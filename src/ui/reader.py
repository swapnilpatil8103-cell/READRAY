"""Reader: PDF viewer (left, ~60%) and context-aware AI assistant (right, ~40%)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from .. import charts
from ..ai_service import ChatMessage
from ..notes import make_note
from ..pdf_processor import (
    PDFProcessingError, EncryptedPDFError, crop_region_png, extract_tables, highlight_rects,
    ocr_available, ocr_page_text, search_document, text_in_region,
)
from ..pdf_viewer import ZOOM_OPTIONS, pdf_viewer, render_scale_for
from ..prompts import FOLLOWUP_PREFIX, LENGTHS, LEVELS, MODES, QUIZ_EVALUATE, QUIZ_GENERATE, build_user_prompt
from ..retrieval import SourcePassage, build_paper_context, build_section_context, build_selection_context, section_names
from ..utils import (
    MAX_QUESTION_CHARS, ValidationError, extract_json, format_pages, validate_image_upload, validate_text,
)
from . import state
from .common import ai_disabled_notice, ai_footer, badge, complete_ai, quote, stream_ai

SAMPLE_PDF = Path(__file__).resolve().parents[2] / "assets" / "sample_paper.pdf"


# --------------------------------------------------------------------------- upload


def _ingest(data: bytes, filename: str, password: str | None = None, origin: str = "") -> bool:
    try:
        with st.spinner(f"Reading “{filename}”…"):
            record, is_new = state.ingest_pdf(data, filename, password)
    except EncryptedPDFError as exc:
        st.session_state["rr_pending_pdf"] = (data, filename, origin)
        st.warning(str(exc) + " Enter the password below to open it.", icon=":material/lock:")
        return False
    except (ValidationError, PDFProcessingError) as exc:
        st.error(f"“{filename}”: {exc}", icon=":material/error:")
        return False
    st.session_state.pop("rr_pending_pdf", None)
    state.open_paper(record["id"])
    if not is_new:
        st.toast(f"“{record['title'][:60]}” was already in your library — reopened it.")
    return True


def upload_widget(key: str, label: str = "Upload a research paper (PDF)") -> None:
    files = st.file_uploader(label, type=["pdf"], key=key, help=f"Up to {state.policy().max_pdf_mb:g} MB. Drag and drop works.")
    done: set = st.session_state.setdefault("_rr_done_uploads", set())
    if files is not None and files.file_id not in done:
        done.add(files.file_id)
        if _ingest(files.getvalue(), files.name, origin=key):
            st.rerun()
    pending_password_form(key)


def pending_password_form(key: str) -> None:
    """Password prompt for an encrypted PDF, shown next to the uploader that received it."""
    pending = st.session_state.get("rr_pending_pdf")
    if pending and pending[2] in (key, ""):
        with st.form(f"{key}_pw", clear_on_submit=True):
            pw = st.text_input(f"Password for “{pending[1]}”", type="password")
            if st.form_submit_button("Open"):
                if _ingest(pending[0], pending[1], pw, origin=key):
                    st.rerun()


def empty_state() -> None:
    st.markdown(
        "<div class='rr-empty'><h3>Open a paper to start reading</h3>"
        "<p>Upload a PDF, then select any passage, equation or figure and ask the assistant to explain it "
        "in the context of the paper, with page citations you can check.</p></div>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        upload_widget("rr_upload_main")
        if SAMPLE_PDF.exists() and st.button("Or open the sample paper", icon=":material/description:", use_container_width=True):
            if _ingest(SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name):
                st.rerun()
        papers = state.workspace()["papers"]
        if papers:
            st.caption(f"{len(papers)} paper(s) in your library — open one from **My Library**.")


# --------------------------------------------------------------------------- navigation callbacks


def _go(page: int) -> None:
    st.session_state["rr_page"] = int(page)


def _shift(delta: int, count: int) -> None:
    st.session_state["rr_page"] = max(1, min(count, int(st.session_state["rr_page"]) + delta))


def _on_map_select() -> None:
    event = st.session_state.get("rr_map")
    try:
        points = event["selection"]["points"]
        if points:
            st.session_state["rr_page"] = int(points[0]["x"])
    except (KeyError, TypeError, ValueError):
        pass


def _use_text(text: str, page: int | None) -> None:
    st.session_state["rr_sel_text"] = text
    st.session_state["rr_sel_page"] = page


# --------------------------------------------------------------------------- left panel


def header(loaded: state.LoadedPaper, record: dict) -> None:
    paper = loaded.paper
    meta = " · ".join(x for x in [paper.authors, paper.year, f"{paper.page_count} pages"] if x)
    st.markdown(f"<p class='rr-title'>{_esc(paper.title)}</p><p class='rr-sub'>{_esc(meta)}</p>", unsafe_allow_html=True)
    for w in paper.warnings:
        st.warning(w, icon=":material/warning:")


def _esc(text: str) -> str:
    import html

    return html.escape(text or "")


def toolbar(count: int) -> None:
    c1, c2, c3, c4, c5 = st.columns([0.6, 1.3, 0.6, 1.4, 3.1], vertical_alignment="bottom")
    c1.button("", icon=":material/chevron_left:", on_click=_shift, args=(-1, count), disabled=st.session_state["rr_page"] <= 1,
              help="Previous page", use_container_width=True)
    c2.number_input(f"Page (of {count})", min_value=1, max_value=count, step=1, key="rr_page")
    c3.button("", icon=":material/chevron_right:", on_click=_shift, args=(1, count), disabled=st.session_state["rr_page"] >= count,
              help="Next page", use_container_width=True)
    c4.selectbox("Zoom", list(ZOOM_OPTIONS), key="rr_zoom")
    c5.text_input("Search this paper", key="rr_search", placeholder="e.g. instrumental variable", max_chars=200)


def viewer_panel(loaded: state.LoadedPaper) -> None:
    paper = loaded.paper
    page_no = st.session_state["rr_page"]
    page = paper.page(page_no)
    query = st.session_state["rr_search"].strip()

    if query:
        hits = search_document(paper, query)
        with st.expander(f"Search: {sum(h.count for h in hits)} match(es) on {len(hits)} page(s)", expanded=True):
            if not hits:
                st.caption("No matches in the extracted text. Scanned pages cannot be searched without OCR.")
            else:
                if len(hits) > 1:
                    st.plotly_chart(charts.search_hits_chart(hits, paper.page_count, query), use_container_width=True,
                                    config={"displayModeBar": False})
                for h in hits[:30]:
                    cols = st.columns([1, 6], vertical_alignment="center")
                    cols[0].button(f"p. {h.page}", key=f"hit_{h.page}", on_click=_go, args=(h.page,),
                                   type="primary" if h.page == page_no else "secondary", use_container_width=True)
                    cols[1].caption(h.snippet)
                if len(hits) > 30:
                    st.caption(f"Showing the first 30 of {len(hits)} pages.")

    if page.image_only:
        _scanned_page_notice(loaded, page_no)

    zoom = ZOOM_OPTIONS[st.session_state["rr_zoom"]]
    try:
        png = state.page_png(loaded, page_no, render_scale_for(zoom))
        highlights = highlight_rects(loaded.pdf, page_no, query) if query else []
    except Exception:
        st.error(f"Page {page_no} could not be rendered.", icon=":material/error:")
        return
    event = pdf_viewer(png, page, paper.doc_id, zoom, highlights, height=860, key="rr_viewer")
    _handle_viewer_event(event, loaded)

    _fallback_selection(loaded, page_no)
    _paper_tools(loaded, page_no)


def _handle_viewer_event(event, loaded: state.LoadedPaper) -> None:
    if not isinstance(event, dict) or event.get("nonce") == st.session_state["_rr_viewer_nonce"]:
        return
    st.session_state["_rr_viewer_nonce"] = event.get("nonce")
    page = int(event.get("page") or st.session_state["rr_page"])
    if event.get("event") == "selection" and event.get("text"):
        _use_text(str(event["text"])[:20_000], page)
        if st.session_state["rr_mode"] in ("figure", "teach", "quiz", "summarize"):
            st.session_state["rr_mode"] = "explain"
    elif event.get("event") == "crop":
        try:
            rect = tuple(float(v) for v in event["rect"])
            png = crop_region_png(loaded.pdf, page, rect)
        except (KeyError, TypeError, ValueError) as exc:
            st.warning(f"Could not crop that region: {exc}")
            return
        region_text = text_in_region(loaded.paper, page, rect)
        st.session_state["rr_crop"] = {"png": png, "page": page, "rect": rect, "text": region_text, "origin": "crop"}
        st.session_state["rr_mode"] = "equation" if region_text and _looks_like_equation(region_text) else "figure"
        if region_text:
            _use_text(region_text, page)


def _looks_like_equation(text: str) -> bool:
    from ..pdf_processor import _is_equation

    return _is_equation(text)


def _scanned_page_notice(loaded: state.LoadedPaper, page_no: int) -> None:
    if ocr_available():
        st.warning("This page looks like a scanned image with no text layer.", icon=":material/document_scanner:")
        if st.button("Run OCR on this page", key=f"ocr_{page_no}"):
            try:
                with st.spinner("Running OCR…"):
                    text = ocr_page_text(loaded.pdf, page_no)
                st.session_state[f"rr_ocr_{loaded.paper.doc_id}_{page_no}"] = text
            except Exception as exc:
                st.error(f"OCR failed: {exc}")
        text = st.session_state.get(f"rr_ocr_{loaded.paper.doc_id}_{page_no}")
        if text:
            with st.expander("OCR text (machine-read — may contain errors)", expanded=True):
                st.text(text[:5000])
                st.button("Use OCR text as the selection", on_click=_use_text, args=(text[:20_000], page_no))
    else:
        st.warning(
            "This page looks like a scanned image with no text layer, and OCR is not installed on this server. "
            "You can still **crop a region** (switch the viewer to *Crop region*) and use *Figure / table* mode with a "
            "vision-capable model, or paste the text into the assistant manually.",
            icon=":material/document_scanner:",
        )


def _fallback_selection(loaded: state.LoadedPaper, page_no: int) -> None:
    page = loaded.paper.page(page_no)
    with st.expander("Can't select text? Choose a paragraph from this page"):
        if not page.blocks:
            st.caption("No extracted text on this page.")
            return
        options = list(range(len(page.blocks)))
        idx = st.selectbox(
            "Paragraph", options, key=f"rr_block_{loaded.paper.doc_id}_{page_no}",
            format_func=lambda i: f"[{page.blocks[i].kind}] " + " ".join(page.blocks[i].text.split())[:110],
        )
        quote(page.blocks[idx].text)
        st.button("Use this paragraph", on_click=_use_text, args=(page.blocks[idx].text, page_no), icon=":material/arrow_forward:")


def _paper_tools(loaded: state.LoadedPaper, page_no: int) -> None:
    paper = loaded.paper
    with st.expander("Paper map, sections, figures & equations"):
        st.plotly_chart(charts.paper_map(paper), use_container_width=True, key="rr_map", on_select=_on_map_select,
                        selection_mode="points", config={"displayModeBar": False})
        tabs = st.tabs(["Sections", "Figures & tables", "Equations", "Abstract"])
        with tabs[0]:
            if not paper.headings:
                st.caption("No section headings were detected (common in PDFs without a clear heading style).")
            for i, h in enumerate(paper.headings[:80]):
                st.button(f"p. {h.page} — {h.text[:90]}", key=f"hd_{i}", on_click=_go, args=(h.page,), type="tertiary")
        with tabs[1]:
            if not paper.captions:
                st.caption("No figure or table captions were detected.")
            for i, c in enumerate(paper.captions[:80]):
                st.button(f"p. {c.page} — {c.text[:110]}", key=f"cap_{i}", on_click=_go, args=(c.page,), type="tertiary")
        with tabs[2]:
            st.caption("Lines that *look* like equations (heuristic). Extracted math is often imperfect — verify against the page.")
            for i, e in enumerate(paper.equations[:60]):
                cols = st.columns([1, 5])
                cols[0].button(f"p. {e.page}", key=f"eq_{i}", on_click=_go, args=(e.page,))
                cols[1].code(e.text, language=None)
            if not paper.equations:
                st.caption("None detected.")
        with tabs[3]:
            st.write(paper.abstract or "_No abstract was detected automatically._")

    with st.expander("Tables & charts"):
        _tables_and_charts(loaded, page_no)


def _tables_and_charts(loaded: state.LoadedPaper, page_no: int) -> None:
    key = f"rr_tables_{loaded.paper.doc_id}_{page_no}"
    if st.button("Detect tables on this page", icon=":material/table:"):
        with st.spinner("Looking for tables…"):
            st.session_state[key] = extract_tables(loaded.pdf, page_no)
    frames = st.session_state.get(key)
    if frames is not None and not frames:
        st.caption("No tables were detected on this page. Tables drawn as images cannot be extracted; crop them instead.")
    for i, df in enumerate(frames or []):
        st.markdown(f"**Table {i + 1}, p. {page_no}** " + badge("extracted from the paper", "rr-src"), unsafe_allow_html=True)
        st.caption("Detected automatically by PyMuPDF. Cells can be merged or misaligned — compare with the page before relying on it.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        _chart_builder(df, "paper", f"Table detected on p. {page_no} of “{loaded.paper.title[:60]}”", f"tbl_{page_no}_{i}")

    st.divider()
    st.markdown("**Plot your own data** " + badge("user-supplied", "rr-src"), unsafe_allow_html=True)
    st.caption("Paste CSV (first row = headers), e.g. values you read off a figure. The chart will be labelled as your data.")
    raw = st.text_area("CSV data", key="rr_user_csv", height=110, placeholder="year,rate\n2019,1.5\n2020,0.25")
    if raw.strip():
        import io

        try:
            df = pd.read_csv(io.StringIO(raw[:50_000]))
            _chart_builder(df, "user", "Values entered by the reader", "user")
        except Exception:
            st.error("Could not read that CSV. Check the separators and header row.")


def _chart_builder(df: pd.DataFrame, provenance: str, note: str, key: str) -> None:
    df = charts.coerce_numeric(df)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric or len(df.columns) < 2:
        st.caption("Not enough numeric columns to chart.")
        return
    cols = st.columns(3)
    x = cols[0].selectbox("X axis", list(df.columns), key=f"{key}_x")
    ys = cols[1].multiselect("Y values", [c for c in numeric if c != x], default=[c for c in numeric if c != x][:1], key=f"{key}_y")
    kind = cols[2].selectbox("Chart", ["bar", "line", "scatter"], key=f"{key}_k")
    if ys:
        st.plotly_chart(charts.data_chart(df, x, ys, kind, provenance, note), use_container_width=True)


# --------------------------------------------------------------------------- right panel


def assistant_panel(loaded: state.LoadedPaper, record: dict) -> None:
    st.markdown("##### Reading assistant")
    ready = state.ai_ready()
    if not ready:
        ai_disabled_notice(compact=True)

    if st.session_state.get("rr_mode") not in MODES:
        st.session_state["rr_mode"] = "explain"
    st.pills("Mode", list(MODES), key="rr_mode", format_func=lambda k: MODES[k].label, label_visibility="collapsed")
    mode_key = st.session_state.get("rr_mode") or "explain"
    mode = MODES[mode_key]
    st.caption(mode.help)

    if mode.scope == "quiz":
        _quiz_panel(loaded, ready)
        _notes_box(loaded, record)
        return

    c1, c2 = st.columns([3, 2])
    c1.segmented_control("Level", list(LEVELS), key="rr_level")
    c2.segmented_control("Length", list(LENGTHS), key="rr_length")

    images: list[tuple[bytes, str]] = []
    section = None
    problem = None
    if mode.scope in {"selection", "figure"}:
        crop = st.session_state.get("rr_crop")
        if mode.scope == "figure" or (mode_key == "equation" and crop):
            images, problem = _image_inputs(mode_key)
        label = "Caption or related text (optional)" if mode.scope == "figure" else "Selected passage"
        sel_page = st.session_state.get("rr_sel_page")
        st.text_area(
            label + (f" — from p. {sel_page}" if sel_page else ""), key="rr_sel_text", height=130,
            placeholder="Select text in the viewer and click “Use selection”, choose a paragraph below the viewer, or paste text here.",
        )
        if mode_key == "equation":
            latex = st.text_input("Or type the equation in LaTeX (optional)", key="rr_eq_latex", placeholder=r"r_t = \alpha + \beta m_t + \varepsilon_t")
            if latex.strip():
                try:
                    st.latex(latex)
                except Exception:
                    st.caption("Could not render that LaTeX.")
        text = st.session_state["rr_sel_text"].strip() or (st.session_state.get("rr_eq_latex", "").strip() if mode_key == "equation" else "")
        if mode.scope == "selection" and not text and not images:
            problem = "Select or paste a passage first."
        if mode.scope == "figure" and not images and not text:
            problem = "Crop a region in the viewer, upload an image, or paste the caption."
    elif mode.scope == "section":
        names = section_names(loaded.chunks)
        if names:
            section = st.selectbox("Section", names, key="rr_section")
        else:
            st.caption("No sections were detected; the selected passage will be summarised instead.")
            st.text_area("Passage to summarise", key="rr_sel_text", height=130)
            if not st.session_state["rr_sel_text"].strip():
                problem = "No sections detected — paste or select the text to summarise."
    elif mode.scope == "paper":
        st.caption("Uses the title, abstract, section outline and retrieved excerpts — not the entire paper.")

    st.text_input("Anything specific you want to know? (optional)", key="rr_extra", max_chars=500)
    if st.button(f"{mode.label}", type="primary", icon=mode.icon, disabled=not ready or problem is not None,
                 help=problem or None, use_container_width=True):
        _start_thread(loaded, mode_key, images, section)
    elif problem and ready:
        st.caption(problem)

    _thread_view(loaded, record)
    _notes_box(loaded, record)
    ai_footer()


def _image_inputs(mode_key: str) -> tuple[list[tuple[bytes, str]], str | None]:
    crop = st.session_state.get("rr_crop")
    images: list[tuple[bytes, str]] = []
    if crop:
        st.image(crop["png"], caption=f"Region from p. {crop['page']}" if crop.get("page") else "Uploaded image", use_container_width=True)
        if st.button("Clear image", icon=":material/close:"):
            st.session_state["rr_crop"] = None
            st.rerun()
        images.append((crop["png"], crop.get("media_type", "image/png")))
    else:
        up = st.file_uploader("Or upload an image of the figure/table/equation", type=["png", "jpg", "jpeg", "gif"], key="rr_img_upload")
        if up is not None:
            try:
                img = validate_image_upload(up.getvalue())
                st.session_state["rr_crop"] = {"png": img.data, "media_type": img.media_type, "page": None, "origin": "upload"}
                st.rerun()
            except ValidationError as exc:
                return [], str(exc)
        if mode_key == "figure":
            st.caption("Tip: switch the viewer to **Crop region** and drag a box around the figure.")
    return images, None


def _start_thread(loaded: state.LoadedPaper, mode_key: str, images: list, section: str | None) -> None:
    ss = st.session_state
    paper, budget = loaded.paper, int(ss["rr_budget"])
    try:
        extra = validate_text(ss.get("rr_extra"), MAX_QUESTION_CHARS) if (ss.get("rr_extra") or "").strip() else ""
        scope = MODES[mode_key].scope
        selection = ""
        if scope == "paper":
            bundle = build_paper_context(paper, loaded.chunks, loaded.index, budget)
        elif scope == "section" and section:
            bundle = build_section_context(paper, loaded.chunks, section, budget)
        else:
            selection = (ss["rr_sel_text"] or "").strip()
            latex = (ss.get("rr_eq_latex") or "").strip() if mode_key == "equation" else ""
            if latex:
                selection = (selection + "\n\n" if selection else "") + f"Equation provided by the reader (LaTeX): {latex}"
            crop = ss.get("rr_crop")
            page_hint = ss.get("rr_sel_page") or (crop or {}).get("page") or ss["rr_page"]
            if not selection and images:
                caps = [c.text for c in paper.captions if c.page == page_hint]
                selection = ("Captions on this page: " + " | ".join(caps)) if caps else f"[Image of a region on page {page_hint}; no text selected]"
            selection = validate_text(selection, 20_000, "The selected passage")
            bundle = build_selection_context(paper, loaded.chunks, loaded.index, selection, page_hint, budget)
            if images and (crop or {}).get("origin") == "upload":
                bundle.sources.append(SourcePassage("image", None, "", "[The reader uploaded an image; its page in the paper is unknown.]"))
    except ValidationError as exc:
        st.error(str(exc))
        return
    prompt = build_user_prompt(mode_key, bundle.to_prompt(), ss["rr_level"] or "Intermediate", ss["rr_length"] or "Concise", extra)
    cfg = state.active_ai_config()
    ss["rr_thread"] = {
        "paper_id": paper.doc_id, "mode": mode_key, "level": ss["rr_level"], "length": ss["rr_length"],
        "selection": selection or (f"Section: {section}" if section else ""), "page": bundle.selection_page,
        "section": bundle.section, "located": bundle.located, "truncated": bundle.truncated,
        "sources": [asdict(s) for s in bundle.sources], "pages": bundle.pages(),
        "messages": [{"role": "user", "text": prompt}], "images": images,
        "model": cfg.model if cfg else "", "pending": True,
    }


def _thread_messages(thread: dict) -> list[ChatMessage]:
    out = []
    for i, m in enumerate(thread["messages"]):
        out.append(ChatMessage(m["role"], m["text"], thread["images"] if i == 0 else []))
    return out


def _thread_view(loaded: state.LoadedPaper, record: dict) -> None:
    thread = st.session_state.get("rr_thread")
    if not thread or thread.get("paper_id") != loaded.paper.doc_id:
        return
    mode = MODES[thread["mode"]]
    st.divider()
    where = f"p. {thread['page']}" if thread.get("page") else "page not located"
    st.markdown(f"**{mode.label}** · {where} · {thread.get('level')} · {thread.get('length')}  "
                + badge("AI-generated", "rr-ai"), unsafe_allow_html=True)
    if thread.get("selection") and mode.scope in {"selection", "figure"}:
        quote(thread["selection"][:1500] + ("…" if len(thread["selection"]) > 1500 else ""))
    if mode.scope in {"selection", "figure"} and not thread.get("located"):
        st.caption("The passage could not be matched to the extracted text, so its page and surroundings are unverified.")

    for m in thread["messages"][1:]:
        with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
            st.markdown(m["text"] if m["role"] == "assistant" else m["text"].replace(FOLLOWUP_PREFIX, ""))

    if thread.get("pending"):
        with st.chat_message("assistant"):
            text = stream_ai(_thread_messages(thread))
        thread["pending"] = False
        if text:
            thread["messages"].append({"role": "assistant", "text": text})
        elif len(thread["messages"]) > 1:
            thread["messages"].pop()  # drop the failed follow-up so it can be retried
        else:
            st.session_state["rr_thread"] = None
            return

    with st.expander(f"Source excerpts sent to the AI ({format_pages(thread['pages']) or 'no page numbers'}) — verify here"):
        st.caption("These are the only parts of the paper the model saw. Page references in the answer should match them.")
        for i, s in enumerate(thread["sources"]):
            page = s.get("page")
            cols = st.columns([5, 1], vertical_alignment="center")
            cols[0].markdown(f"**{s['role']}** · {'p. ' + str(page) if page else 'no page'}" + (f" · {s['section']}" if s.get("section") else ""))
            if page:
                cols[1].button("Go", key=f"src_go_{i}", on_click=_go, args=(page,), help=f"Open page {page}")
            quote(s["text"][:2500])
        if thread.get("truncated"):
            st.caption("Excerpts were shortened to fit the context budget (adjustable in Settings).")

    answers = [m for m in thread["messages"] if m["role"] == "assistant"]
    if answers:
        cols = st.columns(2)
        if cols[0].button("Save explanation to notes", icon=":material/bookmark_add:", use_container_width=True):
            _save_note(loaded, "ai", answers[-1]["text"], source_text=thread.get("selection", ""), page=thread.get("page"),
                       mode=mode.label, model=thread.get("model", ""))
        if cols[1].button("Clear conversation", icon=":material/delete_sweep:", use_container_width=True):
            st.session_state["rr_thread"] = None
            st.rerun()
        question = st.chat_input("Ask a follow-up about this passage…", key="rr_followup", disabled=not state.ai_ready(), max_chars=MAX_QUESTION_CHARS)
        if question and question.strip():
            thread["messages"].append({"role": "user", "text": FOLLOWUP_PREFIX + question.strip()})
            thread["pending"] = True
            st.rerun()


def _save_note(loaded: state.LoadedPaper, kind: str, content: str, **kwargs) -> None:
    try:
        note = make_note(loaded.paper.doc_id, loaded.paper.title, kind, content, **kwargs)
    except ValueError as exc:
        st.error(str(exc))
        return
    state.workspace()["notes"].append(note)
    state.save_workspace()
    st.toast("Saved to Research Notes.", icon=":material/bookmark:")


def _notes_box(loaded: state.LoadedPaper, record: dict) -> None:
    sel = (st.session_state.get("rr_sel_text") or "").strip()
    page = st.session_state.get("rr_sel_page") or st.session_state["rr_page"]
    with st.popover("Save passage / add my note", icon=":material/edit_note:", use_container_width=True):
        with st.form("rr_note_form", clear_on_submit=True, border=False):
            include = st.checkbox("Include the selected passage", value=bool(sel), disabled=not sel)
            body = st.text_area("My note", placeholder="Your own thoughts (stored as a personal note)")
            tags = st.text_input("Tags", placeholder="method, identification")
            pg = st.number_input("Page", 1, loaded.paper.page_count, int(page))
            if st.form_submit_button("Save", type="primary"):
                kind = "personal" if body.strip() else "passage"
                _save_note(loaded, kind, body, source_text=sel if include else "", page=int(pg), tags=tags)


# --------------------------------------------------------------------------- quiz


def _quiz_panel(loaded: state.LoadedPaper, ready: bool) -> None:
    quiz = st.session_state.get("rr_quiz")
    if quiz and quiz.get("paper_id") != loaded.paper.doc_id:
        quiz = None
    c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
    n = c1.number_input("Questions", 3, 10, 5)
    c2.segmented_control("Level", list(LEVELS), key="rr_level")
    if st.button("Generate quiz" if not quiz else "New quiz", type="primary", icon=":material/quiz:", disabled=not ready, use_container_width=True):
        bundle = build_paper_context(loaded.paper, loaded.chunks, loaded.index, int(st.session_state["rr_budget"]))
        prompt = QUIZ_GENERATE.format(n=int(n), level=LEVELS[st.session_state["rr_level"] or "Intermediate"], context=bundle.to_prompt())
        text = complete_ai([ChatMessage("user", prompt)], spinner="Writing questions from the paper…")
        if text:
            try:
                questions = [q for q in extract_json(text).get("questions", []) if isinstance(q, dict) and q.get("question")]
            except ValueError:
                questions = []
            if not questions:
                st.error("The model did not return a usable quiz. Please try again.")
            else:
                valid = set(bundle.pages())
                for q in questions:
                    q["pages"] = [p for p in q.get("pages", []) if isinstance(p, int) and p in valid]
                st.session_state["rr_quiz"] = {"paper_id": loaded.paper.doc_id, "questions": questions[:10], "feedback": {}}
                st.rerun()
    if not quiz:
        st.caption("Questions are generated from retrieved excerpts of this paper, and your answers are checked against them.")
        return
    for i, q in enumerate(quiz["questions"]):
        with st.container(border=True):
            st.markdown(f"**Q{i + 1}.** {q['question']}  " + badge(q.get("type", "question"), "rr-src"), unsafe_allow_html=True)
            answer = st.text_area("Your answer", key=f"rr_quiz_ans_{i}", height=80, label_visibility="collapsed", placeholder="Your answer…")
            cols = st.columns(2)
            if cols[0].button("Check answer", key=f"rr_quiz_chk_{i}", disabled=not ready or not answer.strip()):
                prompt = QUIZ_EVALUATE.format(level=LEVELS[st.session_state["rr_level"] or "Intermediate"], question=q["question"],
                                              answer_key=q.get("answer_key", ""), pages=format_pages(q.get("pages", [])) or "not given",
                                              answer=answer.strip()[:MAX_QUESTION_CHARS])
                fb = complete_ai([ChatMessage("user", prompt)], spinner="Checking your answer…")
                if fb:
                    quiz["feedback"][str(i)] = fb
            if quiz["feedback"].get(str(i)):
                st.markdown(badge("AI feedback", "rr-ai"), unsafe_allow_html=True)
                st.markdown(quiz["feedback"][str(i)])
            with cols[1].popover("Show model answer"):
                st.markdown(q.get("answer_key", "_No answer key._"))
                if q.get("pages"):
                    st.caption("Supporting pages: " + format_pages(q["pages"]))
                    for p in q["pages"][:4]:
                        st.button(f"Open p. {p}", key=f"rr_quiz_go_{i}_{p}", on_click=_go, args=(p,))


# --------------------------------------------------------------------------- page


def render() -> None:
    ws = state.workspace()
    state.close_paper_if_removed()
    loaded = state.get_loaded(st.session_state.get("rr_current"))
    if loaded is None:
        if st.session_state.get("rr_current"):
            st.warning("The PDF for this paper is not available in this session. Upload it again to keep reading.")
            st.session_state["rr_current"] = None
        empty_state()
        return
    record = ws["papers"][loaded.paper.doc_id]
    state.apply_user_metadata(loaded, record)
    count = loaded.paper.page_count
    st.session_state["rr_page"] = max(1, min(count, int(st.session_state.get("rr_page") or 1)))

    header(loaded, record)
    toolbar(count)
    if record.get("last_page") != st.session_state["rr_page"]:
        record["last_page"] = st.session_state["rr_page"]
        state.save_workspace()

    left, right = st.columns([3, 2], gap="medium")
    with left:
        viewer_panel(loaded)
    with right:
        assistant_panel(loaded, record)

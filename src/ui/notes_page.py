"""Research Notes: browse, search, filter, edit, delete and export notes."""

from __future__ import annotations

import streamlit as st

from ..notes import NOTE_KINDS, all_tags, export_markdown, export_txt, make_note, parse_tags, search_notes
from ..utils import safe_filename
from . import state
from .common import badge, quote

KIND_BADGE = {"ai": ("AI explanation", "rr-ai"), "personal": ("My note", "rr-me"), "passage": ("Saved passage", "rr-src")}


def render() -> None:
    ws = state.workspace()
    notes = ws["notes"]
    st.markdown("### Research Notes")
    st.caption("Personal notes are yours; **AI explanations** are model output saved alongside the original passage. "
               "Notes are stored in Readray, not inside the PDF file.")

    papers = ws["papers"]
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    query = c1.text_input("Search notes", placeholder="Search text, passages, tags…")
    paper_opts = ["All papers"] + list(papers)
    paper_id = c2.selectbox("Paper", paper_opts, format_func=lambda pid: pid if pid == "All papers" else papers[pid]["title"][:50])
    kinds = c3.multiselect("Type", list(NOTE_KINDS), format_func=NOTE_KINDS.get)
    tags = c4.multiselect("Tags", all_tags(notes))
    results = search_notes(notes, query, None if paper_id == "All papers" else paper_id, kinds or None, tags or None)

    with st.expander("Add a note", icon=":material/add:"):
        if not papers:
            st.caption("Add a paper to your library first.")
        else:
            with st.form("notes_add", clear_on_submit=True):
                pid = st.selectbox("Paper", list(papers), format_func=lambda x: papers[x]["title"][:80])
                page = st.number_input("Page (0 = not specific)", 0, 10_000, 0)
                passage = st.text_area("Passage from the paper (optional)", height=80)
                body = st.text_area("My note", height=100)
                tag_text = st.text_input("Tags (comma-separated)")
                if st.form_submit_button("Save note", type="primary"):
                    try:
                        ws["notes"].append(make_note(pid, papers[pid]["title"], "personal" if body.strip() else "passage", body,
                                                     source_text=passage, page=int(page) or None, tags=tag_text))
                        state.save_workspace()
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

    e1, e2, _ = st.columns([1.3, 1.3, 4])
    title = "Readray notes"
    e1.download_button("Export Markdown", export_markdown(results, title), file_name=safe_filename("readray_notes") + ".md",
                       mime="text/markdown", disabled=not results, icon=":material/download:", use_container_width=True)
    e2.download_button("Export TXT", export_txt(results, title), file_name=safe_filename("readray_notes") + ".txt",
                       mime="text/plain", disabled=not results, icon=":material/download:", use_container_width=True)
    st.caption(f"{len(results)} of {len(notes)} note(s) shown — exports include the notes currently shown.")

    if not notes:
        st.markdown("<div class='rr-empty'><h3>No notes yet</h3><p>In the Reader, save passages, AI explanations or your "
                    "own thoughts — they will appear here.</p></div>", unsafe_allow_html=True)
        return
    for n in results:
        _note_card(ws, n)


def _note_card(ws: dict, n: dict) -> None:
    label, cls = KIND_BADGE.get(n["kind"], (n["kind"], "rr-src"))
    with st.container(border=True):
        meta = f"{n.get('paper_title', '')[:80]} · p. {n.get('page') or '—'} · {n.get('created_at', '')[:16].replace('T', ' ')}"
        if n.get("mode"):
            meta += f" · {n['mode']}"
        st.markdown(badge(label, cls) + f"<span class='rr-muted'>{meta}</span>", unsafe_allow_html=True)
        if n.get("source_text"):
            quote(n["source_text"][:3000])
        if n.get("content"):
            st.markdown(n["content"])
        if n.get("tags"):
            st.caption(" ".join(f"#{t}" for t in n["tags"]))
        cols = st.columns([1, 1, 6])
        with cols[0].popover("Edit"):
            with st.form(f"edit_{n['id']}", border=False):
                body = st.text_area("Content", n.get("content", ""), height=200,
                                    help="Editing an AI explanation keeps it labelled as AI-generated.")
                tags = st.text_input("Tags", ", ".join(n.get("tags", [])))
                if st.form_submit_button("Save"):
                    n["content"], n["tags"] = body.strip(), parse_tags(tags)
                    state.save_workspace()
                    st.rerun()
        if cols[1].button("Delete", key=f"del_{n['id']}"):
            ws["notes"] = [x for x in ws["notes"] if x["id"] != n["id"]]
            state.save_workspace()
            st.rerun()

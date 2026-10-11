"""Literature Review workspace: topics, editable matrix with provenance, AI drafting, synthesis, export."""

from __future__ import annotations

import streamlit as st

from .. import charts
from .. import literature_review as lr
from ..ai_service import ChatMessage
from ..retrieval import build_paper_context
from ..utils import now_iso, safe_filename
from . import state
from .common import ai_disabled_notice, badge, complete_ai


def render() -> None:
    ws = state.workspace()
    st.markdown("### Literature Review")
    st.caption("Group papers under a research topic and build a comparison matrix. Every cell records whether it was "
               "entered by you, auto-detected from the PDF, or drafted by AI from retrieved excerpts (unverified).")

    topics = ws["topics"]
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    with c2.popover("New topic", icon=":material/add:", use_container_width=True):
        with st.form("lr_new", clear_on_submit=True, border=False):
            name = st.text_input("Topic name", placeholder="e.g. Monetary policy transmission to bank lending")
            desc = st.text_area("What is your review about? (helps judge relevance)", height=80)
            if st.form_submit_button("Create", type="primary"):
                try:
                    topic = lr.create_topic(ws, name, desc)
                    state.save_workspace()
                    st.session_state["lr_topic"] = topic["id"]
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    if not topics:
        st.markdown("<div class='rr-empty'><h3>No review topics yet</h3><p>Create a topic, then add papers from your library.</p></div>",
                    unsafe_allow_html=True)
        return
    if st.session_state.get("lr_topic") not in topics:
        st.session_state["lr_topic"] = next(iter(topics))
    tid = c1.selectbox("Topic", list(topics), key="lr_topic", format_func=lambda t: topics[t]["name"])
    topic = topics[tid]
    if topic.get("description"):
        st.caption(topic["description"])

    papers = ws["papers"]
    chosen = st.multiselect("Papers in this review", list(papers), default=[p for p in topic["paper_ids"] if p in papers],
                            format_func=lambda pid: papers[pid]["title"][:90], key=f"lr_papers_{tid}")
    if set(chosen) != set(topic["paper_ids"]):
        for pid in list(topic["paper_ids"]):
            if pid not in chosen:
                lr.remove_paper_from_topic(topic, pid)
        for pid in chosen:
            lr.add_paper_to_topic(topic, papers[pid])
        state.save_workspace()
        st.rerun()
    if not topic["paper_ids"]:
        st.info("Add papers from your library to start the matrix.")
        _topic_admin(ws, topic)
        return

    _matrix_editor(topic)
    st.plotly_chart(charts.matrix_coverage_chart(topic, lr.FIELDS), use_container_width=True, config={"displayModeBar": False})
    _ai_tools(topic, papers)
    _exports(topic)
    _topic_admin(ws, topic)


def _matrix_editor(topic: dict) -> None:
    show_prov = st.toggle("Show where each value came from", value=False)
    df = lr.matrix_dataframe(topic, with_provenance=show_prov)
    if show_prov:
        st.dataframe(df.drop(columns=["paper_id"]), use_container_width=True, hide_index=True)
        return
    st.caption("Edit cells directly — edited cells become **manual** and AI drafting will not overwrite them.")
    edited = st.data_editor(
        df, use_container_width=True, hide_index=True, num_rows="fixed", key=f"lr_editor_{topic['id']}",
        column_config={"paper_id": None, **{label: st.column_config.TextColumn(label, width="medium") for _, label in lr.FIELDS}},
    )
    if not edited.equals(df):
        changed = lr.apply_matrix_edits(topic, edited)
        if changed:
            state.save_workspace()
            st.toast(f"Saved {changed} edit(s).")


def _ai_tools(topic: dict, papers: dict) -> None:
    st.markdown("#### AI assistance " + badge("drafts — verify before use", "rr-ai"), unsafe_allow_html=True)
    if not state.ai_ready():
        ai_disabled_notice(compact=True)
        return
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Draft matrix cells from a paper**")
        st.caption("Retrieves relevant excerpts and asks the model to fill research question, method, data, findings, "
                   "limitations and gap — or mark them *not stated*. Manual cells are kept.")
        pid = st.selectbox("Paper", topic["paper_ids"], format_func=lambda x: papers[x]["title"][:70], key="lr_draft_pid")
        overwrite = st.checkbox("Overwrite my manual edits", value=False)
        loaded = state.get_loaded(pid)
        if loaded is None:
            st.caption("The PDF for this paper is not available in this session; re-upload it to draft cells.")
        if st.button("Draft cells", disabled=loaded is None, icon=":material/auto_awesome:"):
            state.apply_user_metadata(loaded, papers[pid])
            bundle = build_paper_context(loaded.paper, loaded.chunks, loaded.index, int(st.session_state["rr_budget"]))
            text = complete_ai([ChatMessage("user", lr.build_extraction_prompt(topic, bundle.to_prompt()))],
                               spinner="Reading excerpts and drafting cells…")
            if text:
                try:
                    cells = lr.parse_extraction(text, set(bundle.pages()), bundle.excerpt_pages())
                except ValueError:
                    st.error("The model's answer could not be parsed. Please try again.")
                else:
                    updated = lr.apply_extraction(topic, pid, cells, overwrite_manual=overwrite)
                    state.save_workspace()
                    st.success(f"Drafted {len(updated)} cell(s). Check them against the cited pages.")
                    st.rerun()
    with c2.container(border=True):
        st.markdown("**Cross-paper comparison**")
        st.caption("Compares papers using only the matrix contents and separates directly supported comparisons from synthesis.")
        enough = len(topic["paper_ids"]) >= 2 and lr.filled_ratio(topic) > 0.3
        if not enough:
            st.caption("Add at least two papers and fill more of the matrix first.")
        if st.button("Compare papers", disabled=not enough, icon=":material/compare_arrows:"):
            text = complete_ai([ChatMessage("user", lr.build_synthesis_prompt(topic))], spinner="Comparing papers…")
            if text:
                cfg = state.active_ai_config()
                topic.setdefault("syntheses", []).insert(0, {"text": text, "created_at": now_iso(), "model": cfg.model if cfg else "",
                                                             "paper_ids": list(topic["paper_ids"])})
                state.save_workspace()
    for i, s in enumerate(topic.get("syntheses", [])[:3]):
        with st.expander(f"AI synthesis · {s['created_at'][:16].replace('T', ' ')} · {s.get('model', '')}", expanded=i == 0):
            st.markdown(badge("AI-generated synthesis", "rr-ai"), unsafe_allow_html=True)
            st.markdown(s["text"])
            if st.button("Delete this synthesis", key=f"lr_syn_del_{i}"):
                topic["syntheses"].pop(i)
                state.save_workspace()
                st.rerun()


def _exports(topic: dict) -> None:
    c1, c2, _ = st.columns([1.3, 1.3, 4])
    name = safe_filename(f"review_{topic['name']}")
    c1.download_button("Export CSV", lr.export_csv(topic), file_name=f"{name}.csv", mime="text/csv",
                       icon=":material/download:", use_container_width=True)
    c2.download_button("Export Markdown", lr.export_markdown(topic), file_name=f"{name}.md", mime="text/markdown",
                       icon=":material/download:", use_container_width=True)


def _topic_admin(ws: dict, topic: dict) -> None:
    with st.expander("Topic settings"):
        with st.form(f"lr_edit_{topic['id']}", border=False):
            name = st.text_input("Name", topic["name"])
            desc = st.text_area("Description", topic.get("description", ""))
            if st.form_submit_button("Save") and name.strip():
                topic["name"], topic["description"] = name.strip()[:120], desc.strip()[:2000]
                state.save_workspace()
                st.rerun()
        if st.button("Delete topic", icon=":material/delete:"):
            ws["topics"].pop(topic["id"], None)
            state.save_workspace()
            st.rerun()

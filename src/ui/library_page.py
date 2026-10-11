"""My Library: upload, list, search, edit metadata, reopen, remove, backup/restore."""

from __future__ import annotations

import streamlit as st

from .. import library
from ..notes import NOTE_KINDS, search_notes
from ..utils import ValidationError, safe_filename
from . import state
from .common import quote
from .reader import _ingest, pending_password_form


def _open(paper_id: str) -> None:
    state.open_paper(paper_id)
    st.switch_page(st.session_state["_rr_pages"]["reader"])


def render() -> None:
    ws = state.workspace()
    st.markdown("### My Library")
    _storage_notice()

    with st.expander("Add papers", expanded=not ws["papers"], icon=":material/upload_file:"):
        files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True, key="lib_upload",
                                 help=f"Up to {state.policy().max_pdf_mb:g} MB each.")
        done: set = st.session_state.setdefault("_rr_done_uploads", set())
        new = [f for f in files or [] if f.file_id not in done]
        if new:
            for f in new:
                done.add(f.file_id)
                _ingest(f.getvalue(), f.name, origin="lib_upload")
            st.rerun()
        pending_password_form("lib_upload")

    if not ws["papers"]:
        st.markdown("<div class='rr-empty'><h3>Your library is empty</h3><p>Upload a PDF above or open one in the Reader.</p></div>",
                    unsafe_allow_html=True)
        _backup_section(ws)
        return

    query = st.text_input("Search by title, author or file name", key="lib_query", placeholder="Search your library…")
    papers = library.search_papers(ws, query)
    st.caption(f"{len(papers)} of {len(ws['papers'])} paper(s)")
    for p in papers:
        _paper_card(ws, p)
    _backup_section(ws)


def _storage_notice() -> None:
    if state.store().kind == "local":
        st.caption(f"Storage: **local folder** `{state.policy().data_dir}` on the machine running Readray (single-user mode).")
    else:
        st.caption("Storage: **this browser session only.** Papers, notes and reviews are kept in server memory for this session "
                   "and disappear when it ends. Use **Backup** below to keep your work.")


def _paper_card(ws: dict, p: dict) -> None:
    pid = p["id"]
    notes = search_notes(ws["notes"], paper_id=pid)
    has_pdf = state.store().get_pdf(pid) is not None
    with st.container(border=True):
        top = st.columns([6, 1.2, 1.2], vertical_alignment="center")
        meta = " · ".join(x for x in [p.get("authors"), p.get("year"), f"{p.get('page_count', '?')} pages",
                                       f"{len(notes)} note(s)"] if x)
        top[0].markdown(f"**{p.get('title') or p.get('filename')}**  \n<span class='rr-muted'>{meta}</span>", unsafe_allow_html=True)
        if top[1].button("Open", key=f"lib_open_{pid}", type="primary", disabled=not has_pdf,
                         help=None if has_pdf else "The PDF is not in this session — upload it again.", use_container_width=True):
            _open(pid)
        with top[2].popover("More", use_container_width=True):
            _edit_form(ws, p)
            st.divider()
            delete_notes = st.checkbox("Also delete this paper's notes", key=f"lib_delnotes_{pid}")
            if st.button("Remove from library", key=f"lib_rm_{pid}", icon=":material/delete:"):
                library.remove_paper(ws, state.store(), pid, delete_notes=delete_notes)
                state.save_workspace()
                st.session_state["_rr_docs"].pop(pid, None)
                state.close_paper_if_removed()
                st.rerun()
            st.caption("Removing deletes Readray's copy in this storage. It does not affect files elsewhere on your computer.")
        with st.expander("Details and notes"):
            st.caption(f"File: {p.get('filename')} · added {p.get('added_at', '')[:10]} · metadata: {p.get('metadata_source', 'auto-detected')}")
            if p.get("abstract"):
                st.markdown("**Abstract (auto-detected)**")
                st.write(p["abstract"])
            if not notes:
                st.caption("No notes yet.")
            for n in notes[:20]:
                st.markdown(f"*{NOTE_KINDS.get(n['kind'], n['kind'])}* · p. {n.get('page') or '?'} · {n.get('created_at', '')[:10]}")
                if n.get("source_text"):
                    quote(n["source_text"][:600])
                if n.get("content"):
                    st.markdown(n["content"][:1200])


def _edit_form(ws: dict, p: dict) -> None:
    with st.form(f"lib_edit_{p['id']}", border=False):
        st.markdown("**Correct metadata**")
        title = st.text_input("Title", p.get("title", ""))
        authors = st.text_input("Authors", p.get("authors", ""))
        year = st.text_input("Year", p.get("year", ""), max_chars=10)
        if st.form_submit_button("Save metadata"):
            try:
                library.update_metadata(ws, p["id"], title=title, authors=authors, year=year)
                state.save_workspace()
                st.toast("Metadata updated.")
                st.rerun()
            except ValidationError as exc:
                st.error(str(exc))


def _backup_section(ws: dict) -> None:
    with st.expander("Backup & restore", icon=":material/backup:"):
        st.caption("A backup is a JSON file with your library metadata, notes and literature reviews. Including PDFs makes "
                   "it larger but lets you restore everything in a new session. Store it somewhere you trust.")
        include_pdfs = st.checkbox("Include PDFs", value=False)
        if st.button("Prepare backup file"):
            pdfs = None
            if include_pdfs:
                pdfs = {pid: d for pid in ws["papers"] if (d := state.store().get_pdf(pid))}
            st.session_state["lib_backup"] = library.export_backup(ws, pdfs)
        if st.session_state.get("lib_backup"):
            st.download_button("Download backup", st.session_state["lib_backup"], file_name=safe_filename("readray_backup") + ".json",
                               mime="application/json", icon=":material/download:")
        st.divider()
        up = st.file_uploader("Restore from a backup (merged into the current workspace)", type=["json"], key="lib_restore")
        if up is not None and st.button("Restore"):
            try:
                incoming, pdfs = library.import_backup(up.getvalue())
            except ValidationError as exc:
                st.error(str(exc))
                return
            library.merge_workspaces(ws, incoming)
            for pid, data in pdfs.items():
                state.store().put_pdf(pid, data)
            state.save_workspace()
            st.success(f"Restored {len(incoming['papers'])} paper record(s), {len(incoming['notes'])} note(s), "
                       f"{len(incoming['topics'])} topic(s); {len(pdfs)} PDF(s).")

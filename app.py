"""Readray — AI-assisted research paper reader.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Readray", page_icon=":material/menu_book:", layout="wide", initial_sidebar_state="collapsed")

from src.ui import library_page, notes_page, reader, review_page, settings_page, state  # noqa: E402
from src.ui.common import inject_css  # noqa: E402

state.init_state()
inject_css()

pages = {
    "reader": st.Page(reader.render, title="Reader", icon=":material/menu_book:", url_path="reader", default=True),
    "library": st.Page(library_page.render, title="My Library", icon=":material/library_books:", url_path="library"),
    "notes": st.Page(notes_page.render, title="Research Notes", icon=":material/edit_note:", url_path="notes"),
    "review": st.Page(review_page.render, title="Literature Review", icon=":material/table_view:", url_path="review"),
    "settings": st.Page(settings_page.render, title="Settings", icon=":material/settings:", url_path="settings"),
}
st.session_state["_rr_pages"] = pages

with st.sidebar:
    st.markdown("### Readray")
    st.caption("Read → select → understand → verify → annotate → synthesize.")
    current = state.workspace()["papers"].get(st.session_state.get("rr_current") or "")
    if current:
        st.markdown(f"**Open paper**  \n{current['title'][:120]}")
        if st.button("Close paper", icon=":material/close:"):
            st.session_state["rr_current"] = None
            st.session_state["rr_thread"] = None
            st.rerun()
    reader.upload_widget("rr_upload_sidebar", "Open another PDF")
    st.caption(("AI: " + state.active_ai_config().display_name) if state.ai_ready() else "AI: not configured (see Settings)")

st.navigation(list(pages.values()), position="top").run()

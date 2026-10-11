"""Session state, storage and AI configuration helpers shared by all UI pages."""

from __future__ import annotations

import hmac
import os
from collections import OrderedDict
from dataclasses import dataclass

import streamlit as st

from .. import library
from ..ai_service import AIConfig, AppPolicy, get_provider, load_policy, load_server_config
from ..pdf_processor import PaperDocument, PDFProcessingError, process_pdf, render_page_png
from ..retrieval import BM25Index, Chunk, build_chunks
from ..utils import ValidationError, sha256_hex, validate_pdf_upload

MAX_DOCS_IN_MEMORY = 4
MAX_RENDERED_PAGES = 16
DEFAULT_CONTEXT_BUDGET = 16_000


def _secrets():
    try:
        # Accessing st.secrets raises if no secrets.toml exists; treat that as "no secrets".
        return st.secrets if len(st.secrets) >= 0 else None
    except Exception:
        return None


def policy() -> AppPolicy:
    if "_rr_policy" not in st.session_state:
        st.session_state["_rr_policy"] = load_policy(_secrets(), os.environ)
    return st.session_state["_rr_policy"]


@st.cache_resource(show_spinner=False)
def _local_store(path: str) -> library.LocalStore:
    # Shared across sessions on purpose: local mode is documented as single-user.
    return library.LocalStore(path)


def init_state() -> None:
    ss = st.session_state
    if "_rr_store" not in ss:
        pol = policy()
        ss["_rr_store"] = _local_store(pol.data_dir) if pol.storage == "local" else library.SessionStore(ss)
    defaults = {
        "rr_current": None,
        "rr_page": 1,
        "rr_zoom": "Fit width",
        "rr_search": "",
        "rr_sel_text": "",
        "rr_sel_page": None,
        "rr_crop": None,
        "rr_thread": None,
        "rr_quiz": None,
        "rr_level": "Intermediate",
        "rr_length": "Concise",
        "rr_mode": "explain",
        "rr_budget": DEFAULT_CONTEXT_BUDGET,
        "rr_ai_requests": 0,
        "rr_user_ai": None,
        "rr_access_ok": False,
        "_rr_docs": OrderedDict(),
        "_rr_pngs": OrderedDict(),
        "_rr_viewer_nonce": None,
    }
    for key, value in defaults.items():
        ss.setdefault(key, value)


def store():
    return st.session_state["_rr_store"]


def workspace() -> dict:
    return store().load()


def save_workspace() -> None:
    store().save(workspace())


# --------------------------------------------------------------------------- documents


@dataclass
class LoadedPaper:
    paper: PaperDocument
    chunks: list[Chunk]
    index: BM25Index
    pdf: bytes


def _remember(paper_id: str, loaded: LoadedPaper) -> None:
    docs: OrderedDict = st.session_state["_rr_docs"]
    docs[paper_id] = loaded
    docs.move_to_end(paper_id)
    while len(docs) > MAX_DOCS_IN_MEMORY:
        docs.popitem(last=False)


def ingest_pdf(data: bytes, filename: str, password: str | None = None) -> tuple[dict, bool]:
    """Validate, parse and add a PDF to the library. Raises ValidationError / PDFProcessingError."""
    validate_pdf_upload(data, filename, policy().max_pdf_mb)
    paper_id = sha256_hex(data)[:16]
    docs = st.session_state["_rr_docs"]
    if paper_id in docs:
        loaded = docs[paper_id]
    else:
        paper = process_pdf(data, filename, password)
        loaded = LoadedPaper(paper, (chunks := build_chunks(paper)), BM25Index(chunks), data)
    _remember(paper_id, loaded)
    record, is_new = library.add_paper(workspace(), library.paper_metadata(loaded.paper, len(data)))
    if is_new or store().get_pdf(paper_id) is None:
        store().put_pdf(paper_id, data)
    if password:
        record["password_protected"] = True
    save_workspace()
    return record, is_new


def get_loaded(paper_id: str | None) -> LoadedPaper | None:
    """Processed paper for the current session, re-parsing from the store when needed."""
    if not paper_id or paper_id not in workspace()["papers"]:
        return None
    docs = st.session_state["_rr_docs"]
    if paper_id in docs:
        docs.move_to_end(paper_id)
        return docs[paper_id]
    data = store().get_pdf(paper_id)
    if data is None:
        return None
    try:
        record = workspace()["papers"][paper_id]
        paper = process_pdf(data, record.get("filename", "paper.pdf"))
    except PDFProcessingError:
        return None
    chunks = build_chunks(paper)
    loaded = LoadedPaper(paper, chunks, BM25Index(chunks), data)
    _remember(paper_id, loaded)
    return loaded


def apply_user_metadata(loaded: LoadedPaper, record: dict) -> None:
    """User-corrected metadata overrides auto-detected values for prompts and display."""
    loaded.paper.title = record.get("title") or loaded.paper.title
    loaded.paper.authors = record.get("authors", loaded.paper.authors)
    loaded.paper.year = record.get("year", loaded.paper.year)


def page_png(loaded: LoadedPaper, page: int, scale: float) -> bytes:
    cache: OrderedDict = st.session_state["_rr_pngs"]
    key = (loaded.paper.doc_id, page, round(scale, 2))
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    png = render_page_png(loaded.pdf, page, scale)
    cache[key] = png
    while len(cache) > MAX_RENDERED_PAGES:
        cache.popitem(last=False)
    return png


def open_paper(paper_id: str) -> None:
    ss = st.session_state
    if ss.get("rr_current") != paper_id:
        ss["rr_current"] = paper_id
        record = workspace()["papers"].get(paper_id, {})
        ss["rr_page"] = int(record.get("last_page", 1) or 1)
        ss["rr_thread"] = None
        ss["rr_quiz"] = None
        ss["rr_crop"] = None
        ss["rr_sel_text"] = ""
        ss["rr_sel_page"] = None


def close_paper_if_removed() -> None:
    if st.session_state.get("rr_current") not in workspace()["papers"]:
        st.session_state["rr_current"] = None


# --------------------------------------------------------------------------- AI


def active_ai_config() -> AIConfig | None:
    user_cfg = st.session_state.get("rr_user_ai")
    if user_cfg is not None:
        return user_cfg
    cfg = load_server_config(_secrets(), os.environ)
    if cfg is None:
        return None
    if policy().access_code and not st.session_state.get("rr_access_ok"):
        return None
    return cfg


def server_ai_needs_access_code() -> bool:
    return bool(policy().access_code) and load_server_config(_secrets(), os.environ) is not None and not st.session_state.get("rr_access_ok")


def check_access_code(code: str) -> bool:
    expected = policy().access_code
    ok = bool(expected) and hmac.compare_digest(code.strip().encode(), expected.encode())
    st.session_state["rr_access_ok"] = ok
    return ok


def ai_ready() -> bool:
    return active_ai_config() is not None


def quota_left() -> int | None:
    """Remaining requests for this session on the owner's key (None = unlimited / user's own key)."""
    cfg = active_ai_config()
    if cfg is None or cfg.source == "session":
        return None
    return max(0, policy().max_requests_per_session - st.session_state["rr_ai_requests"])


def provider_for_request():
    """Provider for one request, enforcing the per-session quota on the server key."""
    cfg = active_ai_config()
    provider = get_provider(cfg)
    left = quota_left()
    if left is not None and left <= 0:
        raise ValidationError(
            "This session has used its AI request allowance on the shared key. Add your own API key in Settings to continue."
        )
    st.session_state["rr_ai_requests"] += 1
    return provider

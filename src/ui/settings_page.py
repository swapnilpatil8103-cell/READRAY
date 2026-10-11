"""Settings: AI configuration status, per-session key, access code, reading preferences, privacy."""

from __future__ import annotations

import os

import streamlit as st

from ..ai_service import DEFAULT_ANTHROPIC_MODEL, DEFAULT_GEMINI_MODEL, GEMINI_BASE_URL, PROVIDERS, AIConfig, load_server_config
from ..pdf_processor import ocr_available
from . import state
from .common import ai_disabled_notice


def render() -> None:
    st.markdown("### Settings")
    pol = state.policy()
    cfg = state.active_ai_config()

    st.markdown("#### AI provider")
    if cfg:
        st.success(f"AI is enabled: {cfg.display_name}.", icon=":material/check_circle:")
        left = state.quota_left()
        if left is not None:
            st.caption(f"This session can make {left} more request(s) on the shared key (limit {pol.max_requests_per_session}).")
    else:
        ai_disabled_notice()

    if state.server_ai_needs_access_code():
        with st.form("access_code"):
            code = st.text_input("Access code", type="password", help="Provided by whoever runs this deployment.")
            if st.form_submit_button("Unlock AI"):
                if state.check_access_code(code):
                    st.success("Access granted for this session.")
                    st.rerun()
                else:
                    st.error("That access code is not correct.")

    if pol.allow_user_keys:
        with st.expander("Use your own API key for this session", expanded=cfg is None, icon=":material/key:"):
            st.caption("The key is held in this session's server memory only. It is not written to disk, logged, or shown "
                       "again, and it is discarded when the session ends. Requests are billed to your account.")
            with st.form("user_key", clear_on_submit=True):
                provider = st.selectbox("Provider", list(PROVIDERS), format_func=PROVIDERS.get)
                key = st.text_input("API key", type="password",
                                    help="Optional for local OpenAI-compatible servers such as Ollama.")
                model = st.text_input("Model (optional for Anthropic and Gemini)",
                                      placeholder=f"{DEFAULT_ANTHROPIC_MODEL} (Anthropic) · {DEFAULT_GEMINI_MODEL} (Gemini)")
                base_url = st.text_input("Base URL (OpenAI-compatible only)", placeholder="https://api.groq.com/openai/v1")
                if st.form_submit_button("Use this key", type="primary"):
                    model = model.strip() or {"anthropic": DEFAULT_ANTHROPIC_MODEL, "gemini": DEFAULT_GEMINI_MODEL}.get(provider, "")
                    if provider in ("anthropic", "gemini") and not key.strip():
                        st.error(f"Enter a {PROVIDERS[provider]} API key.")
                    elif provider == "openai_compatible" and not model:
                        st.error("Enter the model name for your OpenAI-compatible provider.")
                    elif provider == "openai_compatible" and not (key.strip() or base_url.strip()):
                        st.error("Enter an API key or a base URL.")
                    else:
                        st.session_state["rr_user_ai"] = AIConfig(
                            provider=provider, api_key=key.strip() or "not-needed", model=model,
                            base_url={"gemini": GEMINI_BASE_URL, "openai_compatible": base_url.strip() or None}.get(provider),
                            source="session",
                        )
                        st.rerun()
            if st.session_state.get("rr_user_ai") is not None and st.button("Forget my key"):
                st.session_state["rr_user_ai"] = None
                st.rerun()
    elif cfg is None:
        st.caption("This deployment does not accept user-supplied keys.")

    st.markdown("#### Reading")
    st.slider("Context budget (characters of the paper sent per request)", 4_000, 40_000, step=1_000, key="rr_budget",
              help="Higher sends more surrounding text (better context, higher cost). Roughly 4 characters ≈ 1 token.")
    c1, c2 = st.columns(2)
    c1.segmented_control("Default explanation level", ["Beginner", "Intermediate", "Advanced"], key="rr_level")
    c2.segmented_control("Default length", ["Concise", "Detailed"], key="rr_length")

    st.markdown("#### Data, privacy and cost")
    storage = ("**This session only** — papers, notes and reviews live in server memory for your session and are not shared "
               "with other visitors. They are lost when the session ends; use *Backup* in My Library to keep them."
               if state.store().kind == "session" else
               f"**Local folder** `{pol.data_dir}` — persistent on this machine. Intended for single-user local use only.")
    st.markdown(
        f"- Storage: {storage}\n"
        "- When you use an AI feature, the selected text, retrieved excerpts, your question and any image you crop or upload "
        "are sent to the configured AI provider and handled under that provider's terms. Readray cannot guarantee "
        "confidentiality once text leaves this server — avoid AI features for papers you may not share.\n"
        "- AI API usage may incur charges for whoever owns the key.\n"
        "- Readray does not log paper contents or API keys. The hosting platform may keep its own operational logs.\n"
        "- Removing a paper deletes Readray's copy from its storage; it cannot remove copies already sent to an AI provider."
    )
    st.markdown("#### Environment")
    server = load_server_config(state._secrets(), os.environ)
    st.caption(
        f"OCR (Tesseract): {'available' if ocr_available() else 'not installed — scanned pages cannot be read automatically'} · "
        f"Max PDF size: {pol.max_pdf_mb:g} MB · Server key configured: {'yes' if server else 'no'} · "
        f"User keys allowed: {'yes' if pol.allow_user_keys else 'no'}"
    )

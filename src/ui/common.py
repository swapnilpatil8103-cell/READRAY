"""Shared UI pieces: styling, AI streaming with friendly errors, small widgets."""

from __future__ import annotations

import streamlit as st

from ..ai_service import AIError, ChatMessage, missing_config_help
from ..prompts import SYSTEM_PROMPT
from ..utils import ValidationError
from . import state

CSS = """
<style>
:root { --rr-accent: #3b5b92; --rr-ai: #c8803a; --rr-me: #2f6f4e; --rr-muted: #6b7686; }
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.rr-title { font-size: 1.35rem; font-weight: 650; line-height: 1.3; margin: 0; }
.rr-sub { color: var(--rr-muted); font-size: .9rem; margin: .15rem 0 .4rem 0; }
.rr-badge { display: inline-block; font-size: .72rem; font-weight: 600; padding: 1px 8px; border-radius: 999px;
            border: 1px solid currentColor; margin-right: 6px; vertical-align: middle; }
.rr-ai { color: var(--rr-ai); } .rr-me { color: var(--rr-me); } .rr-src { color: var(--rr-accent); }
.rr-quote { border-left: 3px solid var(--rr-accent); padding: .35rem .75rem; margin: .3rem 0 .6rem 0;
            color: inherit; opacity: .92; font-size: .92rem; white-space: pre-wrap; max-height: 240px; overflow: auto; }
.rr-muted { color: var(--rr-muted); font-size: .85rem; }
.rr-empty { text-align: center; padding: 2.5rem 1rem; color: var(--rr-muted); }
.rr-empty h3 { color: inherit; }
/* Let pills / segmented controls wrap instead of scrolling sideways (falls back gracefully). */
[data-testid="stButtonGroup"] > div { flex-wrap: wrap; row-gap: .4rem; }
@media (max-width: 640px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(text: str, cls: str) -> str:
    return f"<span class='rr-badge {cls}'>{text}</span>"


def quote(text: str) -> None:
    import html

    st.markdown(f"<div class='rr-quote'>{html.escape(text)}</div>", unsafe_allow_html=True)


def ai_disabled_notice(compact: bool = False) -> None:
    if state.server_ai_needs_access_code():
        st.info("AI features on this deployment need an access code, or your own API key. Enter either on the **Settings** page.",
                icon=":material/lock:")
        return
    if compact:
        st.info("AI is not configured, so AI explanations are off. Reading, search and notes still work. "
                "See **Settings** to enable AI.", icon=":material/info:")
    else:
        st.info(missing_config_help(), icon=":material/info:")


def stream_ai(messages: list[ChatMessage], system: str = SYSTEM_PROMPT) -> str | None:
    """Stream a response into the page. Returns the full text, or None if the request failed."""
    try:
        provider = state.provider_for_request()
    except (AIError, ValidationError) as exc:
        st.error(str(exc), icon=":material/error:")
        return None
    try:
        text = st.write_stream(provider.stream(system, messages))
    except AIError as exc:
        st.error(str(exc), icon=":material/error:")
        return None
    except Exception:  # never leak raw provider errors (they may echo request details)
        st.error("The AI request failed unexpectedly. Please try again.", icon=":material/error:")
        return None
    return text if isinstance(text, str) else "".join(map(str, text or []))


def complete_ai(messages: list[ChatMessage], system: str = SYSTEM_PROMPT, spinner: str = "Thinking…") -> str | None:
    try:
        provider = state.provider_for_request()
        with st.spinner(spinner):
            return provider.complete(system, messages)
    except (AIError, ValidationError) as exc:
        st.error(str(exc), icon=":material/error:")
    except Exception:
        st.error("The AI request failed unexpectedly. Please try again.", icon=":material/error:")
    return None


def ai_footer() -> None:
    cfg = state.active_ai_config()
    if cfg is None:
        return
    left = state.quota_left()
    quota = f" · {left} request(s) left this session" if left is not None else ""
    st.caption(f"AI: {cfg.display_name}{quota}. Selected text and excerpts are sent to this provider; "
               "API usage may incur charges. AI output can be wrong — check the cited pages.")

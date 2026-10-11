"""Streamlit startup smoke tests using streamlit.testing (no browser, no network, no API key)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "READRAY_AI_PROVIDER", "READRAY_STORAGE"):
        monkeypatch.delenv(var, raising=False)


def _errors(at):
    return [e.value for e in at.exception]


def test_app_starts_without_api_key():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not _errors(at)
    page_text = " ".join(m.value for m in at.markdown)
    assert "Open a paper to start reading" in page_text


def test_reader_with_sample_paper_and_ai_disabled():
    at = AppTest.from_file(APP, default_timeout=60).run()
    sample = next(b for b in at.button if "sample paper" in b.label.lower())
    sample.click().run()
    assert not _errors(at)
    md = " ".join(m.value for m in at.markdown)
    assert "Interest Rate Announcements" in md
    assert any("AI is not configured" in i.value for i in at.info)
    # The AI action button exists but is disabled without credentials.
    run_btn = next(b for b in at.button if b.label == "Explain")
    assert run_btn.disabled
    # Page navigation works.
    at.number_input(key="rr_page").set_value(2).run()
    assert not _errors(at) and at.session_state["rr_page"] == 2
    # Pasting a passage into the assistant works without AI configured.
    at.text_area(key="rr_sel_text").input("Standard errors are clustered by bank.").run()
    assert not _errors(at)
    assert len(at.session_state["_rr_workspace"]["papers"]) == 1


PAGE_SCRIPT = """
import sys
sys.path.insert(0, {root!r})
import streamlit as st
from src.ui import state, {module}
from src import library, literature_review as lr
from src.notes import make_note
state.init_state()
ws = state.workspace()
if {seed!r} and not ws["papers"]:
    ws["papers"]["abc123"] = {{"id": "abc123", "title": "Seeded paper", "authors": "A. B.", "year": "2020", "page_count": 3,
                              "filename": "x.pdf", "added_at": "2026-01-01T00:00:00+00:00"}}
    ws["notes"].append(make_note("abc123", "Seeded paper", "ai", "AI text", source_text="passage", page=2, tags="x"))
    t = lr.create_topic(ws, "Topic")
    lr.add_paper_to_topic(t, ws["papers"]["abc123"])
st.session_state["_rr_pages"] = {{}}
{module}.render()
"""


@pytest.mark.parametrize("module", ["library_page", "notes_page", "review_page", "settings_page"])
@pytest.mark.parametrize("seed", [False, True])
def test_other_pages_render(module, seed):
    root = str(Path(APP).parent)
    at = AppTest.from_string(PAGE_SCRIPT.format(root=root, module=module, seed=seed), default_timeout=60).run()
    assert not _errors(at)

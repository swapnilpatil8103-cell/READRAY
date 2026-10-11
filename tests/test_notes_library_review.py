import json

import pandas as pd
import pytest

from src import library
from src import literature_review as lr
from src.notes import all_tags, export_markdown, export_txt, make_note, parse_tags, search_notes
from src.pdf_processor import process_pdf
from src.utils import ValidationError

# --------------------------------------------------------------------------- notes


@pytest.fixture
def notes():
    return [
        make_note("p1", "Paper One", "ai", "Beta measures **sensitivity**.", source_text="β is the slope", page=3,
                  mode="Explain", model="claude-opus-5-5", tags="Method, regression"),
        make_note("p1", "Paper One", "personal", "Check robustness table.", page=5, tags=["todo"]),
        make_note("p2", "Paper Two", "passage", "", source_text="Lending falls by two percent.", page=2),
    ]


def test_make_note_validation():
    with pytest.raises(ValueError):
        make_note("p", "t", "personal", "   ")
    with pytest.raises(ValueError):
        make_note("p", "t", "unknown", "x")
    assert parse_tags("A b, c;#c\n D") == ["a-b", "c", "d"]


def test_search_notes(notes):
    assert len(search_notes(notes, "robustness")) == 1
    assert len(search_notes(notes, "two percent")) == 1  # searches the saved passage too
    assert len(search_notes(notes, paper_id="p1")) == 2
    assert [n["kind"] for n in search_notes(notes, kinds=["ai"])] == ["ai"]
    assert len(search_notes(notes, tags=["todo"])) == 1
    assert all_tags(notes) == ["method", "regression", "todo"]


def test_export_markdown_distinguishes_ai_and_personal(notes):
    md = export_markdown(notes, "My notes")
    assert md.startswith("# My notes")
    assert "## Paper One" in md and "## Paper Two" in md
    assert "**AI-generated explanation (unverified):**" in md
    assert "**My note:**" in md
    assert "> β is the slope" in md
    assert "p. 3" in md and "model: claude-opus-5-5" in md


def test_export_txt(notes):
    txt = export_txt(notes)
    assert "AI-GENERATED EXPLANATION (UNVERIFIED):" in txt and "MY NOTE:" in txt
    assert "ORIGINAL PASSAGE:" in txt and "Lending falls" in txt


# --------------------------------------------------------------------------- library


def test_session_store_add_update_search_remove(simple_pdf):
    state: dict = {}
    store = library.SessionStore(state)
    ws = store.load()
    paper = process_pdf(simple_pdf, "simple.pdf")
    record, is_new = library.add_paper(ws, library.paper_metadata(paper, len(simple_pdf)))
    store.put_pdf(record["id"], simple_pdf)
    assert is_new and library.add_paper(ws, library.paper_metadata(paper, 1))[1] is False
    ws["notes"].append(make_note(record["id"], record["title"], "personal", "hi"))
    topic = lr.create_topic(ws, "Banks")
    lr.add_paper_to_topic(topic, record)

    library.update_metadata(ws, record["id"], title="Corrected Title", year="2021")
    assert ws["papers"][record["id"]]["metadata_source"] == "edited by user"
    assert ws["notes"][0]["paper_title"] == "Corrected Title"
    assert topic["rows"][record["id"]]["title"]["value"] == "Corrected Title"
    with pytest.raises(ValidationError):
        library.update_metadata(ws, record["id"], title="  ")

    assert library.search_papers(ws, "corrected")
    assert library.search_papers(ws, "nothing-matches") == []

    library.remove_paper(ws, store, record["id"], delete_notes=True)
    assert ws["papers"] == {} and ws["notes"] == [] and topic["paper_ids"] == []
    assert store.get_pdf(record["id"]) is None


def test_local_store_roundtrip_and_corruption(tmp_path, simple_pdf):
    store = library.LocalStore(tmp_path)
    ws = store.load()
    ws["papers"]["abc123"] = {"id": "abc123", "title": "T"}
    store.save(ws)
    store.put_pdf("abc123", simple_pdf)
    reopened = library.LocalStore(tmp_path)
    assert reopened.load()["papers"]["abc123"]["title"] == "T"
    assert reopened.get_pdf("abc123") == simple_pdf
    assert reopened.get_pdf("../../etc/passwd") is None  # path traversal is rejected
    (tmp_path / "workspace.json").write_text("{broken", encoding="utf-8")
    assert library.LocalStore(tmp_path).load() == library.empty_workspace()
    assert list(tmp_path.glob("workspace.corrupt-*.json"))


def test_backup_roundtrip_and_validation(simple_pdf):
    ws = library.empty_workspace()
    ws["papers"]["p1"] = {"id": "p1", "title": "T"}
    ws["notes"].append(make_note("p1", "T", "personal", "note"))
    raw = library.export_backup(ws, {"p1": simple_pdf})
    restored, pdfs = library.import_backup(raw)
    assert restored["papers"]["p1"]["title"] == "T" and pdfs["p1"] == simple_pdf

    merged = library.merge_workspaces(library.empty_workspace(), restored)
    merged = library.merge_workspaces(merged, restored)
    assert len(merged["notes"]) == 1  # no duplicates on repeated import

    for bad in (b"not json", json.dumps({"x": 1}).encode(),
                json.dumps({"readray_backup": True, "workspace": {"version": 99}}).encode(),
                json.dumps({"readray_backup": True, "workspace": {"version": 1, "papers": {"../x": {"id": "../x"}},
                                                                  "notes": [], "topics": {}}}).encode()):
        with pytest.raises(ValidationError):
            library.import_backup(bad)


# --------------------------------------------------------------------------- literature review


@pytest.fixture
def topic():
    ws = library.empty_workspace()
    t = lr.create_topic(ws, "Bank lending", "How monetary policy reaches firms")
    lr.add_paper_to_topic(t, {"id": "p1", "title": "Paper One", "authors": "A. Author", "year": "2020"})
    lr.add_paper_to_topic(t, {"id": "p2", "title": "Paper Two", "authors": "", "year": ""})
    return t


def test_topic_rows_prefilled_from_metadata(topic):
    assert topic["rows"]["p1"]["title"] == {**topic["rows"]["p1"]["title"], "value": "Paper One", "source": "metadata"}
    assert topic["rows"]["p2"]["authors"]["source"] == ""
    with pytest.raises(ValueError):
        lr.create_topic(library.empty_workspace(), "  ")


def test_manual_edits_and_dataframe_roundtrip(topic):
    assert lr.set_cell(topic, "p1", "methodology", "Diff-in-diff") is True
    assert lr.set_cell(topic, "p1", "methodology", "Diff-in-diff") is False
    df = lr.matrix_dataframe(topic)
    assert list(df["paper_id"]) == ["p1", "p2"]
    edited = df.copy()
    edited.loc[edited["paper_id"] == "p2", "Main findings"] = "Lending falls"
    edited.loc[edited["paper_id"] == "p2", "Year"] = pd.NA
    assert lr.apply_matrix_edits(topic, edited) == 1
    assert topic["rows"]["p2"]["findings"]["source"] == "manual"


def test_ai_extraction_parsing_drops_unverifiable_pages_and_marks_missing(topic):
    response = """```json
    {"research_question": {"value": "Does policy affect lending?", "pages": [1, 99]},
     "methodology": {"value": null, "pages": []},
     "data": "500 banks",
     "findings": {"value": "N/A", "pages": [2]},
     "limitations": {"value": "Single country", "pages": ["3"]}}
    ```"""
    cells = lr.parse_extraction(response, valid_pages={1, 2, 3})
    assert cells["research_question"]["pages"] == [1]  # page 99 was not in the excerpts
    assert cells["methodology"]["value"] == lr.NOT_STATED
    assert cells["findings"]["value"] == lr.NOT_STATED
    assert cells["data"]["value"] == "500 banks"
    assert cells["limitations"]["pages"] == [3]
    assert cells["gap"]["value"] == lr.NOT_STATED  # absent from the response
    assert all(c["source"] == "ai" for c in cells.values())

    lr.set_cell(topic, "p1", "data", "My manual sample description")
    updated = lr.apply_extraction(topic, "p1", cells)
    assert "data" not in updated and topic["rows"]["p1"]["data"]["value"] == "My manual sample description"
    assert "data" in lr.apply_extraction(topic, "p1", cells, overwrite_manual=True)


def test_exports_include_provenance(topic):
    lr.set_cell(topic, "p1", "findings", "Lending | falls")
    lr.apply_extraction(topic, "p2", {"gap": lr.cell("Needs micro data", "ai", [4])})
    csv = lr.export_csv(topic)
    assert "Main findings — source" in csv and "manual" in csv and "AI draft (pp. 4)" in csv
    md = lr.export_markdown(topic)
    assert "Lending \\| falls" in md and "_(AI draft; pp. 4)_" in md
    assert md.startswith("# Literature review: Bank lending")


def test_synthesis_prompt_uses_only_matrix_and_tags_provenance(topic):
    lr.set_cell(topic, "p1", "findings", "Effect is large")
    lr.apply_extraction(topic, "p2", {"findings": lr.cell("Effect is small", "ai", [2])})
    prompt = lr.build_synthesis_prompt(topic)
    assert "Effect is large" in prompt and "[manual]" in prompt and "[AI draft]" in prompt
    assert "do not claim two papers disagree" in prompt
    assert 0 < lr.filled_ratio(topic) < 1


def test_ai_extraction_repairs_excerpt_ids_and_string_pages():
    # Real failure mode seen with a small local model: excerpt ids instead of page numbers (invalid JSON).
    response = """```
    {"research_question": {"value": "Q", "pages": [E3]},
     "findings": {"value": "F", "pages": [E2, E10, "p. 4", "unknown"]},
     "data": {"value": "D", "pages": []},
     "limitations": {"value": ["Not causal", "One country"], "pages": [3]}}
    ```"""
    cells = lr.parse_extraction(response, valid_pages={1, 2, 3}, excerpt_pages={"E2": 1, "E3": 2})
    assert cells["research_question"]["pages"] == [2]
    assert cells["findings"]["pages"] == [1]  # E10 unknown -> dropped; p. 4 not among sent pages -> dropped
    assert cells["data"]["value"] == "D"
    assert cells["limitations"]["value"] == "Not causal; One country"


def test_excerpt_pages_match_prompt_labels(sample_pdf):
    from src.retrieval import BM25Index, build_chunks, build_paper_context

    paper = process_pdf(sample_pdf)
    chunks = build_chunks(paper)
    bundle = build_paper_context(paper, chunks, BM25Index(chunks), 8000)
    prompt = bundle.to_prompt()
    for label, page in bundle.excerpt_pages().items():
        assert f"[{label} | p. {page} |" in prompt

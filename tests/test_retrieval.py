from src.pdf_processor import process_pdf
from src.retrieval import (
    BM25Index, build_chunks, build_paper_context, build_section_context, build_selection_context,
    locate_selection, section_names,
)


def _prep(data):
    paper = process_pdf(data)
    chunks = build_chunks(paper)
    return paper, chunks, BM25Index(chunks)


def test_chunks_never_span_pages_and_keep_sections(sample_pdf):
    paper, chunks, _ = _prep(sample_pdf)
    assert chunks
    for c in chunks:
        assert c.text.strip()
        assert c.text.splitlines()[0] in paper.page(c.page).text
    assert any(c.section == "3 Methodology" and c.kind == "equation" for c in chunks)
    assert all(c.in_references for c in chunks if c.section == "References")


def test_bm25_ranks_relevant_chunk_first(sample_pdf):
    _, chunks, index = _prep(sample_pdf)
    top = index.top("limitations external validity credit rationing", k=1)[0]
    assert top.section == "5 Limitations"
    assert index.top("", k=3) == []


def test_bm25_downweights_references(sample_pdf):
    _, _, index = _prep(sample_pdf)
    ranked = index.score("fictional review bank lending channel")
    assert ranked and not ranked[0][1].in_references


def test_locate_selection_exact_and_fuzzy(sample_pdf):
    _, chunks, _ = _prep(sample_pdf)
    exact = locate_selection(chunks, "Standard errors are clustered by bank.")
    assert exact is not None and exact.page == 2 and exact.section == "3 Methodology"
    # extraction-style differences: hyphenation, case, punctuation
    fuzzy = locate_selection(chunks, "STANDARD ERRORS are clus-\ntered by bank")
    assert fuzzy is not None and fuzzy.index == exact.index
    assert locate_selection(chunks, "completely unrelated words about astronomy and telescopes") is None


def test_selection_context_includes_neighbours_captions_and_pages(sample_pdf):
    paper, chunks, index = _prep(sample_pdf)
    bundle = build_selection_context(paper, chunks, index, "The coefficient δ captures the additional spread response of small firms", 2)
    roles = [s.role for s in bundle.sources]
    assert bundle.located and bundle.selection_page == 2 and bundle.section == "3 Methodology"
    assert "containing paragraph" in roles and "before" in roles
    assert any(s.role == "equation" or "(1)" in s.text for s in bundle.sources)
    prompt = bundle.to_prompt()
    assert "SELECTED PASSAGE (page 2" in prompt and "| p. 2 |" in prompt
    assert set(bundle.pages()) <= set(range(1, paper.page_count + 1))


def test_selection_referencing_table_pulls_caption(sample_pdf):
    paper, chunks, index = _prep(sample_pdf)
    bundle = build_selection_context(paper, chunks, index, "Table 2 presents the main estimates.", 2)
    assert any(s.text.startswith("Table 2:") for s in bundle.sources)  # as neighbour or referenced caption


def test_unlocated_selection_is_flagged(sample_pdf):
    paper, chunks, index = _prep(sample_pdf)
    bundle = build_selection_context(paper, chunks, index, "Text the reader typed that is not in the paper at all, about volcanoes.", 1)
    assert not bundle.located
    assert "could not be located" in bundle.to_prompt()


def test_context_respects_budget(sample_pdf):
    paper, chunks, index = _prep(sample_pdf)
    small = build_selection_context(paper, chunks, index, "Standard errors are clustered by bank.", 2, budget_chars=1500)
    total = sum(len(s.text) for s in small.sources) + len(small.abstract)
    assert total <= 1500 + 200  # selection + small slack for truncation markers
    assert small.truncated


def test_paper_and_section_context(sample_pdf):
    paper, chunks, index = _prep(sample_pdf)
    overview = build_paper_context(paper, chunks, index, budget_chars=8000)
    assert any(s.role == "section outline" for s in overview.sources)
    assert overview.truncated  # a sample of the paper, never the whole thing
    names = section_names(chunks)
    assert "References" not in names and "4 Results" in names
    sec = build_section_context(paper, chunks, "4 Results")
    assert sec.located and all(s.section == "4 Results" for s in sec.sources)
    assert build_section_context(paper, chunks, "No such section").located is False

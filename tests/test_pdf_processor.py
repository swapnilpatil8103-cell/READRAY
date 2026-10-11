import pymupdf
import pytest

from src.pdf_processor import (
    CorruptPDFError, EmptyPDFError, EncryptedPDFError, crop_region_png, extract_tables, highlight_rects,
    process_pdf, render_page_png, search_document, text_in_region,
)


def test_extracts_text_per_page_with_correct_numbers(simple_pdf):
    paper = process_pdf(simple_pdf, "simple.pdf")
    assert paper.page_count == 3
    assert [p.number for p in paper.pages] == [1, 2, 3]
    assert "500 banks" in paper.page(2).text
    assert "500 banks" not in paper.page(1).text
    assert "two percent" in paper.page(3).text


def test_page_out_of_range(simple_pdf):
    paper = process_pdf(simple_pdf)
    with pytest.raises(IndexError):
        paper.page(4)
    with pytest.raises(IndexError):
        paper.page(0)


def test_title_from_largest_text_when_metadata_empty(simple_pdf):
    paper = process_pdf(simple_pdf)
    assert paper.title == "Liquidity Shocks and Bank Lending"
    assert "Jane Doe" in paper.authors


def test_headings_captions_and_abstract(simple_pdf):
    paper = process_pdf(simple_pdf)
    headings = [(h.page, h.text) for h in paper.headings]
    assert (1, "1 Introduction") in headings
    assert (2, "2 Data") in headings
    assert [(c.page, c.kind) for c in paper.captions] == [(2, "table")]
    assert paper.abstract.startswith("We examine how liquidity shocks")


def test_section_for_page(simple_pdf):
    paper = process_pdf(simple_pdf)
    assert paper.section_for(3) == "3 Results"


def test_sample_paper_structure(sample_pdf):
    paper = process_pdf(sample_pdf, "sample_paper.pdf")
    assert paper.title.startswith("Interest Rate Announcements")
    assert paper.authors == "A. Example and B. Sample"
    texts = [h.text for h in paper.headings]
    assert "3 Methodology" in texts and "References" in texts
    assert not any("Fictional Event Study" in t for t in texts), "title lines must not become headings"
    assert {c.kind for c in paper.captions} == {"table", "figure"}
    assert not any(c.text.startswith("Table 1 reports") for c in paper.captions)
    assert any("(1)" in e.text for e in paper.equations)


@pytest.mark.parametrize("data", [b"", b"not a pdf at all", b"%PDF-1.4\n%garbage without objects"])
def test_invalid_pdfs_raise_user_friendly_errors(data):
    with pytest.raises((CorruptPDFError, EmptyPDFError)) as exc:
        process_pdf(data)
    assert str(exc.value)


def test_encrypted_pdf_requires_password(simple_pdf):
    doc = pymupdf.open(stream=simple_pdf, filetype="pdf")
    encrypted = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    doc.close()
    with pytest.raises(EncryptedPDFError):
        process_pdf(encrypted)
    with pytest.raises(EncryptedPDFError, match="not accepted"):
        process_pdf(encrypted, password="wrong")
    assert process_pdf(encrypted, password="secret").page_count == 3


def test_scanned_page_detection():
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), False)
    pix.clear_with(200)
    page.insert_image(pymupdf.Rect(72, 72, 400, 400), pixmap=pix)
    data = doc.tobytes()
    paper = process_pdf(data)
    assert paper.scanned_pages == [1]
    assert paper.is_mostly_scanned
    assert any("scanned" in w for w in paper.warnings)


def test_search_reports_pages_and_counts(simple_pdf):
    paper = process_pdf(simple_pdf)
    hits = search_document(paper, "LIQUIDITY")
    assert [h.page for h in hits] == [1, 2, 3]
    assert hits[0].count >= 2
    assert search_document(paper, "x") == []
    assert search_document(paper, "nonexistent phrase") == []


def test_render_crop_and_highlights(simple_pdf):
    png = render_page_png(simple_pdf, 1, zoom=1.0)
    assert png.startswith(b"\x89PNG")
    crop = crop_region_png(simple_pdf, 1, (0.1, 0.05, 0.9, 0.3))
    assert crop.startswith(b"\x89PNG")
    with pytest.raises(ValueError):
        crop_region_png(simple_pdf, 1, (0.5, 0.5, 0.5, 0.5))
    rects = highlight_rects(simple_pdf, 2, "500 banks")
    assert rects and all(0 <= v <= 1 for r in rects for v in r)


def test_text_in_region(simple_pdf):
    paper = process_pdf(simple_pdf)
    assert "Liquidity Shocks" in text_in_region(paper, 1, (0, 0, 1, 0.2))
    assert text_in_region(paper, 1, (0, 0.95, 1, 1)) == ""


def test_extract_tables_from_sample(sample_pdf):
    frames = extract_tables(sample_pdf, 1)
    assert isinstance(frames, list)
    if frames:  # table detection is heuristic; when found, the shape must match Table 1
        assert frames[0].shape[1] == 5

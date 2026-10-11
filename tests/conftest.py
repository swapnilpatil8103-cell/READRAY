import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_pdf(pages: list[list[tuple[str, float]]], metadata: dict | None = None) -> bytes:
    """Build a PDF where each page is a list of (text, fontsize) lines placed top to bottom."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page(width=612, height=792)
        y = 72
        for text, size in lines:
            rect = pymupdf.Rect(72, y, 540, y + size * 6)
            page.insert_textbox(rect, text, fontsize=size, fontname="helv")
            used = max(1, len(text) // int(540 / (size * 0.5))) + 1
            y += size * 1.4 * used + 10
    if metadata:
        doc.set_metadata(metadata)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def simple_pdf() -> bytes:
    return make_pdf(
        [
            [("Liquidity Shocks and Bank Lending", 20), ("Jane Doe and John Roe", 12),
             ("Abstract: We examine how liquidity shocks affect bank lending to small firms using a panel of banks.", 10),
             ("1 Introduction", 14),
             ("Banks transmit monetary policy to firms. This paper studies the lending channel in detail.", 10)],
            [("2 Data", 14),
             ("The sample contains 500 banks observed quarterly. We measure liquidity as cash over assets.", 10),
             ("Table 1: Summary statistics for the bank sample.", 9)],
            [("3 Results", 14),
             ("Lending falls by two percent after a negative liquidity shock. The effect is larger for small banks.", 10)],
        ],
        metadata={"title": "", "author": ""},
    )


@pytest.fixture
def sample_pdf() -> bytes:
    return (ROOT / "assets" / "sample_paper.pdf").read_bytes()

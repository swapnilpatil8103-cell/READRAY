"""Streamlit wrapper for the bundled PDF viewer component.

The component shows a PyMuPDF-rendered page image with an invisible, positioned text layer built from
PyMuPDF's line boxes, so the browser's native text selection works. It also supports drag-to-crop and
search-hit highlighting. It loads no external scripts.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import streamlit.components.v1 as components

from .pdf_processor import PageContent

_FRONTEND = Path(__file__).parent / "pdf_viewer_frontend"
_component = components.declare_component("readray_pdf_viewer", path=str(_FRONTEND))

ZOOM_OPTIONS = {"Fit width": "fit-width", "Fit page": "fit-page", "75%": 0.75, "100%": 1.0, "125%": 1.25, "150%": 1.5, "200%": 2.0}


def render_scale_for(zoom) -> float:
    """Raster scale for PyMuPDF so text stays crisp at the chosen zoom (and on high-DPI screens)."""
    if isinstance(zoom, str):
        return 2.0
    return max(1.5, min(3.0, float(zoom) * 2.0))


def text_layer_lines(page: PageContent) -> list[list]:
    w, h = page.width or 1, page.height or 1
    out = []
    for ln in page.lines:
        text = ln.text.rstrip()
        if not text.strip():
            continue
        x0, y0, x1, y1 = ln.bbox
        out.append([round(x0 / w, 5), round(y0 / h, 5), round(x1 / w, 5), round(y1 / h, 5), text])
    return out


def pdf_viewer(
    png: bytes,
    page: PageContent,
    doc_id: str,
    zoom,
    highlights: list[tuple[float, float, float, float]] | None = None,
    height: int = 860,
    key: str = "pdf_viewer",
):
    """Render the viewer. Returns the last event dict ({'event': 'selection'|'crop', ...}) or None."""
    image = "data:image/png;base64," + base64.b64encode(png).decode()
    highlights = [list(r) for r in (highlights or [])]
    sig = hashlib.md5(f"{doc_id}|{page.number}|{zoom}|{height}|{highlights}|{len(png)}".encode()).hexdigest()
    return _component(
        image=image,
        page=page.number,
        doc=doc_id,
        page_width=page.width,
        page_height=page.height,
        lines=text_layer_lines(page),
        highlights=highlights,
        zoom=zoom,
        height=height,
        sig=sig,
        key=key,
        default=None,
    )

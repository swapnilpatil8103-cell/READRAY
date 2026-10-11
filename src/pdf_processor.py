"""PDF loading, page-aware text extraction, structure detection, rendering and cropping.

Everything here works on raw PDF bytes so it is easy to test and never touches disk.
Page numbers exposed by this module are 1-based (what a reader sees).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf as fitz  # PyMuPDF

from .utils import clean_text, sha256_hex

MAX_PAGES = 600
SCANNED_TEXT_THRESHOLD = 25  # characters; below this a page with images is treated as scanned

HEADING_WORDS = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*\.?|[IVX]+\.|[A-H]\.)\s*)?"
    r"(abstract|introduction|background|motivation|related work|literature review|literature|"
    r"theory|theoretical framework|model|the model|hypothes[ie]s|data|data and methodology|"
    r"methodology|methods?|research design|empirical (?:strategy|results|analysis|framework)|"
    r"identification|estimation|results|findings|analysis|robustness(?: checks)?|discussion|"
    r"implications|limitations|conclusions?|concluding remarks|summary|references|bibliography|"
    r"appendix(?: [a-z0-9]+)?|acknowledg(?:e)?ments?)\s*:?$",
    re.IGNORECASE,
)
NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+){0,3}\.?|[IVX]{1,5}\.)\s+[A-Z][^.!?]{2,90}$")
# A caption label must be followed by punctuation, a line end, or a capitalised word — so body text such as
# "Table 1 reports…" is not mistaken for a caption.
CAPTION_RE = re.compile(
    r"^(?i:fig(?:ure)?s?\.?|table|exhibit|chart|panel|graph)\s*[A-Z]?\d+[A-Za-z]?(?:\s*[.:|\-–—]|\s*$|\s*\n|\s+[A-Z(])"
)
MATH_CHARS = set("=+−–×÷∑∏∫√≤≥≈≠∂∇∞αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ^_|‖∈∉⊂⊆∪∩→←↔⇒∀∃")
EQ_NUMBER_RE = re.compile(r"\(\s*[A-Z]?\d{1,3}[a-z]?\s*\)\s*$")


class PDFProcessingError(Exception):
    """Base class. `str(error)` is always safe and useful to show to an end user."""


class CorruptPDFError(PDFProcessingError):
    pass


class EncryptedPDFError(PDFProcessingError):
    pass


class EmptyPDFError(PDFProcessingError):
    pass


class TooManyPagesError(PDFProcessingError):
    pass


@dataclass
class TextLine:
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool


@dataclass
class TextBlock:
    text: str
    bbox: tuple[float, float, float, float]
    page: int
    size: float
    bold: bool
    n_lines: int
    kind: str = "body"  # body | heading | caption | equation


@dataclass
class PageContent:
    number: int
    width: float
    height: float
    text: str
    blocks: list[TextBlock]
    lines: list[TextLine]
    has_images: bool
    image_only: bool


@dataclass
class Heading:
    page: int
    text: str


@dataclass
class Caption:
    page: int
    kind: str  # figure | table
    text: str


@dataclass
class EquationCandidate:
    page: int
    text: str


@dataclass
class SearchHit:
    page: int
    snippet: str
    count: int


@dataclass
class PaperDocument:
    doc_id: str
    filename: str
    page_count: int
    pages: list[PageContent]
    title: str
    authors: str
    year: str
    abstract: str
    headings: list[Heading] = field(default_factory=list)
    captions: list[Caption] = field(default_factory=list)
    equations: list[EquationCandidate] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    body_font_size: float = 10.0

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def is_mostly_scanned(self) -> bool:
        return self.page_count > 0 and len(self.scanned_pages) / self.page_count >= 0.5

    def page(self, number: int) -> PageContent:
        if not 1 <= number <= self.page_count:
            raise IndexError(f"Page {number} is out of range 1–{self.page_count}.")
        return self.pages[number - 1]

    def section_for(self, page: int, y: float | None = None) -> str:
        """Most recent heading at or before (page, y)."""
        current = ""
        for p in self.pages[:page]:
            for block in p.blocks:
                if p.number == page and y is not None and block.bbox[1] > y:
                    return current
                if block.kind == "heading":
                    current = block.text
        return current


# --------------------------------------------------------------------------- opening


def open_pdf(data: bytes, password: str | None = None) -> fitz.Document:
    """Open PDF bytes, translating PyMuPDF failures into user-facing errors."""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several types for malformed files
        raise CorruptPDFError("The PDF could not be opened. It may be corrupted or not a real PDF.") from exc
    if doc.needs_pass:
        if not password or not doc.authenticate(password):
            doc.close()
            raise EncryptedPDFError(
                "This PDF is password-protected." + (" The password was not accepted." if password else "")
            )
    if doc.page_count == 0:
        doc.close()
        raise EmptyPDFError("The PDF contains no pages.")
    if doc.page_count > MAX_PAGES:
        count = doc.page_count
        doc.close()
        raise TooManyPagesError(f"The PDF has {count} pages; Readray supports up to {MAX_PAGES}.")
    return doc


# --------------------------------------------------------------------------- extraction


def _extract_page(page: fitz.Page, number: int) -> PageContent:
    raw = page.get_text("dict", sort=True)
    blocks: list[TextBlock] = []
    lines: list[TextLine] = []
    has_images = bool(page.get_images(full=False))
    for b in raw.get("blocks", []):
        if b.get("type") != 0:
            has_images = True
            continue
        block_lines: list[str] = []
        sizes: list[float] = []
        bold_chars = total_chars = 0
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans)
            if not text.strip():
                continue
            size = max((s.get("size", 0.0) for s in spans), default=0.0)
            bold = any((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower() for s in spans)
            lines.append(TextLine(text=text, bbox=tuple(line["bbox"]), size=size, bold=bold))
            block_lines.append(text)
            for s in spans:
                n = len(s.get("text", "").strip())
                total_chars += n
                if (s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower():
                    bold_chars += n
                sizes.extend([s.get("size", 0.0)] * max(1, n // 20))
        text = clean_text("\n".join(block_lines))
        if not text:
            continue
        blocks.append(
            TextBlock(
                text=text,
                bbox=tuple(b["bbox"]),
                page=number,
                size=round(Counter(round(s, 1) for s in sizes).most_common(1)[0][0], 1) if sizes else 0.0,
                bold=total_chars > 0 and bold_chars / total_chars > 0.6,
                n_lines=len(block_lines),
            )
        )
    page_text = "\n\n".join(b.text for b in blocks)
    image_only = has_images and len(page_text.strip()) < SCANNED_TEXT_THRESHOLD
    return PageContent(
        number=number,
        width=page.rect.width,
        height=page.rect.height,
        text=page_text,
        blocks=blocks,
        lines=lines,
        has_images=has_images,
        image_only=image_only,
    )


def _body_font_size(pages: list[PageContent]) -> float:
    counter: Counter[float] = Counter()
    for p in pages:
        for b in p.blocks:
            counter[b.size] += len(b.text)
    return counter.most_common(1)[0][0] if counter else 10.0


def _is_heading(block: TextBlock, body_size: float) -> bool:
    text = block.text.strip()
    if block.n_lines > 2 or len(text) > 110 or len(text) < 3:
        return False
    one_line = " ".join(text.split())
    if HEADING_WORDS.match(one_line):
        return True
    if one_line.endswith((".", ",", ";")) and not re.match(r"^\d+\.\s*$", one_line):
        return False
    if sum(c.isdigit() for c in one_line) > len(one_line) * 0.4:
        return False
    larger = block.size >= body_size + 1.0
    if NUMBERED_HEADING.match(one_line) and (block.bold or larger):
        return True
    return larger and block.bold and len(one_line.split()) <= 12


def _math_density(text: str) -> float:
    stripped = text.replace(" ", "")
    if not stripped:
        return 0.0
    return sum(c in MATH_CHARS for c in stripped) / len(stripped)


def _is_equation(text: str) -> bool:
    one_line = " ".join(text.split())
    if len(one_line) < 3 or len(one_line) > 300:
        return False
    density = _math_density(one_line)
    words = re.findall(r"[A-Za-z]{4,}", one_line)
    if EQ_NUMBER_RE.search(one_line) and ("=" in one_line or density > 0.05) and len(words) <= 6:
        return True
    return density >= 0.12 and "=" in one_line and len(words) <= 4


def _classify_blocks(pages: list[PageContent], body_size: float) -> None:
    for p in pages:
        for b in p.blocks:
            if CAPTION_RE.match(b.text):
                b.kind = "caption"
            elif _is_heading(b, body_size):
                b.kind = "heading"
            elif _is_equation(b.text):
                b.kind = "equation"


# --------------------------------------------------------------------------- metadata

BAD_TITLE = re.compile(r"^(untitled|microsoft word|document\d*|slide\s*\d*|\s*)$|\.(docx?|tex|pdf|dvi)$|^[\w\-]+$", re.IGNORECASE)


def _guess_title(doc: fitz.Document, first: PageContent) -> str:
    meta_title = (doc.metadata or {}).get("title", "") or ""
    meta_title = " ".join(meta_title.split())
    if meta_title.lower().startswith("microsoft word - "):
        meta_title = meta_title[17:]
    if len(meta_title) >= 8 and not BAD_TITLE.search(meta_title):
        return meta_title
    candidates = [
        b
        for b in first.blocks
        if b.bbox[1] < first.height * 0.5 and 8 <= len(b.text) <= 300 and b.kind != "caption"
        and not re.match(r"(?i)^(abstract|keywords|jel|arxiv|doi|http|received|journal|vol\.|working paper)", b.text)
    ]
    if not candidates:
        return ""
    best = max(candidates, key=lambda b: (b.size, -b.bbox[1]))
    # Multi-line titles are often split into several blocks of the same size: merge them.
    parts, last = [best.text], best
    for b in first.blocks[first.blocks.index(best) + 1 :]:
        if abs(b.size - best.size) <= 0.5 and 0 <= b.bbox[1] - last.bbox[3] < best.size * 1.5 and len(" ".join(parts)) < 300:
            parts.append(b.text)
            last = b
        else:
            break
    return " ".join(" ".join(parts).split())


def _mark_title_blocks(first: PageContent, title: str) -> int:
    """Tag page-1 blocks that make up the title (so they are not treated as section headings).
    Returns the index of the last title block, or -1."""
    norm_title = " ".join(title.lower().split())
    last = -1
    for i, b in enumerate(first.blocks):
        text = " ".join(b.text.lower().split())
        if b.bbox[1] < first.height * 0.5 and len(text) >= 6 and text in norm_title:
            b.kind = "title"
            last = i
    return last


def _guess_authors(doc: fitz.Document, first: PageContent, title: str) -> str:
    meta_author = " ".join(((doc.metadata or {}).get("author", "") or "").split())
    if 3 <= len(meta_author) <= 300 and not re.search(r"(?i)^(user|admin|owner|unknown|author)$", meta_author):
        return meta_author
    last_title = next((i for i in range(len(first.blocks) - 1, -1, -1) if first.blocks[i].kind == "title"), -1)
    if not title or last_title < 0:
        return ""
    for nxt in first.blocks[last_title + 1 : last_title + 4]:
        line = " ".join(nxt.text.split())
        if re.match(r"(?i)^(abstract|keywords|jel|\d)", line) or len(line) > 250:
            break
        words = re.findall(r"[A-Za-z][A-Za-z.'\-]+", line)
        caps = [w for w in words if w[0].isupper() or w.lower() in {"and", "de", "van", "von"}]
        if 2 <= len(words) <= 40 and len(caps) / max(1, len(words)) > 0.7 and not line.endswith("."):
            return re.sub(r"[\d*†‡§¶]+", "", line).strip(" ,")
    return ""


def _guess_year(first: PageContent) -> str:
    head = first.text[:2500]
    for pattern in (r"©\s*((?:19|20)\d{2})", r"(?i)(?:published|accepted|received)[^\n]{0,40}?((?:19|20)\d{2})",
                    r"(?i)\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+((?:19|20)\d{2})"):
        m = re.search(pattern, head)
        if m:
            return m.group(1)
    return ""


def _extract_abstract(pages: list[PageContent]) -> str:
    for p in pages[:2]:
        for i, b in enumerate(p.blocks):
            m = re.match(r"(?is)^\s*abstract\s*[:.\-–—]?\s*(.*)$", b.text)
            if not m:
                continue
            parts = [m.group(1).strip()] if m.group(1).strip() else []
            for nxt in p.blocks[i + 1 :]:
                if nxt.kind == "heading" or re.match(r"(?i)^(keywords|key words|jel|1\.?\s+introduction|introduction)", nxt.text):
                    break
                parts.append(nxt.text)
                if sum(len(x) for x in parts) > 2500:
                    break
            abstract = " ".join(" ".join(parts).split())
            if len(abstract) > 80:
                return abstract[:3000]
    return ""


# --------------------------------------------------------------------------- public API


def process_pdf(data: bytes, filename: str = "paper.pdf", password: str | None = None) -> PaperDocument:
    """Parse PDF bytes into a page-aware PaperDocument. Raises PDFProcessingError subclasses."""
    doc = open_pdf(data, password)
    try:
        pages: list[PageContent] = []
        warnings: list[str] = []
        for index in range(doc.page_count):
            try:
                pages.append(_extract_page(doc.load_page(index), index + 1))
            except Exception:  # one bad page must not sink the whole paper
                pages.append(PageContent(index + 1, 612, 792, "", [], [], False, False))
                warnings.append(f"Text on page {index + 1} could not be extracted.")
        body = _body_font_size(pages)
        _classify_blocks(pages, body)
        first = pages[0]
        title = _guess_title(doc, first)
        if title:
            _mark_title_blocks(first, title)
        authors = _guess_authors(doc, first, title)
        scanned = [p.number for p in pages if p.image_only]
        if scanned:
            warnings.append(
                f"{len(scanned)} page(s) appear to be scanned images without a text layer "
                f"({', '.join(map(str, scanned[:10]))}{'…' if len(scanned) > 10 else ''}). "
                "Their text cannot be read or sent to the AI unless OCR is available."
            )
        if sum(len(p.text) for p in pages) == 0 and not scanned:
            warnings.append("No extractable text was found in this PDF.")
        return PaperDocument(
            doc_id=sha256_hex(data)[:16],
            filename=filename,
            page_count=doc.page_count,
            pages=pages,
            title=title or filename.rsplit(".", 1)[0],
            authors=authors,
            year=_guess_year(first),
            abstract=_extract_abstract(pages),
            headings=[Heading(b.page, " ".join(b.text.split())) for p in pages for b in p.blocks if b.kind == "heading"],
            captions=[
                Caption(b.page, "table" if b.text.lower().startswith("table") else "figure", " ".join(b.text.split()))
                for p in pages for b in p.blocks if b.kind == "caption"
            ],
            equations=[EquationCandidate(b.page, b.text) for p in pages for b in p.blocks if b.kind == "equation"],
            scanned_pages=scanned,
            warnings=warnings,
            body_font_size=body,
        )
    finally:
        doc.close()


def search_document(paper: PaperDocument, query: str, max_hits: int = 200) -> list[SearchHit]:
    """Case-insensitive search across pages, one hit per page with a context snippet."""
    query = " ".join((query or "").split())
    if len(query) < 2:
        return []
    pattern = re.compile(re.escape(query).replace(r"\ ", r"\s+"), re.IGNORECASE)
    hits: list[SearchHit] = []
    for p in paper.pages:
        matches = list(pattern.finditer(p.text))
        if not matches:
            continue
        m = matches[0]
        start, end = max(0, m.start() - 90), min(len(p.text), m.end() + 90)
        snippet = ("…" if start else "") + " ".join(p.text[start:end].split()) + ("…" if end < len(p.text) else "")
        hits.append(SearchHit(page=p.number, snippet=snippet, count=len(matches)))
        if len(hits) >= max_hits:
            break
    return hits


def render_page_png(data: bytes, page_number: int, zoom: float = 1.5, password: str | None = None) -> bytes:
    doc = open_pdf(data, password)
    try:
        page = doc.load_page(page_number - 1)
        zoom = max(0.5, min(zoom, 4.0))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def highlight_rects(data: bytes, page_number: int, query: str, password: str | None = None) -> list[tuple[float, float, float, float]]:
    """Normalised (0–1) rectangles for every occurrence of `query` on a page."""
    query = " ".join((query or "").split())
    if len(query) < 2:
        return []
    doc = open_pdf(data, password)
    try:
        page = doc.load_page(page_number - 1)
        w, h = page.rect.width, page.rect.height
        return [(r.x0 / w, r.y0 / h, r.x1 / w, r.y1 / h) for r in page.search_for(query)[:200]]
    finally:
        doc.close()


def crop_region_png(
    data: bytes, page_number: int, rect: tuple[float, float, float, float], zoom: float = 3.0, password: str | None = None
) -> bytes:
    """Render a normalised (x0, y0, x1, y1) region of a page at high resolution."""
    x0, y0, x1, y1 = (max(0.0, min(1.0, float(v))) for v in rect)
    if x1 - x0 < 0.01 or y1 - y0 < 0.01:
        raise ValueError("The selected region is too small.")
    doc = open_pdf(data, password)
    try:
        page = doc.load_page(page_number - 1)
        w, h = page.rect.width, page.rect.height
        clip = fitz.Rect(x0 * w, y0 * h, x1 * w, y1 * h)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def text_in_region(paper: PaperDocument, page_number: int, rect: tuple[float, float, float, float]) -> str:
    """Extracted text whose lines fall inside a normalised region (used alongside crops)."""
    page = paper.page(page_number)
    x0, y0, x1, y1 = rect[0] * page.width, rect[1] * page.height, rect[2] * page.width, rect[3] * page.height
    inside = [
        ln.text for ln in page.lines
        if ln.bbox[0] >= x0 - 2 and ln.bbox[2] <= x1 + 2 and ln.bbox[1] >= y0 - 2 and ln.bbox[3] <= y1 + 2
    ]
    return clean_text("\n".join(inside))


def extract_tables(data: bytes, page_number: int, password: str | None = None) -> list:
    """Detect tables on a page with PyMuPDF and return pandas DataFrames (may be empty)."""
    import pandas as pd

    doc = open_pdf(data, password)
    try:
        page = doc.load_page(page_number - 1)
        try:
            found = page.find_tables()
        except Exception:
            return []
        frames = []
        for table in found.tables:
            try:
                df = table.to_pandas()
            except Exception:
                continue
            if df.shape[0] >= 1 and df.shape[1] >= 2:
                frames.append(df)
        return frames
    finally:
        doc.close()


def ocr_available() -> bool:
    """True when PyMuPDF can find Tesseract language data (optional system dependency)."""
    try:
        return bool(fitz.get_tessdata())
    except Exception:
        return False


def ocr_page_text(data: bytes, page_number: int, language: str = "eng", password: str | None = None) -> str:
    """OCR one page. Raises RuntimeError when Tesseract is unavailable."""
    if not ocr_available():
        raise RuntimeError("OCR is unavailable: Tesseract is not installed on this server.")
    doc = open_pdf(data, password)
    try:
        page = doc.load_page(page_number - 1)
        textpage = page.get_textpage_ocr(language=language, dpi=200, full=True)
        return clean_text(page.get_text("text", textpage=textpage))
    finally:
        doc.close()

"""Page-aware chunking, BM25 retrieval and context assembly for AI requests.

The goal is to send the model *enough* of the paper to explain a passage in context —
the passage itself, its neighbours, its section, related captions/equations, and the
abstract — without sending the whole paper.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .pdf_processor import PaperDocument
from .utils import normalize_for_match, tokenize, truncate

CHUNK_TARGET_CHARS = 1100
REFERENCE_HEADINGS = re.compile(r"(?i)^(?:\d+\.?\s*)?(references|bibliography|works cited)\s*$")


@dataclass
class Chunk:
    index: int
    page: int
    section: str
    text: str
    kind: str = "body"  # body | caption | equation | heading
    in_references: bool = False

    @property
    def label(self) -> str:
        section = f" · {self.section}" if self.section else ""
        return f"p. {self.page}{section}"


@dataclass
class SourcePassage:
    role: str  # selection | before | after | related | caption | equation | abstract | section
    page: int | None
    section: str
    text: str


@dataclass
class ContextBundle:
    title: str
    authors: str
    abstract: str
    selection: str
    selection_page: int | None
    section: str
    sources: list[SourcePassage] = field(default_factory=list)
    located: bool = False
    truncated: bool = False

    def pages(self) -> list[int]:
        return sorted({s.page for s in self.sources if s.page})

    def _labelled(self) -> list[SourcePassage]:
        return [s for s in self.sources if s.role not in {"selection", "abstract"}]

    def excerpt_pages(self) -> dict[str, int]:
        """Map excerpt labels used in the prompt ('E1', 'E2', …) to page numbers."""
        return {f"E{i}": s.page for i, s in enumerate(self._labelled(), 1) if s.page}

    def to_prompt(self) -> str:
        """Render the bundle as clearly delimited source material for the model."""
        parts = [f"PAPER TITLE: {self.title or 'Unknown'}"]
        if self.authors:
            parts.append(f"AUTHORS (auto-detected, may be incomplete): {self.authors}")
        if self.abstract:
            parts.append(f"ABSTRACT:\n{self.abstract}")
        if self.selection:
            where = f"page {self.selection_page}" if self.selection_page else "page unknown"
            section = f", section “{self.section}”" if self.section else ""
            located = "" if self.located else " (could not be located in the extracted text — treat its page as unverified)"
            parts.append(f"SELECTED PASSAGE ({where}{section}){located}:\n<<<\n{self.selection}\n>>>")
        labelled = self._labelled()
        if labelled:
            parts.append("SURROUNDING AND RELATED EXCERPTS FROM THE PAPER:")
            for i, s in enumerate(labelled, 1):
                page = f"p. {s.page}" if s.page else "page unknown"
                section = f" | section: {s.section}" if s.section else ""
                parts.append(f"[E{i} | {page} | {s.role}{section}]\n{s.text}")
        if self.truncated:
            parts.append("NOTE: Excerpts were shortened to fit the context budget; the full paper was not sent.")
        return "\n\n".join(parts)


# --------------------------------------------------------------------------- chunking


def build_chunks(paper: PaperDocument, target_chars: int = CHUNK_TARGET_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""
    in_refs = False
    buffer: list[str] = []
    buffer_page = 0

    def flush() -> None:
        nonlocal buffer
        if buffer:
            chunks.append(Chunk(len(chunks), buffer_page, section, "\n".join(buffer), "body", in_refs))
            buffer = []

    for page in paper.pages:
        for block in page.blocks:
            if block.kind == "heading":
                flush()
                section = " ".join(block.text.split())
                in_refs = bool(REFERENCE_HEADINGS.match(section)) or (in_refs and not re.match(r"(?i)^appendix", section))
                continue
            if block.kind in {"caption", "equation"}:
                flush()
                chunks.append(Chunk(len(chunks), page.number, section, block.text, block.kind, in_refs))
                continue
            if buffer and (buffer_page != page.number or sum(len(t) for t in buffer) + len(block.text) > target_chars):
                flush()
            if not buffer:
                buffer_page = page.number
            buffer.append(block.text)
        flush()  # never let a chunk span pages, so page citations stay exact
    return chunks


# --------------------------------------------------------------------------- BM25


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text + " " + c.section)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> list[tuple[float, Chunk]]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        results: list[tuple[float, Chunk]] = []
        for doc, length, chunk in zip(self.docs, self.lengths, self.chunks):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                s += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / denom
            if chunk.in_references:
                s *= 0.3  # bibliography entries match lots of keywords but rarely explain anything
            if s > 0:
                results.append((s, chunk))
        results.sort(key=lambda x: (-x[0], x[1].index))
        return results

    def top(self, query: str, k: int = 5, exclude: set[int] | None = None) -> list[Chunk]:
        exclude = exclude or set()
        return [c for _, c in self.score(query) if c.index not in exclude][:k]


# --------------------------------------------------------------------------- locating a selection


def locate_selection(chunks: list[Chunk], selection: str, page_hint: int | None = None) -> Chunk | None:
    """Find the chunk containing the selected text (exact normalised match, then token overlap)."""
    needle = normalize_for_match(selection)
    if not needle or not chunks:
        return None
    probe = needle[:160]
    ordered = sorted(chunks, key=lambda c: (c.page != page_hint, c.index)) if page_hint else chunks
    for chunk in ordered:
        if probe in normalize_for_match(chunk.text):
            return chunk
    # Selections can span chunk boundaries or contain extraction differences: fall back to overlap.
    sel_tokens = set(tokenize(selection, drop_stopwords=False))
    if len(sel_tokens) < 3:
        return None
    best, best_score = None, 0.0
    for chunk in ordered:
        tokens = set(tokenize(chunk.text, drop_stopwords=False))
        if not tokens:
            continue
        score = len(sel_tokens & tokens) / len(sel_tokens)
        if page_hint and chunk.page == page_hint:
            score += 0.05
        if score > best_score:
            best, best_score = chunk, score
    return best if best_score >= 0.6 else None


FIG_REF_RE = re.compile(r"(?i)\b(fig(?:ure)?s?\.?|table|exhibit)\s*([A-Z]?\d+[a-z]?)")
EQ_REF_RE = re.compile(r"(?i)\b(?:eq(?:uation)?s?\.?)\s*\(?(\d+)\)?")


def _referenced_captions(chunks: list[Chunk], text: str) -> list[Chunk]:
    wanted = {(("table" if m.group(1).lower().startswith("table") else "figure"), m.group(2).lower()) for m in FIG_REF_RE.finditer(text)}
    if not wanted:
        return []
    found = []
    for c in chunks:
        if c.kind != "caption":
            continue
        m = FIG_REF_RE.match(c.text)
        if m:
            key = ("table" if m.group(1).lower().startswith("table") else "figure", m.group(2).lower())
            if key in wanted:
                found.append(c)
    return found


def _referenced_equations(chunks: list[Chunk], text: str) -> list[Chunk]:
    numbers = {m.group(1) for m in EQ_REF_RE.finditer(text)}
    if not numbers:
        return []
    return [c for c in chunks if c.kind == "equation" and any(re.search(rf"\(\s*{n}\s*\)\s*$", c.text) for n in numbers)]


# --------------------------------------------------------------------------- context assembly


class _Budget:
    def __init__(self, total: int):
        self.remaining = total
        self.truncated = False

    def take(self, text: str, cap: int) -> str:
        allowed = min(cap, self.remaining)
        if allowed <= 80:
            self.truncated = True
            return ""
        out = truncate(text, allowed)
        if len(out) < len(text):
            self.truncated = True
        self.remaining -= len(out)
        return out


def build_selection_context(
    paper: PaperDocument,
    chunks: list[Chunk],
    index: BM25Index,
    selection: str,
    page_hint: int | None = None,
    budget_chars: int = 16_000,
    neighbours: int = 2,
) -> ContextBundle:
    """Context for explaining one selected passage."""
    budget = _Budget(budget_chars)
    selection = selection.strip()
    sel_text = budget.take(selection, max(2000, budget_chars // 3))
    anchor = locate_selection(chunks, selection, page_hint)
    page = anchor.page if anchor else page_hint
    section = anchor.section if anchor else (paper.section_for(page_hint) if page_hint else "")
    abstract = budget.take(paper.abstract, 1500) if paper.abstract else ""

    bundle = ContextBundle(
        title=paper.title, authors=paper.authors, abstract=abstract, selection=sel_text,
        selection_page=page, section=section, located=anchor is not None,
    )
    bundle.sources.append(SourcePassage("selection", page, section, sel_text))
    used: set[int] = set()

    def add(chunk: Chunk, role: str, cap: int) -> None:
        if chunk.index in used:
            return
        text = budget.take(chunk.text, cap)
        if text:
            used.add(chunk.index)
            bundle.sources.append(SourcePassage(role, chunk.page, chunk.section, text))

    if anchor is not None:
        # The anchor chunk usually contains more than the selection: include it as immediate context.
        if len(normalize_for_match(anchor.text)) > len(normalize_for_match(selection)) + 40:
            add(anchor, "containing paragraph", 2200)
        used.add(anchor.index)
        for offset in range(1, neighbours + 1):
            for idx, role in ((anchor.index - offset, "before"), (anchor.index + offset, "after")):
                if 0 <= idx < len(chunks) and not chunks[idx].in_references:
                    add(chunks[idx], role, 1400)
    elif page_hint:
        for c in [c for c in chunks if c.page == page_hint and c.kind == "body"][:2]:
            add(c, "same page", 1400)

    for c in _referenced_captions(chunks, selection)[:3]:
        add(c, "caption", 600)
    for c in _referenced_equations(chunks, selection)[:3]:
        add(c, "equation", 500)
    if page:
        for c in [c for c in chunks if c.page == page and c.kind in {"caption", "equation"}][:4]:
            add(c, c.kind, 500)
    for c in index.top(selection, k=4, exclude=used):
        add(c, "related", 1200)
    bundle.truncated = budget.truncated
    return bundle


OVERVIEW_QUERIES = [
    ("research question", "research question aim objective we study investigate examine contribution motivation"),
    ("data", "data sample dataset observations period firms countries survey collected source"),
    ("method", "methodology model estimate regression identification approach framework specification"),
    ("results", "results find show evidence significant effect estimate coefficient findings"),
    ("limitations", "limitation limitations caveat future research however cannot caution"),
    ("conclusion", "conclude conclusion implications policy summary contribution"),
]


def build_paper_context(paper: PaperDocument, chunks: list[Chunk], index: BM25Index, budget_chars: int = 20_000) -> ContextBundle:
    """Context for whole-paper modes (Teach me this paper, Quiz me, literature review extraction)."""
    budget = _Budget(budget_chars)
    abstract = budget.take(paper.abstract, 2500) if paper.abstract else ""
    bundle = ContextBundle(paper.title, paper.authors, abstract, "", None, "")
    outline = [f"p. {h.page}: {h.text}" for h in paper.headings][:60]
    if outline:
        bundle.sources.append(SourcePassage("section outline", None, "", budget.take("\n".join(outline), 2500)))
    used: set[int] = set()
    body = [c for c in chunks if c.kind == "body" and not c.in_references]
    # Opening of the paper (introduction usually states the question and contribution).
    for c in body[:2]:
        text = budget.take(c.text, 1500)
        if text:
            used.add(c.index)
            bundle.sources.append(SourcePassage("opening", c.page, c.section, text))
    per_topic = max(1200, budget.remaining // (len(OVERVIEW_QUERIES) * 2 + 1))
    for role, query in OVERVIEW_QUERIES:
        for c in index.top(query, k=2, exclude=used):
            if c.in_references:
                continue
            text = budget.take(c.text, per_topic)
            if text:
                used.add(c.index)
                bundle.sources.append(SourcePassage(role, c.page, c.section, text))
    for cap in [c for c in chunks if c.kind == "caption"][:8]:
        text = budget.take(cap.text, 300)
        if text:
            bundle.sources.append(SourcePassage("caption", cap.page, cap.section, text))
    bundle.sources.sort(key=lambda s: (s.page or 0))
    bundle.truncated = True  # by construction this is a sample of the paper, never the whole thing
    return bundle


def section_names(chunks: list[Chunk]) -> list[str]:
    seen: list[str] = []
    for c in chunks:
        if c.section and c.section not in seen and not c.in_references:
            seen.append(c.section)
    return seen


def build_section_context(paper: PaperDocument, chunks: list[Chunk], section: str, budget_chars: int = 18_000) -> ContextBundle:
    """Context for summarising one section: its full text in reading order, within budget."""
    budget = _Budget(budget_chars)
    abstract = budget.take(paper.abstract, 1000) if paper.abstract else ""
    bundle = ContextBundle(paper.title, paper.authors, abstract, "", None, section)
    members = [c for c in chunks if c.section == section]
    if members:
        bundle.selection_page = members[0].page
    for c in members:
        text = budget.take(c.text, budget_chars)
        if not text:
            break
        bundle.sources.append(SourcePassage("section text", c.page, section, text))
    bundle.located = bool(members)
    bundle.truncated = budget.truncated
    return bundle

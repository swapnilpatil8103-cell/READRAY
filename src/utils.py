"""Small, dependency-free helpers shared across Readray modules."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

MAX_PDF_MB_DEFAULT = 50
MAX_IMAGE_MB = 8
MAX_PASTED_CHARS = 20_000
MAX_QUESTION_CHARS = 4_000

PDF_MAGIC = b"%PDF-"
IMAGE_SIGNATURES = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
}


class ValidationError(ValueError):
    """Raised when user input fails validation. Messages are safe to show to users."""


@dataclass(frozen=True)
class ValidatedImage:
    data: bytes
    media_type: str


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_pdf_upload(data: bytes | None, filename: str = "", max_mb: float = MAX_PDF_MB_DEFAULT) -> bytes:
    """Validate an uploaded PDF by size and magic bytes (not by trusting the extension)."""
    if not data:
        raise ValidationError("The uploaded file is empty.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValidationError(f"The file is {size_mb:.1f} MB; the limit is {max_mb:g} MB.")
    if filename and not filename.lower().endswith(".pdf"):
        raise ValidationError("Only PDF files are supported.")
    # Some PDFs have a few junk bytes before the header; the spec tolerates up to 1024.
    if PDF_MAGIC not in data[:1024]:
        raise ValidationError("This file does not look like a PDF (missing %PDF header).")
    return data


def validate_image_upload(data: bytes | None, max_mb: float = MAX_IMAGE_MB) -> ValidatedImage:
    """Validate an image by signature. Only PNG, JPEG and GIF are accepted."""
    if not data:
        raise ValidationError("The image is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise ValidationError(f"Images must be smaller than {max_mb:g} MB.")
    for signature, media_type in IMAGE_SIGNATURES.items():
        if data.startswith(signature):
            return ValidatedImage(data=data, media_type=media_type)
    raise ValidationError("Unsupported image format. Use PNG, JPEG or GIF.")


def validate_text(text: str | None, max_chars: int = MAX_PASTED_CHARS, field: str = "Text") -> str:
    text = (text or "").strip()
    if not text:
        raise ValidationError(f"{field} is empty.")
    if len(text) > max_chars:
        raise ValidationError(f"{field} is {len(text):,} characters; the limit is {max_chars:,}.")
    return text


def clean_text(text: str) -> str:
    """Normalise extracted PDF text: unicode, ligatures, hyphenated line breaks, whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate across line breaks
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_match(text: str) -> str:
    """Aggressive normalisation for fuzzy matching a selection against extracted text."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have he her his i if in
    into is it its may might more most of on or our she should so such than that the their them then there
    these they this those to was we were what when where which while who will with would you your also
    however thus therefore using used use based between both each other we us""".split()
)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall((text or "").lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def truncate(text: str, max_chars: int, marker: str = " […]") -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[: max(0, max_chars - len(marker))]
    space = cut.rfind(" ")
    if space > len(cut) * 0.8:
        cut = cut[:space]
    return cut + marker


def safe_filename(name: str, default: str = "readray") -> str:
    name = re.sub(r"[^\w\-. ]+", "", name or "").strip().replace(" ", "_")
    return (name or default)[:80]


def redact_secrets(message: str, secrets: list[str] | None = None) -> str:
    """Remove anything that looks like an API key from a message before it is shown or logged."""
    message = str(message or "")
    for secret in secrets or []:
        if secret and len(secret) >= 8:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"\b(sk-[A-Za-z0-9_\-]{8,}|sk-ant-[A-Za-z0-9_\-]{8,}|gsk_[A-Za-z0-9]{8,}|AIza[0-9A-Za-z_\-]{20,})", "[REDACTED]", message)
    message = re.sub(r"(?i)(api[_-]?key|authorization|x-api-key|bearer)(\s*[:=]\s*|\s+)([^\s,'\"]{6,})", r"\1\2[REDACTED]", message)
    return message


def format_pages(pages: list[int] | set[int]) -> str:
    """[1,2,3,5] -> 'pp. 1–3, 5'."""
    pages = sorted({int(p) for p in pages if p})
    if not pages:
        return ""
    ranges: list[str] = []
    start = prev = pages[0]
    for p in pages[1:] + [None]:  # type: ignore[list-item]
        if p is not None and p == prev + 1:
            prev = p
            continue
        ranges.append(f"{start}–{prev}" if start != prev else f"{start}")
        if p is not None:
            start = prev = p
    prefix = "p." if len(pages) == 1 else "pp."
    return f"{prefix} {', '.join(ranges)}"


def extract_json(text: str):
    """Parse the first JSON object in a model response (tolerates ```json fences and surrounding prose)."""
    import json

    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in the response.")
    depth, in_string, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("The JSON object in the response is incomplete.")

import pytest

from src.utils import (
    ValidationError, clean_text, extract_json, format_pages, normalize_for_match, redact_secrets, safe_filename,
    tokenize, truncate, validate_image_upload, validate_pdf_upload, validate_text,
)


def test_validate_pdf_upload_accepts_real_pdf(simple_pdf):
    assert validate_pdf_upload(simple_pdf, "paper.pdf") == simple_pdf


@pytest.mark.parametrize(
    "data,name,msg",
    [
        (b"", "a.pdf", "empty"),
        (b"hello", "a.pdf", "does not look like a PDF"),
        (b"%PDF-1.7 ...", "a.exe", "Only PDF"),
    ],
)
def test_validate_pdf_upload_rejects(data, name, msg):
    with pytest.raises(ValidationError, match=msg):
        validate_pdf_upload(data, name)


def test_validate_pdf_upload_size_limit():
    with pytest.raises(ValidationError, match="limit"):
        validate_pdf_upload(b"%PDF-" + b"0" * (2 * 1024 * 1024), "big.pdf", max_mb=1)


def test_validate_image_upload():
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 10
    assert validate_image_upload(png).media_type == "image/png"
    assert validate_image_upload(b"\xff\xd8\xff\xe0rest").media_type == "image/jpeg"
    with pytest.raises(ValidationError):
        validate_image_upload(b"<svg></svg>")
    with pytest.raises(ValidationError):
        validate_image_upload(png + b"0" * (2 * 1024 * 1024), max_mb=1)


def test_validate_text():
    assert validate_text("  hi  ") == "hi"
    with pytest.raises(ValidationError, match="empty"):
        validate_text("   ")
    with pytest.raises(ValidationError, match="limit"):
        validate_text("x" * 11, max_chars=10)


def test_clean_text_dehyphenates_and_normalises():
    assert clean_text("eco-\nnomics  is fun\n\n\n\nyes") == "economics is fun\n\nyes"
    assert clean_text("ﬁnance") == "finance"  # ligature


def test_normalize_and_tokenize():
    assert normalize_for_match("The Bank-\n Lending, Channel!") == "the banklending channel"
    assert tokenize("The effect of the policy on 2 banks") == ["effect", "policy", "banks"]


def test_truncate():
    assert truncate("short", 10) == "short"
    out = truncate("word " * 50, 40)
    assert len(out) <= 40 and out.endswith("[…]")
    assert truncate("anything", 0) == ""


def test_redact_secrets():
    msg = "Error: invalid x-api-key sk-ant-api03-ABCDEFGHIJKLMNOP for request"
    out = redact_secrets(msg)
    assert "ABCDEFGHIJ" not in out and "[REDACTED]" in out
    assert "mysecretvalue123" not in redact_secrets("failed with mysecretvalue123", ["mysecretvalue123"])
    assert "AIza" not in redact_secrets("key=AIzaSyA1234567890abcdefghijklmnop")


def test_format_pages():
    assert format_pages([3, 1, 2, 5]) == "pp. 1–3, 5"
    assert format_pages([4]) == "p. 4"
    assert format_pages([]) == ""


def test_extract_json():
    assert extract_json('Sure!\n```json\n{"a": {"b": "}"}}\n```') == {"a": {"b": "}"}}
    assert extract_json('prefix {"x": [1, 2]} suffix') == {"x": [1, 2]}
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json('{"unterminated": 1')


def test_safe_filename():
    assert safe_filename("My Review: banks/credit?") == "My_Review_bankscredit"
    assert safe_filename("???") == "readray"

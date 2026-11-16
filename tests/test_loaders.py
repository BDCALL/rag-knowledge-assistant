from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter

from rag_assistant.ingestion.loaders import (
    DocumentParseError,
    UnsupportedFormatError,
    load_document,
    normalise_whitespace,
)


def test_text_loader_uses_filename_as_title() -> None:
    doc = load_document(b"Hello   world.\r\n\r\n\r\n\r\nSecond para.", "notes.txt")
    assert doc.title == "notes"
    assert doc.media_type == "text/plain"
    assert doc.text == "Hello world.\n\nSecond para."


def test_markdown_loader_takes_first_heading_and_strips_front_matter() -> None:
    md = b"---\ntags: [a, b]\n---\n\n# Vector Databases\n\nThey store embeddings.\n"
    doc = load_document(md, "vectors.md")
    assert doc.title == "Vector Databases"
    assert "tags" not in doc.text
    assert doc.media_type == "text/markdown"


def test_markdown_without_heading_falls_back_to_filename() -> None:
    assert load_document(b"just text", "readme.markdown").title == "readme"


def test_media_type_fallback_when_extension_unknown() -> None:
    doc = load_document(b"plain body", "upload.bin", media_type="text/plain; charset=utf-8")
    assert doc.text == "plain body"


def test_unsupported_format_raises() -> None:
    with pytest.raises(UnsupportedFormatError):
        load_document(b"\x89PNG", "image.png", media_type="image/png")


def test_empty_document_raises() -> None:
    with pytest.raises(DocumentParseError):
        load_document(b"   \n\n  ", "empty.txt")


def test_corrupt_pdf_raises_parse_error() -> None:
    with pytest.raises(DocumentParseError):
        load_document(b"%PDF-1.4 not really a pdf", "broken.pdf")


def test_blank_pdf_has_no_text() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(DocumentParseError, match="No extractable text"):
        load_document(buf.getvalue(), "blank.pdf")


def test_pdf_with_text_is_extracted(sample_pdf: bytes) -> None:
    doc = load_document(sample_pdf, "guide.pdf")
    assert doc.media_type == "application/pdf"
    assert doc.metadata["pages"] == 2
    assert "pgvector" in doc.text
    assert "\f" in doc.text  # page separator preserved for the chunker


def test_content_hash_is_stable() -> None:
    a = load_document(b"same text", "a.txt")
    b = load_document(b"same text", "b.txt")
    assert a.content_hash == b.content_hash
    assert a.id != b.id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("a\t\tb", "a b"), ("a \n b", "a\nb"), ("\n\na\n\n\n\nb\n", "a\n\nb")],
)
def test_normalise_whitespace(raw: str, expected: str) -> None:
    assert normalise_whitespace(raw) == expected

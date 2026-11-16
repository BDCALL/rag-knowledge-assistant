"""Turn raw uploaded bytes into a :class:`Document` with extracted text.

Each loader handles one family of formats. ``load_document`` picks a loader from
the file extension (falling back to the declared media type) so the API layer never
needs to know about individual formats.
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag_assistant.domain import Document


class UnsupportedFormatError(ValueError):
    """Raised when no loader can handle the given file."""


class DocumentParseError(ValueError):
    """Raised when a file claims a supported format but cannot be parsed."""


_FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text:
            return text
    return data.decode("latin-1")


def normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and >2 blank lines while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_text(data: bytes, filename: str) -> Document:
    text = normalise_whitespace(_decode(data))
    return Document(
        title=PurePath(filename).stem,
        text=text,
        source=filename,
        media_type="text/plain",
    )


def load_markdown(data: bytes, filename: str) -> Document:
    raw = _decode(data)
    raw = _FRONT_MATTER.sub("", raw, count=1)
    heading = _MD_HEADING.search(raw)
    title = heading.group(1).strip() if heading else PurePath(filename).stem
    return Document(
        title=title,
        text=normalise_whitespace(raw),
        source=filename,
        media_type="text/markdown",
    )


def load_pdf(data: bytes, filename: str) -> Document:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError, OSError) as exc:
        raise DocumentParseError(f"Could not read PDF {filename!r}: {exc}") from exc

    meta_title = None
    if reader.metadata is not None and reader.metadata.title:
        meta_title = str(reader.metadata.title).strip() or None

    # Keep a form-feed between pages so the chunker can record page numbers.
    text = "\f".join(normalise_whitespace(p) for p in pages)
    return Document(
        title=meta_title or PurePath(filename).stem,
        text=text,
        source=filename,
        media_type="application/pdf",
        metadata={"pages": len(pages)},
    )


Loader = Callable[[bytes, str], Document]

LOADERS: dict[str, Loader] = {
    ".txt": load_text,
    ".text": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".pdf": load_pdf,
}

_MEDIA_TYPES: dict[str, Loader] = {
    "text/plain": load_text,
    "text/markdown": load_markdown,
    "application/pdf": load_pdf,
}


def load_document(data: bytes, filename: str, media_type: str | None = None) -> Document:
    """Dispatch to the right loader and reject empty results."""
    loader = LOADERS.get(PurePath(filename).suffix.lower())
    if loader is None and media_type:
        loader = _MEDIA_TYPES.get(media_type.split(";")[0].strip().lower())
    if loader is None:
        supported = ", ".join(sorted(LOADERS))
        raise UnsupportedFormatError(f"Unsupported file {filename!r}; supported: {supported}")

    document = loader(data, filename)
    if not document.text.replace("\f", "").strip():
        raise DocumentParseError(f"No extractable text in {filename!r}")
    return document

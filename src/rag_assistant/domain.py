"""Core domain objects shared by ingestion, storage and the API.

These are plain dataclasses with no framework dependencies, so every layer
(loaders, chunker, stores, HTTP) can depend on them without coupling to each other.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class Document:
    """A source document after text extraction."""

    title: str
    text: str
    source: str
    media_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def content_hash(self) -> str:
        """SHA-256 of the extracted text; used to skip duplicate uploads."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous span of a document's text, the unit we embed and retrieve."""

    document_id: str
    index: int
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    """What a store returns when listing documents (no full text)."""

    id: str
    title: str
    source: str
    media_type: str
    content_hash: str
    chunk_count: int
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A retrieved chunk with its score (higher is better).

    ``ranks`` records the 1-based position the chunk had in each retriever that
    found it (e.g. ``{"dense": 1, "lexical": 4}``), which makes hybrid results
    explainable in the API and in evaluations.
    """

    chunk: Chunk
    score: float
    document_title: str
    ranks: dict[str, int] = field(default_factory=dict)

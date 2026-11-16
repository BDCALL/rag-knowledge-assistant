"""Exact (brute-force) vector search in memory with NumPy.

Used for local development and the test suite. Brute force is O(n·d) per query,
which is fine for thousands of chunks; pgvector + HNSW takes over beyond that.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from rag_assistant.domain import Chunk, Document, DocumentSummary, SearchHit
from rag_assistant.embeddings import Vector
from rag_assistant.store.base import validate_add


@dataclass(slots=True)
class _Entry:
    document: Document
    chunks: list[Chunk]


class InMemoryVectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._docs: dict[str, _Entry] = {}
        self._chunks: list[Chunk] = []
        self._matrix: Vector = np.zeros((0, dim), dtype=np.float32)
        self._lock = threading.RLock()

    def add(self, document: Document, chunks: Sequence[Chunk], embeddings: Vector) -> None:
        validate_add(chunks, embeddings, self.dim)
        with self._lock:
            if document.id in self._docs:
                raise ValueError(f"Document {document.id} already stored")
            self._docs[document.id] = _Entry(document, list(chunks))
            self._chunks.extend(chunks)
            self._matrix = np.vstack([self._matrix, embeddings.astype(np.float32)])

    def _summary(self, entry: _Entry) -> DocumentSummary:
        doc = entry.document
        return DocumentSummary(
            id=doc.id,
            title=doc.title,
            source=doc.source,
            media_type=doc.media_type,
            content_hash=doc.content_hash,
            chunk_count=len(entry.chunks),
            created_at=doc.created_at,
            metadata=dict(doc.metadata),
        )

    def find_by_hash(self, content_hash: str) -> DocumentSummary | None:
        with self._lock:
            for entry in self._docs.values():
                if entry.document.content_hash == content_hash:
                    return self._summary(entry)
        return None

    def get(self, document_id: str) -> DocumentSummary | None:
        with self._lock:
            entry = self._docs.get(document_id)
            return self._summary(entry) if entry else None

    def list_documents(self) -> list[DocumentSummary]:
        with self._lock:
            summaries = [self._summary(e) for e in self._docs.values()]
        return sorted(summaries, key=lambda s: s.created_at, reverse=True)

    def delete(self, document_id: str) -> bool:
        with self._lock:
            if self._docs.pop(document_id, None) is None:
                return False
            keep = [i for i, c in enumerate(self._chunks) if c.document_id != document_id]
            self._chunks = [self._chunks[i] for i in keep]
            self._matrix = self._matrix[keep] if keep else np.zeros((0, self.dim), np.float32)
            return True

    def search(
        self,
        query: Vector,
        k: int = 5,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        if k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"Query dim {q.shape[0]} != store dim {self.dim}")
        with self._lock:
            if not self._chunks:
                return []
            scores = self._matrix @ q
            if document_ids is not None:
                allowed = set(document_ids)
                mask = np.array([c.document_id in allowed for c in self._chunks])
                scores = np.where(mask, scores, -np.inf)
            k = min(k, len(self._chunks))
            # argpartition is O(n); only the top-k slice is fully sorted.
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [
                SearchHit(
                    chunk=self._chunks[i],
                    score=float(scores[i]),
                    document_title=self._docs[self._chunks[i].document_id].document.title,
                )
                for i in top
                if np.isfinite(scores[i])
            ]

    def count_chunks(self) -> int:
        with self._lock:
            return len(self._chunks)

    def iter_chunks(self) -> Iterator[tuple[Chunk, str]]:
        with self._lock:
            snapshot = [(c, self._docs[c.document_id].document.title) for c in self._chunks]
        yield from snapshot

    def ping(self) -> bool:
        return True

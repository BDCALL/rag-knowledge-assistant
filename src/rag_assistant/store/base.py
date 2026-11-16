"""Storage interface shared by the in-memory and Postgres implementations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from rag_assistant.domain import Chunk, Document, DocumentSummary, SearchHit
from rag_assistant.embeddings import Vector


class VectorStore(Protocol):
    """Persists documents + chunk embeddings and runs nearest-neighbour search.

    Embeddings passed in are expected to be L2-normalised, so the score returned by
    :meth:`search` is cosine similarity in ``[-1, 1]`` (higher is more similar).
    """

    def add(self, document: Document, chunks: Sequence[Chunk], embeddings: Vector) -> None: ...

    def find_by_hash(self, content_hash: str) -> DocumentSummary | None: ...

    def get(self, document_id: str) -> DocumentSummary | None: ...

    def list_documents(self) -> list[DocumentSummary]: ...

    def delete(self, document_id: str) -> bool: ...

    def search(
        self,
        query: Vector,
        k: int = 5,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchHit]: ...

    def count_chunks(self) -> int: ...

    def iter_chunks(self) -> Iterator[tuple[Chunk, str]]:
        """Yield every stored chunk with its document title (used to build the BM25 index)."""
        ...

    def ping(self) -> bool:
        """Cheap health check used by ``/health``."""
        ...


def validate_add(chunks: Sequence[Chunk], embeddings: Vector, dim: int) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Expected {len(chunks)} embeddings, got array of shape {embeddings.shape}"
        )
    if embeddings.shape[1] != dim:
        raise ValueError(f"Embedding dim {embeddings.shape[1]} != store dim {dim}")

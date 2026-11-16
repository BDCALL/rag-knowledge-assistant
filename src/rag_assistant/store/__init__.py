"""Vector stores behind one :class:`VectorStore` protocol."""

from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.store.base import VectorStore
from rag_assistant.store.memory import InMemoryVectorStore


def build_store(settings: Settings, dim: int) -> VectorStore:
    """Factory: choose a backend from configuration and make sure it's ready."""
    if settings.store_backend == "pgvector":
        from rag_assistant.store.pgvector import PgVectorStore

        store = PgVectorStore(settings.database_url, dim=dim)
        store.create_schema()
        return store
    return InMemoryVectorStore(dim=dim)


__all__ = ["InMemoryVectorStore", "VectorStore", "build_store"]

"""Embedding providers behind one :class:`Embedder` protocol."""

from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.embeddings.base import Embedder, Vector, l2_normalise
from rag_assistant.embeddings.hashing import HashingEmbedder
from rag_assistant.embeddings.providers import OpenAIEmbedder, SentenceTransformerEmbedder


def build_embedder(settings: Settings) -> Embedder:
    """Factory: choose a provider from configuration."""
    match settings.embedding_provider:
        case "hashing":
            return HashingEmbedder(dim=settings.embedding_dim)
        case "sentence-transformers":
            return SentenceTransformerEmbedder(settings.embedding_model)
        case "openai":
            assert settings.openai_api_key is not None  # enforced by Settings validator
            return OpenAIEmbedder(api_key=settings.openai_api_key, dim=settings.embedding_dim)


__all__ = [
    "Embedder",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "Vector",
    "build_embedder",
    "l2_normalise",
]

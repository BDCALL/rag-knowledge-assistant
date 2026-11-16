from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rag_assistant.config import Settings
from rag_assistant.embeddings import (
    Embedder,
    HashingEmbedder,
    OpenAIEmbedder,
    build_embedder,
    l2_normalise,
)
from rag_assistant.embeddings.hashing import tokenize


def test_hashing_embedder_shape_and_norm() -> None:
    vectors = HashingEmbedder(dim=64).embed(["hello world", "another text", ""])
    assert vectors.shape == (3, 64)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors[:2], axis=1), 1.0, rtol=1e-5)
    assert np.linalg.norm(vectors[2]) == 0.0  # empty text stays a zero vector


def test_hashing_embedder_is_deterministic() -> None:
    a = HashingEmbedder(dim=128).embed(["pgvector stores embeddings"])
    b = HashingEmbedder(dim=128).embed(["pgvector stores embeddings"])
    np.testing.assert_array_equal(a, b)


def test_similar_texts_score_higher_than_unrelated() -> None:
    emb = HashingEmbedder(dim=384)
    query, related, unrelated = emb.embed(
        [
            "How does pgvector store embeddings?",
            "pgvector is a Postgres extension that stores embeddings as vectors.",
            "The recipe needs flour, butter and two eggs.",
        ]
    )
    assert float(query @ related) > float(query @ unrelated) + 0.2


def test_tokenize_drops_stopwords_and_punctuation() -> None:
    assert tokenize("What is the BEST vector-db?") == ["best", "vector", "db"]


def test_implements_protocol() -> None:
    assert isinstance(HashingEmbedder(), Embedder)


def test_l2_normalise_handles_zero_rows() -> None:
    out = l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_invalid_dim() -> None:
    with pytest.raises(ValueError, match="positive"):
        HashingEmbedder(dim=0)


class _FakeOpenAI:
    """Mimics ``client.embeddings.create`` and returns items out of order."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, *, model: str, input: list[str], dimensions: int) -> SimpleNamespace:
        self.calls.append({"model": model, "input": input, "dimensions": dimensions})
        data = [
            SimpleNamespace(index=i, embedding=[float(i + 1)] * dimensions)
            for i in range(len(input))
        ]
        return SimpleNamespace(data=list(reversed(data)))


def test_openai_embedder_orders_and_normalises() -> None:
    fake = _FakeOpenAI()
    emb = OpenAIEmbedder(api_key="sk-test", dim=4, client=fake)
    out = emb.embed(["a", "b"])
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out, 0.5)
    assert fake.calls[0]["dimensions"] == 4
    assert emb.name == "openai:text-embedding-3-small"
    assert emb.embed([]).shape == (0, 4)


def test_factory_defaults_to_hashing() -> None:
    emb = build_embedder(Settings(embedding_dim=32))
    assert isinstance(emb, HashingEmbedder)
    assert emb.dim == 32


def test_openai_requires_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings(embedding_provider="openai", openai_api_key=None)

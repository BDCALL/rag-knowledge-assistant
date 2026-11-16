"""Real embedding providers. Their heavy imports happen lazily inside ``__init__``
so the package (and the test suite) works without them installed."""

from __future__ import annotations

from typing import Any

import numpy as np

from rag_assistant.embeddings.base import Vector, l2_normalise


class SentenceTransformerEmbedder:
    """Local model via sentence-transformers (``pip install .[local-embeddings]``)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "sentence-transformers is not installed; run pip install '.[local-embeddings]'"
            ) from exc
        self._model: Any = SentenceTransformer(model_name)
        self._name = model_name
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> Vector:
        vectors = self._model.encode(texts, batch_size=32, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbedder:
    """OpenAI embeddings API (``pip install .[openai]``, needs ``RAG_OPENAI_API_KEY``)."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = 384,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError("openai is not installed; run pip install '.[openai]'") from exc
            client = OpenAI(api_key=api_key)
        self._client = client
        self._model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def embed(self, texts: list[str]) -> Vector:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        # text-embedding-3-* models support server-side dimensionality reduction.
        response = self._client.embeddings.create(
            model=self._model, input=texts, dimensions=self._dim
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return l2_normalise(np.asarray([item.embedding for item in ordered], dtype=np.float32))

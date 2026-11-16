"""A dependency-free, deterministic embedder for tests, CI and offline demos.

It uses the "hashing trick": each token (and each character trigram, for some
robustness to typos and plurals) is hashed into one of ``dim`` buckets with a
signed weight. The result is a sparse bag-of-features vector. It has no semantic
understanding, but texts that share words land close together, which is enough
to test retrieval end-to-end without downloading a model or calling an API.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from rag_assistant.embeddings.base import Vector, l2_normalise

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
    }
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class HashingEmbedder:
    def __init__(self, dim: int = 384, use_trigrams: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._use_trigrams = use_trigrams

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        sign = 1.0 if value & 1 else -1.0
        return (value >> 1) % self._dim, sign

    def _features(self, text: str) -> Counter[str]:
        tokens = tokenize(text)
        feats: Counter[str] = Counter(f"w:{t}" for t in tokens)
        if self._use_trigrams:
            for token in tokens:
                padded = f"#{token}#"
                feats.update(f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2))
        return feats

    def embed(self, texts: list[str]) -> Vector:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, count in self._features(text).items():
                bucket, sign = self._bucket(feature)
                # Word features weigh more than trigram features; sublinear tf.
                weight = (1.0 if feature.startswith("w:") else 0.3) * (1 + math.log(count))
                out[row, bucket] += sign * weight
        return l2_normalise(out)

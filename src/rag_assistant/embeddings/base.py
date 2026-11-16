"""The embedding interface every provider implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to L2-normalised vectors of a fixed dimension.

    Normalising up front means cosine similarity == dot product, which keeps the
    in-memory store fast and lets pgvector use its inner-product operator.
    """

    @property
    def dim(self) -> int: ...

    @property
    def name(self) -> str: ...

    def embed(self, texts: list[str]) -> Vector:
        """Return an array of shape ``(len(texts), dim)``."""
        ...


def l2_normalise(matrix: Vector) -> Vector:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)

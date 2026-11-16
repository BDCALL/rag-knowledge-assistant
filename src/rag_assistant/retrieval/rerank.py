"""Second-stage reranking of retrieved candidates.

Retrievers (BM25, bi-encoder embeddings) score the query and each chunk
*independently*, which is what makes them fast enough to search a whole corpus.
A cross-encoder reads the query and a chunk *together* in one forward pass, so it
can model word-level interactions ("not", word order, which entity a number
belongs to) and is usually much more accurate, but far too slow to run over
everything. The standard compromise is retrieve-then-rerank:

1. cheaply fetch ~20-50 candidates with hybrid search,
2. rescore only those with the expensive model,
3. keep the top ``k``.

Every reranker implements the small :class:`Reranker` protocol, so the service
does not care whether scores come from a neural model or the offline fallback.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, Protocol, runtime_checkable

from rag_assistant.domain import SearchHit
from rag_assistant.retrieval.bm25 import tokenize


@runtime_checkable
class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score per passage (higher is better)."""
        ...


class TermOverlapReranker:
    """Offline, deterministic fallback: no model download, no network.

    Scores a passage by how much of the query it covers, weighting each query term
    by its rarity *within the candidate set* (a mini-IDF), plus a small bonus when
    query terms appear next to each other in the same order (phrase matches).
    It is a weak reranker, but it exercises the exact same code path as the
    cross-encoder, which is what tests and CI need.
    """

    def __init__(self, phrase_bonus: float = 0.25) -> None:
        self.phrase_bonus = phrase_bonus

    @property
    def name(self) -> str:
        return "term-overlap"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q_terms = list(dict.fromkeys(tokenize(query)))  # unique, order kept
        if not q_terms or not passages:
            return [0.0] * len(passages)
        tokenised = [tokenize(p) for p in passages]
        sets = [set(tokens) for tokens in tokenised]
        n = len(passages)
        weights = {t: math.log(1 + (n + 1) / (1 + sum(t in s for s in sets))) for t in q_terms}
        total = sum(weights.values())
        bigrams = set(pairwise(q_terms))
        scores = []
        for tokens, present in zip(tokenised, sets, strict=True):
            coverage = sum(w for t, w in weights.items() if t in present) / total
            phrases = len(bigrams & set(pairwise(tokens)))
            bonus = self.phrase_bonus * phrases / len(bigrams) if bigrams else 0.0
            scores.append(round(coverage + bonus, 6))
        return scores


class CrossEncoderReranker:
    """A sentence-transformers cross-encoder (``pip install .[local-embeddings]``).

    The default MS MARCO MiniLM model is ~90 MB and reranks 20 passages in tens
    of milliseconds on CPU. ``model`` can be injected so tests never load one.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model: Any = None,
        batch_size: int = 16,
    ) -> None:
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    "sentence-transformers is not installed; run pip install '.[local-embeddings]'"
                ) from exc
            model = CrossEncoder(model_name)  # pragma: no cover - downloads a model
        self._model = model
        self._name = model_name
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return f"cross-encoder:{self._name}"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        raw = self._model.predict(pairs, batch_size=self.batch_size)
        return [float(s) for s in raw]


def rerank(
    reranker: Reranker, query: str, hits: Sequence[SearchHit], limit: int
) -> list[SearchHit]:
    """Rescore ``hits`` with ``reranker`` and return the best ``limit``.

    The new score replaces the retrieval score, and the 1-based position after
    reranking is recorded under ``ranks["rerank"]`` next to the original
    retriever ranks, so you can see how much the reranker moved each chunk.
    Ties keep the original retrieval order (Python's sort is stable).
    """
    if not hits or limit <= 0:
        return []
    scores = reranker.score(query, [h.chunk.text for h in hits])
    if len(scores) != len(hits):
        raise ValueError(f"{reranker.name} returned {len(scores)} scores for {len(hits)} hits")
    order = sorted(range(len(hits)), key=lambda i: -scores[i])[:limit]
    return [
        SearchHit(
            chunk=hits[i].chunk,
            score=scores[i],
            document_title=hits[i].document_title,
            ranks={**hits[i].ranks, "rerank": position},
        )
        for position, i in enumerate(order, start=1)
    ]

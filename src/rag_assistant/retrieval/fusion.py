"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).

Dense cosine scores (~0.2-0.9) and BM25 scores (unbounded, corpus-dependent) live on
different scales, so adding them directly lets one retriever drown out the other.
RRF ignores raw scores and only uses rank::

    rrf(d) = Σ_r  w_r / (k + rank_r(d))

``k`` (60 in the paper) damps the advantage of the very top ranks so that a chunk
ranked well by *both* retrievers beats one ranked first by only one of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from rag_assistant.domain import Chunk, SearchHit

DEFAULT_RRF_K = 60


class _Ranked(Protocol):
    @property
    def chunk(self) -> Chunk: ...

    @property
    def document_title(self) -> str: ...


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[_Ranked]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[SearchHit]:
    """Fuse named ranked lists into one list of :class:`SearchHit` sorted by RRF score."""
    if k < 0:
        raise ValueError("RRF k must be non-negative")
    scores: dict[str, float] = {}
    best: dict[str, _Ranked] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, hits in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        for position, hit in enumerate(hits, start=1):
            cid = hit.chunk.id
            if name in ranks.get(cid, {}):
                continue  # ignore duplicates within one list
            scores[cid] = scores.get(cid, 0.0) + weight / (k + position)
            best.setdefault(cid, hit)
            ranks.setdefault(cid, {})[name] = position
    ordered = sorted(scores, key=lambda cid: (-scores[cid], min(ranks[cid].values()), cid))
    if limit is not None:
        ordered = ordered[:limit]
    return [
        SearchHit(
            chunk=best[cid].chunk,
            score=scores[cid],
            document_title=best[cid].document_title,
            ranks=ranks[cid],
        )
        for cid in ordered
    ]

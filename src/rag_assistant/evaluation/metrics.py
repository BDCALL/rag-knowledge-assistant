"""Ranking metrics for retrieval evaluation.

Each query in a labelled set has a ranked list of retrieved ids and a set of ids
judged relevant. ``recall_at_k`` asks "how much of what matters made the top k?";
``reciprocal_rank`` asks "how far down is the first useful result?". Averaging the
latter over queries gives MRR.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of ``relevant`` ids that appear in the first ``k`` of ``retrieved``."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant:
        raise ValueError("a query needs at least one relevant id")
    found = set(retrieved[:k]) & set(relevant)
    return len(found) / len(set(relevant))


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str]) -> float:
    """1 / rank of the first relevant id (rank starts at 1), or 0.0 if none is retrieved."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(results: Sequence[tuple[Sequence[str], Collection[str]]]) -> float:
    """Average :func:`reciprocal_rank` over ``(retrieved, relevant)`` pairs."""
    if not results:
        raise ValueError("need at least one query to average over")
    return sum(reciprocal_rank(r, rel) for r, rel in results) / len(results)

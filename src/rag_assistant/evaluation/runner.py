"""Run a labelled query set through a knowledge base and score the rankings.

Scoring is at document level: chunk hits are mapped to the file stem of the
document they came from (``leave-policy.md`` -> ``leave-policy``), the same names
the query set uses as labels, and repeat hits from one document count once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePath

from rag_assistant.evaluation.dataset import LabelledQuery
from rag_assistant.evaluation.metrics import mean_reciprocal_rank, recall_at_k
from rag_assistant.service import KnowledgeBase, SearchMode


@dataclass(frozen=True)
class RetrievalReport:
    mode: SearchMode
    k: int
    recall: float
    mrr: float


def evaluate_retrieval(
    kb: KnowledgeBase, queries: Sequence[LabelledQuery], k: int = 3, mode: SearchMode = "hybrid"
) -> RetrievalReport:
    """Average recall@k and MRR over ``queries`` for one search ``mode`` (no reranking)."""
    if not queries:
        raise ValueError("need at least one query to evaluate")
    labels = {d.id: PurePath(d.source).stem for d in kb.store.list_documents()}
    results = []
    for query in queries:
        hits = kb.search(query.question, k=k, mode=mode, rerank=False)
        ranked = list(dict.fromkeys(labels[h.chunk.document_id] for h in hits))
        results.append((ranked, query.relevant))
    recall = sum(recall_at_k(r, rel, k) for r, rel in results) / len(results)
    return RetrievalReport(mode, k, recall, mean_reciprocal_rank(results))

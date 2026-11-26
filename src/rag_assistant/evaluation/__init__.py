"""Retrieval evaluation: ranking metrics over a labelled query set."""

from rag_assistant.evaluation.dataset import LabelledQuery, load_queries
from rag_assistant.evaluation.metrics import mean_reciprocal_rank, recall_at_k, reciprocal_rank
from rag_assistant.evaluation.runner import RetrievalReport, evaluate_retrieval

__all__ = [
    "LabelledQuery",
    "RetrievalReport",
    "evaluate_retrieval",
    "load_queries",
    "mean_reciprocal_rank",
    "recall_at_k",
    "reciprocal_rank",
]

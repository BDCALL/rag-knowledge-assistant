"""Lexical retrieval, rank fusion and reranking used for hybrid search."""

from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.retrieval.bm25 import BM25Index, LexicalHit, tokenize
from rag_assistant.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from rag_assistant.retrieval.rerank import (
    CrossEncoderReranker,
    Reranker,
    TermOverlapReranker,
    rerank,
)


def build_reranker(settings: Settings) -> Reranker | None:
    """Factory: ``None`` when reranking is disabled."""
    match settings.reranker:
        case "none":
            return None
        case "term-overlap":
            return TermOverlapReranker()
        case "cross-encoder":
            return CrossEncoderReranker(settings.reranker_model)


__all__ = [
    "DEFAULT_RRF_K",
    "BM25Index",
    "CrossEncoderReranker",
    "LexicalHit",
    "Reranker",
    "TermOverlapReranker",
    "build_reranker",
    "reciprocal_rank_fusion",
    "rerank",
    "tokenize",
]

"""Application service: ingestion, retrieval and answering, independent of HTTP.

Keeping this layer framework-free means the same logic can back the REST API, a
CLI, or a background worker, and it can be unit-tested without a web server.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from rag_assistant.domain import Chunk, DocumentSummary, SearchHit
from rag_assistant.embeddings import Embedder
from rag_assistant.generation import (
    LLM,
    NO_ANSWER,
    FakeLLM,
    build_messages,
    extract_citations,
)
from rag_assistant.ingestion import RecursiveChunker, load_document
from rag_assistant.retrieval import BM25Index, Reranker, reciprocal_rank_fusion
from rag_assistant.retrieval import rerank as _rerank
from rag_assistant.store import VectorStore

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64
# How many candidates each retriever contributes before fusion, relative to k.
HYBRID_CANDIDATE_FACTOR = 4
MIN_HYBRID_CANDIDATES = 20

DEFAULT_RERANK_CANDIDATES = 20

SearchMode = Literal["hybrid", "dense", "lexical"]


class RerankerNotConfiguredError(RuntimeError):
    """``rerank=True`` was requested but no reranker is configured."""


@dataclass(frozen=True, slots=True)
class IngestResult:
    document: DocumentSummary
    created: bool  # False when an identical document was already indexed
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class Answer:
    """A generated answer plus the evidence it was built from.

    ``sources`` are the retrieved hits in the order they were shown to the model,
    so citation ``[n]`` refers to ``sources[n - 1]``. ``citations`` holds the
    valid source numbers the answer actually cites, in first-seen order.
    """

    question: str
    text: str
    sources: list[SearchHit]
    citations: list[int]
    llm: str
    elapsed_ms: float = 0.0

    @property
    def cited_sources(self) -> list[tuple[int, SearchHit]]:
        return [(n, self.sources[n - 1]) for n in self.citations]


@dataclass(slots=True)
class AnswerStream:
    """Sources up front, then the answer as text deltas.

    Retrieval runs eagerly (so errors surface before any bytes are streamed);
    the model is only called once ``deltas`` is iterated.
    """

    question: str
    sources: list[SearchHit]
    deltas: Iterator[str]
    llm: str
    started: float  # perf_counter() when the request began, for elapsed_ms


class KnowledgeBase:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        chunker: RecursiveChunker,
        lexical: BM25Index | None = None,
        reranker: Reranker | None = None,
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
        llm: LLM | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunker = chunker
        self.lexical = lexical or BM25Index()
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.llm: LLM = llm or FakeLLM()
        self._rebuild_lexical_index()

    def _rebuild_lexical_index(self) -> None:
        """BM25 lives in process memory, so rebuild it from the store at startup."""
        started = time.perf_counter()
        titles: dict[str, str] = {}
        by_doc: dict[str, list[Chunk]] = {}
        for chunk, title in self.store.iter_chunks():
            titles[chunk.document_id] = title
            by_doc.setdefault(chunk.document_id, []).append(chunk)
        for document_id, chunks in by_doc.items():
            self.lexical.add(chunks, titles[document_id])
        if by_doc:
            logger.info(
                "Rebuilt BM25 index: %d chunks in %.1f ms", len(self.lexical), _ms_since(started)
            )

    def ingest(self, data: bytes, filename: str, media_type: str | None = None) -> IngestResult:
        started = time.perf_counter()
        document = load_document(data, filename, media_type)

        existing = self.store.find_by_hash(document.content_hash)
        if existing is not None:
            logger.info("Skipping duplicate upload %s (matches %s)", filename, existing.id)
            return IngestResult(existing, created=False, elapsed_ms=_ms_since(started))

        chunks = self.chunker.split(document)
        texts = [c.text for c in chunks]
        vectors = [
            self.embedder.embed(texts[i : i + EMBED_BATCH_SIZE])
            for i in range(0, len(texts), EMBED_BATCH_SIZE)
        ]
        embeddings = np.vstack(vectors) if vectors else np.zeros((0, self.embedder.dim), np.float32)
        self.store.add(document, chunks, embeddings)
        self.lexical.add(chunks, document.title)

        summary = self.store.get(document.id)
        assert summary is not None
        elapsed = _ms_since(started)
        logger.info("Ingested %s: %d chunks in %.1f ms", filename, len(chunks), elapsed)
        return IngestResult(summary, created=True, elapsed_ms=elapsed)

    def delete(self, document_id: str) -> bool:
        deleted = self.store.delete(document_id)
        self.lexical.remove_document(document_id)
        return deleted

    def search(
        self,
        query: str,
        k: int = 5,
        document_ids: Sequence[str] | None = None,
        mode: SearchMode = "hybrid",
        rerank: bool | None = None,
    ) -> list[SearchHit]:
        """Retrieve the top-``k`` chunks.

        ``dense`` uses embeddings only, ``lexical`` BM25 only, and ``hybrid`` (default)
        over-fetches from both and merges them with Reciprocal Rank Fusion.

        ``rerank`` (default: on whenever a reranker is configured) retrieves
        ``rerank_candidates`` chunks first and lets the reranker choose the top ``k``.
        """
        query = query.strip()
        if not query or k <= 0:
            return []
        use_reranker = self.reranker is not None if rerank is None else rerank
        if not use_reranker:
            return self._retrieve(query, k, document_ids, mode)
        if self.reranker is None:
            raise RerankerNotConfiguredError("rerank requested but RAG_RERANKER is 'none'")
        candidates = self._retrieve(query, max(k, self.rerank_candidates), document_ids, mode)
        return _rerank(self.reranker, query, candidates, limit=k)

    def stream_answer(
        self,
        question: str,
        k: int = 5,
        document_ids: Sequence[str] | None = None,
        mode: SearchMode = "hybrid",
        rerank: bool | None = None,
    ) -> AnswerStream:
        """Retrieve sources for ``question`` and return a lazy stream of the answer.

        With no sources there is nothing to ground an answer in, so the model is
        skipped entirely and the stream yields the fixed "don't know" reply.
        """
        started = time.perf_counter()
        question = question.strip()
        hits = self.search(question, k, document_ids, mode, rerank)
        no_answer = iter([NO_ANSWER])
        deltas = self.llm.stream(build_messages(question, hits)) if hits else no_answer
        return AnswerStream(question, hits, deltas, self.llm.name, started)

    def ask(
        self,
        question: str,
        k: int = 5,
        document_ids: Sequence[str] | None = None,
        mode: SearchMode = "hybrid",
        rerank: bool | None = None,
    ) -> Answer:
        """Retrieve, generate and collect a complete answer with its citations."""
        stream = self.stream_answer(question, k, document_ids, mode, rerank)
        text = "".join(stream.deltas)
        elapsed = _ms_since(stream.started)
        logger.info(
            "Answered with %s from %d sources in %.1f ms", stream.llm, len(stream.sources), elapsed
        )
        return Answer(
            question=stream.question,
            text=text,
            sources=stream.sources,
            citations=extract_citations(text, len(stream.sources)),
            llm=stream.llm,
            elapsed_ms=elapsed,
        )

    def _retrieve(
        self, query: str, k: int, document_ids: Sequence[str] | None, mode: SearchMode
    ) -> list[SearchHit]:
        if mode == "lexical":
            return [
                SearchHit(h.chunk, h.score, h.document_title, {"lexical": i})
                for i, h in enumerate(self.lexical.search(query, k, document_ids), start=1)
            ]
        pool = k if mode == "dense" else max(k * HYBRID_CANDIDATE_FACTOR, MIN_HYBRID_CANDIDATES)
        vector = self.embedder.embed([query])[0]
        dense = self.store.search(vector, k=pool, document_ids=document_ids)
        if mode == "dense":
            return [
                SearchHit(h.chunk, h.score, h.document_title, {"dense": i})
                for i, h in enumerate(dense, start=1)
            ]
        lexical = self.lexical.search(query, pool, document_ids)
        return reciprocal_rank_fusion({"dense": dense, "lexical": lexical}, limit=k)


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)

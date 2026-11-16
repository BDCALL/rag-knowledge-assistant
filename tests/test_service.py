from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.ingestion import RecursiveChunker
from rag_assistant.service import KnowledgeBase, RerankerNotConfiguredError
from rag_assistant.store import InMemoryVectorStore

DOCS = {
    "errors.md": (
        "# Error codes\n\nERR_7731 is raised when the embedding dimension of the model does "
        "not match the vector column. Re-create the schema to fix it."
    ),
    "storage.md": (
        "# Storage\n\nDocuments are persisted in Postgres. Each chunk keeps its vector in a "
        "pgvector column with an HNSW index for fast approximate nearest neighbour lookups."
    ),
    "cats.txt": "Cats are small domesticated carnivores. They sleep for most of the day.",
}


def _kb(store: InMemoryVectorStore | None = None) -> KnowledgeBase:
    embedder = HashingEmbedder(dim=256)
    return KnowledgeBase(embedder, store or InMemoryVectorStore(dim=256), RecursiveChunker(300, 30))


@pytest.fixture
def kb() -> KnowledgeBase:
    kb = _kb()
    for name, text in DOCS.items():
        kb.ingest(text.encode(), name)
    return kb


def test_lexical_mode_finds_exact_code(kb: KnowledgeBase) -> None:
    hits = kb.search("ERR_7731", mode="lexical")
    assert hits[0].document_title == "Error codes"
    assert hits[0].ranks == {"lexical": 1}


def test_dense_mode_reports_dense_ranks(kb: KnowledgeBase) -> None:
    hits = kb.search("pgvector HNSW index", k=2, mode="dense")
    assert [h.ranks for h in hits] == [{"dense": 1}, {"dense": 2}]
    assert hits[0].score >= hits[1].score


def test_hybrid_combines_both_retrievers(kb: KnowledgeBase) -> None:
    hits = kb.search("ERR_7731 vector column", k=3)
    assert hits[0].document_title == "Error codes"
    assert set(hits[0].ranks) == {"dense", "lexical"}
    assert len(hits) == 3


def test_hybrid_respects_document_filter(kb: KnowledgeBase) -> None:
    cats = next(d for d in kb.store.list_documents() if d.title == "cats")
    hits = kb.search("vector postgres cats", document_ids=[cats.id])
    assert hits
    assert {h.chunk.document_id for h in hits} == {cats.id}


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query(kb: KnowledgeBase, query: str) -> None:
    assert kb.search(query) == []


def test_delete_removes_from_both_indexes(kb: KnowledgeBase) -> None:
    errors = next(d for d in kb.store.list_documents() if d.title == "Error codes")
    assert kb.delete(errors.id)
    assert kb.search("ERR_7731", mode="lexical") == []
    assert all(h.chunk.document_id != errors.id for h in kb.search("ERR_7731"))
    assert not kb.delete(errors.id)


def test_lexical_index_is_rebuilt_from_existing_store(kb: KnowledgeBase) -> None:
    restarted = _kb(store=kb.store)  # type: ignore[arg-type]
    assert len(restarted.lexical) == kb.store.count_chunks()
    assert restarted.search("ERR_7731", mode="lexical")[0].document_title == "Error codes"


def test_duplicate_ingest_does_not_double_index(kb: KnowledgeBase) -> None:
    before = len(kb.lexical)
    result = kb.ingest(DOCS["cats.txt"].encode(), "cats-again.txt")
    assert not result.created
    assert len(kb.lexical) == before


def test_lexical_mode_returns_raw_bm25_scores(kb: KnowledgeBase) -> None:
    hits = kb.search("ERR_7731", mode="lexical")
    assert hits[0].score == kb.lexical.search("ERR_7731")[0].score


class ReverseLengthReranker:
    """Prefers the shortest passage, so its effect on the order is easy to see."""

    name = "reverse-length"

    def __init__(self) -> None:
        self.seen: list[int] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.seen.append(len(passages))
        return [-float(len(p)) for p in passages]


def _reranked_kb(reranker: ReverseLengthReranker, candidates: int = 20) -> KnowledgeBase:
    embedder = HashingEmbedder(dim=256)
    kb = KnowledgeBase(
        embedder,
        InMemoryVectorStore(dim=256),
        RecursiveChunker(300, 30),
        reranker=reranker,
        rerank_candidates=candidates,
    )
    for name, text in DOCS.items():
        kb.ingest(text.encode(), name)
    return kb


def test_configured_reranker_is_used_by_default() -> None:
    reranker = ReverseLengthReranker()
    kb = _reranked_kb(reranker)
    hits = kb.search("vector postgres cats", k=2, mode="lexical")
    assert reranker.seen == [3]  # every lexical match became a candidate
    assert [h.ranks["rerank"] for h in hits] == [1, 2]
    assert len(hits[0].chunk.text) <= len(hits[1].chunk.text)
    assert "lexical" in hits[0].ranks  # retriever ranks are kept for explainability


def test_rerank_candidate_pool_is_at_least_k() -> None:
    reranker = ReverseLengthReranker()
    kb = _reranked_kb(reranker, candidates=1)
    assert len(kb.search("vector postgres cats", k=3, mode="dense")) == 3
    assert reranker.seen == [3]


def test_rerank_can_be_disabled_per_query() -> None:
    reranker = ReverseLengthReranker()
    hits = _reranked_kb(reranker).search("ERR_7731", rerank=False)
    assert reranker.seen == []
    assert all("rerank" not in h.ranks for h in hits)


def test_rerank_requested_without_reranker_raises(kb: KnowledgeBase) -> None:
    with pytest.raises(RerankerNotConfiguredError):
        kb.search("ERR_7731", rerank=True)

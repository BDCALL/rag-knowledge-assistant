from pathlib import Path

import pytest

from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.evaluation import LabelledQuery, evaluate_retrieval, load_queries
from rag_assistant.ingestion import RecursiveChunker
from rag_assistant.service import KnowledgeBase, SearchMode
from rag_assistant.store import InMemoryVectorStore

DATA = Path(__file__).parents[1] / "data"


@pytest.fixture
def sample_kb() -> KnowledgeBase:
    kb = KnowledgeBase(HashingEmbedder(dim=256), InMemoryVectorStore(dim=256), RecursiveChunker())
    for path in sorted((DATA / "sample").glob("*.md")):
        kb.ingest(path.read_bytes(), path.name)
    return kb


@pytest.mark.parametrize("mode", ["hybrid", "dense", "lexical"])
def test_sample_set_scores_are_valid(sample_kb: KnowledgeBase, mode: SearchMode) -> None:
    queries = load_queries(DATA / "eval" / "sample-queries.json")
    report = evaluate_retrieval(sample_kb, queries, k=3, mode=mode)
    assert (report.mode, report.k) == (mode, 3)
    assert 0.0 <= report.recall <= 1.0
    assert 0.0 <= report.mrr <= 1.0


def test_lexical_finds_exact_alert_code(sample_kb: KnowledgeBase) -> None:
    queries = [LabelledQuery("alert NWR-4031", frozenset({"incident-runbook"}))]
    report = evaluate_retrieval(sample_kb, queries, k=1, mode="lexical")
    assert report.recall == 1.0
    assert report.mrr == 1.0


def test_paraphrases_separate_lexical_from_hybrid(sample_kb: KnowledgeBase) -> None:
    # Paraphrases share no words with the documents, so BM25 alone finds nothing;
    # fusing in dense scores recovers them. This is why the labelled set includes them.
    queries = [
        LabelledQuery("vacation rollover deadline", frozenset({"leave-policy"})),
        LabelledQuery("who approves my time off", frozenset({"leave-policy"})),
    ]
    lexical = evaluate_retrieval(sample_kb, queries, k=3, mode="lexical")
    hybrid = evaluate_retrieval(sample_kb, queries, k=3, mode="hybrid")
    assert lexical.recall == 0.0
    assert hybrid.recall == 1.0


def test_evaluate_rejects_empty_query_set(sample_kb: KnowledgeBase) -> None:
    with pytest.raises(ValueError, match="at least one query"):
        evaluate_retrieval(sample_kb, [])

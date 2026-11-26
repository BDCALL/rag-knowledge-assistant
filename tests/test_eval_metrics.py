import pytest

from rag_assistant.evaluation import mean_reciprocal_rank, recall_at_k, reciprocal_rank


def test_recall_at_k_counts_relevant_ids_in_top_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, {"b", "d"}, k=2) == 0.5
    assert recall_at_k(retrieved, {"b", "d"}, k=4) == 1.0
    assert recall_at_k(retrieved, {"z"}, k=4) == 0.0


def test_recall_at_k_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="k must be"):
        recall_at_k(["a"], {"a"}, k=0)
    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(["a"], set(), k=1)


def test_reciprocal_rank_and_mrr() -> None:
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0
    results = [(["a"], {"a"}), (["x", "a"], {"a"}), (["x"], {"a"})]
    assert mean_reciprocal_rank(results) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="at least one query"):
        mean_reciprocal_rank([])

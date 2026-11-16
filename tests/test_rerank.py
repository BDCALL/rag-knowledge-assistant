from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_assistant.domain import Chunk, SearchHit
from rag_assistant.retrieval import (
    CrossEncoderReranker,
    Reranker,
    TermOverlapReranker,
    rerank,
)


def _hit(text: str, rank: int) -> SearchHit:
    chunk = Chunk(document_id="d", index=rank, text=text, start_char=0, end_char=len(text))
    return SearchHit(chunk, score=1.0 / rank, document_title="doc", ranks={"dense": rank})


class FakeCrossEncoderModel:
    """Mimics ``CrossEncoder.predict``: scores each (query, passage) pair by length."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
        self.calls.append(pairs)
        return [float(len(passage)) for _, passage in pairs]


class WrongLengthReranker:
    name = "broken"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [1.0]


def test_rerankers_satisfy_protocol() -> None:
    assert isinstance(TermOverlapReranker(), Reranker)
    assert isinstance(CrossEncoderReranker(model=FakeCrossEncoderModel()), Reranker)


def test_overlap_prefers_full_query_coverage() -> None:
    scores = TermOverlapReranker().score(
        "reset api key",
        ["how to reset your api key", "the api is rate limited", "cats sleep a lot"],
    )
    assert scores[0] > scores[1] > scores[2]
    assert scores[2] == 0.0


def test_overlap_rewards_rare_terms_more() -> None:
    # "hnsw" appears in 2 of 4 passages, "index" in 3, so covering "hnsw" counts more.
    scores = TermOverlapReranker(phrase_bonus=0).score(
        "hnsw index", ["hnsw graph", "btree index", "gin index", "hnsw index"]
    )
    assert scores[0] > scores[1]
    assert scores[3] == pytest.approx(1.0)


def test_overlap_phrase_bonus_breaks_ties() -> None:
    scores = TermOverlapReranker().score(
        "vector column", ["column of the vector", "the vector column"]
    )
    assert scores[1] > scores[0]


@pytest.mark.parametrize("query", ["", "the of and"])
def test_overlap_without_content_terms_scores_zero(query: str) -> None:
    assert TermOverlapReranker().score(query, ["anything", "else"]) == [0.0, 0.0]


def test_cross_encoder_passes_query_passage_pairs() -> None:
    model = FakeCrossEncoderModel()
    reranker = CrossEncoderReranker(model_name="tiny", model=model)
    assert reranker.score("q", ["aa", "a"]) == [2.0, 1.0]
    assert model.calls == [[("q", "aa"), ("q", "a")]]
    assert reranker.name == "cross-encoder:tiny"
    assert reranker.score("q", []) == []


def test_rerank_reorders_and_records_positions() -> None:
    hits = [_hit("short", 1), _hit("much longer text", 2), _hit("medium one", 3)]
    out = rerank(CrossEncoderReranker(model=FakeCrossEncoderModel()), "q", hits, limit=2)
    assert [h.chunk.text for h in out] == ["much longer text", "medium one"]
    assert out[0].ranks == {"dense": 2, "rerank": 1}
    assert out[1].ranks == {"dense": 3, "rerank": 2}
    assert out[0].score == 16.0


def test_rerank_ties_keep_retrieval_order() -> None:
    hits = [_hit("aaa", 1), _hit("bbb", 2), _hit("ccc", 3)]
    out = rerank(CrossEncoderReranker(model=FakeCrossEncoderModel()), "q", hits, limit=3)
    assert [h.ranks["dense"] for h in out] == [1, 2, 3]


def test_rerank_edge_cases() -> None:
    reranker = TermOverlapReranker()
    assert rerank(reranker, "q", [], limit=5) == []
    assert rerank(reranker, "q", [_hit("x", 1)], limit=0) == []
    with pytest.raises(ValueError, match="1 scores for 2 hits"):
        rerank(WrongLengthReranker(), "q", [_hit("a", 1), _hit("b", 2)], limit=2)

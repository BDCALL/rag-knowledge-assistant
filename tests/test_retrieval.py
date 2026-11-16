from __future__ import annotations

import math

import pytest

from rag_assistant.domain import Chunk, SearchHit
from rag_assistant.retrieval import BM25Index, reciprocal_rank_fusion, tokenize


def _chunk(cid: str, text: str, doc: str = "d1", index: int = 0) -> Chunk:
    return Chunk(document_id=doc, index=index, text=text, start_char=0, end_char=len(text), id=cid)


@pytest.fixture
def index() -> BM25Index:
    idx = BM25Index()
    idx.add(
        [
            _chunk("c1", "Postgres stores embeddings with the pgvector extension."),
            _chunk(
                "c2", "The HNSW index makes approximate nearest neighbour search fast.", index=1
            ),
            _chunk("c3", "Error ERR_42 means the embedding dimension does not match.", index=2),
        ],
        "Guide",
    )
    idx.add([_chunk("c4", "Cats sleep most of the day.", doc="d2")], "Cats")
    return idx


class TestTokenize:
    def test_lowercases_and_drops_stopwords(self) -> None:
        assert tokenize("The Vector IS in Postgres") == ["vector", "postgres"]

    def test_keeps_identifiers_whole(self) -> None:
        assert tokenize("call pg_vector v1.2 ERR_42") == ["call", "pg_vector", "v1.2", "err_42"]

    def test_keeps_negation(self) -> None:
        assert "not" in tokenize("this is not supported")


class TestBM25:
    def test_exact_rare_token_ranks_first(self, index: BM25Index) -> None:
        hits = index.search("ERR_42", k=3)
        assert [h.chunk.id for h in hits] == ["c3"]
        assert hits[0].document_title == "Guide"

    def test_scores_are_descending(self, index: BM25Index) -> None:
        hits = index.search("pgvector embeddings index search", k=4)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert len(hits) >= 2

    def test_no_overlap_returns_nothing(self, index: BM25Index) -> None:
        assert index.search("quantum chromodynamics") == []

    def test_stopword_only_query_returns_nothing(self, index: BM25Index) -> None:
        assert index.search("the of and") == []

    def test_non_positive_k(self, index: BM25Index) -> None:
        assert index.search("postgres", k=0) == []

    def test_document_filter(self, index: BM25Index) -> None:
        assert index.search("day", document_ids=["d1"]) == []
        assert [h.chunk.id for h in index.search("day", document_ids=["d2"])] == ["c4"]

    def test_rarer_terms_have_higher_idf(self) -> None:
        idx = BM25Index()
        idx.add(
            [_chunk(f"c{i}", "common word" + (" rare" if i == 0 else "")) for i in range(5)], "T"
        )
        assert idx.idf("rare") > idx.idf("common") > 0
        assert math.isclose(idx.idf("missing"), math.log(1 + 5.5 / 0.5))

    def test_term_frequency_saturates(self) -> None:
        idx = BM25Index(k1=1.2, b=0.0)
        idx.add([_chunk("a", "apple"), _chunk("b", "apple " * 50), _chunk("c", "pear")], "T")
        by_id = {h.chunk.id: h.score for h in idx.search("apple", k=3)}
        assert by_id["b"] > by_id["a"]
        assert by_id["b"] < by_id["a"] * (1.2 + 1)  # bounded by (k1 + 1)

    def test_length_normalisation_prefers_shorter_chunk(self) -> None:
        idx = BM25Index(b=1.0)
        filler = " ".join(f"filler{i}" for i in range(40))
        idx.add([_chunk("short", "vector search"), _chunk("long", f"vector search {filler}")], "T")
        assert idx.search("vector", k=2)[0].chunk.id == "short"

    def test_remove_document(self, index: BM25Index) -> None:
        assert index.remove_document("d1") == 3
        assert len(index) == 1
        assert index.search("postgres") == []
        assert index.remove_document("d1") == 0
        assert index.search("cats sleep")[0].chunk.id == "c4"

    def test_re_adding_same_chunk_is_idempotent(self, index: BM25Index) -> None:
        before = len(index)
        index.add([_chunk("c1", "Postgres stores embeddings with the pgvector extension.")], "G")
        assert len(index) == before

    def test_empty_index(self) -> None:
        assert BM25Index().search("anything") == []

    @pytest.mark.parametrize(("k1", "b"), [(-0.1, 0.5), (1.2, 1.5), (1.2, -0.1)])
    def test_rejects_invalid_parameters(self, k1: float, b: float) -> None:
        with pytest.raises(ValueError, match="BM25"):
            BM25Index(k1=k1, b=b)


def _hits(*ids: str) -> list[SearchHit]:
    return [SearchHit(chunk=_chunk(i, i), score=1.0, document_title="T") for i in ids]


class TestRRF:
    def test_agreement_beats_single_first_place(self) -> None:
        fused = reciprocal_rank_fusion(
            {"dense": _hits("a", "both"), "lexical": _hits("b", "both")}, k=60
        )
        assert fused[0].chunk.id == "both"
        assert fused[0].ranks == {"dense": 2, "lexical": 2}
        assert math.isclose(fused[0].score, 2 / 62)

    def test_scores_and_ranks(self) -> None:
        fused = reciprocal_rank_fusion({"dense": _hits("a", "b"), "lexical": _hits("b")}, k=0)
        assert [h.chunk.id for h in fused] == ["b", "a"]
        assert math.isclose(fused[0].score, 1 / 2 + 1)
        assert fused[1].ranks == {"dense": 1}

    def test_weights(self) -> None:
        fused = reciprocal_rank_fusion(
            {"dense": _hits("a"), "lexical": _hits("b")}, weights={"lexical": 2.0}
        )
        assert fused[0].chunk.id == "b"

    def test_limit_and_duplicates(self) -> None:
        fused = reciprocal_rank_fusion({"dense": _hits("a", "a", "b", "c")}, limit=2)
        assert [h.chunk.id for h in fused] == ["a", "b"]
        assert fused[0].ranks == {"dense": 1}

    def test_empty(self) -> None:
        assert reciprocal_rank_fusion({"dense": [], "lexical": []}) == []

    def test_negative_k_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            reciprocal_rank_fusion({}, k=-1)

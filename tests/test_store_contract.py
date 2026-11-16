"""One behavioural contract, run against every VectorStore implementation.

The pgvector variant needs a database: set TEST_DATABASE_URL, e.g.
``postgresql+psycopg://postgres:postgres@localhost:5432/rag_test``. CI provides one
via a ``pgvector/pgvector`` service container.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import pytest

from rag_assistant.domain import Chunk, Document
from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.ingestion import RecursiveChunker
from rag_assistant.store import InMemoryVectorStore, VectorStore

DIM = 64
EMBEDDER = HashingEmbedder(dim=DIM)
TEST_DB = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(
    params=[
        "memory",
        pytest.param(
            "pgvector",
            marks=[
                pytest.mark.integration,
                pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL not set"),
            ],
        ),
    ]
)
def store(request: pytest.FixtureRequest) -> Iterator[VectorStore]:
    if request.param == "memory":
        yield InMemoryVectorStore(dim=DIM)
        return
    from rag_assistant.store.pgvector import PgVectorStore

    assert TEST_DB is not None
    pg = PgVectorStore(TEST_DB, dim=DIM)
    pg.drop_schema()
    pg.create_schema()
    try:
        yield pg
    finally:
        pg.drop_schema()
        pg.engine.dispose()


def _ingest(store: VectorStore, title: str, text: str) -> Document:
    doc = Document(title=title, text=text, source=f"{title}.txt", media_type="text/plain")
    chunks = RecursiveChunker(chunk_size=200, overlap=20).split(doc)
    store.add(doc, chunks, EMBEDDER.embed([c.text for c in chunks]))
    return doc


def _query(text: str) -> np.ndarray:
    return EMBEDDER.embed([text])[0]


def test_empty_store(store: VectorStore) -> None:
    assert store.list_documents() == []
    assert store.count_chunks() == 0
    assert store.search(_query("anything")) == []
    assert store.ping()


def test_add_list_get(store: VectorStore) -> None:
    doc = _ingest(store, "vectors", "pgvector stores embeddings. " * 20)
    listed = store.list_documents()
    assert [d.id for d in listed] == [doc.id]
    summary = store.get(doc.id)
    assert summary is not None
    assert summary.title == "vectors"
    assert summary.chunk_count == store.count_chunks() > 1
    assert store.get("missing") is None


def test_find_by_hash(store: VectorStore) -> None:
    doc = _ingest(store, "a", "unique content here")
    found = store.find_by_hash(doc.content_hash)
    assert found is not None
    assert found.id == doc.id
    assert store.find_by_hash("0" * 64) is None


def test_search_ranks_relevant_document_first(store: VectorStore) -> None:
    _ingest(store, "cooking", "Whisk the eggs with flour and butter, then bake the cake.")
    db = _ingest(store, "databases", "pgvector adds vector similarity search to Postgres.")
    _ingest(store, "travel", "The train to Edinburgh leaves from Kings Cross station.")

    hits = store.search(_query("vector similarity search in Postgres"), k=2)
    assert len(hits) == 2
    assert hits[0].chunk.document_id == db.id
    assert hits[0].document_title == "databases"
    assert hits[0].score >= hits[1].score
    assert -1.0 <= hits[1].score <= hits[0].score <= 1.0 + 1e-6


def test_search_can_filter_by_document(store: VectorStore) -> None:
    a = _ingest(store, "a", "apples and oranges")
    _ingest(store, "b", "apples and pears")
    hits = store.search(_query("apples"), k=10, document_ids=[a.id])
    assert hits
    assert {h.chunk.document_id for h in hits} == {a.id}


def test_search_k_larger_than_corpus(store: VectorStore) -> None:
    _ingest(store, "tiny", "one small chunk")
    assert len(store.search(_query("small"), k=50)) == 1
    assert store.search(_query("small"), k=0) == []


def test_chunk_round_trip_preserves_fields(store: VectorStore) -> None:
    doc = _ingest(store, "rt", "Round trip text for offsets.")
    hit = store.search(_query("round trip"), k=1)[0]
    assert hit.chunk.text == "Round trip text for offsets."
    assert (hit.chunk.start_char, hit.chunk.end_char) == (0, len(doc.text))
    assert hit.chunk.metadata["source"] == "rt.txt"


def test_delete_cascades_to_chunks(store: VectorStore) -> None:
    keep = _ingest(store, "keep", "keep this document around")
    drop = _ingest(store, "drop", "drop this document please")
    assert store.delete(drop.id) is True
    assert store.delete(drop.id) is False
    assert [d.id for d in store.list_documents()] == [keep.id]
    assert all(h.chunk.document_id == keep.id for h in store.search(_query("document"), k=10))


def test_rejects_mismatched_embeddings(store: VectorStore) -> None:
    doc = Document(title="x", text="x", source="x", media_type="text/plain")
    chunk = Chunk(document_id=doc.id, index=0, text="x", start_char=0, end_char=1)
    with pytest.raises(ValueError, match="dim"):
        store.add(doc, [chunk], np.zeros((1, DIM + 1), dtype=np.float32))
    with pytest.raises(ValueError, match="Expected 1"):
        store.add(doc, [chunk], np.zeros((2, DIM), dtype=np.float32))


def test_iter_chunks_yields_every_chunk_with_title(store: VectorStore) -> None:
    a = _ingest(store, "alpha", "First paragraph about vectors.\n\n" * 20)
    b = _ingest(store, "beta", "Second document about cats.")
    pairs = list(store.iter_chunks())
    assert len(pairs) == store.count_chunks()
    titles = {chunk.document_id: title for chunk, title in pairs}
    assert titles == {a.id: "alpha", b.id: "beta"}
    alpha_idx = [c.index for c, _ in pairs if c.document_id == a.id]
    assert sorted(alpha_idx) == list(range(len(alpha_idx)))

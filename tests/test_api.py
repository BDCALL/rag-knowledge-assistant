from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api.app import create_app
from rag_assistant.config import Settings


@pytest.fixture
def client() -> Iterator[TestClient]:
    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_dim=128,
        chunk_size=200,
        chunk_overlap=20,
        max_upload_mb=0.01,  # ~10 KB, so the size limit is easy to test
    )
    with TestClient(create_app(settings)) as c:
        yield c


def _upload(client: TestClient, name: str, body: bytes, ctype: str = "text/plain") -> dict:
    response = client.post("/documents", files={"file": (name, body, ctype)})
    assert response.status_code in (200, 201), response.text
    return response.json()


def test_health_reports_config(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["store"] == "memory"
    assert body["embedder"] == "hashing-128"
    assert body["documents"] == 0


def test_upload_then_list_and_get(client: TestClient) -> None:
    created = _upload(client, "notes.md", b"# Vector Search\n\npgvector indexes embeddings.")
    assert created["created"] is True
    doc = created["document"]
    assert doc["title"] == "Vector Search"
    assert doc["chunk_count"] >= 1

    listed = client.get("/documents").json()
    assert listed["total"] == 1
    assert listed["documents"][0]["id"] == doc["id"]
    assert client.get(f"/documents/{doc['id']}").json()["title"] == "Vector Search"


def test_duplicate_upload_returns_200_and_existing_doc(client: TestClient) -> None:
    first = client.post("/documents", files={"file": ("a.txt", b"same body", "text/plain")})
    second = client.post("/documents", files={"file": ("b.txt", b"same body", "text/plain")})
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["document"]["id"] == first.json()["document"]["id"]
    assert client.get("/documents").json()["total"] == 1


def test_pdf_upload_records_pages(client: TestClient, sample_pdf: bytes) -> None:
    doc = _upload(client, "guide.pdf", sample_pdf, "application/pdf")["document"]
    assert doc["media_type"] == "application/pdf"
    assert doc["metadata"]["pages"] == 2
    hits = client.post("/search", json={"query": "cosine distance nearest"}).json()["hits"]
    assert hits[0]["page"] == 2


def test_search_returns_ranked_hits_with_offsets(client: TestClient) -> None:
    _upload(client, "cake.txt", b"Whisk eggs, flour and butter, then bake the cake for 30 minutes.")
    db = _upload(client, "db.txt", b"pgvector adds vector similarity search to Postgres.")
    response = client.post("/search", json={"query": "Postgres vector similarity", "k": 2})
    assert response.status_code == 200
    hits = response.json()["hits"]
    assert len(hits) == 2
    assert hits[0]["document_id"] == db["document"]["id"]
    assert hits[0]["score"] > hits[1]["score"]
    assert hits[0]["start_char"] == 0


def test_search_filter_by_document(client: TestClient) -> None:
    a = _upload(client, "a.txt", b"apples and oranges")["document"]["id"]
    _upload(client, "b.txt", b"apples and pears")
    hits = client.post("/search", json={"query": "apples", "document_ids": [a]}).json()["hits"]
    assert {h["document_id"] for h in hits} == {a}


def test_delete_document(client: TestClient) -> None:
    doc_id = _upload(client, "x.txt", b"temporary")["document"]["id"]
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.delete(f"/documents/{doc_id}").status_code == 404
    assert client.get(f"/documents/{doc_id}").status_code == 404
    assert client.post("/search", json={"query": "temporary"}).json()["hits"] == []


@pytest.mark.parametrize(
    ("name", "body", "ctype", "code"),
    [
        ("image.png", b"\x89PNG....", "image/png", 415),
        ("empty.txt", b"   ", "text/plain", 422),
        ("broken.pdf", b"%PDF-1.4 nope", "application/pdf", 422),
        ("huge.txt", b"a" * 20_000, "text/plain", 413),
    ],
)
def test_upload_errors(client: TestClient, name: str, body: bytes, ctype: str, code: int) -> None:
    response = client.post("/documents", files={"file": (name, body, ctype)})
    assert response.status_code == code
    assert "detail" in response.json()


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   \n\t"},
        {"query": "x" * 2001},
        {"query": "x", "k": 0},
        {"query": "x", "k": 51},
        {},
    ],
)
def test_search_validation(client: TestClient, payload: dict) -> None:
    assert client.post("/search", json=payload).status_code == 422


def test_openapi_schema_lists_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/documents", "/documents/{document_id}", "/search"} <= set(paths)


@pytest.mark.parametrize("mode", ["hybrid", "dense", "lexical"])
def test_search_modes(client: TestClient, mode: str) -> None:
    _upload(client, "codes.md", b"# Codes\n\nERR_1234 means the index is stale.")
    _upload(client, "other.md", b"# Other\n\nSomething unrelated about gardening.")
    body = client.post("/search", json={"query": "ERR_1234", "k": 2, "mode": mode}).json()
    assert body["mode"] == mode
    assert body["hits"][0]["document_title"] == "Codes"
    assert body["hits"][0]["ranks"]


def test_search_rejects_unknown_mode(client: TestClient) -> None:
    response = client.post("/search", json={"query": "x", "mode": "fuzzy"})
    assert response.status_code == 422


def test_search_without_reranker(client: TestClient) -> None:
    _upload(client, "codes.md", b"# Codes\n\nERR_1234 means the index is stale.")
    body = client.post("/search", json={"query": "ERR_1234"}).json()
    assert body["reranker"] is None
    assert "rerank" not in body["hits"][0]["ranks"]
    assert client.get("/health").json()["reranker"] is None
    response = client.post("/search", json={"query": "ERR_1234", "rerank": True})
    assert response.status_code == 400
    assert "RAG_RERANKER" in response.json()["detail"]


def test_search_with_configured_reranker() -> None:
    settings = Settings(environment="test", embedding_dim=128, reranker="term-overlap")
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").json()["reranker"] == "term-overlap"
        _upload(c, "codes.md", b"# Codes\n\nERR_1234 means the index is stale.")
        _upload(c, "other.md", b"# Other\n\nSomething unrelated about gardening.")
        body = c.post("/search", json={"query": "stale index", "k": 1}).json()
        assert body["reranker"] == "term-overlap"
        assert body["hits"][0]["document_title"] == "Codes"
        assert body["hits"][0]["ranks"]["rerank"] == 1
        opted_out = c.post("/search", json={"query": "stale index", "rerank": False}).json()
        assert opted_out["reranker"] is None
        assert all("rerank" not in h["ranks"] for h in opted_out["hits"])

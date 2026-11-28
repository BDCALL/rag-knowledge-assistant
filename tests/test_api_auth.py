from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from rag_assistant.api.app import create_app
from rag_assistant.api.auth import API_KEY_HEADER, api_key_guard
from rag_assistant.config import Settings


def _client(expected: str | None) -> TestClient:
    app = FastAPI()

    @app.get("/private", dependencies=[Depends(api_key_guard(expected))])
    def private() -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app)


def test_auth_off_lets_every_request_through() -> None:
    # "" is what an empty RAG_API_KEY= line in .env loads as.
    for expected in (None, ""):
        assert _client(expected).get("/private").status_code == 200


def test_missing_or_wrong_key_is_rejected() -> None:
    client = _client("s3cret")
    for headers in ({}, {API_KEY_HEADER: "wrong"}):
        response = client.get("/private", headers=headers)
        assert response.status_code == 401
        assert response.json()["detail"] == "Missing or invalid API key"


def test_correct_key_is_accepted() -> None:
    response = _client("s3cret").get("/private", headers={API_KEY_HEADER: "s3cret"})
    assert response.status_code == 200


def test_app_guards_every_route_except_health() -> None:
    settings = Settings(
        environment="test", store_backend="memory", embedding_dim=64, api_key="s3cret"
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/documents").status_code == 401
        assert client.post("/ask", json={"question": "hi"}).status_code == 401
        ok = client.get("/documents", headers={API_KEY_HEADER: "s3cret"})
        assert ok.status_code == 200

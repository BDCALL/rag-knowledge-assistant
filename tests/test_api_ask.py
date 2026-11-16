"""POST /ask and the SSE variant POST /ask/stream."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api.app import create_app, sse
from rag_assistant.config import Settings
from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.generation import NO_ANSWER, ChatMessage
from rag_assistant.ingestion import RecursiveChunker
from rag_assistant.service import KnowledgeBase
from rag_assistant.store import InMemoryVectorStore

LEAVE = b"# Leave policy\n\nStaff get 25 days of annual leave. Unused days expire in March."
ONCALL = b"# On-call\n\nPage the on-call engineer first. Escalate after 15 minutes."


def _settings() -> Settings:
    return Settings(environment="test", embedding_dim=128, chunk_size=200, chunk_overlap=20)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(_settings())) as c:
        for name, body in (("leave.md", LEAVE), ("oncall.md", ONCALL)):
            c.post("/documents", files={"file": (name, body, "text/markdown")})
        yield c


def _events(raw: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    events = []
    for block in raw.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_ask_returns_answer_with_numbered_cited_sources(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "How many days of annual leave?", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["llm"] == "fake"
    assert body["citations"]
    assert [s["number"] for s in body["sources"]] == [1, 2]
    first = body["sources"][0]
    assert first["document_title"] == "Leave policy"
    assert first["cited"] is True
    assert f"[{first['number']}]" in body["answer"]


def test_ask_with_no_documents_says_it_does_not_know() -> None:
    with TestClient(create_app(_settings())) as c:
        body = c.post("/ask", json={"question": "Anything?"}).json()
    assert body["answer"] == NO_ANSWER
    assert body["sources"] == []
    assert body["citations"] == []


@pytest.mark.parametrize(
    "payload", [{"question": ""}, {"question": "   "}, {"question": "q", "k": 21}, {}]
)
@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
def test_ask_validates_input(client: TestClient, path: str, payload: dict) -> None:
    assert client.post(path, json=payload).status_code == 422


def test_ask_rerank_without_reranker_is_400(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "leave", "rerank": True})
    assert response.status_code == 400
    assert client.post("/ask/stream", json={"question": "leave", "rerank": True}).status_code == 400


def test_health_reports_llm(client: TestClient) -> None:
    assert client.get("/health").json()["llm"] == "fake"


def test_ask_stream_sends_sources_then_tokens_then_done(client: TestClient) -> None:
    response = client.post("/ask/stream", json={"question": "When do I escalate?", "k": 2})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert names[-1] == "done"
    assert set(names[1:-1]) == {"token"}

    sources = events[0][1]
    assert sources["question"] == "When do I escalate?"
    assert sources["sources"][0]["number"] == 1
    answer = "".join(data["text"] for name, data in events if name == "token")
    # The streamed answer matches the non-streaming endpoint exactly.
    blocking = client.post("/ask", json={"question": "When do I escalate?", "k": 2}).json()
    assert answer == blocking["answer"]
    assert events[-1][1]["citations"] == blocking["citations"]


class ExplodingLLM:
    name = "exploding"

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        yield "Partial "
        raise RuntimeError("provider timed out")


def test_ask_stream_reports_generation_errors_in_band() -> None:
    kb = KnowledgeBase(
        HashingEmbedder(dim=128),
        InMemoryVectorStore(dim=128),
        RecursiveChunker(200, 20),
        llm=ExplodingLLM(),
    )
    kb.ingest(LEAVE, "leave.md")
    with TestClient(create_app(_settings(), kb=kb)) as c:
        events = _events(c.post("/ask/stream", json={"question": "leave"}).text)
    assert [name for name, _ in events] == ["sources", "token", "error"]
    assert events[-1][1] == {"detail": "Answer generation failed"}


def test_sse_keeps_multiline_tokens_on_one_data_line() -> None:
    assert sse("token", {"text": "a\nb"}) == 'event: token\ndata: {"text": "a\\nb"}\n\n'

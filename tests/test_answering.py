"""KnowledgeBase.ask / stream_answer and the LLM factory."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from rag_assistant.config import Settings
from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.generation import NO_ANSWER, ChatMessage, FakeLLM, OpenAIChatLLM, build_llm
from rag_assistant.ingestion import RecursiveChunker
from rag_assistant.service import KnowledgeBase
from rag_assistant.store import InMemoryVectorStore

DOCS = {
    "leave.md": "# Leave policy\n\nStaff get 25 days of annual leave. Unused days expire in March.",
    "oncall.md": "# On-call\n\nPage the on-call engineer first. Escalate after 15 minutes.",
}


class RecordingLLM:
    """Returns a canned answer and remembers the prompt it was given."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[Sequence[ChatMessage]] = []

    @property
    def name(self) -> str:
        return "recording"

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        self.calls.append(messages)
        yield from self.answer.split("|")


def _kb(llm: RecordingLLM | FakeLLM | None = None, ingest: bool = True) -> KnowledgeBase:
    kb = KnowledgeBase(
        HashingEmbedder(dim=256), InMemoryVectorStore(dim=256), RecursiveChunker(300, 30), llm=llm
    )
    if ingest:
        for name, text in DOCS.items():
            kb.ingest(text.encode(), name)
    return kb


def test_default_llm_is_the_offline_fake() -> None:
    assert _kb(ingest=False).llm.name == "fake"


def test_ask_grounds_the_answer_in_retrieved_sources() -> None:
    answer = _kb(FakeLLM(max_sources=1)).ask("How many days of annual leave?", k=2)
    assert answer.sources[0].document_title == "Leave policy"
    assert answer.text.endswith("[1]")
    assert "25 days of annual leave" in answer.text
    assert answer.citations == [1]
    assert answer.cited_sources == [(1, answer.sources[0])]
    assert answer.llm == "fake"
    assert answer.elapsed_ms >= 0


def test_ask_drops_hallucinated_citations_and_maps_numbers_to_sources() -> None:
    llm = RecordingLLM("Escalate after 15 minutes |[2]. See also [7].")
    answer = _kb(llm).ask("  When do I escalate?  ", k=2)
    assert answer.question == "When do I escalate?"
    assert answer.text == "Escalate after 15 minutes [2]. See also [7]."
    assert answer.citations == [2]  # [7] was never a source
    assert answer.cited_sources[0][1] is answer.sources[1]
    # The prompt shows exactly the hits the answer is mapped back to.
    user = llm.calls[0][-1].content
    assert f"[1] {answer.sources[0].document_title}" in user
    assert user.endswith("Question: When do I escalate?")


def test_ask_without_sources_skips_the_model() -> None:
    llm = RecordingLLM("should not be used")
    answer = _kb(llm, ingest=False).ask("Anything?")
    assert answer.text == NO_ANSWER
    assert answer.sources == []
    assert answer.citations == []
    assert llm.calls == []


def test_stream_answer_retrieves_eagerly_but_generates_lazily() -> None:
    llm = RecordingLLM("a|b|c")
    stream = _kb(llm).stream_answer("leave", k=1)
    assert len(stream.sources) == 1
    assert llm.calls == []  # nothing generated until the stream is consumed
    assert list(stream.deltas) == ["a", "b", "c"]
    assert len(llm.calls) == 1


def test_build_llm_fake_by_default() -> None:
    assert isinstance(build_llm(Settings()), FakeLLM)


def test_build_llm_openai_uses_model_settings() -> None:
    pytest.importorskip("openai")
    llm = build_llm(Settings(llm_provider="openai", openai_api_key="sk-test", llm_model="gpt-x"))
    assert isinstance(llm, OpenAIChatLLM)
    assert llm.name == "openai:gpt-x"


def test_openai_llm_requires_api_key() -> None:
    with pytest.raises(ValueError, match="llm_provider=openai"):
        Settings(llm_provider="openai", openai_api_key=None)

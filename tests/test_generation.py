from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from rag_assistant.domain import Chunk, SearchHit
from rag_assistant.generation import (
    LLM,
    NO_ANSWER,
    SYSTEM_PROMPT,
    FakeLLM,
    OpenAIChatLLM,
    build_messages,
    complete,
    extract_citations,
    format_sources,
    source_label,
)


def _hit(text: str, title: str = "Handbook", page: int | None = None) -> SearchHit:
    metadata: dict[str, object] = {} if page is None else {"page": page}
    chunk = Chunk("d", 0, text, 0, len(text), metadata=metadata)
    return SearchHit(chunk, score=1.0, document_title=title)


HITS = [
    _hit("Staff get 25 days of leave. Unused days expire.", "Leave policy", page=2),
    _hit("  The API runs on FastAPI.\nIt uses pgvector.  ", "Architecture"),
    _hit("Page the on-call engineer first!", "Runbook"),
]


def test_source_label_includes_page_only_when_known() -> None:
    assert source_label(HITS[0]) == "Leave policy (p. 2)"
    assert source_label(HITS[1]) == "Architecture"


def test_format_sources_numbers_blocks_and_strips_text() -> None:
    text = format_sources(HITS[:2])
    assert text == (
        "[1] Leave policy (p. 2)\nStaff get 25 days of leave. Unused days expire.\n\n"
        "[2] Architecture\nThe API runs on FastAPI.\nIt uses pgvector."
    )


def test_build_messages_has_system_then_user_turn() -> None:
    system, user = build_messages("  How much leave?  ", HITS)
    assert (system.role, system.content) == ("system", SYSTEM_PROMPT)
    assert user.role == "user"
    assert user.content.startswith("Sources:\n\n[1] Leave policy (p. 2)")
    assert user.content.endswith("\n\nQuestion: How much leave?")


def test_extract_citations_dedupes_keeps_order_and_drops_out_of_range() -> None:
    answer = "Leave is 25 days [2][1]. Also [2], see [0] and [9]."
    assert extract_citations(answer, source_count=3) == [2, 1]
    assert extract_citations("no citations here", source_count=3) == []


def test_extract_citations_understands_grouped_and_range_citations() -> None:
    assert extract_citations("See [3, 1] and [1,2].", source_count=3) == [3, 1, 2]
    assert extract_citations("Covered in [2-4].", source_count=5) == [2, 3, 4]
    assert extract_citations("En dash [1\u20132].", source_count=5) == [1, 2]


def test_extract_citations_clips_ranges_and_ignores_reversed_ones() -> None:
    assert extract_citations("Everything [1-999]", source_count=3) == [1, 2, 3]
    assert extract_citations("Backwards [3-1]", source_count=5) == []
    assert extract_citations("Not a citation [a, b] or [1,]", source_count=5) == []


def test_fake_llm_answers_from_first_sentences_with_citations() -> None:
    llm = FakeLLM()
    assert isinstance(llm, LLM)
    answer = complete(llm, build_messages("How much leave?", HITS))
    assert answer == "Staff get 25 days of leave. [1] The API runs on FastAPI. [2]"
    assert extract_citations(answer, len(HITS)) == [1, 2]


def test_fake_llm_streams_word_deltas() -> None:
    deltas = list(FakeLLM(max_sources=1).stream(build_messages("q", HITS)))
    assert len(deltas) > 1
    assert "".join(deltas) == "Staff get 25 days of leave. [1]"


def test_fake_llm_says_it_does_not_know_without_sources() -> None:
    assert complete(FakeLLM(), build_messages("Anything?", [])) == NO_ANSWER


class FakeOpenAIClient:
    """Mimics ``client.chat.completions.create(stream=True)``."""

    def __init__(self, deltas: list[str | None]) -> None:
        self.deltas = deltas
        self.kwargs: dict[str, Any] = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.kwargs = kwargs
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=d))])
            for d in self.deltas
        ]
        return [*chunks, SimpleNamespace(choices=[])]  # final usage-only chunk


def test_openai_llm_streams_content_and_skips_empty_deltas() -> None:
    client = FakeOpenAIClient(["Leave is ", None, "25 days [1].", ""])
    llm = OpenAIChatLLM(api_key="unused", model="gpt-test", client=client)
    assert llm.name == "openai:gpt-test"
    assert complete(llm, build_messages("q", HITS[:1])) == "Leave is 25 days [1]."
    assert client.kwargs["model"] == "gpt-test"
    assert client.kwargs["stream"] is True
    assert client.kwargs["temperature"] == 0.0
    assert [m["role"] for m in client.kwargs["messages"]] == ["system", "user"]

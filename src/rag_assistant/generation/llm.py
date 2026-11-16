"""Chat LLM providers behind one small :class:`LLM` protocol.

``stream`` yields text deltas so the API can forward tokens over SSE as they
arrive; ``complete`` is just the joined stream. :class:`FakeLLM` is
deterministic and needs no network, so tests and CI exercise the full answer
path without an API key.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

from rag_assistant.generation.prompt import NO_ANSWER, ChatMessage


@runtime_checkable
class LLM(Protocol):
    @property
    def name(self) -> str: ...

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        """Yield the answer as a sequence of text deltas."""
        ...


def complete(llm: LLM, messages: Sequence[ChatMessage]) -> str:
    """Collect a full answer from any provider's stream."""
    return "".join(llm.stream(messages))


# A source block in the user turn: "[n] label" on one line, the text below it,
# up to the next "[m] " header or the question.
_SOURCE_BLOCK = re.compile(r"^\[(\d+)\] [^\n]*\n(.*?)(?=\n\n\[\d+\] |\n\nQuestion:)", re.M | re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


class FakeLLM:
    """Extractive stand-in for a real model.

    It reads the numbered sources back out of the prompt and answers with the
    first sentence of each of the first ``max_sources`` sources, cited ``[n]``.
    Not clever, but it obeys the same contract as a real model (grounded text with
    valid citations), streamed word by word.
    """

    def __init__(self, max_sources: int = 2) -> None:
        self.max_sources = max_sources

    @property
    def name(self) -> str:
        return "fake"

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        sentences = [
            f"{_first_sentence(text)} [{number}]"
            for number, text in _SOURCE_BLOCK.findall(user)[: self.max_sources]
        ]
        answer = " ".join(sentences) if sentences else NO_ANSWER
        words = answer.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "


def _first_sentence(text: str) -> str:
    flat = " ".join(text.split())
    sentence = _SENTENCE_END.split(flat, maxsplit=1)[0]
    return sentence.rstrip(".!?") + "."


class OpenAIChatLLM:
    """OpenAI Chat Completions with streaming (``pip install .[openai]``)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError("openai is not installed; run pip install '.[openai]'") from exc
            client = OpenAI(api_key=api_key)
        self._client = client
        self._model = model
        self._temperature = temperature

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self._temperature,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

"""Answer generation: grounded prompts with numbered citations, and LLM providers."""

from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.generation.llm import LLM, FakeLLM, OpenAIChatLLM, complete
from rag_assistant.generation.prompt import (
    NO_ANSWER,
    SYSTEM_PROMPT,
    ChatMessage,
    build_messages,
    extract_citations,
    format_sources,
    source_label,
)


def build_llm(settings: Settings) -> LLM:
    """Factory for the configured answer model."""
    match settings.llm_provider:
        case "fake":
            return FakeLLM()
        case "openai":
            assert settings.openai_api_key is not None  # enforced by Settings
            return OpenAIChatLLM(
                settings.openai_api_key,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
            )


__all__ = [
    "LLM",
    "NO_ANSWER",
    "SYSTEM_PROMPT",
    "ChatMessage",
    "FakeLLM",
    "OpenAIChatLLM",
    "build_llm",
    "build_messages",
    "complete",
    "extract_citations",
    "format_sources",
    "source_label",
]

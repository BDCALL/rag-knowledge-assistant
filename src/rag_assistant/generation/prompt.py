"""Turn retrieved chunks into a grounded, citable prompt.

The model sees each retrieved chunk as a numbered source block::

    [1] Leave policy (p. 2)
    Employees accrue 25 days of annual leave...

and is told to answer *only* from those sources and to cite them inline as
``[1]``, ``[2]``. Numbering the sources (rather than passing chunk IDs) keeps the
citations short for the model and easy to map back to chunks afterwards.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rag_assistant.domain import SearchHit

Role = Literal["system", "user", "assistant"]

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions about the user's documents.\n"
    "Use ONLY the numbered sources provided. Cite every claim inline with the source "
    "number in square brackets, e.g. [1] or [2][3]. If the sources do not contain the "
    "answer, say you don't know instead of guessing."
)

NO_ANSWER = "I don't know based on the provided documents."

# One bracket group: "[1]", "[1, 3]" or "[2-4]" (models often group citations).
# Ranges accept a hyphen or an en dash (\u2013), which some models emit.
_DASH = "-\u2013"
_CITATION = re.compile(rf"\[(\d+(?:\s*[{_DASH},]\s*\d+)*)\]")
_ITEM = re.compile(rf"(\d+)(?:\s*[{_DASH}]\s*(\d+))?")


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


def source_label(hit: SearchHit) -> str:
    """Human-readable label for a source: its document title, plus page if known."""
    page = hit.chunk.metadata.get("page")
    return f"{hit.document_title} (p. {page})" if page is not None else hit.document_title


def format_sources(hits: Sequence[SearchHit]) -> str:
    """Render hits as numbered source blocks separated by blank lines."""
    return "\n\n".join(
        f"[{number}] {source_label(hit)}\n{hit.chunk.text.strip()}"
        for number, hit in enumerate(hits, start=1)
    )


def build_messages(question: str, hits: Sequence[SearchHit]) -> list[ChatMessage]:
    """System instructions plus one user turn holding the sources and the question."""
    user = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question.strip()}"
    return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user)]


def extract_citations(answer: str, source_count: int) -> list[int]:
    """Source numbers cited in ``answer``, in first-seen order, without duplicates.

    Grouped forms are understood too: ``[1, 3]`` cites 1 and 3, and ``[2-4]``
    cites 2, 3 and 4. Numbers outside ``1..source_count`` are dropped: a model
    citing ``[7]`` when it was only given five sources has hallucinated that
    citation. Ranges are clipped to that window, so ``[1-999]`` can't blow up.
    """
    seen: list[int] = []
    for match in _CITATION.finditer(answer):
        for item in _ITEM.finditer(match.group(1)):
            start = int(item.group(1))
            end = int(item.group(2)) if item.group(2) else start
            for number in range(max(start, 1), min(end, source_count) + 1):
                if number not in seen:
                    seen.append(number)
    return seen

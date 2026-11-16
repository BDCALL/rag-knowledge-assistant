"""Recursive, structure-aware text chunking with overlap.

Strategy (similar in spirit to LangChain's RecursiveCharacterTextSplitter, written
from scratch so every decision is explainable):

1. Try to split on the "largest" natural boundary first: pages, then paragraphs,
   then lines, then sentences, then words.
2. Greedily pack the resulting pieces into chunks of at most ``chunk_size`` chars.
3. Never merge across a page break (``\f``), so every chunk maps to one page and
   citations can name it.
4. Carry the last ``overlap`` characters (rounded to a word boundary) into the next
   chunk so an answer that straddles a boundary is still retrievable.

Character offsets into the original text are tracked for every chunk, which lets
the UI highlight exactly where a citation came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_assistant.domain import Chunk, Document

# Ordered from coarsest to finest. The sentence pattern keeps the punctuation.
_SEPARATORS: tuple[str, ...] = ("\f", "\n\n", "\n", r"(?<=[.!?])\s+", " ")


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    end: int


class RecursiveChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    # -- public API -----------------------------------------------------------------

    def split(self, document: Document) -> list[Chunk]:
        text = document.text
        # Pages are always split first, even when the whole text would fit in one chunk.
        atoms: list[_Span] = []
        page_start = 0
        for match in re.finditer("\f", text):
            atoms.extend(self._atomic_spans(text, page_start, match.end(), 1))
            page_start = match.end()
        atoms.extend(self._atomic_spans(text, page_start, len(text), 1))
        spans = self._merge(text, atoms)
        page_starts = [0] + [m.end() for m in re.finditer("\f", text)]

        chunks: list[Chunk] = []
        for span in spans:
            body = text[span.start : span.end].replace("\f", "\n").strip()
            if not body:
                continue
            metadata: dict[str, object] = {"source": document.source}
            if len(page_starts) > 1:
                metadata["page"] = _page_of(span.start, page_starts)
            chunks.append(
                Chunk(
                    document_id=document.id,
                    index=len(chunks),
                    text=body,
                    start_char=span.start,
                    end_char=span.end,
                    metadata=metadata,
                )
            )
        return chunks

    # -- internals ------------------------------------------------------------------

    def _atomic_spans(self, text: str, start: int, end: int, level: int) -> list[_Span]:
        """Split [start, end) into pieces no longer than chunk_size."""
        if end - start <= self.chunk_size:
            return [_Span(start, end)] if text[start:end].strip() else []
        if level >= len(_SEPARATORS):
            # No separator left: hard-cut the run (e.g. a very long URL).
            return [
                _Span(i, min(i + self.chunk_size, end)) for i in range(start, end, self.chunk_size)
            ]

        pieces: list[_Span] = []
        cursor = start
        for match in re.finditer(_SEPARATORS[level], text[start:end]):
            sep_end = start + match.end()
            pieces.append(_Span(cursor, sep_end))
            cursor = sep_end
        pieces.append(_Span(cursor, end))

        if len(pieces) == 1:
            return self._atomic_spans(text, start, end, level + 1)

        out: list[_Span] = []
        for piece in pieces:
            out.extend(self._atomic_spans(text, piece.start, piece.end, level + 1))
        return out

    def _merge(self, text: str, atoms: list[_Span]) -> list[_Span]:
        """Greedily pack atoms into chunks, adding overlap between neighbours."""
        merged: list[_Span] = []
        current: _Span | None = None
        for atom in atoms:
            if current is None:
                current = atom
            elif atom.end - current.start <= self.chunk_size and not _crosses_page(
                text, current, atom
            ):
                current = _Span(current.start, atom.end)
            else:
                merged.append(current)
                overlap_start = self._overlap_start(text, current, atom)
                current = _Span(overlap_start, atom.end)
        if current is not None:
            merged.append(current)
        return merged

    def _overlap_start(self, text: str, previous: _Span, nxt: _Span) -> int:
        if self.overlap == 0:
            return nxt.start
        # Never let overlap push the next chunk past chunk_size.
        budget = min(self.overlap, self.chunk_size - (nxt.end - nxt.start))
        if budget <= 0:
            return nxt.start
        candidate = max(previous.start, previous.end - budget)
        # Snap forward to the next word boundary so we don't start mid-word.
        space = text.find(" ", candidate, previous.end)
        if space != -1 and space + 1 < previous.end:
            candidate = space + 1
        # Don't carry content across a page break.
        page_break = text.rfind("\f", candidate, previous.end)
        if page_break != -1:
            return nxt.start
        return candidate


def _page_of(offset: int, page_starts: list[int]) -> int:
    """1-based page number of a character offset."""
    page = 1
    for i, start in enumerate(page_starts):
        if offset >= start:
            page = i + 1
        else:
            break
    return page


def _crosses_page(text: str, current: _Span, atom: _Span) -> bool:
    """True if joining ``current`` and ``atom`` would span a page break."""
    return "\f" in text[current.end - 1 : atom.start + 1]

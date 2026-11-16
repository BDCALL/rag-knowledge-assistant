from __future__ import annotations

from itertools import pairwise

import pytest

from rag_assistant.domain import Document
from rag_assistant.ingestion.chunking import RecursiveChunker


def _doc(text: str) -> Document:
    return Document(title="t", text=text, source="t.txt", media_type="text/plain")


def test_short_text_is_a_single_chunk() -> None:
    chunks = RecursiveChunker(chunk_size=200, overlap=20).split(_doc("A short note."))
    assert len(chunks) == 1
    assert chunks[0].text == "A short note."
    assert (chunks[0].start_char, chunks[0].end_char) == (0, 13)


def test_chunks_respect_size_limit(long_document: Document) -> None:
    chunker = RecursiveChunker(chunk_size=300, overlap=50)
    chunks = chunker.split(long_document)
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_offsets_point_back_into_source(long_document: Document) -> None:
    for chunk in RecursiveChunker(chunk_size=250, overlap=40).split(long_document):
        assert long_document.text[chunk.start_char : chunk.end_char].strip() == chunk.text


def test_every_word_is_covered(long_document: Document) -> None:
    chunks = RecursiveChunker(chunk_size=250, overlap=40).split(long_document)
    covered = " ".join(c.text for c in chunks)
    for word in long_document.text.split():
        assert word in covered


def test_neighbouring_chunks_overlap(long_document: Document) -> None:
    chunks = RecursiveChunker(chunk_size=250, overlap=60).split(long_document)
    overlapping = sum(1 for a, b in pairwise(chunks) if b.start_char < a.end_char)
    assert overlapping >= len(chunks) // 2


def test_zero_overlap_gives_disjoint_chunks(long_document: Document) -> None:
    chunks = RecursiveChunker(chunk_size=250, overlap=0).split(long_document)
    for a, b in pairwise(chunks):
        assert b.start_char >= a.end_char


def test_prefers_paragraph_boundaries() -> None:
    text = ("alpha " * 30).strip() + "\n\n" + ("beta " * 30).strip()
    chunks = RecursiveChunker(chunk_size=200, overlap=0).split(_doc(text))
    assert len(chunks) == 2
    assert chunks[0].text.startswith("alpha")
    assert "beta" not in chunks[0].text
    assert chunks[1].text.startswith("beta")


def test_unbroken_run_is_hard_cut() -> None:
    chunks = RecursiveChunker(chunk_size=100, overlap=0).split(_doc("x" * 350))
    assert [len(c.text) for c in chunks] == [100, 100, 100, 50]


def test_page_numbers_recorded_and_not_overlapped() -> None:
    text = ("page one text. " * 10).strip() + "\f" + ("page two text. " * 10).strip()
    chunks = RecursiveChunker(chunk_size=160, overlap=40).split(_doc(text))
    pages = [c.metadata["page"] for c in chunks]
    assert pages[0] == 1
    assert pages[-1] == 2
    assert all("\f" not in c.text for c in chunks)
    second_page = [c for c in chunks if c.metadata["page"] == 2]
    assert all("one" not in c.text for c in second_page)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="must be"):
        RecursiveChunker(chunk_size=size, overlap=overlap)


def test_short_pages_are_never_merged() -> None:
    chunks = RecursiveChunker(chunk_size=500, overlap=50).split(_doc("Page one.\fPage two."))
    assert [c.text for c in chunks] == ["Page one.", "Page two."]
    assert [c.metadata["page"] for c in chunks] == [1, 2]

"""An incremental Okapi BM25 index over chunks.

Dense embeddings are good at paraphrase ("how is data persisted?" ≈ "storage"), but
weak on exact tokens: error codes, function names, acronyms, rare proper nouns.
BM25 is the opposite. Running both and fusing the rankings (see :mod:`fusion`)
gets the best of each, which is why hybrid search is the default in most
production RAG systems.

BM25 score for query terms ``q`` and document ``d``::

    Σ_t∈q  idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

The index keeps an inverted list ``term -> {chunk_id: tf}`` so a query only touches
chunks that share at least one term with it, instead of scoring the whole corpus.
"""

from __future__ import annotations

import heapq
import math
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rag_assistant.domain import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[_.'-][a-z0-9]+)*")

# Deliberately short: function words carry no signal, but negations ("not", "no") and
# domain words must survive, and IDF already down-weights merely common terms.
# fmt: off
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "will", "with",
})
# fmt: on


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; keeps identifiers like ``pg_vector`` and ``v1.2`` whole."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True, slots=True)
class LexicalHit:
    chunk: Chunk
    score: float
    document_title: str


class BM25Index:
    """Thread-safe in-process BM25 index supporting add and delete-by-document."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._lengths: dict[str, int] = {}
        self._chunks: dict[str, tuple[Chunk, str]] = {}
        self._by_document: dict[str, list[str]] = defaultdict(list)
        self._total_len = 0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: Iterable[Chunk], document_title: str) -> None:
        with self._lock:
            for chunk in chunks:
                if chunk.id in self._chunks:
                    continue
                counts = Counter(tokenize(chunk.text))
                for term, tf in counts.items():
                    self._postings[term][chunk.id] = tf
                length = sum(counts.values())
                self._lengths[chunk.id] = length
                self._total_len += length
                self._chunks[chunk.id] = (chunk, document_title)
                self._by_document[chunk.document_id].append(chunk.id)

    def remove_document(self, document_id: str) -> int:
        """Drop every chunk of a document; returns how many were removed."""
        with self._lock:
            chunk_ids = self._by_document.pop(document_id, [])
            for cid in chunk_ids:
                chunk, _ = self._chunks.pop(cid)
                self._total_len -= self._lengths.pop(cid)
                for term in set(tokenize(chunk.text)):
                    postings = self._postings.get(term)
                    if postings is not None:
                        postings.pop(cid, None)
                        if not postings:
                            del self._postings[term]
            return len(chunk_ids)

    def idf(self, term: str) -> float:
        n = len(self._chunks)
        df = len(self._postings.get(term, ()))
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(
        self, query: str, k: int = 5, document_ids: Sequence[str] | None = None
    ) -> list[LexicalHit]:
        terms = tokenize(query)
        if k <= 0 or not terms:
            return []
        allowed = set(document_ids) if document_ids is not None else None
        with self._lock:
            if not self._chunks:
                return []
            avgdl = self._total_len / len(self._chunks) or 1.0
            scores: dict[str, float] = defaultdict(float)
            for term, qtf in Counter(terms).items():
                postings = self._postings.get(term)
                if not postings:
                    continue
                idf = self.idf(term)
                for cid, tf in postings.items():
                    if allowed is not None and self._chunks[cid][0].document_id not in allowed:
                        continue
                    norm = self.k1 * (1 - self.b + self.b * self._lengths[cid] / avgdl)
                    scores[cid] += qtf * idf * tf * (self.k1 + 1) / (tf + norm)
            # Ties broken by chunk id so results are deterministic.
            top = heapq.nsmallest(k, scores.items(), key=lambda kv: (-kv[1], kv[0]))
            return [
                LexicalHit(chunk=self._chunks[cid][0], score=s, document_title=self._chunks[cid][1])
                for cid, s in top
            ]

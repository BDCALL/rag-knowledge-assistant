"""Labelled query sets for retrieval evaluation.

A query set is a JSON list of ``{"question": str, "relevant": [str, ...]}`` objects,
where ``relevant`` holds the titles of the documents that answer the question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelledQuery:
    question: str
    relevant: frozenset[str]


def load_queries(path: Path) -> list[LabelledQuery]:
    """Read and validate a labelled query set from ``path``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty JSON list of queries")
    queries = []
    for i, item in enumerate(raw):
        question = str(item.get("question", "")).strip()
        relevant = frozenset(str(r) for r in item.get("relevant", []))
        if not question or not relevant:
            raise ValueError(f"{path}: query {i} needs a question and relevant titles")
        queries.append(LabelledQuery(question, relevant))
    return queries

import json
from pathlib import Path

import pytest

from rag_assistant.evaluation import load_queries

SAMPLE = Path(__file__).parents[1] / "data"


def test_sample_query_set_points_at_real_documents() -> None:
    queries = load_queries(SAMPLE / "eval" / "sample-queries.json")
    titles = {p.stem for p in (SAMPLE / "sample").glob("*.md")}
    assert len(queries) >= 8
    assert all(q.relevant <= titles for q in queries)


@pytest.mark.parametrize("bad", [[], [{"question": " ", "relevant": ["a"]}], [{"question": "q"}]])
def test_load_queries_rejects_bad_files(tmp_path: Path, bad: object) -> None:
    path = tmp_path / "q.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match=r"q\.json"):
        load_queries(path)

from __future__ import annotations

import io
from pathlib import Path

import pytest

from rag_assistant.cli import SAMPLE_DIR, iter_files, main
from rag_assistant.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(environment="test", embedding_dim=128, chunk_size=300, chunk_overlap=30)


def _run(argv: list[str], settings: Settings) -> tuple[int, str]:
    out = io.StringIO()
    code = main(argv, settings=settings, out=out)
    return code, out.getvalue()


def test_sample_corpus_is_shipped() -> None:
    names = [p.name for p in iter_files([SAMPLE_DIR])]
    assert "incident-runbook.md" in names
    assert "README.md" not in names


def test_iter_files_filters_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.md").write_text("c")
    assert [p.name for p in iter_files([tmp_path])] == ["a.txt", "b.md", "c.md"]


def test_iter_files_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_files([tmp_path / "nope"]))


def test_ingest_reports_each_file_and_duplicates(tmp_path: Path, settings: Settings) -> None:
    (tmp_path / "one.md").write_text("# One\n\nAlpha beta gamma.")
    (tmp_path / "copy.md").write_text("# One\n\nAlpha beta gamma.")
    code, out = _run(["ingest", str(tmp_path)], settings)
    assert code == 0
    assert "copy.md: indexed" in out
    assert "one.md: skipped (duplicate)" in out


def test_ingest_failure_sets_exit_code(tmp_path: Path, settings: Settings) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf")
    code, out = _run(["ingest", str(bad)], settings)
    assert code == 1
    assert "FAILED" in out


def test_ingest_missing_path_exits_2(tmp_path: Path, settings: Settings) -> None:
    code, _ = _run(["ingest", str(tmp_path / "missing")], settings)
    assert code == 2


def test_search_on_empty_store(settings: Settings) -> None:
    code, out = _run(["search", "anything", "--mode", "lexical"], settings)
    assert code == 0
    assert "(no results)" in out


def test_demo_compares_modes_on_sample_corpus(settings: Settings) -> None:
    code, out = _run(["demo", "-k", "1", "NWR-4031"], settings)
    assert code == 0
    assert out.count("-- ") == 3
    hybrid = out.split("-- hybrid")[1]
    assert "Runbook: Fleet Telemetry Ingestion" in hybrid
    assert "dense#" in hybrid
    assert "lexical#" in hybrid


def test_long_snippets_are_truncated(tmp_path: Path, settings: Settings) -> None:
    (tmp_path / "long.txt").write_text("needle " + "word " * 200)
    code, out = _run(["demo", "--corpus", str(tmp_path), "-k", "1", "needle"], settings)
    assert code == 0
    assert "…" in out


def test_sample_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from rag_assistant import cli

    monkeypatch.setenv("RAG_SAMPLE_DIR", str(tmp_path))
    assert cli._default_sample_dir() == tmp_path


def test_demo_adds_rerank_row_when_reranker_selected(settings: Settings) -> None:
    code, out = _run(["--reranker", "term-overlap", "demo", "-k", "1", "NWR-4031"], settings)
    assert code == 0
    assert out.count("-- ") == 4
    reranked = out.split("-- hybrid + rerank (term-overlap)")[1]
    assert "Runbook: Fleet Telemetry Ingestion" in reranked
    assert "rerank#1" in reranked


def test_demo_plain_rows_are_not_reranked(settings: Settings) -> None:
    code, out = _run(["--reranker", "term-overlap", "demo", "-k", "1", "NWR-4031"], settings)
    assert code == 0
    plain = out.split("-- hybrid + rerank")[0]
    assert "rerank#" not in plain


def test_search_reranks_by_default_when_configured(settings: Settings) -> None:
    configured = settings.model_copy(update={"reranker": "term-overlap"})
    code, out = _run(["search", "anything"], configured)
    assert code == 0  # empty store, but the rerank path runs without error
    assert "(no results)" in out


def test_search_rerank_without_reranker_exits_2(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(["search", "anything", "--rerank"], settings)
    assert code == 2
    assert "needs a reranker" in capsys.readouterr().err


def test_reranker_flag_none_overrides_settings(settings: Settings) -> None:
    configured = settings.model_copy(update={"reranker": "term-overlap"})
    code, out = _run(["--reranker", "none", "demo", "-k", "1", "NWR-4031"], configured)
    assert code == 0
    assert "+ rerank" not in out


def test_ask_streams_answer_and_marks_cited_sources(settings: Settings) -> None:
    code, out = _run(
        ["ask", "how many holiday days do employees get", "-k", "3", "--ingest", str(SAMPLE_DIR)],
        settings,
    )
    assert code == 0
    answer, _, sources = out.rpartition("Sources (")
    assert "[1]" in answer  # the fake LLM cites the sources it quotes
    assert sources.startswith("fake")
    assert " *[1] " in sources
    assert sources.count("[") == 3


def test_ask_with_empty_index_says_it_does_not_know(settings: Settings) -> None:
    code, out = _run(["ask", "what is the capital of France"], settings)
    assert code == 0
    assert "Sources" not in out
    assert out.strip()  # the fixed "don't know" reply, with no citations


def test_ask_missing_ingest_path_exits_2(tmp_path: Path, settings: Settings) -> None:
    code, _ = _run(["ask", "anything", "--ingest", str(tmp_path / "nope")], settings)
    assert code == 2

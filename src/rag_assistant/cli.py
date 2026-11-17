"""Command-line interface: ingest files, search, and a side-by-side retrieval demo.

Examples::

    rag-cli ingest data/sample            # index a folder (PDF / Markdown / text)
    rag-cli search "NWR-4031" --mode lexical
    rag-cli demo                          # ingest the sample corpus and compare modes
    rag-cli --reranker term-overlap demo  # ...and add a reranked row per query
    rag-cli ask "how many holiday days do employees get" --ingest data/sample

The CLI uses the same :class:`KnowledgeBase` as the API, so with
``RAG_STORE_BACKEND=pgvector`` documents ingested here are served by the API too.
With the default in-memory store each command starts empty, which is what ``demo`` is for.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TextIO, cast

from rag_assistant.api.app import build_knowledge_base
from rag_assistant.config import RerankerProvider, Settings, get_settings
from rag_assistant.domain import SearchHit
from rag_assistant.generation import extract_citations
from rag_assistant.ingestion import DocumentParseError, UnsupportedFormatError
from rag_assistant.service import KnowledgeBase, RerankerNotConfiguredError, SearchMode

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


def _default_sample_dir() -> Path:
    """Repo checkout (editable install) first, then ./data/sample (Docker image)."""
    if env := os.environ.get("RAG_SAMPLE_DIR"):
        return Path(env)
    repo = Path(__file__).resolve().parents[2] / "data" / "sample"
    return repo if repo.is_dir() else Path("data/sample")


SAMPLE_DIR = _default_sample_dir()
DEMO_QUERIES = (
    "NWR-4031",
    "how many holiday days do employees get",
    "what database stores robot telemetry",
)


def iter_files(paths: Sequence[Path]) -> Iterator[Path]:
    """Expand directories recursively; keep only supported, non-README files, sorted."""
    for path in paths:
        if path.is_dir():
            yield from sorted(
                p
                for p in path.rglob("*")
                if p.is_file()
                and p.suffix.lower() in SUPPORTED_SUFFIXES
                and p.stem.lower() != "readme"
            )
        elif path.is_file():
            yield path
        else:
            raise FileNotFoundError(path)


def ingest_paths(kb: KnowledgeBase, paths: Sequence[Path], out: TextIO) -> int:
    """Ingest files, printing one line each. Returns the number of failures."""
    failures = 0
    for path in iter_files(paths):
        try:
            result = kb.ingest(path.read_bytes(), path.name)
        except (UnsupportedFormatError, DocumentParseError) as exc:
            failures += 1
            print(f"  FAILED {path}: {exc}", file=out)
            continue
        verb = "indexed" if result.created else "skipped (duplicate)"
        doc = result.document
        print(
            f"  OK {path.name}: {verb}, {doc.chunk_count} chunks, {result.elapsed_ms} ms", file=out
        )
    return failures


def print_hits(hits: Sequence[SearchHit], out: TextIO, width: int = 90) -> None:
    if not hits:
        print("  (no results)", file=out)
    for i, hit in enumerate(hits, start=1):
        ranks = ", ".join(f"{name}#{rank}" for name, rank in sorted(hit.ranks.items()))
        snippet = " ".join(hit.chunk.text.split())
        if len(snippet) > width:
            snippet = snippet[: width - 1] + "…"
        print(f"  {i}. [{hit.score:.4f}] {hit.document_title} ({ranks})", file=out)
        print(f"     {snippet}", file=out)


def print_answer(
    kb: KnowledgeBase, question: str, out: TextIO, *, k: int, mode: SearchMode
) -> list[int]:
    """Print the answer as it streams, then a numbered source list.

    Cited sources are marked with ``*`` so it is easy to see which retrieved
    chunks the model actually used. Returns the valid citation numbers.
    """
    stream = kb.stream_answer(question, k=k, mode=mode)
    parts: list[str] = []
    for delta in stream.deltas:
        parts.append(delta)
        out.write(delta)
        out.flush()
    print(file=out)
    citations = extract_citations("".join(parts), len(stream.sources))
    if stream.sources:
        print(f"\nSources ({stream.llm}):", file=out)
    for n, hit in enumerate(stream.sources, start=1):
        marker = "*" if n in citations else " "
        print(f" {marker}[{n}] {hit.document_title}", file=out)
    return citations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-cli", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reranker",
        choices=["none", "term-overlap", "cross-encoder"],
        help="override RAG_RERANKER for this command",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index files or folders")
    ingest.add_argument("paths", nargs="+", type=Path)

    search = sub.add_parser("search", help="Search the indexed documents")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--mode", choices=["hybrid", "dense", "lexical"], default="hybrid")
    search.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force reranking on/off (default: on when a reranker is configured)",
    )

    ask = sub.add_parser("ask", help="Answer a question from the indexed documents")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=5)
    ask.add_argument("--mode", choices=["hybrid", "dense", "lexical"], default="hybrid")
    ask.add_argument(
        "--ingest",
        type=Path,
        nargs="+",
        default=[],
        metavar="PATH",
        help="index these files/folders first (handy with the in-memory store)",
    )

    demo = sub.add_parser("demo", help="Ingest the sample corpus and compare search modes")
    demo.add_argument("--corpus", type=Path, default=SAMPLE_DIR)
    demo.add_argument("-k", type=int, default=3)
    demo.add_argument("queries", nargs="*", default=list(DEMO_QUERIES))
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    out: TextIO | None = None,
) -> int:
    out = out or sys.stdout
    args = _build_parser().parse_args(argv)
    settings = settings or get_settings()
    if args.reranker is not None:
        provider = cast(RerankerProvider, args.reranker)
        settings = settings.model_copy(update={"reranker": provider})
    kb = build_knowledge_base(settings)

    try:
        if args.command == "ingest":
            print(f"Ingesting into {kb.store.__class__.__name__}:", file=out)
            return 1 if ingest_paths(kb, args.paths, out) else 0

        if args.command == "search":
            mode = cast(SearchMode, args.mode)
            print_hits(kb.search(args.query, k=args.k, mode=mode, rerank=args.rerank), out)
            return 0

        if args.command == "ask":
            failures = ingest_paths(kb, args.ingest, out) if args.ingest else 0
            if args.ingest:
                print(file=out)
            mode = cast(SearchMode, args.mode)
            print_answer(kb, args.question, out, k=args.k, mode=mode)
            return 1 if failures else 0

        print(f"Ingesting sample corpus from {args.corpus}:", file=out)
        failures = ingest_paths(kb, [args.corpus], out)
        for query in args.queries:
            print(f'\n=== "{query}" ===', file=out)
            for mode in ("dense", "lexical", "hybrid"):
                print(f"-- {mode}", file=out)
                print_hits(kb.search(query, k=args.k, mode=mode, rerank=False), out)
            if kb.reranker is not None:
                print(f"-- hybrid + rerank ({kb.reranker.name})", file=out)
                print_hits(kb.search(query, k=args.k, mode="hybrid", rerank=True), out)
        return 1 if failures else 0
    except FileNotFoundError as exc:
        print(f"error: no such file or directory: {exc}", file=sys.stderr)
        return 2
    except RerankerNotConfiguredError:
        print("error: --rerank needs a reranker (set RAG_RERANKER or --reranker)", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

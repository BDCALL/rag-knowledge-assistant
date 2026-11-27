# RAG Knowledge Assistant

[![CI](https://github.com/BDCALL/rag-knowledge-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/BDCALL/rag-knowledge-assistant/actions/workflows/ci.yml)

Upload documents (PDF, Markdown, text), index them into Postgres + pgvector, and ask questions that are answered with citations.

> Work in progress: Day 3 of 5. Backend core, hybrid retrieval, reranking and cited answers (blocking and SSE-streamed) are done. Next: a Next.js chat UI (Day 4), and an evaluation harness plus full docs (Day 5).

## What works today

- **Ingestion**: PDF (page-aware), Markdown (front matter stripped, title from first heading) and plain text, with duplicate detection by content hash.
- **Chunking**: structure-aware recursive splitter (pages → paragraphs → lines → sentences → words) with overlap and character offsets for precise citations.
- **Embeddings**: pluggable providers behind one protocol: an offline deterministic hashing embedder (default, used in CI), sentence-transformers, or OpenAI.
- **Vector store**: Postgres + pgvector with an HNSW cosine index, or an in-memory NumPy store for development. One shared contract test suite runs against both.
- **Hybrid retrieval**: an incremental Okapi BM25 index (identifier-aware tokenizer, so `NWR-4031` or `pg_vector` match exactly) fused with dense vector search via Reciprocal Rank Fusion. `POST /search` takes `mode: hybrid | dense | lexical` and every hit reports its rank in each retriever, so results are explainable.
- **Reranking (retrieve-then-rerank)**: set `RAG_RERANKER=cross-encoder` to rescore the top `RAG_RERANK_CANDIDATES` hybrid candidates with a sentence-transformers cross-encoder, or `term-overlap` for a model-free offline fallback. Both implement one `Reranker` protocol; `POST /search` accepts `rerank: true | false` per query and each hit gains a `ranks.rerank` position next to its retriever ranks.
- **Answers with citations**: `POST /ask` retrieves the top-`k` chunks, shows them to the model as numbered sources (`[1] Leave policy (p. 2)`), and returns the answer, the cited source numbers and every source with a `cited` flag. Citations outside the given sources are dropped as hallucinated, and with no sources the model is skipped and the reply is "I don't know". `RAG_LLM_PROVIDER=openai` uses OpenAI Chat Completions; the default `fake` provider is an offline, extractive stand-in so tests and CI need no key.
- **Streaming (SSE)**: `POST /ask/stream` sends one `sources` event (so a UI can render citations straight away), then `token` events as the model writes, then `done` with the final citations. Retrieval runs before the response starts, so bad requests still get a normal 4xx; a model failure mid-stream becomes an in-band `error` event.
- **API**: FastAPI with `POST /documents`, `GET /documents`, `GET/DELETE /documents/{id}`, `POST /search`, `POST /ask`, `POST /ask/stream`, `GET /health`, and OpenAPI docs at `/docs`.
- **Observability**: JSON logs, one access-log line per request, and an `X-Request-ID` header propagated through every log line via a `ContextVar`.
- **CLI**: `rag-cli ingest <paths>`, `rag-cli search "<query>" --mode lexical`, and `rag-cli demo`, which indexes a small sample corpus and prints dense vs lexical vs hybrid results side by side. Add `--reranker term-overlap` (or `cross-encoder`) before the command to get an extra "hybrid + rerank" row in the demo; `search` also takes `--rerank / --no-rerank` to force the second stage on or off for one query. `rag-cli ask "<question>" --ingest data/sample` streams a grounded answer to the terminal and then lists the retrieved sources, marking the ones the answer cited with `*`. `rag-cli eval -k 3` indexes the sample corpus, runs the labelled queries in `data/eval/` and prints recall@k and MRR for dense, lexical and hybrid search.

## Quickstart

```bash
cp .env.example .env
docker compose up --build        # API on http://localhost:8000, Postgres + pgvector on 5432

curl -F file=@notes.md localhost:8000/documents
curl localhost:8000/search -H 'content-type: application/json' -d '{"query": "How are embeddings stored?", "mode": "hybrid"}'
curl localhost:8000/ask -H 'content-type: application/json' -d '{"question": "How are embeddings stored?"}'
curl -N localhost:8000/ask/stream -H 'content-type: application/json' -d '{"question": "How are embeddings stored?"}'

docker compose run --rm api rag-cli demo    # compare retrieval modes on the sample corpus
docker compose run --rm api rag-cli --reranker term-overlap demo   # ...plus a reranked row
docker compose run --rm api rag-cli ask "how many holiday days do employees get" --ingest data/sample
```

## Development

```bash
pip install -e ".[dev]"
pytest                                   # unit + API tests (in-memory store)
TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag_test pytest   # + pgvector
ruff check . && ruff format --check . && mypy
```

### Frontend (Next.js + TypeScript)

The chat UI lives in `frontend/` and talks to the API on `http://localhost:8000` (CORS allows `http://localhost:3000`).

```bash
cd frontend
cp .env.example .env.local   # NEXT_PUBLIC_API_URL
npm install
npm run dev                  # http://localhost:3000
npm run lint && npm run typecheck && npm test && npm run build
```

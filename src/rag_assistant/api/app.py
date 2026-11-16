"""FastAPI application: document upload, search and cited question answering."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from rag_assistant import __version__
from rag_assistant.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentList,
    DocumentOut,
    HealthOut,
    IngestOut,
    SearchHitOut,
    SearchRequest,
    SearchResponse,
    sources_out,
)
from rag_assistant.config import Settings, get_settings
from rag_assistant.embeddings import build_embedder
from rag_assistant.generation import build_llm, extract_citations
from rag_assistant.ingestion import DocumentParseError, RecursiveChunker, UnsupportedFormatError
from rag_assistant.observability import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    configure_logging,
)
from rag_assistant.retrieval import build_reranker
from rag_assistant.service import AnswerStream, KnowledgeBase, RerankerNotConfiguredError
from rag_assistant.store import build_store

logger = logging.getLogger(__name__)


def build_knowledge_base(settings: Settings) -> KnowledgeBase:
    embedder = build_embedder(settings)
    store = build_store(settings, dim=embedder.dim)
    chunker = RecursiveChunker(settings.chunk_size, settings.chunk_overlap)
    return KnowledgeBase(
        embedder,
        store,
        chunker,
        reranker=build_reranker(settings),
        rerank_candidates=settings.rerank_candidates,
        llm=build_llm(settings),
    )


def create_app(settings: Settings | None = None, kb: KnowledgeBase | None = None) -> FastAPI:
    """App factory. Tests inject ``settings`` and/or a prebuilt ``kb``."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.kb = kb or build_knowledge_base(settings)
        logger.info(
            "Started with store=%s embedder=%s",
            settings.store_backend,
            app.state.kb.embedder.name,
        )
        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
        description="Upload documents, search them and ask questions answered with citations.",
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )
    # Added last so it is the outermost layer and wraps CORS + error handlers too.
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(UnsupportedFormatError)
    async def _unsupported(_: Request, exc: UnsupportedFormatError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

    @app.exception_handler(DocumentParseError)
    async def _unparseable(_: Request, exc: DocumentParseError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status.HTTP_422_UNPROCESSABLE_CONTENT)

    @app.exception_handler(RerankerNotConfiguredError)
    async def _no_reranker(_: Request, exc: RerankerNotConfiguredError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status.HTTP_400_BAD_REQUEST)

    _register_routes(app, settings)
    return app


def get_kb(request: Request) -> KnowledgeBase:
    kb: KnowledgeBase = request.app.state.kb
    return kb


KB = Annotated[KnowledgeBase, Depends(get_kb)]


def _register_routes(app: FastAPI, settings: Settings) -> None:
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)

    @app.get("/health", response_model=HealthOut, tags=["meta"])
    def health(kb: KB) -> HealthOut:
        ok = kb.store.ping()
        if not ok:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Vector store unreachable")
        return HealthOut(
            status="ok",
            version=__version__,
            store=settings.store_backend,
            embedder=kb.embedder.name,
            reranker=kb.reranker.name if kb.reranker else None,
            llm=kb.llm.name,
            documents=len(kb.store.list_documents()),
            chunks=kb.store.count_chunks(),
        )

    @app.post(
        "/documents",
        response_model=IngestOut,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
        responses={200: {"description": "Duplicate: document already indexed"}},
    )
    async def upload_document(
        kb: KB, file: Annotated[UploadFile, File(description="PDF, Markdown or text")]
    ) -> JSONResponse:
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"File exceeds {settings.max_upload_mb:g} MB limit",
            )
        filename = file.filename or "upload.txt"
        # Parsing + embedding are CPU-bound; keep the event loop free.
        result = await run_in_threadpool(kb.ingest, data, filename, file.content_type)
        body = IngestOut(
            document=DocumentOut.from_domain(result.document),
            created=result.created,
            elapsed_ms=result.elapsed_ms,
        )
        code = status.HTTP_201_CREATED if result.created else status.HTTP_200_OK
        return JSONResponse(body.model_dump(mode="json"), status_code=code)

    @app.get("/documents", response_model=DocumentList, tags=["documents"])
    def list_documents(kb: KB) -> DocumentList:
        docs = [DocumentOut.from_domain(d) for d in kb.store.list_documents()]
        return DocumentList(documents=docs, total=len(docs))

    @app.get("/documents/{document_id}", response_model=DocumentOut, tags=["documents"])
    def get_document(document_id: str, kb: KB) -> DocumentOut:
        doc = kb.store.get(document_id)
        if doc is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
        return DocumentOut.from_domain(doc)

    @app.delete(
        "/documents/{document_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["documents"],
    )
    def delete_document(document_id: str, kb: KB) -> None:
        if not kb.delete(document_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    @app.post("/search", response_model=SearchResponse, tags=["search"])
    def search(body: SearchRequest, kb: KB) -> SearchResponse:
        hits = kb.search(
            body.query, k=body.k, document_ids=body.document_ids, mode=body.mode, rerank=body.rerank
        )
        reranked = kb.reranker is not None and body.rerank is not False
        return SearchResponse(
            query=body.query,
            mode=body.mode,
            reranker=kb.reranker.name if reranked and kb.reranker else None,
            hits=[SearchHitOut.from_domain(h) for h in hits],
        )

    @app.post("/ask", response_model=AskResponse, tags=["answers"])
    def ask(body: AskRequest, kb: KB) -> AskResponse:
        answer = kb.ask(
            body.question,
            body.k,
            document_ids=body.document_ids,
            mode=body.mode,
            rerank=body.rerank,
        )
        return AskResponse.from_domain(answer)

    @app.post(
        "/ask/stream",
        tags=["answers"],
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    def ask_stream(body: AskRequest, kb: KB) -> StreamingResponse:
        """Server-Sent Events: one ``sources`` event, ``token`` events, then ``done``."""
        # Retrieval runs here, before the response starts, so a bad request still
        # gets a normal 4xx instead of an error halfway through a 200 stream.
        stream = kb.stream_answer(
            body.question,
            body.k,
            document_ids=body.document_ids,
            mode=body.mode,
            rerank=body.rerank,
        )
        return StreamingResponse(
            _sse_events(stream),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def sse(event: str, data: object) -> str:
    """Format one Server-Sent Event (JSON keeps newlines in tokens on one data line)."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _sse_events(stream: AnswerStream) -> Iterator[str]:
    sources = [s.model_dump(mode="json") for s in sources_out(stream.sources)]
    yield sse("sources", {"question": stream.question, "llm": stream.llm, "sources": sources})
    parts: list[str] = []
    try:
        for delta in stream.deltas:
            parts.append(delta)
            yield sse("token", {"text": delta})
    except Exception:
        # Headers are already sent, so report the failure in-band and stop.
        logger.exception("Answer stream failed")
        yield sse("error", {"detail": "Answer generation failed"})
        return
    citations = extract_citations("".join(parts), len(stream.sources))
    yield sse("done", {"citations": citations})


def run() -> None:  # pragma: no cover - thin CLI wrapper
    import uvicorn

    settings = get_settings()
    configure_logging(settings.log_level, settings.json_logs)
    uvicorn.run(
        "rag_assistant.api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        log_config=None,  # keep our JSON handler instead of uvicorn's default
        access_log=False,  # RequestContextMiddleware logs requests with their IDs
    )

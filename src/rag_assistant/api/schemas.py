"""HTTP request/response models (kept separate from the domain dataclasses)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints

from rag_assistant.domain import DocumentSummary, SearchHit
from rag_assistant.service import Answer, SearchMode

# Free text sent to retrieval. Whitespace is stripped *before* the length check,
# so "   " is rejected with a 422 instead of running an empty search.
QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class DocumentOut(BaseModel):
    id: str
    title: str
    source: str
    media_type: str
    content_hash: str
    chunk_count: int
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, doc: DocumentSummary) -> DocumentOut:
        return cls(
            id=doc.id,
            title=doc.title,
            source=doc.source,
            media_type=doc.media_type,
            content_hash=doc.content_hash,
            chunk_count=doc.chunk_count,
            created_at=doc.created_at,
            metadata=doc.metadata,
        )


class IngestOut(BaseModel):
    document: DocumentOut
    created: bool
    elapsed_ms: float


class DocumentList(BaseModel):
    documents: list[DocumentOut]
    total: int


class SearchRequest(BaseModel):
    query: QueryText = Field(examples=["How is data stored?"])
    k: int = Field(default=5, ge=1, le=50)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict search to these documents"
    )
    mode: SearchMode = Field(
        default="hybrid",
        description="hybrid = BM25 + embeddings fused with RRF; dense or lexical alone",
    )
    rerank: bool | None = Field(
        default=None,
        description="Rerank candidates with the configured reranker (default: if configured)",
    )


class SearchHitOut(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    text: str
    score: float
    chunk_index: int
    start_char: int
    end_char: int
    page: int | None = None
    ranks: dict[str, int] = Field(
        default_factory=dict, description="1-based rank in each retriever that found it"
    )

    @classmethod
    def from_domain(cls, hit: SearchHit) -> SearchHitOut:
        page = hit.chunk.metadata.get("page")
        return cls(
            chunk_id=hit.chunk.id,
            document_id=hit.chunk.document_id,
            document_title=hit.document_title,
            text=hit.chunk.text,
            score=round(hit.score, 6),
            chunk_index=hit.chunk.index,
            start_char=hit.chunk.start_char,
            end_char=hit.chunk.end_char,
            page=int(page) if page is not None else None,
            ranks=dict(hit.ranks),
        )


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    reranker: str | None = Field(description="Reranker used for these hits, if any")
    hits: list[SearchHitOut]


class AskRequest(BaseModel):
    question: QueryText = Field(examples=["How much annual leave?"])
    k: int = Field(default=5, ge=1, le=20, description="How many sources to give the model")
    document_ids: list[str] | None = Field(
        default=None, description="Only answer from these documents"
    )
    mode: SearchMode = "hybrid"
    rerank: bool | None = None


class SourceOut(SearchHitOut):
    number: int = Field(description="The n in citation [n]")
    cited: bool = Field(description="Whether the answer cites this source")

    @classmethod
    def numbered(cls, number: int, hit: SearchHit, cited: bool) -> SourceOut:
        return cls(**SearchHitOut.from_domain(hit).model_dump(), number=number, cited=cited)


def sources_out(hits: list[SearchHit], citations: list[int] | None = None) -> list[SourceOut]:
    cited = set(citations or [])
    return [SourceOut.numbered(n, hit, n in cited) for n, hit in enumerate(hits, start=1)]


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[int] = Field(description="Cited source numbers, in first-seen order")
    sources: list[SourceOut]
    llm: str
    elapsed_ms: float

    @classmethod
    def from_domain(cls, answer: Answer) -> AskResponse:
        return cls(
            question=answer.question,
            answer=answer.text,
            citations=answer.citations,
            sources=sources_out(answer.sources, answer.citations),
            llm=answer.llm,
            elapsed_ms=answer.elapsed_ms,
        )


class HealthOut(BaseModel):
    status: str
    version: str
    store: str
    embedder: str
    reranker: str | None
    llm: str
    documents: int
    chunks: int

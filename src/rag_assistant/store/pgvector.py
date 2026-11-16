"""Postgres + pgvector implementation of :class:`VectorStore`.

Schema
------
``documents``  one row per uploaded file (title, source, hash, metadata).
``chunks``     one row per chunk with a ``vector(dim)`` column and an HNSW index
               using ``vector_cosine_ops``; deleting a document cascades to its chunks.

Search orders by the cosine-distance operator ``<=>`` so Postgres can use the HNSW
index, and converts distance back to similarity (``1 - distance``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import Vector as PgVector
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from rag_assistant.domain import Chunk, Document, DocumentSummary, SearchHit
from rag_assistant.embeddings import Vector
from rag_assistant.store.base import validate_add

_JSON = JSON().with_variant(JSONB(), "postgresql")


def _models(dim: int) -> tuple[type[Any], type[Any]]:
    """Build ORM classes for a given embedding dimension.

    The dimension is part of the column type, so models are created per store
    (each with its own metadata) rather than at import time.
    """

    class _Base(DeclarativeBase):
        pass

    class DocumentRow(_Base):
        __tablename__ = "documents"

        id: Mapped[str] = mapped_column(String(32), primary_key=True)
        title: Mapped[str] = mapped_column(Text)
        source: Mapped[str] = mapped_column(Text)
        media_type: Mapped[str] = mapped_column(String(100))
        content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
        meta: Mapped[dict[str, Any]] = mapped_column("metadata", _JSON, default=dict)
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
        chunks: Mapped[list[ChunkRow]] = relationship(
            back_populates="document", cascade="all, delete-orphan", passive_deletes=True
        )

    class ChunkRow(_Base):
        __tablename__ = "chunks"

        id: Mapped[str] = mapped_column(String(32), primary_key=True)
        document_id: Mapped[str] = mapped_column(
            ForeignKey("documents.id", ondelete="CASCADE"), index=True
        )
        idx: Mapped[int] = mapped_column("chunk_index", Integer)
        text: Mapped[str] = mapped_column(Text)
        start_char: Mapped[int] = mapped_column(Integer)
        end_char: Mapped[int] = mapped_column(Integer)
        meta: Mapped[dict[str, Any]] = mapped_column("metadata", _JSON, default=dict)
        embedding: Mapped[Any] = mapped_column(PgVector(dim))
        document: Mapped[DocumentRow] = relationship(back_populates="chunks")

        __table_args__ = (
            Index(
                "ix_chunks_embedding_hnsw",
                "embedding",
                postgresql_using="hnsw",
                postgresql_with={"m": 16, "ef_construction": 64},
                postgresql_ops={"embedding": "vector_cosine_ops"},
            ),
        )

    return DocumentRow, ChunkRow


class PgVectorStore:
    def __init__(self, database_url: str, dim: int, *, engine: Engine | None = None) -> None:
        self.dim = dim
        self.engine = engine or create_engine(database_url, pool_pre_ping=True, future=True)
        self._Document, self._Chunk = _models(dim)
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    # -- lifecycle ------------------------------------------------------------------

    def create_schema(self) -> None:
        """Idempotently enable pgvector and create tables + indexes."""
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        self._Document.metadata.create_all(self.engine)
        self._check_dimension()

    def drop_schema(self) -> None:
        self._Document.metadata.drop_all(self.engine)

    def _check_dimension(self) -> None:
        """Fail fast if the table was created for a different embedding model."""
        query = text(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        )
        with self.engine.connect() as conn:
            existing = conn.execute(query).scalar()
        if existing is not None and existing != self.dim:
            raise RuntimeError(
                f"chunks.embedding is vector({existing}) but the embedder produces {self.dim} "
                "dimensions; re-create the schema or change RAG_EMBEDDING_DIM."
            )

    @contextmanager
    def _tx(self) -> Iterator[Session]:
        with self._session.begin() as session:
            yield session

    # -- mapping helpers ------------------------------------------------------------

    @staticmethod
    def _summary(row: Any, chunk_count: int) -> DocumentSummary:
        created = row.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return DocumentSummary(
            id=row.id,
            title=row.title,
            source=row.source,
            media_type=row.media_type,
            content_hash=row.content_hash,
            chunk_count=chunk_count,
            created_at=created,
            metadata=dict(row.meta or {}),
        )

    def _summaries(self, session: Session, where: Any = None) -> list[DocumentSummary]:
        doc_t, chunk_t = self._Document, self._Chunk
        stmt = (
            select(doc_t, func.count(chunk_t.id))
            .outerjoin(chunk_t, chunk_t.document_id == doc_t.id)
            .group_by(doc_t.id)
            .order_by(doc_t.created_at.desc())
        )
        if where is not None:
            stmt = stmt.where(where)
        return [self._summary(row, int(count)) for row, count in session.execute(stmt)]

    # -- VectorStore API ------------------------------------------------------------

    def add(self, document: Document, chunks: Sequence[Chunk], embeddings: Vector) -> None:
        validate_add(chunks, embeddings, self.dim)
        with self._tx() as session:
            session.add(
                self._Document(
                    id=document.id,
                    title=document.title,
                    source=document.source,
                    media_type=document.media_type,
                    content_hash=document.content_hash,
                    meta=dict(document.metadata),
                    created_at=document.created_at,
                )
            )
            session.add_all(
                self._Chunk(
                    id=chunk.id,
                    document_id=document.id,
                    idx=chunk.index,
                    text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    meta=dict(chunk.metadata),
                    embedding=vector.tolist(),
                )
                for chunk, vector in zip(chunks, embeddings, strict=True)
            )

    def find_by_hash(self, content_hash: str) -> DocumentSummary | None:
        with self._tx() as session:
            found = self._summaries(session, self._Document.content_hash == content_hash)
        return found[0] if found else None

    def get(self, document_id: str) -> DocumentSummary | None:
        with self._tx() as session:
            found = self._summaries(session, self._Document.id == document_id)
        return found[0] if found else None

    def list_documents(self) -> list[DocumentSummary]:
        with self._tx() as session:
            return self._summaries(session)

    def delete(self, document_id: str) -> bool:
        with self._tx() as session:
            result = session.execute(delete(self._Document).where(self._Document.id == document_id))
            return bool(getattr(result, "rowcount", 0))

    def search(
        self,
        query: Vector,
        k: int = 5,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        if k <= 0:
            return []
        q = [float(x) for x in query.reshape(-1)]
        if len(q) != self.dim:
            raise ValueError(f"Query dim {len(q)} != store dim {self.dim}")
        doc_t, chunk_t = self._Document, self._Chunk
        distance = chunk_t.embedding.cosine_distance(q).label("distance")
        stmt = select(chunk_t, doc_t.title, distance).join(doc_t, doc_t.id == chunk_t.document_id)
        if document_ids is not None:
            stmt = stmt.where(chunk_t.document_id.in_(list(document_ids)))
        stmt = stmt.order_by(distance).limit(k)
        with self._tx() as session:
            rows = session.execute(stmt).all()
        return [
            SearchHit(chunk=self._to_chunk(row), score=1.0 - float(dist), document_title=title)
            for row, title, dist in rows
        ]

    @staticmethod
    def _to_chunk(row: Any) -> Chunk:
        return Chunk(
            id=row.id,
            document_id=row.document_id,
            index=row.idx,
            text=row.text,
            start_char=row.start_char,
            end_char=row.end_char,
            metadata=dict(row.meta or {}),
        )

    def iter_chunks(self, batch_size: int = 1000) -> Iterator[tuple[Chunk, str]]:
        """Stream chunks (without embeddings) in batches to keep memory flat."""
        doc_t, chunk_t = self._Document, self._Chunk
        stmt = (
            select(
                chunk_t.id,
                chunk_t.document_id,
                chunk_t.idx,
                chunk_t.text,
                chunk_t.start_char,
                chunk_t.end_char,
                chunk_t.meta,
                doc_t.title,
            )
            .join(doc_t, doc_t.id == chunk_t.document_id)
            .order_by(chunk_t.document_id, chunk_t.idx)
            .execution_options(yield_per=batch_size)
        )
        with self._session() as session:
            for row in session.execute(stmt):
                yield self._to_chunk(row), row.title

    def count_chunks(self) -> int:
        with self._tx() as session:
            return int(session.scalar(select(func.count(self._Chunk.id))) or 0)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            return False
        return True

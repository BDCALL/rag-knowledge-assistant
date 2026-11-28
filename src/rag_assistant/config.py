"""Application settings, loaded from environment variables (prefix ``RAG_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["hashing", "sentence-transformers", "openai"]
StoreBackend = Literal["memory", "pgvector"]
RerankerProvider = Literal["none", "term-overlap", "cross-encoder"]
LLMProvider = Literal["fake", "openai"]


class Settings(BaseSettings):
    """All runtime configuration in one typed object.

    Defaults are chosen so the app runs with zero setup: an in-memory store and a
    deterministic hashing embedder that needs no model download and no API key.
    """

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    app_name: str = "RAG Knowledge Assistant"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    json_logs: bool = True

    # Storage
    store_backend: StoreBackend = "memory"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"

    # Embeddings
    embedding_provider: EmbeddingProvider = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(default=384, gt=0, le=4096)
    openai_api_key: str | None = None

    # Reranking (second stage over the hybrid candidates; "none" disables it)
    reranker: RerankerProvider = "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=20, ge=1, le=200)

    # Answer generation ("fake" is an offline, extractive stand-in for a real model)
    llm_provider: LLMProvider = "fake"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # Chunking
    chunk_size: int = Field(default=800, gt=50, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=120, ge=0)

    # Auth: when set, clients must send this value in the X-API-Key header
    api_key: str | None = None

    # Uploads
    max_upload_mb: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("RAG_OPENAI_API_KEY is required when embedding_provider=openai")
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("RAG_OPENAI_API_KEY is required when llm_provider=openai")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance (FastAPI dependency)."""
    return Settings()

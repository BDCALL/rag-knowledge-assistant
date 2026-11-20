// Response shapes of the FastAPI backend, mirrored by hand from
// src/rag_assistant/api/schemas.py. Keep the two in sync when the API changes.

export interface DocumentOut {
  id: string;
  title: string;
  source: string;
  media_type: string;
  content_hash: string;
  chunk_count: number;
  created_at: string; // ISO 8601 timestamp
  metadata: Record<string, unknown>;
}

export interface DocumentList {
  documents: DocumentOut[];
  total: number;
}

export interface IngestOut {
  document: DocumentOut;
  // false when an identical file (same content hash) was already ingested
  created: boolean;
  elapsed_ms: number;
}

export interface HealthOut {
  status: string;
  version: string;
  store: string;
  embedder: string;
  reranker: string | null;
  llm: string;
  documents: number;
  chunks: number;
}

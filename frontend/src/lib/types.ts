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

export type SearchMode = "hybrid" | "dense" | "lexical";

export interface AskRequest {
  question: string;
  k?: number; // 1-20, how many sources the model sees
  document_ids?: string[];
  mode?: SearchMode;
  rerank?: boolean;
}

export interface SourceOut {
  number: number; // the n in citation [n]
  cited: boolean;
  chunk_id: string;
  document_id: string;
  document_title: string;
  text: string;
  score: number;
  chunk_index: number;
  start_char: number;
  end_char: number;
  page: number | null;
  ranks: Record<string, number>;
}

export interface AskResponse {
  question: string;
  answer: string;
  citations: number[]; // cited source numbers, in first-seen order
  sources: SourceOut[];
  llm: string;
  elapsed_ms: number;
}

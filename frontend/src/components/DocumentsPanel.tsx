"use client";

import { useCallback, useEffect, useState } from "react";

import { api, ApiError, type ApiClient } from "@/lib/api";
import type { DocumentOut } from "@/lib/types";

import { DocumentList } from "./DocumentList";
import { UploadForm } from "./UploadForm";

type DocumentsApi = Pick<ApiClient, "listDocuments" | "uploadDocument" | "deleteDocument">;

interface DocumentsPanelProps {
  /** Defaults to the shared `api` (class instances cannot be passed from a Server Component). */
  client?: DocumentsApi;
}

/** Readable message for a failed list/delete call. */
export function panelErrorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : "Could not reach the API. Is it running?";
}

/** Owns the document list state: loads it on mount and reloads after every upload or delete. */
export function DocumentsPanel({ client = api }: DocumentsPanelProps) {
  const [documents, setDocuments] = useState<DocumentOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const { documents } = await client.listDocuments();
      setDocuments(documents);
      setError(null);
    } catch (err) {
      setError(panelErrorMessage(err));
    }
  }, [client]);

  useEffect(() => void refresh(), [refresh]);

  async function handleDelete(id: string) {
    try {
      await client.deleteDocument(id);
      await refresh();
    } catch (err) {
      setError(panelErrorMessage(err));
    }
  }

  return (
    <section aria-label="Documents">
      <h2>Documents</h2>
      <UploadForm upload={(file) => client.uploadDocument(file)} onUploaded={() => void refresh()} />
      {error && <p role="alert">{error}</p>}
      {documents === null ? (
        <p className="muted">Loading documents...</p>
      ) : (
        <DocumentList documents={documents} onDelete={(id) => void handleDelete(id)} />
      )}
    </section>
  );
}

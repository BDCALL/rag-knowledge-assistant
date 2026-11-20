"use client";

import { useState, type FormEvent } from "react";

import { ApiError } from "@/lib/api";
import type { IngestOut } from "@/lib/types";

interface UploadFormProps {
  /** Usually `api.uploadDocument`; injected so the form has no hard dependency on the client. */
  upload: (file: File) => Promise<IngestOut>;
  onUploaded?: (result: IngestOut) => void;
}

/** Turns an ingest result into the status line shown under the form. */
export function uploadMessage({ document, created }: IngestOut): string {
  if (!created) return `"${document.title}" was already uploaded.`;
  const chunks = document.chunk_count === 1 ? "1 chunk" : `${document.chunk_count} chunks`;
  return `Added "${document.title}" (${chunks}).`;
}

/** Backend errors carry a readable `detail`; anything else (e.g. network down) gets a generic hint. */
export function uploadErrorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : "Upload failed. Is the API running?";
}

export function UploadForm({ upload, onUploaded }: UploadFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    try {
      const result = await upload(file);
      setStatus(uploadMessage(result));
      onUploaded?.(result);
    } catch (error) {
      setStatus(uploadErrorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="upload-form" onSubmit={handleSubmit}>
      <input
        type="file"
        aria-label="Document to upload"
        accept=".pdf,.md,.markdown,.txt"
        onChange={(event) => setFile(event.target.files?.[0] ?? null)}
      />
      <button type="submit" disabled={!file || busy}>
        {busy ? "Uploading..." : "Upload"}
      </button>
      {status && <p role="status">{status}</p>}
    </form>
  );
}

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { ApiError } from "@/lib/api";
import type { IngestOut } from "@/lib/types";

import { UploadForm, uploadErrorMessage, uploadMessage } from "./UploadForm";

const ingest = (created: boolean): IngestOut => ({
  document: {
    id: "doc-1",
    title: "Leave policy",
    source: "leave-policy.md",
    media_type: "text/markdown",
    content_hash: "abc",
    chunk_count: 3,
    created_at: "2026-11-20T10:00:00Z",
    metadata: {},
  },
  created,
  elapsed_ms: 12,
});

describe("UploadForm", () => {
  it("starts with the submit button disabled until a file is picked", () => {
    const html = renderToStaticMarkup(<UploadForm upload={async () => ingest(true)} />);
    expect(html).toContain('accept=".pdf,.md,.markdown,.txt"');
    expect(html).toMatch(/<button type="submit" disabled="">Upload<\/button>/);
  });

  it("describes new and duplicate uploads differently", () => {
    expect(uploadMessage(ingest(true))).toBe('Added "Leave policy" (3 chunks).');
    expect(uploadMessage(ingest(false))).toBe('"Leave policy" was already uploaded.');
  });

  it("shows the backend detail for API errors and a hint otherwise", () => {
    expect(uploadErrorMessage(new ApiError(415, "Unsupported file type"))).toBe("Unsupported file type");
    expect(uploadErrorMessage(new TypeError("fetch failed"))).toBe("Upload failed. Is the API running?");
  });
});

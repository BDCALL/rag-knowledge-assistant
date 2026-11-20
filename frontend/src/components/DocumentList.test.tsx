import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { DocumentOut } from "@/lib/types";

import { DocumentList } from "./DocumentList";

function makeDoc(overrides: Partial<DocumentOut> = {}): DocumentOut {
  return {
    id: "doc-1",
    title: "Leave policy",
    source: "leave-policy.md",
    media_type: "text/markdown",
    content_hash: "abc",
    chunk_count: 3,
    created_at: "2026-11-20T10:00:00Z",
    metadata: {},
    ...overrides,
  };
}

describe("DocumentList", () => {
  it("shows an empty-state hint when there are no documents", () => {
    const html = renderToStaticMarkup(<DocumentList documents={[]} />);
    expect(html).toContain("No documents yet");
    expect(html).not.toContain("<ul");
  });

  it("renders each title with a singular or plural chunk count", () => {
    const html = renderToStaticMarkup(
      <DocumentList documents={[makeDoc(), makeDoc({ id: "doc-2", title: "Runbook", chunk_count: 1 })]} />,
    );
    expect(html).toContain("Leave policy");
    expect(html).toContain("3 chunks");
    expect(html).toContain("1 chunk<");
  });

  it("only renders delete buttons when a handler is given", () => {
    expect(renderToStaticMarkup(<DocumentList documents={[makeDoc()]} />)).not.toContain("<button");
    const html = renderToStaticMarkup(<DocumentList documents={[makeDoc()]} onDelete={() => {}} />);
    expect(html).toContain('aria-label="Delete Leave policy"');
  });
});

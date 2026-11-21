import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";

import { DocumentsPanel, panelErrorMessage } from "./DocumentsPanel";

function stubClient() {
  return {
    listDocuments: vi.fn().mockResolvedValue({ documents: [], total: 0 }),
    uploadDocument: vi.fn(),
    deleteDocument: vi.fn(),
  };
}

describe("DocumentsPanel", () => {
  it("renders the upload form and a loading line before the list arrives", () => {
    const client = stubClient();
    const html = renderToStaticMarkup(<DocumentsPanel client={client} />);
    expect(html).toContain("<h2>Documents</h2>");
    expect(html).toContain('aria-label="Document to upload"');
    expect(html).toContain("Loading documents...");
    // Effects do not run during server rendering, so nothing is fetched yet.
    expect(client.listDocuments).not.toHaveBeenCalled();
  });
});

describe("panelErrorMessage", () => {
  it("uses the backend detail for API errors and a hint otherwise", () => {
    expect(panelErrorMessage(new ApiError(404, "Document not found"))).toBe("Document not found");
    expect(panelErrorMessage(new TypeError("fetch failed"))).toContain("Is it running?");
  });
});

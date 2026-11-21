import { describe, expect, it, vi } from "vitest";

import { ApiClient, ApiError } from "./api";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function clientWith(response: Response) {
  const fetchStub = vi.fn<typeof fetch>().mockResolvedValue(response);
  return { client: new ApiClient("http://api.test/", fetchStub), fetchStub };
}

describe("ApiClient", () => {
  it("lists documents from the base URL without a double slash", async () => {
    const { client, fetchStub } = clientWith(jsonResponse({ documents: [], total: 0 }));

    await expect(client.listDocuments()).resolves.toEqual({ documents: [], total: 0 });
    expect(fetchStub).toHaveBeenCalledWith("http://api.test/documents", undefined);
  });

  it("uploads a file as multipart form data under the 'file' field", async () => {
    const { client, fetchStub } = clientWith(
      jsonResponse({ document: { id: "d1" }, created: true, elapsed_ms: 3 }, 201),
    );
    const file = new File(["hello"], "notes.md", { type: "text/markdown" });

    const result = await client.uploadDocument(file);

    expect(result.created).toBe(true);
    const [url, init] = fetchStub.mock.calls[0]!;
    expect(url).toBe("http://api.test/documents");
    expect(init?.method).toBe("POST");
    const body = init?.body as FormData;
    expect((body.get("file") as File).name).toBe("notes.md");
  });

  it("posts a question to /ask as JSON", async () => {
    const answer = { question: "Leave?", answer: "25 days [1].", citations: [1], sources: [] };
    const { client, fetchStub } = clientWith(jsonResponse(answer));

    await expect(client.ask({ question: "Leave?", k: 3 })).resolves.toMatchObject({
      citations: [1],
    });
    const [url, init] = fetchStub.mock.calls[0]!;
    expect(url).toBe("http://api.test/ask");
    expect(init?.method).toBe("POST");
    expect(new Headers(init?.headers).get("Content-Type")).toBe("application/json");
    expect(JSON.parse(init?.body as string)).toEqual({ question: "Leave?", k: 3 });
  });

  it("returns undefined for a 204 delete and encodes the id", async () => {
    const { client, fetchStub } = clientWith(new Response(null, { status: 204 }));

    await expect(client.deleteDocument("a/b")).resolves.toBeUndefined();
    expect(fetchStub.mock.calls[0]![0]).toBe("http://api.test/documents/a%2Fb");
  });

  it("surfaces the backend's string detail as an ApiError", async () => {
    const { client } = clientWith(jsonResponse({ detail: "Unsupported file type" }, 415));

    const error = await client.uploadDocument(new File(["x"], "x.exe")).catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 415, message: "Unsupported file type" });
  });

  it("uses the first message of a validation error list", async () => {
    const { client } = clientWith(jsonResponse({ detail: [{ msg: "Field required" }] }, 422));

    await expect(client.health()).rejects.toMatchObject({ status: 422, message: "Field required" });
  });

  it("falls back to a generic message when the error body is not JSON", async () => {
    const { client } = clientWith(new Response("<html>Bad gateway</html>", { status: 502 }));

    await expect(client.listDocuments()).rejects.toMatchObject({
      status: 502,
      message: "Request failed with status 502",
    });
  });
});

import type { AskRequest, AskResponse, DocumentList, HealthOut, IngestOut } from "./types";

export const DEFAULT_API_URL = "http://localhost:8000";

/** Raised for any non-2xx response, carrying the backend's `detail` message when present. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type Fetch = typeof fetch;

/**
 * Thin typed wrapper around the backend's REST endpoints.
 *
 * `fetchImpl` is injectable so tests can pass a stub instead of hitting the network.
 */
export class ApiClient {
  private readonly baseUrl: string;

  constructor(
    baseUrl: string = DEFAULT_API_URL,
    private readonly fetchImpl: Fetch = (...args) => fetch(...args),
  ) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  health(): Promise<HealthOut> {
    return this.request<HealthOut>("/health");
  }

  listDocuments(): Promise<DocumentList> {
    return this.request<DocumentList>("/documents");
  }

  uploadDocument(file: File): Promise<IngestOut> {
    const body = new FormData();
    body.append("file", file);
    // No Content-Type header: the browser sets the multipart boundary itself.
    return this.request<IngestOut>("/documents", { method: "POST", body });
  }

  async deleteDocument(id: string): Promise<void> {
    await this.request<void>(`/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
  }

  ask(request: AskRequest): Promise<AskResponse> {
    return this.request<AskResponse>("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, init);
    if (!response.ok) {
      throw new ApiError(response.status, await errorMessage(response));
    }
    if (response.status === 204) {
      return undefined as T;
    }
    return (await response.json()) as T;
  }
}

/** FastAPI errors look like {"detail": "..."} or, for validation errors, {"detail": [...]}. */
async function errorMessage(response: Response): Promise<string> {
  const fallback = `Request failed with status ${response.status}`;
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const { detail } = body as { detail: unknown };
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail) && detail.length > 0) {
        const first = detail[0] as { msg?: unknown };
        if (typeof first.msg === "string") return first.msg;
      }
    }
  } catch {
    // Body was not JSON (e.g. a proxy error page); use the generic message.
  }
  return fallback;
}

export const api = new ApiClient(process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL);

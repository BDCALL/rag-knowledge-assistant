"use client";

import { useState, type FormEvent } from "react";

import { api, ApiError, type ApiClient } from "@/lib/api";
import type { AskResponse } from "@/lib/types";

import { AnswerView } from "./AnswerView";

interface ChatPanelProps {
  /** Defaults to the shared `api` (class instances cannot be passed from a Server Component). */
  client?: Pick<ApiClient, "ask">;
}

/** Readable message for a failed /ask call. */
export function askErrorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : "Could not get an answer. Is the API running?";
}

/** Question box that calls POST /ask and shows the latest answer with its sources. */
export function ChatPanel({ client = api }: ChatPanelProps) {
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;
    setBusy(true);
    setError(null);
    try {
      setResponse(await client.ask({ question: trimmed }));
    } catch (err) {
      setError(askErrorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section aria-label="Ask">
      <h2>Ask a question</h2>
      <form onSubmit={(event) => void handleSubmit(event)}>
        <input
          type="text"
          aria-label="Question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
        />
        <button type="submit" disabled={busy || !question.trim()}>
          {busy ? "Thinking..." : "Ask"}
        </button>
      </form>
      {error && <p role="alert">{error}</p>}
      {response && <AnswerView response={response} />}
    </section>
  );
}

import type { AskStreamEvent } from "./sse";
import type { AskResponse } from "./types";

/**
 * Folds one /ask/stream event into the answer shown so far.
 *
 * `sources` starts a fresh answer, each `token` appends text and `done` marks
 * the cited sources. `error` leaves the answer as it is; the caller shows the
 * detail. Streams do not report timing, so `elapsed_ms` stays 0.
 */
export function applyStreamEvent(
  current: AskResponse | null,
  event: AskStreamEvent,
): AskResponse | null {
  switch (event.event) {
    case "sources":
      return { ...event.data, answer: "", citations: [], elapsed_ms: 0 };
    case "token":
      return current && { ...current, answer: current.answer + event.data.text };
    case "done": {
      if (!current) return current;
      const cited = new Set(event.data.citations);
      return {
        ...current,
        citations: event.data.citations,
        sources: current.sources.map((source) => ({ ...source, cited: cited.has(source.number) })),
      };
    }
    case "error":
      return current;
  }
}

/**
 * Reads a whole /ask/stream, passing each partial answer to `onUpdate`.
 *
 * Returns the error detail if the stream ends with an `error` event, else null.
 * Failures before the stream starts (e.g. a 422) are thrown by the client.
 */
export async function readAnswerStream(
  events: AsyncIterable<AskStreamEvent>,
  onUpdate: (response: AskResponse | null) => void,
): Promise<string | null> {
  let current: AskResponse | null = null;
  for await (const event of events) {
    current = applyStreamEvent(current, event);
    onUpdate(current);
    if (event.event === "error") return event.data.detail;
  }
  return null;
}

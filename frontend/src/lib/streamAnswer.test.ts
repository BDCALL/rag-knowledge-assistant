import { describe, expect, it } from "vitest";

import type { AskStreamEvent } from "./sse";
import { applyStreamEvent } from "./streamAnswer";
import type { AskResponse, SourceOut } from "./types";

function source(number: number): SourceOut {
  return {
    number,
    cited: false,
    chunk_id: `c${number}`,
    document_id: "d1",
    document_title: "Handbook",
    text: `chunk ${number}`,
    score: 1,
    chunk_index: number,
    start_char: 0,
    end_char: 7,
    page: null,
    ranks: {},
  };
}

describe("applyStreamEvent", () => {
  it("builds an answer from sources, tokens and done", () => {
    const events: AskStreamEvent[] = [
      { event: "sources", data: { question: "Q?", llm: "fake", sources: [source(1), source(2)] } },
      { event: "token", data: { text: "Yes " } },
      { event: "token", data: { text: "[2]." } },
      { event: "done", data: { citations: [2] } },
    ];
    const result = events.reduce<AskResponse | null>(applyStreamEvent, null);
    expect(result?.answer).toBe("Yes [2].");
    expect(result?.citations).toEqual([2]);
    expect(result?.sources.map((s) => s.cited)).toEqual([false, true]);
  });

  it("ignores tokens before sources and keeps the answer on error", () => {
    expect(applyStreamEvent(null, { event: "token", data: { text: "x" } })).toBeNull();
    const partial = applyStreamEvent(null, {
      event: "sources",
      data: { question: "Q?", llm: "fake", sources: [] },
    });
    expect(applyStreamEvent(partial, { event: "error", data: { detail: "boom" } })).toBe(partial);
  });
});

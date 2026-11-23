import { describe, expect, it } from "vitest";

import { parseSseChunk } from "./sse";

describe("parseSseChunk", () => {
  it("parses complete events and keeps a partial one as rest", () => {
    const text =
      'event: token\ndata: {"text": "Hel"}\n\n' +
      'event: token\ndata: {"text": "lo\\n"}\n\n' +
      'event: done\ndata: {"cita';
    const { events, rest } = parseSseChunk(text);
    expect(events).toEqual([
      { event: "token", data: { text: "Hel" } },
      { event: "token", data: { text: "lo\n" } },
    ]);
    expect(rest).toBe('event: done\ndata: {"cita');

    const next = parseSseChunk(rest + 'tions": [1, 2]}\n\n');
    expect(next.events).toEqual([{ event: "done", data: { citations: [1, 2] } }]);
    expect(next.rest).toBe("");
  });

  it("handles CRLF line endings and skips comment-only blocks", () => {
    const text = ": keep-alive\r\n\r\nevent: error\r\ndata: {\"detail\": \"boom\"}\r\n\r\n";
    expect(parseSseChunk(text).events).toEqual([{ event: "error", data: { detail: "boom" } }]);
  });
});

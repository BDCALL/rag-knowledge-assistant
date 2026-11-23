import type { SourceOut } from "./types";

/** Events sent by POST /ask/stream, in order: sources, token*, then done or error. */
export type AskStreamEvent =
  | { event: "sources"; data: { question: string; llm: string; sources: SourceOut[] } }
  | { event: "token"; data: { text: string } }
  | { event: "done"; data: { citations: number[] } }
  | { event: "error"; data: { detail: string } };

/**
 * Splits buffered SSE text into complete events.
 *
 * Network chunks can end mid-event, so anything after the last blank line is
 * returned as `rest` for the caller to prepend to the next chunk.
 */
export function parseSseChunk(buffer: string): { events: AskStreamEvent[]; rest: string } {
  const blocks = buffer.replace(/\r\n/g, "\n").split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: AskStreamEvent[] = [];
  for (const block of blocks) {
    let event = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (dataLines.length === 0) continue; // comments or keep-alives
    events.push({ event, data: JSON.parse(dataLines.join("\n")) } as AskStreamEvent);
  }
  return { events, rest };
}

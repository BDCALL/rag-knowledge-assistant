import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { AskResponse, SourceOut } from "@/lib/types";

import { AnswerView, sourceLabel } from "./AnswerView";

const source = (number: number, cited: boolean, page: number | null): SourceOut => ({
  number,
  cited,
  chunk_id: `chunk-${number}`,
  document_id: "doc-1",
  document_title: "Leave policy",
  text: `Source text ${number}`,
  score: 0.5,
  chunk_index: number,
  start_char: 0,
  end_char: 10,
  page,
  ranks: {},
});

const response = (sources: SourceOut[]): AskResponse => ({
  question: "How many leave days?",
  answer: "Staff get 25 days [1].",
  citations: [1],
  sources,
  llm: "fake",
  elapsed_ms: 5,
});

describe("AnswerView", () => {
  it("labels sources with their number and page when known", () => {
    expect(sourceLabel(source(2, false, 4))).toBe("[2] Leave policy, p. 4");
    expect(sourceLabel(source(3, false, null))).toBe("[3] Leave policy");
  });

  it("renders the answer and marks only cited sources", () => {
    const html = renderToStaticMarkup(<AnswerView response={response([source(1, true, 1), source(2, false, null)])} />);
    expect(html).toContain("Staff get 25 days [1].");
    expect(html).toContain('<li class="source cited">');
    expect(html).toContain('<li class="source">');
    expect(html.match(/\(cited\)/g)).toHaveLength(1);
  });

  it("omits the source list when nothing was retrieved", () => {
    const html = renderToStaticMarkup(<AnswerView response={response([])} />);
    expect(html).not.toContain("<ol");
  });
});

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";

import { askErrorMessage, ChatPanel } from "./ChatPanel";

describe("ChatPanel", () => {
  it("renders an empty question box with the Ask button disabled", () => {
    const client = { askStream: vi.fn() };
    const html = renderToStaticMarkup(<ChatPanel client={client} />);
    expect(html).toContain("<h2>Ask a question</h2>");
    expect(html).toContain('aria-label="Question"');
    expect(html).toMatch(/<button type="submit" disabled="">Ask<\/button>/);
    expect(client.askStream).not.toHaveBeenCalled();
  });
});

describe("askErrorMessage", () => {
  it("uses the backend detail for API errors and a hint otherwise", () => {
    expect(askErrorMessage(new ApiError(422, "Question must not be blank"))).toBe(
      "Question must not be blank",
    );
    expect(askErrorMessage(new TypeError("fetch failed"))).toContain("Is the API running?");
  });
});

import type { AskResponse, SourceOut } from "@/lib/types";

/** Label shown before each source, e.g. "[2] Leave policy, p. 4". */
export function sourceLabel({ number, document_title, page }: SourceOut): string {
  const pageLabel = page === null ? "" : `, p. ${page}`;
  return `[${number}] ${document_title}${pageLabel}`;
}

export type AnswerPart = { kind: "text"; text: string } | { kind: "citation"; number: number };

/** Split answer text into plain text and `[n]` citation markers whose number is a known source. */
export function splitCitations(answer: string, sourceNumbers: ReadonlySet<number>): AnswerPart[] {
  const parts: AnswerPart[] = [];
  let last = 0;
  for (const match of answer.matchAll(/\[(\d+)\]/g)) {
    const number = Number(match[1]);
    if (!sourceNumbers.has(number)) continue;
    if (match.index > last) parts.push({ kind: "text", text: answer.slice(last, match.index) });
    parts.push({ kind: "citation", number });
    last = match.index + match[0].length;
  }
  if (last < answer.length) parts.push({ kind: "text", text: answer.slice(last) });
  return parts;
}

/** Presentational view of an /ask response: the answer, then its numbered sources. */
export function AnswerView({ response }: { response: AskResponse }) {
  const numbers = new Set(response.sources.map((source) => source.number));
  return (
    <section className="answer" aria-label="Answer">
      <p className="answer-text">
        {splitCitations(response.answer, numbers).map((part, i) =>
          part.kind === "text" ? (
            part.text
          ) : (
            <a key={i} href={`#source-${part.number}`} className="citation">
              [{part.number}]
            </a>
          ),
        )}
      </p>
      {response.sources.length > 0 && (
        <ol className="source-list">
          {response.sources.map((source) => (
            <li key={source.chunk_id} id={`source-${source.number}`} className={source.cited ? "source cited" : "source"}>
              <strong>{sourceLabel(source)}</strong>
              {source.cited && <span className="muted"> (cited)</span>}
              <p className="source-text">{source.text}</p>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

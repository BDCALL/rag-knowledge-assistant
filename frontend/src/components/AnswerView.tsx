import type { AskResponse, SourceOut } from "@/lib/types";

/** Label shown before each source, e.g. "[2] Leave policy, p. 4". */
export function sourceLabel({ number, document_title, page }: SourceOut): string {
  const pageLabel = page === null ? "" : `, p. ${page}`;
  return `[${number}] ${document_title}${pageLabel}`;
}

/** Presentational view of an /ask response: the answer, then its numbered sources. */
export function AnswerView({ response }: { response: AskResponse }) {
  return (
    <section className="answer" aria-label="Answer">
      <p className="answer-text">{response.answer}</p>
      {response.sources.length > 0 && (
        <ol className="source-list">
          {response.sources.map((source) => (
            <li key={source.chunk_id} className={source.cited ? "source cited" : "source"}>
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

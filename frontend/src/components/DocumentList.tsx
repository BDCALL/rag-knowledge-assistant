import type { DocumentOut } from "@/lib/types";

interface DocumentListProps {
  documents: DocumentOut[];
  onDelete?: (id: string) => void;
}

/** Presentational list of ingested documents; data loading lives in the parent. */
export function DocumentList({ documents, onDelete }: DocumentListProps) {
  if (documents.length === 0) {
    return <p className="muted">No documents yet. Upload a PDF, Markdown or text file to start.</p>;
  }
  return (
    <ul className="document-list">
      {documents.map((doc) => (
        <li key={doc.id}>
          <span className="document-title">{doc.title}</span>
          <span className="muted">
            {" "}
            {doc.chunk_count} {doc.chunk_count === 1 ? "chunk" : "chunks"}
          </span>
          {onDelete && (
            <button type="button" aria-label={`Delete ${doc.title}`} onClick={() => onDelete(doc.id)}>
              Delete
            </button>
          )}
        </li>
      ))}
    </ul>
  );
}

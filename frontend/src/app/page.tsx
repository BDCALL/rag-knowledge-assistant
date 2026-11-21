import { DocumentsPanel } from "@/components/DocumentsPanel";

export default function HomePage() {
  return (
    <main className="container">
      <h1>RAG Knowledge Assistant</h1>
      <p className="muted">
        Upload documents, then ask questions answered with numbered citations to the sources.
      </p>
      <DocumentsPanel />
    </main>
  );
}

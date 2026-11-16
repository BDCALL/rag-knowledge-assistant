from __future__ import annotations

import pytest

from rag_assistant.domain import Document


def _build_pdf(pages: list[str], title: str = "Test Guide") -> bytes:
    """Hand-roll a minimal valid PDF with one line of Helvetica text per page.

    Avoids a test-only dependency on reportlab while still exercising real PDF parsing.
    """
    objects: list[bytes] = []
    n_pages = len(pages)
    font_id = 3 + 2 * n_pages
    info_id = font_id + 1
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(f"<< /Title ({title}) >>".encode())

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {info_id} 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


@pytest.fixture
def sample_pdf() -> bytes:
    return _build_pdf(
        [
            "Postgres with the pgvector extension stores embeddings.",
            "Cosine distance ranks the nearest chunks for a query.",
        ]
    )


@pytest.fixture
def long_document() -> Document:
    paragraphs = [
        f"Paragraph {i}. " + " ".join(f"word{i}_{j}." for j in range(40)) for i in range(12)
    ]
    return Document(
        title="Long", text="\n\n".join(paragraphs), source="long.txt", media_type="text/plain"
    )

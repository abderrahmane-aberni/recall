"""Turns an uploaded file into a Document row plus embedded Chunk rows.

Supports PDF (one "page" per source page, via pypdf) and plain text/markdown
(treated as a single page) - covers both lecture-slide PDFs and typed-up
course notes.
"""
from __future__ import annotations

import io

from sqlalchemy.orm import Session

from .chunking import chunk_pages
from .config import Settings
from .db import Chunk, Document
from .embeddings import EmbeddingBackend


class UnsupportedFileType(Exception):
    pass


class EmptyDocument(Exception):
    """Raised when extraction produced no usable text at all (e.g. a
    scanned/image-only PDF with no OCR)."""


def extract_pages(filename: str, data: bytes) -> list[str]:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf_pages(data)
    if lower.endswith(".txt") or lower.endswith(".md"):
        return [data.decode("utf-8", errors="replace")]
    raise UnsupportedFileType(
        f"Unsupported file type for {filename!r}. Supported: .pdf, .txt, .md"
    )


def _extract_pdf_pages(data: bytes) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return pages


def ingest_document(
    db: Session,
    *,
    filename: str,
    title: str,
    data: bytes,
    settings: Settings,
    embedding_backend: EmbeddingBackend,
) -> Document:
    pages = extract_pages(filename, data)
    records = chunk_pages(
        pages,
        chunk_size_chars=settings.chunk_size_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
    )
    if not records:
        raise EmptyDocument(
            f"No extractable text found in {filename!r}. If this is a scanned "
            "PDF (images of pages, no selectable text), it needs OCR before "
            "it can be used here - that isn't supported yet."
        )

    texts = [r.text for r in records]
    vectors = embedding_backend.embed(texts, task="document")

    document = Document(
        filename=filename,
        title=title,
        embedding_backend=embedding_backend.name,
        num_chunks=len(records),
    )
    db.add(document)
    db.flush()  # assigns document.id without committing yet

    for record, vector in zip(records, vectors):
        db.add(
            Chunk(
                document_id=document.id,
                chunk_index=record.index,
                text=record.text,
                source_label=record.source_label,
                embedding=vector,
            )
        )

    db.commit()
    db.refresh(document)
    return document

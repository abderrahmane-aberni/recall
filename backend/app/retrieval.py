"""Top-k similarity search over stored chunks via pgvector."""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Chunk, Document
from .embeddings import EmbeddingBackend


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    document_title: str
    source_label: str
    text: str
    distance: float  # cosine distance: 0 = identical, 2 = opposite


def retrieve(
    db: Session,
    query: str,
    embedding_backend: EmbeddingBackend,
    *,
    top_k: int = 5,
    document_ids: list[int] | None = None,
) -> list[RetrievedChunk]:
    [query_vector] = embedding_backend.embed([query], task="query")
    distance_col = Chunk.embedding.cosine_distance(query_vector).label("distance")

    stmt = (
        select(Chunk, Document, distance_col)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance_col)
        .limit(top_k)
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))

    results = db.execute(stmt).all()

    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=document.id,
            document_title=document.title,
            source_label=chunk.source_label,
            text=chunk.text,
            distance=float(distance) if distance is not None else float("nan"),
        )
        for chunk, document, distance in results
    ]

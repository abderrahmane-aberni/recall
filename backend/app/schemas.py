"""Pydantic models: API request/response bodies, plus the schema-constrained
shape we force Gemini's chat output into.

Citation safety design: we never ask the model to recall or re-type a page
number - it only has to say *which* of the chunks we handed it (by 1-indexed
position) it actually used. The server already knows exactly what document,
page and text each of those positions maps to, so citations attached to the
final answer are always accurate, even if the model hallucinates everywhere
else.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---- Documents ----


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    title: str
    embedding_backend: str
    num_chunks: int
    created_at: datetime


class IngestResponse(BaseModel):
    document: DocumentOut
    message: str


# ---- Chat ----


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    document_ids: list[int] | None = None  # None = search across all documents


class Citation(BaseModel):
    marker: int  # the [n] number shown inline in the answer
    document_id: int
    document_title: str
    source_label: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunk_count: int


# ---- LLM-constrained structured output (internal, not exposed over the API) ----


class RetrievalAnswer(BaseModel):
    """What we force Gemini's response into. `used_chunk_indices` are
    1-indexed positions into the list of retrieved chunks we gave it in the
    prompt - never a page number or anything else the model has to recall
    correctly on its own."""

    answer: str
    used_chunk_indices: list[int] = Field(default_factory=list)
    # True when none of the retrieved chunks actually answer the question -
    # lets the UI show "not found in these documents" instead of a confident
    # guess dressed up with citations that don't support it.
    answerable_from_context: bool

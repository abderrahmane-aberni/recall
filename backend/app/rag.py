"""Grounded answer generation: builds a prompt from retrieved chunks, asks
Gemini for a schema-constrained answer, and attaches citations the server
itself verified (see schemas.py for why used_chunk_indices rather than
model-authored page numbers)."""
from __future__ import annotations

from .retrieval import RetrievedChunk
from .schemas import Citation, RetrievalAnswer

SYSTEM_INSTRUCTION = """You are a study assistant. Answer the user's question using ONLY the
numbered context passages provided below - do not use outside knowledge, even if you're
confident it's correct. Each passage is numbered [1], [2], etc.

When you use information from a passage, cite it inline in your answer with its number in
square brackets, e.g. "Mitosis has five phases [2]." You may cite more than one passage for a
single sentence, e.g. "[1][3]".

Set answerable_from_context to false (and give a brief, honest answer explaining what's
missing) if the passages don't actually contain the answer - never fill the gap with outside
knowledge or a guess. List in used_chunk_indices the numbers of every passage you actually
relied on, in the order you first cited them. If you didn't use any passage, leave it empty.
"""


def _build_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[{i}] (from \"{chunk.document_title}\", {chunk.source_label})\n{chunk.text}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "(no passages retrieved)"
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"


def build_citations(
    parsed: RetrievalAnswer, retrieved: list[RetrievedChunk]
) -> list[Citation]:
    """Pure mapping from the model's declared chunk-index usage back to the
    server's own retrieval metadata. Separated out from generate_answer so it
    can be unit-tested without calling the live Gemini API."""
    citations: list[Citation] = []
    seen: set[int] = set()
    for marker in parsed.used_chunk_indices:
        if marker in seen:
            continue
        if marker < 1 or marker > len(retrieved):
            # Model referenced a passage number that doesn't exist - skip it
            # rather than attaching a bogus citation. The inline [n] marker
            # stays in the answer text as-is; it just won't resolve to a
            # citation card in the UI.
            continue
        seen.add(marker)
        chunk = retrieved[marker - 1]
        citations.append(
            Citation(
                marker=marker,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                source_label=chunk.source_label,
                snippet=chunk.text[:300],
            )
        )
    return citations


def generate_answer(
    *,
    api_key: str,
    model_name: str,
    question: str,
    retrieved: list[RetrievedChunk],
) -> tuple[str, list[Citation]]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(question, retrieved)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=RetrievalAnswer,
        ),
    )
    parsed: RetrievalAnswer = response.parsed
    citations = build_citations(parsed, retrieved)
    return parsed.answer, citations

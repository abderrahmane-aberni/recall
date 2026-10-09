"""Turns extracted page text into overlapping chunks, each tagged with the
page number(s) it came from so later citations can point back to a real
source location instead of a chunk index.

Pure logic, no I/O - fully unit-testable without a live Gemini key or a
database connection.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkRecord:
    index: int
    text: str
    source_label: str  # e.g. "page 3" or "pages 3-4"


def _page_label(start_page: int, end_page: int) -> str:
    if start_page == end_page:
        return f"page {start_page}"
    return f"pages {start_page}-{end_page}"


def chunk_pages(
    pages: list[str],
    chunk_size_chars: int = 1200,
    chunk_overlap_chars: int = 200,
) -> list[ChunkRecord]:
    """Split a list of per-page text into overlapping chunks.

    `pages` is 1 string per source page, in order (page numbers are implied by
    list position, starting at 1). Chunking works on the concatenated text so
    a chunk can span a page boundary; the resulting source_label reflects
    whichever page(s) each chunk actually overlaps.
    """
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be positive")

    # Build the full text plus a parallel array mapping each character offset
    # to its 1-indexed source page.
    full_text_parts: list[str] = []
    offset_to_page: list[int] = []
    for page_num, page_text in enumerate(pages, start=1):
        full_text_parts.append(page_text)
        offset_to_page.extend([page_num] * len(page_text))
        # separator between pages, attributed to the page it follows
        full_text_parts.append("\n\n")
        offset_to_page.extend([page_num] * 2)

    full_text = "".join(full_text_parts)
    total_len = len(full_text)
    if total_len == 0:
        return []

    step = chunk_size_chars - chunk_overlap_chars
    chunks: list[ChunkRecord] = []
    start = 0
    index = 0
    while start < total_len:
        end = min(start + chunk_size_chars, total_len)
        raw = full_text[start:end]
        stripped = raw.strip()
        if stripped:
            start_page = offset_to_page[start]
            end_page = offset_to_page[end - 1]
            chunks.append(
                ChunkRecord(
                    index=index,
                    text=stripped,
                    source_label=_page_label(start_page, end_page),
                )
            )
            index += 1
        if end == total_len:
            break
        start += step

    return chunks

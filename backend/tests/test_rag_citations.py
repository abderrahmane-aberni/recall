from app.rag import build_citations
from app.retrieval import RetrievedChunk
from app.schemas import RetrievalAnswer


def _chunk(chunk_id: int, doc_id: int = 1, title: str = "Doc", label: str = "page 1", text: str = "text"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        document_title=title,
        source_label=label,
        text=text,
        distance=0.1,
    )


def test_maps_used_indices_to_correct_chunks():
    retrieved = [
        _chunk(1, label="page 1", text="first passage"),
        _chunk(2, label="page 2", text="second passage"),
        _chunk(3, label="page 3", text="third passage"),
    ]
    parsed = RetrievalAnswer(
        answer="Uses [1] and [3].",
        used_chunk_indices=[1, 3],
        answerable_from_context=True,
    )
    citations = build_citations(parsed, retrieved)
    assert [c.marker for c in citations] == [1, 3]
    assert citations[0].source_label == "page 1"
    assert citations[1].source_label == "page 3"


def test_out_of_range_index_is_dropped_not_crashed():
    retrieved = [_chunk(1)]
    parsed = RetrievalAnswer(
        answer="hallucinated a citation",
        used_chunk_indices=[1, 5, 0, -1],
        answerable_from_context=True,
    )
    citations = build_citations(parsed, retrieved)
    assert [c.marker for c in citations] == [1]


def test_duplicate_indices_are_deduplicated():
    retrieved = [_chunk(1), _chunk(2)]
    parsed = RetrievalAnswer(
        answer="repeats itself",
        used_chunk_indices=[1, 1, 2, 1],
        answerable_from_context=True,
    )
    citations = build_citations(parsed, retrieved)
    assert [c.marker for c in citations] == [1, 2]


def test_empty_used_indices_gives_no_citations():
    retrieved = [_chunk(1)]
    parsed = RetrievalAnswer(
        answer="not found in these documents",
        used_chunk_indices=[],
        answerable_from_context=False,
    )
    assert build_citations(parsed, retrieved) == []


def test_snippet_is_truncated_to_300_chars():
    long_text = "a" * 1000
    retrieved = [_chunk(1, text=long_text)]
    parsed = RetrievalAnswer(answer="x", used_chunk_indices=[1], answerable_from_context=True)
    citations = build_citations(parsed, retrieved)
    assert len(citations[0].snippet) == 300

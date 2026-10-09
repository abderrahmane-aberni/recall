from app.chunking import chunk_pages


def test_empty_pages_returns_no_chunks():
    assert chunk_pages([]) == []


def test_single_short_page_is_one_chunk():
    chunks = chunk_pages(["hello world"], chunk_size_chars=1200, chunk_overlap_chars=200)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].source_label == "page 1"
    assert chunks[0].index == 0


def test_long_single_page_produces_overlapping_chunks():
    text = "x" * 3000
    chunks = chunk_pages([text], chunk_size_chars=1000, chunk_overlap_chars=200)
    # step = 800, total_len = 3000 -> starts at 0, 800, 1600, 2400 -> 4 chunks
    assert len(chunks) == 4
    assert all(c.source_label == "page 1" for c in chunks)
    assert [c.index for c in chunks] == [0, 1, 2, 3]


def test_chunk_spanning_page_boundary_gets_range_label():
    # page 1 is short, page 2 is long enough that a chunk starting near the
    # end of page 1 extends into page 2.
    page1 = "A" * 100
    page2 = "B" * 2000
    chunks = chunk_pages([page1, page2], chunk_size_chars=500, chunk_overlap_chars=100)
    # the very first chunk starts at offset 0 (page 1) and extends to offset
    # 500, which is inside page 2 (page1 + separator is 102 chars)
    assert chunks[0].source_label == "pages 1-2"
    # a later chunk fully inside page 2 should just say "page 2"
    assert any(c.source_label == "page 2" for c in chunks)


def test_whitespace_only_page_produces_no_chunk():
    chunks = chunk_pages(["   \n\n  "], chunk_size_chars=1200, chunk_overlap_chars=200)
    assert chunks == []


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_pages(["hello"], chunk_size_chars=100, chunk_overlap_chars=100)


def test_chunk_size_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        chunk_pages(["hello"], chunk_size_chars=0, chunk_overlap_chars=0)


def test_consecutive_chunks_actually_overlap():
    # distinct characters at every position so overlap can be checked exactly
    import string

    alphabet = string.ascii_letters + string.digits  # 62 distinct chars
    text = "".join(alphabet[i % len(alphabet)] for i in range(2000))
    chunk_size, overlap = 500, 100
    step = chunk_size - overlap
    chunks = chunk_pages([text], chunk_size_chars=chunk_size, chunk_overlap_chars=overlap)

    for i, (a, b) in enumerate(zip(chunks, chunks[1:])):
        start_a = i * step
        start_b = (i + 1) * step
        expected_overlap = text[start_b : start_a + chunk_size]
        # the tail of chunk a and the head of chunk b should both contain
        # exactly this shared slice of the source text
        assert a.text.endswith(expected_overlap)
        assert b.text.startswith(expected_overlap)

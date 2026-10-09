import pytest

from app.ingest import UnsupportedFileType, extract_pages


def test_txt_file_is_single_page():
    pages = extract_pages("notes.txt", b"hello world")
    assert pages == ["hello world"]


def test_md_file_is_single_page():
    pages = extract_pages("notes.md", "# Title\n\nbody".encode("utf-8"))
    assert pages == ["# Title\n\nbody"]


def test_unsupported_extension_raises():
    with pytest.raises(UnsupportedFileType):
        extract_pages("notes.docx", b"irrelevant")


def test_pdf_extraction_returns_one_page_per_pdf_page():
    # build a tiny real PDF in-memory with reportlab (already a project
    # dependency path via the pdf skill) rather than depending on a fixture
    # file, so this test is self-contained.
    import io

    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 700, "Page one text")
    c.showPage()
    c.drawString(100, 700, "Page two text")
    c.showPage()
    c.save()

    pages = extract_pages("doc.pdf", buf.getvalue())
    assert len(pages) == 2
    assert "Page one text" in pages[0]
    assert "Page two text" in pages[1]

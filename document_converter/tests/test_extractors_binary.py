"""Extractor tests that need a real .docx / .pdf on disk, built on the fly."""
from document_converter.config import OCRConfig
from document_converter.extractors.docx_extractor import DocxExtractor
from document_converter.extractors.pdf_extractor import PdfExtractor
from document_converter.model import HeadingBlock, ParagraphBlock, ListBlock, TableBlock

OCR_OFF = OCRConfig(mode="off")


def _make_docx(path):
    from docx import Document as DocxDocument
    d = DocxDocument()
    d.add_heading("Doc Title", level=1)
    d.add_paragraph("An intro paragraph with some words.")
    d.add_heading("Section One", level=2)
    d.add_paragraph("bullet a", style="List Bullet")
    d.add_paragraph("bullet b", style="List Bullet")
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "h1"
    table.cell(0, 1).text = "h2"
    table.cell(1, 0).text = "v1"
    table.cell(1, 1).text = "v2"
    d.save(str(path))


def test_docx_extractor_produces_headings_lists_and_tables(tmp_path):
    p = tmp_path / "doc.docx"
    _make_docx(p)
    doc, log = DocxExtractor().extract(p, OCR_OFF)
    kinds = {type(b).__name__ for b in doc.blocks}
    assert "HeadingBlock" in kinds
    assert "ListBlock" in kinds
    assert "TableBlock" in kinds
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert headings[0].text == "Doc Title" and headings[0].level == 1
    lists = [b for b in doc.blocks if isinstance(b, ListBlock)]
    assert len(lists[0].items) == 2


def _make_pdf(path, with_body_text=True):
    import fitz
    d = fitz.open()
    page = d.new_page()
    page.insert_text((72, 72), "Document Title", fontsize=22)
    if with_body_text:
        page.insert_text((72, 130), "This is a body paragraph with normal sized text.", fontsize=10)
    d.save(str(path))
    d.close()


def test_pdf_extractor_extracts_text(tmp_path):
    p = tmp_path / "doc.pdf"
    _make_pdf(p)
    doc, log = PdfExtractor().extract(p, OCR_OFF)
    assert doc.blocks
    all_text = " ".join(getattr(b, "text", "") or getattr(b, "code", "") for b in doc.blocks)
    assert "body paragraph" in all_text.lower() or "document title" in all_text.lower()


def test_pdf_extractor_flags_scanned_page_when_ocr_off(tmp_path):
    import fitz
    d = fitz.open()
    d.new_page()  # blank page, looks scanned
    p = tmp_path / "blank.pdf"
    d.save(str(p))
    d.close()
    doc, log = PdfExtractor().extract(p, OCRConfig(mode="off"))
    assert any(f.code in ("page_looks_scanned_ocr_disabled", "empty_pdf_extraction") for f in log.findings)

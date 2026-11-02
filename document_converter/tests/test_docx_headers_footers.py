from docx import Document as DocxDocument

from document_converter.extractors.docx_extractor import DocxExtractor
from document_converter.config import OCRConfig

OCR_OFF = OCRConfig(mode="off")


def test_header_footer_excluded_and_logged_not_leaked_into_body(tmp_path):
    d = DocxDocument()
    d.sections[0].header.paragraphs[0].text = "Confidential - Internal Use Only"
    d.sections[0].footer.paragraphs[0].text = "Page footer text"
    d.add_heading("Doc Title", level=1)
    d.add_paragraph("Body content that should remain.")
    path = tmp_path / "headers.docx"
    d.save(str(path))

    doc, log = DocxExtractor().extract(path, OCR_OFF)
    body_texts = [getattr(b, "text", None) for b in doc.blocks]
    assert not any("Confidential" in (t or "") for t in body_texts)
    assert not any("Page footer" in (t or "") for t in body_texts)
    assert any(f.code == "docx_header_footer_excluded" for f in log.findings)


def test_no_header_footer_no_exclusion_finding(tmp_path):
    d = DocxDocument()
    d.add_heading("Plain Doc", level=1)
    d.add_paragraph("Just body text, no header or footer set.")
    path = tmp_path / "plain.docx"
    d.save(str(path))

    doc, log = DocxExtractor().extract(path, OCR_OFF)
    assert not any(f.code == "docx_header_footer_excluded" for f in log.findings)

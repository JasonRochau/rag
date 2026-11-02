import shutil
import subprocess

import pytest
from docx import Document as DocxDocument

from document_converter.extractors.doc_extractor import DocExtractor
from document_converter.config import OCRConfig
from document_converter.model import HeadingBlock, ListBlock

OCR_OFF = OCRConfig(mode="off")
HAS_SOFFICE = shutil.which("soffice") is not None or shutil.which("libreoffice") is not None


@pytest.fixture
def legacy_doc_file(tmp_path):
    """Builds a .docx with python-docx, then converts it to legacy .doc with
    LibreOffice -- the same conversion path a real legacy .doc file exercises
    on the way in, just run in reverse to produce a realistic fixture without
    committing a binary file to the repo."""
    if not HAS_SOFFICE:
        pytest.skip("LibreOffice not installed")
    src = tmp_path / "source.docx"
    d = DocxDocument()
    d.add_heading("Legacy Doc Test", level=1)
    d.add_paragraph("A paragraph that will round-trip through legacy .doc format.")
    d.add_heading("Section Two", level=2)
    d.add_paragraph("bullet a", style="List Bullet")
    d.add_paragraph("bullet b", style="List Bullet")
    d.save(str(src))

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    subprocess.run([soffice, "--headless", "--norestore", "--convert-to", "doc", "--outdir", str(tmp_path), str(src)],
                    capture_output=True, timeout=60, check=True)
    doc_path = tmp_path / "source.doc"
    assert doc_path.exists()
    return doc_path


@pytest.mark.skipif(not HAS_SOFFICE, reason="LibreOffice not installed")
def test_doc_extractor_converts_and_extracts(legacy_doc_file):
    doc, log = DocExtractor().extract(legacy_doc_file, OCR_OFF)
    assert doc.metadata.source_format == "doc"
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    lists = [b for b in doc.blocks if isinstance(b, ListBlock)]
    assert len(headings) >= 2
    assert len(lists) == 1
    assert any(f.code == "doc_converted_via_libreoffice" for f in log.findings)


def test_doc_extractor_reports_clean_error_when_libreoffice_unavailable(monkeypatch, tmp_path):
    fake_doc = tmp_path / "fake.doc"
    fake_doc.write_bytes(b"not a real doc file")
    monkeypatch.setattr("shutil.which", lambda name: None)

    doc, log = DocExtractor().extract(fake_doc, OCR_OFF)
    assert doc.blocks == []
    errors = [f for f in log.findings if f.severity.value == "ERROR"]
    assert any(f.code == "doc_conversion_unavailable" for f in errors)

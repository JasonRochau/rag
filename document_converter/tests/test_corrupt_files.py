"""Corrupt/unreadable input must never crash the pipeline or claim success (spec section 3.6)."""
from document_converter.pipeline import DocumentPipeline


def test_corrupt_pdf_does_not_crash(pipeline, write_source):
    src = write_source("corrupt.pdf", b"%PDF-1.4\nnot a real pdf structure\x00\x01\x02", binary=True)
    result = pipeline.preview(str(src))
    assert result.chunking_result.chunks == []
    errors = [f for f in result.validation_report.findings.findings if f.severity.value == "ERROR"]
    assert any(f.code == "extraction_failed" for f in errors)
    assert result.validation_report.retrieval_ready is False


def test_corrupt_docx_does_not_crash(pipeline, write_source):
    src = write_source("corrupt.docx", b"not a real docx / not a zip file at all", binary=True)
    result = pipeline.preview(str(src))
    assert result.chunking_result.chunks == []
    errors = [f for f in result.validation_report.findings.findings if f.severity.value == "ERROR"]
    assert any(f.code == "extraction_failed" for f in errors)


def test_corrupt_file_process_does_not_crash_and_writes_a_report(pipeline, write_source):
    src = write_source("corrupt.pdf", b"garbage not a pdf", binary=True)
    result = pipeline.process(str(src))
    assert result.report_path
    import os
    assert os.path.exists(result.report_path)

"""
Tests for the PDF extractor's content-aware fixes: compound-block code
detection, multi-column reading order, the flat-OCR-transcript fix, and
encrypted-PDF handling (universal formatting spec sections 3.3-3.5, 9, 11).
"""
import shutil
from pathlib import Path

import pytest
import fitz

from document_converter.extractors.pdf_extractor import PdfExtractor
from document_converter.config import OCRConfig
from document_converter.model import CodeBlock, HeadingBlock, ParagraphBlock

OCR_OFF = OCRConfig(mode="off")
HAS_TESSERACT = shutil.which("tesseract") is not None


def test_content_based_code_detection_without_monospace_font(tmp_path):
    """Reproduces the real bug: a PDF that renders code in the same font as
    body text (no monospace signal) must still be recognized as code from
    its content, not left as prose that then trips the HTML-remnant check
    on '#include <iostream>'."""
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "Example", fontsize=18)
    p.insert_text((72, 110), "Intro paragraph before the code sample below here.", fontsize=11)
    code_lines = ["#include <iostream>", "using namespace std;", "int main() {", "    cout << 1;", "}"]
    y = 140
    for line in code_lines:
        p.insert_text((72, y), line, fontsize=10)  # same size/family as body text
        y += 14
    path = tmp_path / "code.pdf"
    d.save(str(path))
    d.close()

    doc, log = PdfExtractor().extract(path, OCR_OFF)
    code_blocks = [b for b in doc.blocks if isinstance(b, CodeBlock)]
    assert code_blocks, "code content was not classified as a CodeBlock"
    assert "#include <iostream>" in code_blocks[0].code
    assert "cout << 1;" in code_blocks[0].code
    assert any(f.code == "content_based_code_detection" for f in log.findings)


def test_prose_and_captions_with_bare_equals_not_misclassified_as_code(tmp_path):
    """Reproduces a real false positive: technical prose and figure
    captions routinely state a parameter value with a bare "=" ("Figure 2
    MLP results with learning rate = 0.0001", "initial value = 1 and
    learning rate = 0.01 show better convergence") -- none of that is code,
    and neither the equals sign nor the caption/prose shape it sits in
    should be enough, alone or diluted through a run, to reclassify it as
    one CodeBlock."""
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "Results", fontsize=18)
    lines = [
        "Criteria of accepting the results of each test case is",
        "how fast w converges to 0.9394 shown in Type 1 in Fig.",
        "1 with minimal cost.",
        "Figure 2 MLP results with learning rate = 0.0001",
        "Figure 3 MLP results with learning rate = 0.001",
        "Figure 4 MLP results with learning rate = 0.01",
        "From Fig. 2, 3 and 4, initial value = 1 and learning",
        "rate = 0.01 show better convergence to 0.9394 than",
        "others. When the cost of each learning rates is compared",
        "as shown in Fig. 5, epoch = 300 is enough for the",
        "convergence because more than 300 epochs does not",
        "lower the cost.",
    ]
    y = 110
    for line in lines:
        p.insert_text((72, y), line, fontsize=11)  # same size/family as body text, no monospace flag
        y += 14
    path = tmp_path / "captions.pdf"
    d.save(str(path))
    d.close()

    doc, log = PdfExtractor().extract(path, OCR_OFF)
    assert not any(isinstance(b, CodeBlock) for b in doc.blocks), [
        b.code for b in doc.blocks if isinstance(b, CodeBlock)
    ]
    assert not any(f.code == "content_based_code_detection" for f in log.findings)


def test_assignment_heavy_pseudocode_still_detected_as_code(tmp_path):
    """The bare-equals signal is downgraded to corroborating (not
    conclusive) evidence -- this must not silently stop catching genuine
    assignment-heavy pseudocode/code that has no other distinguishing
    signal (no keywords, no braces, no monospace font): a short run where
    *every* line is a bare assignment, undiluted by real prose, still
    accumulates enough density to be recognized as code."""
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "Computation", fontsize=18)
    p.insert_text((72, 100), "The following steps compute the result:", fontsize=11)
    code_lines = ["x = 5", "y = 10", "z = x + y", "w = z * 2"]
    y = 130
    for line in code_lines:
        p.insert_text((72, y), line, fontsize=10)  # same size/family as body text
        y += 14
    path = tmp_path / "assignments.pdf"
    d.save(str(path))
    d.close()

    doc, log = PdfExtractor().extract(path, OCR_OFF)
    code_blocks = [b for b in doc.blocks if isinstance(b, CodeBlock)]
    assert code_blocks, "assignment-heavy pseudocode was not classified as a CodeBlock"
    joined = "\n".join(b.code for b in code_blocks)
    for line in code_lines:
        assert line in joined


def test_isolated_semicolon_sentence_not_misclassified_as_code(tmp_path):
    """A single prose line that happens to end with a semicolon-like
    punctuation pattern should not, on its own, be reclassified as code --
    the weak-signal path requires at least two corroborating lines."""
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "Notes", fontsize=18)
    p.insert_text((72, 110), "This sentence is just prose and mentions nothing code-like at all.", fontsize=11)
    path = tmp_path / "prose.pdf"
    d.save(str(path))
    d.close()

    doc, log = PdfExtractor().extract(path, OCR_OFF)
    assert not any(isinstance(b, CodeBlock) for b in doc.blocks)


def test_multi_column_reading_order(tmp_path):
    d = fitz.open()
    p = d.new_page(width=600, height=800)
    p.insert_text((50, 60), "Two Column Page", fontsize=16)
    left_lines = ["Left column line one.", "Left column line two.", "Left column line three."]
    right_lines = ["Right column line one.", "Right column line two.", "Right column line three."]
    y = 100
    for line in left_lines:
        p.insert_text((50, y), line, fontsize=11)
        y += 20
    y = 100
    for line in right_lines:
        p.insert_text((320, y), line, fontsize=11)
        y += 20
    path = tmp_path / "columns.pdf"
    d.save(str(path))
    d.close()

    doc, log = PdfExtractor().extract(path, OCR_OFF)
    texts = [getattr(b, "text", None) for b in doc.blocks if hasattr(b, "text")]
    left_idx = [i for i, t in enumerate(texts) if t and t.startswith("Left column")]
    right_idx = [i for i, t in enumerate(texts) if t and t.startswith("Right column")]
    assert left_idx and right_idx
    assert max(left_idx) < min(right_idx), "columns were interleaved instead of read left-then-right"
    # Each column line stayed its own paragraph rather than merging across the column boundary.
    assert texts.count("Left column line three.") == 1
    assert texts.count("Right column line one.") == 1
    assert any(f.code == "multi_column_layout_detected" for f in log.findings)


def test_single_column_page_unaffected(tmp_path):
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "Title", fontsize=16)
    p.insert_text((72, 110), "Just one paragraph on an ordinary single-column page.", fontsize=11)
    path = tmp_path / "single.pdf"
    d.save(str(path))
    d.close()

    doc, log = PdfExtractor().extract(path, OCR_OFF)
    assert not any(f.code == "multi_column_layout_detected" for f in log.findings)


def test_encrypted_pdf_produces_clean_error_not_a_crash(tmp_path):
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "Secret content", fontsize=14)
    path = tmp_path / "encrypted.pdf"
    d.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="hunter2", owner_pw="ownerpw",
           permissions=int(fitz.PDF_PERM_PRINT))
    d.close()

    doc, log = PdfExtractor().extract(path, OCR_OFF)
    assert doc.blocks == []
    errors = [f for f in log.findings if f.severity.value == "ERROR"]
    assert any(f.code == "pdf_password_protected" for f in errors)


def _render_page_as_image_pdf(tmp_path, name: str, draw_fn) -> Path:
    d = fitz.open()
    p = d.new_page()
    draw_fn(p)
    pix = p.get_pixmap(dpi=150)
    img_doc = fitz.open()
    img_page = img_doc.new_page(width=pix.width, height=pix.height)
    img_page.insert_image(img_page.rect, pixmap=pix)
    path = tmp_path / name
    img_doc.save(str(path))
    img_doc.close()
    d.close()
    return path


@pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract binary not installed")
def test_ocr_reconstructs_paragraphs_and_headings_not_one_flat_block(tmp_path):
    """Regression test for the 'flat OCR transcript' failure mode (spec
    section 1): before the fix, every line OCR'd from a page collapsed into
    a single ParagraphBlock regardless of blank-line paragraph breaks or
    isolated heading-like lines in the source image."""
    def draw(p):
        p.insert_text((72, 72), "Scanned Notice", fontsize=20)
        p.insert_text((72, 140), "First paragraph about the notice with enough words to look real.", fontsize=11)
        p.insert_text((72, 200), "Second paragraph, a separate idea entirely from the first one here.", fontsize=11)

    path = _render_page_as_image_pdf(tmp_path, "scanned.pdf", draw)
    doc, log = PdfExtractor().extract(path, OCRConfig(mode="on", min_text_chars_per_page=10000))

    assert doc.metadata.ocr_used is True
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    paragraphs = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert len(headings) >= 1
    assert len(paragraphs) >= 2, "OCR text collapsed into fewer paragraphs than the source actually had"


@pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract binary not installed")
def test_ocr_page_accounting_recorded_on_metadata(tmp_path):
    def draw(p):
        p.insert_text((72, 72), "Just a scanned page", fontsize=14)

    path = _render_page_as_image_pdf(tmp_path, "scanned2.pdf", draw)
    doc, log = PdfExtractor().extract(path, OCRConfig(mode="on", min_text_chars_per_page=10000))
    assert doc.metadata.extra.get("ocr_pages") == 1
    assert doc.metadata.extra.get("native_text_pages") == 0

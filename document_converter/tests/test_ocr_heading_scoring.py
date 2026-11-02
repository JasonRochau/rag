"""
Tests for the OCR multi-signal heading scorer and typed-block routing in
pdf_extractor.py (prevention half of the OCR-heading-promotion fix): a
figure-heavy, fully-OCR'd page must not promote figure labels, diagram
annotations, or repeated labels to headings just because they're isolated,
and those rejected candidates must survive as typed, searchable blocks
instead of being silently merged into surrounding paragraph text.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import fitz

from document_converter.extractors.pdf_extractor import PdfExtractor
from document_converter.config import OCRConfig
from document_converter.model import (
    HeadingBlock, CaptionBlock, FigureLabelBlock, DiagramAnnotationBlock, ParagraphBlock,
)

HAS_TESSERACT = shutil.which("tesseract") is not None
pytestmark = pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract binary not installed")


def _render_page_as_image_pdf(tmp_path: Path, name: str, draw_fn) -> Path:
    d = fitz.open()
    p = d.new_page(width=612, height=792)
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


def _texts(doc):
    return [(type(b).__name__, (getattr(b, "text", None) or "").strip()) for b in doc.blocks]


def test_figure_labels_and_axis_annotations_not_promoted_to_headings(tmp_path):
    def draw(p):
        y = 72
        p.insert_text((72, y), "A Figure-Heavy Report", fontsize=18); y += 40
        p.insert_text((72, y), "This report walks through several diagrams and their outputs below.", fontsize=11); y += 40
        p.insert_text((72, y), "Original Image", fontsize=11); y += 40
        p.insert_text((72, y), "Input Image", fontsize=11); y += 40
        p.insert_text((72, y), "Output Image", fontsize=11); y += 40
        p.insert_text((72, y), "Y - Axis", fontsize=11); y += 40
        p.insert_text((72, y), "North Direction Edges", fontsize=11); y += 40
        p.insert_text((72, y), "Color: Red", fontsize=11); y += 40
        p.insert_text((72, y), "This concluding paragraph wraps up the report with a final summary.", fontsize=11)

    path = _render_page_as_image_pdf(tmp_path, "figures.pdf", draw)
    doc, log = PdfExtractor().extract(path, OCRConfig(mode="on", min_text_chars_per_page=10000))
    assert doc.metadata.ocr_used is True

    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    heading_texts = {h.text.strip() for h in headings}
    for label in ("Original Image", "Input Image", "Output Image", "Y - Axis",
                  "North Direction Edges", "Color: Red"):
        assert label not in heading_texts, f"{label!r} was wrongly promoted to a heading"

    kinds = {t for _, t in _texts(doc)}
    assert "Input Image" in kinds or "Output Image" in kinds or "Original Image" in kinds
    typed = [b for b in doc.blocks if isinstance(b, (FigureLabelBlock, DiagramAnnotationBlock))]
    assert typed, "figure/diagram labels must survive as typed, searchable blocks, not vanish"


def test_repeated_verbatim_label_not_promoted_to_heading(tmp_path):
    def draw(p):
        y = 72
        p.insert_text((72, y), "Comparison Report", fontsize=18); y += 40
        p.insert_text((72, y), "Below are three repeated measurements from the same experiment run.", fontsize=11); y += 40
        for _ in range(4):
            p.insert_text((72, y), "Sample Value:", fontsize=11); y += 40
        p.insert_text((72, y), "The measurements above were consistent across every trial performed.", fontsize=11)

    path = _render_page_as_image_pdf(tmp_path, "repeated.pdf", draw)
    doc, log = PdfExtractor().extract(path, OCRConfig(mode="on", min_text_chars_per_page=10000))
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert not any(h.text.strip() == "Sample Value:" for h in headings)


def test_isolated_real_title_and_numbered_sections_still_promoted(tmp_path):
    def draw(p):
        y = 72
        p.insert_text((72, y), "Report On Widget Performance", fontsize=20); y += 50
        p.insert_text((72, y), "This document summarizes the performance of the new widget across tests.", fontsize=11); y += 40
        p.insert_text((72, y), "1 Introduction", fontsize=11); y += 40
        p.insert_text((72, y), "This introduction explains the motivation for the widget performance study", fontsize=11); y += 16
        p.insert_text((72, y), "and describes the methodology used throughout the rest of this report.", fontsize=11); y += 40
        p.insert_text((72, y), "2 Results", fontsize=11); y += 40
        p.insert_text((72, y), "The results below show consistent improvement across every benchmark tested", fontsize=11); y += 16
        p.insert_text((72, y), "and confirm the hypothesis stated in the introduction above.", fontsize=11)

    path = _render_page_as_image_pdf(tmp_path, "report.pdf", draw)
    doc, log = PdfExtractor().extract(path, OCRConfig(mode="on", min_text_chars_per_page=10000))
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    heading_texts = [h.text.strip() for h in headings]
    assert any("Introduction" in t for t in heading_texts)
    assert any("Results" in t for t in heading_texts)
    assert len(headings) >= 1 and headings[0].level == 1


def test_caption_line_becomes_caption_block_not_heading_or_paragraph_merge(tmp_path):
    def draw(p):
        y = 72
        p.insert_text((72, y), "Diagram Walkthrough", fontsize=18); y += 40
        p.insert_text((72, y), "The diagram below illustrates the overall pipeline used in this system.", fontsize=11); y += 40
        p.insert_text((72, y), "Figure 1: Overall pipeline architecture", fontsize=11); y += 40
        p.insert_text((72, y), "As shown above, data flows from ingestion through to the final output stage.", fontsize=11)

    path = _render_page_as_image_pdf(tmp_path, "caption.pdf", draw)
    doc, log = PdfExtractor().extract(path, OCRConfig(mode="on", min_text_chars_per_page=10000))
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert not any("Figure 1" in h.text for h in headings)
    captions = [b for b in doc.blocks if isinstance(b, CaptionBlock)]
    assert any("Figure 1" in c.text for c in captions), "caption line should survive as a typed CaptionBlock"
    paragraphs = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert not any("Figure 1" in p_.text for p_ in paragraphs), "caption must not be merged into a paragraph"

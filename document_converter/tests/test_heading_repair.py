"""
Unit tests for the bodyless-heading repair pass (heading_repair.py): demotes
unsupported, OCR-derived headings found in bodyless chains to their typed
block, while leaving native/DOCX headings, confidently-scored OCR headings,
and legitimate parent/container headings untouched.
"""
from __future__ import annotations

from document_converter.model import (
    Document, DocumentMetadata, HeadingBlock, ParagraphBlock, FigureLabelBlock, UnknownBlock,
)
from document_converter.findings import FindingLog
from document_converter.heading_repair import repair_bodyless_heading_chains


def _doc(blocks):
    d = Document(metadata=DocumentMetadata(source_path="x.pdf", source_format="pdf"))
    d.blocks = blocks
    return d


def _heading(block_id, order, level, text, page=1, from_ocr=False, score=None):
    h = HeadingBlock(block_id=block_id, order=order, source_format="pdf", level=level,
                      text=text, normalized=text, page=page)
    if from_ocr:
        h.style["from_ocr"] = True
        if score is not None:
            h.style["ocr_heading_score"] = score
    return h


def _para(block_id, order, text, page=1):
    return ParagraphBlock(block_id=block_id, order=order, source_format="pdf", text=text,
                           normalized=text, page=page)


def test_weak_ocr_figure_label_chain_demoted():
    blocks = [
        _heading("b0001", 1, 1, "Report", from_ocr=True, score=4),
        _para("b0002", 2, "Intro paragraph with enough words to carry real body content here."),
        _heading("b0003", 3, 2, "Input Image", from_ocr=True, score=2),
        _heading("b0004", 4, 2, "Output Image", from_ocr=True, score=2),
        _para("b0005", 5, "Real content finally appears here under the next heading only now."),
        _heading("b0006", 6, 2, "Results", from_ocr=True, score=5),
        _para("b0007", 7, "Body text about the results that follows normally after this heading."),
    ]
    doc = _doc(blocks)
    log = FindingLog()
    stats = repair_bodyless_heading_chains(doc, log)

    assert stats["bodyless_heading_chains_detected"] == 1
    assert stats["bodyless_heading_chains_repaired"] == 1
    assert stats["ocr_headings_demoted"] == 2
    assert isinstance(doc.blocks[2], FigureLabelBlock)
    assert isinstance(doc.blocks[3], FigureLabelBlock)
    assert doc.blocks[2].text == "Input Image"
    assert doc.blocks[2].style.get("demoted_from_heading") is True
    # "Results" is confidently scored and not label-shaped -- must remain a heading.
    assert isinstance(doc.blocks[5], HeadingBlock)
    assert any(f.code == "bodyless_heading_demoted" for f in log.findings)


def test_native_pdf_heading_never_touched():
    """A native-PDF heading (no `from_ocr` style flag) is out of scope for
    this pass entirely, even if it sits in a bodyless chain shape."""
    blocks = [
        _heading("b0001", 1, 1, "Title"),
        _heading("b0002", 2, 2, "Input Image"),  # not from_ocr -- untouched
        _para("b0003", 3, "Some content that follows the second heading eventually here."),
    ]
    doc = _doc(blocks)
    stats = repair_bodyless_heading_chains(doc, FindingLog())
    assert stats["ocr_headings_demoted"] == 0
    assert isinstance(doc.blocks[1], HeadingBlock)


def test_confident_unsupported_heading_left_for_empty_section_validator():
    """A confidently-scored OCR heading with genuinely no body at all (not
    label-shaped either) is a real document defect, not something this pass
    should silently paper over -- spec: 'Do not blindly collapse every
    bodyless heading.'"""
    blocks = [
        _heading("b0001", 1, 1, "Title", from_ocr=True, score=5),
        _para("b0002", 2, "Intro paragraph with enough words to carry real body content here."),
        _heading("b0003", 3, 2, "Orphan Section", from_ocr=True, score=5),
    ]
    doc = _doc(blocks)
    stats = repair_bodyless_heading_chains(doc, FindingLog())
    assert stats["ocr_headings_demoted"] == 0
    assert isinstance(doc.blocks[2], HeadingBlock)


def test_legitimate_parent_container_heading_preserved():
    """A heading immediately followed by a *deeper* heading is a container
    with real child sections and must never be treated as bodyless."""
    blocks = [
        _heading("b0001", 1, 1, "Title", from_ocr=True, score=5),
        _para("b0002", 2, "Intro paragraph with enough words to carry real body content here."),
        _heading("b0003", 3, 2, "Parent Section", from_ocr=True, score=2),  # weak, but has a child
        _heading("b0004", 4, 3, "Child Section", from_ocr=True, score=5),
        _para("b0005", 5, "The child section has real body content following it right here now."),
    ]
    doc = _doc(blocks)
    stats = repair_bodyless_heading_chains(doc, FindingLog())
    assert stats["ocr_headings_demoted"] == 0
    assert isinstance(doc.blocks[2], HeadingBlock)
    assert doc.blocks[2].text == "Parent Section"


def test_trailing_bodyless_heading_at_end_of_document_demoted():
    blocks = [
        _heading("b0001", 1, 1, "Title", from_ocr=True, score=5),
        _para("b0002", 2, "Intro paragraph with enough words to carry real body content here."),
        _heading("b0003", 3, 2, "Color: Red", from_ocr=True, score=2),
    ]
    doc = _doc(blocks)
    stats = repair_bodyless_heading_chains(doc, FindingLog())
    assert stats["ocr_headings_demoted"] == 1
    assert not isinstance(doc.blocks[2], HeadingBlock)


def test_weak_unlabeled_heading_demoted_to_unknown_block():
    """Weak-scored but not matching any specific label shape -- still
    demoted (spec: unsupported headings must not remain headings), falling
    back to the generic UnknownBlock so it stays searchable."""
    blocks = [
        _heading("b0001", 1, 1, "Title", from_ocr=True, score=5),
        _para("b0002", 2, "Intro paragraph with enough words to carry real body content here."),
        _heading("b0003", 3, 2, "Some Odd Fragment", from_ocr=True, score=2),
        _heading("b0004", 4, 2, "Another Fragment", from_ocr=True, score=2),
        _para("b0005", 5, "Real content finally appears here under the next heading only now."),
    ]
    doc = _doc(blocks)
    stats = repair_bodyless_heading_chains(doc, FindingLog())
    assert stats["ocr_headings_demoted"] == 2
    assert isinstance(doc.blocks[2], UnknownBlock)
    assert isinstance(doc.blocks[3], UnknownBlock)

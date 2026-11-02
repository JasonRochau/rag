"""Structural-readiness gates for title resolution, front-matter grouping,
and paragraph continuity (spec sections 3, 5, 6, 7, 14). These construct a
`Document` directly (rather than going through a PDF) so the *validator*
itself is under test, independent of whether the PDF extractor's own
prevention logic (tested separately in test_pdf_front_matter.py) catches
the defect first."""
from document_converter.model import (
    Document, DocumentMetadata, HeadingBlock, ParagraphBlock, AuthorBlock, AffiliationBlock,
)
from document_converter.chunker import MarkdownChunker
from document_converter.config import ChunkingConfig
from document_converter.validation import validate, completion_state
from document_converter.findings import FindingLog


def _validate_doc(doc, config=None):
    config = config or ChunkingConfig()
    result = MarkdownChunker().chunk(doc, "x.pdf", config, document_id="d1")
    return validate(doc, result, config, "x.pdf", "x_normalized.md", "pdf", False, FindingLog())


def test_abstract_promoted_to_h1_is_blocking():
    doc = Document(metadata=DocumentMetadata(source_path="x.pdf", source_format="pdf"))
    doc.add(HeadingBlock(block_id="b1", order=0, source_format="pdf", level=1, text="Abstract", normalized="Abstract"))
    doc.add(ParagraphBlock(block_id="b2", order=1, source_format="pdf", text="Some abstract text here.",
                            normalized="Some abstract text here."))
    report = _validate_doc(doc)
    codes = [f.code for f in report.findings.findings]
    assert "abstract_promoted_to_h1" in codes
    assert report.stats["document_title_valid"] is False
    assert report.stats["structural_readiness"] is False
    state = completion_state(report, indexed=False)
    assert state["structural_readiness"] is False


def test_valid_title_passes_title_gate():
    doc = Document(metadata=DocumentMetadata(source_path="x.pdf", source_format="pdf"))
    doc.add(HeadingBlock(block_id="b1", order=0, source_format="pdf", level=1,
                          text="A Real Document Title", normalized="A Real Document Title"))
    doc.add(HeadingBlock(block_id="b2", order=1, source_format="pdf", level=2, text="Abstract", normalized="Abstract"))
    doc.add(ParagraphBlock(block_id="b3", order=2, source_format="pdf", text="Some abstract text here.",
                            normalized="Some abstract text here."))
    report = _validate_doc(doc)
    assert report.stats["document_title_valid"] is True
    assert report.stats["h1_count"] == 1
    assert report.stats["h1_is_first_meaningful_block"] is True


def test_front_matter_interleaved_with_body_is_blocking():
    """An author block stranded *after* real body content (not grouped
    contiguously after the title) must be caught even though this shape
    cannot occur through the PDF extractor's own grouping pass -- the
    validator has to independently back the gate for any construction
    path, not just assume upstream code always behaves."""
    doc = Document(metadata=DocumentMetadata(source_path="x.pdf", source_format="pdf"))
    doc.add(HeadingBlock(block_id="b1", order=0, source_format="pdf", level=1,
                          text="A Real Document Title", normalized="A Real Document Title"))
    doc.add(HeadingBlock(block_id="b2", order=1, source_format="pdf", level=2,
                          text="1 Introduction", normalized="1 Introduction"))
    doc.add(ParagraphBlock(block_id="b3", order=2, source_format="pdf", text="The introduction begins here today.",
                            normalized="The introduction begins here today."))
    doc.add(AuthorBlock(block_id="b4", order=3, source_format="pdf", text="Jane Doe", normalized="Jane Doe"))
    doc.add(AffiliationBlock(block_id="b5", order=4, source_format="pdf", text="Some University",
                              normalized="Some University"))
    report = _validate_doc(doc)
    codes = [f.code for f in report.findings.findings]
    assert "author_metadata_interleaved_with_body" in codes
    assert report.stats["front_matter_grouped"] is False
    assert report.stats["reading_order_valid"] is False
    assert report.stats["structural_readiness"] is False


def test_front_matter_grouped_right_after_title_is_valid():
    doc = Document(metadata=DocumentMetadata(source_path="x.pdf", source_format="pdf"))
    doc.add(HeadingBlock(block_id="b1", order=0, source_format="pdf", level=1,
                          text="A Real Document Title", normalized="A Real Document Title"))
    doc.add(AuthorBlock(block_id="b2", order=1, source_format="pdf", text="Jane Doe", normalized="Jane Doe"))
    doc.add(AffiliationBlock(block_id="b3", order=2, source_format="pdf", text="Some University",
                              normalized="Some University"))
    doc.add(HeadingBlock(block_id="b4", order=3, source_format="pdf", level=2, text="Abstract", normalized="Abstract"))
    doc.add(ParagraphBlock(block_id="b5", order=4, source_format="pdf", text="Some abstract text here.",
                            normalized="Some abstract text here."))
    report = _validate_doc(doc)
    assert report.stats["front_matter_grouped"] is True
    assert report.stats["author_metadata_interleaved_with_body"] is False
    assert report.stats["reading_order_valid"] is True


def test_paragraph_interrupted_by_front_matter_detected():
    doc = Document(metadata=DocumentMetadata(source_path="x.pdf", source_format="pdf"))
    doc.add(HeadingBlock(block_id="b1", order=0, source_format="pdf", level=1,
                          text="A Real Document Title", normalized="A Real Document Title"))
    text = (
        "The introduction continues here. Copyright 2024 ACM. All rights reserved. "
        "And then the sentence resumes after the interruption for good measure now."
    )
    doc.add(ParagraphBlock(block_id="b2", order=1, source_format="pdf", text=text, normalized=text))
    report = _validate_doc(doc)
    codes = [f.code for f in report.findings.findings]
    assert "paragraph_interrupted_by_front_matter" in codes
    assert report.stats["paragraph_continuity_valid"] is False


def test_non_blocking_long_paragraph_does_not_block_indexing():
    doc = Document(metadata=DocumentMetadata(source_path="x.pdf", source_format="pdf"))
    doc.add(HeadingBlock(block_id="b1", order=0, source_format="pdf", level=1,
                          text="A Real Document Title", normalized="A Real Document Title"))
    long_text = "This is a perfectly coherent sentence about systems. " * 20
    doc.add(ParagraphBlock(block_id="b2", order=1, source_format="pdf", text=long_text, normalized=long_text))
    report = _validate_doc(doc)
    codes = [f.code for f in report.findings.findings]
    assert "long_paragraph" in codes
    assert report.stats["structural_readiness"] is True
    assert report.stats["retrieval_ready"] is True

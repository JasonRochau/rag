"""Unit tests for heading_sibling_audit.py (spec section 9): recovering a
high-confidence numbered section heading that extraction rendered as
prose, without inventing anything the source text doesn't actually
contain."""
from document_converter.model import Document, DocumentMetadata, HeadingBlock, ParagraphBlock
from document_converter.findings import FindingLog
from document_converter.heading_sibling_audit import audit_numbered_sibling_headings


def _doc(*blocks):
    d = Document(metadata=DocumentMetadata(source_path="x.pdf", source_format="pdf"))
    for b in blocks:
        d.add(b)
    return d


def test_missing_sibling_heading_recovered():
    doc = _doc(
        HeadingBlock(block_id="b1", order=0, source_format="pdf", level=2,
                     text="1 Introduction", normalized="1 Introduction"),
        ParagraphBlock(block_id="b2", order=1, source_format="pdf",
                        text="Unfortunately most curricula do not cover this.",
                        normalized="Unfortunately most curricula do not cover this."),
        ParagraphBlock(
            block_id="b3", order=2, source_format="pdf",
            text="2 Logistical Issues With the ever-increasing use of systems, we cover "
                 "these topics over two courses.",
            normalized="2 Logistical Issues With the ever-increasing use of systems, we cover "
                       "these topics over two courses.",
        ),
        HeadingBlock(block_id="b4", order=3, source_format="pdf", level=2,
                     text="3 Data Representations", normalized="3 Data Representations"),
    )
    log = FindingLog()
    stats = audit_numbered_sibling_headings(doc, log)

    assert stats["high_confidence_headings_recovered"] == 1
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert [h.text for h in headings] == ["1 Introduction", "2 Logistical Issues", "3 Data Representations"]
    assert headings[1].level == headings[0].level == headings[2].level
    codes = [f.code for f in log.findings]
    assert "high_confidence_heading_recovered" in codes


def test_no_gap_no_audit_action():
    doc = _doc(
        HeadingBlock(block_id="b1", order=0, source_format="pdf", level=2,
                     text="1 Introduction", normalized="1 Introduction"),
        HeadingBlock(block_id="b2", order=1, source_format="pdf", level=2,
                     text="2 Results", normalized="2 Results"),
    )
    log = FindingLog()
    stats = audit_numbered_sibling_headings(doc, log)
    assert stats["high_confidence_headings_recovered"] == 0
    assert stats["outline_gaps_checked"] == 0


def test_legitimate_numbering_gap_without_matching_text_is_not_invented():
    """A gap in the numbering alone is not evidence of anything -- nothing
    should be promoted when the source text has no matching candidate
    (spec: "must not invent a missing section merely because numbering
    skips")."""
    doc = _doc(
        HeadingBlock(block_id="b1", order=0, source_format="pdf", level=2,
                     text="1 Introduction", normalized="1 Introduction"),
        ParagraphBlock(block_id="b2", order=1, source_format="pdf",
                        text="This chapter intentionally skips ahead to the next topic.",
                        normalized="This chapter intentionally skips ahead to the next topic."),
        HeadingBlock(block_id="b3", order=2, source_format="pdf", level=2,
                     text="3 Data Representations", normalized="3 Data Representations"),
    )
    log = FindingLog()
    stats = audit_numbered_sibling_headings(doc, log)
    assert stats["high_confidence_headings_recovered"] == 0
    headings = [b.text for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert headings == ["1 Introduction", "3 Data Representations"]


def test_ambiguous_multiple_candidates_reported_not_guessed():
    doc = _doc(
        HeadingBlock(block_id="b1", order=0, source_format="pdf", level=2,
                     text="1 Introduction", normalized="1 Introduction"),
        ParagraphBlock(
            block_id="b2", order=1, source_format="pdf",
            text="2 Logistical Issues With some content that could be a heading candidate.",
            normalized="2 Logistical Issues With some content that could be a heading candidate.",
        ),
        ParagraphBlock(
            block_id="b3", order=2, source_format="pdf",
            text="2 Scheduling Concerns With other content shaped the exact same way here.",
            normalized="2 Scheduling Concerns With other content shaped the exact same way here.",
        ),
        HeadingBlock(block_id="b4", order=3, source_format="pdf", level=2,
                     text="3 Data Representations", normalized="3 Data Representations"),
    )
    log = FindingLog()
    stats = audit_numbered_sibling_headings(doc, log)
    assert stats["high_confidence_headings_recovered"] == 0
    assert stats["high_confidence_headings_ambiguous"] == 1
    codes = [f.code for f in log.findings]
    assert "high_confidence_heading_not_preserved" in codes

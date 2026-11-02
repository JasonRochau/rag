"""
Regression tests for the front-matter/title normalization fixes made while
investigating "Regression Analyses and Their Particularities in
Observational Studies.pdf" (author_metadata_interleaved_with_body /
publication_metadata_interleaved_with_body, both false positives caused by
a later normalize.py pass breaking an already-correct front-matter/title
adjacency -- see DELIVERABLES.md).

Covers, at the unit level (bypassing OCR so these run fast and
deterministically):
  1. A title wrapped across two adjacent title-shaped regions is merged.
  2. A complete short title followed by a legitimate bodyless subtitle is
     *not* merged.
  3. A document-type label ("Review Article") near authors is not
     classified as a person.
  4. `_enforce_single_first_h1`'s H1-move-to-front repair carries an
     already-contiguous front-matter run along instead of stranding it
     behind stray page-furniture text.
  5/9. The independent validation.py placement check: a genuinely valid
     run passes, and a genuinely interleaved one (a second author block
     stranded deep in the body) still fails.
  6. A recurring front-matter-shaped line recovered *mid-document*, after
     the initial contiguous grouping already ran (paragraph_continuity_repair's
     residual repair pass), is never promoted to a front-matter type -- so
     it can never retroactively break a front-matter run that was already
     correct.
"""
from __future__ import annotations

from document_converter.model import (
    Document, DocumentMetadata, HeadingBlock, ParagraphBlock, UnknownBlock,
    AuthorBlock, AffiliationBlock, PublicationMetadataBlock, FRONT_MATTER_BLOCK_TYPES,
)
from document_converter.normalize import normalize_document
from document_converter.front_matter import classify_front_matter_line
from document_converter.validation import _validate_front_matter_placement
from document_converter.paragraph_continuity_repair import repair_interrupted_paragraphs
from document_converter.findings import FindingLog


def _mk(cls, block_id, order, **kw):
    return cls(block_id=block_id, order=order, source_format="pdf", **kw)


def _doc(blocks):
    d = Document(metadata=DocumentMetadata(source_path="x.pdf", source_format="pdf"))
    d.blocks = blocks
    return d


# ---------------------------------------------------------------------- #
# classify_front_matter_line: document-type labels vs. real author names

def test_review_article_label_is_not_classified_as_author():
    assert classify_front_matter_line("Review Article") == "publication_metadata"


def test_other_article_type_labels_not_classified_as_author():
    for label in ("Case Report", "Original Article", "Systematic Review", "Editorial"):
        assert classify_front_matter_line(label) != "author"


def test_genuine_two_word_author_name_still_classified_as_author():
    assert classify_front_matter_line("Randal Bryant") == "author"


def test_doi_line_classified_as_publication_metadata():
    assert classify_front_matter_line("DOI: 10.3238/arztebl.m2023.0278") == "publication_metadata"


# ---------------------------------------------------------------------- #
# _reconstruct_split_title: wrapped title vs. legitimate subtitle

def test_split_title_across_two_dangling_regions_is_merged():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="Regression Analyses and Their", page=1),
        _mk(AffiliationBlock, "b2", 1, text="Institute of Medicine", page=1),
        _mk(AuthorBlock, "b3", 2, text="Jane Doe", page=1),
        _mk(HeadingBlock, "b4", 3, level=2, text="Particularities in Observational Studies", page=1),
        _mk(ParagraphBlock, "b5", 4, text="Body text starts here.", page=1),
    ]
    doc = _doc(blocks)
    log = normalize_document(doc)
    h1s = [b for b in doc.blocks if isinstance(b, HeadingBlock) and b.level == 1]
    assert len(h1s) == 1
    assert h1s[0].text == "Regression Analyses and Their Particularities in Observational Studies"
    assert any(f.code == "title_continuation_merged" for f in log.findings)


def test_complete_short_title_with_legitimate_subtitle_not_merged():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="Deep Learning", page=1),
        _mk(HeadingBlock, "b2", 1, level=2, text="A Practical Introduction", page=1),
        _mk(ParagraphBlock, "b3", 2, text="Body text starts here.", page=1),
    ]
    doc = _doc(blocks)
    log = normalize_document(doc)
    h1s = [b for b in doc.blocks if isinstance(b, HeadingBlock) and b.level == 1]
    assert h1s[0].text == "Deep Learning"
    assert not any(f.code == "title_continuation_merged" for f in log.findings)
    subtitle = [b for b in doc.blocks if isinstance(b, HeadingBlock) and b.level == 2]
    assert subtitle and subtitle[0].text == "A Practical Introduction"


# ---------------------------------------------------------------------- #
# h1_moved_to_front must carry an already-contiguous front-matter run along

def test_h1_move_carries_contiguous_front_matter_run():
    blocks = [
        _mk(ParagraphBlock, "b1", 0, text="Stray masthead fragment.", page=1),
        _mk(HeadingBlock, "b2", 1, level=1, text="A Study of Something", page=1),
        _mk(AuthorBlock, "b3", 2, text="Jane Doe", page=1),
        _mk(AffiliationBlock, "b4", 3, text="Some University", page=1),
        _mk(ParagraphBlock, "b5", 4, text="Body paragraph begins here.", page=1),
    ]
    doc = _doc(blocks)
    log = normalize_document(doc)
    h1_idx = next(i for i, b in enumerate(doc.blocks) if isinstance(b, HeadingBlock) and b.level == 1)
    fm_indices = [i for i, b in enumerate(doc.blocks) if isinstance(b, FRONT_MATTER_BLOCK_TYPES)]
    assert h1_idx == 0
    assert fm_indices == [1, 2]
    assert any(f.code == "h1_moved_to_front" for f in log.findings)
    placement = _validate_front_matter_placement(doc, FindingLog())
    assert placement["author_metadata_interleaved_with_body"] is False


# ---------------------------------------------------------------------- #
# validation.py: end-to-end front-matter placement check

def test_valid_front_matter_run_passes_placement_check():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="A Study of Something", page=1),
        _mk(AuthorBlock, "b2", 1, text="Jane Doe", page=1),
        _mk(AffiliationBlock, "b3", 2, text="Some University", page=1),
        _mk(ParagraphBlock, "b4", 3, text="Body paragraph begins here.", page=1),
    ]
    doc = _doc(blocks)
    result = _validate_front_matter_placement(doc, FindingLog())
    assert result["author_metadata_interleaved_with_body"] is False
    assert result["publication_metadata_interleaved_with_body"] is False


def test_genuinely_interleaved_front_matter_still_fails():
    """A second, physically separate author block stranded deep in the body
    (not adjacent to the title's front-matter group) must still fail --
    this is the real defect the check exists to catch, and none of the
    normalize.py repairs above should ever paper over it."""
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="A Study of Something", page=1),
        _mk(AuthorBlock, "b2", 1, text="Jane Doe", page=1),
        _mk(AffiliationBlock, "b3", 2, text="Some University", page=1),
        _mk(ParagraphBlock, "b4", 3, text="Body paragraph begins here.", page=1),
        _mk(HeadingBlock, "b5", 4, level=2, text="Section Two", page=2),
        _mk(AuthorBlock, "b6", 5, text="John Smith", page=2),
        _mk(ParagraphBlock, "b7", 6, text="More body content.", page=2),
    ]
    doc = _doc(blocks)
    log = FindingLog()
    result = _validate_front_matter_placement(doc, log)
    assert result["author_metadata_interleaved_with_body"] is True
    assert any(f.code == "author_metadata_interleaved_with_body" for f in log.findings)


# ---------------------------------------------------------------------- #
# A front-matter-shaped line recovered mid-document, after the initial
# grouping already ran, must never be promoted to a front-matter type.

def test_mid_document_repeated_doi_recovered_from_paragraph_is_not_front_matter_typed():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="A Study of Something", page=1),
        _mk(AuthorBlock, "b2", 1, text="Jane Doe", page=1),
        _mk(
            ParagraphBlock, "b3", 2, page=8,
            text=(
                "The analysis concluded with several caveats. "
                "DOI: 10.3238/arztebl.m2023.0278 "
                "Further replication is warranted."
            ),
        ),
    ]
    doc = _doc(blocks)
    log = FindingLog()
    repair_interrupted_paragraphs(doc, log)
    assert not any(isinstance(b, FRONT_MATTER_BLOCK_TYPES) and b.page == 8 for b in doc.blocks)
    recovered = [b for b in doc.blocks if isinstance(b, UnknownBlock) and "DOI" in (b.text or "")]
    assert recovered, "the DOI sentence must still be excised into its own block, just not front-matter-typed"
    placement = _validate_front_matter_placement(doc, FindingLog())
    assert placement["publication_metadata_interleaved_with_body"] is False

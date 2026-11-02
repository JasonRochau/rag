"""
Tests for the generalization-fixture repair passes added alongside fixture 3
(`A Brief Review of a Machine Learning Programming of Simple Linear
Regression`): title-adjacent front-matter reconstruction and figure-caption
normalization (normalize.py), repeated running-header removal
(running_header_repair.py), and repeated top-level section-numbering audit
(section_numbering_audit.py). All operate on the common Document/Block
model, so a hand-built Document exercises them the same way any parser's
output would -- no PDF/OCR involved.
"""
from __future__ import annotations

from document_converter.model import (
    Document, DocumentMetadata, HeadingBlock, ParagraphBlock, CaptionBlock,
    AuthorBlock, AffiliationBlock, ConferenceMetadataBlock,
)
from document_converter.findings import FindingLog
from document_converter.normalize import normalize_document
from document_converter.running_header_repair import repair_running_headers
from document_converter.section_numbering_audit import audit_repeated_top_level_numbering


def _mk(cls, block_id, order, **kw):
    return cls(block_id=block_id, order=order, source_format="pdf", **kw)


def _doc(blocks):
    doc = Document(metadata=DocumentMetadata(source_path="x.pdf", source_format="pdf"))
    doc.blocks = blocks
    return doc


# ---------------------------------------------------------------------- #
# normalize.py: leading front-matter reconstruction + caption normalization

def test_compound_author_paragraph_after_title_reconstructed_as_front_matter():
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="A Paper Title", normalized="A Paper Title", page=1),
        _mk(AffiliationBlock, "b0002", 1, text="Some Research Institute", normalized="Some Research Institute", page=1),
        _mk(ParagraphBlock, "b0003", 2, page=1,
            text='Jane A. Doe, John Q. Public, Main Street, Springfield, 62704, USA '
                 'Corresponding author: jane.doe@example.edu',
            normalized='Jane A. Doe, John Q. Public, Main Street, Springfield, 62704, USA '
                       'Corresponding author: jane.doe@example.edu'),
        _mk(ParagraphBlock, "b0004", 3, page=1, text="This paper studies an ordinary topic in depth.",
            normalized="This paper studies an ordinary topic in depth."),
    ]
    doc = _doc(blocks)
    normalize_document(doc)
    assert isinstance(doc.blocks[2], AuthorBlock)
    assert isinstance(doc.blocks[3], ParagraphBlock)  # body content past front matter is left untouched


def test_reconstruction_stops_at_first_non_front_matter_block():
    """The leading-run scan must never reach into the body: a body
    paragraph sitting between two front-matter-shaped lines (an unusual but
    possible extraction order) must stop the scan, not be skipped over."""
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="A Paper Title", normalized="A Paper Title", page=1),
        _mk(ParagraphBlock, "b0002", 1, page=1, text="Jane A. Doe and John Q. Public",
            normalized="Jane A. Doe and John Q. Public"),
        _mk(ParagraphBlock, "b0003", 2, page=1, text="This is ordinary body prose that starts the paper.",
            normalized="This is ordinary body prose that starts the paper."),
        _mk(ParagraphBlock, "b0004", 3, page=1, text="Some Research Institute",
            normalized="Some Research Institute"),
    ]
    doc = _doc(blocks)
    normalize_document(doc)
    assert isinstance(doc.blocks[1], AuthorBlock)
    assert isinstance(doc.blocks[2], ParagraphBlock)
    assert doc.blocks[2].text.startswith("This is ordinary body prose")
    # b0004 ("Some Research Institute") is affiliation-shaped but sits past
    # the body paragraph that already closed the leading front-matter run.
    assert isinstance(doc.blocks[3], ParagraphBlock)


def test_figure_caption_paragraph_normalized_to_caption_block():
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="Figure 2 MLP results with learning rate = 0.01",
            normalized="Figure 2 MLP results with learning rate = 0.01"),
        _mk(ParagraphBlock, "b0003", 2, text="Table 1 Types of values used as training data",
            normalized="Table 1 Types of values used as training data"),
    ]
    doc = _doc(blocks)
    normalize_document(doc)
    assert isinstance(doc.blocks[1], CaptionBlock)
    assert isinstance(doc.blocks[2], CaptionBlock)


def test_ordinary_paragraph_mentioning_a_figure_number_not_normalized():
    """Only a line that *starts* with the figure/table label is a caption --
    a body sentence merely referencing one elsewhere must stay a paragraph."""
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="The Y values used for training are shown in Table 1.",
            normalized="The Y values used for training are shown in Table 1."),
    ]
    doc = _doc(blocks)
    normalize_document(doc)
    assert isinstance(doc.blocks[1], ParagraphBlock)


# ---------------------------------------------------------------------- #
# running_header_repair.py

def test_repeated_page_boundary_text_removed_preserving_front_matter_occurrence():
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title", page=1),
        _mk(ConferenceMetadataBlock, "b0002", 1, page=1, text="Proceedings of Some Venue, 2020",
            normalized="Proceedings of Some Venue, 2020"),
        _mk(ParagraphBlock, "b0003", 2, page=1, text="Body content on page one begins here.",
            normalized="Body content on page one begins here."),
        _mk(ParagraphBlock, "b0004", 3, page=2, text="Proceedings of Some Venue, 2020",
            normalized="Proceedings of Some Venue, 2020"),
        _mk(ParagraphBlock, "b0005", 4, page=2, text="Body content on page two continues here.",
            normalized="Body content on page two continues here."),
    ]
    doc = _doc(blocks)
    log = FindingLog()
    stats = repair_running_headers(doc, log)
    assert stats["running_headers_removed"] == 1
    remaining_texts = [getattr(b, "text", "") for b in doc.blocks]
    assert remaining_texts.count("Proceedings of Some Venue, 2020") == 1
    assert isinstance(doc.blocks[1], ConferenceMetadataBlock)  # the preserved occurrence, untouched
    assert any(f.code == "running_header_removed" for f in log.findings)


def test_repeated_mid_page_text_not_treated_as_running_header():
    """Repetition alone is not enough -- the text must also sit at a
    page-boundary position on every occurrence. A phrase repeated in the
    middle of page content (not near either edge) must be left alone."""
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title", page=1),
        _mk(ParagraphBlock, "b0002", 1, page=1, text="Intro paragraph one.", normalized="Intro paragraph one."),
        _mk(ParagraphBlock, "b0003", 2, page=1, text="Intro paragraph two.", normalized="Intro paragraph two."),
        _mk(ParagraphBlock, "b0004", 3, page=1, text="the important term recurs here",
            normalized="the important term recurs here"),
        _mk(ParagraphBlock, "b0005", 4, page=1, text="Intro paragraph four.", normalized="Intro paragraph four."),
        _mk(ParagraphBlock, "b0006", 5, page=1, text="Intro paragraph five.", normalized="Intro paragraph five."),
        _mk(ParagraphBlock, "b0007", 6, page=2, text="Body paragraph one.", normalized="Body paragraph one."),
        _mk(ParagraphBlock, "b0008", 7, page=2, text="Body paragraph two.", normalized="Body paragraph two."),
        _mk(ParagraphBlock, "b0009", 8, page=2, text="the important term recurs here",
            normalized="the important term recurs here"),
        _mk(ParagraphBlock, "b0010", 9, page=2, text="Body paragraph four.", normalized="Body paragraph four."),
        _mk(ParagraphBlock, "b0011", 10, page=2, text="Body paragraph five.", normalized="Body paragraph five."),
    ]
    doc = _doc(blocks)
    log = FindingLog()
    stats = repair_running_headers(doc, log)
    assert stats["running_headers_removed"] == 0
    assert len(doc.blocks) == 11


# ---------------------------------------------------------------------- #
# section_numbering_audit.py

def test_all_top_level_headings_sharing_one_number_renumbered_in_document_order():
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(HeadingBlock, "b0002", 1, level=2, text="1. Introduction", normalized="1. Introduction"),
        _mk(HeadingBlock, "b0003", 2, level=2, text="1. Methods", normalized="1. Methods"),
        _mk(HeadingBlock, "b0004", 3, level=2, text="1. Results", normalized="1. Results"),
        _mk(HeadingBlock, "b0005", 4, level=2, text="1. Discussion", normalized="1. Discussion"),
        _mk(HeadingBlock, "b0006", 5, level=2, text="1. Conclusions", normalized="1. Conclusions"),
    ]
    doc = _doc(blocks)
    log = FindingLog()
    stats = audit_repeated_top_level_numbering(doc, log)
    assert stats["headings_renumbered"] == 4  # "1. Introduction" already matches position 1, untouched
    texts = [b.text for b in doc.blocks if isinstance(b, HeadingBlock) and b.level > 1]
    assert texts == ["1. Introduction", "2. Methods", "3. Results", "4. Discussion", "5. Conclusions"]
    assert not any(f.code == "repeated_top_level_section_numbering" for f in log.findings)


def test_numbering_restart_at_one_starts_new_segment_not_flagged():
    """A restart to 1 after a higher number starts a new segment (an
    anthology's next article), so cross-segment duplicates are not flagged
    and nothing is rewritten."""
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(HeadingBlock, "b0002", 1, level=2, text="1 Introduction", normalized="1 Introduction"),
        _mk(HeadingBlock, "b0003", 2, level=2, text="2 Background", normalized="2 Background"),
        _mk(HeadingBlock, "b0004", 3, level=2, text="1 Duplicated Section", normalized="1 Duplicated Section"),
        _mk(HeadingBlock, "b0005", 4, level=2, text="3 Conclusions", normalized="3 Conclusions"),
    ]
    doc = _doc(blocks)
    original_texts = [b.text for b in blocks if isinstance(b, HeadingBlock)]
    log = FindingLog()
    stats = audit_repeated_top_level_numbering(doc, log)
    # The "1" after "2" is a reset to a new segment; no within-segment repeat.
    assert stats["headings_renumbered"] == 0
    assert stats["repeated_number_groups_flagged"] == 0
    texts = [b.text for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert texts == original_texts  # nothing rewritten
    assert not any(f.code == "repeated_top_level_section_numbering" for f in log.findings)


def test_repeated_numbering_inside_one_segment_is_flagged():
    """Synthetic corrupted numbering that repeats inside a single segment
    (no intervening reset) must still be flagged and left unrepaired."""
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(HeadingBlock, "b0002", 1, level=2, text="1 Introduction", normalized="1 Introduction"),
        _mk(HeadingBlock, "b0003", 2, level=2, text="2 Background", normalized="2 Background"),
        _mk(HeadingBlock, "b0004", 3, level=2, text="2 Duplicated Section", normalized="2 Duplicated Section"),
        _mk(HeadingBlock, "b0005", 4, level=2, text="3 Conclusions", normalized="3 Conclusions"),
    ]
    doc = _doc(blocks)
    original_texts = [b.text for b in blocks if isinstance(b, HeadingBlock)]
    log = FindingLog()
    stats = audit_repeated_top_level_numbering(doc, log)
    assert stats["headings_renumbered"] == 0
    assert stats["repeated_number_groups_flagged"] == 1  # "2" repeats within the segment
    texts = [b.text for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert texts == original_texts  # nothing rewritten
    errors = [f for f in log.findings if f.code == "repeated_top_level_section_numbering"]
    assert len(errors) == 2  # both "2" headings flagged


def test_decrease_to_a_number_above_one_is_not_a_segment_restart():
    """Only a restart to 0/1 opens a new segment; dropping back to 3 after 5
    is corruption within the segment and must still be flagged."""
    nums = [1, 2, 3, 4, 5, 3, 6]
    blocks = [_mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title")]
    for i, n in enumerate(nums, start=2):
        t = f"{n} Section {i}"
        blocks.append(_mk(HeadingBlock, f"b{i:04d}", i - 1, level=2, text=t, normalized=t))
    doc = _doc(blocks)
    log = FindingLog()
    stats = audit_repeated_top_level_numbering(doc, log)
    assert stats["headings_renumbered"] == 0
    assert stats["repeated_number_groups_flagged"] == 1
    errors = [f for f in log.findings if f.code == "repeated_top_level_section_numbering"]
    assert len(errors) == 2  # both "3" headings flagged


def test_all_same_number_run_after_restart_is_repaired_per_segment():
    """Anthology whose second article's numbering is all-"1" corrupted: the
    first article is untouched, the second is renumbered in document order."""
    nums = [1, 2, 3, 1, 1, 1]
    blocks = [_mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title")]
    for i, n in enumerate(nums, start=2):
        t = f"{n}. Part {i}"
        blocks.append(_mk(HeadingBlock, f"b{i:04d}", i - 1, level=2, text=t, normalized=t))
    doc = _doc(blocks)
    log = FindingLog()
    audit_repeated_top_level_numbering(doc, log)
    texts = [b.text for b in doc.blocks if isinstance(b, HeadingBlock) and b.level > 1]
    assert texts == ["1. Part 2", "2. Part 3", "3. Part 4", "1. Part 5", "2. Part 6", "3. Part 7"]
    assert not any(f.code == "repeated_top_level_section_numbering" for f in log.findings)


def test_numbering_glued_to_following_word_not_treated_as_top_level():
    """A parser artifact (observed: Docling) that glues a sub-heading's
    numbering directly onto the next word with no space ("3.2Tracking
    Vertex Visit Order") must not be misread as a top-level heading numbered
    "3" -- `numbering_tuple`'s prefix regex backtracks to the short match in
    that shape, and this audit must recognize and exclude it rather than
    flag/rewrite based on a number that was never really there."""
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(HeadingBlock, "b0002", 1, level=2, text="3 Real Top Level Section", normalized="3 Real Top Level Section"),
        _mk(HeadingBlock, "b0003", 2, level=3, text="3.2Tracking Vertex Visit Order",
            normalized="3.2Tracking Vertex Visit Order"),
        _mk(HeadingBlock, "b0004", 3, level=3, text="3.3Stack Operations", normalized="3.3Stack Operations"),
    ]
    doc = _doc(blocks)
    log = FindingLog()
    stats = audit_repeated_top_level_numbering(doc, log)
    assert stats["repeated_number_groups_flagged"] == 0
    assert stats["headings_renumbered"] == 0
    assert not any(f.code == "repeated_top_level_section_numbering" for f in log.findings)


def test_fullwidth_digit_callout_headings_not_treated_as_numbering_corruption():
    """Microsoft Learn PDFs render the "docons" Note-icon glyph as the
    Unicode fullwidth digit '７' (U+FF17) in extracted text. Python's \d
    matches it, so before the ASCII-only fix every Note callout parsed as a
    top-level heading numbered 7, and N callouts looked like unambiguous
    numbering corruption -- the audit then REWROTE the callout text in place
    ("1 Note", "2 Note", ...). Fullwidth digits are not section numbers, so
    the audit must see no numbered headings here at all: no repair, no flag,
    no text mutation."""
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(HeadingBlock, "b0002", 1, level=2, text="７ Note", normalized="７ Note"),
        _mk(ParagraphBlock, "b0003", 2, text="First callout body.", normalized="First callout body."),
        _mk(HeadingBlock, "b0004", 3, level=2, text="７ Note", normalized="７ Note"),
        _mk(ParagraphBlock, "b0005", 4, text="Second callout body.", normalized="Second callout body."),
        _mk(HeadingBlock, "b0006", 5, level=2, text="７ Note", normalized="７ Note"),
    ]
    doc = _doc(blocks)
    original_texts = [b.text for b in blocks]
    log = FindingLog()
    stats = audit_repeated_top_level_numbering(doc, log)
    assert stats["headings_renumbered"] == 0
    assert stats["repeated_number_groups_flagged"] == 0
    assert not any(f.code == "repeated_top_level_numbering_repaired" for f in log.findings)
    assert not any(f.code == "repeated_top_level_section_numbering" for f in log.findings)
    assert [b.text for b in doc.blocks] == original_texts  # callout text untouched

"""
Unit tests for retrieval_view.py's MAIN_TEXT filtering and title
resolution (spec "MAIN_TEXT vs FULL_FIDELITY ingestion profiles",
Phases 2/3/5). Hand-built `Document`/block objects, no OCR/PDF dependency,
so these run fast and deterministically.
"""
from __future__ import annotations

from document_converter.model import (
    Document, DocumentMetadata, HeadingBlock, ParagraphBlock, UnknownBlock,
    AuthorBlock, AffiliationBlock, ContactBlock, KeywordBlock, PublicationMetadataBlock,
    CopyrightNoticeBlock, PermissionNoticeBlock, ConferenceMetadataBlock,
    CaptionBlock, FigureLabelBlock, DiagramAnnotationBlock, TableCellBlock, ImageBlock,
)
from document_converter.retrieval_view import build_main_text_document, resolve_main_text_title, apply_resolved_title
from document_converter.render import render_markdown


def _mk(cls, block_id, order, **kw):
    return cls(block_id=block_id, order=order, source_format="pdf", **kw)


def _doc(blocks, title=None):
    d = Document(metadata=DocumentMetadata(source_path="x.pdf", source_format="pdf", title=title))
    d.blocks = blocks
    return d


def _types(blocks):
    return [type(b).__name__ for b in blocks]


# ---------------------------------------------------------------------- #
# Exclusion categories (spec items 1-11)

def test_excludes_author_block():
    blocks = [_mk(HeadingBlock, "b1", 0, level=1, text="T", page=1),
              _mk(AuthorBlock, "b2", 1, text="Jane Doe", page=1),
              _mk(ParagraphBlock, "b3", 2, text="Body text.", page=1)]
    filtered, stats = build_main_text_document(_doc(blocks))
    assert "AuthorBlock" not in _types(filtered.blocks)
    assert stats.excluded_by_type.get("AuthorBlock") == 1


def test_excludes_affiliation_block():
    blocks = [_mk(HeadingBlock, "b1", 0, level=1, text="T", page=1),
              _mk(AffiliationBlock, "b2", 1, text="Some University", page=1),
              _mk(ParagraphBlock, "b3", 2, text="Body text.", page=1)]
    filtered, _ = build_main_text_document(_doc(blocks))
    assert "AffiliationBlock" not in _types(filtered.blocks)


def test_excludes_contact_and_keyword_blocks():
    blocks = [_mk(HeadingBlock, "b1", 0, level=1, text="T", page=1),
              _mk(ContactBlock, "b2", 1, text="jane@example.com", page=1),
              _mk(KeywordBlock, "b3", 2, text="Keywords: a, b, c", page=1),
              _mk(ParagraphBlock, "b4", 3, text="Body text.", page=1)]
    filtered, _ = build_main_text_document(_doc(blocks))
    kinds = _types(filtered.blocks)
    assert "ContactBlock" not in kinds
    assert "KeywordBlock" not in kinds


def test_excludes_copyright_and_permission_blocks():
    blocks = [_mk(HeadingBlock, "b1", 0, level=1, text="T", page=1),
              _mk(CopyrightNoticeBlock, "b2", 1, text="Copyright 2024 ACM.", page=1),
              _mk(PermissionNoticeBlock, "b3", 2, text="Permission to make digital copies...", page=1),
              _mk(ParagraphBlock, "b4", 3, text="Body text.", page=1)]
    filtered, _ = build_main_text_document(_doc(blocks))
    kinds = _types(filtered.blocks)
    assert "CopyrightNoticeBlock" not in kinds
    assert "PermissionNoticeBlock" not in kinds


def test_excludes_publication_history_metadata_block():
    blocks = [_mk(HeadingBlock, "b1", 0, level=1, text="T", page=1),
              _mk(ConferenceMetadataBlock, "b2", 1, text="Proceedings of X, 2024", page=1),
              _mk(PublicationMetadataBlock, "b3", 2, text="Received Sept 2018; accepted Jan 2019", page=1),
              _mk(ParagraphBlock, "b4", 3, text="Body text.", page=1)]
    filtered, _ = build_main_text_document(_doc(blocks))
    kinds = _types(filtered.blocks)
    assert "ConferenceMetadataBlock" not in kinds
    assert "PublicationMetadataBlock" not in kinds


def test_excludes_figures_captions_and_image_descriptions():
    # The heading is followed by a real paragraph first, not graphical
    # debris, so this isolates the per-type exclusion from the separate
    # shell-heading rule (covered by its own tests below).
    blocks = [_mk(HeadingBlock, "b1", 0, level=1, text="T", page=1),
              _mk(ParagraphBlock, "b2", 1, text="Body text.", page=1),
              _mk(ImageBlock, "b3", 2, page=1),
              _mk(FigureLabelBlock, "b4", 3, text="Modern processor", page=1),
              _mk(CaptionBlock, "b5", 4, text="Fig. 1. A diagram.", page=1),
              _mk(DiagramAnnotationBlock, "b6", 5, text="Y - Axis", page=1),
              _mk(TableCellBlock, "b7", 6, text="cell", page=1)]
    filtered, stats = build_main_text_document(_doc(blocks))
    kinds = _types(filtered.blocks)
    for excluded in ("ImageBlock", "FigureLabelBlock", "CaptionBlock", "DiagramAnnotationBlock", "TableCellBlock"):
        assert excluded not in kinds
    assert kinds == ["HeadingBlock", "ParagraphBlock"]


def test_excludes_table_of_contents_section():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="T", page=1),
        _mk(HeadingBlock, "b2", 1, level=2, text="Table of Contents", page=1),
        _mk(ParagraphBlock, "b3", 2, text="1 Introduction .... 1", page=1),
        _mk(ParagraphBlock, "b4", 3, text="2 Methods .... 5", page=1),
        _mk(HeadingBlock, "b5", 4, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b6", 5, text="Real introduction body text.", page=1),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    texts = [getattr(b, "text", "") for b in filtered.blocks]
    assert "Table of Contents" not in texts
    assert "1 Introduction .... 1" not in texts
    assert "Real introduction body text." in texts
    assert stats.excluded_toc_blocks == 3  # heading + 2 entries


def test_excludes_references_section_content():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="T", page=1),
        _mk(ParagraphBlock, "b2", 1, text="Real body text.", page=1),
        _mk(HeadingBlock, "b3", 2, level=2, text="References", page=1),
        _mk(ParagraphBlock, "b4", 3, text="Smith, J. 2020. Some Paper. DOI: 10.1000/x", page=1),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    texts = [getattr(b, "text", "") for b in filtered.blocks]
    assert "References" not in texts
    assert not any("Smith, J." in t for t in texts)
    assert "Real body text." in texts
    assert stats.excluded_references_blocks == 2


def test_full_fidelity_preserves_all_blocks_unfiltered():
    """build_main_text_document is never called for FULL_FIDELITY -- the
    pipeline uses the original, unfiltered Document as-is (see
    pipeline.py's `_prepare_for_profile`). This test documents the
    invariant at the Document level: nothing here mutates the input."""
    blocks = [_mk(HeadingBlock, "b1", 0, level=1, text="T", page=1),
              _mk(AuthorBlock, "b2", 1, text="Jane Doe", page=1),
              _mk(CaptionBlock, "b3", 2, text="Fig. 1.", page=1),
              _mk(ParagraphBlock, "b4", 3, text="Body text.", page=1)]
    original = _doc(blocks)
    build_main_text_document(original)  # MAIN_TEXT view derived...
    assert _types(original.blocks) == [  # ...but original is untouched.
        "HeadingBlock", "AuthorBlock", "CaptionBlock", "ParagraphBlock",
    ]


# ---------------------------------------------------------------------- #
# Shell-heading exclusion: a figure-adjacent heading must not survive as
# a section title (spec Phase 5's "Modern processor" requirement).

def test_shell_heading_immediately_followed_by_figure_debris_is_excluded():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="Real Title", page=1),
        _mk(ParagraphBlock, "b2", 1, text="Real body text.", page=1),
        _mk(HeadingBlock, "b3", 2, level=1, text="Modern processor", page=4),
        _mk(PublicationMetadataBlock, "b4", 3, text="https://doi.org/10.1145/x", page=1),
        _mk(CaptionBlock, "b5", 4, text="Fig. 1. A diagram.", page=4),
        _mk(HeadingBlock, "b6", 5, level=2, text="2.2 Execution Model", page=4),
        _mk(ParagraphBlock, "b7", 6, text="More real body text.", page=4),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    texts = [getattr(b, "text", "") for b in filtered.blocks]
    assert "Modern processor" not in texts
    assert "Real body text." in texts
    assert "2.2 Execution Model" in texts
    assert "More real body text." in texts
    assert stats.excluded_shell_heading_blocks == 1


def test_ordinary_heading_followed_by_paragraph_is_not_excluded():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="Real Title", page=1),
        _mk(HeadingBlock, "b2", 1, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b3", 2, text="Introduction body text.", page=1),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    assert "1 Introduction" in [getattr(b, "text", "") for b in filtered.blocks]
    assert stats.excluded_shell_heading_blocks == 0


# ---------------------------------------------------------------------- #
# Title resolution (spec items 21-25)

def test_figure_label_cannot_become_title():
    # Matches the real OpenCL-paper shape: a genuine first/root heading
    # exists ahead of the figure-label heading -- "Modern processor" is a
    # *later*, non-title heading, which is what makes it eligible for the
    # shell-heading exclusion in the first place (a figure label that
    # happened to be a document's *only* heading would instead be handled
    # by `document.metadata.title` resolving from PDF metadata, as it did
    # for the real document -- see test_actual_article_title_becomes_sole_root_title).
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="2 Overview and Challenges", page=3),
        _mk(ParagraphBlock, "b2", 1, text="Overview body text.", page=3),
        _mk(HeadingBlock, "b3", 2, level=1, text="Modern processor", page=4),
        _mk(PublicationMetadataBlock, "b4", 3, text="https://doi.org/x", page=1),
        _mk(HeadingBlock, "b5", 4, level=2, text="2.2 Execution Model", page=4),
        _mk(ParagraphBlock, "b6", 5, text="Real body text.", page=4),
    ]
    doc = _doc(blocks, title=None)
    filtered, _ = build_main_text_document(doc)
    title, source = resolve_main_text_title(doc, filtered.blocks)
    assert title != "Modern processor"
    assert title == "2 Overview and Challenges"
    assert source == "first_body_heading"


def test_actual_article_title_becomes_sole_root_title():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="Some Broken Body Heading", page=3),
        _mk(ParagraphBlock, "b2", 1, text="Body text.", page=3),
    ]
    doc = _doc(blocks, title="Augmenting Operating Systems with OpenCL Accelerators")
    filtered, _ = build_main_text_document(doc)
    title, source = resolve_main_text_title(doc, filtered.blocks)
    assert title == "Augmenting Operating Systems with OpenCL Accelerators"
    assert source == "metadata"


def test_toc_entry_cannot_become_root_title():
    blocks = [_mk(HeadingBlock, "b1", 0, level=2, text="Table of Contents", page=1)]
    doc = _doc(blocks, title="Table of Contents")
    filtered, _ = build_main_text_document(doc)
    title, source = resolve_main_text_title(doc, filtered.blocks)
    assert title is None
    assert source == "unresolved"


def test_running_header_cannot_become_root_title():
    """A running header/footer line the extractor already classified as
    front-matter-shaped publication metadata must not resolve as title
    even if it happens to match `document.metadata.title` verbatim."""
    blocks = [_mk(PublicationMetadataBlock, "b1", 0, text="Deutsches Arzteblatt International", page=3)]
    doc = _doc(blocks, title="Deutsches Arzteblatt International")
    filtered, _ = build_main_text_document(doc)
    title, source = resolve_main_text_title(doc, filtered.blocks)
    assert title is None


def test_publication_metadata_cannot_become_root_title():
    blocks = [_mk(ConferenceMetadataBlock, "b1", 0, text="Proceedings of ACM SIGOPS 2019", page=1)]
    doc = _doc(blocks, title="Proceedings of ACM SIGOPS 2019")
    filtered, _ = build_main_text_document(doc)
    title, source = resolve_main_text_title(doc, filtered.blocks)
    assert title is None


def test_bare_arxiv_id_metadata_title_cannot_become_root_title():
    """A "Print to PDF" of an arXiv page stamps the arXiv accession number
    into the PDF's /Title metadata (the source browser tab's title, not
    the paper's) -- real-world case found via the Bayesian Networks PDF
    acceptance test, where `document.metadata.title` came back as the bare
    string "1804.08137" with no document text to match it against
    verbatim, so the exact-block-text guard alone can't catch it."""
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="Real Article Title", page=1),
        _mk(ParagraphBlock, "b2", 1, text="Real body text.", page=1),
    ]
    doc = _doc(blocks, title="1804.08137")
    filtered, _ = build_main_text_document(doc)
    title, source = resolve_main_text_title(doc, filtered.blocks)
    assert title == "Real Article Title"
    assert source == "first_body_heading"


def test_bare_arxiv_id_with_version_suffix_cannot_become_root_title():
    doc = _doc([_mk(HeadingBlock, "b1", 0, level=1, text="Real Title", page=1)], title="1804.08137v2")
    filtered, _ = build_main_text_document(doc)
    title, source = resolve_main_text_title(doc, filtered.blocks)
    assert title == "Real Title"


def test_bare_doi_metadata_title_cannot_become_root_title():
    doc = _doc([_mk(HeadingBlock, "b1", 0, level=1, text="Real Title", page=1)], title="10.1145/1234567.1234568")
    filtered, _ = build_main_text_document(doc)
    title, source = resolve_main_text_title(doc, filtered.blocks)
    assert title == "Real Title"


def test_title_resembling_but_not_matching_arxiv_id_pattern_is_still_accepted():
    # A real title should never be false-positived just for containing digits/dots.
    doc = _doc([], title="Section 1804.08137 of the Handbook")
    filtered, _ = build_main_text_document(doc)
    title, source = resolve_main_text_title(doc, filtered.blocks)
    assert title == "Section 1804.08137 of the Handbook"


# ---------------------------------------------------------------------- #
# Abstract / Acknowledgements section exclusion (spec EXCLUDE "abstract",
# "acknowledgements") -- same stack-based subtree mechanism already proven
# for References/TOC, extended to two more named sections.

def test_excludes_abstract_section_and_its_body():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="Real Title", page=1),
        _mk(HeadingBlock, "b2", 1, level=2, text="Abstract", page=1),
        _mk(ParagraphBlock, "b3", 2, text="This paper studies something interesting in depth.", page=1),
        _mk(HeadingBlock, "b4", 3, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b5", 4, text="Real introduction body text.", page=1),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    texts = [getattr(b, "text", "") for b in filtered.blocks]
    assert "Abstract" not in texts
    assert "This paper studies something interesting in depth." not in texts
    assert "1 Introduction" in texts
    assert "Real introduction body text." in texts
    assert stats.excluded_abstract_blocks == 2  # heading + its one paragraph


def test_excludes_acknowledgements_section():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="Real Title", page=1),
        _mk(HeadingBlock, "b2", 1, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b3", 2, text="Real introduction body text.", page=1),
        _mk(HeadingBlock, "b4", 3, level=2, text="Acknowledgements", page=9),
        _mk(ParagraphBlock, "b5", 4, text="We thank our colleagues for their support.", page=9),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    texts = [getattr(b, "text", "") for b in filtered.blocks]
    assert "Acknowledgements" not in texts
    assert "We thank our colleagues for their support." not in texts
    assert "Real introduction body text." in texts
    assert stats.excluded_acknowledgements_blocks == 2


# ---------------------------------------------------------------------- #
# Junk-heading exclusion (spec "do not promote algorithms, equations,
# diagram labels, table headings, or visual-region text into document
# headings") -- reuses `heading_rejection_reason`, already trusted at
# extraction time and by validation.py's retrospective safety net.

def test_bare_identifier_heading_is_excluded_even_as_the_first_heading():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="1701.05226", page=1),
        _mk(HeadingBlock, "b2", 1, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b3", 2, text="Real introduction body text.", page=1),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    texts = [getattr(b, "text", "") for b in filtered.blocks]
    assert "1701.05226" not in texts
    assert "1 Introduction" in texts
    assert stats.excluded_junk_heading_blocks == 1


def test_equation_shaped_body_heading_is_excluded():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="Real Title", page=1),
        _mk(HeadingBlock, "b2", 1, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b3", 2, text="Real introduction body text.", page=1),
        _mk(HeadingBlock, "b4", 3, level=2, text="F(x) = x squared plus one", page=2),
        _mk(ParagraphBlock, "b5", 4, text="More real body text right after it.", page=2),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    texts = [getattr(b, "text", "") for b in filtered.blocks]
    assert "F(x) = x squared plus one" not in texts
    assert "More real body text right after it." in texts  # content itself is kept, just not the heading
    assert stats.excluded_junk_heading_blocks == 1


def test_short_real_title_is_not_treated_as_junk():
    # A real single/two-word title must not be swept up by body-heading
    # reasons (e.g. "ocr_fragment"'s all-words-too-short check) -- the title
    # has its own, narrower validity rules (spec, and validation.py's own
    # precedent of exempting level-1 from the general heading-quality scan).
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="T", page=1),
        _mk(ParagraphBlock, "b2", 1, text="Real body text.", page=1),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    texts = [getattr(b, "text", "") for b in filtered.blocks]
    assert "T" in texts
    assert stats.excluded_junk_heading_blocks == 0


# ---------------------------------------------------------------------- #
# Front-matter zone (spec EXCLUDE "arXiv identifier and header", "abstract",
# "authors", "affiliations", "email addresses" -- caught even when the
# extractor left them as plain, untyped Paragraph/Unknown blocks instead of
# a dedicated front-matter block type) and title-region reconstruction
# (spec "reconstruct the title from adjacent title regions").

def test_arxiv_stamp_line_is_excluded_from_body():
    blocks = [
        _mk(ParagraphBlock, "b1", 0, text="1701.05226v2 [cs.AI] 1 Mar 2017 arXiv", page=1),
        _mk(ParagraphBlock, "b2", 1, text="A Wrapped Title Line About Reasoning", page=1),
        _mk(HeadingBlock, "b3", 2, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b4", 3, text="Real introduction body text.", page=1),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    texts = [getattr(b, "text", "") for b in filtered.blocks]
    assert "1701.05226v2 [cs.AI] 1 Mar 2017 arXiv" not in texts
    assert "A Wrapped Title Line About Reasoning" not in texts  # became the title, not a body paragraph
    assert stats.excluded_front_matter_zone_blocks == 2
    assert stats.title_region_text == "A Wrapped Title Line About Reasoning"


def test_reserved_label_mid_line_paragraph_is_excluded():
    blocks = [
        _mk(ParagraphBlock, "b1", 0,
            text="Dept. of Computer Science, City University London Abstract. This paper studies uncertainty in depth "
                 "across several different sections and examples drawn from the literature.",
            page=1),
        _mk(HeadingBlock, "b2", 1, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b3", 2, text="Real introduction body text.", page=1),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    texts = [getattr(b, "text", "") for b in filtered.blocks]
    assert not any("Abstract." in t for t in texts)
    assert not any("City University London" in t for t in texts)
    assert "Real introduction body text." in texts


def test_untyped_numbered_section_start_ends_the_zone_and_is_kept():
    # A real numbered section heading that never got promoted to
    # HeadingBlock (extraction gap) must still end the front-matter zone --
    # otherwise real body content silently gets swept up as "front matter."
    blocks = [
        _mk(ParagraphBlock, "b1", 0, text="Some Wrapped Title Fragment", page=1),
        _mk(UnknownBlock, "b2", 1, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b3", 2, text="Real introduction body text that must never be dropped.", page=1),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    texts = [getattr(b, "text", "") for b in filtered.blocks]
    assert "1 Introduction" in texts
    assert "Real introduction body text that must never be dropped." in texts


def test_comma_bearing_short_fragment_excluded_but_not_folded_into_title():
    # An author-list-shaped line the shared classifier happens to miss
    # (unusual punctuation) must still not pollute the reconstructed title --
    # a bare comma is a strong, generic signal it's not a title line. Needs
    # a prior strong front-matter signal (here, the arXiv stamp) for either
    # short fragment to be considered part of the zone at all -- otherwise a
    # short comma-bearing sentence right after the title is at least as
    # likely to just be the start of ordinary body prose.
    blocks = [
        _mk(ParagraphBlock, "b1", 0, text="1701.05226v2 [cs.AI] 1 Mar 2017 arXiv", page=1),
        _mk(ParagraphBlock, "b2", 1, text="A Real Title Fragment Here", page=1),
        _mk(ParagraphBlock, "b3", 2, text="Jane X, John Y, Amit Z", page=1),
        _mk(HeadingBlock, "b4", 3, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b5", 4, text="Real introduction body text.", page=1),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    texts = [getattr(b, "text", "") for b in filtered.blocks]
    assert "Jane X, John Y, Amit Z" not in texts
    assert stats.title_region_text == "A Real Title Fragment Here"


def test_ordinary_document_with_real_h1_and_intro_leaves_zone_untouched():
    # The common, well-formed case (real H1 immediately followed by a real
    # section) must produce zero front-matter-zone exclusions.
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="A Perfectly Normal Title", page=1),
        _mk(HeadingBlock, "b2", 1, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b3", 2, text="Real introduction body text.", page=1),
    ]
    filtered, stats = build_main_text_document(_doc(blocks))
    assert stats.excluded_front_matter_zone_blocks == 0
    assert stats.title_region_text is None


# ---------------------------------------------------------------------- #
# Title + subtitle reconstruction (spec "support a main title plus
# subtitle", "reconstruct the title from adjacent title regions").

def test_reconstructs_title_from_wrapped_paragraph_fragments():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="1701.05226", page=1),
        _mk(ParagraphBlock, "b2", 1, text="1701.05226v2 [cs.AI] 1 Mar 2017 arXiv", page=1),
        _mk(ParagraphBlock, "b3", 2, text="A Wrapped Title About Reasoning", page=1),
        _mk(ParagraphBlock, "b4", 3, text="In Uncertain Domains", page=1),
        _mk(HeadingBlock, "b5", 4, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b6", 5, text="Real introduction body text.", page=1),
    ]
    doc = _doc(blocks, title="1701.05226")
    filtered, stats = build_main_text_document(doc)
    title, source = resolve_main_text_title(doc, filtered.blocks, stats.title_region_text)
    assert title == "A Wrapped Title About Reasoning In Uncertain Domains"
    assert source == "reconstructed_title_region"


def test_reconstructs_main_title_and_subtitle_combination():
    # Real-world shape found via the Bayesian Networks PDF: the actual
    # subtitle survived extraction as the document's first (real, non-junk)
    # heading, while the main title landed as a separate, un-headinged
    # paragraph further down, past an arXiv stamp line.
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="Computational Complexity and Efficient Implementation", page=1),
        _mk(ParagraphBlock, "b2", 1, text="1804.08137v2 [stat.CO] 30 Nov 2018 arXiv", page=1),
        _mk(ParagraphBlock, "b3", 2, text="Learning Bayesian Networks from Big Data with Greedy Search", page=1),
        _mk(HeadingBlock, "b4", 3, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b5", 4, text="Real introduction body text.", page=1),
    ]
    doc = _doc(blocks, title="1804.08137")
    filtered, stats = build_main_text_document(doc)
    title, source = resolve_main_text_title(doc, filtered.blocks, stats.title_region_text)
    assert title == (
        "Learning Bayesian Networks from Big Data with Greedy Search: "
        "Computational Complexity and Efficient Implementation"
    )
    assert source == "reconstructed_title_and_subtitle"


def test_metadata_title_still_wins_over_reconstruction_when_usable():
    blocks = [
        _mk(ParagraphBlock, "b1", 0, text="A Wrapped Title Fragment", page=1),
        _mk(HeadingBlock, "b2", 1, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b3", 2, text="Real introduction body text.", page=1),
    ]
    doc = _doc(blocks, title="The Real Metadata Title")
    filtered, stats = build_main_text_document(doc)
    title, source = resolve_main_text_title(doc, filtered.blocks, stats.title_region_text)
    assert title == "The Real Metadata Title"
    assert source == "metadata"


# ---------------------------------------------------------------------- #
# apply_resolved_title (spec "the resulting Markdown must begin with the
# resolved title") -- the renderer has no special handling of "the title"
# at all, it just walks `active_document.blocks` in order, so title
# resolution must be synced back into the block list, not just returned as
# a separate string.

def test_apply_resolved_title_synthesizes_h1_when_none_present():
    blocks = [_mk(ParagraphBlock, "b1", 0, text="Real body text.", page=1)]
    doc = _doc(blocks)
    apply_resolved_title(doc, "The Real Title")
    assert isinstance(doc.blocks[0], HeadingBlock)
    assert doc.blocks[0].level == 1
    assert doc.blocks[0].text == "The Real Title"
    assert render_markdown(doc).startswith("# The Real Title\n")


def test_apply_resolved_title_deduplicates_matching_body_heading():
    # title came from metadata/reconstruction but a body heading with the
    # exact same text also survived filtering -- must not render twice.
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b2", 1, text="Real introduction body text.", page=1),
        _mk(HeadingBlock, "b3", 2, level=1, text="The Real Title", page=1),
    ]
    doc = _doc(blocks)
    apply_resolved_title(doc, "The Real Title")
    h1_count = sum(1 for b in doc.blocks if isinstance(b, HeadingBlock) and b.text == "The Real Title")
    assert h1_count == 1
    assert doc.blocks[0].text == "The Real Title"
    assert doc.blocks[1].text == "1 Introduction"  # everything else stays in place


def test_apply_resolved_title_is_a_noop_when_already_first():
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="The Real Title", page=1),
        _mk(ParagraphBlock, "b2", 1, text="Real body text.", page=1),
    ]
    doc = _doc(blocks)
    original_first = doc.blocks[0]
    apply_resolved_title(doc, "The Real Title")
    assert doc.blocks[0] is original_first  # untouched, not replaced with a new block
    assert len(doc.blocks) == 2


def test_apply_resolved_title_is_noop_for_falsy_title():
    blocks = [_mk(ParagraphBlock, "b1", 0, text="Real body text.", page=1)]
    doc = _doc(blocks)
    apply_resolved_title(doc, None)
    assert doc.blocks == blocks


def test_full_pipeline_markdown_begins_with_resolved_reconstructed_title():
    """End-to-end (still hand-built blocks, no PDF): the exact real-world
    shape found via the Bayesian Networks PDF acceptance test -- a
    reconstructed main title + subtitle, with the actual subtitle heading
    surviving deep in `document.blocks` (not at position 0) -- must still
    end up as the Markdown's opening H1."""
    blocks = [
        _mk(HeadingBlock, "b1", 0, level=1, text="Computational Complexity and Efficient Implementation", page=1),
        _mk(ParagraphBlock, "b2", 1, text="1804.08137v2 [stat.CO] 30 Nov 2018 arXiv", page=1),
        _mk(ParagraphBlock, "b3", 2, text="Learning Bayesian Networks from Big Data with Greedy Search", page=1),
        _mk(HeadingBlock, "b4", 3, level=2, text="1 Introduction", page=1),
        _mk(ParagraphBlock, "b5", 4, text="Real introduction body text.", page=1),
    ]
    doc = _doc(blocks, title="1804.08137")
    filtered, stats = build_main_text_document(doc)
    title, source = resolve_main_text_title(doc, filtered.blocks, stats.title_region_text)
    apply_resolved_title(filtered, title)
    md = render_markdown(filtered)
    assert md.startswith(
        "# Learning Bayesian Networks from Big Data with Greedy Search: "
        "Computational Complexity and Efficient Implementation\n"
    )
    assert "## 1 Introduction" in md
    assert "1804.08137" not in md

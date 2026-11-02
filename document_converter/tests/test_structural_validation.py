"""
Structural over/under-segmentation validation (spec section 16-17, 21):
empty vs. tiny vs. parent-container sections, the suspicious/hard-reject
heading safety net, and heading-hierarchy invariants (single H1, no skipped
levels). Uses raw Markdown source through the same parse -> normalize ->
chunk -> validate pipeline the other validation tests use, since heading
hierarchy/section structure is format-agnostic once it reaches this layer.
"""
from __future__ import annotations

from document_converter.md_parser import parse_markdown
from document_converter.normalize import normalize_document
from document_converter.chunker import MarkdownChunker
from document_converter.config import ChunkingConfig
from document_converter.validation import validate
from document_converter.findings import FindingLog
from document_converter.model import (
    Document, DocumentMetadata, HeadingBlock, ParagraphBlock, ImageBlock, CodeBlock, FigureLabelBlock,
    CaptionBlock,
)


def _validated(text, config=None):
    config = config or ChunkingConfig()
    doc, extract_log = parse_markdown(text, source_path="x.md")
    norm_log = normalize_document(doc)
    upstream = FindingLog()
    upstream.extend(extract_log)
    upstream.extend(norm_log)
    result = MarkdownChunker().chunk(doc, "x.md", config, document_id="d1")
    report = validate(doc, result, config, "x.md", "x_normalized.md", "markdown", False, upstream)
    return doc, report


def _validate_document(doc: Document, config=None):
    """Runs a manually-constructed Document (bypassing markdown parsing, for
    tests that need precise block adjacency/type control) through the same
    chunk -> validate pipeline."""
    config = config or ChunkingConfig()
    result = MarkdownChunker().chunk(doc, "x.md", config, document_id="d1")
    report = validate(doc, result, config, "x.md", "x_normalized.md", "markdown", False, FindingLog())
    return report


def _mk(cls, block_id, order, **kw):
    return cls(block_id=block_id, order=order, source_format="markdown", **kw)


# ---------------------------------------------------------------------- #
# H1 invariants

def test_exactly_one_h1_for_normal_document():
    doc, report = _validated("# Title\n\nSome content.\n\n## Section\n\nMore content here.\n")
    h1s = [b for b in doc.blocks if isinstance(b, HeadingBlock) and b.level == 1]
    assert len(h1s) == 1
    assert report.stats["headers"]["h1"] == 1


def test_multiple_h1_headings_get_demoted_by_normalize():
    doc, report = _validated("# First Title\n\nContent.\n\n# Second Title\n\nMore content.\n")
    h1s = [b for b in doc.blocks if isinstance(b, HeadingBlock) and b.level == 1]
    assert len(h1s) == 1
    # normalize.py must have logged the demotion (repair, not a silent drop).
    assert report.stats["repairs"] > 0


# ---------------------------------------------------------------------- #
# Empty vs. tiny vs. parent-container sections

def test_sibling_heading_with_no_body_is_flagged_empty():
    text = (
        "# Title\n\nIntro paragraph with enough words to not be tiny itself here now.\n\n"
        "## Empty Section\n\n## Next Section\n\nThis section actually has real body content in it.\n"
    )
    _, report = _validated(text)
    assert any(f.code == "empty_section" for f in report.findings.findings)
    assert report.stats["empty_section_count"] >= 1


def test_parent_heading_with_only_child_sections_is_not_empty():
    text = (
        "# Title\n\nIntro paragraph with enough words to not be tiny itself here now.\n\n"
        "## Parent Section\n\n"
        "### Child Section\n\nThe parent has no direct body text, only this child section's content.\n"
    )
    _, report = _validated(text)
    empty_codes = [f for f in report.findings.findings if f.code == "empty_section"]
    assert not any("Parent Section" in f.message for f in empty_codes)


def test_excessive_tiny_sections_fail_structural_readiness():
    # 12 leaf sections, each with a one-word body (well under the 10-word
    # tiny threshold) -- comfortably above both the minimum leaf count and
    # the ratio gate.
    parts = ["# Title\n\nIntro paragraph with enough words to not be tiny itself here now.\n"]
    for i in range(12):
        parts.append(f"\n## Section {i}\n\nshort.\n")
    _, report = _validated("".join(parts))
    assert report.stats["tiny_section_ratio"] > 0
    assert any(f.code == "excessive_tiny_sections" for f in report.findings.findings)
    assert report.stats["structural_readiness"] is False


def test_few_tiny_sections_do_not_fail_short_documents():
    text = "# Title\n\n## Only Section\n\nshort.\n"
    _, report = _validated(text)
    assert not any(f.code == "excessive_tiny_sections" for f in report.findings.findings)


# ---------------------------------------------------------------------- #
# Retrospective heading safety net (catches bad headings regardless of
# source format / extractor, e.g. a Word "Heading 2" style applied to code)

def test_code_heading_caught_by_retrospective_safety_net():
    text = (
        "# Title\n\nIntro paragraph with enough words to not be tiny itself here now.\n\n"
        "## LinearSearch(numbers, key) {\n\nBody text.\n"
    )
    _, report = _validated(text)
    assert any(f.code == "code_promoted_to_heading" for f in report.findings.findings)
    assert report.stats["code_like_heading_count"] >= 1
    assert report.stats["structural_readiness"] is False


def test_ordinary_headings_do_not_trigger_safety_net():
    text = "# Title\n\n## Scope\n\nBody text about scope.\n\n## Risks\n\nBody text about risks.\n"
    _, report = _validated(text)
    assert report.stats["code_like_heading_count"] == 0
    assert report.stats["equation_like_heading_count"] == 0
    assert report.stats["fragment_heading_count"] == 0


def test_caption_heading_caught_by_retrospective_safety_net():
    text = (
        "# Title\n\nIntro paragraph with enough words to not be tiny itself here now.\n\n"
        "## Table 1: An LCP array\n\nBody text describing the table.\n"
    )
    _, report = _validated(text)
    assert any(f.code == "caption_promoted_to_heading" for f in report.findings.findings)
    assert report.stats["caption_heading_count"] >= 1
    assert report.stats["structural_readiness"] is False


# ---------------------------------------------------------------------- #
# Fix 4: remaining_html must not scan CodeBlock content, and must
# aggregate repeated identical tags outside code into one finding

def _doc_with_blocks(blocks):
    doc = Document(metadata=DocumentMetadata(source_path="x.md", source_format="markdown"))
    doc.blocks = blocks
    return doc


def test_remaining_html_ignores_tags_inside_code_blocks():
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="Intro paragraph with enough words to not be tiny at all here.",
            normalized="Intro paragraph with enough words to not be tiny at all here."),
        _mk(HeadingBlock, "b0003", 2, level=2, text="DOM Example", normalized="DOM Example"),
        _mk(CodeBlock, "b0004", 3, language="html",
            code="<html>\n<head><title>Example</title></head>\n<body></body>\n</html>",
            normalized="<html>\n<head><title>Example</title></head>\n<body></body>\n</html>"),
    ]
    report = _validate_document(_doc_with_blocks(blocks))
    assert not any(f.code == "remaining_html" for f in report.findings.findings)


def test_remaining_html_aggregates_repeated_identical_tags_outside_code():
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="A paragraph mentioning a <config> placeholder inline for context here.",
            normalized="A paragraph mentioning a <config> placeholder inline for context here."),
        _mk(HeadingBlock, "b0003", 2, level=2, text="Section Two", normalized="Section Two"),
        _mk(ParagraphBlock, "b0004", 3, text="Another paragraph that again references a <config> placeholder here too.",
            normalized="Another paragraph that again references a <config> placeholder here too."),
        _mk(HeadingBlock, "b0005", 4, level=2, text="Section Three", normalized="Section Three"),
        _mk(ParagraphBlock, "b0006", 5, text="A third mention of the same <config> placeholder appears in this text.",
            normalized="A third mention of the same <config> placeholder appears in this text."),
    ]
    report = _validate_document(_doc_with_blocks(blocks))
    html_findings = [f for f in report.findings.findings if f.code == "remaining_html"]
    assert len(html_findings) == 1, html_findings
    assert "3 places" in html_findings[0].message


# ---------------------------------------------------------------------- #
# Fix 5: retrospective, whole-document context checks

def test_heading_embedded_in_caption_still_flagged_suspicious():
    """A heading whose own text is chrome/fragment-shaped (fails
    `looks_like_title_candidate` -- here a trailing-colon lead-in, the same
    shape a caption fragment or a stray label out of a figure region would
    have) sitting next to a caption is exactly the "false heading spliced
    into the caption stream" case detection must not lose: adjacency plus
    a genuinely suspicious heading shape."""
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="Intro paragraph with enough words to not be tiny at all here.",
            normalized="Intro paragraph with enough words to not be tiny at all here."),
        _mk(ParagraphBlock, "b0003", 2, text="Figure 2: Overview of the pipeline architecture",
            normalized="Figure 2: Overview of the pipeline architecture"),
        _mk(HeadingBlock, "b0004", 3, level=2, text="See below:", normalized="See below:"),
        _mk(ParagraphBlock, "b0005", 4, text="This label describes the ground-truth values used for evaluation here.",
            normalized="This label describes the ground-truth values used for evaluation here."),
    ]
    report = _validate_document(_doc_with_blocks(blocks))
    suspicious = [f for f in report.findings.findings if f.code == "suspicious_heading"]
    assert any("adjacent_to_figure_or_caption" in f.message for f in suspicious), suspicious


def test_heading_incorrectly_inserted_before_image_still_flagged_suspicious():
    """Same "chrome-shaped text + adjacency" true positive as the caption
    case above, but for a heading landing directly in front of an
    `ImageBlock` -- the "heading incorrectly inserted between a figure and
    its caption" shape."""
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="Intro paragraph with enough words to not be tiny at all here.",
            normalized="Intro paragraph with enough words to not be tiny at all here."),
        _mk(HeadingBlock, "b0003", 2, level=2, text="Continue", normalized="Continue"),
        _mk(ImageBlock, "b0004", 3, alt="chart", src="chart.png"),
    ]
    report = _validate_document(_doc_with_blocks(blocks))
    suspicious = [f for f in report.findings.findings if f.code == "suspicious_heading"]
    assert any("adjacent_to_figure_or_caption" in f.message for f in suspicious), suspicious


def test_legitimate_heading_adjacent_to_bare_image_not_flagged_suspicious():
    """A real, descriptive heading (passes `looks_like_title_candidate`)
    sitting next to a bare/uncaptioned `ImageBlock` (no alt text -- an icon,
    not a figure) must not be flagged: coincidental proximity to a
    contentless image is not, by itself, evidence anything is wrong. Applies
    to any parser that can emit a bare `ImageBlock` next to a real heading
    (Docling's finer-grained picture segmentation triggered this in
    practice; the check itself is parser-agnostic, operating only on the
    common `Document`/`Block` model)."""
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="Intro paragraph with enough words to not be tiny at all here.",
            normalized="Intro paragraph with enough words to not be tiny at all here."),
        _mk(ImageBlock, "b0003", 2, alt="", src="icon.png"),
        _mk(HeadingBlock, "b0004", 3, level=2, text="Recursion Trees", normalized="Recursion Trees"),
        _mk(ParagraphBlock, "b0005", 4, text="This section explains how recursion trees model call depth here.",
            normalized="This section explains how recursion trees model call depth here."),
    ]
    report = _validate_document(_doc_with_blocks(blocks))
    suspicious = [f for f in report.findings.findings if f.code == "suspicious_heading"]
    assert not suspicious, suspicious


def test_legitimate_heading_near_paired_figure_and_caption_not_flagged_suspicious():
    """A real heading that legitimately introduces a labeled figure (or
    starts right after one ends) must not be penalized merely for sitting
    next to it -- the same "coincidental proximity in an illustrated
    document is not evidence" principle already applied to numbered
    headings, extended to any heading whose own text is not chrome-shaped."""
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="Intro paragraph with enough words to not be tiny at all here.",
            normalized="Intro paragraph with enough words to not be tiny at all here."),
        _mk(HeadingBlock, "b0003", 2, level=2, text="Common Big O Complexities", normalized="Common Big O Complexities"),
        _mk(ImageBlock, "b0004", 3, alt="Chart of growth rates for common complexity classes", src="chart.png"),
        _mk(CaptionBlock, "b0005", 4, text="Figure 3: Growth rates for common complexity classes",
            normalized="Figure 3: Growth rates for common complexity classes"),
        _mk(ParagraphBlock, "b0006", 5, text="Many commonly used algorithms have running times in one of these classes.",
            normalized="Many commonly used algorithms have running times in one of these classes."),
    ]
    report = _validate_document(_doc_with_blocks(blocks))
    suspicious = [f for f in report.findings.findings if f.code == "suspicious_heading"]
    assert not suspicious, suspicious


def test_heading_with_high_symbol_density_flagged_suspicious():
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="Intro paragraph with enough words to not be tiny at all here.",
            normalized="Intro paragraph with enough words to not be tiny at all here."),
        _mk(HeadingBlock, "b0003", 2, level=2, text="12/34/56/78 Data", normalized="12/34/56/78 Data"),
        _mk(ParagraphBlock, "b0004", 3, text="Body text following the numeric-heavy heading for this section here.",
            normalized="Body text following the numeric-heavy heading for this section here."),
    ]
    report = _validate_document(_doc_with_blocks(blocks))
    suspicious = [f for f in report.findings.findings if f.code == "suspicious_heading"]
    assert any("high_symbol_or_digit_density" in f.message for f in suspicious), suspicious


def test_heading_interrupting_a_sentence_flagged_suspicious():
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="The system was evaluated using a benchmark and the results were as follows",
            normalized="The system was evaluated using a benchmark and the results were as follows"),
        _mk(HeadingBlock, "b0003", 2, level=2, text="Interruption", normalized="Interruption"),
        _mk(ParagraphBlock, "b0004", 3, text="continues here with the rest of the same sentence after the break.",
            normalized="continues here with the rest of the same sentence after the break."),
    ]
    report = _validate_document(_doc_with_blocks(blocks))
    suspicious = [f for f in report.findings.findings if f.code == "suspicious_heading"]
    assert any("interrupts_sentence" in f.message for f in suspicious), suspicious


def test_interrupts_sentence_ignores_repeated_page_furniture_block():
    """A running watermark/page-stamp string that recurs verbatim 3+ times
    across the document (here as the block immediately before two different
    headings) is not a real sentence a heading could plausibly interrupt --
    comparing across it produces a false positive out of two unrelated real
    paragraphs the watermark merely sits between. This must not fire even
    though the punctuation/case shape (`prev` has no terminal punctuation,
    `next` starts with a lowercase letter) otherwise matches."""
    furniture = "zyBooks User CSC000-1_1"
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="Intro paragraph with enough words to not be tiny at all here.",
            normalized="Intro paragraph with enough words to not be tiny at all here."),
        _mk(ParagraphBlock, "b0003", 2, text=furniture, normalized=furniture),
        _mk(HeadingBlock, "b0004", 3, level=2, text="Recursion Trees", normalized="Recursion Trees"),
        _mk(ParagraphBlock, "b0005", 4, text="continues here with unrelated real prose after the watermark line.",
            normalized="continues here with unrelated real prose after the watermark line."),
        _mk(ParagraphBlock, "b0006", 5, text="More body content so this section is not flagged tiny or empty here.",
            normalized="More body content so this section is not flagged tiny or empty here."),
        _mk(ParagraphBlock, "b0007", 6, text=furniture, normalized=furniture),
        _mk(HeadingBlock, "b0008", 7, level=2, text="Growth Rates", normalized="Growth Rates"),
        _mk(ParagraphBlock, "b0009", 8, text="continues again with more unrelated real prose right here too.",
            normalized="continues again with more unrelated real prose right here too."),
        _mk(ParagraphBlock, "b0010", 9, text=furniture, normalized=furniture),
    ]
    report = _validate_document(_doc_with_blocks(blocks))
    suspicious = [f for f in report.findings.findings if f.code == "suspicious_heading"]
    assert not any("interrupts_sentence" in f.message for f in suspicious), suspicious


def test_bodyless_same_level_heading_chain_flagged_suspicious():
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="Intro paragraph with enough words to not be tiny at all here.",
            normalized="Intro paragraph with enough words to not be tiny at all here."),
        _mk(HeadingBlock, "b0003", 2, level=2, text="First Fake Heading", normalized="First Fake Heading"),
        _mk(HeadingBlock, "b0004", 3, level=2, text="Second Fake Heading", normalized="Second Fake Heading"),
        _mk(ParagraphBlock, "b0005", 4, text="Real content finally appears here under the second heading only.",
            normalized="Real content finally appears here under the second heading only."),
    ]
    report = _validate_document(_doc_with_blocks(blocks))
    suspicious = [f for f in report.findings.findings if f.code == "suspicious_heading"]
    assert any("bodyless_heading_chain" in f.message for f in suspicious), suspicious


def test_normal_headings_do_not_trigger_context_checks():
    text = (
        "# Title\n\n## Scope\n\nBody text about scope that reads as an ordinary paragraph here.\n\n"
        "## Risks\n\nBody text about risks that reads as an ordinary paragraph as well.\n"
    )
    _, report = _validated(text)
    suspicious = [f for f in report.findings.findings if f.code == "suspicious_heading"]
    assert not suspicious, suspicious


# ---------------------------------------------------------------------- #
# OCR heading evidence metrics/gates (spec "Validation"): a figure/diagram/
# table-cell-shaped heading that somehow survives both extraction-time
# prevention and the repair pass must be caught here, retrospectively, same
# as the existing code/equation/caption hard-reject categories.

def test_figure_label_heading_fails_the_hard_gate_even_if_not_from_ocr():
    """The retrospective check is format-agnostic: any HeadingBlock whose
    text has a figure/diagram/table-cell shape trips the gate, regardless of
    which extractor or repair stage should have caught it earlier."""
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="Intro paragraph with enough words to not be tiny at all here.",
            normalized="Intro paragraph with enough words to not be tiny at all here."),
        _mk(HeadingBlock, "b0003", 2, level=2, text="Input Image", normalized="Input Image"),
        _mk(ParagraphBlock, "b0004", 3, text="Body text describing the input image that follows this heading.",
            normalized="Body text describing the input image that follows this heading."),
    ]
    report = _validate_document(_doc_with_blocks(blocks))
    assert report.stats["figure_labels_promoted_to_heading"] == 1
    assert any(f.code == "unsupported_ocr_heading" for f in report.findings.findings)
    assert report.stats["structural_readiness"] is False


def test_native_diagram_annotation_shaped_heading_is_not_flagged():
    """Regression for the D5 finding (artifacts/heading_classifier_step1/D5.md,
    2026-09-14): `looks_like_diagram_annotation` is a pure text-shape
    classifier, and on native-path headings -- which carry no OCR provenance
    (`from_ocr`/`ocr_heading_score` style keys are only persisted by
    pdf_extractor.py's OCR path) -- it produced nothing but false positives
    on legitimate subsection headings ("Values", "Counts", "Dynamic errors").
    A native heading with that text shape must skip the check entirely."""
    for text in ("Values", "Counts"):
        blocks = [
            _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
            _mk(ParagraphBlock, "b0002", 1, text="Intro paragraph with enough words to not be tiny at all here.",
                normalized="Intro paragraph with enough words to not be tiny at all here."),
            _mk(HeadingBlock, "b0003", 2, level=2, text=text, normalized=text),
            _mk(ParagraphBlock, "b0004", 3, text="Body text describing the quantity table that follows this heading.",
                normalized="Body text describing the quantity table that follows this heading."),
        ]
        report = _validate_document(_doc_with_blocks(blocks))
        assert report.stats["diagram_annotations_promoted_to_heading"] == 0, text
        assert not any(f.code == "unsupported_ocr_heading" for f in report.findings.findings), text
        assert report.stats["structural_readiness"] is True, text


def test_ocr_diagram_annotation_shaped_heading_is_still_flagged():
    """The gate must not defang the check where it belongs: an OCR-path
    heading (carrying the `from_ocr`/`ocr_heading_score` evidence
    pdf_extractor.py persists) whose text has a diagram-annotation shape
    still trips the hard gate exactly as before."""
    heading = _mk(HeadingBlock, "b0003", 2, level=2, text="Values", normalized="Values")
    heading.style["from_ocr"] = True
    heading.style["ocr_heading_score"] = 5
    heading.style["ocr_heading_signals"] = ["font_size_jump"]
    heading.style["ocr_heading_negatives"] = []
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="Intro paragraph with enough words to not be tiny at all here.",
            normalized="Intro paragraph with enough words to not be tiny at all here."),
        heading,
        _mk(ParagraphBlock, "b0004", 3, text="Body text describing the quantity table that follows this heading.",
            normalized="Body text describing the quantity table that follows this heading."),
    ]
    report = _validate_document(_doc_with_blocks(blocks))
    assert report.stats["diagram_annotations_promoted_to_heading"] == 1
    assert any(f.code == "unsupported_ocr_heading" for f in report.findings.findings)
    assert report.stats["structural_readiness"] is False


def test_clean_document_reports_zero_ocr_heading_gates():
    text = "# Title\n\n## Scope\n\nBody text about scope that reads as an ordinary paragraph here.\n"
    _, report = _validated(text)
    for key in (
        "figure_labels_promoted_to_heading", "diagram_annotations_promoted_to_heading",
        "table_cells_promoted_to_heading", "unsupported_ocr_headings_remaining",
    ):
        assert report.stats[key] == 0, key
    assert report.stats["structural_readiness"] is True


def test_typed_ocr_block_counts_as_atomic_section_content():
    """A heading followed only by a FigureLabelBlock (no ParagraphBlock at
    all) must not be flagged as an empty section -- spec: 'Atomic child
    content counts as meaningful section content,' and the OCR typed blocks
    are exactly that kind of atomic content."""
    blocks = [
        _mk(HeadingBlock, "b0001", 0, level=1, text="Title", normalized="Title"),
        _mk(ParagraphBlock, "b0002", 1, text="Intro paragraph with enough words to not be tiny at all here.",
            normalized="Intro paragraph with enough words to not be tiny at all here."),
        _mk(HeadingBlock, "b0003", 2, level=2, text="Diagram", normalized="Diagram"),
        _mk(FigureLabelBlock, "b0004", 3, text="Input Image", normalized="Input Image"),
    ]
    report = _validate_document(_doc_with_blocks(blocks))
    empty_codes = [f for f in report.findings.findings if f.code == "empty_section"]
    assert not any("Diagram" in f.message for f in empty_codes)

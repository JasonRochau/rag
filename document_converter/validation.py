"""
Validation and reporting (spec section 10).

Aggregates findings produced during parsing/normalization/chunking with the
document-level checks that can only be made once the whole picture is
available (paragraph length, leaf-section size, HTML remnants, source-block
coverage), and produces the statistics block used in reports and API
responses.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .model import (
    Document, HeadingBlock, ParagraphBlock, ListBlock, TableBlock, CodeBlock, QuoteBlock,
    ImageBlock, WarningBlock, UnknownBlock, CaptionBlock, FigureLabelBlock,
    DiagramAnnotationBlock, TableCellBlock,
    AuthorBlock, AffiliationBlock, ContactBlock, KeywordBlock, FRONT_MATTER_BLOCK_TYPES,
)
from .chunker import ChunkingResult, build_sections, Section
from .findings import FindingLog, Severity
from .config import ChunkingConfig
from .extractors import (
    heading_rejection_reason, looks_like_caption_text, looks_like_title_candidate,
    classify_ocr_nonheading_kind, OCR_HEADING_CONFIDENT_SCORE, numbering_tuple,
)
from .front_matter import classify_front_matter_line, is_references_heading

_PAGE_FURNITURE_REPEAT_THRESHOLD = 3


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

_AUTHOR_LIKE_FRONT_MATTER_TYPES = (AuthorBlock, AffiliationBlock, ContactBlock, KeywordBlock)
# A section-label line ("Abstract", "Keywords") is never a legitimate
# document title (spec section 3's `abstract_promoted_to_h1` gate) --
# checked defensively here even though `pdf_extractor.py`'s title
# resolver and `_resolve_heading_level`'s title-candidate guard already
# keep this from happening at extraction time, since this validator must
# independently back the gate for any source format/path, not just PDF.
_TITLE_BLOCKING_LABELS = {"abstract", "keywords", "index terms"}

_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_HTML_TAG_RE = re.compile(r"<([a-zA-Z!/][^<>\n]{0,60})>")
# The OCR-derived typed label/caption/annotation blocks (spec "Typed block
# handling") are real, searchable content -- they count toward source
# coverage the same as any other content block, and they count as atomic
# section content the same as a list/table/code/image (spec "A heading
# followed by an atomic figure, table, equation, code block, or list counts
# as meaningful section content").
_OCR_TYPED_BLOCK_TYPES = (CaptionBlock, FigureLabelBlock, DiagramAnnotationBlock, TableCellBlock)
_CONTENT_BLOCK_TYPES = (
    ParagraphBlock, ListBlock, TableBlock, CodeBlock, QuoteBlock, ImageBlock, WarningBlock,
    UnknownBlock,
) + _OCR_TYPED_BLOCK_TYPES
_ATOMIC_BLOCK_TYPES = (ListBlock, TableBlock, CodeBlock, ImageBlock) + _OCR_TYPED_BLOCK_TYPES

# Maps a `heading_rejection_reason` code to the specific spec section-21
# finding code for a heading that should never have been promoted at all --
# this is a retrospective, format-agnostic safety net independent of
# whichever extractor produced the heading (PDF font-size buckets, DOCX
# "Heading N" style, OCR isolated-line heuristic, ...), since none of those
# extraction paths can be fully trusted to keep code/equations/page-numbers/
# fragments out of a heading on their own (spec section 9.5: even a reliable
# Word heading style doesn't excuse code pasted into it).
_HARD_REJECT_FINDING_CODES = {
    "code_or_algorithm": "code_promoted_to_heading",
    "equation_or_matrix": "equation_promoted_to_heading",
    "no_letters": "page_number_promoted_to_heading",
    "punctuation_only": "ocr_fragment_promoted_to_heading",
    "ocr_fragment": "ocr_fragment_promoted_to_heading",
    "figure_or_table_caption": "caption_promoted_to_heading",
}


@dataclass
class ValidationReport:
    findings: FindingLog
    stats: Dict[str, Any]
    retrieval_ready: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"stats": self.stats, "findings": self.findings.to_list(), "retrieval_ready": self.retrieval_ready}

    def to_human_text(self) -> str:
        lines = [f"# Validation report: {self.stats.get('source_file', '')}", ""]
        lines.append("## Summary")
        for k, v in self.stats.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("## Findings")
        if not self.findings.findings:
            lines.append("(none)")
        for f in self.findings.findings:
            loc = []
            if f.block_id:
                loc.append(f"block={f.block_id}")
            if f.line:
                loc.append(f"line={f.line}")
            if f.page:
                loc.append(f"page={f.page}")
            loc_s = f" [{', '.join(loc)}]" if loc else ""
            lines.append(f"- **{f.severity.value}** `{f.code}`{loc_s}: {f.message}")
            if f.suggested_action:
                lines.append(f"  - suggested action: {f.suggested_action}")
        return "\n".join(lines) + "\n"


def find_html_remnants(text: str) -> List[str]:
    stripped = _INLINE_CODE_RE.sub(" ", text or "")
    return [m.group(0) for m in _HTML_TAG_RE.finditer(stripped)]


def _validate_paragraph_lengths(document: Document, config: ChunkingConfig, log: FindingLog) -> None:
    for b in document.blocks:
        if not isinstance(b, ParagraphBlock):
            continue
        wc = b.word_count
        if wc > config.hard_max_words * 0.85:
            log.warning(
                "paragraph_approaching_hard_limit",
                f"Paragraph ({wc} words) is approaching the hard chunk budget ({config.hard_max_words} words)",
                block_id=b.block_id,
            )
        elif wc > config.preferred_paragraph_words:
            log.warning(
                "long_paragraph",
                f"Paragraph ({wc} words) exceeds the preferred length of {config.preferred_paragraph_words} words",
                block_id=b.block_id,
            )


def _validate_html_remnants(document: Document, log: FindingLog) -> None:
    """Spec Fix 4: a document *about* markup (papers, tutorials, specs) can
    legitimately contain literal HTML/XML examples -- those belong in a
    fenced code block (pdf_extractor.py detects and groups them), and this
    must never re-scan CodeBlock content as if it were leftover conversion
    debris. For any tags that do remain outside a code block, identical
    repeated tags are collapsed into one aggregated finding with a count
    instead of one warning per occurrence -- one defect, one finding."""
    tag_locations: Dict[str, List[str]] = {}
    for b in document.blocks:
        if isinstance(b, CodeBlock):
            continue
        text = getattr(b, "text", None) or b.normalized or ""
        for tag in find_html_remnants(text):
            tag_locations.setdefault(tag, []).append(b.block_id)
    for tag, block_ids in tag_locations.items():
        if len(block_ids) == 1:
            log.warning("remaining_html", f"Unconverted HTML-like markup {tag!r} found outside code",
                        block_id=block_ids[0])
        else:
            log.warning(
                "remaining_html",
                f"Unconverted HTML-like markup {tag!r} found outside code in {len(block_ids)} places",
                block_id=block_ids[0],
            )


def _validate_heading_quality(document: Document, log: FindingLog) -> Dict[str, int]:
    """Retrospective safety net (spec sections 9, 16.3, 21): re-runs the same
    classifier used to keep bad candidates out of extraction against every
    HeadingBlock that actually made it into the final document, regardless
    of source format or extraction path. A hard-reject category (code,
    equation/matrix, page number, OCR fragment/punctuation-only) is always
    an ERROR -- these headings must never occur at all. Anything else the
    classifier would also have rejected (a full sentence, trailing
    punctuation, too many words) is logged as a lower-severity
    `suspicious_heading`, and only fails structural readiness in aggregate
    once it exceeds `suspicious_heading_ratio_limit` -- a single odd heading
    is not proof of a broken document, but a high proportion of them is."""
    headings = [b for b in document.blocks if isinstance(b, HeadingBlock)]
    counts = {
        "code_like_heading_count": 0,
        "equation_like_heading_count": 0,
        "page_furniture_heading_count": 0,
        "fragment_heading_count": 0,
        "caption_heading_count": 0,
        "suspicious_heading_count": 0,
        "single_character_heading_count": 0,
        "non_alphabetic_heading_count": 0,
    }
    for h in headings:
        text = (h.text or h.normalized or "").strip()
        if h.level == 1:
            # The title (H1) has its own validity rules (spec section 11,
            # title resolution) rather than the body-heading classifier used
            # here -- a short/minimal title is not the same failure mode as
            # a code/equation/fragment line getting promoted to a body
            # heading, so it's exempt from this particular scan.
            continue
        if len(text) == 1:
            counts["single_character_heading_count"] += 1
        if text and not any(c.isalpha() for c in text):
            counts["non_alphabetic_heading_count"] += 1
        reason = heading_rejection_reason(text)
        if reason is None:
            continue
        code = _HARD_REJECT_FINDING_CODES.get(reason)
        if code is not None:
            if reason == "code_or_algorithm":
                counts["code_like_heading_count"] += 1
            elif reason == "equation_or_matrix":
                counts["equation_like_heading_count"] += 1
            elif reason == "no_letters":
                counts["page_furniture_heading_count"] += 1
            elif reason == "figure_or_table_caption":
                counts["caption_heading_count"] += 1
            else:
                counts["fragment_heading_count"] += 1
            log.error(code, f"Heading {text!r} looks like {reason.replace('_', ' ')}, not a real heading",
                      block_id=h.block_id)
        else:
            counts["suspicious_heading_count"] += 1
            log.warning("suspicious_heading",
                        f"Heading {text!r} does not read as a genuine heading ({reason}); manual review recommended",
                        block_id=h.block_id)
    return counts


def _block_text(block) -> str:
    return (getattr(block, "text", None) or getattr(block, "normalized", None) or "").strip()


def _repeated_block_texts(document: Document, threshold: int = _PAGE_FURNITURE_REPEAT_THRESHOLD) -> set:
    """Normalized text of any block (any type, not just headings) that
    recurs verbatim `threshold`+ times across the whole document -- a
    running watermark/header/footer string misplaced mid-stream by the
    parser (e.g. a page stamp Docling read as a picture region's OCR text
    or as an ordinary paragraph) reliably repeats this way, exactly the same
    "identical text, 3+ times, carries no distinguishing identity" signal
    `heading_demotion.py` already uses for repeated boilerplate headings,
    just computed over every block so non-heading page furniture can be
    recognized as unreliable *evidence* too, not only rejected when it's
    itself masquerading as a heading."""
    counts = Counter(_norm_ws(_block_text(b)) for b in document.blocks if _block_text(b))
    return {t for t, c in counts.items() if c >= threshold}


def _validate_heading_context(document: Document, log: FindingLog) -> int:
    """Whole-document context checks the per-line/per-heading classifier
    structurally cannot perform (spec Fix 5): a false heading that happens
    to capture unrelated trailing text is "non-empty" and slips past
    `_validate_structural_segmentation`'s empty/tiny-section checks, so this
    looks at document-order adjacency instead of section body size. Returns
    the number of newly-flagged headings, folded into the same
    `suspicious_heading_count`/ratio gate `_validate_heading_quality`
    already populates -- re-running Fix 2's caption/axis/function-call/
    numeric detectors "retrospectively" falls out for free from that
    existing mechanism once `heading_rejection_reason` itself recognizes
    those shapes, so this only adds the genuinely new contextual signals.

    Both context signals below require a *second*, independent piece of
    evidence alongside raw adjacency, not adjacency alone -- proximity to a
    figure/caption/paragraph is extremely common in an illustrated document
    and is not, by itself, proof anything is wrong (see each signal's own
    comment for what the corroborating evidence is)."""
    blocks = document.blocks
    repeated_texts = _repeated_block_texts(document)
    bodyless_chain_idx = {
        i for i, b in enumerate(blocks)
        if isinstance(b, HeadingBlock) and i + 1 < len(blocks)
        and isinstance(blocks[i + 1], HeadingBlock) and blocks[i + 1].level == b.level
    }
    flagged = 0
    for i, b in enumerate(blocks):
        if not isinstance(b, HeadingBlock) or b.level == 1:
            continue
        text = _block_text(b)
        if not text:
            continue
        reasons = []
        prev_b = blocks[i - 1] if i > 0 else None
        next_b = blocks[i + 1] if i + 1 < len(blocks) else None

        # A numbered heading ("4 Linear Algebraic DFS on Binary Trees") has
        # its own strong, independent outline evidence -- an illustrated
        # document legitimately places a real section heading right next to
        # a figure/caption all the time, and that coincidental proximity
        # must not by itself count as evidence against it. The same
        # "independent evidence of being real" precedent extends to any
        # heading whose *text* passes `looks_like_title_candidate` (spec
        # section 11's stricter title-shape check: rejects a trailing-colon
        # lead-in, a shouted single word, a short imperative UI-chrome verb
        # like "Check"/"Continue") -- a heading that reads like an ordinary
        # descriptive title is not made suspicious by merely sitting next to
        # a real figure/table it introduces or follows; a heading that is
        # itself chrome/fragment-shaped stays flagged exactly as before.
        if (
            numbering_tuple(text) is None and not looks_like_title_candidate(text)
            and (
                (prev_b is not None and (isinstance(prev_b, ImageBlock) or looks_like_caption_text(_block_text(prev_b))))
                or (next_b is not None and (isinstance(next_b, ImageBlock) or looks_like_caption_text(_block_text(next_b))))
            )
        ):
            reasons.append("adjacent_to_figure_or_caption")

        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.5:
            reasons.append("high_symbol_or_digit_density")

        # Corroborating evidence here is the *reliability* of `prev_text`
        # itself: a running watermark/page-stamp string that recurs
        # verbatim throughout the document (`repeated_texts`) is not a real
        # sentence a heading could plausibly interrupt -- comparing across
        # it produces a false "interrupted sentence" out of two unrelated
        # real paragraphs the watermark merely sits between. An ordinary
        # paragraph, repeated or not, that is not part of that page-furniture
        # set is unaffected -- this narrows *which* adjacent block counts as
        # sentence evidence, it does not loosen the punctuation/case check.
        if (
            prev_b is not None and not isinstance(prev_b, HeadingBlock)
            and _norm_ws(_block_text(prev_b)) not in repeated_texts
        ):
            prev_text = _block_text(prev_b)
            next_text = _block_text(next_b) if next_b is not None else ""
            if (
                prev_text and prev_text[-1] not in ".!?:;"
                and next_text and next_text[0].isalpha() and next_text[0].islower()
            ):
                reasons.append("interrupts_sentence")

        if i in bodyless_chain_idx:
            reasons.append("bodyless_heading_chain")

        if reasons:
            flagged += 1
            log.warning(
                "suspicious_heading",
                f"Heading {text!r} looks structurally suspicious in context ({', '.join(reasons)})",
                block_id=b.block_id,
            )
    return flagged


def _is_bodyless_heading(blocks: List, i: int) -> bool:
    """Same "no body of its own" test `heading_repair.py`'s repair pass uses:
    nothing follows the heading, or the very next block is a heading at an
    equal-or-shallower level. A heading immediately followed by a *deeper*
    heading is a legitimate parent/container and is never bodyless."""
    b = blocks[i]
    if not isinstance(b, HeadingBlock):
        return False
    nxt = blocks[i + 1] if i + 1 < len(blocks) else None
    if nxt is None:
        return True
    return isinstance(nxt, HeadingBlock) and nxt.level <= b.level


def _validate_ocr_heading_evidence(document: Document, log: FindingLog) -> Dict[str, int]:
    """Post-repair verification (spec "Validation" metrics/gates): scans the
    *final* block list (after extraction-time prevention and the bodyless-
    heading repair pass have both already run) for any OCR-derived heading
    that should never have survived either stage. This is deliberately
    independent of the repair pass's own self-reported counters -- it is the
    thing that actually backs the `== 0` gates, so a bug in either the
    scorer or the repair pass shows up here instead of only in an internal
    counter nobody re-checks.

    Provenance gate: the `diagram_annotation` shape is evaluated only for
    headings that carry OCR evidence (`style["from_ocr"]`, persisted together
    with `ocr_heading_score`/`ocr_heading_signals`/`ocr_heading_negatives` by
    pdf_extractor.py's OCR path). `looks_like_diagram_annotation` is a pure
    text-shape classifier (axis/direction/quantity-word regexes) with no
    geometric signal, and a 2026-09-14 measurement
    (artifacts/heading_classifier_step1/D5.md) found 10/10 sampled
    `diagram_annotations_promoted_to_heading` flags were false positives on
    legitimate native-path headings ("Dynamic errors", "Result values",
    "DEFAULT VALUES", ...) that merely happen to match a bare-word/quantity
    shape. For native text there is no way to distinguish a real subsection
    heading from a diagram label without figure-geometry context, so the
    check is meaningless -- and only harmful -- off the OCR path. The
    `figure_label`/`caption`/`table_cell` shapes stay format-agnostic: those
    patterns ("Figure 3:", "Input Image", key:value rows) are unambiguous
    non-heading text regardless of provenance (see
    `classify_ocr_nonheading_kind`'s docstring and
    test_figure_label_heading_fails_the_hard_gate_even_if_not_from_ocr)."""
    blocks = document.blocks
    counts = {
        "figure_labels_promoted_to_heading": 0,
        "diagram_annotations_promoted_to_heading": 0,
        "table_cells_promoted_to_heading": 0,
        "unsupported_ocr_headings_remaining": 0,
    }
    _KIND_TO_METRIC = {
        "figure_label": "figure_labels_promoted_to_heading",
        "diagram_annotation": "diagram_annotations_promoted_to_heading",
        "table_cell": "table_cells_promoted_to_heading",
    }
    for i, b in enumerate(blocks):
        if not isinstance(b, HeadingBlock) or b.level == 1:
            continue
        text = (b.text or b.normalized or "").strip()
        kind = classify_ocr_nonheading_kind(text)
        if kind == "diagram_annotation" and not b.style.get("from_ocr"):
            # Native-path heading: the diagram-annotation shape is pure text
            # shape with no geometry behind it, and measured 10/10 false
            # positives on legitimate native headings (D5.md) -- skip.
            continue
        metric = _KIND_TO_METRIC.get(kind)
        if metric is not None:
            counts[metric] += 1
            log.error(
                "unsupported_ocr_heading",
                f"Heading {text!r} looks like a {kind.replace('_', ' ')}, not a real heading, "
                f"and survived both extraction-time prevention and the repair pass",
                block_id=b.block_id, page=b.page,
            )
            continue
        if not b.style.get("from_ocr"):
            continue
        score = b.style.get("ocr_heading_score", OCR_HEADING_CONFIDENT_SCORE)
        if score < OCR_HEADING_CONFIDENT_SCORE and _is_bodyless_heading(blocks, i):
            counts["unsupported_ocr_headings_remaining"] += 1
            log.error(
                "unsupported_ocr_heading",
                f"Heading {text!r} is a weakly-scored, bodyless OCR heading that survived the "
                f"repair pass",
                block_id=b.block_id, page=b.page,
            )
    return counts


def _validate_title(document: Document, log: FindingLog) -> Dict[str, Any]:
    """Spec sections 2/3/14: the document title has its own validity rules,
    independent of the ordinary body-heading classifier. Runs against the
    *final* block list, after normalization/front-matter grouping have
    already run, so it backs the gate for every source format/path (not
    just PDF's own title resolver and `_resolve_heading_level` guard)."""
    headings = [b for b in document.blocks if isinstance(b, HeadingBlock)]
    h1s = [h for h in headings if h.level == 1]
    document_title_valid = True
    h1_is_first_meaningful_block = False

    if len(h1s) == 0:
        document_title_valid = False
        log.error("invalid_document_title", "No H1 (document title) is present in the final document")
    elif len(h1s) > 1:
        document_title_valid = False
        log.error("invalid_document_title",
                   f"{len(h1s)} H1 headings are present in the final document; exactly one is required")

    if h1s:
        h1 = h1s[0]
        idx = document.blocks.index(h1)
        # Front-matter blocks (author/affiliation/contact/keywords/...) are
        # the only kind of content allowed to precede the title in a
        # normalized document -- anything else ahead of the H1 means the
        # title genuinely isn't the first meaningful block.
        preceding_non_front_matter = any(
            not isinstance(b, FRONT_MATTER_BLOCK_TYPES) for b in document.blocks[:idx]
        )
        h1_is_first_meaningful_block = idx == 0 or not preceding_non_front_matter
        if idx > 0 and preceding_non_front_matter:
            document_title_valid = False
            log.error(
                "root_title_not_rendered_as_h1",
                f"Title heading {h1.text!r} is not the first meaningful block (position {idx})",
                block_id=h1.block_id,
            )

        title_norm = h1.text.strip().lower().rstrip(":.")
        if title_norm in _TITLE_BLOCKING_LABELS:
            document_title_valid = False
            log.error(
                "abstract_promoted_to_h1",
                f"The H1 title is {h1.text!r}, which is a section label, not a document title",
                block_id=h1.block_id,
            )

    return {
        "document_title_valid": document_title_valid,
        "h1_count": len(h1s),
        "h1_is_first_meaningful_block": h1_is_first_meaningful_block,
    }


def _validate_front_matter_placement(document: Document, log: FindingLog) -> Dict[str, bool]:
    """Spec sections 5/6/14: author/affiliation/contact/keyword front matter
    must form one contiguous group immediately after the title, and
    publication/copyright/permission/conference metadata must never be
    interleaved with body content. Front-matter extraction/grouping already
    guarantees this by construction for PDFs (`pdf_extractor.py`); this is
    the independent, format-agnostic check that actually backs the gate."""
    blocks = document.blocks
    fm_indices = [i for i, b in enumerate(blocks) if isinstance(b, FRONT_MATTER_BLOCK_TYPES)]
    author_interleaved = False
    publication_interleaved = False
    if fm_indices:
        contiguous = fm_indices == list(range(fm_indices[0], fm_indices[0] + len(fm_indices)))
        h1_idx = next((i for i, b in enumerate(blocks) if isinstance(b, HeadingBlock) and b.level == 1), None)
        expected_start = (h1_idx + 1) if h1_idx is not None else 0
        starts_right_after_title = fm_indices[0] == expected_start
        if not (contiguous and starts_right_after_title):
            author_interleaved = any(isinstance(blocks[i], _AUTHOR_LIKE_FRONT_MATTER_TYPES) for i in fm_indices)
            publication_interleaved = any(
                not isinstance(blocks[i], _AUTHOR_LIKE_FRONT_MATTER_TYPES) for i in fm_indices
            )
            if author_interleaved:
                log.error(
                    "author_metadata_interleaved_with_body",
                    "Author/affiliation/contact front matter is not grouped as one contiguous "
                    "block immediately after the title",
                )
            if publication_interleaved:
                log.error(
                    "publication_metadata_interleaved_with_body",
                    "Publication/copyright/permission/conference metadata is not grouped as one "
                    "contiguous block immediately after the title",
                )
    return {
        "front_matter_grouped": not (author_interleaved or publication_interleaved),
        "author_metadata_interleaved_with_body": author_interleaved,
        "publication_metadata_interleaved_with_body": publication_interleaved,
    }


def _references_section_block_ids(document: Document) -> set:
    """Block ids of every paragraph inside a References/Bibliography-style
    section, at any depth -- see `front_matter.is_references_heading`."""
    ids: set = set()
    for s in build_sections(document):
        if any(is_references_heading(h) for h in s.header_path):
            ids.update(cb.block_id for cb in s.content_blocks)
    return ids


def _validate_paragraph_continuity(document: Document, log: FindingLog) -> int:
    """Defense-in-depth (spec section 7): flags a body paragraph whose text
    has front-matter content embedded mid-string -- the residual case
    extraction-time excision can miss (e.g. a multi-line copyright/
    permission notice whose continuation line doesn't independently match
    the line-level classifier on its own). Scoped to the strong, low-
    false-positive signals (contact/permission/copyright/conference/
    publication-metadata shape) -- author-name shape is deliberately
    excluded here for the same reason `pdf_extractor.py`'s title resolver
    excludes it: a short capitalized phrase embedded in ordinary prose is
    far more often coincidental than a real interleaving defect. Paragraphs
    inside the document's own References/Bibliography section are exempt
    entirely -- see `_references_section_block_ids`."""
    flagged = 0
    strong_kinds = {"contact", "permission", "copyright", "conference", "publication_metadata"}
    reference_ids = _references_section_block_ids(document)
    for b in document.blocks:
        if not isinstance(b, ParagraphBlock) or b.block_id in reference_ids:
            continue
        text = b.text or b.normalized or ""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) < 2:
            continue
        for s in sentences:
            s = s.strip()
            if s and classify_front_matter_line(s) in strong_kinds:
                flagged += 1
                log.error(
                    "paragraph_interrupted_by_front_matter",
                    f"Paragraph contains embedded front-matter text not fully excised from body "
                    f"flow: {s!r}",
                    block_id=b.block_id,
                )
                break
    return flagged


def _leaf_sections(document: Document) -> List[Section]:
    """Leaf sections: heading-rooted sections with no descendant heading.
    A heading whose only content is child sections (a chapter/part heading)
    is a parent container, not an empty leaf, and must not be flagged as
    empty (spec section 16.1)."""
    sections = [s for s in build_sections(document) if s.heading_block is not None]
    paths = [tuple(s.header_path) for s in sections]
    leaves = []
    for i, s in enumerate(sections):
        p = paths[i]
        has_descendant = any(len(q) > len(p) and q[:len(p)] == p for j, q in enumerate(paths) if j != i)
        if not has_descendant:
            leaves.append(s)
    return leaves


def _validate_structural_segmentation(document: Document, config: ChunkingConfig, log: FindingLog) -> Dict[str, Any]:
    """Over-segmentation checks that a text-only heading classifier can't
    catch: a short interactive-courseware UI label ("Show answer") or a
    caption fragment can be a perfectly grammatical, innocuous-looking
    heading on its own, but a document where a large share of its leaf
    sections carry no real body content is still exhibiting the "arbitrary
    fragment promoted to heading" failure mode (spec section 16)."""
    leaves = _leaf_sections(document)
    empty = 0
    tiny = 0
    body_words: List[int] = []
    for s in leaves:
        wc = sum(cb.word_count for cb in s.content_blocks)
        has_atomic = any(isinstance(cb, _ATOMIC_BLOCK_TYPES) for cb in s.content_blocks)
        body_words.append(wc)
        if wc == 0 and not has_atomic:
            empty += 1
            log.error("empty_section", f"Section {' > '.join(s.header_path)!r} has no body content",
                      block_id=s.heading_block.block_id if s.heading_block else None)
        elif wc < config.tiny_section_body_words and not has_atomic:
            tiny += 1

    leaf_count = len(leaves)
    tiny_ratio = (tiny / leaf_count) if leaf_count else 0.0
    if leaf_count >= config.min_leaf_sections_for_ratio_gates and tiny_ratio > config.tiny_section_ratio_limit:
        log.error(
            "excessive_tiny_sections",
            f"{tiny}/{leaf_count} leaf sections ({tiny_ratio:.0%}) have fewer than "
            f"{config.tiny_section_body_words} words of body content, exceeding the "
            f"{config.tiny_section_ratio_limit:.0%} limit -- indicates over-segmentation "
            f"(fragments or UI labels promoted to headings) rather than genuine structure",
        )

    return {
        "empty_section_count": empty,
        "tiny_section_count": tiny,
        "leaf_section_count_for_ratio": leaf_count,
        "tiny_section_ratio": tiny_ratio,
        "median_section_body_words": _median(body_words),
        "mean_section_body_words": (sum(body_words) / len(body_words)) if body_words else 0.0,
    }


def _median(values: List[int]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _validate_suspicious_heading_ratio(heading_count: int, suspicious_count: int, config: ChunkingConfig,
                                        log: FindingLog) -> float:
    ratio = (suspicious_count / heading_count) if heading_count else 0.0
    if heading_count >= config.min_leaf_sections_for_ratio_gates and ratio > config.suspicious_heading_ratio_limit:
        log.error(
            "excessive_suspicious_headings",
            f"{suspicious_count}/{heading_count} headings ({ratio:.0%}) do not read as genuine headings, "
            f"exceeding the {config.suspicious_heading_ratio_limit:.0%} limit",
        )
    return ratio


def _validate_source_coverage(document: Document, chunking_result: ChunkingResult, log: FindingLog) -> None:
    content_block_ids = {b.block_id for b in document.blocks if isinstance(b, _CONTENT_BLOCK_TYPES)}
    used: Dict[str, int] = {}
    for chunk in chunking_result.chunks:
        for bid in chunk.source_block_ids:
            used[bid] = used.get(bid, 0) + 1
    dropped = set(chunking_result.dropped_block_ids)
    split_map = chunking_result.split_block_provenance
    for bid in content_block_ids:
        if bid in dropped:
            continue
        if bid in split_map:
            # A subdivided atomic block is represented by its *parts*, not
            # the (no-longer-chunked) parent id -- every part must appear in
            # exactly one chunk for the parent to count as fully covered,
            # with no gap (a missing part) or duplication (a repeated part).
            part_ids = split_map[bid]
            counts = [used.get(pid, 0) for pid in part_ids]
            if all(c == 0 for c in counts):
                log.error("missing_source_block", f"Source block {bid} does not appear in any chunk", block_id=bid)
                continue
            for pid, count in zip(part_ids, counts):
                if count == 0:
                    log.error(
                        "missing_split_block_part",
                        f"Part {pid} of subdivided source block {bid} does not appear in any chunk",
                        block_id=pid,
                    )
                elif count > 1:
                    log.error(
                        "duplicate_split_block_part",
                        f"Part {pid} of subdivided source block {bid} appears in {count} chunks",
                        block_id=pid,
                    )
            continue
        count = used.get(bid, 0)
        if count == 0:
            log.error("missing_source_block", f"Source block {bid} does not appear in any chunk", block_id=bid)
        elif count > 1:
            log.error("duplicate_source_block", f"Source block {bid} appears in {count} chunks", block_id=bid)


def _source_block_counts(document: Document, chunking_result: ChunkingResult) -> Dict[str, int]:
    """Pure counting companion to `_validate_source_coverage`'s findings, used
    to surface the same coverage numbers in the quality report (spec section
    24's source_blocks/represented/excluded/duplicate fields). A subdivided
    atomic block counts as one "represented" source block (not one per part)
    as long as every part it was split into is represented exactly once."""
    content_block_ids = {b.block_id for b in document.blocks if isinstance(b, _CONTENT_BLOCK_TYPES)}
    used: Dict[str, int] = {}
    for chunk in chunking_result.chunks:
        for bid in chunk.source_block_ids:
            used[bid] = used.get(bid, 0) + 1
    dropped = set(chunking_result.dropped_block_ids)
    split_map = chunking_result.split_block_provenance
    represented = excluded = duplicated = 0
    for bid in content_block_ids:
        if bid in dropped:
            excluded += 1
            continue
        if bid in split_map:
            counts = [used.get(pid, 0) for pid in split_map[bid]]
            if all(c == 0 for c in counts):
                excluded += 1
            else:
                represented += 1
                if any(c != 1 for c in counts):
                    duplicated += 1
            continue
        count = used.get(bid, 0)
        if count == 0:
            excluded += 1
        else:
            represented += 1
            if count > 1:
                duplicated += 1
    return {
        "source_blocks": len(content_block_ids),
        "represented_source_blocks": represented,
        "excluded_source_blocks": excluded,
        "duplicate_source_blocks": duplicated,
    }


def _heading_hierarchy_valid(headings: List[HeadingBlock]) -> bool:
    if not headings:
        return True
    if headings[0].level != 1:
        return False
    max_seen = headings[0].level
    for h in headings[1:]:
        if h.level > max_seen + 1:
            return False
        max_seen = max(max_seen, h.level)
    return True


def compute_stats(document: Document, chunking_result: ChunkingResult, source_file: str,
                   normalized_file: str, source_format: str, ocr_used: bool,
                   findings: FindingLog, heading_quality: Optional[Dict[str, int]] = None,
                   structural: Optional[Dict[str, Any]] = None,
                   ocr_evidence: Optional[Dict[str, int]] = None,
                   title_info: Optional[Dict[str, Any]] = None,
                   front_matter_info: Optional[Dict[str, bool]] = None,
                   interrupted_paragraphs: int = 0) -> Dict[str, Any]:
    heading_quality = heading_quality or {}
    structural = structural or {}
    ocr_evidence = ocr_evidence or {}
    title_info = title_info or {}
    front_matter_info = front_matter_info or {}
    extra = document.metadata.extra or {}
    heading_repair = extra.get("heading_repair") or {}
    heading_sibling_audit = extra.get("heading_sibling_audit") or {}
    paragraph_continuity_repair = extra.get("paragraph_continuity_repair") or {}
    headers = {f"h{i}": 0 for i in range(1, 7)}
    heading_blocks: List[HeadingBlock] = []
    for b in document.blocks:
        if isinstance(b, HeadingBlock):
            headers[f"h{b.level}"] += 1
            heading_blocks.append(b)

    paragraphs = [b for b in document.blocks if isinstance(b, ParagraphBlock)]
    sections = build_sections(document)
    leaf_words = [sum(cb.word_count for cb in s.content_blocks) for s in sections if s.content_blocks]

    errors = findings.count(Severity.ERROR)
    warnings = findings.count(Severity.WARNING)
    finding_codes = [f.code for f in findings.findings if f.severity == Severity.ERROR]

    # Derived from the findings themselves (rather than recomputing the
    # preferred-length threshold here) so this can never drift from what
    # _validate_paragraph_lengths actually warned about.
    long_paragraph_count = len({f.block_id for f in findings.findings if f.code == "long_paragraph"})
    oversized_paragraph_count = sum(1 for c in finding_codes if c == "oversized_paragraph")
    embedding_truncation_count = sum(1 for c in finding_codes if c == "chunk_exceeds_token_limit")
    empty_chunk_count = sum(1 for c in chunking_result.chunks if not c.body.strip())

    coverage = _source_block_counts(document, chunking_result)
    page_meta = document.metadata.extra or {}
    native_text_pages = page_meta.get("native_text_pages")
    ocr_pages = page_meta.get("ocr_pages")
    failed_pages = page_meta.get("failed_pages")
    pages_accounted_for = (
        (native_text_pages or 0) + (ocr_pages or 0) + (failed_pages or 0)
        if any(v is not None for v in (native_text_pages, ocr_pages, failed_pages)) else None
    )

    structural_readiness = errors == 0
    retrieval_ready = structural_readiness and len(chunking_result.chunks) > 0

    stats: Dict[str, Any] = {
        "source_file": source_file,
        "normalized_file": normalized_file,
        "source_format": source_format,
        "ocr_used": ocr_used,
        "page_count": document.metadata.page_count,
        "native_text_pages": native_text_pages,
        "ocr_pages": ocr_pages,
        "failed_pages": failed_pages,
        "pages_accounted_for": pages_accounted_for,
        "headers": headers,
        "heading_count": len(heading_blocks),
        "heading_hierarchy_valid": _heading_hierarchy_valid(heading_blocks),
        "unique_header_paths": len({tuple(c.header_path) for c in chunking_result.chunks}),
        "paragraph_count": len(paragraphs),
        "largest_paragraph_words": max((p.word_count for p in paragraphs), default=0),
        "long_paragraph_count": long_paragraph_count,
        "oversized_paragraph_count": oversized_paragraph_count,
        "list_count": sum(1 for b in document.blocks if isinstance(b, ListBlock)),
        "table_count": sum(1 for b in document.blocks if isinstance(b, TableBlock)),
        "code_block_count": sum(1 for b in document.blocks if isinstance(b, CodeBlock)),
        "leaf_section_count": len(leaf_words),
        "largest_leaf_section_words": max(leaf_words, default=0),
        "chunk_count": len(chunking_result.chunks),
        "empty_chunk_count": empty_chunk_count,
        "largest_chunk_words": chunking_result.largest_chunk_words,
        "largest_chunk_tokens": chunking_result.largest_chunk_tokens,
        "max_chunk_words": chunking_result.largest_chunk_words,
        "max_chunk_tokens": chunking_result.largest_chunk_tokens,
        "forced_fallback_splits": chunking_result.forced_fallback_splits,
        "forced_fallback_split_count": chunking_result.forced_fallback_splits,
        # `oversized_atomic_blocks` now counts only blocks subdivision could
        # not resolve (dropped, hard-stop) -- `oversized_unresolved_blocks`
        # is the same number under the name used elsewhere in this spec.
        # `atomic_blocks_split`/`forced_split_atomic_parts` report on the
        # (expected common) case where an oversized block was successfully
        # subdivided instead.
        "oversized_atomic_blocks": chunking_result.oversized_atomic_blocks,
        "oversized_atomic_block_count": chunking_result.oversized_atomic_blocks,
        "oversized_unresolved_blocks": chunking_result.oversized_atomic_blocks,
        "atomic_blocks_split": chunking_result.atomic_blocks_split,
        "forced_split_atomic_parts": chunking_result.forced_split_atomic_parts,
        "embedding_truncation_count": embedding_truncation_count,
        "repairs": findings.count(Severity.FIXED),
        "warnings": warnings,
        "errors": errors,
        "structural_readiness": structural_readiness,
        "retrieval_ready": retrieval_ready,
        "heading_density_per_1000_words": (
            (len(heading_blocks) / max(sum(p.word_count for p in paragraphs), 1)) * 1000
        ),
        "heading_depth_distribution": headers,
        "code_like_heading_count": heading_quality.get("code_like_heading_count", 0),
        "equation_like_heading_count": heading_quality.get("equation_like_heading_count", 0),
        "page_furniture_heading_count": heading_quality.get("page_furniture_heading_count", 0),
        "fragment_heading_count": heading_quality.get("fragment_heading_count", 0),
        "caption_heading_count": heading_quality.get("caption_heading_count", 0),
        "single_character_heading_count": heading_quality.get("single_character_heading_count", 0),
        "non_alphabetic_heading_count": heading_quality.get("non_alphabetic_heading_count", 0),
        "suspicious_heading_count": heading_quality.get("suspicious_heading_count", 0),
        "suspicious_heading_ratio": heading_quality.get("suspicious_heading_ratio", 0.0),
        "empty_section_count": structural.get("empty_section_count", 0),
        "empty_leaf_sections": structural.get("empty_section_count", 0),
        "tiny_section_count": structural.get("tiny_section_count", 0),
        "tiny_section_ratio": structural.get("tiny_section_ratio", 0.0),
        "median_section_body_words": structural.get("median_section_body_words", 0.0),
        "mean_section_body_words": structural.get("mean_section_body_words", 0.0),
        # OCR heading promotion (spec "OCR heading promotion" / "Validation"):
        # candidates/accepted/rejected are extraction-time tallies from
        # pdf_extractor.py's `_OcrHeadingScorer`; demoted and the chain
        # counts come from heading_repair.py's post-classification repair
        # pass; the four "promoted_to_heading"/"remaining" counts are
        # computed fresh here, directly off the final block list, so they
        # verify what the scorer and repair pass actually achieved rather
        # than trusting their own self-reported counters.
        "ocr_heading_candidates": extra.get("ocr_heading_candidates", 0),
        "ocr_headings_accepted": extra.get("ocr_headings_accepted", 0),
        "ocr_headings_rejected": extra.get("ocr_headings_rejected", 0),
        "ocr_headings_demoted": heading_repair.get("ocr_headings_demoted", 0),
        "bodyless_heading_chains_detected": heading_repair.get("bodyless_heading_chains_detected", 0),
        "bodyless_heading_chains_repaired": heading_repair.get("bodyless_heading_chains_repaired", 0),
        "figure_labels_promoted_to_heading": ocr_evidence.get("figure_labels_promoted_to_heading", 0),
        "diagram_annotations_promoted_to_heading": ocr_evidence.get("diagram_annotations_promoted_to_heading", 0),
        "table_cells_promoted_to_heading": ocr_evidence.get("table_cells_promoted_to_heading", 0),
        "unsupported_ocr_headings_remaining": ocr_evidence.get("unsupported_ocr_headings_remaining", 0),
        # Title/front-matter/reading-order structural gates (spec section 14).
        "document_title_valid": title_info.get("document_title_valid", True),
        "h1_count": title_info.get("h1_count", headers.get("h1", 0)),
        "h1_is_first_meaningful_block": title_info.get("h1_is_first_meaningful_block", True),
        "title_resolution": extra.get("title_resolution"),
        "title_distinctiveness_score": (extra.get("title_distinctiveness") or {}).get("title_distinctiveness_score"),
        "filename_title_similarity": (extra.get("title_distinctiveness") or {}).get("filename_title_similarity"),
        "root_heading_title_score": (extra.get("title_distinctiveness") or {}).get("root_heading_title_score"),
        "title_fallback_reason": (extra.get("title_distinctiveness") or {}).get("title_fallback_reason"),
        "front_matter_grouped": front_matter_info.get("front_matter_grouped", True),
        "author_metadata_interleaved_with_body": front_matter_info.get("author_metadata_interleaved_with_body", False),
        "publication_metadata_interleaved_with_body": front_matter_info.get(
            "publication_metadata_interleaved_with_body", False),
        "paragraphs_repaired": paragraph_continuity_repair.get("paragraphs_repaired", 0),
        "front_matter_sentences_excised": paragraph_continuity_repair.get("front_matter_sentences_excised", 0),
        "interrupted_paragraph_count": interrupted_paragraphs,
        "paragraph_continuity_valid": interrupted_paragraphs == 0,
        "reading_order_valid": not (
            front_matter_info.get("author_metadata_interleaved_with_body", False)
            or front_matter_info.get("publication_metadata_interleaved_with_body", False)
        ) and interrupted_paragraphs == 0,
        "outline_gaps_checked": heading_sibling_audit.get("outline_gaps_checked", 0),
        "high_confidence_headings_recovered": heading_sibling_audit.get("high_confidence_headings_recovered", 0),
        "high_confidence_headings_ambiguous": heading_sibling_audit.get("high_confidence_headings_ambiguous", 0),
        "high_confidence_headings_preserved": heading_sibling_audit.get("high_confidence_headings_ambiguous", 0) == 0,
        "layout_hyphenation": extra.get("layout_hyphenation"),
        "front_matter_block_count": sum(
            1 for b in document.blocks
            if type(b).__name__ in (
                "AuthorBlock", "AffiliationBlock", "ContactBlock", "KeywordBlock",
                "PublicationMetadataBlock", "CopyrightNoticeBlock", "PermissionNoticeBlock",
                "ConferenceMetadataBlock",
            )
        ),
        "excluded_front_matter_count": len(extra.get("excluded_front_matter") or []),
    }
    stats.update(coverage)
    return stats


def completion_state(report: "ValidationReport", indexed: bool) -> Dict[str, Any]:
    """Section 8's per-document completion object. `indexed` is a pipeline-level
    fact (whether an index entry was actually written) that validate() itself
    can't know, so callers (DocumentPipeline) supply it explicitly."""
    s = report.stats
    unresolved_blocking_list_errors = sum(
        1 for f in report.findings.findings if f.severity == Severity.ERROR and "list" in f.code
    )
    return {
        "processing_complete": True,
        "source_coverage_complete": s.get("excluded_source_blocks", 0) == 0,
        "structural_readiness": s.get("structural_readiness", False),
        "retrieval_ready": report.retrieval_ready,
        "blocking_errors": s.get("errors", 0),
        "forced_fallback_splits": s.get("forced_fallback_split_count", 0),
        "embedding_truncations": s.get("embedding_truncation_count", 0),
        "indexed": indexed,
        # Spec section 14's structural readiness gate.
        "document_title_valid": s.get("document_title_valid", False),
        "h1_count": s.get("h1_count", 0),
        "h1_is_first_meaningful_block": s.get("h1_is_first_meaningful_block", False),
        "front_matter_grouped": s.get("front_matter_grouped", False),
        "reading_order_valid": s.get("reading_order_valid", False),
        "paragraph_continuity_valid": s.get("paragraph_continuity_valid", False),
        "high_confidence_headings_preserved": s.get("high_confidence_headings_preserved", False),
        "heading_hierarchy_valid": s.get("heading_hierarchy_valid", False),
        "publication_metadata_interleaved_with_body": s.get("publication_metadata_interleaved_with_body", True),
        "author_metadata_interleaved_with_body": s.get("author_metadata_interleaved_with_body", True),
        "unresolved_blocking_list_errors": unresolved_blocking_list_errors,
    }


def validate(document: Document, chunking_result: ChunkingResult, config: ChunkingConfig,
             source_file: str, normalized_file: str, source_format: str, ocr_used: bool,
             upstream_findings: FindingLog) -> ValidationReport:
    log = FindingLog()
    log.extend(upstream_findings)
    log.extend(chunking_result.findings)
    _validate_paragraph_lengths(document, config, log)
    _validate_html_remnants(document, log)
    _validate_source_coverage(document, chunking_result, log)

    heading_quality = _validate_heading_quality(document, log)
    heading_quality["suspicious_heading_count"] += _validate_heading_context(document, log)
    heading_count = sum(1 for b in document.blocks if isinstance(b, HeadingBlock))
    heading_quality["suspicious_heading_ratio"] = _validate_suspicious_heading_ratio(
        heading_count, heading_quality["suspicious_heading_count"], config, log,
    )
    structural = _validate_structural_segmentation(document, config, log)
    ocr_evidence = _validate_ocr_heading_evidence(document, log)
    title_info = _validate_title(document, log)
    front_matter_info = _validate_front_matter_placement(document, log)
    interrupted_paragraphs = _validate_paragraph_continuity(document, log)

    stats = compute_stats(document, chunking_result, source_file, normalized_file, source_format, ocr_used, log,
                           heading_quality=heading_quality, structural=structural, ocr_evidence=ocr_evidence,
                           title_info=title_info, front_matter_info=front_matter_info,
                           interrupted_paragraphs=interrupted_paragraphs)
    return ValidationReport(findings=log, stats=stats, retrieval_ready=stats["retrieval_ready"])

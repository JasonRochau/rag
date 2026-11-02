"""
MAIN_TEXT retrieval-readiness policy (see DELIVERABLES.md, "MAIN_TEXT vs
FULL_FIDELITY ingestion profiles").

Answers exactly one question: "is the filtered main-text retrieval view
safe to index?" -- separately from FULL_FIDELITY's full structural
perfection gate (`validation.validate()`'s `errors == 0`).

Reuses `validate()` itself (no parallel validation system): the caller
runs `validate()` against the *filtered* MAIN_TEXT document, and this
module re-derives a MAIN_TEXT-specific `retrieval_ready`/blocking-error
list from that report's findings, by name-based exemption of a fixed,
narrow set of finding codes that are only ever about content MAIN_TEXT has
already excluded (front matter placement, figure/caption-shaped headings,
title/H1 structural gates -- MAIN_TEXT resolves its own title separately,
see retrieval_view.py). Every other finding code -- token-budget overruns,
missing/duplicate source-block coverage, code/equation/fragment headings
-- still blocks, unchanged. (Oversized paragraphs and unsplittable atomic
blocks no longer appear here at all: since the 2026-09-23 drop-warn-succeed
policy, `chunker.py` force-splits the former and drops the latter with a
WARNING, so neither ever reaches this error gate.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .findings import Finding, Severity
from .validation import ValidationReport

# Findings that are only ever about content this profile has already
# excluded from the retrieval view (front-matter placement, a figure/
# caption/page-furniture line that got promoted to a heading, or the
# full-document H1/title structural gate -- MAIN_TEXT resolves its own
# title via retrieval_view.resolve_main_text_title instead of requiring a
# single well-placed body H1). Deliberately narrow and enumerated by name,
# not a blanket "ignore everything front-matter/figure-related" rule --
# every other error code (chunk_exceeds_token_limit, missing/duplicate_
# source_block, code/equation/fragment headings, ...) is never exempted and
# still blocks indexing. Note `oversized_paragraph` and
# `oversized_atomic_block` are deliberately absent from BOTH lists: they
# ceased to be error codes on 2026-09-23 (oversized paragraphs are
# force-split in every profile; an unsplittable atomic block is dropped
# from the index with a WARNING), so they can no longer reach this gate.
MAIN_TEXT_EXEMPT_ERROR_CODES = frozenset({
    "author_metadata_interleaved_with_body",
    "publication_metadata_interleaved_with_body",
    "paragraph_interrupted_by_front_matter",
    "invalid_document_title",
    "root_title_not_rendered_as_h1",
    "abstract_promoted_to_h1",
    "empty_section",
    "excessive_tiny_sections",
    "excessive_suspicious_headings",
    "caption_promoted_to_heading",
    "page_number_promoted_to_heading",
    # A heading that reads as a diagram/axis/chart-annotation label
    # ("Values", "Counts") is the exact "graphical axis labels, diagram
    # labels" category Phase 2 excludes -- it just surfaced through the
    # heading-quality OCR-evidence check rather than a dedicated block
    # type. Note this only exempts the *finding*; the heading text itself
    # still remains in the MAIN_TEXT view (no block-type signal exists to
    # remove it generically the way retrieval_view.py's shell-heading rule
    # does for a figure-adjacent heading), a disclosed, narrow limitation.
    "unsupported_ocr_heading",
})


@dataclass
class MainTextReadiness:
    ready: bool
    title: Optional[str]
    title_source: str
    blocking_findings: List[Finding] = field(default_factory=list)
    exempted_findings: List[Finding] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "retrieval_ready": self.ready,
            "title": self.title,
            "title_source": self.title_source,
            "blocking_error_count": len(self.blocking_findings),
            "blocking_error_codes": sorted({f.code for f in self.blocking_findings}),
            "exempted_error_count": len(self.exempted_findings),
            "exempted_error_codes": sorted({f.code for f in self.exempted_findings}),
            "reasons": self.reasons,
        }


def evaluate_main_text_readiness(
    report: ValidationReport,
    title: Optional[str],
    title_source: str,
    has_body_content: bool,
) -> MainTextReadiness:
    """`report` must already be `validate()`'s output against the *filtered*
    MAIN_TEXT document + its own chunking result -- this function does not
    re-run validation, only re-interprets it under MAIN_TEXT's policy."""
    all_errors = [f for f in report.findings.findings if f.severity == Severity.ERROR]
    blocking = [f for f in all_errors if f.code not in MAIN_TEXT_EXEMPT_ERROR_CODES]
    exempted = [f for f in all_errors if f.code in MAIN_TEXT_EXEMPT_ERROR_CODES]

    reasons: List[str] = []
    if not title:
        reasons.append("no title could be resolved from title-eligible content")
    if not has_body_content:
        reasons.append("no main body text remains after MAIN_TEXT filtering")
    if report.stats.get("chunk_count", 0) == 0:
        reasons.append("zero chunks were produced")
    if report.stats.get("empty_chunk_count", 0) > 0:
        reasons.append(f"{report.stats['empty_chunk_count']} empty chunk(s) produced")
    for f in blocking:
        reasons.append(f"{f.code}: {f.message}")

    ready = not blocking and bool(title) and has_body_content and report.stats.get("chunk_count", 0) > 0 \
        and report.stats.get("empty_chunk_count", 0) == 0

    return MainTextReadiness(
        ready=ready, title=title, title_source=title_source,
        blocking_findings=blocking, exempted_findings=exempted, reasons=reasons,
    )


def apply_main_text_overlay(report: ValidationReport, readiness: MainTextReadiness) -> None:
    """Mutates `report` in place so its `retrieval_ready`/stats reflect the
    MAIN_TEXT policy instead of the raw full-error-count gate `validate()`
    itself computes. Deliberately does not touch `stats["errors"]` (the
    total error count, including exempted ones, stays visible for
    diagnostics) -- only what actually *gates* indexing changes."""
    report.retrieval_ready = readiness.ready
    report.stats["retrieval_ready"] = readiness.ready
    report.stats["structural_readiness"] = not readiness.blocking_findings
    report.stats["main_text_title"] = readiness.title
    report.stats["main_text_title_source"] = readiness.title_source
    report.stats["main_text_blocking_error_count"] = len(readiness.blocking_findings)
    report.stats["main_text_blocking_error_codes"] = sorted({f.code for f in readiness.blocking_findings})
    report.stats["main_text_exempted_error_count"] = len(readiness.exempted_findings)
    report.stats["main_text_exempted_error_codes"] = sorted({f.code for f in readiness.exempted_findings})

"""
Bodyless-heading repair pass (post-classification, pre-validation).

Prevention (pdf_extractor.py's `_OcrHeadingScorer`) keeps most figure
labels, diagram annotations, table-cell fragments, and unsupported OCR
fragments out of the heading stream in the first place, by requiring
multiple independent corroborating signals before a line from OCR is ever
promoted to a `HeadingBlock`. This module is the second line of defense for
whatever still gets through: a run of consecutive, bodyless headings is the
structural signature of exactly the failure mode a single-line-at-a-time
scorer cannot fully rule out (e.g. a caption/annotation that scored just high
enough to pass in isolation, but whose neighbors reveal it was never a real
section).

Scope is deliberately narrow and format-agnostic:
  * Only headings that came from OCR (`style["from_ocr"]`) are ever touched
    -- a native PDF/DOCX heading has real font metadata behind it and is out
    of scope for this pass entirely.
  * Only headings that are structurally "bodyless" (immediately followed by
    another heading of equal-or-shallower level, or by nothing at all) are
    considered -- a heading immediately followed by a *deeper* heading is a
    legitimate parent/container and is left alone (see "Preserve legitimate
    container headings").
  * Every candidate, regardless of how long a bodyless run it sits in, is
    judged by the same per-member evidence: only a heading that is either
    weakly-scored (below `OCR_HEADING_CONFIDENT_SCORE`) or structurally
    label/caption/annotation/table-cell-shaped is demoted. Clustering
    (sitting in a 2+-member run) is *not* by itself disqualifying evidence
    for an individual member -- a run of chart axis/data-point labels (e.g.
    "O exponential", "O quadratic", "O linear-quadratic") is still caught
    because each one is independently weak/label-shaped, but a genuine
    numbered section heading ("2.4 Constant time operations") that happens
    to sit at the tail of a run started by weak neighbors (e.g. a
    "Participation Activity"-style UI prompt immediately before it) is not
    swept up just for being adjacent to them. An earlier version of this
    pass blanket-demoted every OCR member of a 2+ run regardless of its own
    score or shape on the theory that clustering alone was disqualifying;
    in practice that destroyed real chapter/section headings that happened
    to close out a run of genuinely bad neighbors (see
    `test_regression_fixtures.py` / the CSC506 regression fixture, where
    "2.3 Python: Linear and binary search", "2.5 O notation", etc. were all
    wrongly demoted to `UnknownBlock`). A confidently-scored, unlabeled
    heading with genuinely no body at all is a real document defect (spec:
    "Do not blindly collapse every bodyless heading") and is left for
    `validation.py`'s existing `empty_section` gate to report honestly
    instead of being silently papered over here.

Demoted headings are replaced in place (same `block_id`, `order`, `page`,
text, and source order) with the typed block their text shape implies
(`CaptionBlock` / `FigureLabelBlock` / `DiagramAnnotationBlock` /
`TableCellBlock` / `UnknownBlock`), so they stay searchable and keep their
page/source-region provenance -- they simply fall out of the heading tree
and become ordinary content of whichever real heading precedes them once
`build_sections`/`render_markdown`/`validate` are re-run on the mutated
block list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .model import (
    Document, HeadingBlock, CaptionBlock, FigureLabelBlock, DiagramAnnotationBlock,
    TableCellBlock, UnknownBlock,
)
from .findings import FindingLog
from .extractors import classify_ocr_nonheading_kind, OCR_HEADING_CONFIDENT_SCORE

_KIND_CTORS = {
    "caption": CaptionBlock,
    "figure_label": FigureLabelBlock,
    "diagram_annotation": DiagramAnnotationBlock,
    "table_cell": TableCellBlock,
    "unknown": UnknownBlock,
}


@dataclass
class RepairStats:
    bodyless_heading_chains_detected: int = 0
    bodyless_heading_chains_repaired: int = 0
    ocr_headings_demoted: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "bodyless_heading_chains_detected": self.bodyless_heading_chains_detected,
            "bodyless_heading_chains_repaired": self.bodyless_heading_chains_repaired,
            "ocr_headings_demoted": self.ocr_headings_demoted,
        }


def _is_weak_ocr_heading(block: HeadingBlock) -> bool:
    if not block.style.get("from_ocr"):
        return False
    score = block.style.get("ocr_heading_score", OCR_HEADING_CONFIDENT_SCORE)
    return score < OCR_HEADING_CONFIDENT_SCORE


def _find_bodyless_runs(blocks: List) -> List[List[int]]:
    """Maximal runs of consecutive headings with no body between any of
    them: each transition heading[k] -> heading[k+1] within a run has
    nothing in between (they are adjacent in the flat block list) and is
    not a parent -> deeper-child relationship. Membership in a run is about
    *adjacency*, not about what eventually follows the last member -- a
    trailing "Output Image" heading immediately followed by real paragraph
    content still had nothing between *it* and the "Input Image" heading
    right before it, so it belongs in the same cluster and gets evaluated
    for demotion just like its neighbor, regardless of what comes after it.

    A single heading not adjacent to another heading only forms its own
    (length-1) run when nothing follows it at all (true end-of-document
    bodylessness) -- a lone heading followed by ordinary content has a real
    body and is never a repair candidate. A heading directly followed by a
    *deeper* heading is a container with real child sections and never
    starts or extends a run."""
    runs: List[List[int]] = []
    i = 0
    n = len(blocks)
    while i < n:
        if not isinstance(blocks[i], HeadingBlock):
            i += 1
            continue
        j = i
        while (
            j + 1 < n and isinstance(blocks[j + 1], HeadingBlock)
            and blocks[j + 1].level <= blocks[j].level
        ):
            j += 1
        if j > i:
            runs.append(list(range(i, j + 1)))
            i = j + 1
        else:
            if i == n - 1:
                runs.append([i])
            i += 1
    return runs


def repair_bodyless_heading_chains(document: Document, log: FindingLog) -> Dict[str, int]:
    """Mutates `document.blocks` in place, demoting unsupported OCR headings
    found in bodyless chains to their appropriate typed block. Returns the
    repair-pass metrics (spec "Validation" section)."""
    stats = RepairStats()
    blocks = document.blocks
    runs = _find_bodyless_runs(blocks)

    for run in runs:
        # Every structurally bodyless run found counts as "detected", even
        # if it turns out nothing in it needs repair (e.g. it's entirely
        # native-PDF/DOCX headings, or every member is confidently-scored
        # and unlabeled) -- "repaired" below is the subset actually fixed.
        stats.bodyless_heading_chains_detected += 1

        members = [blocks[i] for i in run]
        # A genuine multi-heading run (2+) is disqualifying by clustering
        # alone; a length-one run needs its own weak-score or label-shape
        # evidence (see module docstring). The document title (H1) has its
        # own resolution rules (spec section 11) and is exempt from the
        # body-heading classifier everywhere else in this pipeline too --
        # never a demotion target, regardless of run length.
        if len(run) >= 2:
            candidates = [
                (i, b) for i, b in zip(run, members)
                if b.level > 1 and b.style.get("from_ocr")
            ]
        else:
            candidates = [
                (i, b) for i, b in zip(run, members)
                if b.level > 1 and b.style.get("from_ocr")
                and (_is_weak_ocr_heading(b) or classify_ocr_nonheading_kind(b.text) != "unknown")
            ]
        if not candidates:
            continue

        repaired_any = False
        for i, b in candidates:
            # No further per-member guard here: the outer `candidates`
            # filter above already encodes the len(run)==1-vs-2+ distinction
            # (a length-one candidate only ever reaches this point if it was
            # already weak-scored or label-shaped; a 2+-run candidate is
            # here because the run itself is the evidence). `kind` falls
            # back to "unknown" -> UnknownBlock when the text doesn't match
            # any specific figure/diagram/table-cell/caption shape, which
            # keeps it searchable without mis-typing it.
            kind = classify_ocr_nonheading_kind(b.text)
            ctor = _KIND_CTORS[kind]
            replacement = ctor(
                block_id=b.block_id, order=b.order, source_format=b.source_format,
                raw=b.raw, page=b.page, text=b.text, normalized=b.normalized or b.text,
            )
            replacement.style = dict(b.style)
            replacement.style["demoted_from_heading"] = True
            replacement.style["demoted_reason"] = "bodyless_heading_chain"
            blocks[i] = replacement
            stats.ocr_headings_demoted += 1
            repaired_any = True
            log.fixed(
                "bodyless_heading_demoted",
                f"Demoted bodyless-chain heading {b.text!r} to a {ctor.__name__} "
                f"(source region {b.block_id}, page {b.page})",
                block_id=b.block_id, page=b.page,
            )
        if repaired_any:
            stats.bodyless_heading_chains_repaired += 1

    return stats.to_dict()

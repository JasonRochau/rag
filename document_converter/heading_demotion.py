"""
Active false-heading demotion pass, run for every extractor's output.

`validation.py`'s `heading_rejection_reason` classifier already recognizes
every false-heading shape found in the layout PoC (repeated UI-chrome labels,
pseudocode-step fragments, captions, equations/matrices, ...) -- but only to
*flag* it after the fact (an `ERROR`/`WARNING` finding on an otherwise
unmodified `HeadingBlock`). The legacy PDF/OCR path originally went without
an active demotion pass keyed off that classifier because its own
font-size/isolation gating (`pdf_extractor.py`'s `_OcrHeadingScorer`) keeps
most of this shape of text out of the heading stream in the first place, and
what still gets through is caught by `heading_repair.py`'s bodyless-chain
pass -- which only touches OCR-sourced headings (`style["from_ocr"]`).
Experience showed that net still leaks (verbatim-repeated boilerplate
headings on otherwise "clean" legacy-path documents, checkbox-shaped
headings like the Apple 10-K's "Yes ☐"), so this pass now runs on the
shared pipeline path for both extractors, complementing rather than
replacing the OCR-scoped gating.

This module is the parser-neutral demotion pass promised as PoC condition 1,
built from two signals, not one:

  1. `heading_rejection_reason`'s text-shape classifier -- catches
     pseudocode-step fragments, equation/matrix lines, captions, and other
     clearly-invalid-shaped text (fixture 2's 9 false headings, and part of
     fixture 1's). *Acts* on a hard-reject verdict instead of only recording
     a finding, the way `validation.py` does.
  2. Verbatim-repetition: a non-title heading whose exact (whitespace-
     normalized) text recurs 3+ times across the document. This is
     necessary, not optional -- fixture 1's dominant false-heading pattern
     (interactive-courseware UI chrome: "Animation content:", "PARTICIPATION
     ACTIVITY", "Check", ...) is grammatically fine short text that
     `heading_rejection_reason` correctly does *not* reject on shape alone
     (see that function's own docstring: isolated short phrases are valid
     headings elsewhere in a document, and over-rejecting them would be the
     spec's warned-against "isolated short line is invalid" over-reach).
     What actually marks them false is that the identical label appears
     dozens of times, never carrying section-distinguishing identity --
     the same repetition signal this PoC's own `check_criteria.py` mechanical
     count already uses to grade "false heading", so this closes the fix
     against the same yardstick the defect was measured with, not a looser
     one. Still fully parser/document-neutral: no fixture-specific text is
     matched, only the general shape "identical heading text, 3+ times".

Intended to run exactly once per document, wired into the shared
`_build_document()` stage chain in `pipeline.py` (every extractor, not just
Docling): after `normalize_document`, running-header repair, semantic
structuring, and bodyless-heading repair, and before the heading-sibling /
section-numbering audits, so a demoted block is ordinary content (never a
heading) for the audits and every downstream stage. The pass is
idempotent-safe (demoted blocks are no longer headings, repaired text no
longer matches the duplication shape, re-leveled subheadings no longer
mis-nest), but the pipeline is the single call site by design.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from .model import Block, Document, HeadingBlock, CodeBlock, ImageBlock, UnknownBlock
from .findings import FindingLog
from .extractors import (
    heading_rejection_reason, numbering_tuple, numbering_prefix_remainder,
    looks_like_algorithm_line, looks_like_caption_text, looks_like_title_candidate,
)

_REPEAT_THRESHOLD = 3


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _block_text(b: Block) -> str:
    return (getattr(b, "text", None) or getattr(b, "normalized", None) or "").strip()


# --------------------------------------------------------------------- #
# Procedural / pseudocode step-label demotion.
#
# `heading_rejection_reason`'s short-colon carve-out ("Abstract:" must stay a
# heading) means a short (<=4 word) colon-terminated line, or a "Step N"
# label, is never hard-rejected on shape alone -- correctly so, since either
# shape is common to plenty of genuine short headings. What actually marks a
# candidate like "Step 2) Pop:" or "Build permutation matrix:" false is
# *context* no single line carries: it sits inside a run of algorithm/
# pseudocode content (immediately touching a code block or an
# unrendered-formula region) alongside *other* short procedural labels doing
# the same thing -- a genuine section heading never clusters three-in-a-row
# with nothing but a formula/code block between each. Both conditions
# (algorithm-adjacent, and not alone) are required so a single legitimate
# short heading that happens to introduce one code/equation block (e.g. a
# "Constraint:" heading over its one formula) is never touched.
_STEP_LABEL_RE = re.compile(r"^step\s+\d+\s*[):.]?", re.IGNORECASE)


def _looks_like_procedural_label_shape(text: str) -> bool:
    words = text.split()
    if not words:
        return False
    if _STEP_LABEL_RE.match(text):
        return True
    return text.endswith(":") and len(words) <= 4


def _is_bare_image(b: Optional[Block]) -> bool:
    """True for an `ImageBlock` with no alt/caption content of its own --
    an icon or decorative glyph, structurally incapable of being the
    "figure" a real section heading would introduce or follow. Adjacency
    to one is therefore not, by itself, evidence a heading is real *or*
    false; it only matters combined with the heading's own text already
    reading as UI chrome (see `_ui_chrome_label_reason`)."""
    return isinstance(b, ImageBlock) and not (b.alt or "").strip()


def _ui_chrome_label_reason(blocks: List[Block], i: int) -> Optional[str]:
    """A heading whose text is UI-chrome-shaped (`looks_like_title_candidate`
    already says no -- a shouted single word or a short imperative
    interaction verb like "Check"/"Continue") *and* sits immediately next to
    a bare, contentless image (an icon, not a figure) is a widget button/
    label caught by the parser's heading-level font sizing on this
    particular page, not a section title. Both conditions are required:
    chrome-shaped text alone is common and often a legitimate short heading
    elsewhere in a document (spec's own "isolated short line is invalid" is
    the over-reach to avoid); a bare icon alone proves nothing either."""
    b = blocks[i]
    text = _block_text(b)
    if looks_like_title_candidate(text):
        return None
    prev_b = blocks[i - 1] if i > 0 else None
    next_b = blocks[i + 1] if i + 1 < len(blocks) else None
    if _is_bare_image(prev_b) or _is_bare_image(next_b):
        return "ui_chrome_label"
    return None


def _is_algorithm_shaped_block(b: Optional[Block]) -> bool:
    if b is None:
        return False
    if isinstance(b, CodeBlock):
        return True
    if isinstance(b, UnknownBlock) and b.style.get("parser_region_type") == "formula":
        return True
    return looks_like_algorithm_line(_block_text(b))


def _procedural_cluster_positions(blocks: List[Block], body_headings) -> set:
    """Returns the set of `body_headings` list-positions (not block indices)
    that are both procedural-label-shaped and algorithm-block-adjacent, and
    have at least one other such position immediately before/after them in
    heading order -- the "repeated procedural labels in the same section"
    signal, computed once so the per-heading loop below can just look it
    up."""
    qualifies = {}
    for pos, (i, b) in enumerate(body_headings):
        text = _block_text(b)
        if not _looks_like_procedural_label_shape(text):
            continue
        prev_blk = blocks[i - 1] if i > 0 else None
        next_blk = blocks[i + 1] if i + 1 < len(blocks) else None
        if _is_algorithm_shaped_block(prev_blk) or _is_algorithm_shaped_block(next_blk):
            qualifies[pos] = True
    return {pos for pos in qualifies if (pos - 1) in qualifies or (pos + 1) in qualifies}


# --------------------------------------------------------------------- #
# Garbled/malformed heading text (conservative: repair when the content is
# safely reconstructable, demote when it is not -- never invent a title).
#
# Docling's reading-order reconstruction is observed (both fixtures) to
# occasionally emit the *same* heading's text twice back-to-back (two
# overlapping layout regions read as one), and separately to occasionally
# split a single numbered heading into a corrupted numbered fragment
# immediately followed by a second, unnumbered HeadingBlock carrying the
# real (clean) title text. These are different shapes needing different
# handling: the first is a duplicate of content already known good (safe to
# collapse in place); the second's first block carries no reliably
# recoverable title of its own (safe only to drop, keeping the clean
# second heading as the section's real title).
def _dedupe_self_repeated_heading_text(text: str) -> Optional[str]:
    """If `text` is, character-for-character (ignoring digits/punctuation/
    whitespace/case), two back-to-back copies of the same content, returns
    the text truncated to just the first copy; otherwise None. Generic
    duplicate-layout-region detector -- no vocabulary, no fixture text."""
    words = text.split()
    if len(words) < 6:
        return None
    alpha = re.sub(r"[^a-z]", "", text.lower())
    n = len(alpha)
    if n < 8 or n % 2 != 0:
        return None
    half = n // 2
    if alpha[:half] != alpha[half:]:
        return None
    cum = 0
    for idx, w in enumerate(words):
        cum += len(re.sub(r"[^a-z]", "", w.lower()))
        if cum >= half:
            return " ".join(words[: idx + 1]).strip()
    return None


# A remainder that itself starts with another numbering-shaped token
# ("2.7F", "3.1") is a duplicated/garbled outline prefix, not a title --
# the same duplicate-layout-region phenomenon as above, just landing on the
# numbering instead of the words.
_NUMBERING_LIKE_TOKEN_RE = re.compile(r"^\d{1,2}(?:\.\d{1,3}){1,4}[A-Za-z]?[.:)]?$")


def _garbled_numbered_heading_reason(blocks: List[Block], i: int) -> Optional[str]:
    """Only ever considered for a heading `numbering_tuple` already exempted
    from `heading_rejection_reason` (see `demote_false_headings`), and only
    acted on when it is structurally orphaned -- immediately followed by
    another heading at an equal-or-shallower level, i.e. it contributes no
    body of its own (spec's bodyless-chain shape) -- so a numbered heading
    that merely has unusual-looking (but real) title text is never touched
    on text shape alone."""
    b = blocks[i]
    text = _block_text(b)
    nxt = blocks[i + 1] if i + 1 < len(blocks) else None
    if not (isinstance(nxt, HeadingBlock) and nxt.level <= b.level):
        return None
    remainder = numbering_prefix_remainder(text)
    if not remainder:
        return None
    reason = heading_rejection_reason(remainder)
    if reason is not None:
        return f"garbled_heading_fragment_{reason}"
    if _NUMBERING_LIKE_TOKEN_RE.match(remainder.split()[0]):
        return "garbled_heading_fragment_duplicated_numbering_token"
    return None


# --------------------------------------------------------------------- #
# Mis-leveled subheading repair.
#
# The Docling adapter assigns every *unnumbered* heading a flat level (its
# own numbering-free default), independent of what precedes it. When an
# unnumbered heading immediately follows a numbered one with no content of
# its own in between, and its assigned level is shallower-or-equal to that
# numbered heading's level, the numbered heading reads as contributing no
# body -- an `empty_section` structural error -- even though it plainly has
# real content: the unnumbered heading right after it, mis-leveled as a
# sibling instead of a child. Promoting that heading's level to
# `parent.level + 1` is a safe, generic repair (spec's own bodyless-chain
# rule: "a heading immediately followed by a *deeper* heading is a
# legitimate parent/container, never bodyless") -- it changes no heading's
# text or existence, only its place in the hierarchy.
def _repair_misleveled_subheadings(blocks: List[Block], log: FindingLog) -> int:
    repaired = 0
    for i, b in enumerate(blocks):
        if not isinstance(b, HeadingBlock) or b.level <= 1:
            continue
        if numbering_tuple(_block_text(b)) is None:
            continue
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        if not isinstance(nxt, HeadingBlock):
            continue
        if numbering_tuple(_block_text(nxt)) is not None:
            continue  # the next heading has its own numbering -- a real sibling, not a mis-leveled child
        if nxt.level > b.level:
            continue  # already a legitimate deeper child
        old_level = nxt.level
        nxt.level = min(b.level + 1, 6)
        repaired += 1
        log.fixed(
            "misleveled_subheading_repaired",
            f"Promoted heading {nxt.text!r} from level {old_level} to level {nxt.level} "
            f"so it nests under {b.text!r} instead of appearing to close it out with no body "
            f"(source region {nxt.block_id}, page {nxt.page})",
            block_id=nxt.block_id, page=nxt.page,
        )
    return repaired


@dataclass
class HeadingDemotionStats:
    headings_checked: int = 0
    headings_demoted: int = 0
    demoted_by_reason: Dict[str, int] = None
    headings_text_repaired: int = 0
    subheadings_releveled: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "headings_checked": self.headings_checked,
            "headings_demoted": self.headings_demoted,
            "demoted_by_reason": dict(self.demoted_by_reason or {}),
            "headings_text_repaired": self.headings_text_repaired,
            "subheadings_releveled": self.subheadings_releveled,
        }


def demote_false_headings(document: Document, log: FindingLog) -> Dict[str, int]:
    """Mutates `document.blocks` in place. Runs, in order:

    1. Self-duplicated heading text repair -- an exact-duplicate-content
       heading (`_dedupe_self_repeated_heading_text`) is safely
       reconstructable, so its text is collapsed to a single copy in place;
       it stays a heading.
    2. False-heading demotion: any non-title `HeadingBlock` (level > 1 -- the
       document title has its own resolution rules, spec section 11, and is
       exempt from every body-heading classifier in this pipeline, same rule
       `heading_repair.py` follows) that trips any of:
         a. `heading_rejection_reason`'s hard-reject rules (unnumbered only,
            see the numbering exemption below),
         b. exact text recurring `_REPEAT_THRESHOLD` (3) or more times among
            this document's non-title headings ("repeated_boilerplate"),
         c. the procedural-pseudocode-label cluster shape (short/"Step N",
            algorithm-block-adjacent, not alone -- see
            `_procedural_cluster_positions`),
         d. a numbered heading whose text is structurally orphaned (another
            heading immediately follows with no body) *and* whose title
            portion (after its own numbering) is unreconstructable --
            shape-rejected or itself a duplicated numbering token (see
            `_garbled_numbered_heading_reason`),
         e. a bodyless heading immediately preceded by caption-shaped text
            (a mis-split table-header cell, not a section title),
       is replaced, in place, with an `UnknownBlock` carrying the same
       `block_id`/`order`/`page`/text so it stays searchable and keeps its
       source-region provenance -- it simply falls out of the heading tree
       before `build_sections`/`render_markdown`/`validate` ever run
       against it.
    3. Mis-leveled-subheading repair (`_repair_misleveled_subheadings`),
       against the post-demotion block list, so a numbered heading whose
       real content is a mis-leveled child heading (not a false heading at
       all) nests correctly instead of reading as empty.
    """
    stats = HeadingDemotionStats(demoted_by_reason={})
    blocks = document.blocks
    body_headings = [(i, b) for i, b in enumerate(blocks) if isinstance(b, HeadingBlock) and b.level > 1]

    for _, b in body_headings:
        deduped = _dedupe_self_repeated_heading_text(_block_text(b))
        if deduped is not None and deduped != _block_text(b):
            old_text = b.text
            b.text = deduped
            b.normalized = deduped
            stats.headings_text_repaired += 1
            log.fixed(
                "duplicated_heading_text_repaired",
                f"Collapsed duplicated heading text {old_text!r} to {deduped!r} "
                f"(source region {b.block_id}, page {b.page})",
                block_id=b.block_id, page=b.page,
            )

    text_counts = Counter(_norm(b.text) for _, b in body_headings)
    repeated_boilerplate = {t for t, c in text_counts.items() if c >= _REPEAT_THRESHOLD}
    procedural_cluster = _procedural_cluster_positions(blocks, body_headings)

    for pos, (i, b) in enumerate(body_headings):
        stats.headings_checked += 1
        text = (b.text or b.normalized or "").strip()
        # A valid numeric outline prefix ("4.1 Pre-Order Tree Traversal") is
        # strong independent structural evidence a heading is real -- the
        # same precedent `validation.py`'s own context checks and this
        # adapter's own `_heading_level` already give numbering (see
        # docling_adapter.py). It exempts a numbered heading from the
        # *shape*-based rejection only: Docling's reading-order
        # reconstruction is observed (both fixtures, phase6/8 reports) to
        # occasionally glue/duplicate a numbered heading's own text, which
        # trips `too_many_words`/`too_long_or_empty` despite the heading
        # being genuine -- self-duplication is handled above (repair, not
        # rejection), and `_garbled_numbered_heading_reason` below is the
        # narrower, structurally-gated check for the shape that genuinely
        # is unrecoverable. Repetition-based demotion still applies to a
        # numbered heading (below) -- numbering never excuses a label that
        # verbatim-repeats 3+ times, which no genuine numbered section does.
        is_numbered = numbering_tuple(text) is not None
        reason = None if is_numbered else heading_rejection_reason(text)
        if reason is None and _norm(text) in repeated_boilerplate:
            reason = "repeated_boilerplate"
        if reason is None and pos in procedural_cluster:
            reason = "pseudocode_procedural_label"
        if reason is None and is_numbered:
            reason = _garbled_numbered_heading_reason(blocks, i)
        if reason is None and not is_numbered:
            reason = _ui_chrome_label_reason(blocks, i)
        if reason is None:
            prev_b = blocks[i - 1] if i > 0 else None
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            is_bodyless = isinstance(nxt, HeadingBlock) and nxt.level <= b.level
            if is_bodyless and prev_b is not None and looks_like_caption_text(_block_text(prev_b)):
                reason = "table_header_fragment"
            # A heading immediately following a block *this same pass* just
            # demoted as a table-header fragment is the next column header
            # in the same mis-extracted table ("Operation" / "Example"
            # pairs) -- the chain, not just the first link, is the false
            # heading; this reuses this pass's own output rather than a
            # second independent guess at the shape.
            elif (
                prev_b is not None and isinstance(prev_b, UnknownBlock)
                and prev_b.style.get("demoted_reason") == "table_header_fragment"
            ):
                reason = "table_header_fragment"
        if reason is None:
            continue

        replacement = UnknownBlock(
            block_id=b.block_id, order=b.order, source_format=b.source_format,
            raw=b.raw, page=b.page, text=b.text, normalized=b.normalized or b.text,
        )
        replacement.style = dict(b.style)
        replacement.style["demoted_from_heading"] = True
        replacement.style["demoted_reason"] = reason
        blocks[i] = replacement
        stats.headings_demoted += 1
        stats.demoted_by_reason[reason] = stats.demoted_by_reason.get(reason, 0) + 1
        log.fixed(
            "false_heading_demoted",
            f"Demoted heading {b.text!r} to UnknownBlock ({reason}) "
            f"(source region {b.block_id}, page {b.page})",
            block_id=b.block_id, page=b.page,
        )

    stats.subheadings_releveled = _repair_misleveled_subheadings(blocks, log)
    return stats.to_dict()

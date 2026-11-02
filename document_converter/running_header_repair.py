"""
Running-header/footer removal (generalization fixture: repeated publication/
conference lines recurring at page boundaries).

Purely structural, parser-neutral detection: no phrase bank, no vocabulary.
A block's normalized text counts as a running header/footer only when *all*
of the following hold:

  1. It recurs verbatim (whitespace/case-normalized) on 2+ distinct pages.
  2. Every one of those occurrences sits within the first/last
     `_BOUNDARY_WINDOW` blocks of its own page (top-of-page or
     bottom-of-page position) -- ordinary body content repeating a phrase
     mid-page does not qualify, no matter how often it recurs.

This is deliberately the same "identical text, repeated, carries no
section-distinguishing identity" evidence `heading_demotion.py` already uses
for boilerplate headings, just computed over every block type instead of
only headings, and gated on page-boundary position instead of a heading
hierarchy -- a page-boundary running header has no heading-shaped signal to
key off in the first place (spec's own front-matter conference/copyright
lines are ordinary paragraph text, not headings).

One occurrence is always preserved -- whichever one is already a
front-matter-typed block (`FRONT_MATTER_BLOCK_TYPES`, e.g. a
`ConferenceMetadataBlock` `normalize.py`'s front-matter reconstruction or
`pdf_extractor.py`'s extraction-time classifier already produced), or
failing that, the first occurrence in document order. Every other
occurrence is removed from body flow entirely (never left as a stray
duplicate paragraph, never merged into surrounding prose) and logged, the
same "excised and dropped, retained in the audit log" pattern
`paragraph_continuity_repair.py` already uses for copyright/permission
boilerplate.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from .model import Block, Document, FRONT_MATTER_BLOCK_TYPES, HeadingBlock
from .findings import FindingLog

_REPEAT_THRESHOLD = 2
_BOUNDARY_WINDOW = 2


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _block_text(b: Block) -> str:
    return (getattr(b, "text", None) or getattr(b, "normalized", None) or "").strip()


@dataclass
class RunningHeaderStats:
    running_headers_removed: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {"running_headers_removed": self.running_headers_removed}


def repair_running_headers(document: Document, log: FindingLog) -> Dict[str, int]:
    """Mutates `document.blocks` in place. Never touches a `HeadingBlock`
    (a repeated heading is a different defect class, already owned by
    `heading_demotion.py`'s repeated_boilerplate rule where it applies) --
    only plain content blocks (paragraphs, front-matter-typed lines, etc.)
    are eligible."""
    stats = RunningHeaderStats()
    blocks = document.blocks

    pages_order: Dict[int, List[int]] = defaultdict(list)
    for i, b in enumerate(blocks):
        if b.page is not None:
            pages_order[b.page].append(i)

    def is_boundary(i: int) -> bool:
        b = blocks[i]
        idxs = pages_order.get(b.page, [])
        if not idxs:
            return False
        pos = idxs.index(i)
        return pos < _BOUNDARY_WINDOW or pos >= len(idxs) - _BOUNDARY_WINDOW

    occurrences: Dict[str, List[int]] = defaultdict(list)
    for i, b in enumerate(blocks):
        if isinstance(b, HeadingBlock):
            continue
        text = _block_text(b)
        if not text:
            continue
        occurrences[_norm(text)].append(i)

    to_remove: set = set()
    for norm_text, idxs in occurrences.items():
        if len(idxs) < _REPEAT_THRESHOLD:
            continue
        distinct_pages = {blocks[i].page for i in idxs}
        if len(distinct_pages) < _REPEAT_THRESHOLD:
            continue
        if not all(is_boundary(i) for i in idxs):
            continue
        keep_idx = next((i for i in idxs if isinstance(blocks[i], FRONT_MATTER_BLOCK_TYPES)), idxs[0])
        for i in idxs:
            if i == keep_idx:
                continue
            to_remove.add(i)
            stats.running_headers_removed += 1
            log.fixed(
                "running_header_removed",
                f"Removed repeated running header/footer text from body flow, preserving the "
                f"occurrence at {blocks[keep_idx].block_id!r} (source region {blocks[i].block_id}, "
                f"page {blocks[i].page}): {_block_text(blocks[i])!r}",
                block_id=blocks[i].block_id, page=blocks[i].page,
            )

    if to_remove:
        document.blocks = [b for i, b in enumerate(blocks) if i not in to_remove]
        for j, b in enumerate(document.blocks):
            b.order = j

    return stats.to_dict()

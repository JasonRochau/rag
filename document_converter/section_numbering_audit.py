"""
Repeated top-level section-numbering audit (generalization fixture:
"five distinct top-level headings all beginning with 1." class of
corruption).

Purely structural: looks only at the numeric outline prefix each top-level
heading (`numbering_tuple` of length 1 -- "1 Introduction", "2 Methods", the
same "top-level numbered heading" concept `heading_sibling_audit.py` already
uses) already carries and where it sits in document order, never at the
document's own expected/known section names.

Two or more distinct top-level headings sharing the same number is always
flagged (`repeated_top_level_section_numbering`, an ERROR -- this is
numbering corruption or a genuine outline defect either way). It is only
*repaired* -- renumbered 1..N in document order -- when the evidence is
unambiguous: every single top-level numbered heading in the document shares
that one number, so there is no other, differently-numbered heading whose
existence would make "what should this really be numbered" a guess. The
moment a second, distinct number is also present, renumbering would have to
guess which heading belongs where in a sequence this pass cannot actually
reconstruct -- so it only flags, leaving the ambiguous case for manual
review rather than rewriting numbering from anything about the document's
actual (expected) content.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .model import Document, HeadingBlock
from .findings import FindingLog
from .extractors import numbering_tuple, numbering_prefix_remainder

# ASCII-only digits ([0-9], not \d): Unicode decimal digits such as the
# fullwidth '７' (U+FF17) that icon fonts emit for glyph bullets (Microsoft
# Learn's "docons" Note icon) are not real section numbers -- parsing them
# as one makes repeated callouts look like repeated top-level numbering.
_LEADING_NUMBER_RE = re.compile(r"^([0-9]{1,2})(\s*[.):]?\s*)")
# Some parsers (observed: Docling) occasionally glue a heading's numbering
# directly onto the following word with no separating space ("3.2Tracking
# Vertex Visit Order"). `numbering_tuple`'s prefix regex then backtracks to
# a shorter match than the real number ("3" instead of "3.2", since "3."
# alone satisfies its punctuation-boundary check) -- a text-corruption
# artifact of the source, not a genuinely single-component top-level
# heading. The remainder left after that short match starts with the
# missing digit run glued straight to a letter ("2Tracking...") -- a
# reliable, generic tell that the "top-level" number this parsed to isn't
# real, so this heading is excluded from the audit entirely rather than
# treated as evidence of numbering corruption.
_GLUED_SUBNUMBER_RE = re.compile(r"^\d+[A-Za-z]")


@dataclass
class SectionNumberingStats:
    repeated_number_groups_flagged: int = 0
    headings_renumbered: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "repeated_number_groups_flagged": self.repeated_number_groups_flagged,
            "headings_renumbered": self.headings_renumbered,
        }


def _top_level_numbered(document: Document) -> List[Tuple[int, HeadingBlock, int]]:
    out = []
    for i, b in enumerate(document.blocks):
        if not isinstance(b, HeadingBlock) or b.level <= 1:
            continue
        text = (b.text or b.normalized or "").strip()
        nt = numbering_tuple(text)
        if nt is None or len(nt) != 1:
            continue
        remainder = numbering_prefix_remainder(text) or ""
        if _GLUED_SUBNUMBER_RE.match(remainder):
            continue
        out.append((i, b, nt[0]))
    return out


def _split_into_segments(numbered: List[Tuple[int, HeadingBlock, int]]) -> List[List[Tuple[int, HeadingBlock, int]]]:
    """Split the document-ordered list of top-level numbered headings into
    segments at numbering restarts: a heading numbered 0 or 1 that follows a
    higher number starts a new segment (an anthology's next article). Any
    other decrease (1,2,3,5,3,6) is kept in the segment so the repeat is still
    flagged as corruption. An all-same-number run (1,1,1,1) stays one segment
    because no number ever exceeds the max."""
    segments: List[List[Tuple[int, HeadingBlock, int]]] = []
    current: List[Tuple[int, HeadingBlock, int]] = []
    current_max: Optional[int] = None
    for item in numbered:
        n = item[2]
        if current_max is not None and n <= 1 and n < current_max:
            segments.append(current)
            current = [item]
            current_max = n
        else:
            current.append(item)
            if current_max is None or n > current_max:
                current_max = n
    if current:
        segments.append(current)
    return segments


def _audit_segment(
    segment: List[Tuple[int, HeadingBlock, int]], log: FindingLog, stats: SectionNumberingStats,
) -> None:
    """Run the original document-wide repeated-numbering logic on one segment.
    Mutates heading blocks in place only in the unambiguous all-same-number
    repair case."""
    if len(segment) < 2:
        return

    by_number: Dict[int, List[int]] = defaultdict(list)
    for pos, (_, _, n) in enumerate(segment):
        by_number[n].append(pos)
    repeated = {n: positions for n, positions in by_number.items() if len(positions) > 1}
    if not repeated:
        return

    stats.repeated_number_groups_flagged += len(repeated)
    unambiguous_repair = len(by_number) == 1

    if not unambiguous_repair:
        for n, positions in repeated.items():
            for pos in positions:
                _, b, _ = segment[pos]
                log.error(
                    "repeated_top_level_section_numbering",
                    f"Heading {b.text!r} shares top-level number {n} with {len(positions) - 1} other "
                    f"heading(s) -- numbering corruption suspected; left unrepaired (other, "
                    f"differently-numbered top-level headings are also present in this segment, "
                    f"so the correct sequence can't be inferred from document order alone)",
                    block_id=b.block_id, page=b.page,
                )
        return

    for seq, (_, b, old_num) in enumerate(segment, start=1):
        if old_num == seq:
            continue
        text = (b.text or b.normalized or "").strip()
        m = _LEADING_NUMBER_RE.match(text)
        if not m:
            continue
        new_text = f"{seq}{m.group(2)}{text[m.end():]}"
        old_text = b.text
        b.text = new_text
        b.normalized = new_text
        stats.headings_renumbered += 1
        log.fixed(
            "repeated_top_level_numbering_repaired",
            f"Renumbered heading {old_text!r} to {new_text!r} (document-order position {seq} of "
            f"{len(segment)}; every top-level numbered heading in this segment shared number "
            f"{old_num}, so document order is the only, unambiguous evidence for the real sequence)",
            block_id=b.block_id, page=b.page,
        )


def audit_repeated_top_level_numbering(document: Document, log: FindingLog) -> Dict[str, int]:
    """Mutates `document.blocks` in place only in the unambiguous-repair
    case described in the module docstring; otherwise only logs. Returns
    audit metrics."""
    stats = SectionNumberingStats()
    numbered = _top_level_numbered(document)
    if len(numbered) < 2:
        return stats.to_dict()

    segments = _split_into_segments(numbered)
    for segment in segments:
        _audit_segment(segment, log, stats)

    return stats.to_dict()

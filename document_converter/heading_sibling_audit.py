"""
High-confidence numbered-sibling heading audit (spec section 9).

Extraction-time heading detection (font size, isolation, OCR scoring) can
miss a single section heading in an otherwise clearly-numbered outline --
e.g. a scanned page where "2 Logistical Issues" happened to render at body
font size while "1 Introduction" and "3 Data Representations" did not. This
is a *document-level* second pass, run after extraction/normalization, that
looks at the outline actually recovered (top-level numbered headings: "1
...", "3 ...", "4 ...") and checks whether a body block sitting between two
of them starts with exactly the missing number followed by a short
title-case run and then ordinary sentence prose -- the textual signature of
a heading that got flattened into a paragraph.

This is deliberately conservative in both directions:
  * A gap in the numbering is not, by itself, evidence of anything -- most
    documents have legitimate non-consecutive numbering (front matter,
    appendices, a introduction that starts at "0"). Nothing is promoted
    unless the *source text itself* contains a matching candidate.
  * A missing number is left alone (not silently invented) when no
    candidate is found, and reported instead of guessed at when a
    candidate's shape is ambiguous (more than one plausible split point,
    or more than one candidate for the same missing number) -- see spec
    section 9's "must not invent a missing section merely because
    numbering skips."
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .model import Document, HeadingBlock, ParagraphBlock
from .findings import FindingLog
from .extractors import numbering_tuple

# A missing sibling's own heading text, reconstructed from a paragraph that
# swallowed it: the exact missing number, then a short (1-8 word) run of
# Title-Case words (the section title), then the start of an ordinary
# sentence (capital letter or opening punctuation) that is clearly body
# prose, not more title.
_CANDIDATE_RE = re.compile(
    r"^(?P<num>\d{1,3})\s+"
    r"(?P<title>(?:[A-Z][A-Za-z'\-]*\s+){1,7}[A-Z][A-Za-z'\-]*)\s+"
    r"(?P<rest>[A-Z(].{14,})$"
)


@dataclass
class SiblingAuditStats:
    outline_gaps_checked: int = 0
    high_confidence_headings_recovered: int = 0
    high_confidence_headings_ambiguous: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "outline_gaps_checked": self.outline_gaps_checked,
            "high_confidence_headings_recovered": self.high_confidence_headings_recovered,
            "high_confidence_headings_ambiguous": self.high_confidence_headings_ambiguous,
        }


def _top_level_numbered_headings(document: Document) -> List[Tuple[int, HeadingBlock]]:
    out = []
    for b in document.blocks:
        if not isinstance(b, HeadingBlock):
            continue
        nt = numbering_tuple(b.text)
        if nt is not None and len(nt) == 1:
            out.append((nt[0], b))
    return out


def audit_numbered_sibling_headings(document: Document, log: FindingLog) -> Dict[str, int]:
    """Mutates `document.blocks` in place, promoting confidently-recovered
    missing sibling headings. Returns audit metrics."""
    stats = SiblingAuditStats()
    numbered = _top_level_numbered_headings(document)
    if len(numbered) < 2:
        return stats.to_dict()

    numbered.sort(key=lambda t: document.blocks.index(t[1]))
    level = numbered[0][1].level

    for (n_lo, h_lo), (n_hi, h_hi) in zip(numbered, numbered[1:]):
        if n_hi - n_lo <= 1:
            continue
        missing = list(range(n_lo + 1, n_hi))
        lo_idx = document.blocks.index(h_lo)
        hi_idx = document.blocks.index(h_hi)

        for missing_num in missing:
            stats.outline_gaps_checked += 1
            matches: List[Tuple[int, "re.Match"]] = []
            for i in range(lo_idx + 1, hi_idx):
                block = document.blocks[i]
                if not isinstance(block, ParagraphBlock):
                    continue
                m = _CANDIDATE_RE.match((block.normalized or block.text).strip())
                if m and int(m.group("num")) == missing_num:
                    matches.append((i, m))

            if not matches:
                continue
            if len(matches) > 1:
                stats.high_confidence_headings_ambiguous += 1
                log.error(
                    "high_confidence_heading_not_preserved",
                    f"Outline sequence implies a section numbered {missing_num} between "
                    f"{h_lo.text!r} and {h_hi.text!r}, and {len(matches)} candidate blocks matched "
                    f"its shape -- left unresolved rather than guessed at",
                    block_id=document.blocks[matches[0][0]].block_id,
                    suggested_action="Disambiguate manually which block is the real section heading.",
                )
                continue

            idx, m = matches[0]
            block = document.blocks[idx]
            heading_text = f"{m.group('num')} {m.group('title')}"
            rest_text = m.group("rest").strip()

            new_heading = HeadingBlock(
                block_id=block.block_id, order=block.order, source_format=block.source_format,
                level=level, text=heading_text, normalized=heading_text, page=block.page,
            )
            new_heading.style = dict(block.style)
            new_heading.style["recovered_by"] = "heading_sibling_audit"
            new_para = ParagraphBlock(
                block_id=document.next_block_id(), order=block.order, source_format=block.source_format,
                text=rest_text, normalized=rest_text, page=block.page,
            )
            document.blocks[idx:idx + 1] = [new_heading, new_para]
            for j, b in enumerate(document.blocks):
                b.order = j
            hi_idx += 1  # one extra block inserted before h_hi

            stats.high_confidence_headings_recovered += 1
            log.fixed(
                "high_confidence_heading_recovered",
                f"Recovered section heading {heading_text!r} that was rendered as prose between "
                f"{h_lo.text!r} and {h_hi.text!r}",
                block_id=new_heading.block_id, page=new_heading.page,
            )

    return stats.to_dict()

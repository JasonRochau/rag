"""
OCR/layout hyphenation repair (spec section 11) and OCR bullet-glyph
normalization (spec section 12) -- both are format-agnostic text-shape
repairs applied to already-assembled paragraph/line text, not to any
particular document's vocabulary.

Hyphenation repair
-------------------
A PDF/OCR line-wrap artifact ("imple-\\nmented", "imple- mented",
"imple -\\nmented") always collapses, once lines are joined with a single
space (the extractors' own join separator), to one of two literal text
shapes: ``word- word`` or ``word - word`` -- a hyphen with whitespace
*after* it, immediately before a lowercase continuation word. A genuine
semantic hyphen ("machine-level") never has whitespace on either side in
the source text, so the presence of trailing whitespace after the hyphen is
itself the structural signal that something is a line-wrap artifact rather
than a real hyphen -- no line/bbox bookkeeping needs to survive extraction
for this repair to work; it operates on the final joined string.

Given a candidate, the decision of whether to actually dehyphenate is
dictionary-driven wherever a wordlist is available:
  * the two fragments joined with no hyphen is a real dictionary word
    ("ad" + "vanced" -> "advanced") -> repair, high confidence.
  * neither fragment survives a whitespace-based OCR wrap under a joined
    reading, but *both* fragments are independently real dictionary words
    of non-trivial length ("machine" + "level") -> the hyphen is semantic;
    normalize stray whitespace around it but keep the hyphen, high
    confidence.
  * anything else (no dictionary available, or the evidence doesn't clear
    either bar) -> low-confidence: keep the hyphen, normalize whitespace,
    and report it as unresolved rather than guessing.

Known multi-word technical compounds are also checked against a small
built-in preservation list first (spec "Known compound-word preservation"),
since compound tokens like "client-server" are exactly the kind of thing a
generic dictionary won't itself validate as a single entry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

_SYSTEM_DICT_PATHS = (
    "/usr/share/dict/words",
    "/usr/share/dict/american-english",
    "/usr/share/dict/british-english",
)

# Established hyphenated compounds common in technical/systems writing --
# checked before any dictionary lookup so they're preserved even when no
# system dictionary is installed at all (spec "Known compound-word
# preservation").
KNOWN_COMPOUNDS: Set[str] = {
    "machine-level", "client-server", "programmer-centric", "upper-level",
    "bit-level", "byte-level", "real-time", "run-time", "runtime-level",
    "multi-core", "cross-platform", "high-level", "low-level",
    "end-to-end", "object-oriented", "general-purpose", "memory-mapped",
    "user-level", "kernel-level", "open-source", "peer-to-peer",
    "man-in-the-middle", "read-only", "write-back", "write-through",
    "off-the-shelf", "state-of-the-art", "well-known", "self-contained",
}


def _load_system_dictionary() -> Optional[Set[str]]:
    for p in _SYSTEM_DICT_PATHS:
        path = Path(p)
        if path.is_file():
            try:
                words = {
                    w.strip().lower() for w in path.read_text(errors="ignore").splitlines()
                    if w.strip()
                }
                if words:
                    return words
            except OSError:
                continue
    return None


_SYSTEM_DICTIONARY = _load_system_dictionary()

# word2/word1 fragment shapes: a fragment must look like an ordinary
# alphabetic word chunk (letters only, plus an internal apostrophe for
# word1 so "wouldn'- t" style artifacts aren't silently ignored, though
# that spacing pattern is rare). Requires >=1 space after the hyphen --
# the structural tell that separates this from a real inline hyphen.
_HYPHEN_WRAP_RE = re.compile(r"\b([A-Za-z]{2,})\s?-\s+([a-z][a-z]*)\b")


@dataclass
class HyphenationStats:
    layout_hyphenation_candidates: int = 0
    layout_hyphenation_repairs: int = 0
    layout_hyphenation_rejected: int = 0
    unresolved_layout_hyphenation: int = 0
    repairs_log: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, int]:
        return {
            "layout_hyphenation_candidates": self.layout_hyphenation_candidates,
            "layout_hyphenation_repairs": self.layout_hyphenation_repairs,
            "layout_hyphenation_rejected": self.layout_hyphenation_rejected,
            "unresolved_layout_hyphenation": self.unresolved_layout_hyphenation,
        }


def _is_word(token: str, dictionary: Optional[Set[str]]) -> bool:
    return bool(dictionary) and token.lower() in dictionary


def repair_layout_hyphenation(
    text: str, stats: HyphenationStats, block_id: Optional[str] = None,
    dictionary: Optional[Set[str]] = _SYSTEM_DICTIONARY,
) -> str:
    """Repairs layout-wrap hyphenation in `text` in place (returns the
    repaired string) and accumulates metrics/log entries onto `stats`."""

    def _decide(m: re.Match) -> str:
        w1, w2 = m.group(1), m.group(2)
        stats.layout_hyphenation_candidates += 1
        compound_key = f"{w1.lower()}-{w2.lower()}"
        joined = w1 + w2

        if compound_key in KNOWN_COMPOUNDS:
            stats.layout_hyphenation_rejected += 1
            stats.repairs_log.append({
                "block_id": block_id or "", "action": "rejected",
                "original": m.group(0), "result": f"{w1}-{w2}", "reason": "known_compound",
            })
            return f"{w1}-{w2}"

        if _is_word(joined, dictionary):
            stats.layout_hyphenation_repairs += 1
            stats.repairs_log.append({
                "block_id": block_id or "", "action": "repaired",
                "original": m.group(0), "result": joined, "reason": "joined_form_is_dictionary_word",
            })
            return joined

        if dictionary and _is_word(w1, dictionary) and len(w1) >= 3 and _is_word(w2, dictionary) and len(w2) >= 2:
            stats.layout_hyphenation_rejected += 1
            stats.repairs_log.append({
                "block_id": block_id or "", "action": "rejected",
                "original": m.group(0), "result": f"{w1}-{w2}", "reason": "both_fragments_are_independent_words",
            })
            return f"{w1}-{w2}"

        stats.unresolved_layout_hyphenation += 1
        stats.repairs_log.append({
            "block_id": block_id or "", "action": "unresolved",
            "original": m.group(0), "result": f"{w1}-{w2}",
            "reason": "no_dictionary_evidence" if not dictionary else "ambiguous",
        })
        return f"{w1}-{w2}"

    return _HYPHEN_WRAP_RE.sub(_decide, text)


# ---------------------------------------------------------------------------
# OCR bullet-glyph normalization (spec section 12).
#
# OCR frequently misreads a real bullet glyph ("•", "●", a filled circle) as
# whatever Latin letter/punctuation happens to look similar at scan
# resolution -- commonly "e", "o", "O", "@", "*", a degree sign, etc.,
# followed by the item text. A single such line is exactly as likely to be
# ordinary text that happens to start with that character (a sentence
# fragment, a stray "e" from a broken word) as it is to be a real bullet --
# the only thing that distinguishes a genuine glyph-bullet run from that is
# *repetition*: two or more lines sharing the same leading glyph shape and
# the same left-indentation are list siblings; a lone match is left alone
# and reported instead of guessed at (spec: "Do not hard-code `e` or `@` as
# universal bullets... require list-level evidence").
# ---------------------------------------------------------------------------
_GLYPH_BULLET_RE = re.compile(r"^([eE@©°*·»>])\s+([A-Z][A-Za-z].*)$")


def ocr_bullet_glyph_match(text: str) -> Optional[str]:
    """Returns the bullet-item body text if `text` has the shape of an
    OCR-mis-scanned bullet glyph line, else None. This is a per-line shape
    test only -- callers must additionally require run evidence (two or
    more candidates at consistent indentation) before treating a match as a
    confirmed list item; see the module docstring."""
    m = _GLYPH_BULLET_RE.match(text.strip())
    return m.group(2) if m else None

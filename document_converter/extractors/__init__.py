"""Format-specific extractors behind a shared interface (spec section 6)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Tuple

from ..model import Document
from ..findings import FindingLog
from ..config import OCRConfig

# Office export tools (Word, PowerPoint, LibreOffice) routinely leave a
# meaningless default in the file's own title metadata -- "Microsoft Word -
# Document1", "Presentation1", "Untitled" -- which is worse than no title at
# all: normalize.py trusts an explicit title over guessing one from the
# filename, so a junk metadata title wins over a perfectly good filename.
# Publisher/platform names (spec section 11's explicit penalty list) belong
# in the same bucket: a "Print to PDF" export routinely carries the
# authoring *platform's* name in the title slot, not the document's --
# "zyBooks" was found this way on a real regression fixture (an interactive
# CS courseware export whose actual chapter title is a graphic, not
# extractable text, so the platform name was the only metadata title at
# all).
_JUNK_TITLE_RE = re.compile(
    r"^(microsoft word - .*|document\d*|untitled\d*|new document|presentation\d*|"
    r"zybooks|coursera|udemy|canvas|blackboard|moodle|scribd)$",
    re.IGNORECASE,
)



# A scanning/production pipeline sometimes leaves an internal batch, issue,
# or page-range identifier in the PDF's title metadata field instead of the
# document's actual title -- e.g. "Ausgabe_A_S00022P128_S00028P134" (German
# "Ausgabe" = issue/edition, followed by underscore-joined page-range
# codes). Structural, not vocabulary-based, so it generalizes to any such
# pipeline rather than this one string: a real title is essentially never
# one whitespace-free run of underscore/hyphen-joined tokens where at least
# half the tokens are alphanumeric-code-shaped (a handful of letters glued
# to a run of digits, like a page/batch/version stamp) rather than a word.
_CODE_TOKEN_RE = re.compile(r"^[A-Za-z]{0,3}\d+[A-Za-z]*\d*$")


def _looks_like_internal_code(title: str) -> bool:
    if " " in title or "\t" in title:
        return False
    segments = [s for s in re.split(r"[_\-]+", title) if s]
    if len(segments) < 2:
        return False
    code_like = sum(1 for s in segments if _CODE_TOKEN_RE.match(s))
    return code_like / len(segments) >= 0.5


def clean_title(title: Optional[str]) -> Optional[str]:
    """Returns `title` unless it's empty, matches a known default-export
    placeholder, or is shaped like an internal batch/issue code, in which
    case returns None so callers fall back to the filename or an in-body
    heading instead."""
    if not title or not title.strip():
        return None
    title = title.strip()
    if _JUNK_TITLE_RE.match(title):
        return None
    if _looks_like_internal_code(title):
        return None
    return title


# Shared with pdf_extractor.py's OCR path: OCR and plain text both carry no
# font/style signal, so a short, isolated, unpunctuated line surrounded by
# blank lines is the only generic *structural* heading signal available for
# either. `is_heading_candidate` therefore additionally has to do the work a
# font-size/style signal would normally rule out on its own: reject code,
# pseudocode, algorithm-listing lines, equations/matrices, page numbers, bare
# punctuation, and garbled OCR fragments. This one function is shared by the
# PDF OCR path, the plain-text extractor, and (as a retrospective safety net)
# validation.py, so a fix here applies to every format at once instead of
# drifting between independent copies -- which is exactly how the original
# bug happened (pdf_extractor.py's own code-run detector used a different,
# stricter set of signals than this classifier, so code/algorithm lines were
# already committed as headings before the code-run detector ever saw them).
LIST_LINE_RE = re.compile(r'^(\s*)([-*•]|\d{1,9}[.)])\s+(.*)$')

# Pure page numbers / running numeric footers: "94", "(3)", "1245 3", "2400".
_NO_LETTERS_RE = re.compile(r"^[\W\d_]*$")

# Numbering prefix for section headings: "1", "3.1", "3.1.1", "Chapter 3",
# "Section 3.2", "Appendix A". Returns a depth (dot count + 1) via
# `numbering_depth`, used to infer heading hierarchy instead of flattening
# every accepted heading to the same level. The digit classes are deliberately
# ASCII-only ([0-9], not \d): Python's \d also matches Unicode decimal digits
# such as fullwidth '７' (U+FF17), which icon fonts (e.g. Microsoft Learn's
# "docons" Note-icon glyph) emit as text -- those were never real section
# numbers, and parsing them as such makes every Note callout look like a
# top-level heading numbered 7. (re.ASCII is NOT an acceptable substitute:
# it would also narrow \W used by _NO_LETTERS_RE's "no letters" check,
# causing CJK heading text like 日本語 to be thrown out as page numbers.)
_NUMBERING_RE = re.compile(r"^([0-9]{1,2}(?:\.[0-9]{1,3}){0,4})(?:[.:)]|\s+\S|$)")
_NUMBERING_WORD_RE = re.compile(r"^(chapter|section|appendix)\s+(\d+|[A-Z])\b", re.IGNORECASE)
# "N - text" (number separated from its text by a dash) is an enumerated
# parameter-value/list item ("1 - Rowcount check only (the default)",
# "2 - Errors and warnings"), not outline numbering: genuine numbered
# headings separate number from title with a space or "." ("1 Introduction",
# "3.2 Setup"), never a dash. ASCII-only digits, same fullwidth-glyph
# reasoning as _NUMBERING_RE above.
_ENUMERATED_VALUE_ITEM_RE = re.compile(r"^[0-9]{1,2}\s+[-–—]\s+\S")

# Programming keywords/constructs that show up in print-to-PDF/OCR'd code.
# A keyword alone is no longer enough ("Import data" and "Return values" are
# real headings). It must be accompanied by a code operator/terminator, OR the
# keyword must be one of the common-English words that only reads as code in
# its lowercase form.
_CODE_KEYWORD_RE = re.compile(
    r"\b(function|end\s*function|end\s*if|end\s*for|end\s*while|end\s*while|"
    r"return|print|import|include|namespace|public|private|protected|"
    r"def|class|elif|else\s*if|while\s*\(|for\s*\(|if\s*\()\b", re.IGNORECASE,
)
# Common-English words that are also code keywords: only treat as a code signal
# when they appear in their usual lowercase code spelling (e.g. "import os",
# not "Import data").
_CODE_KEYWORD_COMMON_ENGLISH = {
    "import", "return", "class", "function", "def", "print",
    "include", "namespace", "public", "private", "protected",
}
# Code operator/terminator evidence required for keywords that are not in the
# common-English set above, or as an alternative signal for any keyword.
# A parenthetical only counts when it is a call (word immediately before '(');
# "Return values (C#)" is a heading, "return foo(x)" is code. A colon only
# counts at line end ("def foo():", "if (x) > 0:"), not mid-heading.
_CODE_TERMINATOR_RE = re.compile(r"(;|\{|::|\w\([^()]*\)|:\s*$)")
_CODE_OPERATOR_RE = re.compile(
    r"(==|!=|<=|>=|\+\+|--|\+=|-=|->|::|;\s*$|^\s*[{}]\s*;?\s*$|//|/\*|\w+\s*\([^()]*\)\s*[{;]\s*$)"
)
# Numbered algorithm-listing line, e.g. "12: end function", "9: childCount — 0".
_ALGO_LINE_NO_RE = re.compile(r"^\d{1,3}\s*[:.]\s+\S")
# "identifier — value" / "identifier <- value": OCR commonly renders the
# pseudocode assignment arrow (<-, ←) as an em-dash or similar.
_ARROW_ASSIGN_RE = re.compile(r"^[A-Za-z_][\w\[\]]{0,20}\s*[—–]\s*\S")
# A bare "=" almost never appears in real prose heading text; it is one of
# the strongest single-signal indicators of code/equation content.
_HAS_EQUALS_RE = re.compile(r"=")
_NUMERIC_UNIT_TOKEN_RE = re.compile(r"^[\d.,]+[a-zA-Z]{0,3}$")
# Scientific-notation continuation token ("E-3", "e+5") -- OCR/whitespace
# routinely splits "3.1E-3" into two tokens ("3.1", "E-3"); without this the
# second half looks like a plain word and dilutes the numeric-density ratio
# below its trigger threshold.
_SCI_NOTATION_TOKEN_RE = re.compile(r"^[Ee][-+]?\d+(\.\d+)?$")
# A document-universal convention (not tied to any one document/publisher):
# "Figure 3:", "Fig. 7.", "Table 1:" always introduces a caption, never a
# section heading. The colon/period lead-in is the common case, but OCR (and
# some publishers' own house style) routinely drops that punctuation
# entirely -- "Figure 2 MLP results with learning rate = 0.0001" is exactly
# as much a caption as "Figure 2: MLP results..." is, just missing the
# colon; requiring one letter of descriptive text to follow the number
# (instead of punctuation) catches that shape without loosening the rule to
# match a bare "Figure 2" reference embedded mid-sentence (the `^` anchor
# already confines this to a line that *starts* with the label).
_CAPTION_RE = re.compile(r"^(figure|fig\.?|table)\s+\d+(?:\.\d+)?\s*(?:[:.)]|\s+[A-Za-z])", re.IGNORECASE)
# A bare "identifier(args)" expression with no verb/sentence structure
# ("KSP(xj, xj)") -- a function-call/notation shape that isn't caught by the
# existing code-operator bank (no trailing brace/semicolon, no keyword, no
# bare "=").
_BARE_FUNCTION_CALL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,24}\s*\([^()]{1,80}\)[.,]?$")


def _strip_bracket_punct(token: str) -> str:
    return token.strip("()[]{}<>,.;:")


def _looks_like_numeric_token(token: str) -> bool:
    core = _strip_bracket_punct(token)
    if not core:
        return False
    if _NUMERIC_UNIT_TOKEN_RE.match(core):
        return True
    if _SCI_NOTATION_TOKEN_RE.match(core):
        return True
    return False


def looks_like_caption_text(text: str) -> bool:
    """True for a figure/table caption lead-in ("Figure 3: ...", "Table 1.
    ..."). Shared between the heading classifier (Fix 2) and validation.py's
    retrospective context checks (Fix 5, adjacency-to-caption)."""
    return bool(_CAPTION_RE.match(text.strip()))


# A small, generic set of measurement/metric units and axis-label shapes
# that show up in charts across essentially any empirical paper -- not
# vocabulary tied to any one document's subject matter, the same way
# `_CODE_KEYWORD_RE` is a generic programming-keyword bank rather than a
# per-document word list.
_UNIT_WORD_RE = re.compile(
    r"^(ms|milliseconds?|s|secs?|seconds?|micro ?seconds?|us|min|mins?|minutes?|"
    r"hrs?|hours?|accuracy|precision|recall|error|errors|latency|throughput|"
    r"percent|count|counts|time|times|score|scores|rate|rates|loss|value|values)$",
    re.IGNORECASE,
)


def looks_like_axis_or_legend_label(text: str) -> bool:
    """Structural shape of a chart axis/legend label: an isolated 1-2 word
    line that is either a bare unit/metric word or purely numeric tokens.
    This only tests text *shape* -- spec Fix 2 requires the page-region
    adjacency-to-figure-content condition to be checked separately by the
    caller (pdf_extractor.py has the bbox/position information this
    function doesn't); the same bare word standing alone elsewhere in a
    document (not near a figure) must still be a valid heading."""
    s = text.strip()
    words = s.split()
    if not (1 <= len(words) <= 2):
        return False
    if all(_looks_like_numeric_token(w) for w in words):
        return True
    return any(_UNIT_WORD_RE.match(w.strip(".,:%")) for w in words)


def numbering_depth(text: str) -> Optional[int]:
    """Returns the outline depth implied by a numbering prefix ("1" -> 1,
    "3.1" -> 2, "3.1.1" -> 3), or None if the text has no numbering prefix."""
    s = text.strip()
    if _ENUMERATED_VALUE_ITEM_RE.match(s):
        return None
    m = _NUMBERING_RE.match(s)
    if m:
        return m.group(1).count(".") + 1
    if _NUMBERING_WORD_RE.match(s):
        return 1
    return None


def numbering_tuple(text: str) -> Optional[Tuple[int, ...]]:
    """Parses a leading numeric outline prefix ("3.1.2") into a comparable
    tuple (3, 1, 2), used to validate that a numbered heading candidate is a
    plausible successor in the document's own section sequence rather than
    an unrelated numbered artifact (e.g. a figure/table caption number)."""
    s = text.strip()
    if _ENUMERATED_VALUE_ITEM_RE.match(s):
        return None
    m = _NUMBERING_RE.match(s)
    if not m:
        return None
    try:
        return tuple(int(p) for p in m.group(1).split("."))
    except ValueError:
        return None


def numbering_prefix_remainder(text: str) -> Optional[str]:
    """Returns the text remaining after stripping a leading numeric outline
    prefix ("2.5 0 notation" -> "0 notation"), or None if `text` has no such
    prefix. Shared by `heading_demotion.py`'s garbled-numbered-heading check
    so it inspects the same title portion `numbering_tuple` parsed the
    prefix from, rather than a second, potentially-inconsistent regex."""
    s = text.strip()
    if _ENUMERATED_VALUE_ITEM_RE.match(s):
        return None
    m = _NUMBERING_RE.match(s)
    if not m:
        return None
    return s[len(m.group(1)):].strip(" \t.:)")


def is_plausible_outline_successor(prev: Optional[Tuple[int, ...]], candidate: Tuple[int, ...]) -> bool:
    """True if `candidate` is a plausible next number in an outline sequence
    that most recently emitted `prev` (True unconditionally when `prev` is
    None -- the very first numbered heading is always accepted). Used to
    distinguish a genuine section-numbering continuation from an unrelated
    numbered artifact that happens to share the same syntactic shape (a
    citation number, a figure/table number, a data value) -- a document-
    universal outline-continuity rule, not tied to any document's content."""
    if prev is None:
        return True
    if candidate == prev:
        return False
    if len(candidate) == len(prev) + 1 and candidate[:-1] == prev and candidate[-1] == 1:
        return True  # first child of the currently-open section
    max_d = min(len(candidate), len(prev))
    for d in range(1, max_d + 1):
        if candidate[:d - 1] == prev[:d - 1] and candidate[d - 1] == prev[d - 1] + 1:
            if all(v <= 1 for v in candidate[d:]):
                return True  # sibling/ancestor increment, optionally starting fresh sub-numbering
    return False


# OCR frequently emits a numbered section's title and its opening sentence
# on one physical line ("3.2.3 Suffix Extraction. For efficiency, the first
# step..."), which otherwise looks like ordinary prose glued to a heading
# and gets rejected wholesale by `_looks_like_prose_continuation`'s
# mid-string-period check. This decomposes that shape into (heading, body)
# without touching the many genuinely-prose lines that also contain a
# mid-string period (see the abbreviation guard below).
_COMPOUND_HEADING_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,3}){0,4})\s+"
    r"(?P<title>[A-Z][A-Za-z][A-Za-z0-9 ,'\-]{2,70}?)\.\s+"
    r"(?P<rest>[A-Z].{10,})$"
)
# Common scholarly abbreviations ending in a period that is NOT a sentence
# boundary ("et al.", "Fig.", "e.g.") -- a generic English/academic-writing
# convention, not vocabulary tied to any one document.
_ABBREVIATION_LAST_WORDS = {
    "al", "etc", "eg", "ie", "fig", "figs", "vs", "dr", "mr", "mrs", "ms",
    "st", "no", "vol", "pp", "p", "resp", "approx", "ca", "cf",
}


def split_compound_numbered_heading(text: str) -> Optional[Tuple[str, str, str]]:
    """If `text` has the shape "<numbering> <Title words>. <Sentence...>",
    returns (numbering, title, remainder); otherwise None. Purely
    syntactic -- the caller is responsible for outline-sequence-plausibility
    validation via `is_plausible_outline_successor`, since that requires
    document-level state this pure string function doesn't have."""
    s = text.strip()
    m = _COMPOUND_HEADING_RE.match(s)
    if not m:
        return None
    num, title, rest = m.group("num"), m.group("title").strip(), m.group("rest").strip()
    if not title or not any(c.isalpha() for c in title):
        return None
    alpha_chars = sum(1 for c in title if c.isalpha())
    if alpha_chars / max(len(title), 1) < 0.6:
        return None  # too numeric/symbolic to be a real title ("E-3) 3.55")
    if looks_like_equation_or_table_row(title) or looks_like_caption_text(title):
        return None
    title_words = title.split()
    if not (1 <= len(title_words) <= 12):
        return None
    last_word = re.sub(r"[^A-Za-z]", "", title_words[-1]).lower()
    if last_word in _ABBREVIATION_LAST_WORDS:
        return None  # "3.1 See Smith et al. for details..." -- not a heading
    return num, title, rest


def _all_words_too_short(s: str) -> bool:
    """True when every alphabetic "word" in the line is 1-2 letters (e.g. 'T
    Zz', 'SU', 'oro'... well 'oro' is 3 -- see known limitation). Genuine
    headings virtually always contain at least one word of 3+ letters;
    accidental OCR syllables and stray symbol-adjacent letters do not."""
    words = re.findall(r"[A-Za-z]+", s)
    if not words:
        return False
    return all(len(w) <= 2 for w in words)


def _looks_like_prose_continuation(s: str) -> bool:
    """Rejects lines that are grammatically full sentences/clauses rather
    than heading-shaped noun phrases: starts with a lowercase letter (a
    heading fragment glued mid-sentence), or reads as a sentence introducing
    what follows (comma + long, or trailing ':' + long) -- both common
    shapes for OCR'd lead-in prose ("As an example, the following vector
    s:", "is equivalent to the following operation on Ag:") that satisfy a
    naive "short isolated line" rule but are not headings."""
    if not s:
        return True
    first_alpha = next((c for c in s if c.isalpha()), None)
    if first_alpha is not None and first_alpha.islower():
        return True
    words = s.split()
    if "," in s and len(words) >= 5:
        return True
    if s.endswith(":") and len(words) > 4:
        return True
    # A mid-string sentence-ending period followed by more text ("2.1.3:
    # Linear search runtime. L") means this is prose/caption debris, not a
    # single title-shaped line.
    if re.search(r"\.\s+\S", s):
        return True
    return False


def looks_like_algorithm_line(s: str) -> bool:
    """Structural-only algorithm/pseudocode signals: a numbered listing line
    ("12: end function") or an identifier-arrow-value assignment ("childCount
    — 0"). Unlike keyword matching, these essentially never occur in ordinary
    English prose, so this is the variant safe to apply to *runs* of
    continuous text (pdf_extractor.py's code-run grouping scans consecutive
    paragraph lines, not just isolated candidate lines -- keyword matching
    there would flag any paragraph that happens to use "class"/"return"/
    "print"/"public" as an ordinary English word)."""
    return bool(_ALGO_LINE_NO_RE.match(s) or _ARROW_ASSIGN_RE.match(s))


def looks_like_code_or_algorithm_line(s: str) -> bool:
    """Full code/algorithm signal bank, including keyword/operator matching.
    Safe for a single, already-isolated heading-candidate line (surrounded by
    blank lines); NOT safe to scan across a run of ordinary prose lines --
    see `looks_like_algorithm_line` for that case."""
    if looks_like_algorithm_line(s):
        return True
    m = _CODE_KEYWORD_RE.search(s)
    if m is not None:
        matched = m.group(0)
        if _CODE_TERMINATOR_RE.search(s):
            return True
        if matched.lower() in _CODE_KEYWORD_COMMON_ENGLISH and matched == matched.lower():
            return True
    if _CODE_OPERATOR_RE.search(s):
        return True
    return False


def looks_like_equation_or_table_row(s: str) -> bool:
    if _HAS_EQUALS_RE.search(s):
        return True
    if re.match(r"^\([\d]{1,3}\)$", s):  # standalone equation number "(3)"
        return True
    if _BARE_FUNCTION_CALL_RE.match(s):  # "KSP(xj, xj)" -- notation/function-call shape
        return True
    tokens = s.split()
    if len(tokens) >= 3:
        numeric_like = sum(1 for t in tokens if _looks_like_numeric_token(t))
        if numeric_like / len(tokens) >= 0.5:
            return True
    return False


# Microsoft Learn product-badge boilerplate: pages open with an "Applies to:"
# row of product badges, each badge rendering as an isolated short line
# ("Applies to: SQL Server") that is otherwise indistinguishable from a short
# heading by shape alone. The lead-in prefix is publisher boilerplate, never
# a section heading, so a line beginning with it is rejected outright.
_PRODUCT_BADGE_RE = re.compile(r"^applies to\b\s*:?", re.IGNORECASE)


def heading_rejection_reason(line: str) -> Optional[str]:
    """Returns why `line` must NOT be treated as a heading candidate, or
    None if it may be one (structural gating -- font-size/isolation/etc --
    is still the caller's job; this only rules out clearly-invalid text).
    Every reason here corresponds to a spec section-9 rejection category, and
    the returned string doubles as the finding code suffix."""
    s = line.strip()
    if not s or len(s) > 90:
        return "too_long_or_empty"
    if _CAPTION_RE.match(s):
        return "figure_or_table_caption"
    if _PRODUCT_BADGE_RE.match(s):
        return "product_badge_line"
    if _ENUMERATED_VALUE_ITEM_RE.match(s):
        return "enumerated_value_item"
    if s[-1] in ".,;":
        return "trailing_punctuation"
    words = s.split()
    if len(words) > 12:
        return "too_many_words"
    if LIST_LINE_RE.match(line):
        return "list_item"
    first_char = s[0]
    if not (first_char.isalnum()):
        return "punctuation_only"
    if _NO_LETTERS_RE.match(s):
        return "no_letters"
    if not (first_char.isupper() or first_char.isdigit()):
        return "prose_continuation"
    if _all_words_too_short(s):
        return "ocr_fragment"
    if looks_like_code_or_algorithm_line(s):
        return "code_or_algorithm"
    if looks_like_equation_or_table_row(s):
        return "equation_or_matrix"
    if _looks_like_prose_continuation(s):
        return "prose_continuation"
    return None


def is_heading_candidate(line: str) -> bool:
    return heading_rejection_reason(line) is None


# Common interactive-courseware UI chrome: caption/label lead-ins ("Static
# figure:", "Animation content:") and short imperative call-to-action
# phrases ("Show answer", "Continue", "Submit"). These are valid *headings*
# (heading_rejection_reason lets them through -- they're grammatically fine
# short phrases, and demoting every short phrase to non-heading would be the
# "isolated short line is invalid" over-reach the spec warns against), but
# they are never a good *document title* candidate. This is a general
# interactive-document-chrome pattern, not text tied to one publisher.
_TITLE_CANDIDATE_VERB_RE = re.compile(
    r"^(show|hide|reveal|check|submit|continue|next|previous|play|pause|skip|toggle|click)\b",
    re.IGNORECASE,
)


def looks_like_title_candidate(text: str) -> bool:
    """A stricter check than `is_heading_candidate`, used only when picking
    which heading becomes the document title (spec section 11). Rejects
    caption/label lead-ins (trailing ':'), shouted single-word UI chrome
    (a lone all-caps word, e.g. "PARTICIPATION"), and short imperative
    call-to-action phrases -- none of these are descriptive enough to be a
    real document title even though they're valid headings elsewhere in the
    document."""
    if not is_heading_candidate(text):
        return False
    s = text.strip()
    if s.endswith(":"):
        return False
    words = s.split()
    if len(words) == 1 and s.isupper():
        return False
    if len(words) <= 3 and _TITLE_CANDIDATE_VERB_RE.match(s):
        return False
    return True


# ---------------------------------------------------------------------- #
# OCR heading-promotion hardening.
#
# A native PDF/DOCX heading has real font metadata behind it; OCR text has
# none. Isolation (blank space above and below a line) used to be treated as
# if it *were* font evidence -- it is not, it is just a layout gap that any
# figure label, axis tick, or caption fragment produces just as easily as a
# real section title. The functions below are the *negative*-evidence half
# of that fix: recognizing the shapes that a scanned figure/diagram/table
# region produces, so pdf_extractor.py's OCR multi-signal scorer can weigh
# them against a candidate instead of promoting on isolation alone.
_FIGURE_LABEL_RE = re.compile(
    r"^(original|input|output|source|sample|resulting?|final|scanned|cropped)\s+"
    r"(image|figure|photo|picture|diagram|graph|chart|plot|scan)s?\.?$",
    re.IGNORECASE,
)
# "Image 2", "Figure 2", "Photo 3" used as a bare figure label (no colon, no
# "Figure N:" caption lead-in -- that shape is `looks_like_caption_text`'s
# job) -- OCR routinely drops the caption punctuation entirely.
_BARE_FIGURE_NUMBER_RE = re.compile(
    r"^(image|figure|photo|picture|diagram|graph|chart|plot|panel|scan)\s*\d{1,3}\.?$", re.IGNORECASE,
)
_DIRECTION_WORD_RE = re.compile(
    r"^(north|south|east|west|northeast|northwest|southeast|southwest|"
    r"horizontal|vertical|diagonal|clockwise|counter-?clockwise)\b", re.IGNORECASE,
)
_DIRECTION_LABEL_RE = re.compile(
    r"^(north|south|east|west|northeast|northwest|southeast|southwest|"
    r"horizontal|vertical|diagonal|clockwise|counter-?clockwise)[\w\s-]*"
    r"\b(direction|edges?|axis|arrow|vector)s?$", re.IGNORECASE,
)
# "Y - Axis", "X-Axis", "Y Axis" -- a spelled-out axis label, distinct from
# `looks_like_axis_or_legend_label`'s bare-unit-word shape.
_AXIS_SPELLED_RE = re.compile(r"^[xyz]\s*[-–]?\s*axis$", re.IGNORECASE)
# "Color: Red", "Colour:", "Key:", "Legend:" -- a colored/legend swatch label.
_COLOR_LEGEND_RE = re.compile(r"^(colou?r|legend|key|shading|pattern)\s*:\s*\S*$", re.IGNORECASE)
# "8 Gray Levels", "3 Channels", "24 Bit Depth" -- a bare count + short
# descriptive noun phrase describing an image/data property. Structurally
# identical to a level-1 numbered heading ("8 Introduction") without any
# actual outline semantics behind the number -- the caller is responsible
# for cross-checking outline plausibility; this only tests the shape.
_BARE_QUANTITY_LABEL_RE = re.compile(r"^\d{1,4}\s+[A-Za-z][A-Za-z\s]{1,24}$")
# A short introductory phrase that ends a lead-in with a colon ("Solution:",
# "For example:", "Note:") -- legitimate headings can also end in a colon
# ("Abstract:"), so this is intentionally *not* wired into
# `heading_rejection_reason` as a hard reject; it is soft, negative evidence
# for the OCR multi-signal scorer only.
_SHORT_COLON_INTRO_RE = re.compile(r".*:\s*$")


def looks_like_figure_or_diagram_label(text: str) -> bool:
    """True for a bare figure/image label ("Original Image", "Input Image",
    "Image 2") -- never a caption (no "Figure N:" lead-in punctuation, see
    `looks_like_caption_text` for that shape) but still never a section
    heading."""
    s = text.strip()
    return bool(_FIGURE_LABEL_RE.match(s) or _BARE_FIGURE_NUMBER_RE.match(s))


def looks_like_diagram_annotation(text: str) -> bool:
    """True for an axis/direction/legend/color annotation on a chart or
    diagram ("Y - Axis", "North Direction Edges", "Color: Red")."""
    s = text.strip()
    if _AXIS_SPELLED_RE.match(s) or _DIRECTION_LABEL_RE.match(s) or _COLOR_LEGEND_RE.match(s):
        return True
    if looks_like_axis_or_legend_label(s):
        return True
    words = s.split()
    if len(words) <= 3 and _DIRECTION_WORD_RE.match(s):
        return True
    return False


def looks_like_bare_quantity_label(text: str) -> bool:
    """True for a bare count + short noun phrase describing an image/data
    property ("8 Gray Levels", "3 Channels") -- syntactically identical to a
    level-1 numbered heading, but the caller must additionally confirm this
    has no real outline continuity behind it before treating the match as
    negative evidence (a genuine "8 Conclusion" is not this)."""
    s = text.strip()
    if not _BARE_QUANTITY_LABEL_RE.match(s):
        return False
    words = s.split()
    return len(words) <= 4


def looks_like_short_introductory_phrase(text: str) -> bool:
    """Soft negative evidence only (see module docstring above): any line
    ending in a colon, regardless of length. A legitimate heading can end in
    a colon ("Abstract:"), so this must always be combined with other
    signals by the caller, never used as an unconditional rejection."""
    return bool(_SHORT_COLON_INTRO_RE.match(text.strip()))


def looks_like_table_cell_fragment(text: str) -> bool:
    """Structural shape of a single cell/row recovered from a table that
    OCR could not reconstruct as a real grid: a short "label: value" or
    "label  value" pair, or a lone short numeric/short-token value. Distinct
    from `looks_like_diagram_annotation`'s "Color: Red" swatch-label shape
    only by caller context (adjacency to other same-shaped siblings /
    detected table zone) -- this function only tests the generic shape."""
    s = text.strip()
    if not s or len(s) > 60:
        return False
    if re.match(r"^[A-Za-z][\w\s/.'-]{0,30}:\s*[\w.%-]{1,20}$", s):
        return True
    if re.match(r"^[\w.%-]{1,12}(\s{2,}[\w.%-]{1,12}){1,6}$", s):
        return True
    return False


_ALL_CAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \-/,'.]*$")


def heading_capitalization_shape(text: str) -> str:
    """Coarse capitalization "shape" bucket used to compare a candidate
    against headings already confirmed elsewhere in the same document (spec:
    "capitalization consistent with other confirmed headings"). Purely
    syntactic; the caller decides whether a match against confirmed history
    counts as corroborating evidence."""
    s = text.strip()
    if not s:
        return "empty"
    if len(s) > 1 and _ALL_CAPS_RE.match(s) and any(c.isalpha() for c in s):
        return "all_caps"
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", s)
    if not words:
        return "other"
    capitalized = sum(1 for w in words if w[0].isupper())
    if capitalized / len(words) >= 0.8:
        return "title_case"
    if s[0].isupper():
        return "sentence_case"
    return "other"


def ocr_heading_negative_signals(text: str) -> List[str]:
    """Returns the names of every negative-evidence category `text` matches
    (spec "Negative heading evidence"): figure/image labels, diagram
    annotations, bare quantity labels, table-cell fragments, and short
    colon-terminated introductory phrases. An empty list means none of these
    generic shapes matched -- it does NOT by itself mean the text is a good
    heading (the caller still runs the full positive-signal scorer)."""
    reasons: List[str] = []
    if looks_like_caption_text(text):
        reasons.append("caption")
    if looks_like_figure_or_diagram_label(text):
        reasons.append("figure_or_image_label")
    if looks_like_diagram_annotation(text):
        reasons.append("diagram_annotation")
    if looks_like_bare_quantity_label(text):
        reasons.append("bare_quantity_label")
    if looks_like_table_cell_fragment(text):
        reasons.append("table_cell_fragment")
    if looks_like_short_introductory_phrase(text):
        reasons.append("introductory_phrase")
    return reasons


# Shared between pdf_extractor.py's extraction-time scorer and
# heading_repair.py's post-classification repair pass, so "weak" and
# "confident" mean the same thing in both places.
OCR_HEADING_MIN_ACCEPT_SCORE = 2
OCR_HEADING_CONFIDENT_SCORE = 3
OCR_HEADING_NEGATIVE_OVERRIDE_SCORE = 4
# These three negative-evidence categories are an absolute bar to OCR
# heading promotion, regardless of how many positive signals a candidate
# otherwise accumulates -- they back the hard `== 0` validation gates
# (figure_labels_promoted_to_heading, diagram_annotations_promoted_to_heading,
# table_cells_promoted_to_heading). Every other negative category (bare
# quantity label, introductory phrase, low OCR confidence, repeated label
# text, short fragment with no body) only raises the acceptance bar to
# `OCR_HEADING_NEGATIVE_OVERRIDE_SCORE`, since strong contrary evidence can
# still legitimately override those.
OCR_HEADING_HARD_NEGATIVE_CATEGORIES = frozenset({
    "caption", "figure_or_image_label", "diagram_annotation", "table_cell_fragment",
})


def classify_ocr_nonheading_kind(text: str) -> str:
    """Maps a rejected OCR heading candidate to the typed block it should
    become instead of being forced into a heading or merged into surrounding
    prose (spec "Typed block handling"). One of "caption", "figure_label",
    "diagram_annotation", "table_cell", or "unknown".

    Deliberately does NOT use `looks_like_bare_quantity_label` here, unlike
    `ocr_heading_negative_signals` (scoring): that shape is genuinely
    ambiguous with a real numbered heading ("7 Conclusion" and "8 Gray
    Levels" are structurally identical) and its own docstring says as much
    -- the caller there is expected to cross-check outline continuity,
    which only pdf_extractor.py's scorer has the context to do. This
    function backs the *unconditional* `table_cells_promoted_to_heading`
    gate, so it must stick to shapes that are unambiguous from text alone
    (an actual key:value/columnar row, not just "a number followed by
    words")."""
    if looks_like_caption_text(text):
        return "caption"
    if looks_like_figure_or_diagram_label(text):
        return "figure_label"
    if looks_like_diagram_annotation(text):
        return "diagram_annotation"
    if looks_like_table_cell_fragment(text):
        return "table_cell"
    return "unknown"


def safe_progress(progress: Optional[Callable], stage: str, **payload) -> None:
    """Same contract as pipeline.py's `_emit` (a raising callback must never
    break extraction), duplicated here rather than imported to avoid a
    circular import (pipeline.py imports this package at module level)."""
    if progress is None:
        return
    try:
        progress(stage, payload)
    except Exception:
        pass


class DocumentExtractor(Protocol):
    def supports(self, path: Path) -> bool: ...

    def extract(
        self, path: Path, ocr_config: OCRConfig, *, progress: Optional[Callable] = None,
    ) -> Tuple[Document, FindingLog]: ...


def get_extractors() -> List[DocumentExtractor]:
    from .markdown_extractor import MarkdownExtractor
    from .txt_extractor import TxtExtractor
    from .html_extractor import HtmlExtractor
    from .docx_extractor import DocxExtractor
    from .doc_extractor import DocExtractor
    from .pdf_extractor import PdfExtractor
    extractors: List[DocumentExtractor] = [
        MarkdownExtractor(), TxtExtractor(), HtmlExtractor(), DocxExtractor(), DocExtractor(),
    ]
    # `PARSER=docling` feature flag (layout PoC, artifacts/layout_poc/reports/
    # final_recommendation.md): DoclingExtractor also `supports()` .pdf, and
    # `select_extractor()` returns the first match below, so inserting it
    # ahead of PdfExtractor here is what makes it take over .pdf files --
    # `pdf_extractor.py` stays registered as the default and the fallback for
    # every other value of PARSER (unset, "legacy", anything else).
    if os.environ.get("PARSER", "legacy").strip().lower() == "docling":
        from .docling_extractor import DoclingExtractor
        extractors.append(DoclingExtractor())
    extractors.append(PdfExtractor())
    return extractors


def select_extractor(path: Path) -> DocumentExtractor:
    for ex in get_extractors():
        if ex.supports(path):
            return ex
    raise ValueError(f"No extractor supports file type: {path.suffix}")


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm", ".txt", ".md", ".markdown"}

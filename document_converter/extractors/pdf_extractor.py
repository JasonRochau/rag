"""
PDF extractor (spec 6.5 / 6.6). Uses PyMuPDF (fitz) for text + layout, with
an optional OCR fallback for pages that look scanned.

Heading/list/code detection is font-size and font-flag heuristic based,
since PDF carries no semantic structure. This is inherently imperfect for
arbitrary real-world PDFs -- see the deliverables doc's "known limitations"
section. Repeated running headers/footers are only removed when the exact
same normalized line is seen at the top or bottom of a strong majority of
pages, and every removal is logged.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from ..model import (
    Document, DocumentMetadata, HeadingBlock, ParagraphBlock, ListBlock, ListItemNode,
    TableBlock, CodeBlock, CaptionBlock, FigureLabelBlock, DiagramAnnotationBlock,
    TableCellBlock, UnknownBlock,
    AuthorBlock, AffiliationBlock, ContactBlock, KeywordBlock, PublicationMetadataBlock,
    CopyrightNoticeBlock, PermissionNoticeBlock, ConferenceMetadataBlock,
)
from ..findings import FindingLog
from ..config import OCRConfig
from . import (
    clean_title, is_heading_candidate, numbering_depth, numbering_prefix_remainder,
    looks_like_algorithm_line,
    looks_like_title_candidate, looks_like_axis_or_legend_label, looks_like_caption_text,
    numbering_tuple, is_plausible_outline_successor, split_compound_numbered_heading,
    ocr_heading_negative_signals, classify_ocr_nonheading_kind, heading_capitalization_shape,
    OCR_HEADING_MIN_ACCEPT_SCORE, OCR_HEADING_NEGATIVE_OVERRIDE_SCORE,
    OCR_HEADING_HARD_NEGATIVE_CATEGORIES, safe_progress,
)
from .ocr import ocr_image_lines, OCRLine, OCRUnavailableError
from ..text_repair import HyphenationStats, repair_layout_hyphenation
from ..front_matter import classify_front_matter_line

_BULLET_RE = re.compile(r'^([•\-\*•●]|\d{1,3}[.)])\s+(.*)$')
# Front-matter classifications that disqualify a line from ever being
# picked as the document title (spec section 3). "author" is deliberately
# excluded from this set -- see the usage sites for why.
_TITLE_BLOCKING_FRONT_MATTER_KINDS = {
    "contact", "permission", "copyright", "conference", "publication_metadata",
    "keywords", "affiliation",
}
_CODE_FONT_HINTS = ("mono", "courier", "consolas", "cascadia", "menlo", "conso")
_MONO_FLAG = 1 << 3
_BOLD_FLAG = 1 << 4

# Content-based code detection: a secondary signal for when a PDF renders
# code without a distinguishable monospace font -- common for print-to-PDF
# sources that only vary background color, not font family, for code
# blocks (see the "#include <iostream>" false-positive this was written to
# fix). One strong, unambiguous signal is enough; weaker signals need two
# in a row so a single prose sentence ending in a semicolon isn't
# misclassified as code.
_STRONG_CODE_LINE_RE = re.compile(
    r'^\s*(#include\s*[<"]|#define\s|using\s+namespace\s|import\s+\w|from\s+\w+\s+import\s|'
    r'def\s+\w+\s*\(|class\s+\w+\s*[:({]|public\s+(class|static|void|int|string)|'
    r'private\s+\w|protected\s+\w|\$\s+\S|>>>\s|#!/|namespace\s+\w+\s*\{)'
)
_WEAK_CODE_LINE_RE = re.compile(r'(;\s*$|^\s*[{}]\s*;?\s*$|\w+\s*\([^)]*\)\s*[{;]\s*$|->\w|::\w|^\s*//|^\s*/\*)')
# A line starting with a bare "#" followed by text is, in PDF-extracted
# content (never real Markdown source), always a Python-style comment --
# never a legitimate heading or prose sentence. Recognizing it as a strong
# code signal keeps it grouped into the surrounding code block instead of
# leaking out as a stray paragraph, which would otherwise render as a fake
# ATX heading in the output Markdown (see render.py's paragraph escaping,
# which is a second, independent guard against the same failure mode).
_HASH_COMMENT_RE = re.compile(r'^\s*#\s*\S')


# A markup-tag line ("<html>", "</head>", "<title>Foo</title>") is a strong,
# structural code signal in its own right (spec Fix 4) -- documents *about*
# markup (papers, tutorials, specs) legitimately contain literal examples,
# and those examples should become one atomic ```html block, not scattered
# remaining_html findings across ordinary paragraph text.
_HTML_TAG_LINE_RE = re.compile(r'</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>')
# A complete grammatical sentence -- the run-breaker (spec Fix 1.2): this
# must never be absorbed into a code segment just because it sits next to
# symbolic content (a diagram label, an OCR'd arrow-assignment, axis data).
_SENTENCE_TERMINATOR_RE = re.compile(r'[.!?]$')
_CODE_RUN_DENSITY_THRESHOLD = 0.45
# A bare "=" (not part of ==, !=, <=, >=) is corroborating, not conclusive,
# code/assignment evidence: ordinary English prose essentially never
# contains a literal "=" character on its own, but technical/scientific
# prose routinely states a parameter value this way in a full sentence or a
# figure caption ("Figure 2 MLP results with learning rate = 0.0001",
# "initial value = 1 and learning rate = 0.01 show better convergence") --
# neither is code. A real assignment *statement* is one weak signal here,
# same tier as a semicolon-terminated line; it only tips a run's density
# over the code threshold when several such lines dominate the run, not
# when a couple are diluted among genuinely prose-shaped neighbors (see
# `_is_complete_sentence`/`_is_never_code_label`, which remove the clearer
# prose/caption lines from the run before density is even computed).
_BARE_EQUALS_RE = re.compile(r'(?<![=!<>])=(?!=)')
# A "// comment" is a strong signal wherever it appears in the line, not
# just when the whole line starts with it (a trailing "return -1 // not
# found" is exactly as diagnostic as a comment-only line). Excludes "://"
# so a URL doesn't trip this.
_MID_LINE_COMMENT_RE = re.compile(r'(?<!:)//(?!/)')


def _line_code_strength(text: str) -> str:
    """Per-line signal strength used by `_segment_code_candidate_run`'s
    density evidence: 'strong' (algorithm/comment/markup/known-code-idiom
    line), 'weak' (a corroborating but not by itself sufficient signal), or
    'none'."""
    if (
        _STRONG_CODE_LINE_RE.search(text) or looks_like_algorithm_line(text) or _HASH_COMMENT_RE.match(text)
        or _HTML_TAG_LINE_RE.search(text) or _MID_LINE_COMMENT_RE.search(text)
    ):
        return "strong"
    if _WEAK_CODE_LINE_RE.search(text) or _BARE_EQUALS_RE.search(text):
        return "weak"
    return "none"


def _is_complete_sentence(text: str) -> bool:
    """A run-breaker (spec Fix 1.2): a complete grammatical sentence --
    capitalized start, terminal sentence punctuation, at least 8 words, and
    low symbol density -- is prose, full stop, regardless of what code-like
    content sits next to it. This is what stops one diagram label or OCR'd
    arrow-assignment from converting an unbounded run of surrounding prose
    into a single giant code block."""
    s = text.strip()
    if not s or not s[0].isupper():
        return False
    if not _SENTENCE_TERMINATOR_RE.search(s):
        return False
    words = s.split()
    if len(words) < 8:
        return False
    symbol_chars = sum(1 for c in s if not (c.isalnum() or c.isspace() or c in ".,;:'\"()-"))
    return (symbol_chars / max(len(s), 1)) <= 0.15


def _is_never_code_label(text: str) -> bool:
    """True for a line that is, by construction, never code: it matches one
    of the same figure/caption/diagram-annotation/table-cell/introductory-
    phrase shapes the OCR heading scorer already treats as negative heading
    evidence (`ocr_heading_negative_signals`), or the same typed-block
    shapes `classify_ocr_nonheading_kind` recognizes. A scanned interactive-
    courseware page routinely interleaves this kind of UI chrome
    ("Animation content:", "Static figure:") between real code lines --
    without this, a run's density-based code/prose classification (below)
    sees only "not a complete sentence, not a recognized code idiom either"
    for these lines and folds them into whichever segment they're
    physically adjacent to, which is how a diagram-caption paragraph and an
    unrelated code snippet end up merged into one oversized atomic block."""
    s = text.strip()
    if not s:
        return False
    return bool(ocr_heading_negative_signals(s)) or classify_ocr_nonheading_kind(s) != "unknown"


def _segment_code_candidate_run(lines: List["_Line"]) -> List[Tuple[bool, List["_Line"]]]:
    """Splits a run of otherwise-unclassified lines into contiguous
    (is_code, sub_lines) segments (spec Fix 1): density-based evidence
    (a minimum fraction of a segment's own lines must carry a code signal)
    replaces the old "one strong signal anywhere marks the whole run" rule,
    and a complete-sentence line, or a line that is never code by shape
    (figure/caption/diagram-annotation/table-cell/introductory-phrase),
    always closes out and excludes itself from any code segment. This turns
    a prose -> diagram-label -> prose sequence into paragraph / small
    code-or-figure block / paragraph instead of one merged block spanning
    everything."""
    if not lines:
        return []
    segments: List[Tuple[bool, List["_Line"]]] = []
    current: List["_Line"] = []

    def flush():
        if not current:
            return
        strengths = [_line_code_strength(l.text) for l in current]
        strong_count = sum(1 for s in strengths if s == "strong")
        weak_count = sum(1 for s in strengths if s == "weak")
        density = (strong_count + 0.5 * weak_count) / len(current)
        segments.append((density >= _CODE_RUN_DENSITY_THRESHOLD, list(current)))

    for line in lines:
        if _is_complete_sentence(line.text) or _is_never_code_label(line.text):
            flush()
            current = []
            segments.append((False, [line]))
            continue
        current.append(line)
    flush()
    return segments


def _looks_like_html_run(lines_text: List[str]) -> bool:
    if not lines_text:
        return False
    tagged = sum(1 for t in lines_text if _HTML_TAG_LINE_RE.search(t))
    return (tagged / len(lines_text)) >= 0.5


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    mono: bool
    page: int
    y0: float
    y1: float
    x0: float
    # Set by the compound-numbered-heading pre-pass (Fix 3) for a heading
    # half it split out of an OCR line that glued a section title and its
    # opening sentence together. Bypasses the font-size-bucket gate (this
    # synthetic half never had its own font size to bucket in the first
    # place) while still going through the same content classifier.
    forced_heading: bool = False
    # True for every line recovered via OCR rather than native PDF text.
    # `size`/`y0`/`y1`/`x0` are still real, OCR-measured bounding-box
    # evidence (see `_convert_ocr_lines`) -- this flag exists so the heading
    # classifier can require *additional* corroborating signals for OCR
    # lines specifically (font metadata from a real PDF text run is far more
    # trustworthy than a line height measured off a rasterized, potentially
    # skewed/noisy scan), never so it can fabricate evidence that isn't
    # there.
    from_ocr: bool = False
    # Mean per-word OCR confidence (0-100) for this line, read directly from
    # tesseract's own `image_to_data` output; None when the line didn't come
    # from OCR, -1 when OCR produced no confidence value for it.
    ocr_conf: Optional[float] = None
    # Right edge of the line's bounding box, used only for the first-page
    # title resolver's centering check (spec section 2) -- every other
    # consumer of `_Line` only ever needed the left edge (`x0`) for column
    # banding/indentation, which is why this wasn't carried before. Defaults
    # to `x0` (a zero-width line) when unknown so callers that do need it
    # degrade to "not centered" rather than crashing.
    x1: float = 0.0
    # True only for the line(s) the first-page title resolver merged into
    # the document's single resolved title (spec section 2); bypasses
    # ordinary heading-level resolution entirely -- this line is always H1,
    # regardless of position/numbering, distinct from `forced_heading`
    # (compound-heading splitting) which still goes through normal level
    # resolution.
    is_resolved_title: bool = False


def _order_blocks_by_column(blocks: list, page_width: float) -> Tuple[list, bool]:
    """Reorders top-level text blocks into left-column-then-right-column
    reading order when the page looks like a two-column layout (spec
    section 3.3/9's multi-column requirement), so column text doesn't get
    interleaved line-by-line in document order. Falls back to natural
    top-to-bottom order for single-column pages -- most PDFs -- since this
    heuristic requires a clear two-band split to activate at all.
    Returns (ordered_blocks, was_multicolumn)."""
    if len(blocks) < 4 or page_width <= 0:
        return blocks, False
    mid = page_width / 2
    margin = page_width * 0.04
    left, right, wide = [], [], []
    for b in blocks:
        x0, x1 = b["bbox"][0], b["bbox"][2]
        if x1 <= mid + margin:
            left.append(b)
        elif x0 >= mid - margin:
            right.append(b)
        else:
            wide.append(b)
    if len(left) < 2 or len(right) < 2:
        return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])), False
    left.sort(key=lambda b: b["bbox"][1])
    right.sort(key=lambda b: b["bbox"][1])
    wide.sort(key=lambda b: b["bbox"][1])
    top_y = min(left[0]["bbox"][1], right[0]["bbox"][1])
    leading_wide = [b for b in wide if b["bbox"][1] < top_y]
    trailing_wide = [b for b in wide if b["bbox"][1] >= top_y]
    return leading_wide + left + right + trailing_wide, True


def _flatten_lines(page, page_no: int) -> Tuple[List["_Line"], bool]:
    lines: List[_Line] = []
    data = page.get_text("dict")
    text_blocks = [b for b in data.get("blocks", []) if b.get("type") == 0 and b.get("lines")]
    text_blocks, was_multicolumn = _order_blocks_by_column(text_blocks, page.rect.width)
    for block in text_blocks:
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            sizes = [s.get("size", 0) for s in spans]
            fonts = [s.get("font", "").lower() for s in spans]
            flags = [s.get("flags", 0) for s in spans]
            bold = any(f & _BOLD_FLAG for f in flags) or any("bold" in fn for fn in fonts)
            mono = any(f & _MONO_FLAG for f in flags) or any(any(h in fn for h in _CODE_FONT_HINTS) for fn in fonts)
            bbox = line.get("bbox", [0, 0, 0, 0])
            lines.append(_Line(
                text=text, size=max(sizes) if sizes else 0, bold=bold, mono=mono,
                page=page_no, y0=bbox[1], y1=bbox[3], x0=bbox[0], x1=bbox[2],
            ))
    return lines, was_multicolumn


_MIN_OCR_LINE_HEIGHT_PT = 6.0


def _convert_ocr_lines(ocr_lines: List["OCRLine"], page_no: int, dpi: int) -> List["_Line"]:
    """Turns tesseract's per-line OCR output into `_Line` objects on the same
    point-space coordinate system native PDF text already uses, so the
    existing gap-based paragraph logic and font-size-bucket heading logic
    downstream both just work on OCR'd pages without any special-casing --
    and, critically, without *inventing* any of that evidence. Isolation
    (blank space around a line) used to be treated as if it were typography
    evidence and given a synthesized larger font size; it is not, and doing
    that manufactured a false "this looks like a heading" signal for any
    isolated line at all, including figure labels, axis ticks, and captions.
    `image_to_data` gives every OCR'd line a *real* bounding box and mean
    word confidence straight from the OCR engine -- line height here is
    genuine (if noisier than native font metadata) visual evidence, and gets
    treated as exactly that: one signal among several for the multi-signal
    OCR heading scorer (`_OcrHeadingScorer`), never sufficient on its own."""
    scale = 72.0 / dpi if dpi else 1.0
    out: List[_Line] = []
    for ln in ocr_lines:
        text = ln.text.strip()
        if not text:
            continue
        height_pt = max((ln.y1 - ln.y0) * scale, _MIN_OCR_LINE_HEIGHT_PT)
        out.append(_Line(
            text=text, size=height_pt, bold=False, mono=False, page=page_no,
            y0=ln.y0 * scale, y1=ln.y1 * scale, x0=ln.x0 * scale, x1=ln.x1 * scale,
            from_ocr=True, ocr_conf=ln.conf,
        ))
    return out


def _detect_repeated_running_lines(pages_lines: List[List[_Line]]) -> set:
    """Return the set of normalized text strings that repeat as a header/footer.

    Threshold is deliberately lower than "majority of pages": conference/
    journal templates commonly alternate two distinct running headers on
    odd/even pages (e.g. paper title on odd pages, venue/date on even pages),
    so each individual string only appears on roughly half the pages even
    though the *pattern* -- "some running header on nearly every page" -- is
    unambiguous. A 60%-of-all-pages threshold (the previous value) misses
    exactly this alternating case; requiring ~1/3 of pages still guards
    against a coincidental one-or-two-page repeat while catching the
    alternating pattern (see the real regression fixture this was found on:
    a 12-page ACM paper where the venue-string header and the author-name
    footer each appear on exactly 6 of 12 pages).

    Position (top/bottom 8% of the page) is a reliable signal for native
    text, whose y-coordinates are real page geometry. `_convert_ocr_lines`
    now carries real, OCR-measured y-values too (`image_to_data`'s own
    bounding boxes), but tesseract's *reading order* for a two-column page
    still routinely visits a footer/header line somewhere in the middle of
    the page-order stream rather than at its start/end -- the position
    counters below key off the coordinate, not stream order, so they still
    catch it, but this also runs a position-independent frequency count for
    lines with enough words to not plausibly be a coincidentally-reused
    short section heading ("Summary" reappearing once per chapter is
    legitimate content; an 8-word author byline reappearing verbatim on 6 of
    12 pages is not) as a second, redundant net."""
    if len(pages_lines) < 3:
        return set()
    top_counter: Counter = Counter()
    bottom_counter: Counter = Counter()
    any_position_counter: Counter = Counter()
    for lines in pages_lines:
        if not lines:
            continue
        page_top = min(l.y0 for l in lines)
        page_bottom = max(l.y1 for l in lines)
        span = max(page_bottom - page_top, 1.0)
        seen_this_page = set()
        for l in lines:
            norm = re.sub(r"\d+", "#", l.text.strip().lower())
            if (l.y0 - page_top) / span < 0.08:
                top_counter[norm] += 1
            elif (page_bottom - l.y1) / span < 0.08:
                bottom_counter[norm] += 1
            if len(norm.split()) >= 4 and norm not in seen_this_page:
                any_position_counter[norm] += 1
                seen_this_page.add(norm)
    threshold = max(3, math.ceil(len(pages_lines) * 0.34))
    repeated = {t for t, c in top_counter.items() if c >= threshold}
    repeated |= {t for t, c in bottom_counter.items() if c >= threshold}
    repeated |= {t for t, c in any_position_counter.items() if c >= threshold}
    return repeated


_MIN_PREFIX_FRAGMENT_LEN = 4


def _is_repeated_prefix_fragment(norm: str, repeated: set) -> bool:
    """True when `norm` is a short, truncated OCR fragment of a line already
    identified as repeated page furniture -- e.g. a running author-name
    header that OCR only partially recovers on some pages ("Taic" out of a
    longer name). Deliberately narrow (short candidate only) so an unrelated
    legitimate heading that happens to share a first word with a repeated
    header ("The" in both a running title and an ordinary section heading)
    isn't swept up as furniture."""
    if len(norm) < _MIN_PREFIX_FRAGMENT_LEN or len(norm.split()) > 2:
        return False
    return any(len(rep) > len(norm) and rep.startswith(norm) for rep in repeated)


_BULLET_GLYPH_X_TOLERANCE = 3.0


def _normalize_ocr_bullet_glyphs(lines: List["_Line"], log: FindingLog) -> int:
    """Rewrites confirmed OCR bullet-glyph lines (spec section 12) into a
    standard `- ` prefix in place, so the existing `_BULLET_RE`-based list
    assembly in the main extraction loop picks them up as ordinary list
    items with no further special-casing. A candidate is only rewritten
    when it has run evidence: at least one other candidate on the same page
    at matching left-indentation (`x0`, spec "bounding-box alignment") --
    see `ocr_bullet_glyph_match`'s docstring for why a lone match is left
    alone instead of guessed at. Returns the count of lines normalized."""
    from ..text_repair import ocr_bullet_glyph_match

    candidates = []
    for i, line in enumerate(lines):
        if line.mono or line.forced_heading:
            continue
        body = ocr_bullet_glyph_match(line.text)
        if body is not None:
            candidates.append((i, line, body))

    if not candidates:
        return 0

    groups: dict = {}
    for i, line, body in candidates:
        key = None
        for existing_key in groups:
            p, x = existing_key
            if p == line.page and abs(x - line.x0) <= _BULLET_GLYPH_X_TOLERANCE:
                key = existing_key
                break
        if key is None:
            key = (line.page, line.x0)
        groups.setdefault(key, []).append((i, line, body))

    normalized = 0
    for (page, _x0), members in groups.items():
        if len(members) < 2:
            i, line, body = members[0]
            log.warning(
                "probable_bullet_glyph_not_normalized",
                f"Line {line.text!r} on page {page} looks like a single OCR bullet-glyph "
                f"artifact but has no matching sibling item at the same indentation; left as prose",
                page=page,
            )
            continue
        for i, line, body in members:
            line.text = f"- {body}"
            normalized += 1
        log.fixed(
            "ocr_bullet_glyph_normalized",
            f"Normalized {len(members)} OCR bullet-glyph line(s) on page {page} into a Markdown list",
            page=page,
        )
    return normalized


def _extract_tables(page):
    try:
        finder = page.find_tables()
        out = []
        for t in finder.tables:
            try:
                rows = t.extract()
            except Exception:
                continue
            if rows and len(rows) > 1:
                out.append((rows[0], rows[1:], t.bbox))
        return out
    except Exception:
        return []


_TITLE_MAX_CANDIDATE_LINE_INDEX = 8
_TITLE_MIN_WORDS = 2
_TITLE_MAX_WORDS = 24


def _detect_visual_title(page1_lines: List["_Line"], page_width: float) -> Optional[dict]:
    """First-page visual title resolver (spec section 2). Groups the
    leading run of large-font, near-top line(s) on page 1 into a single
    title candidate, merging a wrapped multi-line title ("Introducing
    Computer Systems from a Programmer's" / "Perspective") into one
    logical title rather than leaving it as the first heading candidate
    the ordinary per-line classifier would separately (mis)judge.

    Pure function of already-extracted `_Line`s (no PDF handle needed), so
    it is directly unit-testable. Returns None when no confident title
    region is found -- callers keep the existing fallback chain (first
    heading-shaped line, then `normalize.py`'s metadata/filename
    synthesis) in that case; this function never invents text."""
    if not page1_lines:
        return None
    sizes = [round(l.size, 1) for l in page1_lines if l.size]
    if not sizes:
        return None
    max_size = max(sizes)
    other_sizes = [s for s in sizes if s < max_size]
    # The title's font must be a clear outlier, not merely whatever the
    # first line happens to be sized at -- a document with one uniform
    # font size must not have its opening line crowned "the title" on
    # typography grounds alone.
    if other_sizes and max_size < max(other_sizes) * 1.15:
        return None

    candidate_idx = [
        i for i, l in enumerate(page1_lines)
        if round(l.size, 1) == max_size and i < _TITLE_MAX_CANDIDATE_LINE_INDEX
    ]
    if not candidate_idx:
        return None

    # Maximal consecutive run starting at the first candidate: a wrapped
    # multi-line title, not scattered same-size occurrences elsewhere on
    # the page (e.g. a pull-quote sharing the title's font).
    start = candidate_idx[0]
    run = [start]
    for i in candidate_idx[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            break
    title_lines = [page1_lines[i] for i in run]

    combined = " ".join(l.text.strip() for l in title_lines)
    words = combined.split()
    if not (_TITLE_MIN_WORDS <= len(words) <= _TITLE_MAX_WORDS):
        return None
    if numbering_depth(combined) is not None:
        return None
    if classify_front_matter_line(combined) in _TITLE_BLOCKING_FRONT_MATTER_KINDS:
        return None
    # Basic lexical plausibility (spec section 5: never select clearly
    # garbled text as the title). The large-font region on a scanned page 1
    # is sometimes a logo/graphic that OCR mangles into near-nonsense
    # ("-: [EEEEEES") rather than a real title -- font size alone can't
    # distinguish that from a genuine title, so this requires the combined
    # text to at least look like language: start with an alphanumeric
    # character (not stray punctuation) and be majority-alphabetic overall,
    # the same low bar `is_heading_candidate` applies elsewhere.
    stripped = combined.strip()
    if not stripped or not stripped[0].isalnum():
        return None
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    non_space_chars = sum(1 for c in stripped if not c.isspace())
    if non_space_chars and (alpha_chars / non_space_chars) < 0.6:
        return None

    centered = True
    if page_width > 0:
        for l in title_lines:
            mid = (l.x0 + l.x1) / 2 if l.x1 > l.x0 else l.x0
            if abs(mid - page_width / 2) > page_width * 0.22:
                centered = False
                break

    confidence = 0.85 if centered else 0.7
    if len(run) > 1:
        confidence = min(confidence + 0.08, 0.97)

    return {
        "text": combined,
        "line_ids": [id(l) for l in title_lines],
        "pages": sorted({l.page for l in title_lines}),
        "centered": centered,
        "multi_line": len(run) > 1,
        "confidence": round(confidence, 2),
    }


def _split_compound_heading_lines(lines: List["_Line"]) -> List["_Line"]:
    """Pre-pass (spec Fix 3): decomposes any line matching the compound
    "<numbering> <Title>. <Sentence...>" shape into a forced-heading line
    plus a plain paragraph-text line, gated on outline-sequence plausibility
    so a citation/data/figure number never gets split as if it were a
    section heading. Runs once, in reading order, before any size/heading
    classification -- the split heading half needs `forced_heading=True`
    since it has no font-size signal of its own to be bucketed by."""
    out: List["_Line"] = []
    prev_number: Optional[Tuple[int, ...]] = None
    for line in lines:
        split = split_compound_numbered_heading(line.text)
        handled = False
        if split is not None:
            num_str, title, rest = split
            candidate = numbering_tuple(num_str)
            if candidate is not None and is_plausible_outline_successor(prev_number, candidate):
                prev_number = candidate
                out.append(_Line(text=f"{num_str} {title}", size=line.size, bold=line.bold, mono=False,
                                  page=line.page, y0=line.y0, y1=line.y0 + 1, x0=line.x0, forced_heading=True))
                out.append(_Line(text=rest, size=line.size, bold=line.bold, mono=False,
                                  page=line.page, y0=line.y0 + 1, y1=line.y1, x0=line.x0))
                handled = True
        if not handled:
            if is_heading_candidate(line.text):
                candidate = numbering_tuple(line.text)
                if candidate is not None and is_plausible_outline_successor(prev_number, candidate):
                    prev_number = candidate
            out.append(line)
    return out


def _build_figure_context(
    pages_lines: List[List["_Line"]], pdf, page_count: int, progress: Optional[Callable] = None,
):
    """Returns an `is_near_figure(line) -> bool` predicate for the axis/
    legend heading guard (spec Fix 2): true when `line` sits near a detected
    embedded image on its page, or near a caption line (works for both
    native PDFs, via real image bounding boxes, and OCR'd pages, where a
    caption is the only reliable "a figure is here" signal since the whole
    page is one rasterized image rather than separately-extractable
    objects)."""
    image_zones: dict = {}
    for pno in range(page_count):
        current_page = pno + 1
        # Same throttle as the extraction-pass progress reporter above
        # (every 25 pages and on the final page): frequent enough that a
        # 31,000+ page PDF never looks dead mid-scan, infrequent enough
        # that this never becomes the bottleneck itself.
        if current_page % 25 == 0 or current_page == page_count:
            safe_progress(progress, "scanning_images", current=current_page, total=page_count)
        try:
            infos = pdf[pno].get_image_info()
        except Exception:
            infos = []
        boxes = [info["bbox"] for info in infos if "bbox" in info]
        if boxes:
            image_zones[pno + 1] = boxes

    caption_positions: dict = {}
    for lines in pages_lines:
        for l in lines:
            if looks_like_caption_text(l.text):
                caption_positions.setdefault(l.page, []).append(l.y0)

    def is_near_figure(line: "_Line") -> bool:
        for bbox in image_zones.get(line.page, []):
            if bbox[1] - 80 <= line.y0 <= bbox[3] + 80:
                return True
        for cap_y in caption_positions.get(line.page, []):
            if abs(line.y0 - cap_y) <= 60:
                return True
        return False

    return is_near_figure


_OCR_LOOKAHEAD_BODY_WORDS = 12
_OCR_HIGH_CONF = 80.0
_OCR_LOW_CONF = 40.0
_OCR_STRONG_SIZE_RATIO = 1.25


def _normalize_repeat_key(text: str) -> str:
    s = re.sub(r"\d+", "#", text.strip().lower())
    return re.sub(r"[^\w#\s]", "", s)


class _OcrHeadingScorer:
    """Multi-signal OCR heading promotion (spec "OCR heading promotion"):
    whitespace isolation alone is never sufficient. A candidate must clear a
    minimum count of independent signals -- lexical heading shape, real
    OCR-measured bbox evidence, position near the start of a page/section,
    substantial body content following it, high OCR confidence,
    capitalization or numbering consistency with headings *already
    confirmed earlier in this document*, or a genuinely repeated numbered-
    heading pattern -- weighed against negative evidence for figure/image/
    diagram/table-cell/caption shapes (an absolute bar) and softer negative
    signals (bare quantity labels, short colon-terminated intros, low OCR
    confidence, repeated verbatim label text, short fragments with no
    following body), which only raise the bar rather than blocking outright.

    Must be consulted at most once per physical decision per line, in
    document order, since several signals are judged against state
    (confirmed heading shapes/numbering) accumulated from earlier
    decisions -- results are memoized by line identity because the pipeline
    asks the same line's heading-candidacy question from more than one call
    site (heading-level resolution, code-run classification, final
    assembly), and each of those must see the same answer.
    """

    def __init__(self, lines: List["_Line"], body_size: float):
        self._body_size = body_size
        self._decisions: dict = {}
        self._confirmed_shapes: set = set()
        self._confirmed_numbered_depths: set = set()
        self._prev_number: Optional[Tuple[int, ...]] = None
        # Isolation must be computed first: it gates who even *has* a
        # lookahead value below (only a genuinely isolated, heading-shaped
        # line is a real "next candidate" boundary -- an ordinary,
        # unpunctuated mid-paragraph line that happens to start with a
        # capital letter and stay under 12 words is not one, and treating it
        # as one was cutting the lookahead off after zero words of real body
        # text).
        self._isolated_ids = self._compute_isolated(lines)  # instance method: needs self._body_size
        self._lookahead_words = self._compute_lookahead(lines, self._isolated_ids)
        self._page_start_ids = self._compute_page_starts(lines)
        self._interrupts_next = self._compute_interrupts_next(lines)
        self._exact_counts = Counter(_normalize_repeat_key(l.text) for l in lines if l.from_ocr)
        self.candidates = 0
        self.accepted = 0
        self.rejected = 0

    @staticmethod
    def _compute_lookahead(lines: List["_Line"], isolated_ids: set) -> dict:
        """For each isolated OCR heading-candidate line, the word count of
        the body text that follows it before the next *isolated*
        heading-shaped line (spec: "surrounding body content" / "short
        fragments with no following body"). Must stop only at another
        genuinely isolated candidate, not any capitalized, under-12-word
        line -- an ordinary paragraph-continuation line (OCR wraps a
        sentence across lines with no terminal punctuation on any one of
        them) can trivially match that shape without being a section
        boundary."""
        lookahead: dict = {}
        n = len(lines)
        for i, line in enumerate(lines):
            if not (line.from_ocr and id(line) in isolated_ids and is_heading_candidate(line.text)):
                continue
            words = 0
            for j in range(i + 1, n):
                nxt = lines[j]
                if nxt.page != line.page:
                    break
                if id(nxt) in isolated_ids and is_heading_candidate(nxt.text):
                    break
                words += len(nxt.text.split())
                if words >= _OCR_LOOKAHEAD_BODY_WORDS:
                    break
            lookahead[id(line)] = words
        return lookahead

    @staticmethod
    def _compute_interrupts_next(lines: List["_Line"]) -> set:
        """Lines whose immediately following line starts with a lowercase
        letter -- proof the candidate's own text is grammatically the first
        half of a sentence that continues past it, not a standalone heading
        (spec negative-evidence category "Sentence continuations"). This is
        the "next block continues my sentence" half of the check; the
        candidate's *own* leading-lowercase case is already excluded by
        `is_heading_candidate` itself."""
        interrupts: set = set()
        n = len(lines)
        for i, line in enumerate(lines):
            if not line.from_ocr:
                continue
            nxt = lines[i + 1] if i + 1 < n and lines[i + 1].page == line.page else None
            if nxt is None:
                continue
            nxt_text = nxt.text.strip()
            first_alpha = next((c for c in nxt_text if c.isalpha()), None)
            if first_alpha is not None and first_alpha.islower():
                interrupts.add(id(line))
        return interrupts

    @staticmethod
    def _compute_page_starts(lines: List["_Line"]) -> set:
        """The first couple of OCR lines on each page (spec: "position near
        the beginning of a logical section") -- a coarse proxy for "start of
        a section" that's fair for both a first-page title and a first
        subsection line on a later scanned page."""
        starts: set = set()
        seen_on_page: dict = {}
        for line in lines:
            if not line.from_ocr:
                continue
            count = seen_on_page.get(line.page, 0)
            if count < 2:
                starts.add(id(line))
            seen_on_page[line.page] = count + 1
        return starts

    def _compute_isolated(self, lines: List["_Line"]) -> set:
        """Real whitespace isolation, measured from the genuine OCR bounding
        boxes rather than synthesized -- kept as exactly one signal among
        several, never sufficient by itself (spec: "Whitespace isolation
        alone is insufficient"). The gap threshold is scaled to the
        surrounding *body* line height, not the candidate line's own size:
        a large heading's leading above/below it is routinely less than
        0.9x its own (much taller) font size even though it is clearly
        visually separated from the smaller body text around it, so scaling
        to the candidate's own size under-counts exactly the large,
        genuinely isolated headings this is supposed to catch."""
        isolated: set = set()
        n = len(lines)
        # A real, densely-typeset page (e.g. a two-column ACM template) puts
        # only ~1.5-2 extra points of paragraph-before/after spacing around
        # a heading -- nowhere near a full extra line height -- while an
        # ordinary line-to-line gap *within* the same paragraph measures
        # close to zero. 0.5x body line height comfortably separates those
        # two cases on real OCR'd measurements without requiring a nearly
        # blank-line-sized gap the way native-PDF paragraph-break detection
        # can afford to (that logic is tuned for gaps between whole
        # paragraphs, not for "how much extra space sits around one line").
        gap_scale = max(self._body_size, 6.0) * 0.5
        for i, line in enumerate(lines):
            if not line.from_ocr:
                continue
            prev = lines[i - 1] if i > 0 and lines[i - 1].page == line.page else None
            nxt = lines[i + 1] if i + 1 < n and lines[i + 1].page == line.page else None
            gap_before = (line.y0 - prev.y1) if prev is not None else 999.0
            gap_after = (nxt.y0 - line.y1) if nxt is not None else 999.0
            if gap_before > gap_scale and gap_after > gap_scale:
                isolated.add(id(line))
        return isolated

    def is_isolated(self, line: "_Line") -> bool:
        return id(line) in self._isolated_ids

    def decide(self, line: "_Line") -> bool:
        key = id(line)
        cached = self._decisions.get(key)
        if cached is not None:
            return cached["accept"]

        self.candidates += 1
        text = line.text.strip()
        positives: List[str] = ["lexical_heading_likelihood"]
        matched_negatives = ocr_heading_negative_signals(text)
        hard_hit = [n for n in matched_negatives if n in OCR_HEADING_HARD_NEGATIVE_CATEGORIES]
        soft_hit = [n for n in matched_negatives if n not in OCR_HEADING_HARD_NEGATIVE_CATEGORIES]

        candidate_number = numbering_tuple(text)
        depth = numbering_depth(text)
        if candidate_number is not None:
            if len(candidate_number) >= 2:
                positives.append("numbering_consistent_with_hierarchy")
            elif self._prev_number is not None and is_plausible_outline_successor(self._prev_number, candidate_number):
                positives.append("numbering_consistent_with_hierarchy")

        if key in self._page_start_ids:
            positives.append("position_near_section_start")

        lookahead_words = self._lookahead_words.get(key, 0)
        if lookahead_words >= _OCR_LOOKAHEAD_BODY_WORDS:
            positives.append("surrounding_body_content")
        elif len(text.split()) <= 2:
            soft_hit.append("short_fragment_no_body")

        if line.ocr_conf is not None and line.ocr_conf >= 0:
            if line.ocr_conf >= _OCR_HIGH_CONF:
                positives.append("high_ocr_confidence")
            elif line.ocr_conf < _OCR_LOW_CONF:
                soft_hit.append("low_ocr_confidence")

        shape = heading_capitalization_shape(text)
        if self._confirmed_shapes and shape in self._confirmed_shapes:
            positives.append("capitalization_consistent_with_confirmed_headings")

        if depth is not None and depth in self._confirmed_numbered_depths:
            positives.append("repeated_heading_pattern")

        if line.size > self._body_size * _OCR_STRONG_SIZE_RATIO:
            positives.append("strong_visual_bbox_evidence")

        if key in self._isolated_ids:
            positives.append("whitespace_isolation")

        norm = _normalize_repeat_key(text)
        if len(text.split()) <= 4 and self._exact_counts.get(norm, 0) >= 3:
            soft_hit.append("repeated_label_text")

        if key in self._interrupts_next:
            # A "Sentence continuation" (spec negative-evidence list): the
            # line right after this one starts with a lowercase letter,
            # meaning this candidate's own text is grammatically the first
            # half of that sentence, not a standalone heading -- unlike the
            # other soft categories, this is a structural proof, not a
            # probabilistic pattern, so no amount of positive score should
            # override it.
            hard_hit.append("interrupts_following_sentence")

        score = len(positives)
        if hard_hit:
            accept = False
        elif soft_hit:
            # Each additional independent soft-negative category beyond the
            # first raises the bar further -- one weak negative signal (e.g.
            # just an ambiguous bare numeric prefix) can still be overridden
            # by ordinary strong evidence, but several independent soft
            # negatives compounding (e.g. a colon-terminated intro phrase
            # AND verbatim repetition elsewhere in the document) is a much
            # stronger combined signal that this was never a real heading.
            required = OCR_HEADING_NEGATIVE_OVERRIDE_SCORE + 2 * (len(soft_hit) - 1)
            accept = score >= required
        else:
            accept = score >= OCR_HEADING_MIN_ACCEPT_SCORE

        if accept:
            self.accepted += 1
            self._confirmed_shapes.add(shape)
            if depth is not None:
                self._confirmed_numbered_depths.add(depth)
            if candidate_number is not None:
                self._prev_number = candidate_number
        else:
            self.rejected += 1

        self._decisions[key] = {
            "accept": accept, "score": score, "positives": positives,
            "negatives": hard_hit + soft_hit,
        }
        return accept

    def info(self, line: "_Line") -> dict:
        return self._decisions.get(id(line), {"accept": False, "score": 0, "positives": [], "negatives": []})


class PdfExtractor:
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def extract(
        self, path: Path, ocr_config: OCRConfig, *, progress: Optional[Callable] = None,
    ) -> Tuple[Document, FindingLog]:
        import pymupdf

        log = FindingLog()
        pdf = pymupdf.open(str(path))

        if pdf.needs_pass:
            # Some PDFs restrict printing/editing with an owner password but
            # allow viewing with none; only treat this as a hard failure if
            # an empty password genuinely doesn't unlock it.
            if not pdf.authenticate(""):
                pdf.close()
                log.error(
                    "pdf_password_protected",
                    "This PDF is password-protected and could not be opened with an empty "
                    "password. Nothing was extracted.",
                    suggested_action="Provide a decrypted copy (e.g. `qpdf --decrypt in.pdf out.pdf`) and re-upload.",
                )
                return Document(metadata=DocumentMetadata(source_path=str(path), source_format="pdf")), log

        page_count = pdf.page_count
        title = clean_title((pdf.metadata or {}).get("title"))

        doc = Document(metadata=DocumentMetadata(
            source_path=str(path), source_format="pdf", title=title, page_count=page_count,
        ))
        order = 0

        def new_id() -> str:
            nonlocal order
            order += 1
            return f"b{order:04d}"

        pages_lines: List[List[_Line]] = []
        ocr_used_any = False
        native_text_pages = 0
        ocr_pages = 0
        failed_pages = 0
        multicolumn_pages = 0

        def _report_page_progress(current_page: int) -> None:
            # Every 25 pages and on the final page -- frequent enough that a
            # 31,000+ page PDF never looks dead mid-extraction, infrequent
            # enough that this never becomes the bottleneck itself. Called
            # from every branch below (native-text and OCR alike) so an
            # OCR-heavy document still ticks.
            if current_page % 25 == 0 or current_page == page_count:
                safe_progress(progress, "extracting", current=current_page, total=page_count)

        for pno in range(page_count):
            page = pdf[pno]
            text_density = len(page.get_text().strip())
            is_scanned_candidate = text_density < ocr_config.min_text_chars_per_page
            if is_scanned_candidate:
                if ocr_config.mode == "off":
                    log.warning(
                        "page_looks_scanned_ocr_disabled",
                        f"Page {pno + 1} has almost no extractable text and looks scanned, but OCR is disabled",
                        page=pno + 1,
                        suggested_action="Re-run with OCR enabled (--ocr / ocr mode 'auto' or 'on') to recover this page.",
                    )
                    pages_lines.append([])
                    failed_pages += 1
                    _report_page_progress(pno + 1)
                    continue
                try:
                    pix = page.get_pixmap(dpi=ocr_config.dpi)
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_lines_raw = ocr_image_lines(img, language=ocr_config.language)
                    ocr_used_any = True
                    if not ocr_lines_raw:
                        log.warning("ocr_low_confidence", f"OCR produced no text for page {pno + 1}", page=pno + 1)
                        pages_lines.append([])
                        failed_pages += 1
                    else:
                        converted = _convert_ocr_lines(ocr_lines_raw, pno + 1, ocr_config.dpi)
                        pages_lines.append(converted)
                        ocr_pages += 1
                        char_count = sum(len(l.text) for l in converted)
                        log.info("ocr_page_used", f"Page {pno + 1} extracted via OCR ({char_count} chars)", page=pno + 1)
                except OCRUnavailableError as e:
                    log.error("ocr_unavailable", str(e), page=pno + 1,
                              suggested_action="Install tesseract-ocr or disable OCR for this run.")
                    pages_lines.append([])
                    failed_pages += 1
                _report_page_progress(pno + 1)
                continue
            lines, was_multicolumn = _flatten_lines(page, pno + 1)
            pages_lines.append(lines)
            native_text_pages += 1
            _report_page_progress(pno + 1)
            if was_multicolumn:
                multicolumn_pages += 1

        doc.metadata.ocr_used = ocr_used_any
        doc.metadata.extra["native_text_pages"] = native_text_pages
        doc.metadata.extra["ocr_pages"] = ocr_pages
        doc.metadata.extra["failed_pages"] = failed_pages
        if multicolumn_pages:
            log.info("multi_column_layout_detected",
                      f"Reordered {multicolumn_pages} page(s) from a detected two-column layout into "
                      f"left-column-then-right-column reading order")
        repeated = _detect_repeated_running_lines(pages_lines)
        if repeated:
            log.fixed("removed_repeated_header_footer",
                      f"Removed {len(repeated)} distinct repeated header/footer line(s) seen on a "
                      f"recurring position across pages")

        # The presumptive document title is the first heading-shaped line in
        # reading order. If that same normalized text also got flagged as a
        # repeated running header/footer (a running-title-header template,
        # e.g. "Linear Algebraic Depth-First Search" repeated on every odd
        # page), its first, genuine occurrence must still survive -- only
        # the later repeats are page furniture. Every other repeated line
        # (venue/date footers, author-name footers, page numbers) never
        # legitimately serves as content, so all of its occurrences,
        # including the first, are dropped.
        title_line_norm: Optional[str] = None
        for lines in pages_lines:
            for l in lines:
                if looks_like_title_candidate(l.text):
                    title_line_norm = re.sub(r"\d+", "#", l.text.strip().lower())
                    break
            if title_line_norm is not None:
                break

        all_lines: List[_Line] = []
        title_kept = False
        for lines in pages_lines:
            for l in lines:
                norm = re.sub(r"\d+", "#", l.text.strip().lower())
                if norm in repeated or _is_repeated_prefix_fragment(norm, repeated):
                    if norm == title_line_norm and not title_kept:
                        title_kept = True
                    else:
                        continue
                all_lines.append(l)

        # First-page visual title resolution (spec section 2): a title that
        # PDF metadata didn't already give us and that spans multiple
        # visual lines (a wrapped title) would otherwise never become one
        # coherent H1 candidate -- the per-line classifier only ever judges
        # one line at a time, so the true title's first line (often not a
        # "complete" title by itself) gets passed over and whatever
        # heading-shaped line comes next (frequently the "Abstract" label)
        # wins the H1 slot instead. PDF-metadata title takes priority when
        # present (spec's own evidence-source ordering); this only runs the
        # visual resolver as a second source, never overriding metadata.
        title_evidence: Optional[dict] = None
        if not title:
            title_result = _detect_visual_title(
                pages_lines[0] if pages_lines else [], pdf[0].rect.width if page_count else 0.0,
            )
            if title_result is not None:
                consumed_ids = set(title_result["line_ids"])
                match_positions = [i for i, l in enumerate(all_lines) if id(l) in consumed_ids]
                if match_positions:
                    insert_at = min(match_positions)
                    kept_lines = [l for l in all_lines if id(l) not in consumed_ids]
                    merged_size = max(l.size for l in all_lines if id(l) in consumed_ids)
                    merged = _Line(
                        text=title_result["text"], size=merged_size, bold=True, mono=False,
                        page=title_result["pages"][0], y0=0.0, y1=0.0, x0=0.0,
                        is_resolved_title=True,
                    )
                    all_lines = kept_lines
                    all_lines.insert(min(insert_at, len(all_lines)), merged)
                    title = title_result["text"]
                    doc.metadata.title = title

                    rejected = []
                    for l in all_lines:
                        if l is merged:
                            continue
                        if looks_like_title_candidate(l.text):
                            norm = l.text.strip().lower().rstrip(":.")
                            if norm == "abstract":
                                rejected.append({"text": l.text.strip(), "reason": "section label, not document title"})
                            break
                    title_evidence = {
                        "selected_title": title_result["text"],
                        "source": "first_page_visual_title",
                        "source_block_ids": [],  # filled in once block_ids are assigned, below
                        "confidence": title_result["confidence"],
                        "rejected_candidates": rejected,
                    }
        elif title:
            title_evidence = {
                "selected_title": title,
                "source": "pdf_metadata",
                "source_block_ids": [],
                "confidence": 0.99,
                "rejected_candidates": [],
            }

        # Compound numbered heading recovery (spec Fix 3): OCR frequently
        # emits a numbered section's title and its opening sentence on one
        # physical line ("3.2.3 Suffix Extraction. For efficiency..."),
        # which the plain classifier rejects wholesale as prose. This splits
        # any such line into a real heading half plus a paragraph-text half
        # *before* heading/size classification runs, gated on outline-
        # sequence plausibility so citation/data numbers are never split.
        all_lines = _split_compound_heading_lines(all_lines)

        bullets_normalized = _normalize_ocr_bullet_glyphs(all_lines, log)
        if bullets_normalized:
            log.info("ocr_bullet_glyphs_normalized_total",
                      f"Normalized {bullets_normalized} OCR bullet-glyph line(s) into list items across the document")

        # Page-region context for the axis/legend heading guard (spec Fix
        # 2): a bare unit/metric word ("Accuracy", "Milliseconds") is only
        # rejected as a heading when it sits near detected figure content --
        # the same word standing alone elsewhere (not near a figure) must
        # still be a valid heading.
        is_near_figure = _build_figure_context(pages_lines, pdf, page_count, progress=progress)

        sizes = [round(l.size, 1) for l in all_lines if l.size]
        body_size = Counter(sizes).most_common(1)[0][0] if sizes else 10.0
        distinct_heading_sizes = sorted({s for s in sizes if s > body_size * 1.08}, reverse=True)
        size_to_level = {s: min(i + 1, 6) for i, s in enumerate(distinct_heading_sizes)}

        # OCR heading promotion (spec "OCR heading promotion" / "Negative
        # heading evidence"): a real PDF's font-size bucket is trustworthy
        # typography evidence by itself. An OCR line's measured height is
        # real too (see `_convert_ocr_lines`), but noisier and much easier
        # for a figure label, axis tick, or caption to accidentally clear --
        # so for `from_ocr` lines, clearing the size bucket only makes a
        # line *eligible* for promotion; `_OcrHeadingScorer` still requires
        # multiple independent corroborating signals before it actually
        # becomes a HeadingBlock.
        ocr_scorer = _OcrHeadingScorer(all_lines, body_size)

        # H2a: generic junk-heading promotion rejections (PDF-native evidence).
        # Package-style qualified names (java.lang.NullPointerException):
        # lowercase first segment and at least two dots, so product names
        # like "Node.js", "ASP.NET" and "System.IO" are not caught.
        _FQCN_RE = re.compile(r'^[a-z_][a-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){2,}$')
        # Stack-trace / call-site endings ("Foo.bar(Foo.java:42)") or a run-in
        # label ending in ':'. A parenthetical after a space ("Number of
        # elements (n)", "Appendix B (Optional)") and generics ("List<T>")
        # are real headings.
        _STACK_TRACE_END_RE = re.compile(r'(?:\w\([^()]*\)|:)\s*$')
        _FULLWIDTH_DIGIT_RE = re.compile(r'^[０-９]')

        def _is_fullwidth_digit_leading(text: str) -> bool:
            s = text.strip()
            return bool(s and _FULLWIDTH_DIGIT_RE.match(s))

        def _reject_as_junk_heading(line: "_Line") -> bool:
            text = line.text.strip()
            if not text:
                return False
            if _FQCN_RE.match(text):
                return True
            if _STACK_TRACE_END_RE.search(text):
                return True
            if _is_fullwidth_digit_leading(text):
                return True
            # Monospace at body size is code; a monospace line set clearly
            # larger than body text is an API-reference style heading.
            if line.mono and not (body_size and line.size >= body_size * 1.2):
                return True
            # Letter density is measured on the title, excluding the outline
            # prefix and whitespace, so "3.1.4 Sets" or "10.2 SQL" pass.
            title = numbering_prefix_remainder(text)
            title = text if title is None else title
            chars = [c for c in title if not c.isspace()]
            if not chars:
                return False
            if sum(c.isalpha() for c in chars) / len(chars) < 0.5:
                return True
            return False

        def _is_heading_line(line: "_Line") -> bool:
            if line.forced_heading or line.is_resolved_title:
                return True
            if not is_heading_candidate(line.text):
                return False
            if _reject_as_junk_heading(line):
                return False
            if looks_like_axis_or_legend_label(line.text) and is_near_figure(line):
                return False
            if line.from_ocr:
                # Whitespace isolation is *necessary but not sufficient*
                # (spec: "Whitespace isolation alone is insufficient" -- not
                # "irrelevant"): OCR line height is real but too noisy to
                # gate on the way a native font-size bucket can (a bold
                # same-size heading routinely measures no taller than body
                # text), so isolation is the structural precondition that
                # keeps ordinary paragraph-continuation lines from ever
                # reaching the multi-signal scorer at all, and the scorer
                # requires additional corroborating signals beyond it.
                if not ocr_scorer.is_isolated(line):
                    return False
                return ocr_scorer.decide(line)
            size_key = round(line.size, 1)
            if size_key not in size_to_level:
                return False
            return True

        def _ocr_typed_block_kind(line: "_Line") -> Optional[str]:
            """For an OCR line that is not going to become a heading: if it
            structurally resembles a figure/diagram/table-cell/caption label,
            or it was heading-shaped but scored too weakly to promote, it
            must become a typed block instead of silently merging into
            surrounding paragraph prose (spec "Typed block handling" /
            "Do not concatenate unrelated labels into one prose paragraph
            merely to eliminate errors"). Scoped to isolated lines only --
            a line that is clearly running body text (no whitespace
            isolation around it) is left to the ordinary paragraph
            assembly below."""
            if not line.from_ocr or not ocr_scorer.is_isolated(line):
                return None
            text = line.text.strip()
            if not text:
                return None
            kind = classify_ocr_nonheading_kind(text)
            if kind != "unknown":
                return kind
            if is_heading_candidate(text) and not ocr_scorer.info(line).get("accept", False):
                return "unknown"
            return None

        # Heading-level resolution (spec section 10): numbering depth ("1",
        # "3.1", "3.1.1") is normalized against the *shallowest* numbering
        # depth actually observed in this document (a chapter-only document
        # numbered "2.1", "2.2", ... has no top-level "2" heading of its own,
        # so "2.1" must still map to the level right under the title, not one
        # level deeper) -- otherwise a document like that would skip a level
        # entirely (H1 -> H3, no H2). The very first accepted heading in
        # reading order is the title (H1); every other heading either gets a
        # numbering-derived level or, absent numbering, defaults to H2 (a
        # top-level section) rather than being flattened to the same level
        # as every other heading in the document regardless of depth.
        #
        # A *numbered* heading is never promoted to the title, even if it's
        # the very first accepted heading candidate in the document (spec
        # section 11 explicitly penalizes "section headings mistakenly
        # located first" as a title candidate): a document consisting only
        # of numbered subsections with no separately-extractable chapter
        # title (e.g. a scanned textbook chapter whose real title is a
        # graphic/image, not OCR'able text) would otherwise have its first
        # subsection heading ("2.1 Searching and algorithms") consume the H1
        # slot, silently demoting the rest of the entire chapter (2.2-2.11)
        # into children of that one subsection. Leaving no H1 assigned here
        # is the better outcome in that case: normalize.py's
        # `_enforce_single_first_h1` synthesizes one from document metadata
        # or the filename instead, which is a more honest title than an
        # arbitrary subsection heading.
        _numbered_depths = [
            d for l in all_lines if _is_heading_line(l)
            for d in [numbering_depth(l.text)] if d is not None
        ]
        _min_raw_depth = min(_numbered_depths) if _numbered_depths else None
        _title_assigned = False
        _heading_candidates_seen = 0
        # A real title, when present as extractable text at all, is always
        # among the very first headings encountered -- if the first several
        # heading candidates are all numbered sections or otherwise
        # disqualified (spec section 11: no reliable title region found),
        # the actual title is more likely a graphic/image this pipeline
        # can't OCR than some topical heading buried in the body. Capping
        # candidacy to the first few headings keeps this consistent with
        # normalize.py's own `_enforce_single_first_h1`, which only moves an
        # out-of-place H1 to the front when it's within the first 3 blocks
        # and otherwise leaves it (and everything before it) alone -- so a
        # "title" assigned any later would just end up stranded mid-document
        # instead of fixed. No qualifying candidate here means no H1 is
        # assigned during extraction at all; normalize.py's fallback
        # (document metadata, then the source filename) produces the title
        # instead, which is more honest than an arbitrary body heading.
        _MAX_TITLE_CANDIDATE_POSITION = 3

        def _resolve_heading_level(text: str) -> int:
            nonlocal _title_assigned, _heading_candidates_seen
            depth = numbering_depth(text)
            # A front-matter-shaped line ("Abstract", a copyright notice, an
            # affiliation, ...) is a section label or metadata, never the
            # document title (spec section 3's `abstract_promoted_to_h1`
            # gate) -- excluded here regardless of position, independent of
            # whether a visual/metadata title was separately resolved above.
            # "author" is deliberately *not* in this blocklist: a short
            # Title Case phrase ("Report On Widget Performance") has the
            # exact same shape as a short author name list, so treating an
            # author-shaped classification as disqualifying would reject
            # legitimate titles far more often than it would ever catch a
            # real author line reaching this fallback (which only runs when
            # no metadata/visual title was already resolved, and an author
            # line still has to also pass `looks_like_title_candidate`).
            is_front_matter_shaped = (
                classify_front_matter_line(text) in _TITLE_BLOCKING_FRONT_MATTER_KINDS
                or text.strip().lower().rstrip(":.") == "abstract"
            )
            if (
                not _title_assigned and depth is None
                and _heading_candidates_seen < _MAX_TITLE_CANDIDATE_POSITION
                and looks_like_title_candidate(text)
                and not is_front_matter_shaped
            ):
                _title_assigned = True
                _heading_candidates_seen += 1
                return 1
            _heading_candidates_seen += 1
            if depth is not None and _min_raw_depth is not None:
                return max(2, min(depth - _min_raw_depth + 2, 6))
            return 2

        # Table zones per page, so we can skip lines that fall inside an extracted table.
        table_zones: dict = {}
        for pno in range(page_count):
            current_page = pno + 1
            # Same throttle as the extraction-pass progress reporter above
            # (every 25 pages and on the final page).
            if current_page % 25 == 0 or current_page == page_count:
                safe_progress(progress, "detecting_tables", current=current_page, total=page_count)
            zones = []
            for header, rows, bbox in _extract_tables(pdf[pno]):
                zones.append((bbox, header, rows))
            if zones:
                table_zones[pno + 1] = zones

        def in_table_zone(line: _Line) -> bool:
            for bbox, _, _ in table_zones.get(line.page, []):
                if bbox[1] - 2 <= line.y0 <= bbox[3] + 2:
                    return True
            return False

        # Compound-block decomposition (spec section 11): a run of consecutive
        # lines with no structural signal (not a heading by size, not a bullet,
        # not font-flagged monospace) may still be code if its *content* looks
        # like code -- reclassify by reusing the existing `mono` flag so the
        # main loop below picks it up as a code block with no further changes.
        def _classify_kind(line: _Line) -> str:
            if _is_heading_line(line):
                return "heading"
            if _BULLET_RE.match(line.text):
                return "bullet"
            if line.mono:
                return "mono_code"
            return "plain"

        reclassified = 0
        _run: List[_Line] = []

        def _flush_code_run():
            nonlocal reclassified
            if _run:
                for is_code, sub_lines in _segment_code_candidate_run(_run):
                    if is_code:
                        for l in sub_lines:
                            l.mono = True
                        reclassified += len(sub_lines)
                    # Non-code sub-segments stay mono=False and fall through
                    # to the ordinary paragraph-accumulation logic below --
                    # this is the "segment, don't engulf" requirement: a
                    # prose -> label -> prose run yields three blocks, not
                    # one merged code block spanning all of it.
            _run.clear()

        for line in all_lines:
            if in_table_zone(line) or _classify_kind(line) != "plain":
                _flush_code_run()
                continue
            _run.append(line)
        _flush_code_run()
        if reclassified:
            log.info("content_based_code_detection",
                      f"Reclassified {reclassified} line(s) as code based on content patterns "
                      f"(no monospace font was detected for them)")

        emitted_tables_for_page: set = set()

        paragraph_buf: List[str] = []
        list_buf: List[ListItemNode] = []
        list_ordered = False
        code_buf: List[str] = []
        prev_line: Optional[_Line] = None
        suppress_next_break = False
        hyphen_stats = HyphenationStats()

        # Front-matter grouping (spec sections 4-8): blocks classified out
        # of the body flow are collected here, in the order encountered,
        # and spliced in as one contiguous group right after the title once
        # extraction finishes -- never left at whatever incidental position
        # they were interleaved into the body reading order (spec section
        # 5's "collect all author columns before beginning abstract or body
        # processing"). Not yet threaded from `PipelineConfig` (see
        # `config.py`'s `ContentPolicyConfig`), so these mirror the spec's
        # own example defaults directly.
        _INCLUDE_COPYRIGHT_IN_MARKDOWN = False
        _INCLUDE_PUBLICATION_METADATA_IN_MARKDOWN = True
        _FRONT_MATTER_BLOCK_CTORS = {
            "author": AuthorBlock, "affiliation": AffiliationBlock, "contact": ContactBlock,
            "keywords": KeywordBlock, "copyright": CopyrightNoticeBlock,
            "permission": PermissionNoticeBlock, "conference": ConferenceMetadataBlock,
            "publication_metadata": PublicationMetadataBlock,
        }
        front_matter_buf: List = []
        excluded_front_matter: List[dict] = []

        def _front_matter_kind_for_line(fm_line: "_Line") -> Optional[str]:
            kind = classify_front_matter_line(fm_line.text)
            if kind is None:
                return None
            # Author/affiliation/contact/keywords front matter never occurs
            # past the first page of an academic-paper-style document; this
            # bounds false positives from ordinary body prose elsewhere in
            # the document that happens to match the same shape (an
            # organization name mentioned in a sentence, a short
            # capitalized phrase). Copyright/permission/conference
            # boilerplate legitimately recurs as a footer on every page.
            if kind in ("author", "affiliation", "contact", "keywords") and fm_line.page != 1:
                return None
            return kind

        def _handle_front_matter_line(fm_kind: str, fm_line: "_Line") -> None:
            excludable = fm_kind in ("copyright", "permission") and not _INCLUDE_COPYRIGHT_IN_MARKDOWN
            excludable = excludable or (
                fm_kind in ("conference", "publication_metadata")
                and not _INCLUDE_PUBLICATION_METADATA_IN_MARKDOWN
            )
            if excludable:
                excluded_front_matter.append({"kind": fm_kind, "text": fm_line.text, "page": fm_line.page})
                log.info(
                    f"{fm_kind}_notice_excluded",
                    f"Excluded {fm_kind} text from rendered Markdown per content policy "
                    f"(retained in document metadata): {fm_line.text!r}",
                    page=fm_line.page,
                )
                return
            ctor = _FRONT_MATTER_BLOCK_CTORS[fm_kind]
            block = ctor(block_id=new_id(), order=order, source_format="pdf",
                          text=fm_line.text, normalized=fm_line.text, page=fm_line.page)
            front_matter_buf.append(block)
            log.fixed(
                "front_matter_block_extracted",
                f"Extracted {fm_kind} line out of body flow and grouped it as front matter "
                f"(page {fm_line.page}): {fm_line.text!r}",
                block_id=block.block_id, page=fm_line.page,
            )

        def flush_paragraph():
            nonlocal paragraph_buf
            if paragraph_buf:
                text = " ".join(paragraph_buf)
                bid = new_id()
                text = repair_layout_hyphenation(text, hyphen_stats, block_id=bid)
                doc.add(ParagraphBlock(block_id=bid, order=order, source_format="pdf",
                                        text=text, normalized=text, page=prev_line.page if prev_line else None))
                paragraph_buf = []

        def flush_list():
            nonlocal list_buf
            if list_buf:
                bid = new_id()
                for item in list_buf:
                    item.text = repair_layout_hyphenation(item.text, hyphen_stats, block_id=bid)
                doc.add(ListBlock(block_id=bid, order=order, source_format="pdf",
                                   ordered=list_ordered, items=list_buf,
                                   page=prev_line.page if prev_line else None))
                list_buf = []

        def flush_code():
            nonlocal code_buf
            if code_buf:
                code = "\n".join(code_buf)
                language = "html" if _looks_like_html_run(code_buf) else ""
                doc.add(CodeBlock(block_id=new_id(), order=order, source_format="pdf",
                                   language=language, code=code, normalized=code,
                                   page=prev_line.page if prev_line else None))
                code_buf = []

        _TYPED_BLOCK_CTORS = {
            "caption": CaptionBlock,
            "figure_label": FigureLabelBlock,
            "diagram_annotation": DiagramAnnotationBlock,
            "table_cell": TableCellBlock,
            "unknown": UnknownBlock,
        }

        def emit_tables_for_page(pno: int):
            for bbox, header, rows in table_zones.get(pno, []):
                is_complex = any(len(r) != len(header) for r in rows)
                doc.add(TableBlock(block_id=new_id(), order=order, source_format="pdf",
                                    header=[str(c or "") for c in header],
                                    rows=[[str(c or "") for c in r] for r in rows],
                                    is_complex=is_complex, page=pno))
                if is_complex:
                    log.warning("complex_pdf_table", f"Table on page {pno} has ragged rows; flagged for manual review", page=pno)

        last_page_seen = 0
        for line in all_lines:
            if line.page != last_page_seen:
                if last_page_seen and last_page_seen not in emitted_tables_for_page and last_page_seen in table_zones:
                    pass
                last_page_seen = line.page

            if in_table_zone(line):
                if line.page not in emitted_tables_for_page:
                    flush_paragraph(); flush_list(); flush_code()
                    emit_tables_for_page(line.page)
                    emitted_tables_for_page.add(line.page)
                prev_line = line
                continue

            is_heading = _is_heading_line(line)
            bullet_m = _BULLET_RE.match(line.text)

            if is_heading:
                flush_paragraph(); flush_list(); flush_code()
                if line.is_resolved_title:
                    level = 1
                    _title_assigned = True
                    _heading_candidates_seen += 1
                else:
                    level = _resolve_heading_level(line.text)
                heading = HeadingBlock(block_id=new_id(), order=order, source_format="pdf",
                                        level=level, text=line.text, normalized=line.text,
                                        page=line.page)
                if line.from_ocr:
                    info = ocr_scorer.info(line)
                    heading.style["from_ocr"] = True
                    heading.style["ocr_heading_score"] = info.get("score", 0)
                    heading.style["ocr_heading_signals"] = info.get("positives", [])
                    heading.style["ocr_heading_negatives"] = info.get("negatives", [])
                if line.is_resolved_title and title_evidence is not None:
                    heading.style["title_resolution_source"] = title_evidence["source"]
                    title_evidence["source_block_ids"].append(heading.block_id)
                doc.add(heading)
            elif bullet_m:
                flush_paragraph(); flush_code()
                ordered = bool(re.match(r"\d", bullet_m.group(1)))
                if list_buf and list_ordered != ordered:
                    flush_list()
                list_ordered = ordered
                list_buf.append(ListItemNode(text=bullet_m.group(2).strip()))
            elif line.mono:
                flush_paragraph(); flush_list()
                code_buf.append(line.text)
            else:
                typed_kind = _ocr_typed_block_kind(line)
                if typed_kind is not None:
                    flush_paragraph(); flush_list(); flush_code()
                    typed_block = _TYPED_BLOCK_CTORS[typed_kind](
                        block_id=new_id(), order=order, source_format="pdf",
                        text=line.text, normalized=line.text, page=line.page,
                    )
                    typed_block.style["from_ocr"] = True
                    typed_block.style["ocr_reroute_kind"] = typed_kind
                    doc.add(typed_block)
                    prev_line = line
                    continue

                gap = (line.y0 - prev_line.y1) if (prev_line and prev_line.page == line.page) else 999
                # A large horizontal jump on the same page (e.g. the last line of a
                # left column immediately followed by the first line of the right
                # column after multi-column reordering) is not vertical whitespace,
                # so the gap check above wouldn't catch it on its own.
                x_jump = (
                    prev_line is not None and prev_line.page == line.page and
                    abs(line.x0 - prev_line.x0) > max(line.size, 10) * 5
                )
                new_para = (
                    prev_line is None or prev_line.page != line.page or x_jump or
                    gap > max(line.size, 10) * 0.9 or
                    (paragraph_buf and paragraph_buf[-1].endswith((".", "!", "?", ":")) and gap > line.size * 0.3)
                )

                # Front-matter reclassification (spec sections 4-7): a line
                # that matches an author/affiliation/contact/keywords/
                # copyright/permission/conference shape is pulled out of
                # the body flow entirely -- never appended to
                # `paragraph_buf`, never flushed as its own paragraph break
                # -- so the paragraph running before and after it rejoins
                # into one continuous block once this line is excised
                # (paragraph continuity, spec section 7). Grouped back in
                # at document-title position after the whole page loop
                # finishes (spec section 5/6).
                #
                # Gated on *not clearly being a mid-sentence line-wrap*
                # rather than on the ordinary (much stricter) new-paragraph
                # gap threshold: real front-matter blocks (author/
                # affiliation/email stacked lines) are typically
                # single-spaced, tighter than the gap `new_para` requires,
                # so gating on that would miss exactly the tightly-packed
                # case this exists to catch. An unfinished sentence
                # (buffered text with no terminal punctuation yet, same
                # page, no horizontal jump) is the one case still excluded,
                # so a genuine paragraph line-wrap is never misread as
                # front matter mid-sentence.
                mid_sentence_wrap = (
                    bool(paragraph_buf) and not paragraph_buf[-1].endswith((".", "!", "?", ":"))
                    and not x_jump and prev_line is not None and prev_line.page == line.page
                )
                fm_kind = None
                if not mid_sentence_wrap:
                    fm_kind = _front_matter_kind_for_line(line)
                if fm_kind is not None:
                    _handle_front_matter_line(fm_kind, line)
                    suppress_next_break = True
                    prev_line = line
                    continue

                flush_list(); flush_code()
                if suppress_next_break:
                    new_para = False
                    suppress_next_break = False
                if new_para:
                    flush_paragraph()
                paragraph_buf.append(line.text)

            prev_line = line

        flush_paragraph(); flush_list(); flush_code()

        if front_matter_buf:
            title_idx = next(
                (i for i, b in enumerate(doc.blocks) if isinstance(b, HeadingBlock) and b.level == 1), None,
            )
            insert_at = (title_idx + 1) if title_idx is not None else 0
            doc.blocks[insert_at:insert_at] = front_matter_buf
            for j, b in enumerate(doc.blocks):
                b.order = j
            log.fixed(
                "front_matter_grouped",
                f"Grouped {len(front_matter_buf)} front-matter block(s) (author/affiliation/contact/"
                f"keywords) together, immediately after the document title",
            )
        if excluded_front_matter:
            doc.metadata.extra["excluded_front_matter"] = excluded_front_matter

        doc.metadata.extra["ocr_heading_candidates"] = ocr_scorer.candidates
        doc.metadata.extra["ocr_headings_accepted"] = ocr_scorer.accepted
        doc.metadata.extra["ocr_headings_rejected"] = ocr_scorer.rejected

        if title_evidence is not None:
            doc.metadata.extra["title_resolution"] = title_evidence

        doc.metadata.extra["layout_hyphenation"] = hyphen_stats.to_dict()
        for entry in hyphen_stats.repairs_log:
            if entry["action"] == "repaired":
                log.fixed("layout_hyphenation_repaired",
                          f"Repaired layout-wrap hyphenation {entry['original']!r} -> {entry['result']!r}",
                          block_id=entry["block_id"] or None)
            elif entry["action"] == "unresolved":
                log.warning("unresolved_layout_hyphenation",
                            f"Could not confidently resolve possible layout hyphenation {entry['original']!r} "
                            f"({entry['reason']}); left as {entry['result']!r}",
                            block_id=entry["block_id"] or None,
                            suggested_action="Verify manually whether this is a wrapped word or a real compound.")

        if not doc.blocks:
            log.error("empty_pdf_extraction", "No text could be extracted from this PDF (even after OCR policy was applied)")

        pdf.close()
        return doc, log

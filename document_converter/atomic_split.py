"""
Token-aware subdivision of oversized atomic blocks.

Corrective rule (replaces the old absolute "never split code internally"):

    Never split code, pseudocode, equations, tables, or other atomic blocks
    arbitrarily, mid-line, or without preserving structural provenance.

    When an atomic block exceeds the embedding model's real token limit,
    subdivide it at safe semantic boundaries.

This module is the one canonical implementation `chunker.py` calls from both
the preview/process path (approximate tokenizer, no embedding service) and
the index path (the live embedding service's real tokenizer) -- there is no
separate preview-only splitting logic, so a document's chunk boundaries never
drift between what a user previewed and what actually got embedded.

Splitting always prefers the highest-value semantic boundary available for
the block's type and only forces a mid-line/mid-word split as an absolute
last resort, when a single line/row/item alone exceeds the budget -- and even
then never mid-token (verified against the same tokenizer used for the
accept/reject decision), always flagged `forced_split=True` with full
reconstruction metadata so no content is silently lost, truncated, or
duplicated outside its recorded overlap.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .model import Block, CodeBlock, ListBlock, ListItemNode, TableBlock
from .render import render_block, render_code, render_table

logger = logging.getLogger(__name__)

Tokenizer = Callable[[str], int]

# Small fixed budget reserved for a part's own context header/prefix and
# markdown serialization overhead (fence lines, escaping) on top of whatever
# the caller already reserved for the chunk-level provenance prefix and
# section heading -- keeps every part comfortably inside the real limit
# instead of relying on a single shared estimate for two different things.
DEFAULT_PROVENANCE_TOKEN_RESERVE = 40
DEFAULT_OVERLAP_TOKENS = 30
MAX_CODE_OVERLAP_LINES = 5


@dataclass
class AtomicBlockPart:
    """One token-safe fragment of an oversized atomic block."""

    block: Block
    part_index: int  # 1-based
    part_count: int
    # Reconstruction-unit range, 1-based and inclusive. For line-packed
    # blocks a unit is one of the block's own rendered lines; for an
    # exploded forced split (several parts from ONE source line, or several
    # fragments from one oversized table row) each part earns its own unit,
    # so verify_reconstruction's "every unit covered exactly once, in order"
    # invariant holds without non-overlap duplication.
    line_start: int
    line_end: int
    overlap_line_start: Optional[int]
    overlap_line_end: Optional[int]
    forced_split: bool
    context_header: str
    provenance: Dict[str, Any]


class AtomicSplitError(RuntimeError):
    """Raised only when a part genuinely cannot be brought under budget
    (e.g. max_tokens too small to hold even a single character). The caller
    (`chunker.py`) drops the offending block from the index with a WARNING
    -- the document itself still succeeds; one unsplittable block is never
    a whole-document failure."""


def _default_tokenizer(text: str) -> int:
    return int(round(len(text.split()) * 1.3))


# --------------------------------------------------------------------------- #
# generic line-level packing shared by code/generic-text splitting


def _line_tokens(lines: List[str], tokenizer: Tokenizer) -> List[int]:
    return [tokenizer(line) if line.strip() else 1 for line in lines]


_FUNC_BOUNDARY_RE = re.compile(
    r"^\s*(?:\d+[:.]?\s*)?(algorithm\s+\d+\b|end\s+function\b|end\s+procedure\b|function\b|procedure\b)",
    re.IGNORECASE,
)
_STEP_RE = re.compile(r"^\s*\d+[:.]\s")
_CONTROL_RE = re.compile(
    r"^\s*(?:\d+[:.]?\s*)?(end\s+if\b|end\s+for\b|end\s+while\b|else\s+if\b|if\b|for\b|while\b|else\b)",
    re.IGNORECASE,
)
_CONTINUATION_SUFFIX_RE = re.compile(r"[-,+*/=(&|]\s*$")


def _code_line_priority(lines: List[str], i: int) -> int:
    """Priority of the boundary immediately BEFORE line index `i` (i.e. a
    break between line i-1 and line i). Higher is more preferred. Mirrors
    the required preference order for code/pseudocode: function/procedure
    boundary > algorithm step > blank line > control-flow > complete
    statement > any line (always allowed, the unconditional fallback)."""
    if i <= 0 or i >= len(lines):
        return -1
    prev, cur = lines[i - 1], lines[i]
    if _FUNC_BOUNDARY_RE.match(cur):
        return 5
    if _STEP_RE.match(cur):
        return 4
    if prev.strip() == "" or cur.strip() == "":
        return 3
    if _CONTROL_RE.match(cur):
        return 2
    if not _CONTINUATION_SUFFIX_RE.search(prev):
        return 1
    return 0


def _forced_split_line(line: str, budget: int, tokenizer: Tokenizer) -> List[str]:
    """Split a single line that alone exceeds `budget` into token-safe
    pieces, breaking on whitespace whenever possible and only falling back
    to a character-level cut (verified against the tokenizer, never mid
    -token in practice for real text) for a single pathologically long
    "word" with no internal whitespace."""
    words = line.split(" ")
    pieces: List[str] = []
    current: List[str] = []

    def flush():
        if current:
            pieces.append(" ".join(current))
            current.clear()

    for w in words:
        candidate = current + [w]
        if current and tokenizer(" ".join(candidate)) > budget:
            flush()
            candidate = [w]
        if tokenizer(" ".join(candidate)) > budget and len(candidate) == 1:
            # A single "word" alone is too big -- cut it by characters.
            flush()
            word = w
            while word:
                lo, hi = 1, len(word)
                best = 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if tokenizer(word[:mid]) <= budget:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                pieces.append(word[:best])
                word = word[best:]
            continue
        current.append(w)
    flush()
    return pieces or [line]


def _pack_lines_with_boundaries(
    lines: List[str], budget: int, tokenizer: Tokenizer,
    priority_fn: Callable[[List[str], int], int],
) -> List[Dict[str, Any]]:
    """Greedily fill each part up to `budget` tokens, then retroactively pull
    the break point back to the highest-priority boundary available within
    the filled window (never below the point that keeps every part
    non-empty). A single line that alone exceeds budget is forced-split.
    Returns a list of {start, end, forced, pieces} dicts, `end` exclusive."""
    token_of_line = _line_tokens(lines, tokenizer)
    n = len(lines)
    parts: List[Dict[str, Any]] = []
    i = 0
    while i < n:
        if token_of_line[i] > budget:
            pieces = _forced_split_line(lines[i], budget, tokenizer)
            parts.append({"start": i, "end": i + 1, "forced": True, "pieces": pieces})
            i += 1
            continue
        j = i
        cur = 0
        while j < n and token_of_line[j] <= budget and cur + token_of_line[j] <= budget:
            cur += token_of_line[j]
            j += 1
        end = j
        best_end, best_pri = end, -1
        for k in range(i + 1, end + 1):
            pri = priority_fn(lines, k) if k < n else 6
            if pri >= best_pri:
                best_pri, best_end = pri, k
        end = best_end if best_end > i else end
        parts.append({"start": i, "end": end, "forced": False, "pieces": None})
        i = end
    return parts


def _detect_parent_procedure(lines: List[str], start_idx: int) -> Optional[str]:
    """Scans backward from `start_idx` for the nearest function/procedure/
    algorithm declaration line, to give a split part's context header a
    real anchor (spec section 5's "Parent procedure: ..." line)."""
    name_re = re.compile(r"([A-Z][A-Z0-9_]{2,})\s*\(", re.IGNORECASE)
    for i in range(min(start_idx, len(lines) - 1), -1, -1):
        line = lines[i]
        if _FUNC_BOUNDARY_RE.match(line):
            m = name_re.search(line)
            if m:
                return m.group(1)
            return line.strip()[:60]
    return None


# --------------------------------------------------------------------------- #
# reconstruction verification


def verify_reconstruction(original_lines: List[str], parts: List[AtomicBlockPart]) -> Dict[str, Any]:
    """Checks the four invariants required for a valid subdivision: every
    original line is represented by exactly one part's *own* (non-overlap)
    range, in order, with no gaps and no non-overlap duplication."""
    covered: List[int] = []
    for p in sorted(parts, key=lambda p: p.part_index):
        covered.extend(range(p.line_start, p.line_end + 1))
    expected = list(range(1, len(original_lines) + 1))
    missing = sorted(set(expected) - set(covered))
    seen = set()
    duplicated = []
    for i in covered:
        if i in seen:
            duplicated.append(i)
        seen.add(i)
    return {
        "all_parent_content_represented": covered == expected,
        "missing_lines": missing,
        "duplicated_non_overlap_lines": duplicated,
        "reconstructs_to_original": covered == expected and not duplicated,
    }


# --------------------------------------------------------------------------- #
# per-type splitters


def _split_code_block(
    block: CodeBlock, tokenizer: Tokenizer, content_budget: int, overlap_tokens: int, max_tokens: int,
) -> Tuple[List[AtomicBlockPart], List[str]]:
    lines = block.code.split("\n")
    raw_parts = _pack_lines_with_boundaries(lines, content_budget, tokenizer, _code_line_priority)

    # A forced-split line yields pieces that each fit `content_budget`
    # alone, but the whole set almost never does -- emitting every piece as
    # one part just moves the overflow to the final verification gate (the
    # long-known-failing single-overlong-line case: one 100-word line, a
    # 30-token budget, and a single 113-token "part"). Regroup the pieces
    # into as many budget-fitting parts as needed instead; each piece-group
    # earns its own reconstruction unit below (one unit per part, exactly
    # like an exploded oversized table-row fragment), so
    # verify_reconstruction still sees every unit covered exactly once.
    expanded: List[Dict[str, Any]] = []
    for rp in raw_parts:
        if not rp["forced"]:
            expanded.append(rp)
            continue
        cur: List[str] = []
        cur_tokens = 0
        for piece in rp["pieces"]:
            piece_tokens = tokenizer(piece)
            if cur and cur_tokens + piece_tokens > content_budget:
                expanded.append({"start": rp["start"], "end": rp["end"], "forced": True, "pieces": cur})
                cur, cur_tokens = [], 0
            cur.append(piece)
            cur_tokens += piece_tokens
        if cur:
            expanded.append({"start": rp["start"], "end": rp["end"], "forced": True, "pieces": cur})

    result: List[AtomicBlockPart] = []
    unit_lines: List[str] = []
    part_count = len(expanded)
    unit = 0  # running reconstruction-unit counter (see unit assignment below)
    prev_part: Optional[AtomicBlockPart] = None
    prev_own_end = None  # exclusive index into `lines` of the previous part's own content
    for idx, rp in enumerate(expanded, start=1):
        start, end, forced = rp["start"], rp["end"], rp["forced"]
        own_lines = rp["pieces"] if forced else lines[start:end]

        overlap_lines: List[str] = []
        overlap_line_start = overlap_line_end = None
        if prev_part is not None and not forced:
            k = 0
            while (
                k < MAX_CODE_OVERLAP_LINES
                and k < (prev_part.line_end - prev_part.line_start + 1)
                and prev_own_end - k - 1 >= 0
                and tokenizer("\n".join(lines[prev_own_end - k - 1: prev_own_end])) <= overlap_tokens
            ):
                k += 1
            if k > 0:
                overlap_lines = lines[prev_own_end - k: prev_own_end]
                # Recorded in unit space (the trailing units of the previous
                # part) so the metadata stays consistent when an exploded
                # forced line shifted unit indices off real line indices.
                overlap_line_start, overlap_line_end = prev_part.line_end - k + 1, prev_part.line_end

        parent_proc = _detect_parent_procedure(lines, start)

        def _build(overlap_lines, overlap_line_start, overlap_line_end):
            header_lines = [f"[Part {idx} of {part_count} of code block {block.block_id}]"]
            if parent_proc:
                header_lines.append(f"Parent procedure: {parent_proc}")
            if overlap_line_start:
                header_lines.append(f"Previous context: lines {overlap_line_start}-{overlap_line_end}")
            header_lines.append(f"Current lines: {start + 1}-{end}")
            context_header = "\n".join(header_lines)
            part_code = "\n".join(overlap_lines + own_lines)
            pb = CodeBlock(
                block_id=f"{block.block_id}-part-{idx}", order=block.order, source_format=block.source_format,
                page=block.page, language=block.language, code=part_code, parent_id=block.block_id,
            )
            pb.style = dict(block.style)
            pb.style["atomic_split_context_header"] = context_header
            return pb, context_header

        part_block, context_header = _build(overlap_lines, overlap_line_start, overlap_line_end)
        # Real per-part budget check on what will actually be embedded
        # (header + fenced content); drop overlap first if it doesn't fit.
        if tokenizer(context_header + "\n\n" + render_code(part_block)) > max_tokens and overlap_lines:
            overlap_line_start = overlap_line_end = None
            part_block, context_header = _build([], None, None)

        # Unit assignment: a normal part's units ARE its real source lines
        # (so when nothing exploded, line_start/line_end are exactly the
        # historical real-line indices); a forced piece-group gets one unit
        # of its own, since its pieces all come from the same single source
        # line and several parts may share that line.
        if forced:
            unit += 1
            line_start = line_end = unit
            unit_lines.append("\n".join(own_lines))
        else:
            line_start = unit + 1
            unit += (end - start)
            line_end = unit
            unit_lines.extend(lines[start:end])

        result.append(AtomicBlockPart(
            block=part_block, part_index=idx, part_count=part_count,
            line_start=line_start, line_end=line_end, overlap_line_start=overlap_line_start,
            overlap_line_end=overlap_line_end, forced_split=forced, context_header=context_header,
            provenance={},
        ))
        prev_part = result[-1]
        prev_own_end = end
    return result, unit_lines


def _split_list_block(
    block: ListBlock, tokenizer: Tokenizer, content_budget: int, overlap_tokens: int, max_tokens: int,
) -> Tuple[List[AtomicBlockPart], List[str]]:
    items = block.items
    item_lines = [_render_single_item(it, block.ordered, i) for i, it in enumerate(items)]
    item_tokens = [tokenizer(t) for t in item_lines]

    groups: List[Tuple[int, int]] = []
    i = 0
    n = len(items)
    while i < n:
        if item_tokens[i] > content_budget:
            # A single item's own text exceeds budget: forced-split its text.
            groups.append((i, i + 1))
            i += 1
            continue
        j, cur = i, 0
        while j < n and item_tokens[j] <= content_budget and cur + item_tokens[j] <= content_budget:
            cur += item_tokens[j]
            j += 1
        groups.append((i, j))
        i = j

    part_count = len(groups)
    result: List[AtomicBlockPart] = []
    prev_end = None
    for idx, (start, end) in enumerate(groups, start=1):
        forced = item_tokens[start] > content_budget and end == start + 1
        sub_items: List[ListItemNode]
        if forced:
            pieces = _forced_split_line(items[start].text, content_budget, tokenizer)
            sub_items = [ListItemNode(text=p, checked=items[start].checked) for p in pieces]
        else:
            sub_items = list(items[start:end])

        overlap_start = overlap_end = None
        overlap_items: List[ListItemNode] = []
        if prev_end is not None and not forced and prev_end - 1 >= 0:
            overlap_items = [items[prev_end - 1]]
            overlap_start = overlap_end = prev_end

        start_number = start + 1
        header = (
            f"[Part {idx} of {part_count} of list block {block.block_id}"
            f"{', continuing from item ' + str(start_number) if block.ordered else ''}]"
        )

        def _build_list(overlap_items, overlap_start, overlap_end):
            pb = ListBlock(
                block_id=f"{block.block_id}-part-{idx}", order=block.order, source_format=block.source_format,
                page=block.page, ordered=block.ordered, items=overlap_items + sub_items, parent_id=block.block_id,
            )
            pb.style = dict(block.style)
            pb.style["atomic_split_context_header"] = header
            return pb

        part_block = _build_list(overlap_items, overlap_start, overlap_end)
        if tokenizer(header + "\n\n" + render_block(part_block)) > max_tokens and overlap_items:
            overlap_start = overlap_end = None
            part_block = _build_list([], None, None)

        result.append(AtomicBlockPart(
            block=part_block, part_index=idx, part_count=part_count,
            line_start=start + 1, line_end=end, overlap_line_start=overlap_start,
            overlap_line_end=overlap_end, forced_split=forced, context_header=header, provenance={},
        ))
        prev_end = end
    return result, item_lines


def _render_single_item(item: ListItemNode, ordered: bool, idx: int) -> str:
    marker = f"{idx + 1}." if ordered else "-"
    return f"{marker} {item.text}"


def _render_table_text(header: List[str], rows: List[List[str]]) -> str:
    """The exact Markdown a candidate table part would render as -- built
    via the real `render_table()` code path (same cell escaping, separator
    row, and leading/trailing pipe padding a part actually ships with).
    Used only to MEASURE candidate parts while packing/subdividing; the
    parts themselves are always built from the parsed row/column model
    directly, never string-split back out of this rendered text."""
    tmp = TableBlock(block_id="", order=0, source_format="", header=list(header), rows=[list(r) for r in rows])
    return render_table(tmp)


def _pad_row(row: List[str], width: int) -> List[str]:
    row = list(row) + [""] * (width - len(row))
    return row[:width] if width else row


def _split_cell_text_into_rows(
    key_header: Optional[str], key_cell: Optional[str], col_header: str, cell_text: str,
    max_tokens: int, tokenizer: Tokenizer, table_id: str, row_label: str,
    header_for: Callable[[str], str],
) -> List[Dict[str, Any]]:
    """Ladder rule (b): a single cell -- paired with just the row's key cell,
    if any -- alone exceeds budget. Its content is plain text at this point,
    so it goes through the same structural text splitter code/generic
    blocks already use (`_forced_split_line`), spread one piece per
    continuation row with the key cell repeated on every one so each
    fragment is still self-describing on its own (rule (a)'s contract).

    Every measurement is against THIS fragment's own worst-case context
    header (`header_for(columns_desc)` -- the real descriptor for this one
    column, with only the "Part 99 of 99" and "(piece 99 of 99)" digit
    counts bounded from above, since the real part/piece counts are unknown
    until the whole ladder has run). The piece-count suffix bound is always
    reserved even though a single-piece result omits it -- over-reserving
    is the safe direction (the real, shorter header can only leave more
    room, and the final per-part verification gate re-checks the real
    render regardless).

    Rule (c): raises `AtomicSplitError`, naming the row, when even a single
    indivisible piece still can't fit once the key-cell/column-header
    framing itself is accounted for.
    """
    frame_header = [key_header, col_header] if key_header else [col_header]
    frame_row = [key_cell, ""] if key_header else [""]
    frame_render = _render_table_text(frame_header, [frame_row])
    desc_bound = f"{col_header} (piece 99 of 99)"
    avail = max_tokens - tokenizer(header_for(desc_bound) + "\n\n" + frame_render)
    if avail <= 0:
        raise AtomicSplitError(
            f"Table {table_id} {row_label} cannot be subdivided: even an empty cell for "
            f"column {col_header!r}" + (f" (key {key_cell!r})" if key_header else "") +
            f" exceeds the available budget ({max_tokens} tokens)"
        )
    pieces = _forced_split_line(cell_text, avail, tokenizer)
    count = len(pieces)
    out: List[Dict[str, Any]] = []
    for n, piece in enumerate(pieces, start=1):
        row = [key_cell, piece] if key_header else [piece]
        desc = col_header + (f" (piece {n} of {count})" if count > 1 else "")
        out.append({"header": frame_header, "row": row, "columns": desc})
    return out


def _split_oversized_row(
    header: List[str], row: List[str], max_tokens: int, tokenizer: Tokenizer,
    table_id: str, row_number: int, total_rows: int,
    header_for: Callable[[str], str],
) -> List[Dict[str, Any]]:
    """Escalation ladder for one table row that alone -- with the full
    header -- exceeds `max_tokens`. Purely structural, format-agnostic, no
    document-type detection, applied identically to every table:

    (a) Cell-aware column-group subdivision: partition the row's non-key
        columns into contiguous groups, greedily, each rendered with the
        row's key cell (column 0, by structural convention -- the value a
        reader would use to recognize the row out of context) repeated so
        every fragment stands alone.
    (b) A single column that still doesn't fit alongside just the key cell
        falls to `_split_cell_text_into_rows` -- text-splitting that one
        cell's content across rows.
    (c) A row that can't even be identified by its key cell alone -- or a
        single indivisible text-split piece that still doesn't fit -- raises
        `AtomicSplitError` naming the row. Content is never silently
        dropped or truncated to make it fit.

    `max_tokens` is the REAL per-part budget, and every candidate fragment
    is measured together with its OWN context header via
    `header_for(columns_desc)` -- the fragment's actual columns descriptor
    (a single-column fragment reserves room for "columns col3", not a
    worst-case "columns col1-...-colN" string that a table with
    sentence-length column names would blow the entire budget on before a
    single cell was even considered). `header_for` keeps only the
    "Part 99 of 99" digit-count bound, since part_count is unknown until
    every row's ladder has run.

    Returns an ordered list of {"header", "row", "columns"} fragment specs.
    Every fragment's cells come straight from the parsed `row`/`header`
    lists -- never re-derived by string-splitting rendered Markdown.
    """
    row_label = f"row {row_number} of {total_rows}"
    width = len(header)
    row = _pad_row(row, width)

    def fits(rendered: str, columns_desc: str) -> bool:
        # Exactly what the final per-part verification gate re-checks on the
        # real render: this fragment's own context header + serialized table.
        return tokenizer(header_for(columns_desc) + "\n\n" + rendered) <= max_tokens

    if width == 0:
        cell = row[0] if row else ""
        return _split_cell_text_into_rows(
            None, None, "value", cell, max_tokens, tokenizer, table_id, row_label, header_for,
        )

    key_header, key_cell = header[0], row[0]
    other_headers, other_cells = header[1:], row[1:]

    key_only = _render_table_text([key_header], [[key_cell]])
    if not fits(key_only, key_header):
        raise AtomicSplitError(
            f"Table {table_id} {row_label} cannot be subdivided: its key cell "
            f"({key_header}={key_cell!r}) alone exceeds the available budget ({max_tokens} tokens)"
        )

    if not other_headers:
        return _split_cell_text_into_rows(
            key_header, key_cell, key_header, key_cell, max_tokens, tokenizer, table_id, row_label, header_for,
        )

    fragments: List[Dict[str, Any]] = []
    i, m = 0, len(other_headers)
    while i < m:
        single = _render_table_text([key_header, other_headers[i]], [[key_cell, other_cells[i]]])
        if not fits(single, other_headers[i]):
            fragments.extend(_split_cell_text_into_rows(
                key_header, key_cell, other_headers[i], other_cells[i],
                max_tokens, tokenizer, table_id, row_label, header_for,
            ))
            i += 1
            continue
        j = i + 1
        while j < m:
            trial = _render_table_text(
                [key_header] + other_headers[i:j + 1], [[key_cell] + other_cells[i:j + 1]],
            )
            if not fits(trial, "-".join(other_headers[i:j + 1])):
                break
            j += 1
        group_headers = other_headers[i:j]
        fragments.append({
            "header": [key_header] + group_headers,
            "row": [key_cell] + other_cells[i:j],
            "columns": "-".join(group_headers),
        })
        i = j
    return fragments


def _split_table_block(
    block: TableBlock, tokenizer: Tokenizer, content_budget: int, overlap_tokens: int,
    max_tokens: int, oversized_row_policy: str = "split",
) -> Tuple[List[AtomicBlockPart], List[str]]:
    if oversized_row_policy not in ("split", "error"):
        raise ValueError(f"Unknown oversized_row_policy {oversized_row_policy!r} (expected 'split' or 'error')")

    header = block.header
    rows = block.rows
    n = len(rows)

    # Cached once per table -- the real rendered cost of the header plus
    # separator row alone, via the same render_table() path every candidate
    # part is measured against below. Used only as a cheap floor (if even
    # the bare header can't fit, no row ever will); the authoritative
    # accept/reject decision always re-renders and re-tokenizes the WHOLE
    # candidate part fresh on commit -- never adds this cached number to
    # independently-tokenized row counts.
    header_only_tokens = tokenizer(_render_table_text(header, []))

    groups: List[Tuple[int, int, bool]] = []  # (start, end, oversized_single_row)
    i = 0
    while i < n:
        if header_only_tokens > content_budget:
            groups.append((i, i + 1, True))
            i += 1
            continue
        single_text = _render_table_text(header, [rows[i]])
        if tokenizer(single_text) > content_budget:
            groups.append((i, i + 1, True))
            i += 1
            continue
        j = i + 1
        while j < n:
            # Re-measure the WHOLE candidate part -- header, separator row,
            # and every escaped/padded row packed so far plus the next one
            # -- as one serialized string, exactly as it will actually be
            # rendered and persisted. Never sum independently-tokenized rows
            # (the bug this replaces): token counts are not strictly
            # additive under a real subword tokenizer.
            candidate_text = _render_table_text(header, rows[i:j + 1])
            if tokenizer(candidate_text) > content_budget:
                break
            j += 1
        groups.append((i, j, False))
        i = j

    # Pass 1: resolve every group into its final fragment list before any
    # part_index/part_count is assigned -- both depend on the table-wide
    # total, which isn't known until an oversized row's ladder has actually
    # run (a normal multi-row group is always exactly one fragment; an
    # oversized single row may explode into several).
    resolved: List[Dict[str, Any]] = []
    for start, end, oversized in groups:
        if not oversized:
            resolved.append({"kind": "rows", "start": start, "end": end})
            continue
        if oversized_row_policy == "error":
            row_tokens = tokenizer(_render_table_text(header, [rows[start]]))
            raise AtomicSplitError(
                f"Table {block.block_id} row {start + 1} of {n} ({row_tokens} tokens) exceeds the "
                f"effective budget ({content_budget} tokens) and oversized_row_policy is 'error'"
            )
        # Per-fragment budget accounting: the ladder measures every
        # candidate fragment against the real `max_tokens` together with
        # that fragment's OWN context header -- its actual columns
        # descriptor (a single-column fragment reserves room for
        # "columns col3", not "columns col1-...-colN"), with only the
        # "Part 99 of 99" digit count bounded from above since part_count
        # isn't known until every row's ladder has run. (The old shared
        # worst-case all-columns reserve collapsed to a 1-token budget for
        # tables with sentence-length column names -- Azure docs are the
        # observed case -- and spuriously tripped ladder rule (c), failing
        # the whole document.) The final per-part verification gate still
        # re-checks every real render, so any residual misestimate fails
        # loud rather than shipping an oversized part.
        def header_for(columns_desc: str, row_start: int = start) -> str:
            return (
                f"[Part 99 of 99 of table block {block.block_id} "
                f"(row {row_start + 1} of {n}, columns {columns_desc}, forced split)]"
            )
        fragments = _split_oversized_row(
            header, rows[start], max_tokens, tokenizer, block.block_id, start + 1, n, header_for,
        )
        resolved.append({"kind": "fragments", "start": start, "end": end, "fragments": fragments})

    part_count = sum(1 if r["kind"] == "rows" else len(r["fragments"]) for r in resolved)

    # Pass 2: build the real parts now that part_count is known, with a
    # single running "row unit" counter for line_start/line_end -- one unit
    # per original row for a normal group (unchanged from before), or one
    # unit per fragment for an oversized row that got exploded (each
    # fragment carries different cells from that one row, so each earns its
    # own unit -- verify_reconstruction only cares that every unit is
    # covered exactly once, in order; it never inspects row_texts content).
    result: List[AtomicBlockPart] = []
    row_texts: List[str] = []
    idx = 0
    unit = 0
    for r in resolved:
        if r["kind"] == "rows":
            start, end = r["start"], r["end"]
            idx += 1
            unit_start = unit + 1
            unit += (end - start)
            sub_rows = [list(row) for row in rows[start:end]]
            part_block = TableBlock(
                block_id=f"{block.block_id}-part-{idx}", order=block.order, source_format=block.source_format,
                page=block.page, header=list(header), rows=sub_rows, is_complex=block.is_complex,
                parent_id=block.block_id,
            )
            part_block.style = dict(block.style)
            header_note = "repeated header" if header else "no header"
            context_header = f"[Part {idx} of {part_count} of table block {block.block_id} ({header_note})]"
            part_block.style["atomic_split_context_header"] = context_header
            result.append(AtomicBlockPart(
                block=part_block, part_index=idx, part_count=part_count,
                line_start=unit_start, line_end=unit, overlap_line_start=None, overlap_line_end=None,
                forced_split=False, context_header=context_header, provenance={},
            ))
            row_texts.extend(" | ".join(row) for row in rows[start:end])
        else:
            start = r["start"]
            for frag in r["fragments"]:
                idx += 1
                unit += 1
                part_block = TableBlock(
                    block_id=f"{block.block_id}-part-{idx}", order=block.order, source_format=block.source_format,
                    page=block.page, header=list(frag["header"]), rows=[list(frag["row"])],
                    is_complex=block.is_complex, parent_id=block.block_id,
                )
                part_block.style = dict(block.style)
                context_header = (
                    f"[Part {idx} of {part_count} of table block {block.block_id} "
                    f"(row {start + 1} of {n}, columns {frag['columns']}, forced split)]"
                )
                part_block.style["atomic_split_context_header"] = context_header
                result.append(AtomicBlockPart(
                    block=part_block, part_index=idx, part_count=part_count,
                    line_start=unit, line_end=unit, overlap_line_start=None, overlap_line_end=None,
                    forced_split=True, context_header=context_header, provenance={},
                ))
                row_texts.append(" | ".join(frag["row"]))

    return result, row_texts


def _split_generic_block(
    block: Block, tokenizer: Tokenizer, content_budget: int, overlap_tokens: int, max_tokens: int,
) -> Tuple[List[AtomicBlockPart], List[str]]:
    """Fallback for any other atomic block type (figure annotations, table
    cells, captions, image alt text, ...): split its rendered text by line
    if it has real newlines, else by sentence, packing greedily and forcing
    a word-level split only when a single line/sentence alone is oversized."""
    text = render_block(block)
    lines = text.split("\n") if "\n" in text else _split_sentences_generic(text)
    raw_parts = _pack_lines_with_boundaries(lines, content_budget, tokenizer, _generic_line_priority)

    # Same forced-piece regrouping as `_split_code_block`: pieces of one
    # forced-split line/sentence each fit `content_budget` alone but rarely
    # do together, so they become as many budget-fitting parts as needed,
    # each with its own reconstruction unit.
    expanded: List[Dict[str, Any]] = []
    for rp in raw_parts:
        if not rp["forced"]:
            expanded.append(rp)
            continue
        cur: List[str] = []
        cur_tokens = 0
        for piece in rp["pieces"]:
            piece_tokens = tokenizer(piece)
            if cur and cur_tokens + piece_tokens > content_budget:
                expanded.append({"start": rp["start"], "end": rp["end"], "forced": True, "pieces": cur})
                cur, cur_tokens = [], 0
            cur.append(piece)
            cur_tokens += piece_tokens
        if cur:
            expanded.append({"start": rp["start"], "end": rp["end"], "forced": True, "pieces": cur})

    part_count = len(expanded)
    result: List[AtomicBlockPart] = []
    unit_lines: List[str] = []
    unit = 0
    prev_part: Optional[AtomicBlockPart] = None
    prev_own_end = None
    field_name = "text" if hasattr(block, "text") else None
    for idx, rp in enumerate(expanded, start=1):
        start, end, forced = rp["start"], rp["end"], rp["forced"]
        own_lines = rp["pieces"] if forced else lines[start:end]

        overlap_lines: List[str] = []
        overlap_line_start = overlap_line_end = None
        if prev_part is not None and not forced and prev_own_end - 1 >= 0:
            if tokenizer(lines[prev_own_end - 1]) <= overlap_tokens:
                overlap_lines = [lines[prev_own_end - 1]]
                overlap_line_start = overlap_line_end = prev_part.line_end

        header = f"[Part {idx} of {part_count} of {type(block).__name__} {block.block_id}]"
        joiner = "\n" if "\n" in text else " "

        def _build_generic(overlap_lines):
            part_text = joiner.join(overlap_lines + own_lines)
            pb = type(block)(
                block_id=f"{block.block_id}-part-{idx}", order=block.order, source_format=block.source_format,
                page=block.page, parent_id=block.block_id,
            )
            pb.style = dict(block.style)
            if field_name:
                setattr(pb, field_name, part_text)
            else:
                pb.normalized = part_text
            pb.style["atomic_split_context_header"] = header
            return pb

        part_block = _build_generic(overlap_lines)
        if tokenizer(header + "\n\n" + render_block(part_block)) > max_tokens and overlap_lines:
            overlap_line_start = overlap_line_end = None
            part_block = _build_generic([])

        if forced:
            unit += 1
            line_start = line_end = unit
            unit_lines.append(joiner.join(own_lines))
        else:
            line_start = unit + 1
            unit += (end - start)
            line_end = unit
            unit_lines.extend(lines[start:end])

        result.append(AtomicBlockPart(
            block=part_block, part_index=idx, part_count=part_count,
            line_start=line_start, line_end=line_end, overlap_line_start=overlap_line_start,
            overlap_line_end=overlap_line_end, forced_split=forced, context_header=header, provenance={},
        ))
        prev_part = result[-1]
        prev_own_end = end
    return result, unit_lines


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])")


def _split_sentences_generic(text: str) -> List[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return parts or [text]


def _generic_line_priority(lines: List[str], i: int) -> int:
    if i <= 0 or i >= len(lines):
        return -1
    prev, cur = lines[i - 1], lines[i]
    if prev.strip() == "" or cur.strip() == "":
        return 2
    if not _CONTINUATION_SUFFIX_RE.search(prev):
        return 1
    return 0


def split_oversized_atomic_block(
    block: Block,
    tokenizer: Tokenizer,
    max_tokens: int,
    provenance_token_reserve: int = DEFAULT_PROVENANCE_TOKEN_RESERVE,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    document_id: str = "",
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    oversized_row_policy: str = "split",
) -> List[AtomicBlockPart]:
    """Subdivides `block` into token-safe `AtomicBlockPart`s.

    `max_tokens` is the caller's already-computed effective budget for a
    chunk in this section (i.e. the embedding model's real max sequence
    length minus the chunk-level provenance prefix and heading overhead).
    `provenance_token_reserve` further reserves room for this part's own
    compact context header; `overlap_tokens` caps how much trailing content
    from the previous part is repeated for continuity. Every returned part
    is verified against `tokenizer` to fit within `max_tokens` before being
    returned.

    `oversized_row_policy` (tables only) governs what happens when a single
    table row -- with the header repeated -- alone exceeds budget: "split"
    (the default) escalates through the structural column-group/cell-text
    ladder in `_split_oversized_row`; "error" raises `AtomicSplitError`
    naming the row immediately instead of attempting to subdivide it.
    """
    # A caller-supplied reserve that's smaller than what the real header
    # text actually costs would otherwise make every part's post-hoc
    # verification fail (header text is only known once part_index/
    # part_count/overlap are known, i.e. after packing) -- ground the
    # reserve in a realistic worst-case rendering of this block's own
    # header format instead of trusting the caller's guess alone.
    if isinstance(block, CodeBlock):
        sample_header = (
            f"[Part 99 of 99 of code block {block.block_id}]\n"
            f"Parent procedure: SOME_PROCEDURE_NAME\n"
            f"Previous context: lines 9999-9999\n"
            f"Current lines: 9999-9999"
        )
    elif isinstance(block, ListBlock):
        sample_header = f"[Part 99 of 99 of list block {block.block_id}, continuing from item 999]"
    elif isinstance(block, TableBlock):
        sample_header = f"[Part 99 of 99 of table block {block.block_id} (repeated header)]"
    else:
        sample_header = f"[Part 99 of 99 of {type(block).__name__} {block.block_id}]"
    header_estimate = tokenizer(sample_header) + 5
    effective_reserve = max(provenance_token_reserve, header_estimate)
    content_budget = max(max_tokens - effective_reserve, 1)
    # Code/list/generic parts glue trailing context from the previous part
    # onto every part after the first, so their own packing budget must
    # leave room for that on top of the header reserve above -- tables
    # never add row-level overlap (only the already-reserved header row is
    # repeated), so they pack against the full content budget.
    #
    # The overlap reservation is capped at half of `content_budget` rather
    # than subtracted in full: `overlap_tokens` is a fixed constant
    # (DEFAULT_OVERLAP_TOKENS=30) sized for realistic chunk budgets, but a
    # small `max_tokens` (or a large per-part header, e.g. a CodeBlock's
    # multi-line context header) can leave `content_budget` itself well
    # under 30 -- subtracting the full constant then floors `pack_budget`
    # to 1, which doesn't gracefully shrink packing, it destroys it:
    # `_split_list_block`/`_split_code_block`/`_split_generic_block` all
    # treat "single item/line exceeds budget" as a forced split, so a
    # budget of 1 forces *every* line/item to be torn apart word-by-word
    # regardless of whether it would have fit whole in the real budget --
    # observed directly as a 25-item numbered procedure exploding into 200
    # single-word fragments. Reserving at most half still leaves genuine
    # room for overlap when the budget is comfortable (the common case,
    # where content_budget >> overlap_tokens and this is a no-op), while
    # guaranteeing packing always keeps at least half of whatever budget
    # actually exists -- and the per-part post-hoc check just below
    # (`tokenizer(context_header + ... ) > max_tokens`) already drops
    # overlap entirely for any part where even that doesn't fit, so
    # under-reserving here was never the only safety net.
    overlap_reserve = min(overlap_tokens, content_budget // 2)
    pack_budget = max(content_budget - overlap_reserve, 1)
    if isinstance(block, CodeBlock):
        parts, original_lines = _split_code_block(block, tokenizer, pack_budget, overlap_tokens, max_tokens)
    elif isinstance(block, ListBlock):
        parts, original_lines = _split_list_block(block, tokenizer, pack_budget, overlap_tokens, max_tokens)
    elif isinstance(block, TableBlock):
        parts, original_lines = _split_table_block(
            block, tokenizer, content_budget, overlap_tokens, max_tokens, oversized_row_policy,
        )
    else:
        parts, original_lines = _split_generic_block(block, tokenizer, pack_budget, overlap_tokens, max_tokens)

    if not parts:
        raise AtomicSplitError(f"Block {block.block_id} produced no parts during subdivision")

    page_start = page_start if page_start is not None else block.page
    page_end = page_end if page_end is not None else block.page
    for p in parts:
        p.provenance = {
            "document_id": document_id,
            "parent_block_id": block.block_id,
            "block_part_id": p.block.block_id,
            "part_index": p.part_index,
            "part_count": p.part_count,
            "block_type": type(block).__name__,
            "page_start": page_start,
            "page_end": page_end,
            "source_region_ids": [block.block_id],
            "line_start": p.line_start,
            "line_end": p.line_end,
            "overlap_line_start": p.overlap_line_start,
            "overlap_line_end": p.overlap_line_end,
            "forced_split": p.forced_split,
        }

    # Final hard verification against the real tokenizer/budget on exactly
    # what will be embedded (this part's own context header plus its
    # rendered content) -- the per-splitter overlap-dropping above is a
    # best-effort optimization, this is the authoritative gate.
    for p in parts:
        full_text = p.context_header + "\n\n" + render_block(p.block)
        actual = tokenizer(full_text)
        if actual > max_tokens:
            raise AtomicSplitError(
                f"Part {p.block.block_id} ({actual} tokens) still exceeds the effective "
                f"budget ({max_tokens} tokens) after subdivision"
            )

    recon = verify_reconstruction(original_lines, parts)
    if not recon["reconstructs_to_original"]:
        raise AtomicSplitError(
            f"Subdivision of block {block.block_id} failed reconstruction check: "
            f"missing={recon['missing_lines']} duplicated={recon['duplicated_non_overlap_lines']}"
        )

    forced_parts = [p for p in parts if p.forced_split]
    if forced_parts:
        # A forced split means no safe structural boundary (blank line,
        # function/algorithm boundary, sentence punctuation) existed
        # anywhere in at least one line of this block -- i.e. content shaped
        # unlike normal prose/code (near-newline-free degenerate output is
        # the pathological case this guards, but any parser/extractor could
        # produce it). Never allowed to pass silently: logged here
        # independent of the caller's own FindingLog warnings, so it's
        # visible even to a caller that doesn't surface `forced_split`.
        logger.warning(
            "atomic_split: block %s required %d forced token-boundary "
            "split(s) out of %d part(s) -- no safe structural boundary "
            "(blank line/function/sentence break) was found; content was "
            "recursively split at the token boundary, never silently "
            "dropped or truncated without record",
            block.block_id, len(forced_parts), len(parts),
        )

    return parts

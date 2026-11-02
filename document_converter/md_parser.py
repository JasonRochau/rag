"""
Canonical, fence-aware Markdown parser.

Turns raw Markdown text into the common `Document` representation (model.py).
This is used both as the Markdown format extractor (for uploaded .md/.markdown
files) and internally as the single source of truth for "what is a heading /
list / table / code block" everywhere else in the pipeline -- the normalizer,
the chunker and the validator all consume the `Document` this module builds,
never the raw text again.

Not a full CommonMark implementation (see deliverables doc "known
limitations"). It deliberately covers exactly the constructs the spec
requires to be handled correctly: ATX headings h1-h6, Setext headings,
fenced code (backtick and tilde, length >= 3, closing fence must match char
and be at least as long as the opening fence), paragraphs, ordered/unordered/
nested/task lists, GitHub-flavored pipe tables, and blockquotes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .model import (
    Document, DocumentMetadata, Block, HeadingBlock, ParagraphBlock,
    ListBlock, ListItemNode, TableBlock, CodeBlock, QuoteBlock,
)
from .findings import FindingLog, Severity

ATX_RE = re.compile(r'^(?P<hashes>#{1,6})(?:\s+(?P<text>.*?))?\s*#*\s*$')
FENCE_RE = re.compile(r'^(?P<indent>\s{0,3})(?P<fence>`{3,}|~{3,})\s*(?P<lang>[^\s`]*)\s*$')
SETEXT_H1_RE = re.compile(r'^\s{0,3}=+\s*$')
SETEXT_H2_RE = re.compile(r'^\s{0,3}-{1,}\s*$')
LIST_ITEM_RE = re.compile(
    r'^(?P<indent>\s*)(?:(?P<bullet>[-*+])|(?P<num>\d{1,9})[.)])\s+(?P<content>.*)$'
)
TASK_RE = re.compile(r'^\[(?P<mark>[ xX])\]\s+(?P<rest>.*)$')
BLOCKQUOTE_RE = re.compile(r'^\s{0,3}>\s?(?P<content>.*)$')
TABLE_ROW_RE = re.compile(r'^\s{0,3}\|?(?P<body>.+)\|?\s*$')
TABLE_DELIM_CELL_RE = re.compile(r'^:?-{1,}:?$')


def _is_blank(line: str) -> bool:
    return line.strip() == ""


def _is_fence_open(line: str) -> Optional[Tuple[str, int, str, int]]:
    m = FENCE_RE.match(line)
    if not m:
        return None
    fence = m.group("fence")
    return fence[0], len(fence), m.group("lang") or "", len(m.group("indent"))


def _is_fence_close(line: str, char: str, min_len: int) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(c != char for c in stripped):
        return False
    return len(stripped) >= min_len


def _split_table_row(line: str) -> List[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    cells: List[str] = []
    current = []
    escaped = False
    for ch in body:
        if escaped:
            current.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
            current.append(ch)
        elif ch == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    cells.append("".join(current).strip())
    return cells


def _looks_like_table_delimiter(line: str) -> bool:
    if "|" not in line and "-" not in line:
        return False
    cells = _split_table_row(line)
    if not cells:
        return False
    return all(TABLE_DELIM_CELL_RE.match(c.strip()) for c in cells if c.strip() != "") and \
        any(c.strip() for c in cells)


def _list_indent_level(indent: str) -> int:
    return len(indent.expandtabs(4))


def parse_markdown(text: str, source_path: str = "", source_format: str = "markdown") -> Tuple[Document, FindingLog]:
    """Parse raw Markdown text into a `Document`. Returns (document, findings)."""
    log = FindingLog()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    doc = Document(metadata=DocumentMetadata(source_path=source_path, source_format=source_format))
    order = 0
    n = len(lines)
    i = 0
    seen_h1 = False

    def new_id() -> str:
        nonlocal order
        order += 1
        return f"b{order:04d}"

    while i < n:
        line = lines[i]
        lineno = i + 1

        if _is_blank(line):
            i += 1
            continue

        fence_info = _is_fence_open(line)
        if fence_info:
            char, flen, lang, indent = fence_info
            code_lines: List[str] = []
            j = i + 1
            closed = False
            while j < n:
                if _is_fence_close(lines[j], char, flen):
                    closed = True
                    break
                code_lines.append(lines[j])
                j += 1
            block = CodeBlock(
                block_id=new_id(), order=order, source_format=source_format,
                page=None, language=lang, code="\n".join(code_lines),
                raw="\n".join(lines[i:j + (1 if closed else 0)]),
                normalized="\n".join(code_lines),
            )
            if not closed:
                log.error(
                    "unclosed_fence",
                    f"Fenced code block starting at line {lineno} is never closed",
                    line=lineno, block_id=block.block_id,
                    suggested_action="Add a matching closing fence of the same character and length.",
                )
            doc.add(block)
            i = j + 1 if closed else n
            continue

        m_atx = ATX_RE.match(line)
        if m_atx and (m_atx.group("text") or line.strip("#").strip() == ""):
            level = len(m_atx.group("hashes"))
            htext = (m_atx.group("text") or "").strip()
            if level == 1:
                if seen_h1:
                    log.warning("multiple_h1", f"Additional H1 found at line {lineno}: {htext!r}", line=lineno)
                seen_h1 = True
            block = HeadingBlock(
                block_id=new_id(), order=order, source_format=source_format,
                level=level, text=htext, raw=line, normalized=htext,
            )
            doc.add(block)
            i += 1
            continue

        # Setext heading lookahead: current line + next line is a valid underline,
        # and current line doesn't itself look like another block starter.
        if i + 1 < n and not _is_fence_open(line) and not LIST_ITEM_RE.match(line) and \
                not BLOCKQUOTE_RE.match(line) and "|" not in line:
            nxt = lines[i + 1]
            if SETEXT_H1_RE.match(nxt) and nxt.strip():
                htext = line.strip()
                if htext:
                    if seen_h1:
                        log.warning("multiple_h1", f"Additional H1 (setext) found at line {lineno}: {htext!r}", line=lineno)
                    seen_h1 = True
                    block = HeadingBlock(
                        block_id=new_id(), order=order, source_format=source_format,
                        level=1, text=htext, raw=line + "\n" + nxt, normalized=htext,
                    )
                    log.fixed("setext_to_atx", f"Converted setext H1 at line {lineno} to ATX", line=lineno, block_id=block.block_id)
                    doc.add(block)
                    i += 2
                    continue
            if SETEXT_H2_RE.match(nxt) and nxt.strip() and set(nxt.strip()) == {"-"}:
                htext = line.strip()
                if htext:
                    block = HeadingBlock(
                        block_id=new_id(), order=order, source_format=source_format,
                        level=2, text=htext, raw=line + "\n" + nxt, normalized=htext,
                    )
                    log.fixed("setext_to_atx", f"Converted setext H2 at line {lineno} to ATX", line=lineno, block_id=block.block_id)
                    doc.add(block)
                    i += 2
                    continue

        if BLOCKQUOTE_RE.match(line):
            q_lines = []
            j = i
            while j < n and BLOCKQUOTE_RE.match(lines[j]):
                q_lines.append(BLOCKQUOTE_RE.match(lines[j]).group("content"))
                j += 1
            qtext = " ".join(l.strip() for l in q_lines if l.strip())
            block = QuoteBlock(
                block_id=new_id(), order=order, source_format=source_format,
                raw="\n".join(lines[i:j]), normalized=qtext, text=qtext,
            )
            doc.add(block)
            i = j
            continue

        if LIST_ITEM_RE.match(line):
            j, root_items = _parse_list(lines, i, n)
            ordered = bool(LIST_ITEM_RE.match(line).group("num"))
            block = ListBlock(
                block_id=new_id(), order=order, source_format=source_format,
                ordered=ordered, items=root_items,
                raw="\n".join(lines[i:j]),
            )
            doc.add(block)
            i = j
            continue

        if "|" in line and i + 1 < n and _looks_like_table_delimiter(lines[i + 1]):
            header = _split_table_row(line)
            delim = _split_table_row(lines[i + 1])
            j = i + 2
            rows: List[List[str]] = []
            while j < n and "|" in lines[j] and not _is_blank(lines[j]) and not LIST_ITEM_RE.match(lines[j]):
                rows.append(_split_table_row(lines[j]))
                j += 1
            is_complex = any(len(r) != len(header) for r in rows) or len(delim) != len(header)
            block = TableBlock(
                block_id=new_id(), order=order, source_format=source_format,
                header=header, rows=rows, is_complex=is_complex,
                raw="\n".join(lines[i:j]),
            )
            if is_complex:
                log.warning(
                    "complex_table", "Table has ragged rows relative to its header; flagged for manual review",
                    block_id=block.block_id, line=lineno,
                    suggested_action="Verify the rendered table cells align with the source.",
                )
            doc.add(block)
            i = j
            continue

        # Paragraph: accumulate contiguous plain lines.
        p_lines = [line]
        j = i + 1
        while j < n:
            nl = lines[j]
            if _is_blank(nl):
                break
            if _is_fence_open(nl) or ATX_RE.match(nl) and (ATX_RE.match(nl).group("text") or nl.strip("#").strip() == ""):
                break
            if LIST_ITEM_RE.match(nl) or BLOCKQUOTE_RE.match(nl):
                break
            if j + 1 < n and (SETEXT_H1_RE.match(nl) is None) and "|" in nl and _looks_like_table_delimiter(lines[j + 1]):
                break
            p_lines.append(nl)
            j += 1
        ptext = " ".join(l.strip() for l in p_lines if l.strip() != "")
        block = ParagraphBlock(
            block_id=new_id(), order=order, source_format=source_format,
            raw="\n".join(p_lines), normalized=ptext, text=ptext,
        )
        doc.add(block)
        i = j

    return doc, log


def _parse_list(lines: List[str], start: int, n: int) -> Tuple[int, List[ListItemNode]]:
    """Parse a (possibly nested) list starting at `start`. Returns (next_index, items)."""

    def parse_at_level(idx: int, base_indent: int) -> Tuple[int, List[ListItemNode]]:
        items: List[ListItemNode] = []
        while idx < n:
            line = lines[idx]
            if _is_blank(line):
                # A blank line followed by a non-list, non-indented line ends the list.
                if idx + 1 >= n:
                    idx += 1
                    break
                nxt = lines[idx + 1]
                m_next = LIST_ITEM_RE.match(nxt)
                if m_next and _list_indent_level(m_next.group("indent")) >= base_indent:
                    idx += 1
                    continue
                idx += 1
                break
            m = LIST_ITEM_RE.match(line)
            if not m:
                break
            indent = _list_indent_level(m.group("indent"))
            if indent < base_indent:
                break
            if indent > base_indent:
                # Shouldn't normally happen (caller handles nesting), stop here.
                break
            content = m.group("content")
            checked = None
            tm = TASK_RE.match(content)
            if tm:
                checked = tm.group("mark").lower() == "x"
                content = tm.group("rest")
            node = ListItemNode(text=content.strip(), checked=checked)
            idx += 1
            # Look ahead for a more-indented nested list.
            if idx < n:
                look = idx
                if _is_blank(lines[look]) and look + 1 < n:
                    look += 1
                nm = LIST_ITEM_RE.match(lines[look]) if look < n else None
                if nm and _list_indent_level(nm.group("indent")) > base_indent:
                    idx, children = parse_at_level(look, _list_indent_level(nm.group("indent")))
                    node.children = children
            items.append(node)
        return idx, items

    first_indent = _list_indent_level(LIST_ITEM_RE.match(lines[start]).group("indent"))
    return parse_at_level(start, first_indent)

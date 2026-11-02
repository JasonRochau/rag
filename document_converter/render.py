"""
Deterministic Markdown renderer: `Document` -> Markdown text.

Whitespace correctness (exactly one blank line between blocks, no blank
lines inside a list, ATX-only headings with a single space after the
hashes, ...) is guaranteed *by construction* here rather than by a
regex-based text-repair pass afterward. This is also why the chunker can
reuse `render_block()` to build a chunk's body text and be certain it is
byte-identical to what ends up in the on-disk Markdown file: both walk the
same blocks through the same renderer.
"""
from __future__ import annotations

import re
from typing import List

from .model import (
    Block, HeadingBlock, ParagraphBlock, ListBlock, ListItemNode, TableBlock,
    CodeBlock, QuoteBlock, ImageBlock, PageBreakBlock, WarningBlock, UnknownBlock,
    CaptionBlock, FigureLabelBlock, DiagramAnnotationBlock, TableCellBlock,
    TitleBlock, AuthorBlock, AffiliationBlock, ContactBlock, AbstractBlock, KeywordBlock,
    PublicationMetadataBlock, CopyrightNoticeBlock, PermissionNoticeBlock, ConferenceMetadataBlock,
    Document,
)

# Guards against non-heading text that happens to start with ATX-heading
# syntax -- e.g. a Python comment line ("# Variables to hold...") extracted
# from a PDF/OCR source as plain paragraph text -- from being reinterpreted
# as a real heading by any downstream Markdown parser/viewer. The internal
# pipeline never re-parses its own rendered output (chunker.py walks the
# typed Block list directly), but the .md file written to disk is also a
# deliverable in its own right, and "no heading created from arbitrary
# content" has to hold for that file too, not just for the internal model.
_LEADING_ATX_RE = re.compile(r'^(#{1,6})(\s|$)')


def _escape_leading_atx(text: str) -> str:
    if not text:
        return text
    m = _LEADING_ATX_RE.match(text)
    if m:
        return "\\" + text
    return text


def _render_list_items(items: List[ListItemNode], ordered: bool, depth: int = 0) -> List[str]:
    out: List[str] = []
    indent = "  " * depth
    for idx, item in enumerate(items, 1):
        marker = f"{idx}." if ordered else "-"
        if item.checked is None:
            prefix = f"{indent}{marker} "
        else:
            prefix = f"{indent}{marker} [{'x' if item.checked else ' '}] "
        out.append(prefix + _escape_leading_atx(item.text))
        if item.children:
            out.extend(_render_list_items(item.children, ordered, depth + 1))
    return out


def render_list(block: ListBlock) -> str:
    return "\n".join(_render_list_items(block.items, block.ordered))


def _escape_table_cell(cell: str) -> str:
    return cell.replace("|", "\\|").replace("\n", " ").strip()


def render_table(block: TableBlock) -> str:
    header = [_escape_table_cell(c) for c in block.header]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in block.rows:
        padded = list(row) + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(_escape_table_cell(c) for c in padded[:len(header)]) + " |")
    return "\n".join(lines)


def render_code(block: CodeBlock) -> str:
    code = block.code
    longest_run = 0
    run = 0
    for ch in code:
        if ch == "`":
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 0
    fence_len = max(3, longest_run + 1)
    fence = "`" * fence_len
    return f"{fence}{block.language}\n{code}\n{fence}"


def render_quote(block: QuoteBlock) -> str:
    text = block.text or block.normalized
    return "\n".join("> " + line for line in (text.split("\n") if text else [""]))


def render_heading(block: HeadingBlock) -> str:
    return f"{'#' * block.level} {block.text}"


def render_paragraph(block: ParagraphBlock) -> str:
    return _escape_leading_atx(block.text or block.normalized)


def render_image(block: ImageBlock) -> str:
    return f"![{block.alt}]({block.src})"


def render_page_break(block: PageBreakBlock) -> str:
    return "<!-- page break -->"


def render_warning(block: WarningBlock) -> str:
    return f"> ⚠ {block.text}"


def render_unknown(block: UnknownBlock) -> str:
    return _escape_leading_atx(block.text or block.normalized or block.raw)


def render_caption(block: CaptionBlock) -> str:
    return _escape_leading_atx(block.text or block.normalized)


def render_figure_label(block: FigureLabelBlock) -> str:
    return _escape_leading_atx(block.text or block.normalized)


def render_diagram_annotation(block: DiagramAnnotationBlock) -> str:
    return _escape_leading_atx(block.text or block.normalized)


def render_table_cell(block: TableCellBlock) -> str:
    return _escape_leading_atx(block.text or block.normalized)


def render_title(block: TitleBlock) -> str:
    return _escape_leading_atx(block.text or block.normalized)


def render_front_matter_text(block: Block) -> str:
    """Shared renderer for the plain-text front-matter block types (spec
    section 4) -- author/affiliation/contact/keywords/publication metadata
    all render as ordinary Markdown text, grouped near the top of the
    document by their position in `document.blocks` (see
    `pdf_extractor.py`'s front-matter grouping pass), not by any special
    markup here."""
    return _escape_leading_atx(block.text or block.normalized)


_RENDERERS = {
    HeadingBlock: render_heading,
    ParagraphBlock: render_paragraph,
    ListBlock: render_list,
    TableBlock: render_table,
    CodeBlock: render_code,
    QuoteBlock: render_quote,
    ImageBlock: render_image,
    PageBreakBlock: render_page_break,
    WarningBlock: render_warning,
    UnknownBlock: render_unknown,
    CaptionBlock: render_caption,
    FigureLabelBlock: render_figure_label,
    DiagramAnnotationBlock: render_diagram_annotation,
    TableCellBlock: render_table_cell,
    TitleBlock: render_title,
    AuthorBlock: render_front_matter_text,
    AffiliationBlock: render_front_matter_text,
    ContactBlock: render_front_matter_text,
    AbstractBlock: render_front_matter_text,
    KeywordBlock: render_front_matter_text,
    PublicationMetadataBlock: render_front_matter_text,
    CopyrightNoticeBlock: render_front_matter_text,
    PermissionNoticeBlock: render_front_matter_text,
    ConferenceMetadataBlock: render_front_matter_text,
}


def render_block(block: Block) -> str:
    fn = _RENDERERS.get(type(block))
    if fn is None:
        return block.normalized or block.raw or ""
    return fn(block)


def render_markdown(document: Document) -> str:
    fragments = [render_block(b) for b in document.blocks if _has_content(b)]
    return "\n\n".join(fragments).strip() + "\n"


def _has_content(block: Block) -> bool:
    if isinstance(block, ListBlock):
        return bool(block.items)
    if isinstance(block, TableBlock):
        return bool(block.header)
    if isinstance(block, CodeBlock):
        return True
    text = getattr(block, "text", None) or block.normalized or block.raw
    return bool(text and text.strip())

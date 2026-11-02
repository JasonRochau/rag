"""DOCX extractor (spec 6.4). Uses python-docx; preserves source order via the document body XML."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Tuple, Optional

from docx import Document as DocxDocument
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from docx.oxml.ns import qn

from ..model import (
    Document, DocumentMetadata, HeadingBlock, ParagraphBlock, ListBlock, ListItemNode,
    TableBlock, CodeBlock,
)
from ..findings import FindingLog
from ..config import OCRConfig
from . import clean_title

_HEADING_STYLE_RE = re.compile(r"^Heading\s*(\d)$", re.IGNORECASE)
_CODE_FONTS = {"courier new", "consolas", "cascadia code", "cascadia mono", "menlo", "monaco"}


def _heading_level(style_name: str) -> Optional[int]:
    if not style_name:
        return None
    m = _HEADING_STYLE_RE.match(style_name.strip())
    if m:
        lvl = int(m.group(1))
        return max(1, min(6, lvl))
    if style_name.strip().lower() == "title":
        return 1
    return None


def _is_list_paragraph(p: DocxParagraph) -> Tuple[bool, bool]:
    """Returns (is_list, ordered)."""
    pPr = p._p.pPr
    if pPr is not None and pPr.find(qn("w:numPr")) is not None:
        numPr = pPr.find(qn("w:numPr"))
        numId_el = numPr.find(qn("w:numId"))
        ordered = False
        try:
            ordered = int(numId_el.get(qn("w:val"))) % 2 == 0
        except Exception:
            pass
        return True, ordered
    style = (p.style.name or "").lower() if p.style else ""
    if "list bullet" in style:
        return True, False
    if "list number" in style:
        return True, True
    return False, False


def _is_code_paragraph(p: DocxParagraph) -> bool:
    style = (p.style.name or "").lower() if p.style else ""
    if "code" in style or "preformatted" in style:
        return True
    for run in p.runs:
        font_name = (run.font.name or "").lower()
        if font_name in _CODE_FONTS:
            return True
    return False


def _paragraph_text_with_links(p: DocxParagraph) -> str:
    text = p.text
    try:
        for h in p.hyperlinks:
            url = h.address or ""
            if url and h.text:
                text = text.replace(h.text, f"[{h.text}]({url})", 1)
    except Exception:
        pass
    return text.strip()


def _collect_header_footer_text(d: DocxDocument) -> List[str]:
    """Distinct, non-empty header/footer paragraph text across all sections
    (spec section 3.1's headers/footers requirement). These are running
    boilerplate, not body content, so they're never added as blocks -- but
    per section 17, an exclusion must be recorded, not silent, so callers
    log what was found here."""
    seen = set()
    texts: List[str] = []
    for section in d.sections:
        for attr in ("header", "footer", "first_page_header", "first_page_footer"):
            part = getattr(section, attr, None)
            if part is None:
                continue
            try:
                if part.is_linked_to_previous:
                    continue
                for p in part.paragraphs:
                    t = p.text.strip()
                    if t and t not in seen:
                        seen.add(t)
                        texts.append(t)
            except Exception:
                continue
    return texts


class DocxExtractor:
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".docx"

    def extract(
        self, path: Path, ocr_config: OCRConfig, *, progress: Optional[Callable] = None,
    ) -> Tuple[Document, FindingLog]:
        log = FindingLog()
        d = DocxDocument(str(path))
        title = None
        try:
            title = clean_title(d.core_properties.title)
        except Exception:
            pass

        header_footer_texts = _collect_header_footer_text(d)
        if header_footer_texts:
            preview = header_footer_texts[:5]
            more = f" (+{len(header_footer_texts) - 5} more)" if len(header_footer_texts) > 5 else ""
            log.info(
                "docx_header_footer_excluded",
                f"Excluded {len(header_footer_texts)} distinct header/footer line(s) from the document "
                f"body (running boilerplate, not content): {preview}{more}",
            )

        doc = Document(metadata=DocumentMetadata(source_path=str(path), source_format="docx", title=title))
        order = 0

        def new_id() -> str:
            nonlocal order
            order += 1
            return f"b{order:04d}"

        body_children = list(d.element.body)
        paragraphs_by_id = {id(p._p): p for p in d.paragraphs}
        tables_by_id = {id(t._tbl): t for t in d.tables}

        pending_list_items: List[ListItemNode] = []
        pending_list_ordered = False
        pending_code_lines: List[str] = []

        def flush_list():
            nonlocal pending_list_items
            if pending_list_items:
                doc.add(ListBlock(block_id=new_id(), order=order, source_format="docx",
                                   ordered=pending_list_ordered, items=pending_list_items))
                pending_list_items = []

        def flush_code():
            nonlocal pending_code_lines
            if pending_code_lines:
                code = "\n".join(pending_code_lines)
                doc.add(CodeBlock(block_id=new_id(), order=order, source_format="docx",
                                   language="", code=code, normalized=code))
                pending_code_lines = []

        for el in body_children:
            tag = el.tag.split("}")[-1]
            if tag == "p" and id(el) in paragraphs_by_id:
                p = paragraphs_by_id[id(el)]
                text = _paragraph_text_with_links(p)
                if not text.strip():
                    continue
                level = _heading_level(p.style.name if p.style else "")
                if level:
                    flush_list()
                    flush_code()
                    doc.add(HeadingBlock(block_id=new_id(), order=order, source_format="docx",
                                          level=level, text=text, normalized=text))
                    continue
                if _is_code_paragraph(p):
                    flush_list()
                    pending_code_lines.append(text)
                    continue
                flush_code()
                is_list, ordered = _is_list_paragraph(p)
                if is_list:
                    if pending_list_items and pending_list_ordered != ordered:
                        flush_list()
                    pending_list_ordered = ordered
                    pending_list_items.append(ListItemNode(text=text))
                    continue
                flush_list()
                doc.add(ParagraphBlock(block_id=new_id(), order=order, source_format="docx",
                                        text=text, normalized=text))
            elif tag == "tbl" and id(el) in tables_by_id:
                flush_list()
                flush_code()
                t = tables_by_id[id(el)]
                header, rows, is_complex = _table_rows(t)
                doc.add(TableBlock(block_id=new_id(), order=order, source_format="docx",
                                    header=header, rows=rows, is_complex=is_complex))
                if is_complex:
                    log.warning("complex_docx_table", "DOCX table has ragged rows; flagged for manual review")

        flush_list()
        flush_code()

        if not doc.blocks:
            log.warning("empty_docx", "No extractable paragraphs or tables were found in the DOCX file")

        return doc, log


def _table_rows(t: DocxTable) -> Tuple[List[str], List[List[str]], bool]:
    grid = [[cell.text.strip() for cell in row.cells] for row in t.rows]
    if not grid:
        return [], [], False
    header, rows = grid[0], grid[1:]
    is_complex = any(len(r) != len(header) for r in rows)
    return header, rows, is_complex

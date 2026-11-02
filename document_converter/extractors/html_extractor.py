"""HTML extractor (spec 6.3). Uses BeautifulSoup with a local (no network) parser."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

from ..model import (
    Document, DocumentMetadata, HeadingBlock, ParagraphBlock, ListBlock, ListItemNode,
    TableBlock, CodeBlock, QuoteBlock, ImageBlock, UnknownBlock,
)
from ..findings import FindingLog
from ..config import OCRConfig

_BOILERPLATE_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript", "svg", "form", "iframe"}
_HEADING_TAGS = {f"h{i}": i for i in range(1, 7)}
_UNCONVERTIBLE_BLOCK_TAGS = {"video", "audio", "canvas", "embed", "object", "map"}


def _inline_text(node: Tag) -> str:
    parts: List[str] = []
    for el in node.descendants:
        if isinstance(el, NavigableString):
            if el.parent and el.parent.name == "a":
                continue
            parts.append(str(el))
        elif isinstance(el, Tag) and el.name == "a":
            href = el.get("href", "")
            text = el.get_text(strip=True)
            parts.append(f"[{text}]({href})" if href else text)
    text = " ".join(" ".join(parts).split())
    return text


class HtmlExtractor:
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in (".html", ".htm")

    def extract(
        self, path: Path, ocr_config: OCRConfig, *, progress: Optional[Callable] = None,
    ) -> Tuple[Document, FindingLog]:
        log = FindingLog()
        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "lxml")

        for tag in soup.find_all(list(_BOILERPLATE_TAGS)):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else None
        doc = Document(metadata=DocumentMetadata(source_path=str(path), source_format="html", title=title))
        order = 0

        def new_id() -> str:
            nonlocal order
            order += 1
            return f"b{order:04d}"

        body = soup.body or soup

        def walk(node: Tag):
            nonlocal order
            for child in node.find_all(recursive=False):
                if not isinstance(child, Tag):
                    continue
                name = child.name.lower()
                if name in _HEADING_TAGS:
                    text = child.get_text(" ", strip=True)
                    if text:
                        doc.add(HeadingBlock(block_id=new_id(), order=order, source_format="html",
                                              level=_HEADING_TAGS[name], text=text, normalized=text))
                elif name == "p":
                    text = _inline_text(child)
                    if text:
                        doc.add(ParagraphBlock(block_id=new_id(), order=order, source_format="html",
                                                text=text, normalized=text))
                elif name in ("ul", "ol"):
                    items = _parse_list(child)
                    if items:
                        doc.add(ListBlock(block_id=new_id(), order=order, source_format="html",
                                           ordered=(name == "ol"), items=items))
                elif name == "table":
                    header, rows, is_complex = _parse_table(child)
                    if header or rows:
                        doc.add(TableBlock(block_id=new_id(), order=order, source_format="html",
                                            header=header, rows=rows, is_complex=is_complex))
                        if is_complex:
                            log.warning("complex_html_table", "HTML table has merged/ragged cells; flagged for manual review")
                elif name == "pre":
                    code_tag = child.find("code")
                    lang = ""
                    if code_tag and code_tag.get("class"):
                        for c in code_tag.get("class"):
                            if c.startswith("language-"):
                                lang = c[len("language-"):]
                    code_text = (code_tag or child).get_text()
                    doc.add(CodeBlock(block_id=new_id(), order=order, source_format="html",
                                       language=lang, code=code_text, normalized=code_text))
                elif name == "blockquote":
                    text = _inline_text(child)
                    if text:
                        doc.add(QuoteBlock(block_id=new_id(), order=order, source_format="html", text=text, normalized=text))
                elif name == "img":
                    doc.add(ImageBlock(block_id=new_id(), order=order, source_format="html",
                                        alt=child.get("alt", ""), src=child.get("src", "")))
                elif name in _UNCONVERTIBLE_BLOCK_TAGS:
                    doc.add(UnknownBlock(block_id=new_id(), order=order, source_format="html",
                                          text=f"<{name}> element omitted (unsupported)",
                                          warnings=["remaining_html"]))
                    log.warning("unconvertible_html", f"<{name}> element could not be converted to Markdown and was dropped")
                elif name in ("div", "section", "article", "main", "span", "body", "figure"):
                    walk(child)
                else:
                    walk(child)

        walk(body)
        return doc, log


def _parse_list(node: Tag) -> List[ListItemNode]:
    items: List[ListItemNode] = []
    for li in node.find_all("li", recursive=False):
        nested = li.find(["ul", "ol"], recursive=False)
        text_parts = []
        for el in li.contents:
            if isinstance(el, Tag) and el.name in ("ul", "ol"):
                continue
            if isinstance(el, NavigableString):
                text_parts.append(str(el))
            elif isinstance(el, Tag):
                text_parts.append(el.get_text(" ", strip=True))
        text = " ".join(" ".join(text_parts).split())
        checkbox = li.find("input", attrs={"type": "checkbox"})
        checked = bool(checkbox.get("checked") is not None) if checkbox else None
        node_item = ListItemNode(text=text, checked=checked)
        if nested is not None:
            node_item.children = _parse_list(nested)
        items.append(node_item)
    return items


def _parse_table(node: Tag):
    rows_tags = node.find_all("tr")
    header: List[str] = []
    rows: List[List[str]] = []
    is_complex = False
    for tag in node.find_all(["td", "th"]):
        if tag.get("colspan") or tag.get("rowspan"):
            is_complex = True
            break
    for i, tr in enumerate(rows_tags):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if i == 0 and tr.find("th"):
            header = cells
        elif i == 0 and not header:
            header = cells
        else:
            rows.append(cells)
    if header and any(len(r) != len(header) for r in rows):
        is_complex = True
    return header, rows, is_complex

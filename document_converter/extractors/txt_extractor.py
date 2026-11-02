"""
Plain-text extractor (spec 6.2).

Deliberately conservative: TXT has no structural syntax, so we only ever
promise a title (H1) plus a single flat layer of H2 "heading-looking" lines,
never a deep hierarchy. Anything fancier is left to optional semantic
structuring (section 8).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from ..model import Document, DocumentMetadata, HeadingBlock, ParagraphBlock, ListBlock, ListItemNode, CodeBlock
from ..findings import FindingLog
from ..config import OCRConfig
from . import LIST_LINE_RE as _LIST_RE, is_heading_candidate as _is_heading_candidate

_CMD_RE = re.compile(r'^(\$\s+|>\s+|#\s*[A-Za-z]).{0,200}$')


class TxtExtractor:
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in (".txt", ".text")

    def extract(
        self, path: Path, ocr_config: OCRConfig, *, progress: Optional[Callable] = None,
    ) -> Tuple[Document, FindingLog]:
        log = FindingLog()
        raw = path.read_text(encoding="utf-8", errors="replace")
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        lines = raw.split("\n")

        doc = Document(metadata=DocumentMetadata(source_path=str(path), source_format="txt"))
        order = 0

        def new_id() -> str:
            nonlocal order
            order += 1
            return f"b{order:04d}"

        n = len(lines)
        i = 0
        found_h1 = False
        while i < n:
            line = lines[i]
            if not line.strip():
                i += 1
                continue

            prev_blank = i == 0 or not lines[i - 1].strip()
            next_blank = i + 1 >= n or not lines[i + 1].strip()

            if prev_blank and next_blank and _is_heading_candidate(line):
                level = 1 if not found_h1 else 2
                found_h1 = True
                doc.add(HeadingBlock(
                    block_id=new_id(), order=order, source_format="txt",
                    level=level, text=line.strip(), normalized=line.strip(),
                ))
                i += 1
                continue

            if _LIST_RE.match(line):
                items: List[ListItemNode] = []
                ordered = False
                j = i
                while j < n and (_LIST_RE.match(lines[j]) or (lines[j].strip() == "" and j + 1 < n and _LIST_RE.match(lines[j + 1]))):
                    if lines[j].strip() == "":
                        j += 1
                        continue
                    m = _LIST_RE.match(lines[j])
                    ordered = ordered or bool(re.match(r'\d', m.group(2)))
                    items.append(ListItemNode(text=m.group(3).strip()))
                    j += 1
                doc.add(ListBlock(
                    block_id=new_id(), order=order, source_format="txt",
                    ordered=ordered, items=items,
                    raw="\n".join(lines[i:j]),
                ))
                i = j
                continue

            if _CMD_RE.match(line):
                j = i
                cmd_lines = []
                while j < n and (lines[j].strip() == "" or _CMD_RE.match(lines[j]) or lines[j].startswith("  ")):
                    if lines[j].strip() == "" and (j + 1 >= n or not (_CMD_RE.match(lines[j + 1]) or lines[j + 1].startswith("  "))):
                        break
                    cmd_lines.append(lines[j])
                    j += 1
                while cmd_lines and not cmd_lines[-1].strip():
                    cmd_lines.pop()
                doc.add(CodeBlock(
                    block_id=new_id(), order=order, source_format="txt",
                    language="", code="\n".join(cmd_lines), normalized="\n".join(cmd_lines),
                    raw="\n".join(lines[i:j]),
                ))
                i = j
                continue

            j = i
            p_lines = []
            while j < n and lines[j].strip() and not _LIST_RE.match(lines[j]) and not _CMD_RE.match(lines[j]) and \
                    not (i != j and (j == i or not lines[j - 1].strip()) and (j + 1 >= n or not lines[j + 1].strip()) and _is_heading_candidate(lines[j])):
                p_lines.append(lines[j])
                j += 1
            text = " ".join(l.strip() for l in p_lines if l.strip())
            if text:
                doc.add(ParagraphBlock(
                    block_id=new_id(), order=order, source_format="txt",
                    text=text, normalized=text, raw="\n".join(p_lines),
                ))
            i = max(j, i + 1)

        if not found_h1:
            log.warning("txt_no_title_detected", "No confident title line found in TXT source; a title will be synthesized during normalization")

        return doc, log

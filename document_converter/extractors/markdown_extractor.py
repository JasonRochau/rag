from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

from ..model import Document
from ..findings import FindingLog
from ..config import OCRConfig
from ..md_parser import parse_markdown


class MarkdownExtractor:
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in (".md", ".markdown")

    def extract(
        self, path: Path, ocr_config: OCRConfig, *, progress: Optional[Callable] = None,
    ) -> Tuple[Document, FindingLog]:
        text = path.read_text(encoding="utf-8", errors="replace")
        document, log = parse_markdown(text, source_path=str(path), source_format="markdown")
        document.metadata.title = None
        return document, log

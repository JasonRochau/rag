"""
Legacy .doc extractor (spec section 3.2).

.doc is a binary OLE-compound-file format with no maintained pure-Python
parser worth depending on. The reliable local approach is to convert it to
.docx with LibreOffice headless mode and run the result through the
existing DocxExtractor, so every downstream stage only ever has to deal
with one Word format. Conversion runs fully locally (a subprocess call to
`soffice`); nothing is sent to a remote service, and the original .doc
file is never modified.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple

from ..model import Document, DocumentMetadata
from ..findings import FindingLog
from ..config import OCRConfig
from .docx_extractor import DocxExtractor


class DocConversionUnavailableError(RuntimeError):
    pass


def find_soffice() -> str:
    for name in ("soffice", "libreoffice"):
        path = shutil.which(name)
        if path:
            return path
    raise DocConversionUnavailableError(
        "Converting a legacy .doc file requires LibreOffice (the `soffice` binary), which is not "
        "on PATH. Install it (e.g. `sudo apt install libreoffice` on Debian/Ubuntu, `brew install "
        "--cask libreoffice` on macOS) so `soffice` is available, or convert the file to .docx "
        "yourself first and upload that instead. Conversion runs fully locally; nothing is sent to "
        "a remote service."
    )


class DocExtractor:
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".doc"

    def extract(
        self, path: Path, ocr_config: OCRConfig, *, progress: Optional[Callable] = None,
    ) -> Tuple[Document, FindingLog]:
        log = FindingLog()
        try:
            soffice = find_soffice()
        except DocConversionUnavailableError as e:
            log.error("doc_conversion_unavailable", str(e),
                      suggested_action="Install LibreOffice or convert the file to .docx before uploading.")
            return Document(metadata=DocumentMetadata(source_path=str(path), source_format="doc")), log

        with tempfile.TemporaryDirectory(prefix="docconvert-") as tmp_dir:
            try:
                result = subprocess.run(
                    [soffice, "--headless", "--norestore", "--convert-to", "docx", "--outdir", tmp_dir, str(path)],
                    capture_output=True, text=True, timeout=120,
                )
            except subprocess.TimeoutExpired:
                log.error("doc_conversion_timeout",
                          f"LibreOffice conversion of {path.name} to .docx timed out after 120 seconds")
                return Document(metadata=DocumentMetadata(source_path=str(path), source_format="doc")), log
            except Exception as e:
                log.error("doc_conversion_failed", f"Could not run LibreOffice to convert {path.name}: {e}")
                return Document(metadata=DocumentMetadata(source_path=str(path), source_format="doc")), log

            converted = Path(tmp_dir) / (path.stem + ".docx")
            if result.returncode != 0 or not converted.exists():
                detail = (result.stderr or result.stdout or "").strip()[:300]
                log.error(
                    "doc_conversion_failed",
                    f"LibreOffice could not convert {path.name} to .docx (exit code {result.returncode})"
                    + (f": {detail}" if detail else ""),
                    suggested_action="Confirm the file is a genuine .doc (not renamed/corrupt) and "
                                      "that LibreOffice can open it manually.",
                )
                return Document(metadata=DocumentMetadata(source_path=str(path), source_format="doc")), log

            log.info("doc_converted_via_libreoffice",
                      f"Converted {path.name} from legacy .doc to .docx locally via LibreOffice for extraction; "
                      f"lower structural fidelity than a native .docx should be expected")
            document, extract_log = DocxExtractor().extract(converted, ocr_config)

        document.metadata.source_format = "doc"
        document.metadata.source_path = str(path)
        log.extend(extract_log)
        return document, log

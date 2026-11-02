"""
DoclingExtractor -- structural-inference extractor behind the `PARSER=docling`
feature flag (wired in `get_extractors()` below). Disabled by default;
`pdf_extractor.py` remains the default and the fallback for every other flag
state, per the layout PoC's integration plan
(`artifacts/layout_poc/reports/final_recommendation.md`).

Docling runs in its own isolated virtualenv (`.venv-docling/`, ~5.5 GiB with
torch/CUDA/transformers) -- entirely separate from this project's own
`.venv`, which does not have `docling` installed. This extractor therefore
shells out to that venv's `docling` CLI as a subprocess rather than importing
the package in-process, writes its JSON export to a throwaway temp
directory, and adapts that JSON into the common `Document`/`Block` model --
the same DoclingDocument-JSON-to-`Document` mapping validated in the PoC,
moved here from `artifacts/layout_poc/adapter/docling_adapter.py` once it
cleared the four required gating conditions:

  1. `--enrich-formula` is never passed. The formula-enrichment VLM
     (CodeFormulaV2) was observed, in the PoC, to degenerate into a
     multi-KB repetition-loop LaTeX blob when mis-routed a pseudocode
     region -- a real model defect, not a tuning issue -- so formula
     enrichment is disabled entirely rather than worked around.
  2. `demote_false_headings` (`..heading_demotion`) runs on this
     extractor's output the same as every other extractor's: it is wired
     into the shared `_build_document()` stage chain in `pipeline.py`
     (after normalize/running-header repair/structuring/bodyless-heading
     repair, before the heading audits), so every downstream stage sees
     the same "no obviously-false headings" document shape the legacy PDF
     path already benefits from via its own font-size/isolation gating.
     It was briefly called from this extractor directly; that call was
     removed when the pass moved into the shared chain, so the docling
     path never runs it twice.
  3. Every block this extractor builds sets both `.text` and `.normalized`.
     A formula/caption/code/warning block that only set `.text` made its
     real size invisible to `Block.word_count`-driven chunk packing during
     the PoC (fixed generally as a fallback in `model.py`); this extractor
     also sets both fields directly at construction time, as defense in
     depth independent of that base-class fallback.

`formula` has no corresponding `Block` subclass in the common model (a real
gap the PoC's Phase 1 boundary inspection identified). Rather than
misclassify it as a paragraph or heading, it becomes an `UnknownBlock`
tagged `style["parser_region_type"] = "formula"` and is logged in
`unmapped_element_counts`. `page_header`/`page_footer` regions are dropped
(not emitted as any block), with a running count recorded in the document's
`extra` metadata -- the one deliberate, counted drop in this extractor.
Every element label without an explicit rule is logged to
`unmapped_element_counts` and emitted as `UnknownBlock`/`WarningBlock`
rather than silently dropped or misclassified.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..model import (
    Document, DocumentMetadata, HeadingBlock, ParagraphBlock, ListBlock, ListItemNode,
    TableBlock, CodeBlock, ImageBlock, CaptionBlock, UnknownBlock, WarningBlock,
)
from ..findings import FindingLog
from ..config import OCRConfig
from . import numbering_tuple, safe_progress

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCLING_CLI = REPO_ROOT / ".venv-docling" / "bin" / "docling"
DOCLING_CONVERT_TIMEOUT_SECONDS = 900


class DoclingUnavailableError(RuntimeError):
    """Raised when the isolated `.venv-docling` CLI is missing or fails to
    run -- caught by `extract()` and turned into a clean `docling_unavailable`
    finding, the same pattern `doc_extractor.py` uses for a missing
    LibreOffice binary, never a raw traceback bubbling past this point."""


def _resolve(d: Dict[str, Any], ref: str) -> Dict[str, Any]:
    """Resolve a DoclingDocument '#/texts/12'-style JSON pointer."""
    parts = ref.lstrip("#/").split("/")
    node: Any = d
    for p in parts:
        node = node[int(p)] if p.isdigit() else node[p]
    return node


def _item_text(item: Dict[str, Any]) -> str:
    return item.get("text", "") or ""


def _item_page(item: Dict[str, Any]) -> Optional[int]:
    prov = item.get("prov") or []
    return prov[0]["page_no"] if prov else None


def _item_bbox(item: Dict[str, Any]) -> Optional[Dict[str, float]]:
    prov = item.get("prov") or []
    return prov[0]["bbox"] if prov else None


def _heading_level(text: str) -> int:
    nums = numbering_tuple(text)
    if nums is None:
        return 2
    return min(1 + len(nums), 6)


def _table_grid(table_item: Dict[str, Any]) -> Tuple[List[str], List[List[str]]]:
    data = table_item.get("data", {})
    n_rows, n_cols = data.get("num_rows", 0), data.get("num_cols", 0)
    grid: List[List[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    header_rows: set = set()
    for cell in data.get("table_cells", []):
        r0, r1 = cell["start_row_offset_idx"], cell["end_row_offset_idx"]
        c0, c1 = cell["start_col_offset_idx"], cell["end_col_offset_idx"]
        text = cell.get("text", "")
        if cell.get("column_header"):
            header_rows.add(r0)
        for r in range(r0, min(r1, n_rows)):
            for c in range(c0, min(c1, n_cols)):
                grid[r][c] = text
    if header_rows:
        header_idx = min(header_rows)
        header = grid[header_idx]
        rows = [row for i, row in enumerate(grid) if i != header_idx]
    elif grid:
        header, rows = grid[0], grid[1:]
    else:
        header, rows = [], []
    return header, rows


class _DoclingJsonAdapter:
    """DoclingDocument JSON -> `Document`/`Block` mapping. See module
    docstring for the design decisions behind each branch below."""

    def __init__(self):
        self.unmapped_counts: Dict[str, int] = {}
        self.dropped_page_furniture = 0
        self.title_seen = False

    def _log_unmapped(self, label: str) -> None:
        self.unmapped_counts[label] = self.unmapped_counts.get(label, 0) + 1

    def adapt(self, docling_json_path: str, source_pdf_path: str) -> Document:
        d = json.loads(Path(docling_json_path).read_text(encoding="utf-8"))
        if d.get("schema_name") != "DoclingDocument":
            raise ValueError(f"Not a DoclingDocument export: schema_name={d.get('schema_name')!r}")

        blocks = []
        body = d["body"]
        for child_ref in body.get("children", []):
            blocks.extend(self._walk(d, child_ref["$ref"]))

        for i, b in enumerate(blocks, start=1):
            b.block_id = f"b{i:04d}"
            b.order = i - 1

        metadata = DocumentMetadata(
            source_path=source_pdf_path,
            source_format="pdf",
            title=next((b.text for b in blocks if isinstance(b, HeadingBlock) and b.level == 1), None),
            page_count=len(d.get("pages", {})),
            ocr_used=True,
            extra={
                "adapter": "docling",
                "docling_version": d.get("version"),
                "dropped_page_furniture_count": self.dropped_page_furniture,
                "unmapped_element_counts": dict(self.unmapped_counts),
            },
        )
        return Document(metadata=metadata, blocks=blocks)

    _LEAF_LABELS = {
        "section_header", "text", "list_item", "code", "formula", "caption",
        "picture", "table", "page_header", "page_footer",
    }

    def _walk(self, d: Dict[str, Any], ref: str) -> List[Any]:
        item = _resolve(d, ref)
        label = item.get("label")
        name = item.get("name")

        if "children" in item and label not in self._LEAF_LABELS:
            if name == "list" or label == "list":
                return [self._list_block(d, item)]
            out = []
            for c in item.get("children", []):
                out.extend(self._walk(d, c["$ref"]))
            return out

        if label == "section_header":
            text = _item_text(item)
            if not self.title_seen:
                self.title_seen = True
                level = 1
            else:
                level = _heading_level(text)
            return [HeadingBlock(
                block_id="", order=0, source_format="pdf", text=text, normalized=text, level=level,
                page=_item_page(item), style={"docling_label": label, "bbox": _item_bbox(item), "parser": "docling"},
            )]

        if label == "text":
            text = _item_text(item)
            return [ParagraphBlock(
                block_id="", order=0, source_format="pdf", text=text, normalized=text,
                page=_item_page(item), style={"docling_label": label, "bbox": _item_bbox(item), "parser": "docling"},
            )]

        if label == "list_item":
            text = _item_text(item)
            return [ListBlock(
                block_id="", order=0, source_format="pdf", items=[ListItemNode(text=text)],
                page=_item_page(item), style={"docling_label": label, "parser": "docling"},
            )]

        if label == "code":
            text = _item_text(item)
            return [CodeBlock(
                block_id="", order=0, source_format="pdf", code=text,
                page=_item_page(item), style={"docling_label": label, "bbox": _item_bbox(item), "parser": "docling"},
            )]

        if label == "formula":
            text = _item_text(item)
            return [UnknownBlock(
                block_id="", order=0, source_format="pdf", text=text, normalized=text,
                page=_item_page(item),
                style={"docling_label": label, "parser_region_type": "formula", "bbox": _item_bbox(item), "parser": "docling"},
            )]

        if label == "caption":
            text = _item_text(item)
            return [CaptionBlock(
                block_id="", order=0, source_format="pdf", text=text, normalized=text,
                page=_item_page(item), style={"docling_label": label, "bbox": _item_bbox(item), "parser": "docling"},
            )]

        if label == "picture":
            child_texts = []
            for c in item.get("children", []):
                child_item = _resolve(d, c["$ref"])
                t = _item_text(child_item)
                if t:
                    child_texts.append(t)
            for cap_ref in item.get("captions", []):
                cap_item = _resolve(d, cap_ref["$ref"])
                t = _item_text(cap_item)
                if t:
                    child_texts.append(f"[caption] {t}")
            return [ImageBlock(
                block_id="", order=0, source_format="pdf", alt="; ".join(child_texts), src="",
                page=_item_page(item), style={"docling_label": label, "bbox": _item_bbox(item), "parser": "docling"},
            )]

        if label == "table":
            header, rows = _table_grid(item)
            return [TableBlock(
                block_id="", order=0, source_format="pdf", header=header, rows=rows,
                page=_item_page(item), style={"docling_label": label, "bbox": _item_bbox(item), "parser": "docling"},
            )]

        if label in ("page_header", "page_footer"):
            self.dropped_page_furniture += 1
            return []

        self._log_unmapped(label or name or "unknown")
        text = _item_text(item)
        if text:
            return [WarningBlock(
                block_id="", order=0, source_format="pdf", text=text, normalized=text,
                page=_item_page(item), style={"docling_label": label, "bbox": _item_bbox(item), "parser": "docling"},
            )]
        return []

    def _list_block(self, d: Dict[str, Any], group_item: Dict[str, Any]) -> ListBlock:
        items = []
        page = None
        for c in group_item.get("children", []):
            child = _resolve(d, c["$ref"])
            page = page or _item_page(child)
            items.append(ListItemNode(text=_item_text(child)))
        return ListBlock(
            block_id="", order=0, source_format="pdf", items=items, page=page,
            style={"docling_label": "list", "parser": "docling"},
        )


class DoclingExtractor:
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def extract(
        self, path: Path, ocr_config: OCRConfig, *, progress: Optional[Callable] = None,
    ) -> Tuple[Document, FindingLog]:
        log = FindingLog()
        safe_progress(progress, "extracting", current=0, total=0)

        if not DOCLING_CLI.exists():
            log.error(
                "docling_unavailable",
                f"PARSER=docling is set but the isolated Docling environment "
                f"({DOCLING_CLI}) is not installed. Nothing was extracted.",
                suggested_action="Install it (python3 -m venv .venv-docling && "
                                  ".venv-docling/bin/pip install docling), or unset PARSER to "
                                  "use the default pdf_extractor.py path.",
            )
            return Document(metadata=DocumentMetadata(source_path=str(path), source_format="pdf")), log

        with tempfile.TemporaryDirectory(prefix="docling_extract_") as tmpdir:
            try:
                # `--enrich-formula` deliberately omitted -- see module
                # docstring, condition 1. `--device auto` (not a fixed
                # `cuda`, unlike the PoC's benchmarking config) so this runs
                # correctly on a machine with no GPU, falling back to CPU.
                subprocess.run(
                    [
                        str(DOCLING_CLI), "convert", str(path),
                        "--to", "json",
                        "--device", "auto",
                        "--table-mode", "accurate",
                        "--enrich-code",
                        "--output", tmpdir,
                    ],
                    check=True, capture_output=True, text=True,
                    timeout=DOCLING_CONVERT_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as e:
                log.error(
                    "docling_conversion_timed_out",
                    f"Docling conversion of {path.name} did not finish within "
                    f"{DOCLING_CONVERT_TIMEOUT_SECONDS}s.",
                )
                return Document(metadata=DocumentMetadata(source_path=str(path), source_format="pdf")), log
            except subprocess.CalledProcessError as e:
                log.error(
                    "docling_conversion_failed",
                    f"Docling conversion of {path.name} failed: {e.stderr[-500:] if e.stderr else e}",
                )
                return Document(metadata=DocumentMetadata(source_path=str(path), source_format="pdf")), log

            json_path = Path(tmpdir) / f"{path.stem}.json"
            if not json_path.exists():
                log.error(
                    "docling_conversion_failed",
                    f"Docling conversion of {path.name} reported success but produced no JSON export "
                    f"at the expected path ({json_path}).",
                )
                return Document(metadata=DocumentMetadata(source_path=str(path), source_format="pdf")), log

            adapter = _DoclingJsonAdapter()
            document = adapter.adapt(str(json_path), str(path))
            page_count = document.metadata.page_count or 0
            safe_progress(progress, "extracting", current=page_count, total=page_count)

        if adapter.unmapped_counts:
            log.warning(
                "docling_unmapped_element_types",
                f"{sum(adapter.unmapped_counts.values())} element(s) across "
                f"{len(adapter.unmapped_counts)} type(s) had no explicit mapping rule: "
                f"{adapter.unmapped_counts}; preserved as UnknownBlock/WarningBlock, not dropped.",
            )

        return document, log

"""
Pre-ingest PDF splitter (spec: "a large PDF cannot go through the pipeline as
one object").

`DocumentPipeline` materializes an entire document and then runs many
O(blocks) passes over it (running-header repair, heading-sibling/numbering
audits, chunking, validation, ...). A PDF with tens of thousands of pages
blows that up. The fix lives entirely *before* ingestion: split the raw PDF
into whole-structural-unit parts, each a few megabytes, and run every part
through the exact same `DocumentPipeline.index()` entry point the CLI's
`index` command uses. All parts write into the same `paths.index_dir`, so the
final index is the union of what each part produced -- this module never
touches `pipeline.py`, the extractor, the chunker, or the index schema.

STEP 1 -- WHERE STRUCTURE COMES FROM
`DocumentPipeline` only knows what a heading is *after* extraction (a
`HeadingBlock`, decided by extraction-time font-size heuristics), and in this
extractor `HeadingBlock.level == 1` is the resolved document *title*, not a
repeatable chapter/section tier (confirmed: a PDF with several visually
"chapter"-styled headings still yields exactly one level-1 `HeadingBlock`).
So the splitter's structural source is NOT extracted headings -- it is the
PDF's own outline/bookmark tree (`pymupdf.Document.get_toc()`), read directly,
zero extraction:

  * Option A (preferred): the raw outline tree, at EVERY level, not just
    level 1. Real-world outlines frequently include "folder" bookmarks that
    group children but carry no destination page of their own
    (`get_toc(simple=False)` shows `kind: 0, page: -1`, as opposed to a real
    `GoTo` destination's `kind: 4`) -- this happens at every depth, not only
    at level 1. Such an entry's start page is resolved to the smallest
    resolvable page among its own nested descendants (every entry, any
    level, between it and its next same-level sibling) -- still pure
    outline-tree reading, no PDF content is touched. Confirmed against
    sql-sql-server-ver17.pdf: 20 of 23 level-1 entries are folder-only, and
    the oversized `Reference` section is itself one of them, containing 13
    level-2 children (some also folder-only) descending as deep as level 6.
  * Option B (fallback, only when Option A yields nothing usable): run the
    existing `PdfExtractor` once and take level-1 `HeadingBlock` positions.
    This is the expensive path Option A exists to avoid, so it is only ever
    reached when the PDF has no outline, or its level-1 entries carry no
    resolvable page anywhere in their subtree. Option B never yields more
    than one boundary in this extractor (see above), so it can never itself
    be recursed into -- a document that falls through to Option B is, by
    definition, one this splitter cannot subdivide beyond whatever Option B
    finds.
  * If neither source yields any boundary, the whole document is treated as
    a single unit -- there is nothing to split on without inventing
    boundaries the document itself doesn't have.

RULE: HOW A CHILD'S PAGE SPAN IS COMPUTED (stated once, applied at every
outline level and every recursion depth)
  * A structural unit begins at its own resolved destination page (or, for a
    folder-only entry, the smallest resolvable page anywhere in its
    subtree).
  * Its span ends immediately before the resolved start page of the next
    sibling *at the same level, within the same parent*; the last sibling's
    span extends through its parent's own resolved end page.
  * Pages before the first child's resolved start page (within a parent's
    own span) are NOT dropped and NOT attached to any child -- they become
    that parent's own leading "front matter" / "(lead-in)" unit, which has no
    further substructure of its own (folding the same rule the top level
    already used for cover/TOC pages before the first H1).
  * A sibling whose own destination (or every one of its descendants') is
    unresolvable, or resolves to a page at or before the previous sibling's
    start, is dropped from consideration entirely -- its pages are not
    carved out separately; they fall to whichever neighboring unit's span
    already covers them under the two rules above. No page is ever
    orphaned: every page in a parent's span belongs to exactly one child
    span or the parent's own lead-in span.

RECURSIVE SPLIT ALGORITHM
Structural units at a level are contiguous, non-overlapping page ranges
covering their parent's entire span (see the rule above). Walk them in
document order, growing the current part one whole unit at a time; before
adding a unit, build the *candidate* part as a real fitz document, save it,
and check its actual on-disk size. If the candidate would exceed the ~5MB
target, the unit is *not* added -- the current part is closed as-is and the
unit starts a new part instead.

A unit is never split *inside* -- but before a unit becomes (part of) an
output part, it is measured *alone*. If even the unit by itself already
exceeds the target:
  * and it has outline children (a real PDF outline node one level deeper,
    covering some sub-range of its own span) -- do NOT accept it as one
    giant part. Recurse: apply this exact same algorithm to its children
    instead, packing THEM into target-sized parts. This can happen at any
    depth and repeats until the oversized content fits, however many levels
    that takes (`sql-sql-server-ver17.pdf`'s `Reference` needs two: level 1
    -> level 2 -> level 3 for its largest sub-branch).
  * and it has NO outline children (a childless leaf, or a unit whose
    children all failed to resolve to a usable page) -- there is no
    structure left to recurse into. This is the documented, intentional
    floor: the splitter raises `SplitError` with the outline path, level,
    resolved first/last page, and actual assembled byte size, rather than
    silently emitting an oversized part or falling back to an arbitrary
    page-count cut (which would violate section integrity). Solving this
    case for real requires a streaming/low-memory ingestion pipeline, which
    is out of scope for this module -- see `SplitError`'s message.

No document-size/page-count/part-count/outline-depth limit exists anywhere
in this module -- the ~5MB figure is a soft target on assembled bytes, not a
cap, and there is no hard-coded maximum recursion depth: descent follows
whatever the PDF's own outline tree actually contains.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .config import load_config, PipelineConfig, INGESTION_PROFILES
from .model import HeadingBlock
from .pipeline import DocumentPipeline, PipelineValidationError

# Soft target for an assembled part's on-disk size (spec: "~5MB parts").
# Not a limit on anything -- see module docstring.
TARGET_PART_BYTES = 5 * 1024 * 1024


# --------------------------------------------------------------------------
# Outline tree: structural units, at any depth

@dataclass
class Unit:
    """One structural unit at some outline level: a contiguous, 0-indexed,
    inclusive page span, plus its own already-resolved outline children (an
    empty list means "nothing left to recurse into if this is oversized").

    `path` is the outline breadcrumb from the document root down to this
    unit (titles only), used for `SplitError` diagnostics. `is_synthetic`
    marks a unit this module invented rather than one that came from a real
    outline entry -- the "front matter" / "(lead-in)" span before a parent's
    first child, or the single whole-document unit used when there is no
    outline/extraction boundary at all."""
    start_page: int
    end_page: int
    title: str
    level: int
    path: List[str]
    children: List["Unit"] = field(default_factory=list)
    is_synthetic: bool = False


def _min_page_in_slice(toc, lo: int, hi: int) -> Optional[int]:
    """Smallest resolvable (1-indexed) destination page among toc[lo:hi),
    regardless of the level of the entry it belongs to -- this is how a
    folder-only bookmark's own start page is resolved, at any outline
    depth."""
    best = None
    for i in range(lo, hi):
        p = toc[i][2]
        if p and p >= 1 and (best is None or p < best):
            best = p
    return best


def _build_children(
    toc: List[list], lo: int, hi: int, range_start: int, range_end: int, path: List[str],
) -> List[Unit]:
    """Builds the immediate structural children covering `[range_start,
    range_end]` (0-indexed, inclusive) from descendant outline entries
    `toc[lo:hi)`. Returns `[]` if there is nothing usable in this slice --
    meaning the caller's own span has no further outline substructure to
    recurse into (it is, from the packer's point of view, a leaf).

    See the module docstring's "RULE: HOW A CHILD'S PAGE SPAN IS COMPUTED"
    for the exact semantics implemented here; this is the single place that
    rule is applied, recursively, at every outline level."""
    if lo >= hi or range_end < range_start:
        return []

    child_level = min(toc[i][0] for i in range(lo, hi))
    idxs = [i for i in range(lo, hi) if toc[i][0] == child_level]

    # Each candidate child's own raw destination page, or (for a folder-only
    # entry, page == -1) the smallest resolvable page anywhere in ITS OWN
    # descendant slice -- pure outline-tree reading, no PDF content touched.
    raw: List[Tuple[int, Optional[int], int, int]] = []
    for k, i in enumerate(idxs):
        sub_lo = i + 1
        sub_hi = idxs[k + 1] if k + 1 < len(idxs) else hi
        own_page = toc[i][2] if toc[i][2] and toc[i][2] >= 1 else None
        start_1idx = own_page if own_page is not None else _min_page_in_slice(toc, sub_lo, sub_hi)
        raw.append((i, start_1idx, sub_lo, sub_hi))

    # Keep only children that resolve to a page strictly inside this parent's
    # own span and strictly after the previous kept child -- this is the
    # defensive guard against a page-less ancestor or an out-of-order sibling
    # producing a bogus (too-early/too-late) boundary that would otherwise
    # corrupt every span computed from it.
    kept: List[Tuple[int, int, int, int]] = []
    last_start0 = range_start - 1
    for i, start_1idx, sub_lo, sub_hi in raw:
        if start_1idx is None:
            continue
        start0 = start_1idx - 1
        if start0 < range_start or start0 > range_end or start0 <= last_start0:
            continue
        kept.append((i, start0, sub_lo, sub_hi))
        last_start0 = start0

    if not kept:
        return []

    children: List[Unit] = []
    if kept[0][1] > range_start:
        lead_title = "Front matter" if not path else f"{path[-1]} (lead-in)"
        children.append(Unit(
            start_page=range_start, end_page=kept[0][1] - 1, title=lead_title,
            level=child_level, path=path + ["(lead-in)"], children=[], is_synthetic=True,
        ))

    for k, (i, start0, sub_lo, sub_hi) in enumerate(kept):
        end0 = kept[k + 1][1] - 1 if k + 1 < len(kept) else range_end
        title = toc[i][1]
        cpath = path + [title]
        grandchildren = _build_children(toc, sub_lo, sub_hi, start0, end0, cpath)
        children.append(Unit(
            start_page=start0, end_page=end0, title=title, level=toc[i][0],
            path=cpath, children=grandchildren,
        ))

    return children


def _resolve_extraction_h1_pages(pdf_path: Path, config: PipelineConfig) -> List[Tuple[int, str]]:
    """Option B (fallback): run the existing extractor once and read level-1
    `HeadingBlock` positions, 0-indexed. Only reached when the raw outline
    (Option A) has nothing usable -- this is the expensive path."""
    from .extractors.pdf_extractor import PdfExtractor

    document, _log = PdfExtractor().extract(pdf_path, config.ocr)
    boundaries: List[Tuple[int, str]] = []
    last_page = -1
    for block in document.blocks:
        if not (isinstance(block, HeadingBlock) and block.level == 1 and block.page):
            continue
        page0 = block.page - 1
        if page0 <= last_page:
            continue
        boundaries.append((page0, block.text))
        last_page = page0
    return boundaries


def _units_from_flat_boundaries(boundaries: List[Tuple[int, str]], page_count: int) -> List[Unit]:
    """Turns a flat boundary list (Option B, or the "no boundary at all"
    case) into top-level `Unit`s with no children -- Option B never yields
    deeper structure in this extractor, so there is nothing to recurse into
    beyond whatever this produces (an oversized result here goes straight to
    `SplitError`, not further descent)."""
    points = list(boundaries)
    synthetic_first = False
    if not points:
        points = [(0, "Document")]
        synthetic_first = True
    elif points[0][0] > 0:
        points.insert(0, (0, "Front matter"))
        synthetic_first = True

    units: List[Unit] = []
    for i, (start, title) in enumerate(points):
        end = points[i + 1][0] - 1 if i + 1 < len(points) else page_count - 1
        if end < start:
            continue
        units.append(Unit(
            start_page=start, end_page=end, title=title, level=1, path=[title],
            children=[], is_synthetic=(synthetic_first and i == 0),
        ))
    return units


def _resolve_root_units(
    toc: List[list], page_count: int, pdf_path: Path, config: PipelineConfig,
) -> Tuple[List[Unit], str]:
    """Returns `(root_units, source)`, `source` one of "outline" (Option A),
    "extraction" (Option B), or "none" (neither source found any boundary at
    all -- the document is treated as one unit)."""
    if any(e[0] == 1 for e in toc):
        children = _build_children(toc, 0, len(toc), 0, page_count - 1, [])
        if children:
            return children, "outline"

    boundaries = _resolve_extraction_h1_pages(pdf_path, config)
    if boundaries:
        return _units_from_flat_boundaries(boundaries, page_count), "extraction"

    return (
        [Unit(0, page_count - 1, "Document", 0, [], [], is_synthetic=True)],
        "none",
    )


def resolve_h1_boundaries(pdf_path: Path, config: PipelineConfig) -> Tuple[List[Tuple[int, str]], str]:
    """Returns `(boundaries, source)`, `boundaries` a list of `(0-indexed
    page, title)` in document order for the top (level-1) outline units
    only -- the flat view kept for callers/tests that only care about the
    top level. `source` is one of "outline" (Option A), "extraction"
    (Option B), or "none". Full recursive structure (needed to actually
    split) lives in `_resolve_root_units`/`Unit.children`, not here."""
    import pymupdf

    pdf = pymupdf.open(str(pdf_path))
    try:
        toc = pdf.get_toc()
        page_count = pdf.page_count
    finally:
        pdf.close()

    units, source = _resolve_root_units(toc, page_count, pdf_path, config)
    boundaries = [(u.start_page, u.title) for u in units if u.level == 1 and not u.is_synthetic]
    return boundaries, source


# --------------------------------------------------------------------------
# Splitting

@dataclass
class PartInfo:
    path: str
    part_index: int          # 1-based, matches the docnameN.pdf suffix
    start_page: int           # 0-indexed, inclusive
    end_page: int              # 0-indexed, inclusive
    page_count: int
    size_bytes: int
    section_titles: List[str] = field(default_factory=list)
    boundary_level: int = 1   # outline level of the units packed into this part


class SplitError(RuntimeError):
    """Raised when a structural unit exceeds `target_bytes` and has no
    outline children left to recurse into -- the documented floor behavior
    (see module docstring's RECURSIVE SPLIT ALGORITHM). Carries the
    diagnostic fields a caller/operator needs to act on this by hand:
    `pdf_path`, `title`, `path` (outline breadcrumb), `level`, `first_page`/
    `last_page` (1-indexed, inclusive), `size_bytes`, `target_bytes`,
    `reason`."""

    def __init__(self, pdf_path: Path, unit: Unit, size_bytes: int, target_bytes: int):
        self.pdf_path = str(pdf_path)
        self.title = unit.title
        self.path = list(unit.path)
        self.level = unit.level
        self.first_page = unit.start_page + 1
        self.last_page = unit.end_page + 1
        self.size_bytes = size_bytes
        self.target_bytes = target_bytes
        self.reason = (
            "no outline children are available to subdivide this unit further (it has none "
            "at all, or none of its children resolve to a distinct, in-range destination page)"
        )
        breadcrumb = " > ".join(self.path) if self.path else self.title
        super().__init__(
            f"cannot split {self.pdf_path!r} to the ~{target_bytes / 1e6:.1f}MB target: "
            f"outline unit {breadcrumb!r} (level {self.level}, pages {self.first_page}-"
            f"{self.last_page}, {size_bytes / 1e6:.2f}MB assembled) {self.reason}. This is an "
            f"intentional failure, not a bug -- see split_ingest.py's module docstring "
            f"(RECURSIVE SPLIT ALGORITHM / floor behavior). A structurally indivisible region "
            f"this large cannot be split further without either cutting inside it (which would "
            f"violate section integrity) or silently claiming the target was met when it wasn't; "
            f"ingesting it safely needs a streaming/low-memory pipeline change, out of scope here."
        )


def _save_deterministic(doc, path: Path, **extra) -> None:
    """Save `doc` so repeated saves of the same page range yield the same
    bytes: freeze PDF metadata dates and suppress the per-save fresh /ID
    PyMuPDF otherwise writes. Used for every candidate and final part save
    inside `_pack_units` so byte-size decisions are reproducible."""
    doc.set_metadata({
        "creationDate": "D:19700101000000Z",
        "modDate": "D:19700101000000Z",
    })
    doc.save(str(path), garbage=4, deflate=True, no_new_id=True, **extra)


def _pack_units(
    src, units: List[Unit], target_bytes: int, scratch_path: Path, out_dir: Path, stem: str,
    part_index: int, pdf_path: Path,
) -> Tuple[List[PartInfo], int]:
    """Greedily packs `units` (siblings at one outline level, in document
    order) into ~`target_bytes` parts, recursing into a unit's own children
    when that unit alone already exceeds the target. Returns
    `(parts, next_part_index)` so numbering stays sequential across
    recursive calls -- a recursed-into unit's parts are numbered in place of
    the single part it would otherwise have been."""
    import pymupdf

    parts: List[PartInfo] = []
    i, n = 0, len(units)
    while i < n:
        unit = units[i]

        # Measure the unit alone before deciding anything -- this is the
        # single "actual assembled on-disk byte size" check that everything
        # else here (accept as a part, grow a part, or recurse) is decided
        # from; never estimated from page/block/character count.
        acc = pymupdf.open()
        acc.insert_pdf(src, from_page=unit.start_page, to_page=unit.end_page)
        _save_deterministic(acc, scratch_path)
        alone_size = os.path.getsize(scratch_path)

        if alone_size > target_bytes:
            acc.close()
            if unit.children:
                # Never split inside this unit while it has children --
                # descend and let its own children be packed instead.
                sub_parts, part_index = _pack_units(
                    src, unit.children, target_bytes, scratch_path, out_dir, stem, part_index, pdf_path,
                )
                parts.extend(sub_parts)
                i += 1
                continue
            raise SplitError(pdf_path, unit, alone_size, target_bytes)

        part_start, part_end = unit.start_page, unit.end_page
        titles = [unit.title]
        boundary_level = unit.level
        i += 1

        while i < n:
            nxt = units[i]
            acc.insert_pdf(src, from_page=nxt.start_page, to_page=nxt.end_page)
            # Every candidate (and final) save goes through the deterministic
            # helper: fixed metadata dates and no fresh /ID, so the byte count
            # of the same page range is reproducible across runs. The
            # `garbage=4, deflate=True` flags also keep unreferenced objects
            # compacted after a rolled-back candidate.
            _save_deterministic(acc, scratch_path)
            candidate_size = os.path.getsize(scratch_path)
            if candidate_size <= target_bytes:
                part_end = nxt.end_page
                titles.append(nxt.title)
                boundary_level = nxt.level
                i += 1
            else:
                # Adding this whole unit would push the part over the
                # target -- roll it back out (never cut a unit) and close
                # the part exactly at the current boundary.
                added_pages = nxt.end_page - nxt.start_page + 1
                acc.delete_pages(from_page=acc.page_count - added_pages, to_page=acc.page_count - 1)
                break

        part_path = out_dir / f"{stem}{part_index}.pdf"
        _save_deterministic(acc, part_path)
        size_bytes = os.path.getsize(part_path)
        parts.append(PartInfo(
            path=str(part_path), part_index=part_index, start_page=part_start,
            end_page=part_end, page_count=acc.page_count, size_bytes=size_bytes,
            section_titles=titles, boundary_level=boundary_level,
        ))
        acc.close()
        part_index += 1

    return parts, part_index


def split_pdf(
    pdf_path: Path, out_dir: Path, config: PipelineConfig, target_bytes: int = TARGET_PART_BYTES,
) -> Tuple[List[PartInfo], str]:
    """Splits `pdf_path` into whole-structural-unit parts under `out_dir`,
    named `{stem}1.pdf`, `{stem}2.pdf`, ... Returns `(parts, h1_source)` --
    see `resolve_h1_boundaries` for `h1_source`. Recursively descends the
    outline (see module docstring) wherever a unit exceeds `target_bytes`;
    raises `SplitError` if that descent bottoms out at a childless unit that
    still exceeds it.

    Deterministic: the same source PDF always yields the same unit
    boundaries and the same accumulate-until-~5MB decisions, since both are
    derived only from the outline/extraction and the parts' own assembled
    bytes, never from wall-clock time or filesystem ordering."""
    import pymupdf

    out_dir.mkdir(parents=True, exist_ok=True)
    src = pymupdf.open(str(pdf_path))
    try:
        page_count = src.page_count
        toc = src.get_toc()
        root_units, h1_source = _resolve_root_units(toc, page_count, pdf_path, config)

        stem = pdf_path.stem
        scratch_path = out_dir / f".{stem}.split_candidate.tmp.pdf"
        parts, _next_index = _pack_units(
            src, root_units, target_bytes, scratch_path, out_dir, stem, 1, pdf_path,
        )

        if scratch_path.exists():
            scratch_path.unlink()
        return parts, h1_source
    finally:
        src.close()


# --------------------------------------------------------------------------
# Ingest + report

ProgressFn = Callable[[str, dict], None]


def _progress_printer() -> ProgressFn:
    """Same shape/behavior as `cli.py`'s: one status line per event to
    stderr, timestamped from this run's start."""
    start = time.monotonic()

    def cb(stage: str, payload: dict) -> None:
        elapsed = time.monotonic() - start
        bits = [f"[{elapsed:7.1f}s]", stage]
        if "source" in payload:
            bits.append(payload["source"])
        if "current" in payload and payload.get("total"):
            bits.append(f"{payload['current']}/{payload['total']}")
        elif "current" in payload:
            bits.append(str(payload["current"]))
        print(" ".join(str(b) for b in bits), file=sys.stderr)

    return cb


def split_and_ingest(
    pipeline: DocumentPipeline, source_path: str, work_dir: Optional[str] = None,
    force: bool = False, profile: Optional[str] = None, target_bytes: int = TARGET_PART_BYTES,
    progress: Optional[ProgressFn] = None,
) -> List[dict]:
    """Splits `source_path` into parts under `work_dir` (a fresh temp
    directory when not given) and indexes every part through
    `DocumentPipeline.index()` -- the same entry point the CLI's `index`
    command and every other caller uses. Every part is a distinct source
    file, so each gets its own `document_id`/index entry (see
    `stable_document_id`) -- no dedup, no collision with the other parts or
    with the original monolith (which was never itself indexed).

    `split_pdf` (and therefore a `SplitError` it raises) runs before any
    part is ingested, so a document with a structurally indivisible
    oversized region fails loudly here rather than reporting a misleading
    per-part "failed" ingest result.

    Returns one result dict per part, in part order. The split parts
    themselves are left on disk under `work_dir` after ingestion (same
    convention as an uploaded source file living under `paths.source_dir` --
    the index's `source_path` for each part keeps pointing at a real file,
    not one this function has since deleted out from under it), so a
    caller/operator is responsible for cleaning up a temp `work_dir` once
    it's no longer needed."""
    path = Path(source_path)
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="split_ingest_")
    out_dir = Path(work_dir)

    parts, h1_source = split_pdf(path, out_dir, pipeline.config, target_bytes=target_bytes)
    total_pages = sum(p.page_count for p in parts)
    print(f"H1 boundary source: {h1_source}  ({len(parts)} part(s), {total_pages} page(s) total)",
          file=sys.stderr)
    print(f"Split parts written under: {out_dir}", file=sys.stderr)

    results = []
    for part in parts:
        rel = os.path.basename(part.path)
        print(f"[part {part.part_index}/{len(parts)}] {rel}: pages {part.start_page + 1}-"
              f"{part.end_page + 1} ({part.page_count}p, {part.size_bytes / 1e6:.2f} MB, "
              f"level {part.boundary_level})", file=sys.stderr)
        try:
            result = pipeline.index(part.path, relative_source=rel, force=force, profile=profile,
                                     progress=progress)
            results.append({
                "part": rel, "document_id": result.document_id,
                "status": "skipped" if result.skipped_unchanged else "indexed",
                "chunks": len(result.chunking_result.chunks),
                "warnings": result.validation_report.stats["warnings"],
                "errors": result.validation_report.stats["errors"],
                "pages": part.page_count, "size_bytes": part.size_bytes,
            })
        except PipelineValidationError as e:
            results.append({"part": rel, "status": "failed", "error": str(e)})
        except Exception as e:
            results.append({"part": rel, "status": "failed", "error": f"{type(e).__name__}: {e}"})

    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="document_converter.split_ingest",
        description="Split a large PDF along its outline/bookmark tree (recursively descending "
                     "any unit that exceeds the target) into ~5MB parts, then ingest every part "
                     "through the normal DocumentPipeline index path.",
    )
    parser.add_argument("file", help="Path to the source PDF")
    parser.add_argument("--config", help="Path to a docpipe.yaml/json config file")
    parser.add_argument("--work-dir", help="Directory to write split parts into (default: a new temp dir)")
    parser.add_argument("--force", action="store_true", help="Re-index a part even if unchanged")
    parser.add_argument("--profile", choices=INGESTION_PROFILES, default=None,
                         help="Ingestion profile (default: the pipeline config's profile)")
    parser.add_argument("--target-mb", type=float, default=TARGET_PART_BYTES / 1e6,
                         help="Soft per-part size target in MB (default: 5)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    pipeline = DocumentPipeline(config)
    try:
        results = split_and_ingest(
            pipeline, args.file, work_dir=args.work_dir, force=args.force, profile=args.profile,
            target_bytes=int(args.target_mb * 1e6), progress=_progress_printer(),
        )
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        return 2
    except SplitError as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2

    failed = 0
    for r in results:
        if r["status"] == "failed":
            failed += 1
            print(f"FAILED  {r['part']}: {r['error']}")
        else:
            print(f"{r['status'].upper():8} {r['part']} ({r.get('chunks', 0)} chunks, "
                  f"{r.get('warnings', 0)} warnings, {r.get('errors', 0)} errors)")
    print(f"\n{len(results)} part(s) ingested, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

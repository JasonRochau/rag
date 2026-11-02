"""
Visual region locator for the VISUAL_ANALYSIS pipeline (spec "Phase 2").

Deliberately NOT a second document parser: it never builds a `Document`,
never emits `Block`s, and never participates in MAIN_TEXT filtering,
normalization, semantic structuring, or chunking. It is a small companion
step that opens the *same source PDF* a second time with the *same*
PyMuPDF/`pymupdf` library `pdf_extractor.py` already depends on
(`_build_figure_context`'s `pdf[pno].get_image_info()` call, and
`_extract_tables`'s `page.find_tables()` call), and turns two existing
PyMuPDF page-geometry signals into stable, croppable visual regions --
reusing existing extraction signals rather than inventing a parallel
extraction pipeline:

RASTER visuals -- `page.get_image_info()` bounding boxes (unchanged from
the original phase).

VECTOR-DRAWN visuals (added 2026-08-13 correction phase) -- a chart drawn
as PDF vector primitives (lines/rects/curves with no embedded bitmap, e.g.
an exported matplotlib/plotly chart with no raster fallback) has no
`get_image_info()` signal at all. `page.get_drawings()` exposes those
primitives directly; this module spatially groups nearby/intersecting
primitives into candidate clusters and applies conservative, multi-signal
filtering (primitive count, bounding-box area, shape/orientation
diversity -- never one brittle magic number) before treating a cluster as
a real chart/diagram rather than a table border, page separator,
underline, or other decorative furniture. See `VisualConfig`'s
`vector_*` fields for the exact thresholds and their reasoning.

Scope for this phase: PDF only (the corpus signal from Phase 1 inspection --
every project PDF with real figures/charts/screenshots exposes them as
raster `get_image_info()` bounding boxes or PDF vector-drawing primitives;
DOCX/HTML have no equivalent page-geometry/bounding-box source yet).
`extract_visual_regions` on a non-PDF path returns an empty list rather
than guessing.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

try:
    import pymupdf
except ImportError:  # pragma: no cover - declared dependency, already used by pdf_extractor.py
    pymupdf = None

from .config import VisualConfig


@dataclass
class VisualRegion:
    """One eligible visual region located on a page, with everything needed
    to crop it, cache its analysis, and cite it back to the source PDF."""
    page: int  # 1-indexed, matches Block.page / ChunkRecord.source_pages elsewhere
    bbox: List[float]  # [x0, y0, x1, y1] in PDF point space, page-relative
    order: int  # stable document-order key (page * 100000 + index-on-page)
    width_pt: float
    height_pt: float
    area_fraction: float  # of the page's total area
    image_bytes: bytes = field(repr=False)  # rendered PNG crop
    image_hash: str = ""  # sha256 of image_bytes, computed in __post_init__
    # Detection provenance (spec "vector visual detection requirement"):
    # "raster" (embedded bitmap XObject, `get_image_info()`) or "vector"
    # (grouped PDF drawing primitives, `get_drawings()`). Defaults to
    # "raster" so every pre-existing call site/test that never set this
    # keeps its exact prior behavior.
    source: str = "raster"
    # Number of individual vector-drawing paths grouped into this region --
    # `None` for raster regions, an evidence field for vector ones (used in
    # acceptance reporting and available for future tuning).
    vector_primitive_count: Optional[int] = None

    def __post_init__(self):
        if not self.image_hash:
            self.image_hash = hashlib.sha256(self.image_bytes).hexdigest()


class VisualSkipReason:
    # Covers both "too small to be a meaningful figure" and "a thin
    # decorative separator/rule line" -- a separator is, by construction,
    # thin along one axis, which `min_dimension_pt` already catches (a
    # rule line's short dimension is normally 1-3pt, far under any
    # reasonable minimum-figure-dimension threshold), so this is one
    # reason code rather than two branches that can never actually diverge
    # at the default configuration.
    TOO_SMALL = "visual_skipped_non_retrieval_too_small"
    NESTED_DUPLICATE = "visual_skipped_non_retrieval_nested_duplicate"
    BUDGET = "visual_skipped_budget"
    # Vector-specific skip reasons (spec "vector visual detection
    # requirement" / "VECTOR REGION FILTERING").
    VECTOR_TOO_SMALL = "visual_skipped_vector_too_small"
    VECTOR_TOO_SPARSE = "visual_skipped_vector_too_sparse"
    VECTOR_LOW_DIVERSITY = "visual_skipped_vector_low_diversity"
    VECTOR_TABLE_LIKE = "visual_skipped_vector_table_like"
    VECTOR_GRID_LIKE = "visual_skipped_vector_grid_like"
    VECTOR_TEXT_LIKE = "visual_skipped_vector_text_like"
    VECTOR_PAGE_TOO_DENSE = "visual_skipped_vector_page_too_dense"
    # Cross-source dedup: a vector cluster that overlaps an already-accepted
    # raster region on the same page describes the same source figure (spec
    # "OVERLAP / DEDUPLICATION") -- the raster crop is kept since it already
    # contains the real pixel content.
    DUPLICATE_OF_RASTER = "visual_skipped_duplicate_of_raster"


def _boxes_overlap_contained(a: List[float], b: List[float], tol: float = 2.0) -> bool:
    """True when box `a` is fully contained in (or equal to) box `b`,
    within `tol` points -- used to collapse the nested-bbox duplicates
    PyMuPDF sometimes reports for one logical figure (an outer frame image
    and an inner content image at nearly the same bbox; observed on real
    corpus pages during Phase 1 inspection, e.g. two-bbox pairs at
    (51,462,309,715)/(52,144,306,387)-shaped nesting)."""
    return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol


def _boxes_intersect(a: List[float], b: List[float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _box_area(b: List[float]) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def _iou(a: List[float], b: List[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _dedupe_nested(boxes: List[dict]) -> List[dict]:
    """Keeps only the largest box in each nested/near-duplicate cluster,
    largest-first so a later, smaller, contained box is dropped in favor of
    the one already kept."""
    ordered = sorted(boxes, key=lambda i: -(i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1]))
    kept: List[dict] = []
    for info in ordered:
        b = info["bbox"]
        if any(_boxes_overlap_contained(b, k["bbox"]) for k in kept):
            continue
        kept.append(info)
    return kept


# --------------------------------------------------------------------------- #
# Vector-drawn visual detection (2026-08-13 correction phase).
#
# page -> page.get_drawings() (existing PyMuPDF page geometry API, the same
#         page object pdf_extractor.py already opens)
#      -> spatially group nearby/intersecting drawing primitives (union-find
#         over "expanded bbox intersects")
#      -> per-cluster signals: primitive count, bounding-box area fraction,
#         shape/orientation diversity
#      -> conservative multi-signal acceptance rule (never one magic number)
#      -> suppress clusters that are mostly inside a PyMuPDF-detected table
#         region (page.find_tables(), the same signal pdf_extractor.py's
#         _extract_tables already uses)
#      -> render the accepted bounding region (padded for nearby labels)
#         from the original PDF page to a bitmap crop, exactly like a
#         raster region is rendered
# --------------------------------------------------------------------------- #

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _drawing_bbox(d: dict) -> Optional[List[float]]:
    r = d.get("rect")
    if r is None:
        return None
    bbox = [float(r.x0), float(r.y0), float(r.x1), float(r.y1)]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        # Degenerate (zero-width/height) rect -- e.g. a single point-like
        # drawing op with no extent. Not a useful geometric primitive on
        # its own; give it a tiny nonzero extent so it can still be grouped
        # with real neighbors rather than crashing area math, but it will
        # never itself satisfy an area-fraction threshold.
        bbox[2] = max(bbox[2], bbox[0] + 0.01)
        bbox[3] = max(bbox[3], bbox[1] + 0.01)
    return bbox


def _normalize_bbox(bbox: List[float]) -> List[float]:
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        bbox = list(bbox)
        bbox[2] = max(bbox[2], bbox[0] + 0.01)
        bbox[3] = max(bbox[3], bbox[1] + 0.01)
    return bbox


def _item_bbox(it: tuple) -> Optional[List[float]]:
    """Bounding box of one drawing sub-item (one `re`/`l`/`c`/`qu` op).
    `PyMuPDF` can bundle several such ops into a single `get_drawings()`
    path/drawing entry (e.g. one `finish()` call covering several bars),
    so grouping/signal computation must work at this item granularity, not
    the enclosing drawing's own bbox -- otherwise a bundled multi-shape
    path collapses into one artificially "blob-shaped" primitive and
    defeats the shape-diversity/thin-primitive signals below."""
    op = it[0]
    try:
        if op == "re":
            r = it[1]
            return _normalize_bbox([float(r.x0), float(r.y0), float(r.x1), float(r.y1)])
        if op == "l" and len(it) >= 3:
            p1, p2 = it[1], it[2]
            xs, ys = (p1.x, p2.x), (p1.y, p2.y)
            return _normalize_bbox([min(xs), min(ys), max(xs), max(ys)])
        if op == "c" and len(it) >= 5:
            pts = it[1:5]
            xs, ys = [p.x for p in pts], [p.y for p in pts]
            return _normalize_bbox([min(xs), min(ys), max(xs), max(ys)])
        if op == "qu" and len(it) >= 2:
            q = it[1]
            pts = [q.ul, q.ur, q.ll, q.lr]
            xs, ys = [p.x for p in pts], [p.y for p in pts]
            return _normalize_bbox([min(xs), min(ys), max(xs), max(ys)])
    except (AttributeError, IndexError, TypeError):
        return None
    return None


def _item_kind(it: tuple) -> str:
    """Classifies one drawing sub-item into a coarse geometric "flavor"
    (horizontal/vertical/diagonal line, curve, rect, other). Used for the
    shape/orientation-diversity signal: a real chart/diagram almost always
    mixes flavors (axes vs. bars/points/nodes) even when small, while a
    run of publication furniture (ruled table lines, repeated decorative
    rules) is almost always one flavor repeated many times."""
    op = it[0]
    if op == "re":
        return "rect"
    if op == "l" and len(it) >= 3:
        p1, p2 = it[1], it[2]
        dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
        if dx < 1e-6 and dy < 1e-6:
            return "other"
        if dy <= dx * 0.2:
            return "h"
        if dx <= dy * 0.2:
            return "v"
        return "diag"
    if op in ("c", "qu"):
        return "curve"
    return "other"


def _is_thin_primitive(bbox: List[float], aspect_ratio: float) -> bool:
    """True when `bbox`'s long side is at least `aspect_ratio` times its
    short side -- a stroked line/curve segment (axis, gridline, bar edge,
    plotted-line segment) rather than a compact, roughly-square blob (a
    glyph outline, a node/marker shape). See `VisualConfig
    .vector_thin_primitive_aspect_ratio`'s docstring for why this is the
    key signal that separates real chart/diagram geometry from PDF pages
    that render body text as vector glyph outlines."""
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    short, long = (w, h) if w <= h else (h, w)
    return long >= short * aspect_ratio if short > 1e-6 else True


def _group_drawings_by_proximity(bboxes: List[List[float]], gap_pt: float) -> List[List[int]]:
    """Groups drawing primitives that belong to one visual (axis lines,
    bars, tick marks, node shapes) even though their individual bounding
    boxes are rarely pixel-touching -- conceptually "union any two
    primitives within `gap_pt` of each other", i.e. connected components
    over a proximity graph.

    Naive pairwise bbox comparison is O(n^2), which is fine for a typical
    page (tens to low hundreds of drawings) but infeasible for a page
    whose chart was exported as thousands of tiny vector line segments (a
    real corpus PDF, `Performance-comparison-of-two-virtual-machine-
    scenarios...pdf`, was observed with a genuine high-resolution
    vector-rendered line/scatter chart rendering as 36k-74k drawings on a
    single page -- an O(n^2) pass there does not finish in practical time).

    Instead this inserts each primitive's `gap_pt`-expanded bbox into
    every grid cell (sized `gap_pt`) it overlaps, and unions all
    primitives that ever land in the same cell (chained to the first
    primitive seen there -- no pairwise checks within a cell). Two
    primitives sharing a cell are guaranteed to be within about
    `2 * gap_pt` of each other, which is what "nearby" means here (a
    candidate *region*, not pixel-exact adjacency). Critically, inserting
    by *footprint* rather than by center point (an earlier, faster-seeming
    approach this was changed from) is what makes this correct for
    elongated primitives: an axis line spanning most of a chart's height
    has a center point far from either of its own endpoints, so
    center-only bucketing missed real adjacency between an axis and a
    bar/point near just one end of it (confirmed against a real synthetic
    bar-chart-shaped fixture during acceptance testing). A long primitive
    simply touches more cells along its length; the number of primitives
    long enough for that to matter is always small in practice (a
    handful of axes/gridlines per chart), so this stays close to O(n +
    cells) even though a few primitives contribute more than one cell."""
    n = len(bboxes)
    if n == 0:
        return []
    uf = _UnionFind(n)
    cell_size = max(gap_pt, 1.0)
    # Caps how many cells one single (typically pathological/degenerate)
    # primitive can register into, so one absurdly long primitive can
    # never make this step itself O(page dimension) per item in a way
    # that adds up -- 800 cells at `cell_size=gap_pt` already covers any
    # primitive spanning a full US Letter/A4 page at any plausible gap_pt.
    max_span_cells = 800
    grid_owner: Dict["tuple[int, int]", int] = {}
    for i, b in enumerate(bboxes):
        ex0, ey0, ex1, ey1 = b[0] - gap_pt, b[1] - gap_pt, b[2] + gap_pt, b[3] + gap_pt
        cx0, cy0 = int(ex0 // cell_size), int(ey0 // cell_size)
        cx1 = min(int(ex1 // cell_size), cx0 + max_span_cells)
        cy1 = min(int(ey1 // cell_size), cy0 + max_span_cells)
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                key = (cx, cy)
                owner = grid_owner.get(key)
                if owner is None:
                    grid_owner[key] = i
                else:
                    uf.union(i, owner)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def _cluster_bbox(indices: List[int], bboxes: List[List[float]]) -> List[float]:
    xs0 = [bboxes[i][0] for i in indices]
    ys0 = [bboxes[i][1] for i in indices]
    xs1 = [bboxes[i][2] for i in indices]
    ys1 = [bboxes[i][3] for i in indices]
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def _table_bboxes(page) -> List[List[float]]:
    """Reuses PyMuPDF's own table finder (the same signal
    `pdf_extractor.py`'s `_extract_tables` already uses) purely as a
    suppression signal for vector-cluster false positives -- never a
    second table extractor, never wired into MAIN_TEXT."""
    try:
        finder = page.find_tables()
        return [[float(t.bbox[0]), float(t.bbox[1]), float(t.bbox[2]), float(t.bbox[3])] for t in finder.tables]
    except Exception:
        return []


def _table_overlap_fraction(bbox: List[float], table_bboxes: List[List[float]]) -> float:
    area = _box_area(bbox)
    if area <= 0 or not table_bboxes:
        return 0.0
    covered = 0.0
    for t in table_bboxes:
        ix0, iy0 = max(bbox[0], t[0]), max(bbox[1], t[1])
        ix1, iy1 = min(bbox[2], t[2]), min(bbox[3], t[3])
        covered += max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    return min(covered / area, 1.0)


def _is_meaningful_vector_cluster(
    primitive_count: int, width: float, height: float, area_fraction: float,
    shape_diversity: int, thin_fraction: float, h_count: int, v_count: int, cfg: VisualConfig,
    page_has_sparse_text: bool = False,
) -> "tuple[bool, Optional[str]]":
    """Conservative, multi-signal acceptance rule (spec "VECTOR REGION
    FILTERING": "do not use one brittle magic number"). Every threshold
    used here is a documented `VisualConfig.vector_*` field."""
    if width < cfg.min_dimension_pt or height < cfg.min_dimension_pt:
        return False, VisualSkipReason.VECTOR_TOO_SMALL
    if area_fraction < cfg.vector_min_area_fraction:
        return False, VisualSkipReason.VECTOR_TOO_SMALL
    if primitive_count < cfg.vector_min_primitives_relaxed:
        return False, VisualSkipReason.VECTOR_TOO_SPARSE
    if primitive_count < cfg.vector_min_primitives and shape_diversity < 2:
        return False, VisualSkipReason.VECTOR_TOO_SPARSE
    if shape_diversity < 2 and primitive_count < cfg.vector_high_count_single_orientation:
        return False, VisualSkipReason.VECTOR_LOW_DIVERSITY
    # Ruled-grid suppression (see `VisualConfig.vector_grid_min_line_count`'s
    # docstring): many horizontal AND vertical lines, few/no other shapes
    # -- a bare table/ruling grid `page.find_tables()` didn't itself catch,
    # not a chart (a real chart's axes are normally exactly one line per
    # direction; the rest of its primitives are its actual content --
    # bars/points/curves -- not more grid lines).
    if (min(h_count, v_count) >= cfg.vector_grid_min_line_count
            and (h_count + v_count) / primitive_count >= cfg.vector_grid_line_fraction):
        return False, VisualSkipReason.VECTOR_GRID_LIKE
    # Distinguishes real chart/diagram geometry (dominated by thin,
    # line-like primitives -- axes, gridlines, plotted-line segments, bar
    # edges) from a page whose "vector drawings" are actually glyph
    # outlines standing in for a missing text layer (dominated by compact,
    # roughly-square blob shapes) -- see `VisualConfig
    # .vector_min_thin_primitive_fraction`'s docstring for the real-corpus
    # evidence behind this threshold.
    if thin_fraction < cfg.vector_min_thin_primitive_fraction:
        return False, VisualSkipReason.VECTOR_TEXT_LIKE
    # Secondary guard for a page whose vector-rendered glyphs happen to
    # clear the thin-primitive bar above (see `VisualConfig
    # .vector_whole_page_area_fraction`'s docstring for the real-corpus
    # evidence): a candidate spanning a large share of a page that itself
    # has (near-)zero extractable text is far more likely to be that
    # page's body/metadata text than one real chart/diagram.
    if page_has_sparse_text and area_fraction >= cfg.vector_whole_page_area_fraction:
        return False, VisualSkipReason.VECTOR_TEXT_LIKE
    return True, None


def _vector_candidates(page, cfg: VisualConfig) -> "tuple[List[dict], List[dict]]":
    """Returns `(accepted, skipped)` candidate dicts
    (`{"bbox", "source": "vector", "vector_primitive_count", ...}`) for one
    page's vector-drawn visual clusters -- never raises; a page whose
    `get_drawings()`/`find_tables()` call fails is treated as having no
    vector candidates, exactly like a raster page with no images."""
    try:
        drawings = page.get_drawings()
    except Exception:
        return [], []
    try:
        page_has_sparse_text = len(page.get_text("text").strip()) < cfg.vector_sparse_page_text_chars
    except Exception:
        page_has_sparse_text = False
    if not drawings:
        return [], []
    if len(drawings) > cfg.vector_max_drawings_per_page:
        return [], [{"bbox": [0.0, 0.0, page.rect.width, page.rect.height],
                      "reason": VisualSkipReason.VECTOR_PAGE_TOO_DENSE}]

    # Decide, per drawing, whether to treat it as ONE primitive (its own
    # enclosing bbox) or split it into its individual sub-items. This
    # matters for two real, opposing failure modes found during
    # acceptance testing:
    #  - A single PyMuPDF path can legitimately bundle several *distinct*
    #    same-style chart elements (e.g. one `finish()` call stroking 5
    #    bar rects, or a run of gridlines) -- treating that whole path as
    #    one primitive collapses real chart content into one artificially
    #    "blob-shaped" primitive and defeats the shape-diversity/
    #    thin-primitive signals below. These must be split.
    #  - A single glyph outline (or one continuous plotted curve) is
    #    normally a *sequence of connected bezier curves tracing one
    #    shape* -- splitting THAT into individual curve segments destroys
    #    the very signal (whole-shape aspect ratio: a glyph is compact/
    #    blob-shaped, a real chart curve/line is normally elongated) that
    #    separates real chart geometry from a page whose "vector drawings"
    #    are actually glyph outlines standing in for a missing text layer
    #    (confirmed on a real corpus PDF -- see `VisualConfig
    #    .vector_min_thin_primitive_fraction`'s docstring). These must stay
    #    bundled.
    # Splitting only when every item in a path is the *same simple op type*
    # (`re`-only or `l`-only) covers the first case (distinct same-style
    # shapes really do get bundled this way -- confirmed on the real
    # corpus PDF this phase's acceptance evidence is built on) without
    # touching curve-bearing paths, which covers the second.
    bboxes: List[List[float]] = []
    kinds: List[str] = []
    for d in drawings:
        items = [it for it in d.get("items", []) if it]
        if not items:
            b = _drawing_bbox(d)
            if b is not None:
                bboxes.append(b)
                kinds.append("other")
            continue
        ops = {it[0] for it in items}
        if len(items) > 1 and ops in ({"re"}, {"l"}):
            for it in items:
                b = _item_bbox(it)
                if b is not None:
                    bboxes.append(b)
                    kinds.append(_item_kind(it))
        else:
            b = _drawing_bbox(d)
            if b is not None:
                bboxes.append(b)
                kinds.append("curve" if ops & {"c", "qu"} else _item_kind(items[0]))
    if not bboxes:
        return [], []
    if len(bboxes) > cfg.vector_max_drawings_per_page:
        return [], [{"bbox": [0.0, 0.0, page.rect.width, page.rect.height],
                      "reason": VisualSkipReason.VECTOR_PAGE_TOO_DENSE}]

    pw, ph = page.rect.width, page.rect.height
    page_area = max(pw * ph, 1.0)
    table_bboxes = _table_bboxes(page)

    accepted: List[dict] = []
    skipped: List[dict] = []
    for indices in _group_drawings_by_proximity(bboxes, cfg.vector_group_gap_pt):
        bbox = _cluster_bbox(indices, bboxes)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        area_fraction = _box_area(bbox) / page_area
        shape_diversity = len({kinds[i] for i in indices})
        primitive_count = len(indices)
        thin_count = sum(1 for i in indices if _is_thin_primitive(bboxes[i], cfg.vector_thin_primitive_aspect_ratio))
        thin_fraction = thin_count / len(indices)
        h_count = sum(1 for i in indices if kinds[i] == "h")
        v_count = sum(1 for i in indices if kinds[i] == "v")

        table_overlap = _table_overlap_fraction(bbox, table_bboxes)
        if table_overlap >= cfg.vector_table_overlap_suppress_fraction:
            skipped.append({"bbox": bbox, "reason": VisualSkipReason.VECTOR_TABLE_LIKE})
            continue

        ok, reason = _is_meaningful_vector_cluster(
            primitive_count, w, h, area_fraction, shape_diversity, thin_fraction, h_count, v_count, cfg,
            page_has_sparse_text=page_has_sparse_text,
        )
        if not ok:
            skipped.append({"bbox": bbox, "reason": reason})
            continue

        accepted.append({
            "bbox": bbox, "source": "vector", "vector_primitive_count": primitive_count,
            "shape_diversity": shape_diversity, "thin_fraction": round(thin_fraction, 4),
        })
    return accepted, skipped


def _dedupe_cross_source(candidates: List[dict], iou_threshold: float) -> "tuple[List[dict], List[dict]]":
    """Spec "OVERLAP / DEDUPLICATION": a page may contain both a raster
    image region and vector-drawing geometry describing the same figure
    (e.g. a chart exported with an embedded raster body and a vector axis
    overlay). Rule: an accepted vector cluster that overlaps an accepted
    raster region (IoU above `iou_threshold`, OR either box contains the
    other) is dropped -- the raster crop already contains the real pixel
    content, so keeping both would index the same source figure twice.
    Raster/raster and vector/vector duplicates are handled separately
    (`_dedupe_nested` for raster; clusters are already disjoint by
    construction for vector). Returns `(kept, dropped_as_duplicate)`."""
    rasters = [c for c in candidates if c["source"] == "raster"]
    vectors = [c for c in candidates if c["source"] == "vector"]
    kept_vectors: List[dict] = []
    dropped: List[dict] = []
    for v in vectors:
        is_dup = any(
            _iou(v["bbox"], r["bbox"]) >= iou_threshold
            or _boxes_overlap_contained(v["bbox"], r["bbox"])
            or _boxes_overlap_contained(r["bbox"], v["bbox"])
            for r in rasters
        )
        if is_dup:
            dropped.append({"bbox": v["bbox"], "reason": VisualSkipReason.DUPLICATE_OF_RASTER})
        else:
            kept_vectors.append(v)
    return rasters + kept_vectors, dropped


def extract_visual_regions(
    pdf_path: str, config: Optional[VisualConfig] = None, crop_dpi: int = 300, crop_pad_pt: float = 8.0,
) -> "tuple[List[VisualRegion], List[dict]]":
    """Locates eligible raster-image AND vector-drawn visual regions in
    `pdf_path` and renders a high-resolution crop for each. Returns
    `(regions, skipped)` -- `skipped` records every candidate region that
    was found but filtered as decorative/non-retrieval/duplicate (spec
    Phase 2's "generic classification rules, no per-PDF hardcoding"), each
    as `{"page", "bbox", "reason"}`, so ingestion can report what was
    intentionally not processed rather than silently dropping it.

    Never raises for a PDF that opens successfully but has no eligible
    visuals -- returns `([], [...])` or `([], [])`. Raises only if the file
    itself cannot be opened as a PDF (the caller treats that as
    `visual_crop_failed` for the whole document, exactly like any other
    per-document extraction failure -- MAIN_TEXT extraction is unaffected
    either way, since this module is never on that path)."""
    if pymupdf is None:
        raise RuntimeError("pymupdf is not installed; cannot locate visual regions")
    if not pdf_path.lower().endswith(".pdf"):
        return [], []
    cfg = config or VisualConfig()

    regions: List[VisualRegion] = []
    skipped: List[dict] = []
    doc = pymupdf.open(pdf_path)
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            page_no_1idx = pno + 1
            pw, ph = page.rect.width, page.rect.height
            page_area = max(pw * ph, 1.0)

            # -- raster candidates (unchanged signal: get_image_info()) --
            try:
                infos = page.get_image_info()
            except Exception:
                infos = []
            raster_candidates: List[dict] = []
            if infos:
                deduped = _dedupe_nested(infos)
                for info in deduped:
                    bbox = list(info["bbox"])
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    area_fraction = (w * h) / page_area
                    if w < cfg.min_dimension_pt or h < cfg.min_dimension_pt or area_fraction < cfg.min_area_fraction:
                        skipped.append({"page": page_no_1idx, "bbox": bbox, "reason": VisualSkipReason.TOO_SMALL})
                        continue
                    raster_candidates.append({"bbox": bbox, "source": "raster"})

            # -- vector candidates (new signal: get_drawings()) --
            vector_accepted, vector_skipped = _vector_candidates(page, cfg)
            for s in vector_skipped:
                skipped.append({"page": page_no_1idx, **s})

            # -- cross-source dedup: one figure, one record --
            combined, dropped_dupes = _dedupe_cross_source(
                raster_candidates + vector_accepted, cfg.vector_dedup_iou_threshold,
            )
            for s in dropped_dupes:
                skipped.append({"page": page_no_1idx, **s})
            if not combined:
                continue

            for idx, cand in enumerate(sorted(combined, key=lambda c: (c["bbox"][1], c["bbox"][0]))):
                bbox = cand["bbox"]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                area_fraction = (w * h) / page_area

                if len(regions) >= cfg.max_visuals_per_document:
                    skipped.append({"page": page_no_1idx, "bbox": bbox, "reason": VisualSkipReason.BUDGET})
                    continue

                pad = cfg.vector_crop_pad_pt if cand["source"] == "vector" else crop_pad_pt
                clip = pymupdf.Rect(
                    max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
                    min(bbox[2] + pad, pw), min(bbox[3] + pad, ph),
                )
                try:
                    pix = page.get_pixmap(dpi=crop_dpi, clip=clip)
                    png_bytes = pix.tobytes("png")
                except Exception:
                    skipped.append({"page": page_no_1idx, "bbox": bbox, "reason": "visual_crop_failed"})
                    continue

                regions.append(VisualRegion(
                    page=page_no_1idx, bbox=bbox, order=page_no_1idx * 100000 + idx,
                    width_pt=w, height_pt=h, area_fraction=area_fraction, image_bytes=png_bytes,
                    source=cand["source"], vector_primitive_count=cand.get("vector_primitive_count"),
                ))
    finally:
        doc.close()

    return regions, skipped


def save_crop(index_dir: str, region: VisualRegion) -> str:
    """Persists `region`'s rendered crop under `{index_dir}/_visuals/` keyed
    by content hash (so byte-identical crops across re-indexing runs share
    one file instead of accumulating duplicates), and returns the path.
    `_visuals` is a subdirectory, not a top-level file in `index_dir` --
    kept out of the `docs/.index/*.json` glob every index-file reader
    (DocStore.reload, eval_retrieval.py) already assumes matches only
    versioned per-document index payloads (see title_overrides.py's
    `_meta` subdirectory for the same reasoning)."""
    visuals_dir = os.path.join(index_dir, "_visuals")
    os.makedirs(visuals_dir, exist_ok=True)
    path = os.path.join(visuals_dir, f"{region.image_hash}.png")
    if not os.path.exists(path):
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(region.image_bytes)
        os.replace(tmp_path, path)
    return path

"""
Single shared configuration for the whole document pipeline.

Loaded once, from (in increasing priority):
  1. built-in defaults
  2. a YAML or JSON config file (DOCPIPE_CONFIG env var, or ./docpipe.yaml /
     ./docpipe.json next to this package if present)
  3. environment variables (highest priority, always win)

This is the *one* place embedding host/model, chunk budgets, strict mode,
OCR mode, upload limits and retrieval budget are defined. ingestion.py,
orchestrator.py and control.py all import this module instead of declaring
their own copies.
"""
from __future__ import annotations

import json
import os
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - PyYAML is a declared dependency
    yaml = None

# ---------------------------------------------------------------------------
# Version constants used in the pipeline fingerprint (section 12.1). Bump
# whichever one changes behavior so previously-indexed documents are
# automatically detected as stale and reprocessed.
# ---------------------------------------------------------------------------
NORMALIZER_VERSION = "1.0"
EXTRACTOR_VERSION = "1.0"
CHUNKER_VERSION = "2.0"
PROVENANCE_FORMAT_VERSION = "1.0"
INDEX_SCHEMA_VERSION = 2
PIPELINE_VERSION = "2.0"

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathsConfig:
    source_dir: str = str(REPO_ROOT / "docs" / "newly_formatted_docs_review" / "sources")
    processed_dir: str = str(REPO_ROOT / "docs" / "newly_formatted_docs_review" / "processed")
    report_dir: str = str(REPO_ROOT / "docs" / "newly_formatted_docs_review" / "reports")
    index_dir: str = str(REPO_ROOT / "docs" / ".index")
    manifest_path: str = str(REPO_ROOT / "manifest.json")
    legacy_docs_dir: str = str(REPO_ROOT / "docs")

    def ensure(self) -> None:
        for p in (self.source_dir, self.processed_dir, self.report_dir, self.index_dir):
            os.makedirs(p, exist_ok=True)


@dataclass
class EmbeddingConfig:
    provider: str = "sentence-transformers-service"
    host: str = os.environ.get("EMBEDDING_HOST", "http://127.0.0.1:11437")
    model: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    timeout_seconds: float = 60.0
    # Chunks are embedded in slices of this size (pipeline.py's `index()`)
    # rather than one `embed_batch()` call over the whole corpus -- a
    # multi-hundred-MB single request body cannot succeed against
    # `timeout_seconds` above on a large document (e.g. 30k chunks).
    batch_size: int = int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))
    # Populated at runtime from the embedding service's /health response.
    dimension: Optional[int] = None
    max_sequence_tokens: Optional[int] = None
    normalized_embeddings: Optional[bool] = None


@dataclass
class SemanticStructuringConfig:
    mode: str = "auto"  # off | auto | always
    host: str = os.environ.get("SEMANTIC_HOST", "http://localhost:11434")
    model: str = os.environ.get("SEMANTIC_MODEL", "")
    temperature: float = 0.0
    timeout_seconds: float = 300.0
    # The document's eligible blocks are sent to the model in batches whose
    # serialized preview JSON stays under this many characters
    # (semantic_structuring._batch_eligible_blocks) rather than in one
    # unbounded request -- a large document's entire block list would
    # otherwise exceed the model's context window. A single block whose own
    # serialized preview exceeds the budget still goes out (alone in its
    # batch); the budget bounds batch size, never individual blocks.
    batch_char_budget: int = int(os.environ.get("SEMANTIC_BATCH_CHAR_BUDGET", "24000"))

    @property
    def enabled(self) -> bool:
        return self.mode in ("auto", "always") and bool(self.model)


@dataclass
class ChunkingConfig:
    hard_max_words: int = 512
    target_words: int = 425
    preferred_paragraph_words: int = 150
    include_provenance_prefix: bool = True
    cross_section_packing: bool = False
    strict: bool = True
    overlap_words: int = 0
    # Non-strict emergency fallback only; never used when strict=True.
    allow_emergency_sentence_split: bool = True

    # Token-native packing targets (embedding-model migration: BGE vs. Qwen
    # ingestion profiles). `None` (the default) preserves the exact legacy
    # behavior every existing caller/test already depends on: token-aware
    # packing derives its effective target from `target_words`/`hard_max_words`
    # proportionally against whatever `max_sequence_tokens` the live embedding
    # service reports (chunker.py's `_pack_blocks`) -- this is what keeps a
    # BGE-configured pipeline (max_sequence_tokens=512) byte-for-byte
    # unchanged. When set (e.g. for a Qwen-configured `EmbeddingClient`,
    # whose real context window is far larger than any one chunk should be),
    # these become the packing target/ceiling directly, still capped by
    # `min(max_tokens, max_sequence_tokens)` so a chunk can never exceed what
    # the live model actually accepts -- `max_tokens` is a safety guard
    # against a pathologically long single section, never a required size a
    # chunk is padded up to (spec: "treat any operational maximum as a
    # safety guard, not a required chunk size").
    target_tokens: Optional[int] = None
    max_tokens: Optional[int] = None

    # Structural over-segmentation gates (spec section 16). A leaf section
    # (no child headings) with fewer than `tiny_section_body_words` words and
    # no atomic block (list/table/code/image) of its own is "tiny". Tiny
    # sections are not individually invalid -- a short real section is fine
    # -- but a document where more than `tiny_section_ratio_limit` of its
    # leaf sections are tiny is exhibiting the "arbitrary fragment promoted
    # to heading" failure mode even when no single heading's text is
    # obviously invalid syntax (e.g. interactive-courseware UI labels like
    # "Show answer" that are grammatically fine short headings but carry no
    # real body content). The ratio gate only applies once there are at
    # least `min_leaf_sections_for_ratio_gates` leaf sections, so a short,
    # legitimately flat document is never penalized.
    tiny_section_body_words: int = 10
    tiny_section_ratio_limit: float = 0.05
    suspicious_heading_ratio_limit: float = 0.02
    min_leaf_sections_for_ratio_gates: int = 10

    # Governs what atomic_split._split_table_block does when a single table
    # row -- with the header repeated -- alone exceeds the effective chunk
    # budget. "split" (default) escalates through a structural column-group/
    # cell-text ladder so the row is still represented, never dropped;
    # "error" fails that block immediately (oversized_atomic_block, naming
    # the row) instead of subdividing it. One switch, set once here, applied
    # uniformly to every table in every document -- no per-document override.
    oversized_row_policy: str = "split"

    def fingerprint_fields(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class OCRConfig:
    mode: str = os.environ.get("OCR_MODE", "auto")  # off | auto | on
    language: str = "eng"
    min_text_chars_per_page: int = 40  # below this -> page looks scanned
    dpi: int = 200


# Visual-analysis schema/prompt version (spec "Phase 5" cache key). Bump
# whenever the grounding prompt or the required JSON schema changes in a
# way that should invalidate previously-cached Qwen3-VL analyses -- never
# reuse a cached analysis produced under a different contract.
#
# Bumped 1 -> 2 (2026-08-13 vector-visual-detection correction phase):
# `visual_analysis.build_retrieval_text`'s output contract changed (it now
# performs token-native, active-embedding-tokenizer-driven field reduction
# in addition to the existing word-budget caps -- see
# `build_retrieval_text_within_budget`), so a cached `retrieval_text` from
# under version "1" is no longer guaranteed to reflect what the current
# code would produce, and in at least one real case (the Bayesian-networks
# dependency diagram, see RAG_BUILD_SOURCE_OF_TRUTH.md) the old cached text
# was the exact oversized text that failed to embed. Bumping forces every
# previously-cached analysis to be re-rendered (not re-analyzed by the VLM
# -- the cache stores the validated JSON analysis too, so only
# `build_retrieval_text_within_budget` reruns) through the new bounded
# path at least once.
VISUAL_ANALYSIS_VERSION = "2"


@dataclass
class VisualConfig:
    """Multimodal visual-analysis pipeline (figures/charts/diagrams/
    screenshots), additive to MAIN_TEXT -- never mixed into it. Disabled by
    default so importing/loading this config never changes existing
    ingestion behavior; the isolated validation run enables it explicitly.
    See document_converter/visual_extraction.py / visual_analysis.py /
    visual_cache.py."""
    enabled: bool = os.environ.get("VISUAL_ANALYSIS_ENABLED", "0") not in ("0", "false", "False", "")
    ollama_host: str = os.environ.get("VISUAL_OLLAMA_HOST", "http://127.0.0.1:11434")
    # Locked model for this phase (see RAG_BUILD_SOURCE_OF_TRUTH.md) --
    # deliberately not auto-selected/benchmarked against alternatives.
    model: str = os.environ.get("VISUAL_MODEL", "qwen3-vl:8b-instruct-q4_K_M")
    timeout_seconds: float = float(os.environ.get("VISUAL_TIMEOUT_SECONDS", "180"))
    analysis_version: str = VISUAL_ANALYSIS_VERSION
    # Eligibility filter (spec Phase 2): a raster image region smaller than
    # either bound below is treated as decorative (logo/icon/separator/page
    # furniture) and never sent to the VLM.
    min_area_fraction: float = 0.006  # of the page's total area
    min_dimension_pt: float = 36.0    # both width and height must clear this
    # Hard safety cap so one pathological document (hundreds of tiny inline
    # glyph images from a bad OCR export, see CSC506 chapter in Phase 1
    # inspection) can never turn one `index()` call into hundreds of VLM
    # calls; the remainder are recorded as `visual_skipped_budget`, not
    # silently dropped.
    max_visuals_per_document: int = 24
    # -- Vector-drawn visual detection (spec "vector visual detection
    # requirement", 2026-08-13 correction phase). A chart/diagram drawn as
    # PDF vector primitives (lines/rects/curves, no embedded bitmap) is
    # invisible to `get_image_info()` above; `visual_extraction.py` also
    # inspects `page.get_drawings()`, spatially groups nearby/intersecting
    # primitives into candidate clusters, and applies the conservative,
    # multi-signal rules below (never a single magic number) before a
    # cluster is treated as a real chart/diagram rather than a table
    # border, page separator, underline, or other decorative furniture.
    #
    # Absolute floor: a cluster with fewer primitives than this is never a
    # chart, regardless of shape diversity (a single rule line, a lone
    # rectangle border, or a two-segment tick mark are all below this).
    vector_min_primitives_relaxed: int = 4
    # A cluster at or above `vector_min_primitives_relaxed` but below this
    # count is only accepted if it also mixes at least 2 distinct geometric
    # "flavors" (e.g. horizontal + vertical lines, or lines + rects) --
    # real charts/diagrams almost always mix orientations/shapes (axes vs.
    # bars/points/nodes) even when small; a same-flavor cluster in this
    # range (e.g. a handful of parallel lines) reads as a rule/underline
    # cluster, not a chart.
    vector_min_primitives: int = 8
    # A cluster made of only one geometric flavor (shape_diversity < 2,
    # e.g. many parallel horizontal lines -- the shape of a ruled table
    # grid or a repeated decorative rule pattern) is trusted only once its
    # primitive count clears this much higher bar.
    vector_high_count_single_orientation: int = 40
    # Minimum bounding-box area, as a fraction of the page, for a vector
    # cluster to be considered (higher than the raster `min_area_fraction`
    # since vector chart geometry -- thin lines/points -- naturally covers
    # less inked area than an embedded raster image of the same visual
    # size).
    vector_min_area_fraction: float = 0.02
    # Max gap (points) between two vector-drawing bounding boxes for them
    # to be grouped into the same candidate cluster (connected components
    # over "expanded bbox intersects", computed at individual drawing-item
    # granularity -- see `_item_bbox`). Bundled same-style elements (e.g.
    # several bar rects stroked in one `finish()` call) are split into
    # per-item primitives before grouping (see `_vector_candidates`), so
    # this gap only needs to bridge genuine small visual gaps within one
    # figure (tick marks, adjacent bars/segments), not e.g. matplotlib's
    # full inter-bar spacing -- keeping it small (rather than widening it
    # to compensate) is what keeps two *different* figures, or a figure
    # and nearby body text, from being merged into one cluster (confirmed
    # empirically: widening this to bridge inter-bar gaps instead of
    # splitting bundled items caused real charts on other corpus PDFs to
    # merge with adjacent panels/captions/body text and fail downstream
    # filters).
    vector_group_gap_pt: float = 12.0
    # Extra crop padding (points) applied when rendering a vector cluster's
    # bounding box to a bitmap, wider than the raster crop's padding, so
    # nearby axis/legend/node labels just outside the drawn geometry itself
    # are included in what Qwen3-VL actually sees.
    vector_crop_pad_pt: float = 20.0
    # If a vector cluster's bounding box is mostly (>= this fraction of its
    # own area) inside a table region PyMuPDF's own `page.find_tables()`
    # already detects (the same signal `pdf_extractor.py`'s `_extract_tables`
    # already uses -- no second table detector), it is suppressed as
    # table-grid geometry rather than sent to the VLM as a chart.
    vector_table_overlap_suppress_fraction: float = 0.6
    # Secondary, geometry-only ruled-grid suppression (spec: "if existing
    # table/layout signals can help suppress false positives, reuse them"
    # -- this is the fallback for a ruled grid `page.find_tables()` itself
    # doesn't recognize, e.g. bare lines with no cell text/borders). A
    # cluster whose primitives are overwhelmingly axis-aligned lines split
    # roughly evenly between horizontal and vertical, with few/no other
    # shapes (rects/curves/diagonals -- the bars/points/plotted-curves a
    # real chart's *content* is made of, as opposed to its axes/gridlines)
    # reads as a ruled grid, not a chart. `vector_grid_min_line_count` is
    # the minimum count of BOTH horizontal and vertical lines required
    # before this applies (a real chart's axes are typically exactly one
    # line per direction); `vector_grid_line_fraction` is the minimum
    # share of the cluster's primitives that must be plain h/v lines.
    vector_grid_min_line_count: int = 3
    vector_grid_line_fraction: float = 0.6
    # A primitive whose own bounding box is "thin" (long side at least this
    # many times its short side -- a stroked line/curve segment, like an
    # axis, gridline, bar edge, or plotted-line segment) rather than
    # "blob-shaped" (a filled glyph outline, a node/marker shape). Real
    # charts/diagrams -- even ones exported as thousands of tiny bezier
    # segments approximating a smooth curve -- are dominated by thin
    # primitives; a page whose "vector drawings" are actually glyph
    # outlines standing in for a missing text layer (confirmed on a real
    # corpus PDF, `Learning-Bayesian-Networks-from-Big-Data-with-Greedy-
    # Search.pdf`: `page.get_text()` returns 0 chars, yet `get_drawings()`
    # returns thousands of tiny curve-only paths shaped like the page's
    # ordinary body-text column) is instead dominated by compact,
    # roughly-square glyph blobs (median aspect ratio ~1.3 on that real
    # page, vs. 500-999 on real chart pages -- see acceptance evidence in
    # RAG_BUILD_SOURCE_OF_TRUTH.md).
    vector_thin_primitive_aspect_ratio: float = 4.0
    # A cluster needs at least this fraction of its primitives to be
    # "thin" (above) to be trusted as chart/diagram geometry. Real corpus
    # evidence: the false-positive glyph-text page measured ~1.3% thin
    # primitives; every real chart page measured 43-77%, giving a wide,
    # well-separated margin at this threshold.
    vector_min_thin_primitive_fraction: float = 0.20
    # Secondary text-like guard for a page whose vector-rendered glyphs
    # happen to have a high enough thin-primitive fraction to slip past
    # the check above (confirmed on a real corpus PDF's ResearchGate cover
    # page: title/DOI/author-metadata text with no extractable text layer,
    # thin_fraction 38% -- above the 20% bar -- because a couple of long
    # horizontal divider rules skewed the fraction on an otherwise sparse
    # page). A candidate covering a large share of a page that itself has
    # (near-)zero extractable text (the same `page.get_text()` signal this
    # pipeline's OCR trigger already uses -- reusing an existing signal,
    # not a new one) is far more likely to be that page's body/metadata
    # text than one real chart/diagram spanning almost the whole page.
    # Conservative by design (spec: prefer suppressing an ambiguous
    # candidate): a genuine full-page diagram on a page with no text layer
    # at all is a rarer case than a converted/vector-glyph text page, and
    # the raster path is unaffected either way.
    vector_sparse_page_text_chars: int = 40
    vector_whole_page_area_fraction: float = 0.35
    # IoU threshold above which an accepted vector cluster and an accepted
    # raster region on the same page are treated as the same source figure
    # (see `_dedupe_cross_source`) -- kept low/conservative since the
    # simple containment check (`_boxes_overlap_contained`) already catches
    # the common "one fully inside the other" case; this catches partial
    # overlap that containment alone would miss.
    vector_dedup_iou_threshold: float = 0.15
    # Safety cap: a page with more raw vector-drawing primitives than this
    # skips vector-cluster detection for that page rather than risking a
    # pathological worst case. Grouping itself is spatial-grid-bucketed
    # (not naive O(n^2)), so this can be generous -- a real corpus PDF
    # (Performance-comparison-of-two-virtual-machine-scenarios...) was
    # observed with a genuine high-resolution vector line/scatter chart
    # rendering as 36,000-74,000 individual drawing primitives on one
    # page, comfortably under this cap. Recorded as a skip, never a silent
    # gap, on the rare page that does exceed it (see
    # `VisualSkipReason.VECTOR_PAGE_TOO_DENSE`).
    vector_max_drawings_per_page: int = 150_000
    # How long Ollama keeps the model resident between calls *within* one
    # ingestion batch (Ollama's own `keep_alive`, not a process we manage
    # ourselves) -- then explicitly unloaded (`keep_alive: 0`) once the
    # batch's visual analysis is done, before the embedding pass, so the
    # VLM never competes for VRAM with either the embedding backend or the
    # answer-generation model at question-answering time (spec Phase 11).
    keep_alive_during_batch: str = os.environ.get("VISUAL_KEEP_ALIVE", "5m")
    unload_after_batch: bool = True
    # VRAM barrier (strict-sequential-GPU-lifecycle phase, spec "VRAM
    # BARRIER"): after `unload_model()` above, how long to poll real GPU
    # state for evidence the VLM actually released VRAM before Stage 3
    # (embedding) is allowed to start, and how much slack above the
    # pre-VLM baseline still counts as "released" -- covers desktop/driver
    # noise and any *unrelated* GPU-resident process (spec "EXTERNAL GPU
    # USERS": e.g. a game sharing the same card), never demands exactly 0
    # MiB above baseline (spec "DO NOT DEMAND ZERO VRAM"). Defaults chosen
    # from real measurement on the reference RTX 3060 desktop (see
    # RAG_BUILD_SOURCE_OF_TRUTH.md's strict-GPU-lifecycle change-log
    # entry), not guessed in advance.
    vram_barrier_timeout_s: float = float(os.environ.get("VISUAL_VRAM_BARRIER_TIMEOUT_S", "45"))
    vram_barrier_tolerance_mib: int = int(os.environ.get("VISUAL_VRAM_BARRIER_TOLERANCE_MIB", "1024"))


# Ingestion profiles (see DELIVERABLES.md, "MAIN_TEXT vs FULL_FIDELITY
# ingestion profiles"):
#   MAIN_TEXT     -- the active RAG ingestion path. Chunks/indexes only the
#                    filtered main-text retrieval view (retrieval_view.py)
#                    and gates on retrieval-critical findings only
#                    (retrieval_readiness.py). Default profile.
#   FULL_FIDELITY -- today's original behavior, unchanged: chunks/indexes
#                    the complete unfiltered document and gates on full
#                    structural validation (validation.py's errors == 0).
#                    Selectable for parser diagnostics and any future
#                    document-reconstruction work.
MAIN_TEXT_PROFILE = "main_text"
FULL_FIDELITY_PROFILE = "full_fidelity"
INGESTION_PROFILES = (MAIN_TEXT_PROFILE, FULL_FIDELITY_PROFILE)


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    semantic: SemanticStructuringConfig = field(default_factory=SemanticStructuringConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    strict: bool = os.environ.get("DOCPIPE_STRICT", "1") not in ("0", "false", "False")
    upload_max_bytes: int = int(os.environ.get("UPLOAD_MAX_BYTES", str(50 * 1024 * 1024)))
    retrieval_context_budget_tokens: int = int(os.environ.get("RAG_CONTEXT_BUDGET_TOKENS", "3000"))
    split_ingest_threshold_bytes: int = int(os.environ.get("SPLIT_INGEST_THRESHOLD_BYTES", str(100 * 1024 * 1024)))
    delete_source_on_doc_delete: bool = False
    profile: str = os.environ.get("DOCPIPE_PROFILE", MAIN_TEXT_PROFILE)

    def configuration_fingerprint_payload(self) -> dict:
        """Everything that, if changed, must force reprocessing of a source."""
        return {
            "normalizer_version": NORMALIZER_VERSION,
            "extractor_version": EXTRACTOR_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "provenance_format_version": PROVENANCE_FORMAT_VERSION,
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "chunking": self.chunking.fingerprint_fields(),
            "semantic_mode": self.semantic.mode,
            "semantic_model": self.semantic.model,
            "embedding_model": self.embedding.model,
            "embedding_provider": self.embedding.provider,
            "profile": self.profile,
        }


def _load_file(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError(f"PyYAML required to load {path} but is not installed")
        return yaml.safe_load(text) or {}
    return json.loads(text) if text.strip() else {}


def _apply(dc_instance, data: dict):
    for k, v in (data or {}).items():
        if not hasattr(dc_instance, k):
            continue
        cur = getattr(dc_instance, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _apply(cur, v)
        else:
            setattr(dc_instance, k, v)


_ENV_OVERRIDES = {
    "SOURCE_DIR": ("paths", "source_dir"),
    "PROCESSED_DIR": ("paths", "processed_dir"),
    "REPORT_DIR": ("paths", "report_dir"),
    "INDEX_DIR": ("paths", "index_dir"),
    "MANIFEST_PATH": ("paths", "manifest_path"),
    "EMBEDDING_HOST": ("embedding", "host"),
    "EMBEDDING_MODEL": ("embedding", "model"),
    "EMBEDDING_BATCH_SIZE": ("embedding", "batch_size"),
    "SEMANTIC_HOST": ("semantic", "host"),
    "SEMANTIC_MODEL": ("semantic", "model"),
    "SEMANTIC_MODE": ("semantic", "mode"),
    "SEMANTIC_BATCH_CHAR_BUDGET": ("semantic", "batch_char_budget"),
    "CHUNK_HARD_MAX_WORDS": ("chunking", "hard_max_words"),
    "CHUNK_TARGET_WORDS": ("chunking", "target_words"),
    "CHUNK_PARAGRAPH_WARN_WORDS": ("chunking", "preferred_paragraph_words"),
    "EMBEDDING_CHUNK_TARGET_TOKENS": ("chunking", "target_tokens"),
    "EMBEDDING_CHUNK_MAX_TOKENS": ("chunking", "max_tokens"),
    "CHUNK_OVERSIZED_ROW_POLICY": ("chunking", "oversized_row_policy"),
    "OCR_MODE": ("ocr", "mode"),
    "UPLOAD_MAX_BYTES": (None, "upload_max_bytes"),
    "SPLIT_INGEST_THRESHOLD_BYTES": (None, "split_ingest_threshold_bytes"),
    "RAG_CONTEXT_BUDGET_TOKENS": (None, "retrieval_context_budget_tokens"),
    "DOCPIPE_STRICT": (None, "strict"),
    "DOCPIPE_PROFILE": (None, "profile"),
}

_INT_FIELDS = {
    "hard_max_words", "target_words", "preferred_paragraph_words",
    "upload_max_bytes", "retrieval_context_budget_tokens",
    "split_ingest_threshold_bytes",
    "target_tokens", "max_tokens", "batch_size", "batch_char_budget",
}
_BOOL_FIELDS = {"strict"}


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    cfg = PipelineConfig()

    candidate = config_path or os.environ.get("DOCPIPE_CONFIG")
    if not candidate:
        for name in ("docpipe.yaml", "docpipe.yml", "docpipe.json"):
            p = REPO_ROOT / name
            if p.exists():
                candidate = str(p)
                break
    if candidate:
        _apply_nested(cfg, _load_file(Path(candidate)))

    for env_name, (section, field_name) in _ENV_OVERRIDES.items():
        if env_name not in os.environ:
            continue
        raw = os.environ[env_name]
        value: object = raw
        if field_name in _INT_FIELDS:
            value = int(raw)
        elif field_name in _BOOL_FIELDS:
            value = raw not in ("0", "false", "False")
        target = getattr(cfg, section) if section else cfg
        setattr(target, field_name, value)

    cfg.paths.ensure()
    return cfg


def _apply_nested(cfg: PipelineConfig, data: dict):
    for section in ("paths", "embedding", "semantic", "chunking", "ocr", "visual"):
        if section in (data or {}):
            _apply(getattr(cfg, section), data[section])
    for top in ("strict", "upload_max_bytes", "retrieval_context_budget_tokens", "delete_source_on_doc_delete",
                "split_ingest_threshold_bytes"):
        if top in (data or {}):
            setattr(cfg, top, data[top])


DEFAULT_CONFIG = load_config()

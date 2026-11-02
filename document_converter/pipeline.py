"""
DocumentPipeline: the single service every caller (CLI, ingestion.py,
orchestrator.py, tests) goes through instead of re-implementing any stage.

Four operations, all built from the same `_run` core so preview/validate
output can never drift from what actually gets indexed:
  * preview(path)  -- extract/normalize/structure/chunk/validate, no I/O.
  * process(path)  -- preview() + write normalized markdown & report to disk.
  * validate_only  -- preview() but return just the ValidationReport.
  * index(path)    -- process() + embed chunks + write a versioned index
                       entry. Skips re-embedding if the source content and
                       pipeline configuration fingerprint are unchanged
                       (section 12), unless force=True.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from .config import PipelineConfig, MAIN_TEXT_PROFILE, INGESTION_PROFILES
from .model import Document, DocumentMetadata, ParagraphBlock, ListBlock, CodeBlock, TableBlock
from .findings import FindingLog
from .extractors import select_extractor, SUPPORTED_EXTENSIONS
from .normalize import normalize_document, clean_filename_title
from .semantic_structuring import apply_semantic_structuring
from .heading_demotion import demote_false_headings
from .heading_repair import repair_bodyless_heading_chains
from .heading_sibling_audit import audit_numbered_sibling_headings
from .paragraph_continuity_repair import repair_interrupted_paragraphs
from .running_header_repair import repair_running_headers
from .section_numbering_audit import audit_repeated_top_level_numbering
from .render import render_markdown
from .chunker import MarkdownChunker, ChunkingResult, ChunkRecord
from .validation import validate, ValidationReport, completion_state as _completion_state
from .retrieval_view import (
    build_main_text_document, resolve_main_text_title, apply_resolved_title, MainTextFilterStats,
)
from .retrieval_readiness import evaluate_main_text_readiness, apply_main_text_overlay
from .embedding_client import EmbeddingClient, EmbeddingOversizedInputError, EmbeddingServiceError
from .index_schema import (
    build_index_payload, build_visual_record, write_index_atomic, load_index, is_up_to_date,
    configuration_fingerprint, sha256_text, sha256_hex,
)
from .title_overrides import (
    get_title_override as _get_title_override,
    set_title_override as _set_title_override,
    clear_title_override as _clear_title_override,
)
from . import visual_extraction, visual_association, visual_analysis, visual_cache, gpu_lifecycle

logger = logging.getLogger(__name__)

# Optional progress reporting (spec: "Make document ingestion observable").
# `(stage, payload)` -- e.g. ("embedding", {"current": 128, "total": 4096}).
# Default `None` everywhere reproduces today's behavior byte for byte; every
# existing caller (cli.py, ingestion.py, tests) stays valid unchanged.
ProgressFn = Callable[[str, dict], None]


def _emit(progress: Optional[ProgressFn], stage: str, **payload) -> None:
    """Fires `progress(stage, payload)` if a callback was given, and never
    lets a raising callback break an ingest -- a progress reporter is a
    diagnostic convenience, never a correctness dependency."""
    if progress is None:
        return
    try:
        progress(stage, payload)
    except Exception:
        logger.debug("progress callback failed", exc_info=True)


_MAIN_BODY_CONTENT_TYPES = (ParagraphBlock, ListBlock, CodeBlock, TableBlock)


def _has_main_body_content(document: Document) -> bool:
    return any(isinstance(b, _MAIN_BODY_CONTENT_TYPES) and b.word_count > 0 for b in document.blocks)


class PipelineValidationError(RuntimeError):
    """Raised by index() when strict mode is on and validation found errors."""

    def __init__(self, document_id: str, report: ValidationReport):
        self.document_id = document_id
        self.report = report
        errors = [f for f in report.findings.findings if f.severity.value == "ERROR"]
        # Under MAIN_TEXT, `report.findings` still contains every finding
        # `validate()` produced, including the ones `retrieval_readiness.py`
        # exempted from gating -- showing those first here would point a
        # developer at a red herring (e.g. "no H1 present") instead of the
        # actual reason indexing was blocked. When the MAIN_TEXT overlay ran
        # (`main_text_blocking_error_codes` present in stats), narrow the
        # message to the errors that actually caused the failure.
        blocking_codes = report.stats.get("main_text_blocking_error_codes")
        if blocking_codes is not None:
            errors = [e for e in errors if e.code in blocking_codes]
        detail = "; ".join(f"{e.code}: {e.message}" for e in errors[:5])
        super().__init__(f"Document {document_id!r} failed strict validation ({len(errors)} error(s)): {detail}")


def stable_document_id(relative_source_path: str) -> str:
    """Deterministic id derived from the source's relative path, so re-processing
    the same source always overwrites the same index entry instead of orphaning
    the old one under a content-hash id (see deliverables doc, 'document identity')."""
    norm = relative_source_path.replace(os.sep, "/").strip("/")
    return "d-" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


@dataclass
class ProcessResult:
    document_id: str
    title: str
    # Where `title` came from: "manual_override" when a user-set override
    # (title_overrides.py) was applied, otherwise whatever
    # `resolve_main_text_title`/the FULL_FIDELITY metadata check already
    # produced ("metadata", "front_matter_zone", "first_body_heading",
    # "unresolved", ...) -- see `_prepare_for_profile`.
    title_source: str
    source_path: str
    # The document actually used for this run's chunking/validation --
    # the MAIN_TEXT-filtered view under that profile, the complete
    # extracted-and-normalized document under FULL_FIDELITY. Always
    # consistent with `chunking_result`/`validation_report` (spec: preview
    # output can never drift from what actually gets indexed).
    document: Document
    markdown: str
    chunking_result: ChunkingResult
    validation_report: ValidationReport
    structuring_used: bool
    profile: str
    # The complete extracted-and-normalized document, regardless of
    # profile -- always available for diagnostics even when `document`
    # above is the filtered MAIN_TEXT view (spec: "a developer must still
    # be able to inspect the complete extraction result").
    full_document: Optional[Document] = None
    main_text_filter_stats: Optional[Any] = None
    markdown_path: Optional[str] = None
    report_path: Optional[str] = None
    index_path: Optional[str] = None
    skipped_unchanged: bool = False
    # VISUAL_ANALYSIS ingestion stats for this run (spec Phase 2/14), or
    # None when visual analysis wasn't attempted (preview()/process(), a
    # non-PDF source, or `config.visual.enabled=False`) -- see
    # `DocumentPipeline._analyze_visuals`/`_embed_visuals`.
    visual_stats: Optional[Dict[str, Any]] = None
    # Strict-sequential-GPU-lifecycle evidence for this run's ingestion
    # (only populated by index(), never preview()/process()): per-stage GPU
    # baseline/unload/barrier measurements, so a caller can report real
    # proof of "only one GPU-heavy model was resident at a time" without
    # re-deriving it from logs. None when this run took no GPU-heavy path
    # at all (e.g. a plain re-embed with visual analysis disabled still
    # populates the embedding-stage entry; a fully skipped/unchanged run
    # leaves this None).
    gpu_lifecycle_evidence: Optional[Dict[str, Any]] = None

    @property
    def completion_state(self) -> Dict[str, Any]:
        """Spec section 8's per-document completion object. `indexed` is
        derived from whether this result actually corresponds to a written
        index entry -- true for index() (new write or unchanged-skip), false
        for preview()/process()."""
        return _completion_state(self.validation_report, indexed=self.index_path is not None)


class DocumentPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._embedding_client: Optional[EmbeddingClient] = None

    @property
    def embedding_client(self) -> EmbeddingClient:
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient(
                self.config.embedding.host, self.config.embedding.model, self.config.embedding.timeout_seconds,
            )
        return self._embedding_client

    # ------------------------------------------------------------------ #
    # core

    def _build_document(
        self, path: Path, progress: Optional[ProgressFn] = None,
    ) -> "tuple[Document, FindingLog, bool]":
        log = FindingLog()
        extractor = select_extractor(path)
        _emit(progress, "extracting")
        try:
            document, extract_log = extractor.extract(path, self.config.ocr, progress=progress)
        except Exception as e:
            # A corrupt, truncated, or otherwise unreadable file must produce
            # the same typed FindingLog/ValidationReport shape as any other
            # failure -- never an unhandled traceback bubbling past this
            # point -- and it must never look like an empty-but-successful
            # document (spec section 3.6 / 9's "never claim success" rule).
            log.error(
                "extraction_failed",
                f"Could not extract content from {path.name}: {type(e).__name__}: {e}",
                suggested_action="Confirm the file isn't corrupted or truncated, and that it actually "
                                  "is the format its extension claims; try opening it in its native "
                                  "application to check.",
            )
            empty_doc = Document(metadata=DocumentMetadata(
                source_path=str(path), source_format=path.suffix.lstrip(".").lower() or "unknown",
            ))
            return empty_doc, log, False
        log.extend(extract_log)
        _emit(progress, "normalizing", current=0, total=len(document.blocks))
        log.extend(normalize_document(document))
        _emit(progress, "normalizing", current=len(document.blocks), total=len(document.blocks))
        _emit(progress, "repairing_running_headers", current=0, total=len(document.blocks))
        running_header_stats = repair_running_headers(document, log)
        document.metadata.extra["running_header_repair"] = running_header_stats
        _emit(progress, "repairing_running_headers", current=len(document.blocks), total=len(document.blocks))
        if self.config.semantic.enabled:
            _emit(progress, "structuring")
        structuring = apply_semantic_structuring(document, self.config.semantic, self.config.chunking)
        log.extend(structuring.findings)
        _emit(progress, "repairing_headings", current=0, total=len(structuring.document.blocks))
        repair_stats = repair_bodyless_heading_chains(structuring.document, log)
        structuring.document.metadata.extra["heading_repair"] = repair_stats
        _emit(progress, "repairing_headings", current=len(structuring.document.blocks), total=len(structuring.document.blocks))
        # False-heading demotion runs here -- after normalize/running-header
        # repair/structuring/bodyless-heading repair, before the two heading
        # audits -- for every extractor, not just Docling (it used to be
        # called only inside docling_extractor.py, making it dead code on
        # the default legacy PDF path):
        #   * AFTER running-header repair: that pass never touches
        #     HeadingBlocks (its own docstring defers repeated *headings* to
        #     this demotion pass), and running first de-noises the block
        #     list so demotion's adjacency-based context checks (procedural
        #     clusters, table-header chains, UI-chrome labels) see real
        #     neighbors, not duplicated page-boundary boilerplate.
        #   * AFTER semantic structuring: when structuring is enabled and
        #     used it rebuilds the block list with synthetic plan-generated
        #     HeadingBlocks, so a demotion before it would never reach the
        #     tree downstream stages actually see. When structuring is off
        #     or falls back, `structuring.document` is the same object, so
        #     this placement is identical for the default path.
        #   * AFTER bodyless-heading repair: that pass is the narrow OCR-only
        #     specialist (`style["from_ocr"]` headings, score evidence, typed
        #     CaptionBlock/FigureLabelBlock/... replacements); giving it
        #     first crack keeps its richer typing and chain context intact.
        #     This pass is the parser-neutral generalist and sweeps what the
        #     specialist's scope deliberately excludes.
        #   * BEFORE the heading-sibling and section-numbering audits, so
        #     both audits see the post-demotion heading tree (demoted blocks
        #     are UnknownBlocks, out of the heading hierarchy entirely).
        _emit(progress, "demoting_false_headings", current=0, total=len(structuring.document.blocks))
        demotion_stats = demote_false_headings(structuring.document, log)
        structuring.document.metadata.extra["heading_demotion"] = demotion_stats
        _emit(progress, "demoting_false_headings", current=len(structuring.document.blocks), total=len(structuring.document.blocks))
        _emit(progress, "auditing_heading_siblings", current=0, total=len(structuring.document.blocks))
        sibling_audit_stats = audit_numbered_sibling_headings(structuring.document, log)
        structuring.document.metadata.extra["heading_sibling_audit"] = sibling_audit_stats
        _emit(progress, "auditing_heading_siblings", current=len(structuring.document.blocks), total=len(structuring.document.blocks))
        _emit(progress, "auditing_section_numbering", current=0, total=len(structuring.document.blocks))
        numbering_audit_stats = audit_repeated_top_level_numbering(structuring.document, log)
        structuring.document.metadata.extra["section_numbering_audit"] = numbering_audit_stats
        _emit(progress, "auditing_section_numbering", current=len(structuring.document.blocks), total=len(structuring.document.blocks))
        _emit(progress, "repairing_paragraph_continuity", current=0, total=len(structuring.document.blocks))
        continuity_stats = repair_interrupted_paragraphs(structuring.document, log)
        structuring.document.metadata.extra["paragraph_continuity_repair"] = continuity_stats
        _emit(progress, "repairing_paragraph_continuity", current=len(structuring.document.blocks), total=len(structuring.document.blocks))
        return structuring.document, log, structuring.used

    def _resolve_profile(self, profile: Optional[str]) -> str:
        candidate = profile or self.config.profile
        return candidate if candidate in INGESTION_PROFILES else MAIN_TEXT_PROFILE

    def _prepare_for_profile(
        self, document: Document, profile: str, path: Path, document_id: str,
        relative_source: Optional[str], token_counter, max_seq_tokens: Optional[int], upstream_log: FindingLog,
        progress: Optional[ProgressFn] = None,
    ) -> "tuple[Document, ChunkingResult, ValidationReport, str, str, Optional[MainTextFilterStats]]":
        """The one place a `Document` becomes `(active_document, chunking_result,
        validation_report, title, title_source, filter_stats)` for a given
        profile. Called by
        both `_run` (word-count/offline-tokenizer pass) and `index()`'s
        real-embedding-tokenizer re-chunk pass, so the two can never drift
        apart on which document/config a profile implies."""
        filter_stats: Optional[MainTextFilterStats] = None
        # A manual title override (title_overrides.py) -- set from the UI when
        # the automatic resolver below lands on a technically-valid but
        # useless title (an export tool's "Untitled 1" default document-title
        # metadata) -- always wins over automatic resolution, for both
        # profiles, the same way an explicit `document.metadata.title` would.
        # Checked once here (the single place title resolution happens) so
        # every downstream reader of the resolved title -- MAIN_TEXT's H1
        # synthesis just below, the chunker's header paths, `report`/
        # `result.title`, and the index/citation metadata built from it --
        # sees one corrected value, not a value that only the browser knows
        # about while the index/Markdown/chunks still say "Untitled 1".
        override_title = _get_title_override(self.config.paths.index_dir, document_id)
        if profile == MAIN_TEXT_PROFILE:
            active_document, filter_stats = build_main_text_document(document)
            if override_title:
                resolved_title, title_source = override_title, "manual_override"
            else:
                resolved_title, title_source = resolve_main_text_title(
                    document, active_document.blocks, filter_stats.title_region_text,
                )
            # Strict mode's "never split a paragraph" rule is a FULL_FIDELITY
            # diagnostic guarantee; MAIN_TEXT's job is to keep legitimate long
            # body paragraphs searchable (spec Phase 6), so it opts into the
            # existing non-strict sentence-boundary emergency split instead of
            # dropping the paragraph outright. Every other budget (word/token
            # targets) is identical to the configured chunking policy.
            chunk_config = dataclasses.replace(self.config.chunking, strict=False)
        else:
            active_document = document
            if override_title:
                resolved_title, title_source = override_title, "manual_override"
            else:
                resolved_title = document.metadata.title
                title_source = "metadata" if document.metadata.title else "unresolved"
            chunk_config = self.config.chunking

        # Fall back to the *original* filename's stem, not a temp/upload path's
        # (preview() writes uploads to a randomly-named temp file, so falling
        # back to path.stem there would show the temp name as the title).
        # Run through the same filename->title cleanup normalize.py's H1
        # synthesis uses (hyphens/underscores -> spaces, "Word123" split,
        # acronyms preserved) so a document with no resolvable in-body or
        # metadata title still gets a readable citation title instead of a
        # raw hyphenated slug. Computed *before* chunking (not after, as
        # this used to be) so the title actually used is available to
        # `apply_resolved_title` below -- the renderer, the chunker, and
        # every downstream reader of `active_document` must all see the
        # same title, not just `result.title`/the index metadata.
        fallback_stem = Path(relative_source).stem if relative_source else path.stem
        fallback_title = clean_filename_title(fallback_stem)
        title = resolved_title or fallback_title

        if profile == MAIN_TEXT_PROFILE:
            # Title resolution (`resolve_main_text_title`, above) and what
            # the Markdown renderer/chunker actually see must never drift
            # apart -- without this, `active_document`'s opening heading is
            # whatever `build_main_text_document` happened to leave first
            # (a body subsection, a front-matter-shaped heading that
            # survived, or nothing at all), regardless of what title was
            # actually resolved. Mutates `active_document` in place, so the
            # synthesized/deduplicated H1 is what both the chunker call and
            # `validate()` below see too, not just the eventual
            # `render_markdown()` call in `_run`/`index()`.
            apply_resolved_title(active_document, title)

        # Reported after the fact rather than with a current=0 start tick
        # (unlike the six `_build_document` block stages, which each run
        # exactly once per job): `index()` re-invokes this whole method a
        # second time, with the same document, to re-chunk with the real
        # embedding tokenizer -- a start-tick reset here would make
        # "chunking"/"validating"'s reported `current` non-monotonic across
        # that job even though each individual pass is (see
        # test_ingestion_observability.py's monotonicity contract). Reported
        # in blocks, like every other block-iterating stage, not chunks --
        # `chunking_result`'s own chunk count is a different, unrelated
        # denominator from `active_document.blocks`.
        chunking_result = MarkdownChunker().chunk(
            active_document, source_path=str(path), config=chunk_config, document_id=document_id,
            token_counter=token_counter, max_sequence_tokens=max_seq_tokens,
        )
        _emit(progress, "chunking", current=len(active_document.blocks), total=len(active_document.blocks))
        report = validate(
            active_document, chunking_result, chunk_config, str(path), "",
            document.metadata.source_format, document.metadata.ocr_used, upstream_log,
        )
        _emit(progress, "validating", current=len(active_document.blocks), total=len(active_document.blocks))

        if profile == MAIN_TEXT_PROFILE:
            readiness = evaluate_main_text_readiness(
                report, resolved_title or None, title_source, _has_main_body_content(active_document),
            )
            apply_main_text_overlay(report, readiness)

        return active_document, chunking_result, report, title, title_source, filter_stats

    def _run(self, path: Path, document_id: str, embedding_client: Optional[EmbeddingClient],
              relative_source: Optional[str] = None, profile: Optional[str] = None,
              progress: Optional[ProgressFn] = None) -> ProcessResult:
        document, log, structuring_used = self._build_document(path, progress=progress)
        effective_profile = self._resolve_profile(profile)

        token_counter = None
        max_seq_tokens = None
        if embedding_client is not None:
            info = embedding_client.health()
            token_counter = embedding_client.count_tokens
            max_seq_tokens = info.max_sequence_tokens

        active_document, chunking_result, report, title, title_source, filter_stats = self._prepare_for_profile(
            document, effective_profile, path, document_id, relative_source, token_counter, max_seq_tokens, log,
            progress=progress,
        )
        # No start tick, for the same reason "chunking"/"validating" have
        # none above -- `index()` re-renders a second time in this same job.
        markdown = render_markdown(active_document)
        _emit(progress, "rendering", current=len(active_document.blocks), total=len(active_document.blocks))

        return ProcessResult(
            document_id=document_id, title=title, title_source=title_source, source_path=str(path),
            document=active_document, markdown=markdown, chunking_result=chunking_result, validation_report=report,
            structuring_used=structuring_used, profile=effective_profile,
            full_document=document, main_text_filter_stats=filter_stats,
        )

    def _resolve(self, source_path: str, document_id: Optional[str], relative_source: Optional[str]) -> "tuple[Path, str]":
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(source_path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type {path.suffix!r} for {source_path}")
        doc_id = document_id or stable_document_id(relative_source or path.name)
        return path, doc_id

    # ------------------------------------------------------------------ #
    # public operations

    def preview(self, source_path: str, document_id: Optional[str] = None,
                relative_source: Optional[str] = None, profile: Optional[str] = None,
                progress: Optional[ProgressFn] = None) -> ProcessResult:
        path, doc_id = self._resolve(source_path, document_id, relative_source)
        result = self._run(path, doc_id, embedding_client=None, relative_source=relative_source, profile=profile,
                            progress=progress)
        _emit(progress, "done")
        return result

    def validate_only(self, source_path: str, document_id: Optional[str] = None,
                       relative_source: Optional[str] = None, profile: Optional[str] = None) -> ValidationReport:
        return self.preview(source_path, document_id, relative_source, profile=profile).validation_report

    def process(self, source_path: str, document_id: Optional[str] = None,
                relative_source: Optional[str] = None, profile: Optional[str] = None,
                progress: Optional[ProgressFn] = None) -> ProcessResult:
        path, doc_id = self._resolve(source_path, document_id, relative_source)
        result = self._run(path, doc_id, embedding_client=None, relative_source=relative_source, profile=profile,
                            progress=progress)

        self.config.paths.ensure()
        md_path = os.path.join(self.config.paths.processed_dir, f"{doc_id}.md")
        report_path = os.path.join(self.config.paths.report_dir, f"{doc_id}.report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.markdown)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(result.validation_report.to_human_text())
        result.markdown_path = md_path
        result.report_path = report_path
        return result

    def index(self, source_path: str, document_id: Optional[str] = None,
              relative_source: Optional[str] = None, force: bool = False,
              profile: Optional[str] = None, progress: Optional[ProgressFn] = None) -> ProcessResult:
        path, doc_id = self._resolve(source_path, document_id, relative_source)
        source_hash = sha256_hex(path.read_bytes())
        config_fp = configuration_fingerprint(self.config)
        effective_profile = self._resolve_profile(profile)

        existing_path = os.path.join(self.config.paths.index_dir, f"{doc_id}.json")
        if not force and os.path.exists(existing_path):
            try:
                existing = load_index(existing_path)
            except Exception:
                existing = None
            if existing is not None and is_up_to_date(existing, source_hash, config_fp):
                result = self._run(path, doc_id, embedding_client=None, relative_source=relative_source,
                                    profile=effective_profile, progress=progress)
                result.index_path = existing_path
                result.skipped_unchanged = True
                _emit(progress, "done")
                return result

        result = self.process(source_path, document_id=doc_id, relative_source=relative_source,
                               profile=effective_profile, progress=progress)
        client = self.embedding_client
        info = client.health()

        # Re-chunk with exact token accounting from the live embedding service,
        # now that we're committed to actually embedding these chunks -- via
        # the same `_prepare_for_profile` helper `_run` used, starting again
        # from `full_document` (never from the already-filtered `result.document`
        # -- MAIN_TEXT filtering must always start from the complete extraction).
        active_document, chunking_result, report, title, title_source, filter_stats = self._prepare_for_profile(
            result.full_document, effective_profile, path, doc_id, relative_source,
            client.count_tokens, info.max_sequence_tokens, FindingLog(),
            progress=progress,
        )
        # Re-render from *this* active_document -- the one actually chunked,
        # embedded, and about to be indexed below -- rather than leaving
        # `result.markdown`/the on-disk .md file as whatever `process()`'s
        # earlier offline-tokenizer pass rendered a moment ago. Filtering
        # itself is deterministic and token-counter-independent, so the two
        # passes select the same blocks in practice, but the *renderer*, the
        # *downloadable Markdown*, and the *indexer* must provably share one
        # active_document, not two separately-built-but-hopefully-equal
        # ones -- this closes that gap instead of relying on the two passes
        # happening to agree.
        markdown = render_markdown(active_document)
        _emit(progress, "rendering", current=len(active_document.blocks), total=len(active_document.blocks))
        result.document = active_document
        result.markdown = markdown
        result.chunking_result = chunking_result
        result.validation_report = report
        result.title = title
        result.title_source = title_source
        result.main_text_filter_stats = filter_stats
        if result.markdown_path:
            with open(result.markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown)

        if effective_profile == MAIN_TEXT_PROFILE:
            blocked = not result.validation_report.retrieval_ready
        else:
            blocked = result.validation_report.findings.has_errors()
        if self.config.strict and blocked:
            raise PipelineValidationError(doc_id, result.validation_report)

        # ---------------------------------------------------------------- #
        # Strict sequential GPU model lifecycle (RAG_BUILD_SOURCE_OF_TRUTH.md
        # "strict sequential GPU model lifecycle" phase): STAGE 2 (vision)
        # must fully unload the VLM and clear a VRAM barrier *before* STAGE 3
        # (embedding) does any real `encode()` work, so Qwen3-VL and the
        # (potentially large) embedding model are never GPU-resident at the
        # same time because of this pipeline. See `_analyze_visuals` (Stage
        # 2, no embedding calls) and `_embed_visuals` (second half of Stage
        # 3, called after MAIN_TEXT is embedded below) for the two halves of
        # the old single `_process_visuals`.
        gpu_evidence: Dict[str, Any] = {}

        # STAGE 2 -- VISUAL ANALYSIS: analyze eligible visuals with
        # Qwen3-VL and produce VISUAL_ANALYSIS retrieval-text records.
        # PDF-only (see visual_extraction.py); a non-PDF source or a
        # disabled config both short-circuit to the same all-zero stats
        # block `build_index_payload` builds when `visual_records=None`.
        visual_pending: List[Dict[str, Any]] = []
        visual_stats: Optional[Dict[str, Any]] = None
        if self.config.visual.enabled and effective_profile == MAIN_TEXT_PROFILE and path.suffix.lower() == ".pdf":
            visual_pending, visual_stats = self._analyze_visuals(
                path=path, doc_id=doc_id, chunks=chunking_result.chunks, client=client, progress=progress,
            )
            if visual_stats is not None and "vram_barrier" in visual_stats:
                gpu_evidence["vision_stage_vram_barrier"] = visual_stats["vram_barrier"]

        # STAGE 3 -- EMBEDDING / INDEX BUILD: MAIN_TEXT chunks embedded
        # first (fatal on failure, unchanged behavior), then -- still
        # within this same embedding stage, never with a VLM call in
        # between -- VISUAL_ANALYSIS retrieval text through the identical
        # embedding_client/config (spec "FINAL COMBINATION CONTRACT": one
        # logical retrieval-record set, one embedding configuration).
        embedding_baseline_mib = gpu_lifecycle.read_gpu_memory_used_mib()
        texts = [c.text for c in chunking_result.chunks]
        embeddings: List[List[float]] = []
        if texts:
            batch_size = max(1, self.config.embedding.batch_size)
            total = len(texts)
            # Embedded in slices rather than one `embed_batch(texts)` call over
            # the whole corpus -- at tens of thousands of chunks, a single
            # request body is multi-hundred-MB and cannot succeed against
            # EmbeddingClient's timeout (spec: "batch the embedding call").
            for start in range(0, total, batch_size):
                batch_texts = texts[start:start + batch_size]
                try:
                    batch_embeddings = client.embed_batch(batch_texts)
                except EmbeddingOversizedInputError as e:
                    # `e.oversized_indices`/`e.token_counts` are positions within
                    # `batch_texts` (this slice), not `texts` (the whole corpus)
                    # -- the finding below must name the true global chunk, so
                    # `start` is added before using them for the message/lookup.
                    # `token_counts` stays indexed by the original (slice-local)
                    # position -- it is only ever as long as this one batch.
                    for local_idx in e.oversized_indices:
                        global_idx = start + local_idx
                        chunking_result.findings.error(
                            "chunk_exceeds_token_limit",
                            f"Chunk {global_idx} ({e.token_counts[local_idx]} tokens) exceeds the "
                            f"embedding model's max sequence length and was not embedded",
                            block_id=chunking_result.chunks[global_idx].chunk_id,
                        )
                    # Rewritten to global indices so any caller inspecting the
                    # raised exception itself (not just the findings log above)
                    # also sees the correct chunk positions.
                    e.oversized_indices = [start + i for i in e.oversized_indices]
                    raise
                embeddings.extend(batch_embeddings)
                _emit(progress, "embedding", current=len(embeddings), total=total)
            # `build_index_payload` below zips chunks with embeddings positionally --
            # a length mismatch would otherwise silently drop trailing chunks (or
            # embeddings) instead of failing loudly (spec section 8's "chunk count
            # == embedding count" invariant).
            if len(embeddings) != len(texts):
                raise EmbeddingServiceError(
                    f"Embedding service returned {len(embeddings)} embedding(s) for {len(texts)} "
                    f"chunk(s) -- refusing to write a mismatched index"
                )

        # VISUAL_ANALYSIS embedding (spec Phase 2-7, second half of Stage
        # 3): additive, best-effort, and never allowed to corrupt MAIN_TEXT
        # indexing above -- any failure here (embedding failure for visual
        # text) degrades to "this document has fewer/no visual records"
        # while `embeddings`/`chunking_result` for MAIN_TEXT is written
        # exactly as it already was.
        visual_records: List[Dict[str, Any]] = []
        if visual_pending:
            visual_records, visual_stats = self._embed_visuals(
                visual_pending, visual_stats, doc_id=doc_id, source_path=str(path),
                source_hash=source_hash, client=client, progress=progress,
            )

        # Explicit release (spec "EXPLICITLY RELEASE EMBEDDING MODEL" /
        # "VRAM BARRIER"): best-effort; frees this document's embedding-
        # stage VRAM before the next GPU-heavy stage -- the next
        # document's VLM pass in a batch `scan_and_index()` run, or a
        # concurrent query -- starts.
        gpu_evidence["embedding_stage_baseline_mib"] = embedding_baseline_mib
        gpu_evidence["embedding_stage_released"] = client.release()
        if embedding_baseline_mib is not None:
            gpu_evidence["embedding_stage_vram_barrier"] = gpu_lifecycle.wait_for_vram_release(
                embedding_baseline_mib, tolerance_mib=self.config.visual.vram_barrier_tolerance_mib,
                timeout_s=self.config.visual.vram_barrier_timeout_s,
            ).to_dict()

        _emit(progress, "writing_index")
        payload = build_index_payload(
            document_id=doc_id, title=result.title, title_source=result.title_source, source_path=str(path),
            markdown_path=result.markdown_path or "", report_path=result.report_path or "",
            source_hash=source_hash, normalized_hash=sha256_text(result.markdown),
            extractor=result.document.metadata.source_format, ocr_used=result.document.metadata.ocr_used,
            semantic_structuring_used=result.structuring_used, config_fingerprint=config_fp,
            embedding_info=info, chunks=chunking_result.chunks, embeddings=embeddings,
            retrieval_ready=result.validation_report.retrieval_ready,
            warnings=result.validation_report.stats["warnings"], errors=result.validation_report.stats["errors"],
            profile=effective_profile,
            chunk_target_tokens=self.config.chunking.target_tokens, chunk_max_tokens=self.config.chunking.max_tokens,
            findings=result.validation_report.findings.to_list(), stats=result.validation_report.stats,
            visual_records=visual_records, visual_stats=visual_stats,
        )
        result.index_path = write_index_atomic(self.config.paths.index_dir, doc_id, payload)
        result.visual_stats = visual_stats
        result.gpu_lifecycle_evidence = gpu_evidence
        _emit(progress, "done")
        return result

    def _analyze_visuals(
        self, path: Path, doc_id: str, chunks: List["ChunkRecord"], client: EmbeddingClient,
        progress: Optional[ProgressFn] = None,
    ) -> "tuple[List[Dict[str, Any]], Dict[str, Any]]":
        """STAGE 2 -- VISUAL ANALYSIS (strict sequential GPU model
        lifecycle): locates, crops, and analyzes (Qwen3-VL via Ollama,
        cached) every eligible visual region in `path` (spec Phases 2-6),
        producing retrieval-text records -- but never embeds them. Embedding
        (Stage 3) happens later, in `_embed_visuals`, only after this
        method's `finally` block has explicitly unloaded the VLM and
        evaluated a VRAM barrier, so the VLM and the embedding model are
        never GPU-resident at the same time because of this pipeline.

        Never raises -- every failure mode (crop, VLM, schema) is caught
        per-visual and folded into the returned stats, so one bad visual can
        never make an otherwise-valid document unindexable (spec "FAILURE
        BEHAVIOR"), and the VLM is unloaded even when analysis fails
        unexpectedly partway through the batch (spec "FAILURE HANDLING":
        "VLM exception -> unload VLM anyway")."""
        stats: Dict[str, Any] = {
            "enabled": True, "found": 0, "eligible": 0, "skipped_non_retrieval": 0,
            "analyzed": 0, "cached": 0, "failed": 0, "indexed": 0, "failures": [],
            "token_reduced": 0, "vram_barrier": None,
        }
        vcfg = self.config.visual
        index_dir = self.config.paths.index_dir

        # Token-native sizing (spec "TOKEN VALIDATION REQUIREMENT"): the
        # same tokenizer/limit mechanism `_run`'s MAIN_TEXT chunking already
        # uses (`client.health().max_sequence_tokens` +
        # `client.count_tokens`), never a hardcoded constant, so this stays
        # correct whichever embedding backend (BGE today, Qwen3-Embedding-4B
        # after the still-open migration gates) actually owns the target
        # index. Best-effort: if `health()` fails here, retrieval-text
        # building falls back to the word-budget-only path and the existing
        # `embed_batch` oversized-input handling in `_embed_visuals` below
        # is still the final safety net, exactly as before this phase. This
        # is a metadata/tokenizer call only (no `model.encode()`, no GPU
        # work) -- it does not itself put the embedding model on the GPU.
        count_tokens = None
        max_seq_tokens = None
        try:
            count_tokens = client.count_tokens
            max_seq_tokens = client.health().max_sequence_tokens
        except Exception as e:
            logger.warning(f"[visual] {doc_id}: could not resolve active embedding tokenizer/limit "
                            f"for token-native visual-text sizing, falling back to word-budget caps only: {e}")

        try:
            regions, skipped = visual_extraction.extract_visual_regions(str(path), vcfg)
        except Exception as e:
            logger.warning(f"[visual] {doc_id}: region extraction failed: {type(e).__name__}: {e}")
            stats["failures"].append({"stage": "visual_crop_failed", "error": str(e)})
            return [], stats

        stats["found"] = len(regions) + len(skipped)
        stats["eligible"] = len(regions)
        stats["skipped_non_retrieval"] = len(skipped)
        if not regions:
            return [], stats

        digest = visual_analysis.get_model_digest(vcfg)
        pending: List[Dict[str, Any]] = []
        # Baseline captured before the first possible VLM call so the
        # post-unload VRAM barrier below has real evidence to compare
        # against (spec "VRAM BARRIER": never a fixed sleep, always
        # measured). `None` (no nvidia-smi) degrades to an unmeasured,
        # non-blocking barrier -- see gpu_lifecycle.wait_for_vram_release.
        vlm_baseline_mib = gpu_lifecycle.read_gpu_memory_used_mib()
        vlm_invoked = False
        try:
            for region_idx, region in enumerate(regions):
                _emit(progress, "visual_analysis", current=region_idx + 1, total=len(regions))
                visual_id = f"v-{doc_id}-{region.image_hash[:16]}"
                image_path = visual_extraction.save_crop(index_dir, region)
                association = visual_association.associate_visual(region.page, chunks)

                cached = visual_cache.get_cached_analysis(index_dir, region.image_hash, vcfg.model, vcfg.analysis_version)
                if cached is not None:
                    stats["cached"] += 1
                    analysis_dict = cached["analysis"]
                    retrieval_text = cached["retrieval_text"]
                else:
                    vlm_invoked = True
                    try:
                        analysis, raw_response = visual_analysis.analyze_visual(region.image_bytes, vcfg)
                    except visual_analysis.VisualAnalysisError as e:
                        stats["failed"] += 1
                        stats["failures"].append({
                            "stage": type(e).__name__, "page": region.page, "image_hash": region.image_hash,
                            "error": str(e),
                        })
                        logger.warning(f"[visual] {doc_id} p.{region.page}: {type(e).__name__}: {e}")
                        continue
                    stats["analyzed"] += 1
                    sizing = visual_analysis.build_retrieval_text_within_budget(
                        analysis, association.heading_path, region.page, count_tokens, max_seq_tokens,
                    )
                    retrieval_text = sizing.text
                    if sizing.reduction_steps_applied:
                        stats["token_reduced"] += 1
                        logger.info(
                            f"[visual] {doc_id} p.{region.page}: retrieval text reduced to fit "
                            f"active embedding tokenizer budget ({sizing.token_count}/{sizing.max_tokens} tokens "
                            f"after: {', '.join(sizing.reduction_steps_applied)})"
                        )
                    if not sizing.fits_budget:
                        logger.warning(
                            f"[visual] {doc_id} p.{region.page}: retrieval text still exceeds the active "
                            f"embedding tokenizer budget ({sizing.token_count}/{sizing.max_tokens} tokens) "
                            f"after every reduction step; embedding will likely reject it."
                        )
                    analysis_dict = dataclasses.asdict(analysis)
                    visual_cache.write_cached_analysis(
                        index_dir, region.image_hash, vcfg.model, vcfg.analysis_version,
                        raw_response, analysis_dict, retrieval_text,
                    )

                pending.append({
                    "visual_id": visual_id, "region": region, "image_path": image_path,
                    "association": association, "analysis_dict": analysis_dict,
                    "retrieval_text": retrieval_text, "analysis_model_digest": digest,
                })
        finally:
            # Explicitly unload the VLM even if the loop above raised
            # something unexpected (spec "FAILURE HANDLING") -- and
            # confirm it (Ollama's own `/api/ps`, then a real-nvidia-smi
            # VRAM barrier), never sleep-and-hope, before Stage 3 is
            # allowed to embed anything.
            if vcfg.unload_after_batch and vlm_invoked:
                visual_analysis.unload_model(vcfg)
                gpu_lifecycle.wait_for_ollama_model_unload(
                    vcfg.ollama_host, vcfg.model, timeout_s=vcfg.vram_barrier_timeout_s,
                )
                if vlm_baseline_mib is not None:
                    stats["vram_barrier"] = gpu_lifecycle.wait_for_vram_release(
                        vlm_baseline_mib, tolerance_mib=vcfg.vram_barrier_tolerance_mib,
                        timeout_s=vcfg.vram_barrier_timeout_s,
                    ).to_dict()

        return pending, stats

    def _embed_visuals(
        self, pending: List[Dict[str, Any]], stats: Optional[Dict[str, Any]], *,
        doc_id: str, source_path: str, source_hash: str, client: EmbeddingClient,
        progress: Optional[ProgressFn] = None,
    ) -> "tuple[List[Dict[str, Any]], Dict[str, Any]]":
        """Second half of STAGE 3 (embedding), run immediately after
        MAIN_TEXT's own `embed_batch` call in `index()` -- same
        embedding_client/config, no VLM call in between. Best-effort: an
        embedding failure here degrades to "this document has fewer/no
        visual records" (spec "FAILURE BEHAVIOR") without touching the
        already-written MAIN_TEXT embeddings."""
        stats = dict(stats) if stats is not None else {
            "enabled": True, "found": 0, "eligible": 0, "skipped_non_retrieval": 0,
            "analyzed": 0, "cached": 0, "failed": 0, "indexed": 0, "failures": [], "token_reduced": 0,
        }
        vcfg = self.config.visual
        records: List[Dict[str, Any]] = []
        visual_texts = [p["retrieval_text"] for p in pending]
        if not visual_texts:
            return records, stats

        # Batched the same way as the MAIN_TEXT embedding call in `index()`
        # (Part 1): a large document can carry enough eligible visuals that
        # one `embed_batch()` call over all of them hits the same
        # multi-hundred-MB/timeout problem. The existing all-or-nothing
        # oversized-input handling (comment above, at `_analyze_visuals`'s
        # tokenizer-resolution block) still applies -- any batch failing
        # degrades the whole visual set to "not indexed", same as before
        # batching.
        vis_embeddings: List[List[float]] = []
        try:
            batch_size = max(1, self.config.embedding.batch_size)
            for start in range(0, len(visual_texts), batch_size):
                batch = visual_texts[start:start + batch_size]
                vis_embeddings.extend(client.embed_batch(batch, input_type="document"))
                _emit(progress, "embedding", current=len(vis_embeddings), total=len(visual_texts))
        except (EmbeddingOversizedInputError, EmbeddingServiceError) as e:
            logger.warning(f"[visual] {doc_id}: embedding visual retrieval text failed: {e}")
            stats["failed"] = stats.get("failed", 0) + len(visual_texts)
            stats["failures"] = list(stats.get("failures") or []) + [{"stage": "visual_embedding_failed", "error": str(e)}]
            vis_embeddings = []

        if len(vis_embeddings) == len(visual_texts):
            for p, vec in zip(pending, vis_embeddings):
                region = p["region"]
                records.append(build_visual_record(
                    visual_id=p["visual_id"], document_id=doc_id, source_path=source_path,
                    source_hash=source_hash, page=region.page, bbox=region.bbox,
                    visual_type=p["analysis_dict"].get("visual_type", "other"),
                    heading_path=p["association"].heading_path,
                    linked_chunk_ids=p["association"].linked_chunk_ids,
                    image_path=p["image_path"], image_hash=region.image_hash,
                    analysis_model=vcfg.model, analysis_model_digest=p.get("analysis_model_digest"),
                    analysis_version=vcfg.analysis_version, retrieval_text=p["retrieval_text"],
                    analysis=p["analysis_dict"], document_order=region.order, embedding=vec,
                ))
            stats["indexed"] = len(records)

        return records, stats

    # ------------------------------------------------------------------ #
    # manual title override (title_overrides.py)
    #
    # A UI/document-metadata correction, not a second ingestion path: both
    # methods below persist the override (or its removal) and then rerun
    # the exact same `index()` this module already uses for every other
    # (re)processing request -- for this one `document_id` only -- so
    # MAIN_TEXT's H1, the rewritten Markdown/.md file, chunk header paths,
    # and the index/citation metadata all pick up the corrected title
    # together, the same invariant `index()` already keeps between the
    # renderer, the chunker, and the indexer for automatic title
    # resolution. `force=True` is required here (not merely `index()`'s
    # default): the source file and pipeline configuration are unchanged,
    # so `is_up_to_date` would otherwise skip re-embedding entirely, and
    # the override would only ever reach `title_overrides.json`, never the
    # actual index/Markdown/chunks it's supposed to correct.

    def get_title_override(self, document_id: str) -> Optional[str]:
        """The manually-set title for `document_id`, or None if it has none."""
        return _get_title_override(self.config.paths.index_dir, document_id)

    def set_title_override(
        self, source_path: str, title: str, document_id: Optional[str] = None,
        relative_source: Optional[str] = None, progress: Optional[ProgressFn] = None,
    ) -> ProcessResult:
        """Records `title` as `document_id`'s manual title override and
        immediately reprocesses that one source document so the override
        becomes the actual resolved MAIN_TEXT/index title, not just a
        recorded preference."""
        path, doc_id = self._resolve(source_path, document_id, relative_source)
        _set_title_override(self.config.paths.index_dir, doc_id, title)
        return self.index(str(path), document_id=doc_id, relative_source=relative_source, force=True,
                           progress=progress)

    def clear_title_override(
        self, source_path: str, document_id: Optional[str] = None,
        relative_source: Optional[str] = None,
    ) -> ProcessResult:
        """Removes `document_id`'s manual title override (if any) and
        reprocesses that one source document so it falls back to normal
        automatic title resolution."""
        path, doc_id = self._resolve(source_path, document_id, relative_source)
        _clear_title_override(self.config.paths.index_dir, doc_id)
        return self.index(str(path), document_id=doc_id, relative_source=relative_source, force=True)

    def delete(self, document_id: str, source_path: Optional[str] = None) -> None:
        candidates = [
            os.path.join(self.config.paths.index_dir, f"{document_id}.json"),
            os.path.join(self.config.paths.processed_dir, f"{document_id}.md"),
            os.path.join(self.config.paths.report_dir, f"{document_id}.report.md"),
        ]
        if source_path and self.config.delete_source_on_doc_delete and os.path.exists(source_path):
            candidates.append(source_path)
        for p in candidates:
            if os.path.exists(p):
                os.remove(p)

    def scan_and_index(self, root_dir: str, force: bool = False, profile: Optional[str] = None,
                        progress: Optional[ProgressFn] = None) -> List[Dict[str, Any]]:
        """Walk `root_dir` for supported files and index() each one, keyed by its
        path relative to `root_dir` so re-running is idempotent (spec section 14).

        Skips the pipeline's own working directories (index/processed/report/
        source dirs) so scanning a parent directory that happens to contain them
        -- e.g. the legacy docs/ tree containing docs/newly_formatted_docs_review/
        -- never double-indexes files already reachable through another root."""
        root = Path(root_dir)
        results: List[Dict[str, Any]] = []
        if not root.exists():
            return results
        managed_dirs = [
            Path(self.config.paths.index_dir), Path(self.config.paths.processed_dir),
            Path(self.config.paths.report_dir), Path(self.config.paths.source_dir),
        ]

        def _is_managed(p: Path) -> bool:
            return any(p == m or m in p.parents for m in managed_dirs)

        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if _is_managed(path):
                continue
            rel = str(path.relative_to(root))
            doc_id = stable_document_id(rel)
            _emit(progress, "scanning", source=rel)
            try:
                result = self.index(str(path), document_id=doc_id, relative_source=rel, force=force, profile=profile,
                                     progress=progress)
                results.append({
                    "source": rel, "document_id": doc_id,
                    "status": "skipped" if result.skipped_unchanged else "indexed",
                    "chunks": len(result.chunking_result.chunks),
                    "warnings": result.validation_report.stats["warnings"],
                    "errors": result.validation_report.stats["errors"],
                })
            except PipelineValidationError as e:
                results.append({"source": rel, "document_id": doc_id, "status": "failed", "error": str(e)})
            except Exception as e:
                results.append({"source": rel, "document_id": doc_id, "status": "failed", "error": f"{type(e).__name__}: {e}"})
        return results

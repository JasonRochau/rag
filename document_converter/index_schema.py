"""
Versioned index schema (spec section 11).

Schema version 2 stores one JSON object per chunk instead of parallel
`chunks[]` / `chunks_full[]` / `embeddings[]` arrays associated by `zip()`.
Writers always go through `write_index_atomic`, which writes to a temp file
in the same directory and `os.replace()`s it into place, so a crash or a
failed embedding pass can never leave a half-written index on disk, and a
previously-working index is never removed before its replacement is fully
built and validated.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .config import INDEX_SCHEMA_VERSION, PIPELINE_VERSION, CHUNKER_VERSION, PipelineConfig
from .chunker import ChunkRecord

# `content_kind` values distinguishing a MAIN_TEXT chunk from a
# VISUAL_ANALYSIS record inside the shared `chunks` array (spec Phase 7).
# MAIN_TEXT remains clean body text either way -- this field never changes
# what MAIN_TEXT *contains*, only how a retrieval record already living
# next to it in the index is labeled.
CONTENT_KIND_MAIN_TEXT = "main_text"
CONTENT_KIND_VISUAL_ANALYSIS = "visual_analysis"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def configuration_fingerprint(config: PipelineConfig) -> str:
    payload = json.dumps(config.configuration_fingerprint_payload(), sort_keys=True)
    return sha256_text(payload)


def embedding_configuration_fingerprint(
    embedding_info, chunk_target_tokens: Optional[int], chunk_max_tokens: Optional[int],
) -> str:
    """Identifies the exact embedding-side configuration a batch of vectors
    was produced under -- model id, resolved revision, dimension,
    normalization, query-prompt configuration, tokenizer identity, and the
    token-native chunk target/max -- everything BGE vectors and Qwen
    vectors (or two different Qwen configurations) could disagree on.
    Deliberately separate from `configuration_fingerprint` above (the
    pipeline-wide fingerprint used to decide whether a *source* needs
    reprocessing): this one is compared against the *live embedding
    service* at load/search time (`check_embedding_compatibility`) to
    refuse silently treating one model's vectors as another's, which
    `configuration_fingerprint` alone doesn't guard against -- it's computed
    from static config, not from what the live service actually resolved
    (revision, real dimension, whether a query prompt exists)."""
    payload = json.dumps({
        "model": embedding_info.model,
        "revision": embedding_info.revision,
        "dimension": embedding_info.dimension,
        "normalized_embeddings": embedding_info.normalized_embeddings,
        "query_prompt_name": embedding_info.query_prompt_name,
        "tokenizer_id": embedding_info.tokenizer_id,
        "chunk_target_tokens": chunk_target_tokens,
        "chunk_max_tokens": chunk_max_tokens,
    }, sort_keys=True)
    return sha256_text(payload)


def check_embedding_compatibility(
    payload: Dict[str, Any], embedding_info, chunk_target_tokens: Optional[int], chunk_max_tokens: Optional[int],
) -> Optional[str]:
    """Returns a human-readable mismatch reason if `payload` (an already-
    loaded index) was embedded under different embedding configuration than
    the live `embedding_info` + chunk token settings, or `None` if they
    match -- BGE vectors and Qwen vectors (or two Qwen configurations) must
    never be silently mixed into the same search space. Callers (index
    load/search paths) must skip/refuse a mismatched index rather than
    loading its vectors as if they were compatible.

    A payload with a recorded `config_fingerprint` (every index built since
    the embedding-model-migration schema landed) always takes the strict
    path below: exact fingerprint match or reject. A payload with no
    `config_fingerprint` at all -- e.g. the original production BGE index,
    built before that field existed -- is a *legacy* payload, not
    necessarily an incompatible one, and is handled by
    `_legacy_embedding_compatibility` instead: reject only on missing/
    ambiguous identity or an explicit contradiction, never merely for
    predating the field (see that function's docstring -- treating "no
    fingerprint" as automatically incompatible previously caused every
    legacy BGE document to be silently dropped from search even though
    they were genuinely BGE vectors)."""
    stored = payload.get("embedding", {})
    stored_fp = stored.get("config_fingerprint")

    if not stored_fp:
        return _legacy_embedding_compatibility(payload, stored, embedding_info)

    live_fp = embedding_configuration_fingerprint(embedding_info, chunk_target_tokens, chunk_max_tokens)
    if stored_fp == live_fp:
        return None
    if stored.get("model") and stored.get("model") != embedding_info.model:
        return (
            f"index was embedded with model {stored.get('model')!r}, active embedding service is "
            f"{embedding_info.model!r}"
        )
    if stored.get("dimension") and stored.get("dimension") != embedding_info.dimension:
        return (
            f"index vectors are {stored.get('dimension')}-dimensional, active embedding service produces "
            f"{embedding_info.dimension}-dimensional vectors"
        )
    return (
        f"index embedding configuration fingerprint {stored_fp!r} does not match the active "
        f"configuration's {live_fp!r}"
    )


def _legacy_embedding_compatibility(payload: Dict[str, Any], stored: Dict[str, Any], embedding_info) -> Optional[str]:
    """Backward-compatibility check for a pre-`config_fingerprint` index
    (schema_version 2, but written before the embedding-model-migration
    metadata existed -- `stored` has at most `provider`/`model`/`dimension`/
    `max_sequence_tokens`, none of `normalized_embeddings`/
    `query_prompt_name`/`tokenizer_id`/`config_fingerprint`).

    Accept only when everything the legacy payload DOES record affirmatively
    matches the live embedding service; fail closed the moment identity is
    missing/ambiguous or something recorded actually contradicts the live
    service. This is deliberately narrower than the strict-fingerprint path
    (it can't see revision, normalization, query-prompt config, or chunk
    token budget at all), but "we can't see everything" is not license to
    accept blindly -- it only accepts what it CAN verify."""
    model = stored.get("model")
    dimension = stored.get("dimension")
    if not model or not dimension:
        return (
            "legacy index does not record enough embedding identity (model/dimension) to safely verify "
            "compatibility -- refusing to load an index of unknown embedding space"
        )
    if model != embedding_info.model:
        return (
            f"index was embedded with model {model!r}, active embedding service is {embedding_info.model!r}"
        )
    if dimension != embedding_info.dimension:
        return (
            f"index vectors are {dimension}-dimensional, active embedding service produces "
            f"{embedding_info.dimension}-dimensional vectors"
        )

    # The declared `dimension` field could in principle disagree with what's
    # actually stored on the chunks -- the vectors themselves are the ground
    # truth for which embedding space they occupy, so check them directly
    # rather than trusting metadata alone.
    actual_dims = {len(c["embedding"]) for c in payload.get("chunks", []) if c.get("embedding")}
    if actual_dims and actual_dims != {embedding_info.dimension}:
        return (
            f"index chunk vectors are actually {sorted(actual_dims)}-dimensional, which does not match "
            f"the active embedding service's {embedding_info.dimension}-dimensional output"
        )

    # Any other legacy field that IS present must not contradict the live
    # service -- absence of a field is "no information recorded" and is
    # never itself a mismatch, but a recorded value that disagrees is a
    # hard rejection.
    for field, live_value in (
        ("normalized_embeddings", embedding_info.normalized_embeddings),
        ("query_prompt_name", embedding_info.query_prompt_name),
        ("tokenizer_id", embedding_info.tokenizer_id),
    ):
        stored_value = stored.get(field)
        if stored_value is not None and stored_value != live_value:
            return (
                f"legacy index recorded {field}={stored_value!r}, active embedding service reports "
                f"{field}={live_value!r}"
            )

    return None


def build_index_payload(
    document_id: str,
    title: str,
    source_path: str,
    markdown_path: str,
    report_path: str,
    source_hash: str,
    normalized_hash: str,
    extractor: str,
    ocr_used: bool,
    semantic_structuring_used: bool,
    config_fingerprint: str,
    embedding_info,
    chunks: List[ChunkRecord],
    embeddings: List[List[float]],
    retrieval_ready: bool,
    warnings: int,
    errors: int,
    created_at: Optional[str] = None,
    profile: str = "main_text",
    chunk_target_tokens: Optional[int] = None,
    chunk_max_tokens: Optional[int] = None,
    title_source: str = "",
    findings: Optional[List[Dict[str, Any]]] = None,
    stats: Optional[Dict[str, Any]] = None,
    visual_records: Optional[List[Dict[str, Any]]] = None,
    visual_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    chunk_payload = []
    for c, vec in zip(chunks, embeddings):
        chunk_payload.append({
            "content_kind": CONTENT_KIND_MAIN_TEXT,
            "chunk_id": c.chunk_id,
            "chunk_index": c.chunk_index,
            "header_path": c.header_path,
            "heading": c.heading,
            "provenance": c.provenance,
            "text": c.text,
            "body": c.body,
            "word_count": c.word_count,
            "token_count": c.token_count,
            "source_block_ids": c.source_block_ids,
            "source_pages": c.source_pages,
            "content_types": c.content_types,
            "forced_fallback": c.forced_fallback,
            "split_provenance": c.split_provenance,
            "embedding": vec,
        })

    # VISUAL_ANALYSIS records (spec Phase 7) are appended into the SAME
    # `chunks` array, distinguished only by `content_kind` -- not a parallel
    # array -- so every existing reader that already iterates
    # `payload["chunks"]` for retrieval (DocStore.reload/HybridRanker,
    # eval_retrieval.py) picks them up automatically instead of needing a
    # second read path. Each entry here is already a plain dict (built by
    # `build_visual_record` below), not a `ChunkRecord`, since a visual has
    # no chunk-splitting/provenance concepts of its own.
    chunk_payload.extend(visual_records or [])

    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "document": {
            "document_id": document_id,
            "title": title,
            # Where `title` came from ("manual_override" | "metadata" |
            # "front_matter_zone" | "first_body_heading" | "unresolved" | "";
            # see pipeline.py's `_prepare_for_profile`) -- lets the UI/API
            # distinguish a user-corrected title from an automatically
            # resolved one without re-deriving it.
            "title_source": title_source,
            "source_path": source_path,
            "markdown_path": markdown_path,
            "report_path": report_path,
            "source_hash": source_hash,
            "normalized_hash": normalized_hash,
            "created_at": created_at or now,
            "updated_at": now,
            "profile": profile,
        },
        "pipeline": {
            "pipeline_version": PIPELINE_VERSION,
            "configuration_hash": config_fingerprint,
            "extractor": extractor,
            "ocr_used": ocr_used,
            "semantic_structuring_used": semantic_structuring_used,
            "chunker_version": CHUNKER_VERSION,
        },
        "embedding": {
            "provider": embedding_info.provider,
            "model": embedding_info.model,
            "revision": embedding_info.revision,
            "dimension": embedding_info.dimension,
            "max_sequence_tokens": embedding_info.max_sequence_tokens,
            "normalized_embeddings": embedding_info.normalized_embeddings,
            "query_prompt_name": embedding_info.query_prompt_name,
            "tokenizer_id": embedding_info.tokenizer_id,
            "chunk_target_tokens": chunk_target_tokens,
            "chunk_max_tokens": chunk_max_tokens,
            "config_fingerprint": embedding_configuration_fingerprint(
                embedding_info, chunk_target_tokens, chunk_max_tokens,
            ),
        },
        "validation": {
            "retrieval_ready": retrieval_ready,
            "warnings": warnings,
            "errors": errors,
            # Full diagnostic detail (spec: document-processing diagnostics UI) --
            # the same `FindingLog.to_list()` / `ValidationReport.stats` the
            # convert-time preview/upload response already returns, persisted
            # here too so `/docs/{doc_id}` can serve it for an already-indexed
            # document without ever re-running the pipeline just to look at it.
            # Optional/omittable by callers (e.g. legacy index rebuilds that
            # don't have a report at hand) -- absence just means "no detailed
            # diagnostics recorded for this index entry", never a fabricated one.
            "findings": findings if findings is not None else [],
            "stats": stats if stats is not None else {},
        },
        # Per-document visual-analysis ingestion stats (spec Phase 2/14's
        # "track failures explicitly" / "prove no visual re-runs the VLM
        # unnecessarily") -- counts only, the records themselves are the
        # `content_kind == "visual_analysis"` entries inside `chunks` above.
        # Always present (possibly all-zero) so a reader never has to
        # special-case "visual analysis was never attempted for this
        # document" vs. "it ran and found nothing".
        "visual": visual_stats if visual_stats is not None else {
            "enabled": False, "found": 0, "eligible": 0, "skipped_non_retrieval": 0,
            "analyzed": 0, "cached": 0, "failed": 0, "indexed": 0,
        },
        "chunks": chunk_payload,
    }


def build_visual_record(
    *,
    visual_id: str,
    document_id: str,
    source_path: str,
    source_hash: str,
    page: int,
    bbox: List[float],
    visual_type: str,
    heading_path: List[str],
    linked_chunk_ids: List[str],
    image_path: str,
    image_hash: str,
    analysis_model: str,
    analysis_model_digest: Optional[str],
    analysis_version: str,
    retrieval_text: str,
    analysis: Dict[str, Any],
    document_order: int,
    embedding: List[float],
) -> Dict[str, Any]:
    """One VISUAL_ANALYSIS retrieval record (spec Phase 7's field list),
    shaped to slot directly into `payload["chunks"]` alongside ordinary
    MAIN_TEXT chunk dicts. Field names are reused from the MAIN_TEXT chunk
    schema wherever the concept is equivalent (`chunk_id`/`text`/
    `header_path`/`source_pages`/`embedding`) instead of inventing
    parallel names for the same thing, per spec: "do not duplicate an
    existing field under a new name unnecessarily"."""
    return {
        "content_kind": CONTENT_KIND_VISUAL_ANALYSIS,
        # Reused MAIN_TEXT-chunk-shaped fields, so every existing consumer
        # of `chunks[]` (HybridRanker, DocStore) needs zero special-casing
        # to rank/display this record.
        "chunk_id": visual_id,
        "chunk_index": document_order,
        "header_path": heading_path,
        "heading": heading_path[-1] if heading_path else "",
        "provenance": f"[{os.path.basename(source_path)} > p.{page} > visual {visual_id}]",
        "text": retrieval_text,
        "body": retrieval_text,
        "word_count": len(retrieval_text.split()),
        "token_count": None,
        "source_block_ids": [],
        "source_pages": [page],
        "content_types": ["VisualAnalysis"],
        "forced_fallback": False,
        "split_provenance": None,
        "embedding": embedding,
        # Visual-specific provenance/debug/cache-validation fields (spec
        # "VISUAL RECORD IDENTITY" / "INDEX SCHEMA").
        "visual_id": visual_id,
        "document_id": document_id,
        "source_path": source_path,
        "source_hash": source_hash,
        "page": page,
        "bbox": bbox,
        "visual_type": visual_type,
        "heading_path": heading_path,
        "linked_chunk_ids": linked_chunk_ids,
        "image_path": image_path,
        "image_hash": image_hash,
        "analysis_model": analysis_model,
        "analysis_model_digest": analysis_model_digest,
        "analysis_version": analysis_version,
        "analysis": analysis,
    }


def write_index_atomic(index_dir: str, document_id: str, payload: Dict[str, Any]) -> str:
    os.makedirs(index_dir, exist_ok=True)
    final_path = os.path.join(index_dir, f"{document_id}.json")
    tmp_path = os.path.join(index_dir, f".{document_id}.json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)
    return final_path


def load_index(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_legacy_schema(payload: Dict[str, Any]) -> bool:
    return payload.get("schema_version") != INDEX_SCHEMA_VERSION


def is_up_to_date(payload: Dict[str, Any], source_hash: str, config_fingerprint: str) -> bool:
    if is_legacy_schema(payload):
        return False
    doc = payload.get("document", {})
    pipeline = payload.get("pipeline", {})
    return doc.get("source_hash") == source_hash and pipeline.get("configuration_hash") == config_fingerprint

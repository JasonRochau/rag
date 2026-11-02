"""
Filesystem cache for Qwen3-VL visual analyses (spec "Phase 5").

Cache identity is the crop's content hash plus the exact analysis contract
that produced a result for it (`model` + `analysis_version`, see
`config.VISUAL_ANALYSIS_VERSION`) -- encoded directly into the cache
filename, so "is this cached?" is a single `os.path.exists()` check, and
changing the model or bumping the analysis/prompt version can never
silently reuse a result produced under a different contract (a new
filename, not a stale one, is looked up).

Lives under `{index_dir}/_visuals/`, the same subdirectory
`visual_extraction.save_crop` writes rendered crops into -- one content-
addressed image file per distinct crop, one analysis file per (image,
model, version) triple. Both keyed off `image_hash`, both benefiting from
the same "subdirectory, not a top-level file" reasoning `title_overrides.py`
already documents (kept out of the `docs/.index/*.json` glob every index
reader assumes matches only versioned per-document index payloads).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

_MODEL_TAG_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]+")


def _safe_model_tag(model: str) -> str:
    return _MODEL_TAG_SAFE_RE.sub("_", model)


def _analysis_cache_path(index_dir: str, image_hash: str, model: str, analysis_version: str) -> str:
    tag = _safe_model_tag(model)
    fname = f"{image_hash}__{tag}__v{analysis_version}.json"
    return os.path.join(index_dir, "_visuals", fname)


def get_cached_analysis(
    index_dir: str, image_hash: str, model: str, analysis_version: str,
) -> Optional[Dict[str, Any]]:
    """Returns the cached `{"raw_response": ..., "analysis": {...},
    "retrieval_text": ...}` payload for this exact (image, model, version)
    triple, or `None` on any cache miss/corruption (a corrupt cache file is
    treated as a miss -- re-running the VLM is always safe, never fatal)."""
    path = _analysis_cache_path(index_dir, image_hash, model, analysis_version)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "analysis" not in data or "retrieval_text" not in data:
        return None
    return data


def write_cached_analysis(
    index_dir: str, image_hash: str, model: str, analysis_version: str,
    raw_response: str, analysis: Dict[str, Any], retrieval_text: str,
) -> str:
    path = _analysis_cache_path(index_dir, image_hash, model, analysis_version)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "image_hash": image_hash,
        "model": model,
        "analysis_version": analysis_version,
        "raw_response": raw_response,
        "analysis": analysis,
        "retrieval_text": retrieval_text,
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    return path

"""
Optional local semantic structuring (spec section 8).

Never asked to rewrite content. The model only ever returns a JSON "plan"
that assigns headings and groups *existing* stable block IDs into a
hierarchy; every block referenced must come from the source document, be
used exactly once, and stay in original source order. Any plan that fails
that contract is rejected outright and the pipeline falls back to the
deterministic structure produced by extraction + normalization.

On large documents the eligible block list is split into character-budgeted
batches (``SemanticStructuringConfig.batch_char_budget``) and the model is
called once per batch; the per-batch plans are merged in batch order and the
merged plan is held to exactly the same global validation contract as a
single-call plan. Any batch-level failure (request error, unparseable
response) triggers the same deterministic fallback a failed single call does.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from .model import Document, HeadingBlock
from .findings import FindingLog
from .config import SemanticStructuringConfig, ChunkingConfig
from .chunker import build_sections

_GENERIC_HEADINGS = {
    "introduction", "overview", "summary", "untitled", "notes", "misc",
    "chapter 1", "section 1", "details", "content", "info",
}

PLAN_SYSTEM_PROMPT = """You are structuring a technical document into sections for retrieval.
You will be given a JSON array of content blocks, each with a stable "id" and a text "preview".
Return ONLY a JSON object with this exact shape:
{
  "title": "<document title, descriptive>",
  "sections": [
    {
      "level": 2,
      "heading": "<descriptive heading containing retrieval-relevant terms>",
      "block_ids": ["b0001", "b0002"],
      "children": [
        {"level": 3, "heading": "...", "block_ids": ["b0003"], "children": []}
      ]
    }
  ]
}
Rules you MUST follow:
- Use EVERY block id exactly once across the whole plan (top-level block_ids plus all nested children).
- Never invent a block id that was not given to you.
- Never reorder blocks: within the flattened reading order of your plan, block ids must appear in the
  same relative order they were given to you.
- Headings must be levels 2 through 6. A child's level must be exactly one greater than its parent
  (top-level sections are level 2).
- Do NOT write body text, do NOT summarize, do NOT omit any block. You only choose groupings and headings.
- Prefer descriptive headings ("Recursive DNS resolution steps") over generic ones ("Overview").
"""


@dataclass
class StructuringOutcome:
    document: Document
    used: bool
    findings: FindingLog


def needs_structuring(document: Document, chunking_config: ChunkingConfig) -> bool:
    headings = [b for b in document.blocks if isinstance(b, HeadingBlock)]
    total_words = sum(b.word_count for b in document.blocks)
    if len(headings) <= 1 and total_words > 400:
        return True
    generic = sum(1 for h in headings if h.text.strip().lower() in _GENERIC_HEADINGS)
    if headings and generic / len(headings) > 0.5:
        return True
    sections = build_sections(document)
    for s in sections:
        if sum(b.word_count for b in s.content_blocks) > chunking_config.target_words * 1.8:
            return True
    return False


def _eligible_blocks(document: Document) -> List[Tuple[str, str, str]]:
    """Returns (block_id, kind, preview) for every non-heading block, in source order."""
    out = []
    for b in document.blocks:
        if isinstance(b, HeadingBlock):
            continue
        text = getattr(b, "text", None) or b.normalized or b.raw or ""
        out.append((b.block_id, type(b).__name__, text[:200]))
    return out


def _batch_eligible_blocks(eligible: List[Tuple[str, str, str]],
                            char_budget: int) -> List[List[Tuple[str, str, str]]]:
    """Greedily packs eligible blocks into batches whose serialized preview
    JSON stays under ``char_budget`` characters.

    The budget is measured against the exact ``{"id", "kind", "preview"}``
    item JSON that will be sent to the model (plus separators/brackets), so
    the payload the model receives is what is budgeted, not an estimate.
    Source order is preserved: batches are consecutive slices of ``eligible``.
    A single block whose own serialized item exceeds the budget is placed
    alone in its own batch rather than dropped -- the budget bounds batch
    size, never individual blocks.
    """
    batches: List[List[Tuple[str, str, str]]] = []
    current: List[Tuple[str, str, str]] = []
    current_chars = 2  # the "[" and "]" of the JSON array itself
    for entry in eligible:
        item_chars = len(json.dumps({"id": entry[0], "kind": entry[1], "preview": entry[2]})) + 1  # comma separator
        if current and current_chars + item_chars > char_budget:
            batches.append(current)
            current, current_chars = [], 2
        current.append(entry)
        current_chars += item_chars
    if current:
        batches.append(current)
    return batches


def _call_ollama_chat(host: str, model: str, prompt: str, blocks_json: str,
                       temperature: float, timeout: float) -> str:
    import requests
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": blocks_json},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
    }
    resp = requests.post(host.rstrip("/") + "/api/chat", json=body, timeout=timeout)
    resp.raise_for_status()
    out = resp.json()
    return (out.get("message", {}) or {}).get("content", "") or ""


def _safe_json(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        a, b = raw.find("{"), raw.rfind("}")
        if a != -1 and b != -1 and b > a:
            try:
                return json.loads(raw[a:b + 1])
            except Exception:
                return None
    return None


def _flatten_plan(sections: List[dict]) -> List[Tuple[int, str, List[str]]]:
    """Returns [(level, heading, block_ids), ...] in traversal order."""
    out: List[Tuple[int, str, List[str]]] = []

    def walk(nodes, parent_level):
        for node in nodes:
            level = node.get("level")
            heading = node.get("heading")
            block_ids = node.get("block_ids", [])
            out.append((level, heading, block_ids))
            walk(node.get("children") or [], level)

    walk(sections, 1)
    return out


def validate_plan(plan: dict, eligible_ids_in_order: List[str], log: FindingLog) -> bool:
    if not isinstance(plan, dict) or not plan.get("title") or not isinstance(plan.get("sections"), list):
        log.error("semantic_plan_malformed", "Model response was not a valid structure plan object")
        return False

    flat = _flatten_plan(plan["sections"])
    if not flat:
        log.error("semantic_plan_empty", "Model returned a plan with no sections")
        return False

    seen: Dict[str, int] = {}
    order: List[str] = []
    ok = True
    parent_levels_stack = [1]

    def walk(nodes, parent_level):
        nonlocal ok
        for node in nodes:
            level = node.get("level")
            if not isinstance(level, int) or not (2 <= level <= 6) or level != parent_level + 1:
                log.error("semantic_plan_invalid_level",
                          f"Heading {node.get('heading')!r} has invalid level {level} under parent level {parent_level}")
                ok = False
            for bid in node.get("block_ids", []):
                if bid not in eligible_ids_in_order:
                    log.error("semantic_plan_unknown_block_id", f"Plan references unknown block id {bid!r}")
                    ok = False
                    continue
                seen[bid] = seen.get(bid, 0) + 1
                order.append(bid)
            walk(node.get("children") or [], level if isinstance(level, int) else parent_level + 1)

    walk(plan["sections"], 1)

    for bid, count in seen.items():
        if count > 1:
            log.error("semantic_plan_duplicate_block_id", f"Plan uses block id {bid!r} {count} times")
            ok = False

    missing = [b for b in eligible_ids_in_order if b not in seen]
    if missing:
        log.error("semantic_plan_missing_block_id", f"Plan omits {len(missing)} source block(s): {missing[:5]}...")
        ok = False

    if ok:
        expected_order = [b for b in eligible_ids_in_order if b in seen]
        if order != expected_order:
            log.error("semantic_plan_reordered_blocks", "Plan changes the source order of blocks, which is not permitted")
            ok = False

    return ok


def _apply_plan(document: Document, plan: dict) -> Document:
    by_id = {b.block_id: b for b in document.blocks if not isinstance(b, HeadingBlock)}
    counter = [0]

    def synthetic_id() -> str:
        counter[0] += 1
        return f"bh{counter[0]:04d}"

    new_blocks = []
    title = plan["title"].strip()
    new_blocks.append(HeadingBlock(
        block_id=synthetic_id(), order=0, source_format=document.metadata.source_format,
        level=1, text=title, normalized=title,
    ))

    def emit(nodes):
        for node in nodes:
            heading = node["heading"].strip()
            new_blocks.append(HeadingBlock(
                block_id=synthetic_id(), order=0, source_format=document.metadata.source_format,
                level=node["level"], text=heading, normalized=heading,
            ))
            for bid in node.get("block_ids", []):
                new_blocks.append(by_id[bid])
            emit(node.get("children") or [])

    emit(plan["sections"])
    for i, b in enumerate(new_blocks):
        b.order = i
    document.blocks = new_blocks
    document.metadata.title = title
    return document


def apply_semantic_structuring(document: Document, config: SemanticStructuringConfig,
                                chunking_config: ChunkingConfig) -> StructuringOutcome:
    log = FindingLog()
    if not config.enabled:
        return StructuringOutcome(document=document, used=False, findings=log)
    if config.mode == "auto" and not needs_structuring(document, chunking_config):
        log.info("semantic_structuring_skipped", "Deterministic structure already looks adequate; semantic structuring skipped (mode=auto)")
        return StructuringOutcome(document=document, used=False, findings=log)

    eligible = _eligible_blocks(document)
    eligible_ids_in_order = [e[0] for e in eligible]
    if not eligible:
        log.info("semantic_structuring_skipped", "No content blocks to structure")
        return StructuringOutcome(document=document, used=False, findings=log)

    batches = _batch_eligible_blocks(eligible, config.batch_char_budget)
    merged_title: Optional[str] = None
    merged_sections: List[dict] = []

    for batch_index, batch in enumerate(batches):
        blocks_payload = json.dumps([{"id": bid, "kind": kind, "preview": prev} for bid, kind, prev in batch])
        if len(batches) > 1:
            # The plan prompt demands "EVERY block id exactly once"; with a
            # multi-batch document the model must be told it is looking at a
            # consecutive slice so it structures only these blocks (and so a
            # section heading that spans the batch boundary reads sensibly).
            blocks_payload = (f"You are looking at batch {batch_index + 1} of {len(batches)} "
                              f"of one document's blocks, in source order. Structure only the "
                              f"blocks given below; use every id given here exactly once.\n"
                              + blocks_payload)
        try:
            raw = _call_ollama_chat(config.host, config.model, PLAN_SYSTEM_PROMPT, blocks_payload,
                                     config.temperature, config.timeout_seconds)
        except Exception as e:
            log.warning("semantic_structuring_unavailable", f"Semantic structuring request failed: {e}; falling back to deterministic structure")
            return StructuringOutcome(document=document, used=False, findings=log)

        batch_plan = _safe_json(raw)
        if batch_plan is None:
            log.warning("semantic_plan_unparseable", "Model response was not valid JSON; falling back to deterministic structure")
            return StructuringOutcome(document=document, used=False, findings=log)

        if merged_title is None and batch_plan.get("title"):
            merged_title = batch_plan["title"]
        sections = batch_plan.get("sections")
        if isinstance(sections, list):
            merged_sections.extend(sections)

    # Merged in batch order, so the flattened block-id sequence is still the
    # source order; the existing global validation below enforces the exact
    # same contract as a single-call plan (every id once, in source order).
    plan = {"title": merged_title or document.metadata.title or "Untitled",
            "sections": merged_sections}
    plan_log = FindingLog()
    if not validate_plan(plan, eligible_ids_in_order, plan_log):
        log.extend(plan_log)
        log.warning("semantic_structuring_rejected", "Model plan failed validation; falling back to deterministic structure")
        return StructuringOutcome(document=document, used=False, findings=log)

    new_document = _apply_plan(document, plan)
    log.info("semantic_structuring_applied", f"Applied semantic structuring plan with {len(_flatten_plan(plan['sections']))} sections")
    return StructuringOutcome(document=new_document, used=True, findings=log)

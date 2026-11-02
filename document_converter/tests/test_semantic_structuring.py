"""Tests for semantic_structuring batching: the eligible block list is split
into character-budgeted batches, one Ollama call per batch, per-batch plans
merged in batch order, and the merged plan held to the same global
validation contract as the legacy single-call path. Any batch failure must
fall back to the deterministic structure exactly like a failed single call.
"""
from __future__ import annotations

import json

from document_converter import semantic_structuring as ss
from document_converter.config import ChunkingConfig, SemanticStructuringConfig
from document_converter.model import Document, DocumentMetadata, HeadingBlock, ParagraphBlock


def _make_document(n_blocks: int, preview_len: int = 60) -> Document:
    """A heading-free document of `n_blocks` paragraphs, each with a
    `preview_len`-character body, so `_eligible_blocks` previews are
    predictable in size for the char-budget batching math."""
    doc = Document(metadata=DocumentMetadata(source_path="doc.txt", source_format="txt"))
    for i in range(n_blocks):
        body = f"block {i:04d} " + ("x" * preview_len)
        doc.add(ParagraphBlock(
            block_id=f"b{i + 1:04d}", order=i, source_format="txt",
            raw=body, normalized=body, text=body,
        ))
    return doc


def _make_config(batch_char_budget: int) -> SemanticStructuringConfig:
    return SemanticStructuringConfig(
        mode="always", host="http://localhost:11434", model="fake-model",
        batch_char_budget=batch_char_budget,
    )


def _ids_from_payload(blocks_json: str) -> list:
    """Extracts the block ids from a payload that may carry a batch-position
    note prepended before the JSON array."""
    start = blocks_json.index("[")
    return [b["id"] for b in json.loads(blocks_json[start:])]


def _fake_ollama_call(calls: list):
    """Returns a `_call_ollama_chat` stand-in that answers each batch with a
    valid plan covering exactly the ids it was given, in order."""
    def fake(host, model, prompt, blocks_json, temperature, timeout):
        ids = _ids_from_payload(blocks_json)
        calls.append(blocks_json)
        return json.dumps({
            "title": "Structured Document",
            "sections": [{
                "level": 2,
                "heading": f"Section for {ids[0]} through {ids[-1]}",
                "block_ids": ids,
                "children": [],
            }],
        })
    return fake


def test_multi_batch_plans_merge_into_one_globally_valid_plan(monkeypatch):
    """A small char budget forces >=3 batches; each fake per-batch plan is
    valid locally, and the merged plan must pass the global validate_plan
    (every block id exactly once, in source order) so the plan is applied."""
    calls: list = []
    monkeypatch.setattr(ss, "_call_ollama_chat", _fake_ollama_call(calls))

    doc = _make_document(n_blocks=10)
    original_ids = [b.block_id for b in doc.blocks]
    config = _make_config(batch_char_budget=300)  # ~2-3 blocks per batch
    outcome = ss.apply_semantic_structuring(doc, config, ChunkingConfig())

    assert len(calls) >= 3, f"expected at least 3 batches, got {len(calls)}"
    # Every batch call carries the batch-position note and covers a
    # consecutive slice of the source-order ids.
    for i, payload in enumerate(calls):
        assert f"batch {i + 1} of {len(calls)}" in payload
    batched_ids = [bid for payload in calls for bid in _ids_from_payload(payload)]
    assert batched_ids == original_ids

    assert outcome.used is True
    assert {f.code for f in outcome.findings.findings} >= {"semantic_structuring_applied"}
    # The applied plan kept every original block exactly once, in source
    # order, interleaved with synthetic headings, and set the title.
    new_ids = [b.block_id for b in outcome.document.blocks
               if not isinstance(b, HeadingBlock)]
    assert new_ids == original_ids
    assert outcome.document.metadata.title == "Structured Document"
    assert any(isinstance(b, HeadingBlock) and b.level == 2
               for b in outcome.document.blocks)


def test_single_batch_payload_is_unchanged_from_legacy(monkeypatch):
    """A document that fits in one batch keeps the exact legacy payload
    shape: a bare JSON array with no batch-position note prepended."""
    calls: list = []
    monkeypatch.setattr(ss, "_call_ollama_chat", _fake_ollama_call(calls))

    doc = _make_document(n_blocks=3)
    config = _make_config(batch_char_budget=24000)
    outcome = ss.apply_semantic_structuring(doc, config, ChunkingConfig())

    assert len(calls) == 1
    assert calls[0].lstrip().startswith("[")
    assert outcome.used is True


def test_failing_batch_falls_back_like_a_failed_single_call(monkeypatch):
    """An exception from any batch's request must produce the same outcome
    as a failed single call today: `semantic_structuring_unavailable`
    warning, `used=False`, document untouched."""
    def failing(host, model, prompt, blocks_json, temperature, timeout):
        raise ConnectionError("ollama is down")

    monkeypatch.setattr(ss, "_call_ollama_chat", failing)

    doc = _make_document(n_blocks=10)
    original_blocks = list(doc.blocks)
    config = _make_config(batch_char_budget=300)
    outcome = ss.apply_semantic_structuring(doc, config, ChunkingConfig())

    assert outcome.used is False
    assert outcome.document.blocks == original_blocks
    warnings = [f for f in outcome.findings.findings
                if f.code == "semantic_structuring_unavailable"]
    assert len(warnings) == 1


def test_failing_second_batch_still_falls_back(monkeypatch):
    """The fallback fires even when an earlier batch already succeeded --
    partial plans are never applied."""
    calls: list = []

    def fail_on_second(host, model, prompt, blocks_json, temperature, timeout):
        calls.append(blocks_json)
        if len(calls) == 2:
            raise TimeoutError("second batch timed out")
        ids = _ids_from_payload(blocks_json)
        return json.dumps({
            "title": "Structured Document",
            "sections": [{"level": 2, "heading": "S", "block_ids": ids, "children": []}],
        })

    monkeypatch.setattr(ss, "_call_ollama_chat", fail_on_second)

    doc = _make_document(n_blocks=10)
    original_blocks = list(doc.blocks)
    config = _make_config(batch_char_budget=300)
    outcome = ss.apply_semantic_structuring(doc, config, ChunkingConfig())

    assert len(calls) == 2  # stopped at the failing batch
    assert outcome.used is False
    assert outcome.document.blocks == original_blocks
    assert any(f.code == "semantic_structuring_unavailable"
               for f in outcome.findings.findings)


def test_unparseable_batch_response_falls_back(monkeypatch):
    """A batch whose response is not JSON hits the same
    `semantic_plan_unparseable` fallback as the single-call path."""
    monkeypatch.setattr(ss, "_call_ollama_chat",
                        lambda *a, **kw: "this is not json at all")

    doc = _make_document(n_blocks=10)
    original_blocks = list(doc.blocks)
    config = _make_config(batch_char_budget=300)
    outcome = ss.apply_semantic_structuring(doc, config, ChunkingConfig())

    assert outcome.used is False
    assert outcome.document.blocks == original_blocks
    assert any(f.code == "semantic_plan_unparseable"
               for f in outcome.findings.findings)


def test_batch_eligible_blocks_respects_char_budget():
    """Each batch's serialized payload stays under the budget, batches are
    consecutive source-order slices, and an oversized single block still
    goes out alone in its own batch rather than being dropped."""
    doc = _make_document(n_blocks=8, preview_len=100)
    eligible = ss._eligible_blocks(doc)

    budget = 500
    batches = ss._batch_eligible_blocks(eligible, budget)
    assert len(batches) > 1
    for batch in batches:
        payload = json.dumps([{"id": bid, "kind": kind, "preview": prev}
                              for bid, kind, prev in batch])
        assert len(payload) <= budget
    assert [e for batch in batches for e in batch] == eligible

    oversized = [(f"b{i:04d}", "ParagraphBlock", "y" * 2000) for i in range(1, 4)]
    batches = ss._batch_eligible_blocks(oversized, 500)
    assert [len(b) for b in batches] == [1, 1, 1]
    assert [e for batch in batches for e in batch] == oversized


def test_semantic_batch_char_budget_env_override(monkeypatch):
    """The new config knob is env-overridable like its siblings and parses
    as an int through load_config's override table."""
    from document_converter import config as config_module

    monkeypatch.setattr(config_module.PathsConfig, "ensure", lambda self: None)
    monkeypatch.setenv("SEMANTIC_BATCH_CHAR_BUDGET", "12345")
    cfg = config_module.load_config(config_path="/nonexistent/docpipe.yaml")
    assert cfg.semantic.batch_char_budget == 12345
    assert isinstance(cfg.semantic.batch_char_budget, int)

    assert config_module._ENV_OVERRIDES["SEMANTIC_BATCH_CHAR_BUDGET"] == ("semantic", "batch_char_budget")
    assert "batch_char_budget" in config_module._INT_FIELDS

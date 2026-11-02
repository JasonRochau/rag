import json
import os

from document_converter.chunker import ChunkRecord
from document_converter.embedding_client import EmbeddingModelInfo
from document_converter.index_schema import (
    build_index_payload, write_index_atomic, load_index, is_up_to_date, is_legacy_schema, sha256_text,
    embedding_configuration_fingerprint, check_embedding_compatibility,
)


def _chunk():
    return ChunkRecord(
        chunk_id="c-1", chunk_index=0, header_path=["T"], heading="T", provenance="[x.md > T]",
        text="[x.md > T]\n\n# T\n\nhello", body="# T\n\nhello", word_count=3, token_count=4,
        source_block_ids=["b0001"], source_pages=[],
    )


def _info():
    return EmbeddingModelInfo(provider="fake", model="m", dimension=2, max_sequence_tokens=512, normalized_embeddings=True)


def _qwen_info(**overrides):
    defaults = dict(
        provider="sentence-transformers-service", model="Qwen/Qwen3-Embedding-4B", dimension=2560,
        max_sequence_tokens=40960, normalized_embeddings=True,
        revision="5cf2132abc99cad020ac570b19d031efec650f2b", query_prompt_name="query",
        tokenizer_id="Qwen/Qwen3-Embedding-4B",
    )
    defaults.update(overrides)
    return EmbeddingModelInfo(**defaults)


def test_write_index_atomic_survives_and_reloads(tmp_path):
    payload = build_index_payload(
        document_id="d-1", title="T", source_path="x.md", markdown_path="x.md", report_path="x.report.md",
        source_hash=sha256_text("abc"), normalized_hash=sha256_text("xyz"), extractor="markdown",
        ocr_used=False, semantic_structuring_used=False, config_fingerprint="fp1", embedding_info=_info(),
        chunks=[_chunk()], embeddings=[[0.1, 0.2]], retrieval_ready=True, warnings=0, errors=0,
    )
    path = write_index_atomic(str(tmp_path), "d-1", payload)
    assert os.path.exists(path)
    reloaded = load_index(path)
    assert reloaded["document"]["document_id"] == "d-1"
    assert reloaded["chunks"][0]["embedding"] == [0.1, 0.2]
    assert not is_legacy_schema(reloaded)


def test_build_index_payload_persists_findings_and_stats_when_provided():
    """Document-processing diagnostics UI (`document_converter.html`'s "View
    Details" panel): `validation.findings`/`validation.stats` must round-trip
    through the index payload so `/docs/{doc_id}` can serve them for an
    already-indexed document without re-running the pipeline."""
    findings = [{"code": "long_paragraph", "message": "too long", "severity": "WARNING",
                 "block_id": "b1", "line": None, "page": None, "repaired": False, "suggested_action": None}]
    stats = {"page_count": 3, "heading_count": 2, "chunk_count": 1, "warnings": 1, "errors": 0}
    payload = build_index_payload(
        document_id="d-3", title="T", source_path="x.md", markdown_path="x.md", report_path="x.report.md",
        source_hash="h", normalized_hash="n", extractor="markdown", ocr_used=False,
        semantic_structuring_used=False, config_fingerprint="fp1", embedding_info=_info(),
        chunks=[], embeddings=[], retrieval_ready=True, warnings=1, errors=0,
        findings=findings, stats=stats,
    )
    assert payload["validation"]["findings"] == findings
    assert payload["validation"]["stats"] == stats


def test_build_index_payload_defaults_findings_and_stats_when_omitted():
    """Callers that don't pass findings/stats (none currently do outside
    pipeline.index(), but the parameters are optional) get empty containers,
    not a missing key -- distinguishes "nothing recorded here" (this case)
    from "this index predates the field entirely" (a real legacy payload,
    which simply lacks the `findings`/`stats` keys under `validation`)."""
    payload = build_index_payload(
        document_id="d-4", title="T", source_path="x.md", markdown_path="x.md", report_path="x.report.md",
        source_hash="h", normalized_hash="n", extractor="markdown", ocr_used=False,
        semantic_structuring_used=False, config_fingerprint="fp1", embedding_info=_info(),
        chunks=[], embeddings=[], retrieval_ready=True, warnings=0, errors=0,
    )
    assert payload["validation"]["findings"] == []
    assert payload["validation"]["stats"] == {}


def test_atomic_write_leaves_no_tmp_file_behind(tmp_path):
    payload = build_index_payload(
        document_id="d-2", title="T", source_path="x.md", markdown_path="x.md", report_path="x.report.md",
        source_hash="h", normalized_hash="n", extractor="markdown", ocr_used=False,
        semantic_structuring_used=False, config_fingerprint="fp1", embedding_info=_info(),
        chunks=[], embeddings=[], retrieval_ready=True, warnings=0, errors=0,
    )
    write_index_atomic(str(tmp_path), "d-2", payload)
    files = os.listdir(tmp_path)
    assert files == ["d-2.json"]


def test_is_up_to_date_detects_source_and_config_changes(tmp_path):
    payload = build_index_payload(
        document_id="d-3", title="T", source_path="x.md", markdown_path="x.md", report_path="x.report.md",
        source_hash="hash-a", normalized_hash="n", extractor="markdown", ocr_used=False,
        semantic_structuring_used=False, config_fingerprint="fp-a", embedding_info=_info(),
        chunks=[], embeddings=[], retrieval_ready=True, warnings=0, errors=0,
    )
    assert is_up_to_date(payload, "hash-a", "fp-a") is True
    assert is_up_to_date(payload, "hash-b", "fp-a") is False
    assert is_up_to_date(payload, "hash-a", "fp-b") is False


def test_is_legacy_schema_detects_old_payloads():
    assert is_legacy_schema({"content_hash": "abc", "chunks": []}) is True
    assert is_legacy_schema({"schema_version": 2}) is False


# ---------------------------------------------------------------------- #
# Embedding metadata (embedding-model migration: model id, resolved
# revision, dimension, normalization, query prompt configuration,
# tokenizer identity, chunk target/max, config fingerprint).

def test_build_index_payload_records_full_embedding_metadata():
    payload = build_index_payload(
        document_id="d-1", title="T", source_path="x.md", markdown_path="x.md", report_path="x.report.md",
        source_hash="h", normalized_hash="n", extractor="pdf", ocr_used=False,
        semantic_structuring_used=False, config_fingerprint="fp1", embedding_info=_qwen_info(),
        chunks=[_chunk()], embeddings=[[0.1] * 2560], retrieval_ready=True, warnings=0, errors=0,
        chunk_target_tokens=1000, chunk_max_tokens=4096,
    )
    emb = payload["embedding"]
    assert emb["model"] == "Qwen/Qwen3-Embedding-4B"
    assert emb["revision"] == "5cf2132abc99cad020ac570b19d031efec650f2b"
    assert emb["dimension"] == 2560
    assert emb["normalized_embeddings"] is True
    assert emb["query_prompt_name"] == "query"
    assert emb["tokenizer_id"] == "Qwen/Qwen3-Embedding-4B"
    assert emb["chunk_target_tokens"] == 1000
    assert emb["chunk_max_tokens"] == 4096
    assert emb["config_fingerprint"]  # non-empty


def test_build_index_payload_embedding_metadata_defaults_backward_compatible():
    """Existing call sites that don't pass chunk_target_tokens/chunk_max_tokens
    (e.g. every pre-migration BGE call) must keep working unchanged."""
    payload = build_index_payload(
        document_id="d-1", title="T", source_path="x.md", markdown_path="x.md", report_path="x.report.md",
        source_hash="h", normalized_hash="n", extractor="pdf", ocr_used=False,
        semantic_structuring_used=False, config_fingerprint="fp1", embedding_info=_info(),
        chunks=[], embeddings=[], retrieval_ready=True, warnings=0, errors=0,
    )
    assert payload["embedding"]["chunk_target_tokens"] is None
    assert payload["embedding"]["chunk_max_tokens"] is None
    assert payload["embedding"]["config_fingerprint"]


# ---------------------------------------------------------------------- #
# embedding_configuration_fingerprint / check_embedding_compatibility --
# BGE vectors and Qwen vectors (or two different Qwen configurations) must
# never be silently mixed.

def test_embedding_fingerprint_differs_across_model_and_dimension():
    bge_fp = embedding_configuration_fingerprint(_info(), None, None)
    qwen_fp = embedding_configuration_fingerprint(_qwen_info(), 1000, 4096)
    assert bge_fp != qwen_fp


def test_embedding_fingerprint_stable_for_identical_config():
    a = embedding_configuration_fingerprint(_qwen_info(), 1000, 4096)
    b = embedding_configuration_fingerprint(_qwen_info(), 1000, 4096)
    assert a == b


def test_embedding_fingerprint_changes_with_chunk_token_settings():
    a = embedding_configuration_fingerprint(_qwen_info(), 1000, 4096)
    b = embedding_configuration_fingerprint(_qwen_info(), 2000, 4096)
    assert a != b


def _qwen_payload():
    info = _qwen_info()
    return build_index_payload(
        document_id="d-1", title="T", source_path="x.md", markdown_path="x.md", report_path="x.report.md",
        source_hash="h", normalized_hash="n", extractor="pdf", ocr_used=False,
        semantic_structuring_used=False, config_fingerprint="fp1", embedding_info=info,
        chunks=[], embeddings=[], retrieval_ready=True, warnings=0, errors=0,
        chunk_target_tokens=1000, chunk_max_tokens=4096,
    )


def test_check_embedding_compatibility_matching_config_returns_none():
    payload = _qwen_payload()
    assert check_embedding_compatibility(payload, _qwen_info(), 1000, 4096) is None


def test_check_embedding_compatibility_rejects_bge_index_under_qwen_config():
    """The exact scenario the spec calls out by name: never silently load
    a BGE-embedded index as if it were a Qwen index."""
    bge_payload = build_index_payload(
        document_id="d-1", title="T", source_path="x.md", markdown_path="x.md", report_path="x.report.md",
        source_hash="h", normalized_hash="n", extractor="pdf", ocr_used=False,
        semantic_structuring_used=False, config_fingerprint="fp1", embedding_info=_info(),
        chunks=[], embeddings=[], retrieval_ready=True, warnings=0, errors=0,
    )
    reason = check_embedding_compatibility(bge_payload, _qwen_info(), 1000, 4096)
    assert reason is not None
    assert "Qwen/Qwen3-Embedding-4B" in reason  # names the active (mismatched-against) model


def test_check_embedding_compatibility_rejects_dimension_mismatch():
    payload = _qwen_payload()
    mismatched = _qwen_info(dimension=1024)
    reason = check_embedding_compatibility(payload, mismatched, 1000, 4096)
    assert reason is not None
    assert "2560" in reason and "1024" in reason


def test_check_embedding_compatibility_rejects_chunk_token_setting_mismatch():
    payload = _qwen_payload()
    reason = check_embedding_compatibility(payload, _qwen_info(), 2000, 4096)
    assert reason is not None


def test_check_embedding_compatibility_accepts_legacy_payload_with_fully_matching_fields():
    """A payload missing only `config_fingerprint`, with every other
    embedding field present and matching the live service exactly, is
    accepted -- "predates the fingerprint field" is not itself grounds for
    rejection once the legacy-compatibility check exists (see the
    `_legacy_embedding_compatibility`-backed tests below for the real,
    sparser production-BGE shape and its failure-closed cases)."""
    legacy_style = _qwen_payload()
    legacy_style["embedding"].pop("config_fingerprint", None)
    reason = check_embedding_compatibility(legacy_style, _qwen_info(), 1000, 4096)
    assert reason is None


# ---------------------------------------------------------------------- #
# Legacy (pre-config_fingerprint) index backward compatibility -- the real
# production docs/.index entries indexed 2026-07-29/30 record only
# provider/model/dimension/max_sequence_tokens (schema_version 2, but
# written before the embedding-model-migration metadata existed). Treating
# "no config_fingerprint" as automatically incompatible silently dropped
# every one of those still-genuinely-BGE documents from
# orchestrator.py's DocStore. `_legacy_embedding_compatibility` accepts a
# legacy payload only when everything it DOES record affirmatively matches
# the live service; missing/ambiguous identity or any recorded
# contradiction still fails closed.

def _bge_info(**overrides):
    defaults = dict(
        provider="sentence-transformers-service", model="BAAI/bge-large-en-v1.5",
        dimension=1024, max_sequence_tokens=512, normalized_embeddings=True,
    )
    defaults.update(overrides)
    return EmbeddingModelInfo(**defaults)


def _legacy_bge_payload(vector_dim=1024, n_chunks=1):
    """Shaped exactly like the real pre-migration production index entries."""
    chunks = [ChunkRecord(
        chunk_id=f"c-{i}", chunk_index=i, header_path=["T"], heading="T", provenance="[x.pdf > T]",
        text="[x.pdf > T]\n\n# T\n\nhello", body="# T\n\nhello", word_count=3, token_count=4,
        source_block_ids=[f"b{i:04d}"], source_pages=[1],
    ) for i in range(n_chunks)]
    payload = build_index_payload(
        document_id="d-1", title="T", source_path="x.pdf", markdown_path="x.md", report_path="x.report.md",
        source_hash="h", normalized_hash="n", extractor="pdf", ocr_used=False,
        semantic_structuring_used=False, config_fingerprint="fp1", embedding_info=_bge_info(),
        chunks=chunks, embeddings=[[0.1] * vector_dim for _ in chunks], retrieval_ready=True, warnings=0, errors=0,
    )
    # Strip down to the real legacy shape -- only what pre-migration
    # ingestion actually wrote, none of the migration-era fields.
    payload["embedding"] = {
        "provider": "sentence-transformers-service", "model": "BAAI/bge-large-en-v1.5",
        "dimension": 1024, "max_sequence_tokens": 512,
    }
    return payload


def test_check_embedding_compatibility_accepts_compatible_legacy_bge_index():
    """1. compatible legacy BGE index without fingerprint -> accepted."""
    payload = _legacy_bge_payload()
    assert check_embedding_compatibility(payload, _bge_info(), None, None) is None


def test_check_embedding_compatibility_rejects_legacy_actual_vector_dimension_mismatch():
    """2. legacy BGE vector dimension mismatch -> rejected. Declared
    `dimension` still says 1024, but the vectors actually stored on the
    chunks don't -- vectors are ground truth, not metadata."""
    payload = _legacy_bge_payload(vector_dim=999)
    reason = check_embedding_compatibility(payload, _bge_info(), None, None)
    assert reason is not None
    assert "999" in reason


def test_check_embedding_compatibility_rejects_legacy_index_with_no_model_identity():
    """3. legacy index with unknown/insufficient model identity -> rejected
    (fail closed, not accepted by default)."""
    payload = _legacy_bge_payload()
    payload["embedding"] = {"provider": "sentence-transformers-service"}  # no model, no dimension recorded
    reason = check_embedding_compatibility(payload, _bge_info(), None, None)
    assert reason is not None
    assert "identity" in reason


def test_check_embedding_compatibility_rejects_qwen_index_under_bge_service():
    """4. Qwen index under BGE -> rejected."""
    qwen_payload = _qwen_payload()
    reason = check_embedding_compatibility(qwen_payload, _bge_info(), None, None)
    assert reason is not None
    assert "BAAI/bge-large-en-v1.5" in reason


def test_check_embedding_compatibility_rejects_legacy_bge_index_under_qwen_service():
    """5. BGE index under Qwen -> rejected (legacy-shaped this time, not
    just the already-covered fingerprinted case)."""
    payload = _legacy_bge_payload()
    reason = check_embedding_compatibility(payload, _qwen_info(), 1000, 4096)
    assert reason is not None
    assert "Qwen/Qwen3-Embedding-4B" in reason


def test_check_embedding_compatibility_fingerprinted_index_stays_strict():
    """6. current (fingerprinted) indexes retain strict behavior -- matching
    model/dimension alone is not enough to accept a fingerprinted payload
    whose fingerprint doesn't match; only a genuinely legacy (no-fingerprint)
    payload gets the permissive field-by-field check."""
    payload = _qwen_payload()  # has config_fingerprint, built for chunk_target_tokens=1000
    reason = check_embedding_compatibility(payload, _qwen_info(), 2000, 4096)  # same model/dim, different chunk tokens
    assert reason is not None

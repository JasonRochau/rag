"""
Unit tests for embedding_client.py's input_type plumbing and extended
EmbeddingModelInfo metadata (embedding-model migration). Monkeypatches
`requests.get`/`requests.post` -- no real HTTP, no real embedding service.
"""
from __future__ import annotations

import pytest

from document_converter.embedding_client import (
    EmbeddingClient, EmbeddingModelInfo, EmbeddingCompatibilityError, EmbeddingOversizedInputError,
    EmbeddingServiceError,
)


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = str(json_data)

    def json(self):
        return self._json

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


QWEN_HEALTH = {
    "ok": True, "provider": "sentence-transformers-service", "model": "Qwen/Qwen3-Embedding-4B",
    "revision": "5cf2132abc99cad020ac570b19d031efec650f2b", "dimension": 2560,
    "max_sequence_tokens": 40960, "normalized_embeddings": True,
    "query_prompt_name": "query", "tokenizer_id": "Qwen/Qwen3-Embedding-4B",
}


def _client(monkeypatch, health_payload, model="Qwen/Qwen3-Embedding-4B", captured=None):
    def fake_get(url, timeout=None):
        return _FakeResponse(health_payload)

    def fake_post(url, json=None, timeout=None):
        if captured is not None:
            captured.append(json)
        if url.endswith("/api/embed"):
            n = len(json["input"])
            dim = health_payload.get("dimension", 2)
            return _FakeResponse({
                "embeddings": [[0.1] * dim for _ in range(n)],
                "model": health_payload.get("model"),
                "dimension": dim,
                "token_counts": [3] * n,
                "normalized_embeddings": True,
                "input_type": json.get("input_type", "document"),
            })
        raise AssertionError(f"unexpected POST to {url}")

    monkeypatch.setattr("document_converter.embedding_client.requests.get", fake_get)
    monkeypatch.setattr("document_converter.embedding_client.requests.post", fake_post)
    return EmbeddingClient(host="http://fake:1", model=model)


# ---------------------------------------------------------------------- #
# health() parses the new metadata fields

def test_health_parses_revision_query_prompt_and_tokenizer_id(monkeypatch):
    client = _client(monkeypatch, QWEN_HEALTH)
    info = client.health()
    assert isinstance(info, EmbeddingModelInfo)
    assert info.revision == "5cf2132abc99cad020ac570b19d031efec650f2b"
    assert info.query_prompt_name == "query"
    assert info.tokenizer_id == "Qwen/Qwen3-Embedding-4B"
    assert info.dimension == 2560


def test_health_defaults_new_fields_to_none_when_absent(monkeypatch):
    bge_health = {
        "provider": "sentence-transformers-service", "model": "BAAI/bge-large-en-v1.5",
        "dimension": 1024, "max_sequence_tokens": 512, "normalized_embeddings": True,
    }
    client = _client(monkeypatch, bge_health, model="BAAI/bge-large-en-v1.5")
    info = client.health()
    assert info.revision is None
    assert info.query_prompt_name is None
    assert info.tokenizer_id is None


# ---------------------------------------------------------------------- #
# 2560-dimensional vectors

def test_embed_batch_returns_2560_dimensional_vectors(monkeypatch):
    client = _client(monkeypatch, QWEN_HEALTH)
    vectors = client.embed_batch(["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 2560


# ---------------------------------------------------------------------- #
# input_type plumbing: document is the default, query must be explicit

def test_embed_batch_defaults_to_document_input_type(monkeypatch):
    captured = []
    client = _client(monkeypatch, QWEN_HEALTH, captured=captured)
    client.embed_batch(["a chunk of document text"])
    assert captured[0]["input_type"] == "document"


def test_embed_batch_document_input_type_is_backward_compatible_default(monkeypatch):
    """Every pre-migration ingestion call site (`client.embed_batch(texts)`,
    no `input_type` kwarg at all) must keep behaving exactly as before."""
    captured = []
    client = _client(monkeypatch, QWEN_HEALTH, captured=captured)
    client.embed_batch(["texts"], reject_oversized=True)
    assert captured[0]["input_type"] == "document"


def test_embed_one_query_sends_query_input_type(monkeypatch):
    captured = []
    client = _client(monkeypatch, QWEN_HEALTH, captured=captured)
    client.embed_one("what is a bayesian network?", input_type="query")
    assert captured[0]["input_type"] == "query"


def test_embed_one_default_is_document(monkeypatch):
    captured = []
    client = _client(monkeypatch, QWEN_HEALTH, captured=captured)
    client.embed_one("some text")
    assert captured[0]["input_type"] == "document"


def test_invalid_input_type_rejected_client_side(monkeypatch):
    client = _client(monkeypatch, QWEN_HEALTH)
    with pytest.raises(ValueError):
        client.embed_batch(["x"], input_type="chunk")


# ---------------------------------------------------------------------- #
# Oversized input still raises (unaffected by input_type)

def test_oversized_input_still_raises_regardless_of_input_type(monkeypatch):
    def fake_get(url, timeout=None):
        return _FakeResponse(QWEN_HEALTH)

    def fake_post(url, json=None, timeout=None):
        return _FakeResponse({
            "error": "input_too_long", "message": "too long",
            "oversized_indices": [0], "token_counts": [50000],
            "max_sequence_tokens": 40960,
        }, status_code=413)

    monkeypatch.setattr("document_converter.embedding_client.requests.get", fake_get)
    monkeypatch.setattr("document_converter.embedding_client.requests.post", fake_post)
    client = EmbeddingClient(host="http://fake:1", model="Qwen/Qwen3-Embedding-4B")
    with pytest.raises(EmbeddingOversizedInputError):
        client.embed_batch(["x" * 1000], input_type="query")


# ---------------------------------------------------------------------- #
# Model-mismatch guard (pre-existing behavior) still works with the new fields

def test_health_still_rejects_unexpected_model(monkeypatch):
    client = _client(monkeypatch, QWEN_HEALTH, model="BAAI/bge-large-en-v1.5")
    with pytest.raises(EmbeddingCompatibilityError):
        client.health()


# ---------------------------------------------------------------------- #
# release() -- explicit VRAM-release barrier (strict sequential GPU model
# lifecycle phase)

def test_release_posts_to_api_release_and_returns_true_on_success(monkeypatch):
    captured = []

    def fake_post(url, timeout=None):
        captured.append(url)
        return _FakeResponse({"released": True, "device": "cpu", "cuda_available": True})

    monkeypatch.setattr("document_converter.embedding_client.requests.post", fake_post)
    client = EmbeddingClient(host="http://fake:1", model="BAAI/bge-large-en-v1.5")
    assert client.release() is True
    assert captured == ["http://fake:1/api/release"]


def test_release_returns_false_when_service_reports_not_released(monkeypatch):
    def fake_post(url, timeout=None):
        return _FakeResponse({"released": False, "device": "cuda", "cuda_available": True})

    monkeypatch.setattr("document_converter.embedding_client.requests.post", fake_post)
    client = EmbeddingClient(host="http://fake:1", model="BAAI/bge-large-en-v1.5")
    assert client.release() is False


def test_release_is_best_effort_and_never_raises_when_unreachable(monkeypatch):
    def fake_post(url, timeout=None):
        raise ConnectionError("no route to host")

    monkeypatch.setattr("document_converter.embedding_client.requests.post", fake_post)
    client = EmbeddingClient(host="http://fake:1", model="BAAI/bge-large-en-v1.5")
    assert client.release() is False


# ---------------------------------------------------------------------- #
# Retry policy for transient server-side encode failures (CUDA OOM
# resilience): a 500 carrying the embedding server's `embedding_failed`
# error shape is backed off and retried; everything else fails immediately.

EMBED_OK = {
    "embeddings": [[0.1, 0.2]],
    "model": "BAAI/bge-large-en-v1.5",
    "dimension": 2,
    "token_counts": [3],
    "normalized_embeddings": True,
    "input_type": "document",
}


def _retry_client(monkeypatch, responses, retries=3, base_delay=5.0):
    """Client whose POSTs return the scripted `responses` in order, with
    sleep captured instead of performed. Returns (client, calls, delays)."""
    import document_converter.embedding_client as ec

    calls = []
    delays = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        return responses.pop(0)

    monkeypatch.setattr(ec.requests, "post", fake_post)
    monkeypatch.setattr(ec.time, "sleep", delays.append)
    monkeypatch.setattr(ec, "EMBEDDING_CLIENT_RETRIES", retries)
    monkeypatch.setattr(ec, "EMBEDDING_CLIENT_RETRY_BASE_DELAY", base_delay)
    return EmbeddingClient(host="http://fake:1", model="BAAI/bge-large-en-v1.5"), calls, delays


def _embedding_failed_500():
    return _FakeResponse({"error": "embedding_failed", "message": "CUDA out of memory"}, status_code=500)


def test_embedding_failed_500_is_retried_and_recovers(monkeypatch):
    client, calls, delays = _retry_client(monkeypatch, [_embedding_failed_500(), _FakeResponse(EMBED_OK)])
    vectors = client.embed_batch(["hello world"])
    assert vectors == [[0.1, 0.2]]
    assert len(calls) == 2
    assert delays == [5.0]


def test_backoff_delays_scale_by_3x_per_attempt(monkeypatch):
    responses = [_embedding_failed_500(), _embedding_failed_500(), _FakeResponse(EMBED_OK)]
    client, calls, delays = _retry_client(monkeypatch, responses)
    assert client.embed_batch(["hello world"]) == [[0.1, 0.2]]
    assert len(calls) == 3
    assert delays == [5.0, 15.0]


def test_embedding_failed_500_raises_after_retries_exhausted(monkeypatch):
    responses = [_embedding_failed_500()] * 3
    client, calls, delays = _retry_client(monkeypatch, responses)
    with pytest.raises(EmbeddingServiceError, match="HTTP 500"):
        client.embed_batch(["hello world"])
    assert len(calls) == 3
    assert delays == [5.0, 15.0]


def test_500_with_other_error_shape_is_not_retried(monkeypatch):
    responses = [_FakeResponse({"error": "some_other_bug"}, status_code=500)]
    client, calls, delays = _retry_client(monkeypatch, responses)
    with pytest.raises(EmbeddingServiceError, match="HTTP 500"):
        client.embed_batch(["hello world"])
    assert len(calls) == 1
    assert delays == []


def test_500_with_non_json_body_is_not_retried(monkeypatch):
    class ProxyErrorResponse(_FakeResponse):
        def json(self):
            raise ValueError("no json here")

    client, calls, delays = _retry_client(monkeypatch, [ProxyErrorResponse({}, status_code=500)])
    with pytest.raises(EmbeddingServiceError, match="HTTP 500"):
        client.embed_batch(["hello world"])
    assert len(calls) == 1
    assert delays == []


def test_non_500_5xx_is_not_retried(monkeypatch):
    client, calls, delays = _retry_client(monkeypatch, [_FakeResponse({}, status_code=502)])
    with pytest.raises(EmbeddingServiceError, match="HTTP 502"):
        client.embed_batch(["hello world"])
    assert len(calls) == 1
    assert delays == []


def test_413_oversized_input_is_not_retried(monkeypatch):
    oversized = _FakeResponse({
        "error": "input_too_long", "message": "too long",
        "oversized_indices": [0], "token_counts": [9999], "max_sequence_tokens": 512,
    }, status_code=413)
    client, calls, delays = _retry_client(monkeypatch, [oversized])
    with pytest.raises(EmbeddingOversizedInputError):
        client.embed_batch(["hello world"])
    assert len(calls) == 1
    assert delays == []

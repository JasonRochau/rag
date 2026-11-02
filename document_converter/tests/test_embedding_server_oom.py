"""Tests for embedding_server.py's CUDA OOM resilience: `_is_cuda_oom` and
`encode_with_oom_retry` (the halve-to-batch_size-1 retry ladder).

embedding_server.py loads a real multi-GB SentenceTransformer at import time,
so `sentence_transformers` is replaced with a no-download fake before import.
torch and flask are the real installed packages; CUDA calls the retry path
makes (`is_available`, `empty_cache`) are monkeypatched per test so the suite
behaves identically with or without a GPU present.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeModel:
    """Import-time stand-in for SentenceTransformer -- nothing is downloaded
    or loaded. Only the attributes embedding_server touches at module level
    are provided."""

    max_seq_length = 512
    prompts = None

    def __init__(self, *args, **kwargs):
        self.tokenizer = types.SimpleNamespace(
            name_or_path="fake-tokenizer",
            encode=lambda text, add_special_tokens=True: text.split(),
        )
        self._module = types.SimpleNamespace(
            auto_model=types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash="test-rev"))
        )

    def __getitem__(self, index):
        assert index == 0
        return self._module

    def get_sentence_embedding_dimension(self):
        return 8

    def parameters(self):
        import torch
        return iter([types.SimpleNamespace(device=torch.device("cpu"))])


def _import_embedding_server():
    if "embedding_server" in sys.modules:
        return sys.modules["embedding_server"]
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = _FakeModel
    previous = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = stub
    try:
        return importlib.import_module("embedding_server")
    finally:
        if previous is not None:
            sys.modules["sentence_transformers"] = previous
        else:
            sys.modules.pop("sentence_transformers", None)


embedding_server = _import_embedding_server()


class _ScriptedEncoder:
    """Fake model whose encode() raises the scripted exceptions in order
    (None = succeed), recording the batch_size of every attempt."""

    def __init__(self, script):
        self._script = list(script)
        self.batch_sizes = []

    def encode(self, inputs, **kwargs):
        self.batch_sizes.append(kwargs.get("batch_size"))
        outcome = self._script.pop(0)
        if outcome is not None:
            raise outcome
        return [[0.0] * 8 for _ in inputs]


@pytest.fixture
def cuda_spy(monkeypatch):
    """Deterministic CUDA facade: GPU present, empty_cache recorded."""
    calls = []
    monkeypatch.setattr(embedding_server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(embedding_server.torch.cuda, "empty_cache", lambda: calls.append(True))
    return calls


def _use_encoder(monkeypatch, script):
    encoder = _ScriptedEncoder(script)
    monkeypatch.setattr(embedding_server, "model", encoder)
    return encoder


# ---------------------------------------------------------------------- #
# _is_cuda_oom

def test_is_cuda_oom_matches_canonical_message_on_plain_runtime_error():
    assert embedding_server._is_cuda_oom(
        RuntimeError("CUDA out of memory. Tried to allocate 234.00 MiB")
    ) is True


def test_is_cuda_oom_matches_typed_out_of_memory_error():
    oom_type = getattr(embedding_server.torch.cuda, "OutOfMemoryError", None)
    if oom_type is None:
        pytest.skip("installed torch has no torch.cuda.OutOfMemoryError")
    assert embedding_server._is_cuda_oom(oom_type("CUDA out of memory")) is True


def test_is_cuda_oom_rejects_unrelated_runtime_error():
    assert embedding_server._is_cuda_oom(RuntimeError("shape mismatch")) is False


def test_is_cuda_oom_rejects_non_runtime_error_even_with_oom_message():
    assert embedding_server._is_cuda_oom(ValueError("CUDA out of memory")) is False


# ---------------------------------------------------------------------- #
# encode_with_oom_retry

def test_successful_encode_never_retries(monkeypatch, cuda_spy):
    encoder = _use_encoder(monkeypatch, [None])
    vectors = embedding_server.encode_with_oom_retry(["a", "b"], {"batch_size": 8})
    assert len(vectors) == 2
    assert encoder.batch_sizes == [8]
    assert cuda_spy == []


def test_oom_retries_with_halved_batch_size_until_success(monkeypatch, cuda_spy):
    oom = RuntimeError("CUDA out of memory. Tried to allocate 234.00 MiB")
    encoder = _use_encoder(monkeypatch, [oom, oom, None])
    vectors = embedding_server.encode_with_oom_retry(["a"], {"batch_size": 8})
    assert len(vectors) == 1
    assert encoder.batch_sizes == [8, 4, 2]
    assert cuda_spy == [True, True]


def test_oom_gives_up_at_batch_size_1_and_propagates(monkeypatch, cuda_spy):
    oom = RuntimeError("CUDA out of memory. Tried to allocate 234.00 MiB")
    encoder = _use_encoder(monkeypatch, [oom, oom, oom])
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        embedding_server.encode_with_oom_retry(["a"], {"batch_size": 4})
    assert encoder.batch_sizes == [4, 2, 1]
    assert cuda_spy == [True, True]


def test_oom_at_batch_size_1_raises_immediately_without_retry(monkeypatch, cuda_spy):
    encoder = _use_encoder(monkeypatch, [RuntimeError("CUDA out of memory")])
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        embedding_server.encode_with_oom_retry(["a"], {"batch_size": 1})
    assert encoder.batch_sizes == [1]
    assert cuda_spy == []


def test_unrelated_error_propagates_without_retry(monkeypatch, cuda_spy):
    encoder = _use_encoder(monkeypatch, [RuntimeError("shape mismatch")])
    with pytest.raises(RuntimeError, match="shape mismatch"):
        embedding_server.encode_with_oom_retry(["a"], {"batch_size": 8})
    assert encoder.batch_sizes == [8]
    assert cuda_spy == []


def test_default_batch_size_used_when_kwargs_lack_one(monkeypatch, cuda_spy):
    monkeypatch.setattr(embedding_server, "BATCH_SIZE", 4)
    oom = RuntimeError("CUDA out of memory")
    encoder = _use_encoder(monkeypatch, [oom, None])
    embedding_server.encode_with_oom_retry(["a"], {})
    assert encoder.batch_sizes == [None, 2]

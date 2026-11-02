"""
Unit tests for embedding_backend.py's pure model-selection/encoding
decisions (embedding-model migration: BAAI/bge-large-en-v1.5 ->
Qwen/Qwen3-Embedding-4B). No Flask, no real model load -- see the module
docstring for why this is split out of embedding_server.py.
"""
from __future__ import annotations

import pytest

from embedding_backend import (
    is_qwen3_embedding, qwen3_embedding_load_kwargs, load_kwargs_for,
    resolve_query_prompt_name, build_encode_kwargs, validate_input_type, default_encode_batch_size,
    cuda_lifecycle_managed,
)


# ---------------------------------------------------------------------- #
# Qwen model selection

def test_qwen3_embedding_4b_is_selected():
    assert is_qwen3_embedding("Qwen/Qwen3-Embedding-4B") is True


def test_qwen3_embedding_family_other_sizes_also_selected():
    assert is_qwen3_embedding("Qwen/Qwen3-Embedding-0.6B") is True
    assert is_qwen3_embedding("Qwen/Qwen3-Embedding-8B") is True


def test_bge_is_not_selected_as_qwen():
    assert is_qwen3_embedding("BAAI/bge-large-en-v1.5") is False


def test_unrelated_qwen_chat_model_is_not_selected():
    # A future Qwen *chat* model id must not accidentally pick up
    # embedding-specific load kwargs (attn_implementation, padding_side, ...).
    assert is_qwen3_embedding("Qwen/Qwen2.5-7B-Instruct") is False


def test_load_kwargs_for_bge_is_unchanged_plain_load():
    # Today's exact SentenceTransformer(model_id) call -- no extra kwargs.
    assert load_kwargs_for("BAAI/bge-large-en-v1.5") == {}


def test_load_kwargs_for_qwen_matches_verified_smoke_test_config():
    kwargs = load_kwargs_for("Qwen/Qwen3-Embedding-4B")
    assert kwargs == qwen3_embedding_load_kwargs()
    assert kwargs["device"] == "cuda"


# ---------------------------------------------------------------------- #
# processor_kwargs instead of tokenizer_kwargs (tokenizer_kwargs is
# deprecated in the installed sentence-transformers version)

def test_qwen_load_kwargs_use_processor_kwargs_not_tokenizer_kwargs():
    kwargs = qwen3_embedding_load_kwargs()
    assert "processor_kwargs" in kwargs
    assert kwargs["processor_kwargs"] == {"padding_side": "left"}
    assert "tokenizer_kwargs" not in kwargs


def test_qwen_load_kwargs_model_kwargs_shape():
    kwargs = qwen3_embedding_load_kwargs()
    model_kwargs = kwargs["model_kwargs"]
    assert model_kwargs["attn_implementation"] == "eager"
    # torch_dtype must be the actual torch.float16 dtype object, not a string.
    import torch
    assert model_kwargs["torch_dtype"] is torch.float16


def test_qwen_load_kwargs_do_not_request_accelerate_or_quantization_features():
    kwargs = qwen3_embedding_load_kwargs()
    assert "device_map" not in kwargs
    assert "device_map" not in kwargs.get("model_kwargs", {})
    assert "quantization_config" not in kwargs.get("model_kwargs", {})


# ---------------------------------------------------------------------- #
# Query prompt resolution

def test_resolve_query_prompt_name_present_for_qwen():
    prompts = {"query": "Instruct: ...\nQuery:", "document": ""}
    assert resolve_query_prompt_name("Qwen/Qwen3-Embedding-4B", prompts) == "query"


def test_resolve_query_prompt_name_absent_when_qwen_defines_no_prompts():
    assert resolve_query_prompt_name("Qwen/Qwen3-Embedding-4B", {}) is None
    assert resolve_query_prompt_name("Qwen/Qwen3-Embedding-4B", None) is None


def test_resolve_query_prompt_name_never_applied_to_bge_even_though_bge_defines_one():
    """BGE's own SentenceTransformer config *also* defines a "query" prompt
    key (`{"query": "", "document": ""}`, both empty strings -- numerically
    a no-op, verified separately) -- this migration is targeted at Qwen, so
    BGE's query encoding must never depend on that happening to be empty.
    It must never receive a prompt at all, provably, regardless of what
    BGE's own config contains."""
    bge_shaped_prompts = {"query": "", "document": ""}
    assert resolve_query_prompt_name("BAAI/bge-large-en-v1.5", bge_shaped_prompts) is None


def test_resolve_query_prompt_name_absent_for_unrelated_models():
    assert resolve_query_prompt_name("sentence-transformers/all-MiniLM-L6-v2", {"query": "x"}) is None


# ---------------------------------------------------------------------- #
# Document encoding without the query prompt / query encoding with
# prompt_name="query" -- this one function is the entire ingestion/search
# encoding distinction (spec: "Do not apply the query prompt to documents").

def test_document_encoding_never_receives_a_prompt_name():
    kwargs = build_encode_kwargs("document", query_prompt_name="query")
    assert "prompt_name" not in kwargs
    assert kwargs["normalize_embeddings"] is True


def test_query_encoding_receives_prompt_name_query():
    kwargs = build_encode_kwargs("query", query_prompt_name="query")
    assert kwargs["prompt_name"] == "query"
    assert kwargs["normalize_embeddings"] is True


def test_query_encoding_omits_prompt_name_when_model_has_none_bge_case():
    # BGE (or any model with no query prompt configured): document and
    # query encoding stay identical -- no error, no fabricated prompt.
    kwargs = build_encode_kwargs("query", query_prompt_name=None)
    assert "prompt_name" not in kwargs


def test_encode_kwargs_always_normalize_embeddings():
    for input_type in ("document", "query"):
        kwargs = build_encode_kwargs(input_type, query_prompt_name="query", normalize_embeddings=True)
        assert kwargs["normalize_embeddings"] is True


def test_invalid_input_type_rejected():
    with pytest.raises(ValueError):
        build_encode_kwargs("chunk", query_prompt_name="query")
    with pytest.raises(ValueError):
        validate_input_type("bogus")


def test_valid_input_types_accepted():
    validate_input_type("document")
    validate_input_type("query")


# ---------------------------------------------------------------------- #
# Sub-batching to bound peak per-forward-pass GPU memory (real CUDA OOM
# observed on a 12GB card indexing several real PDFs with up to
# 4096-token chunks -- see embedding_backend.py's module comment).

def test_qwen_gets_a_small_default_batch_size():
    assert default_encode_batch_size("Qwen/Qwen3-Embedding-4B") == 2


def test_bge_keeps_sentence_transformers_own_default_batch_size():
    assert default_encode_batch_size("BAAI/bge-large-en-v1.5") == 32


def test_build_encode_kwargs_includes_batch_size_when_given():
    kwargs = build_encode_kwargs("document", query_prompt_name=None, batch_size=2)
    assert kwargs["batch_size"] == 2


def test_build_encode_kwargs_omits_batch_size_when_not_given():
    kwargs = build_encode_kwargs("document", query_prompt_name=None)
    assert "batch_size" not in kwargs


# ---------------------------------------------------------------------- #
# CUDA lifecycle management (GPU production-readiness test: Qwen3-
# Embedding-4B idle-resident on a 12GB card starves a concurrently loaded
# Ollama generator of VRAM -- measured directly as `ollama ps` reporting
# "77%/23% CPU/GPU" and ~6.6 tok/s. embedding_server.py moves a
# CUDA-lifecycle-managed model to the GPU only for the duration of each
# encode() call; see its module comment.)

def test_qwen_is_cuda_lifecycle_managed():
    assert cuda_lifecycle_managed("Qwen/Qwen3-Embedding-4B") is True


def test_bge_is_not_cuda_lifecycle_managed():
    # BGE (~1.3GB) has no VRAM-contention problem and keeps its original
    # always-resident-on-GPU behavior unchanged.
    assert cuda_lifecycle_managed("BAAI/bge-large-en-v1.5") is False


def test_unrelated_qwen_chat_model_is_not_cuda_lifecycle_managed():
    assert cuda_lifecycle_managed("Qwen/Qwen2.5-7B-Instruct") is False

"""
Embedding-backend selection/encoding decisions (embedding-model migration).

Pure logic, no Flask, no `sentence_transformers` model loading -- kept
separate from embedding_server.py specifically so it's importable and unit-
testable without loading a real (multi-GB, GPU-resident) model, the same
"pure scoring/ranking logic, stateless" split control.py already uses for
BM25/cosine (see its own module docstring). embedding_server.py imports and
calls these; it never re-implements the decisions itself.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def is_qwen3_embedding(model_id: str) -> bool:
    """True for a Qwen3-Embedding-family model id (e.g.
    "Qwen/Qwen3-Embedding-4B") -- the load/encode configuration this module
    provides is specific to that family, not a general "any Qwen model"
    rule (a future Qwen *chat* model id must not accidentally pick this up)."""
    return model_id.startswith("Qwen/Qwen3-Embedding")


def qwen3_embedding_load_kwargs() -> Dict[str, Any]:
    """Exact `SentenceTransformer(...)` loading configuration verified via
    local smoke test on the target GPU (2560-dim vectors, correct
    document ranked first, PASS): device="cuda", fp16 + eager attention (no
    FlashAttention, no quantization, no device_map="auto", no accelerate),
    left-padding via `processor_kwargs` -- `tokenizer_kwargs` is deprecated
    in the installed sentence-transformers version and deliberately not
    used. `torch` is imported lazily inside this function (not at module
    level) so importing this module for its pure string/dict logic never
    requires torch to be importable in a minimal test environment."""
    import torch
    return {
        "device": "cuda",
        "model_kwargs": {
            "torch_dtype": torch.float16,
            "attn_implementation": "eager",
        },
        "processor_kwargs": {
            "padding_side": "left",
        },
    }


def load_kwargs_for(model_id: str) -> Dict[str, Any]:
    """Dispatches by model id: Qwen3-Embedding gets the kwargs above; every
    other model (BGE, ...) gets `{}` -- today's exact, unmodified
    `SentenceTransformer(model_id)` call, so this is a strictly additive,
    targeted change, not a redesign of how every model loads."""
    if is_qwen3_embedding(model_id):
        return qwen3_embedding_load_kwargs()
    return {}


def cuda_lifecycle_managed(model_id: str) -> bool:
    """True when the embedding server must not leave `model_id`'s weights
    CUDA-resident between requests -- i.e. move the model to the GPU only
    for the duration of an actual `model.encode(...)` call, and back to CPU
    (with the CUDA cache released) immediately after.

    Qwen3-Embedding-4B (fp16, ~8GB weights) is large enough that leaving it
    GPU-resident at idle starves a concurrently loaded Ollama generator of
    VRAM: a real end-to-end GPU lifecycle test (embedding_server.py idle-
    resident + Ollama qwen3:14b) measured `ollama ps` reporting
    "77%/23% CPU/GPU" and ~6.6 tokens/sec for the 14B generator -- heavy CPU
    offload, not a clean OOM, but well past "unacceptably slow". BGE
    (~1.3GB) has no such contention and keeps its original always-resident
    behavior -- this is scoped to Qwen3-Embedding specifically, not a
    general embedding-server redesign."""
    return is_qwen3_embedding(model_id)


# `sentence_transformers.SentenceTransformer.encode`'s own `batch_size`
# parameter (default 32) -- a plain, first-party sub-batching knob, not
# accelerate/quantization/device_map/CPU-offloading. With
# EMBEDDING_CHUNK_MAX_TOKENS=4096, a single MAIN_TEXT document can produce
# several chunks each several thousand tokens long; encoding a whole
# document's chunks as one internal batch of up to 32 sequences that long
# needs proportionally more peak activation memory than BGE's much shorter
# (<=512-token) chunks ever did, and can exceed a 12GB card's headroom
# after the 4B-parameter model itself is loaded (~8-9GB in fp16) even
# though the *model* fits comfortably -- observed directly as a CUDA OOM
# on a real shadow-index run against several real PDFs, not a hypothetical
# concern. A small batch size bounds peak memory per forward pass without
# changing a single embedding's value; BGE (short chunks, small model)
# keeps sentence-transformers' own default.
_DEFAULT_QWEN3_EMBEDDING_BATCH_SIZE = 2


def default_encode_batch_size(model_id: str) -> int:
    if is_qwen3_embedding(model_id):
        return _DEFAULT_QWEN3_EMBEDDING_BATCH_SIZE
    return 32  # sentence-transformers' own default, explicit rather than omitted


def resolve_query_prompt_name(model_id: str, prompts: Optional[Dict[str, str]]) -> Optional[str]:
    """The prompt name to apply for `input_type="query"` -- "query" only
    for a Qwen3-Embedding model whose config actually defines one (it ships
    `prompts={"query": "Instruct: ...", "document": ""}`), `None` for
    everything else: document and query encoding then stay identical,
    exactly the pre-migration behavior.

    Deliberately gated on `is_qwen3_embedding(model_id)`, not merely on
    whether `prompts` happens to contain a "query" key: BGE's own config
    *also* defines one (`{"query": "", "document": ""}, both empty
    strings) -- numerically a no-op today (verified: encoding with vs.
    without `prompt_name="query"` against BGE produces bit-identical
    output, empty prompt text), but this migration is targeted at Qwen
    specifically (spec: "do not redesign..."), so BGE's query encoding
    must never depend on that coincidence continuing to hold -- it must
    provably never receive a prompt at all, regardless of what any given
    model's own config happens to contain."""
    if not is_qwen3_embedding(model_id):
        return None
    return "query" if prompts and "query" in prompts else None


_VALID_INPUT_TYPES = ("document", "query")


def validate_input_type(input_type: str) -> None:
    if input_type not in _VALID_INPUT_TYPES:
        raise ValueError(f"input_type must be one of {_VALID_INPUT_TYPES}, got {input_type!r}")


def build_encode_kwargs(
    input_type: str, query_prompt_name: Optional[str], normalize_embeddings: bool = True,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """The exact kwargs to pass to `model.encode(texts, **kwargs)` for a
    given `input_type` -- the entire ingestion/search encoding distinction
    lives in this one function, so it can never silently drift between the
    two call sites. "document" (or a model with no query prompt at all)
    never sets `prompt_name`, matching `model.encode(documents,
    normalize_embeddings=True)` exactly; "query" against a model that
    defines one adds `prompt_name=query_prompt_name`, matching
    `model.encode(queries, prompt_name="query", normalize_embeddings=True)`
    exactly. The query prompt is never applied to documents -- there is no
    code path here that could do that by accident. `batch_size`, when given
    (`default_encode_batch_size`), bounds peak per-forward-pass memory --
    it changes nothing about any single embedding's value, only how many
    texts are forwarded through the model at once."""
    validate_input_type(input_type)
    kwargs: Dict[str, Any] = {"normalize_embeddings": normalize_embeddings}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if input_type == "query" and query_prompt_name:
        kwargs["prompt_name"] = query_prompt_name
    return kwargs

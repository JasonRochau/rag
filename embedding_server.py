"""
LOCAL EMBEDDING SERVICE

Single configured sentence-transformers model, served locally over Flask.
Exposes model metadata (dimension, max sequence length, resolved revision,
query-prompt configuration, tokenizer identity) so ingestion and search
never have to guess or hard-code it, and rejects oversized input instead of
silently truncating it (spec section 13).

Owns the model for the lifetime of this process (spec "server lifecycle"):
starting this script loads the model onto the GPU/CPU; stopping it (SIGTERM/
SIGINT/process kill) releases that process's RAM and VRAM. No per-request
load/unload, no model swapping while running -- one model per running
instance, selected at startup via EMBEDDING_MODEL.

Model-specific load/encode *decisions* (which kwargs to load with, whether
to apply a query prompt) live in embedding_backend.py, not here -- pure
logic with no Flask/model dependency, importable and unit-testable without
loading a real (multi-GB, GPU-resident) model. This file is just the thin
Flask/SentenceTransformer glue around those decisions.
"""
import logging
import os
import threading

import torch
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

from embedding_backend import (
    load_kwargs_for, resolve_query_prompt_name, build_encode_kwargs, default_encode_batch_size,
    cuda_lifecycle_managed,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s | embedding_server | %(message)s')
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
PORT = int(os.environ.get("EMBEDDING_PORT", "11437"))
NORMALIZE = True
# Sub-batching within a single /api/embed call to bound peak per-forward-
# pass GPU memory (see embedding_backend.default_encode_batch_size) --
# overridable for tuning without a code change; defaults to a small batch
# for Qwen3-Embedding, sentence-transformers' own default (32) otherwise.
# Read EMBEDDING_SERVER_BATCH_SIZE first: this variable means "encode
# sub-batch size inside one /api/embed call" in THIS process, while the
# client (a different process, document_converter/config.py) reads
# EMBEDDING_BATCH_SIZE as "number of chunks per HTTP request" (default 64)
# -- same name, different process, different meaning, so a value tuned for
# one must never silently re-tune the other. EMBEDDING_BATCH_SIZE is still
# honored here as a fallback for backward compatibility with existing
# deployments that set it intending the server-side meaning.
BATCH_SIZE = int(os.environ.get(
    "EMBEDDING_SERVER_BATCH_SIZE",
    os.environ.get("EMBEDDING_BATCH_SIZE", default_encode_batch_size(MODEL_ID)),
))


def _resolve_revision(model_id: str, model: SentenceTransformer):
    """The exact commit/snapshot actually loaded -- not just the (possibly
    floating, e.g. "main") configured model id -- so index metadata can
    record what was really embedded, not what was asked for. Tries the
    loaded model's own resolved config first (authoritative: reflects
    exactly what got loaded), then falls back to scanning the local HF
    cache for the same repo id. `None` if neither resolves anything (e.g.
    fully offline/local weights with no cache entry)."""
    try:
        commit = getattr(model[0].auto_model.config, "_commit_hash", None)
        if commit:
            return commit
    except Exception:
        pass
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == model_id:
                revisions = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
                if revisions:
                    return revisions[0].commit_hash
    except Exception:
        pass
    return None


logger.info(f"Loading embedding model {MODEL_ID}...")
model = SentenceTransformer(MODEL_ID, **load_kwargs_for(MODEL_ID))
DIMENSION = (
    model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension")
    else model.get_sentence_embedding_dimension()
)
MAX_SEQ_TOKENS = int(getattr(model, "max_seq_length", 512) or 512)
REVISION = _resolve_revision(MODEL_ID, model)
QUERY_PROMPT_NAME = resolve_query_prompt_name(MODEL_ID, getattr(model, "prompts", None))
TOKENIZER_ID = getattr(model.tokenizer, "name_or_path", MODEL_ID)

# GPU lifecycle correction (production-readiness test, see DELIVERABLES.md):
# a model this size (fp16, ~8GB) left CUDA-resident between requests starves
# a concurrently loaded Ollama generator of VRAM -- measured directly as
# `ollama ps` reporting "77%/23% CPU/GPU" and ~6.6 tok/s for qwen3:14b with
# this server idle-resident. For a CUDA-lifecycle-managed model (Qwen3-
# Embedding; BGE is unaffected, see `cuda_lifecycle_managed`), move the
# model off the GPU immediately after startup -- every `/api/embed` request
# below moves it back to CUDA only for the duration of the actual
# `model.encode(...)` call. Idle VRAM footprint for this process is then
# ~0 regardless of which model is configured.
CUDA_MANAGED = cuda_lifecycle_managed(MODEL_ID) and torch.cuda.is_available()
if CUDA_MANAGED:
    model.to("cpu")
    torch.cuda.empty_cache()
    logger.info("CUDA lifecycle managed: model moved to CPU at idle; CUDA used only during encode()")

# GPU-stage-lifecycle correction (strict sequential GPU model lifecycle
# phase): a single lock around every device transition + encode() call.
# Flask's built-in dev server (app.run() below, threaded=False by default)
# already serializes requests, so this is defense-in-depth rather than a
# fix for an observed race -- it keeps `/api/embed` and the new explicit
# `/api/release` endpoint below from ever being able to interleave their
# `model.to(...)` calls if this server is ever run under a threaded/WSGI
# server in the future.
_device_lock = threading.Lock()


def _current_device() -> "torch.device":
    """The device the model's weights actually live on right now -- read
    from a real parameter, not tracked via a separate boolean that could
    drift from reality."""
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - a model always has parameters
        return torch.device("cpu")


def _is_cuda_oom(exc: BaseException) -> bool:
    """True when `exc` is a CUDA out-of-memory failure, regardless of how
    the installed torch version surfaces it: `torch.cuda.OutOfMemoryError`
    (a `RuntimeError` subclass) where importable, or any `RuntimeError`
    whose message carries the allocator's canonical "CUDA out of memory"
    text (older torch, and OOMs re-raised by wrappers that drop the
    specific type). Matching the message only as a fallback keeps
    unrelated RuntimeErrors (shape errors, dtype mismatches, ...) on the
    ordinary no-retry path."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc)


def encode_with_oom_retry(inputs: list, encode_kwargs: dict):
    """`model.encode(inputs, **encode_kwargs)` with an OOM-specific retry
    ladder: on a CUDA out-of-memory error, release the caching allocator's
    cached-but-unused blocks and retry with `batch_size` halved, repeating
    down to batch_size 1; only then let the exception propagate (the
    caller's generic 500 path). All other exceptions propagate immediately,
    unchanged.

    Why this exists: every other defense here only bounds THIS process's
    own GPU usage (CUDA_MANAGED idle offload, per-request empty_cache, a
    small static batch size). None of them help when the OOM is caused by
    EXTERNAL processes holding the memory -- observed 2026-08-10 as one
    document failing ingestion three times with "CUDA out of memory. Tried
    to allocate 234.00 MiB ... 92.00 MiB free" on an 11.61 GiB card this
    process was mostly not using. Halving the sub-batch shrinks the peak
    activation memory each forward pass asks for, which is exactly the
    margin that lets an encode squeeze into whatever memory the external
    processes left free; batch_size 1 is the floor below which there is
    nothing left to shrink. Embeddings are per-text, so changing
    `batch_size` never changes any embedding's value -- only peak memory."""
    batch_size = int(encode_kwargs.get("batch_size") or BATCH_SIZE)
    while True:
        try:
            return model.encode(inputs, **encode_kwargs)
        except Exception as e:
            if not _is_cuda_oom(e) or batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            logger.warning(
                f"CUDA OOM during encode of {len(inputs)} input(s); "
                f"retrying with batch_size={batch_size} after empty_cache()"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            encode_kwargs = {**encode_kwargs, "batch_size": batch_size}


def _release_model() -> bool:
    """Explicit VRAM release primitive (spec "EMBEDDING SERVER LIFECYCLE" /
    "explicit unload"): moves the model off CUDA and empties the caching
    allocator, for ANY configured backend -- not just the CUDA_MANAGED ones
    (Qwen3-Embedding). This is strictly additive to the existing per-request
    CUDA_MANAGED behavior already proven for Qwen3-Embedding (that behavior
    is unchanged: it still happens automatically on every `/api/embed` call
    with no caller action needed); this gives *any* caller (ingestion's
    Stage 3 end, a query's post-retrieval barrier) an on-demand way to
    guarantee release for whichever backend happens to be active, including
    BGE, which otherwise stays GPU-resident between requests exactly as it
    always has (spec "must work with the existing active embedding
    backend"). Idempotent and safe to call when already off CUDA or when
    CUDA isn't available at all. Returns whether the model is off CUDA when
    this returns."""
    with _device_lock:
        if not torch.cuda.is_available():
            return True
        if _current_device().type == "cuda":
            model.to("cpu")
        torch.cuda.empty_cache()
        return _current_device().type != "cuda"

logger.info(
    f"Model ready: dim={DIMENSION}, max_seq_tokens={MAX_SEQ_TOKENS}, revision={REVISION}, "
    f"query_prompt_name={QUERY_PROMPT_NAME}, tokenizer_id={TOKENIZER_ID}"
)

app = Flask(__name__)


def _tokenizer():
    return model.tokenizer


def count_tokens(text: str) -> int:
    try:
        return len(_tokenizer().encode(text, add_special_tokens=True))
    except Exception:
        # Conservative fallback so we never silently under-count.
        return int(len(text.split()) * 1.4) + 2


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    return [value]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "provider": "sentence-transformers-service",
        "model": MODEL_ID,
        "revision": REVISION,
        "dimension": DIMENSION,
        "max_sequence_tokens": MAX_SEQ_TOKENS,
        "normalized_embeddings": NORMALIZE,
        "query_prompt_name": QUERY_PROMPT_NAME,
        "tokenizer_id": TOKENIZER_ID,
        # Real, current residency -- not a cached assumption -- so a caller
        # (or the acceptance-test evidence gathering) can see whether this
        # process is holding GPU VRAM right now without needing nvidia-smi.
        "device": str(_current_device()),
        "cuda_lifecycle_managed": CUDA_MANAGED,
    })


@app.route("/api/release", methods=["POST"])
def release():
    """Explicit VRAM-release endpoint (spec "EMBEDDING SERVER LIFECYCLE" /
    "VRAM BARRIER"): moves this process's model off CUDA on demand,
    regardless of backend. A no-op-but-successful response when CUDA isn't
    available at all, or when the model is already off CUDA (e.g. a
    CUDA_MANAGED model between requests). Never unloads the model from RAM
    or terminates the process -- the model stays loaded and ready, just not
    occupying GPU memory, so the very next `/api/embed` call still works
    (it moves back to CUDA for the duration of that call) without needing a
    process restart."""
    released = _release_model()
    return jsonify({"released": released, "device": str(_current_device()), "cuda_available": torch.cuda.is_available()})


@app.route("/api/tokenize", methods=["POST"])
def tokenize():
    data = request.get_json(silent=True) or {}
    if "input" not in data:
        return jsonify({"error": "Request body must include an 'input' field (string or list of strings)"}), 400
    inputs = _as_list(data["input"])
    if not all(isinstance(t, str) for t in inputs):
        return jsonify({"error": "'input' must be a string or a list of strings"}), 400
    counts = [count_tokens(t) for t in inputs]
    return jsonify({"token_counts": counts, "max_sequence_tokens": MAX_SEQ_TOKENS})


@app.route("/api/embed", methods=["POST"])
def embed():
    data = request.get_json(silent=True) or {}
    if "input" not in data:
        return jsonify({"error": "Request body must include an 'input' field (string or list of strings)"}), 400
    inputs = _as_list(data["input"])
    if not inputs or not all(isinstance(t, str) and t.strip() for t in inputs):
        return jsonify({"error": "'input' must be a non-empty string or a non-empty list of non-empty strings"}), 400

    # "document" (default -- every pre-migration caller that never sends
    # this field at all keeps encoding exactly as before) or "query".
    # Anything else is a caller bug, not something to guess about.
    input_type = data.get("input_type", "document")
    try:
        encode_kwargs = build_encode_kwargs(
            input_type, QUERY_PROMPT_NAME, normalize_embeddings=NORMALIZE, batch_size=BATCH_SIZE,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    encode_kwargs["show_progress_bar"] = False

    reject = data.get("reject_oversized", True)
    token_counts = [count_tokens(t) for t in inputs]
    oversized = [i for i, c in enumerate(token_counts) if c > MAX_SEQ_TOKENS]
    if oversized and reject:
        logger.warning(f"Rejected embed request: {len(oversized)} of {len(inputs)} inputs exceed {MAX_SEQ_TOKENS} tokens")
        return jsonify({
            "error": "input_too_long",
            "message": (
                f"{len(oversized)} input(s) exceed this model's max sequence length of "
                f"{MAX_SEQ_TOKENS} tokens. Inputs are never silently truncated -- shorten the "
                f"text or split it into smaller chunks before embedding."
            ),
            "oversized_indices": oversized,
            "token_counts": token_counts,
            "max_sequence_tokens": MAX_SEQ_TOKENS,
        }), 413

    # CUDA lifecycle: this process's model may currently be off CUDA either
    # because it's CUDA_MANAGED (Qwen3-Embedding -- sits on CPU between
    # requests, see below) or because a caller explicitly called
    # `/api/release` since its last use (spec "VRAM BARRIER" -- any
    # backend, including BGE, can be explicitly released between GPU
    # stages). Either way, an actual `encode()` call always needs the model
    # on CUDA -- move it there first if it isn't already, so a backend that
    # was explicitly released still serves this request correctly instead
    # of silently falling back to a slow CPU encode.
    with _device_lock:
        if torch.cuda.is_available() and _current_device().type != "cuda":
            model.to("cuda")
    try:
        vectors = encode_with_oom_retry(inputs, encode_kwargs)
    except Exception as e:
        logger.error(f"Embedding failed for a batch of {len(inputs)} input(s): {type(e).__name__}")
        return jsonify({"error": "embedding_failed", "message": str(e)}), 500
    finally:
        # Only a CUDA_MANAGED model (Qwen3-Embedding) automatically returns
        # to CPU after every single request -- BGE's proven "stay resident,
        # explicitly release only when a caller asks" behavior is
        # unchanged, matching its documented production characteristics.
        if CUDA_MANAGED:
            with _device_lock:
                model.to("cpu")
        # Standard CUDA allocator hygiene, not a workaround for a leak: a
        # long-running process encoding batches of widely varying
        # length/size (short queries interleaved with near-4096-token
        # document chunks) fragments the caching allocator's freed blocks
        # over time, which can eventually surface as an out-of-memory error
        # even though total bytes actually in use stay well within the
        # GPU's capacity. Releasing cached-but-unused blocks after every
        # request keeps that fragmentation from accumulating across the
        # process's lifetime -- this changes nothing about how any single
        # request is encoded, and is not accelerate/quantization/
        # device_map/CPU-offloading. For a CUDA_MANAGED model this also
        # releases the VRAM just vacated by the `model.to("cpu")` above, so
        # the next thing to allocate on this GPU (e.g. Ollama) sees it free.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return jsonify({
        "embeddings": [v.tolist() for v in vectors],
        "model": MODEL_ID,
        "dimension": DIMENSION,
        "token_counts": token_counts,
        "normalized_embeddings": NORMALIZE,
        "input_type": input_type,
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=PORT)

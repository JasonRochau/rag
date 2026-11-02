"""
Single reusable embedding client (spec section 3.3 / 13).

Both ingestion and search go through this class so the query and document
vectors are always produced by the same configured model, and so the
service's actual dimension / max-sequence-length are read once and trusted
everywhere instead of being duplicated as constants.
"""
from __future__ import annotations

import os
import time

import requests
from dataclasses import dataclass
from typing import List, Optional


# Retry policy for transient server-side embedding failures (HTTP 500 with
# an `embedding_failed` body -- the shape embedding_server.py returns when
# encode() raised, e.g. a CUDA OOM it could not clear even at batch_size 1
# because EXTERNAL processes hold the GPU memory). Those are worth waiting
# out: the external holder may release the GPU seconds later, whereas the
# alternative is failing the whole document's ingestion. Attempts/waits are
# env-tunable so an operator can lengthen the window without a code change.
# 413 (oversized input) and other 4xx are caller bugs and are never
# retried; neither are non-500 5xx or non-`embedding_failed` bodies, which
# this policy says nothing about.
EMBEDDING_CLIENT_RETRIES = int(os.environ.get("EMBEDDING_CLIENT_RETRIES", "3"))
EMBEDDING_CLIENT_RETRY_BASE_DELAY = float(os.environ.get("EMBEDDING_CLIENT_RETRY_BASE_DELAY", "5.0"))


class EmbeddingServiceError(RuntimeError):
    pass


class EmbeddingOversizedInputError(EmbeddingServiceError):
    def __init__(self, message: str, oversized_indices: List[int], token_counts: List[int]):
        super().__init__(message)
        self.oversized_indices = oversized_indices
        self.token_counts = token_counts


class EmbeddingCompatibilityError(RuntimeError):
    """Raised when an index's recorded embedding metadata does not match the live service."""


def _is_embedding_failed(resp) -> bool:
    """True when a 500 response body is the embedding server's
    `embedding_failed` error shape -- the only server-side failure this
    client retries. Non-JSON or differently-shaped 500 bodies (a proxy
    error page, an unrelated server bug) are not transient-encode failures
    and fall through to the immediate raise."""
    try:
        return resp.json().get("error") == "embedding_failed"
    except Exception:
        return False


@dataclass
class EmbeddingModelInfo:
    provider: str
    model: str
    dimension: int
    max_sequence_tokens: int
    normalized_embeddings: bool
    # Populated for models the embedding service resolves from a versioned
    # hub repo (e.g. Hugging Face) -- the exact commit/snapshot actually
    # loaded, not just the (possibly floating, e.g. "main") configured
    # model id. `None` when the service can't determine one (offline/local
    # weights with no resolvable revision).
    revision: Optional[str] = None
    # The `SentenceTransformer` prompt name the service will apply for
    # `input_type="query"` (e.g. "query" for Qwen3-Embedding), or `None`
    # when the loaded model has no query-specific prompt configured (e.g.
    # BGE) -- document and query encoding are then identical, unchanged
    # from pre-migration behavior.
    query_prompt_name: Optional[str] = None
    # Identity of the tokenizer actually used for `count_tokens`/packing
    # decisions (normally the same repo id as `model`, surfaced separately
    # since it's what chunk-size measurement is actually keyed off).
    tokenizer_id: Optional[str] = None


class EmbeddingClient:
    def __init__(self, host: str, model: str, timeout: float = 60.0):
        self.host = host.rstrip("/")
        self.expected_model = model
        self.timeout = timeout
        self._info: Optional[EmbeddingModelInfo] = None

    def health(self, force: bool = False) -> EmbeddingModelInfo:
        if self._info is not None and not force:
            return self._info
        try:
            resp = requests.get(self.host + "/health", timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise EmbeddingServiceError(f"Embedding service at {self.host} is unreachable: {e}") from e
        info = EmbeddingModelInfo(
            provider=data.get("provider", "sentence-transformers-service"),
            model=data.get("model", ""),
            dimension=int(data.get("dimension", 0)),
            max_sequence_tokens=int(data.get("max_sequence_tokens", 0)),
            normalized_embeddings=bool(data.get("normalized_embeddings", True)),
            revision=data.get("revision"),
            query_prompt_name=data.get("query_prompt_name"),
            tokenizer_id=data.get("tokenizer_id"),
        )
        if self.expected_model and info.model and info.model != self.expected_model:
            raise EmbeddingCompatibilityError(
                f"Configured embedding model {self.expected_model!r} does not match the model "
                f"actually loaded by the embedding service ({info.model!r}). Update EMBEDDING_MODEL "
                f"or restart the embedding service with the expected model."
            )
        self._info = info
        return info

    def count_tokens(self, text: str) -> int:
        try:
            resp = requests.post(self.host + "/api/tokenize", json={"input": text}, timeout=self.timeout)
            resp.raise_for_status()
            return int(resp.json()["token_counts"][0])
        except EmbeddingServiceError:
            raise
        except Exception:
            return int(round(len(text.split()) * 1.4)) + 2

    def embed_batch(
        self, texts: List[str], reject_oversized: bool = True, input_type: str = "document",
    ) -> List[List[float]]:
        """`input_type` distinguishes how the service encodes `texts`:
        "document" (the default -- every existing ingestion call site is
        unchanged) encodes as-is; "query" applies the model's query prompt
        (e.g. Qwen3-Embedding's `prompt_name="query"`) when the loaded model
        defines one, and is a no-op for a model that doesn't (BGE). Never
        pass "query" for chunks being indexed, or "document" for a search
        query -- ingestion and search must stay on their own side of this."""
        if input_type not in ("document", "query"):
            raise ValueError(f"input_type must be 'document' or 'query', got {input_type!r}")
        if not texts:
            return []
        attempts = max(1, EMBEDDING_CLIENT_RETRIES)
        resp = None
        for attempt in range(attempts):
            try:
                resp = requests.post(
                    self.host + "/api/embed",
                    json={"input": texts, "reject_oversized": reject_oversized, "input_type": input_type},
                    timeout=self.timeout,
                )
            except Exception as e:
                raise EmbeddingServiceError(f"Embedding request to {self.host} failed: {e}") from e

            if resp.status_code == 413:
                data = resp.json()
                raise EmbeddingOversizedInputError(
                    data.get("message", "Input exceeds the embedding model's max sequence length"),
                    data.get("oversized_indices", []),
                    data.get("token_counts", []),
                )
            if resp.ok:
                break
            if not (resp.status_code == 500 and _is_embedding_failed(resp)) or attempt == attempts - 1:
                raise EmbeddingServiceError(f"Embedding service returned HTTP {resp.status_code}: {resp.text[:300]}")
            # Transient server-side encode failure (see the module-level
            # policy comment): back off and let the server/GPU recover.
            # Base delay scales by 3x per attempt (5s, 15s, ... by default)
            # -- the first retry catches holders that release immediately,
            # the longer second wait catches slower ones.
            delay = EMBEDDING_CLIENT_RETRY_BASE_DELAY * (3 ** attempt)
            time.sleep(delay)

        data = resp.json()
        model = data.get("model", "")
        if self.expected_model and model and model != self.expected_model:
            raise EmbeddingCompatibilityError(
                f"Embedding service embedded with model {model!r} but pipeline is configured for "
                f"{self.expected_model!r}."
            )
        return data["embeddings"]

    def embed_one(self, text: str, input_type: str = "document") -> List[float]:
        return self.embed_batch([text], input_type=input_type)[0]

    def release(self) -> bool:
        """Explicit VRAM-release barrier (spec "strict sequential GPU model
        lifecycle" / "EMBEDDING SERVER LIFECYCLE"): asks the embedding
        service to move its model off CUDA right now, regardless of
        backend -- proven automatic-per-request for Qwen3-Embedding
        already (see embedding_server.py's `CUDA_MANAGED`); this is the
        same primitive made explicitly callable for any backend (including
        BGE, which otherwise stays GPU-resident between requests) so a
        caller finishing an embedding-dependent stage (query retrieval,
        end of ingestion's embed-everything stage) can hand the GPU back
        before the next stage (answer generation, visual analysis) loads
        its own model. Best-effort: returns False (never raises) if the
        service is unreachable -- failing to release is reported, not
        fatal, the same tolerance every other stage-cleanup call in this
        pipeline already has."""
        try:
            resp = requests.post(self.host + "/api/release", timeout=self.timeout)
            return resp.ok and bool(resp.json().get("released"))
        except Exception:
            return False

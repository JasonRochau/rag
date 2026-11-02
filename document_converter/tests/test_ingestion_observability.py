"""
Tests for the ingestion-observability task:
  * Part 1 -- batched embedding (pipeline.py's `index()`), including the
    global-vs-batch-relative chunk index fix for
    `EmbeddingOversizedInputError`.
  * Part 2 -- the `progress` callback contract threaded through
    `DocumentPipeline` (monotonic `current` within a stage, a raising
    callback never breaks the ingest).

The event-loop / non-blocking regression test lives in
test_orchestrator_docs_api.py (it needs the orchestrator's job registry);
this file only exercises `DocumentPipeline` directly, reusing the
`pipeline`/`pipeline_config`/`write_source` fixtures from conftest.py.
"""
from __future__ import annotations

import pytest

from document_converter.embedding_client import EmbeddingModelInfo, EmbeddingOversizedInputError
from document_converter.index_schema import load_index


def _many_sections_doc(n: int) -> str:
    """A document with `n` H2 sections, each its own paragraph -- with
    `cross_section_packing` off (the default), the chunker gives each
    section its own chunk, so this reliably produces `n` chunks."""
    parts = ["# Title\n"]
    for i in range(n):
        parts.append(
            f"## Section {i}\n\nDistinct padding content for section number {i} so this paragraph "
            f"is a real chunk of its own and never merges with a neighboring section {i}.\n"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------- #
# Part 1: batched embedding

def test_embedding_batching_produces_one_embedding_per_chunk(pipeline, pipeline_config, write_source):
    pipeline_config.embedding.batch_size = 3
    src = write_source("many.md", _many_sections_doc(10))

    result = pipeline.index(str(src))

    n_chunks = len(result.chunking_result.chunks)
    assert n_chunks > pipeline_config.embedding.batch_size  # actually exercises multiple batches
    payload_embeddings = [c["embedding"] for c in load_index(result.index_path)["chunks"]
                           if c.get("content_kind", "main_text") == "main_text"]
    assert len(payload_embeddings) == n_chunks

    calls = pipeline._embedding_client.embed_calls
    assert len(calls) > 1  # more than one embed_batch() call was actually made
    for call in calls:
        assert len(call) <= pipeline_config.embedding.batch_size
    assert sum(len(c) for c in calls) == n_chunks


class _OversizedOnSecondBatchClient:
    """Deterministic stand-in for EmbeddingClient whose second `embed_batch`
    call fails exactly the way the real service does on a 413 (see
    embedding_server.py): `oversized_indices`/`token_counts` are positions
    within *that call's* input list, not the whole corpus."""

    def __init__(self, dimension: int = 8, max_sequence_tokens: int = 512, local_index: int = 1):
        self.dimension = dimension
        self.max_sequence_tokens = max_sequence_tokens
        self.local_index = local_index
        self.calls = 0
        self.release_calls = 0

    def health(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(provider="fake", model="fake-model", dimension=self.dimension,
                                   max_sequence_tokens=self.max_sequence_tokens, normalized_embeddings=True)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_batch(self, texts, reject_oversized: bool = True, input_type: str = "document"):
        call_index = self.calls
        self.calls += 1
        if call_index == 1:
            raise EmbeddingOversizedInputError(
                "too long", oversized_indices=[self.local_index], token_counts=[99999] * len(texts),
            )
        return [[0.1] * self.dimension for _ in texts]

    def embed_one(self, text: str, input_type: str = "document"):
        return self.embed_batch([text], input_type=input_type)[0]

    def release(self) -> bool:
        self.release_calls += 1
        return True


def test_oversized_error_from_non_first_batch_reports_global_chunk_index(pipeline, pipeline_config, write_source):
    batch_size = 3
    pipeline_config.embedding.batch_size = batch_size
    fake = _OversizedOnSecondBatchClient(local_index=1)
    pipeline._embedding_client = fake
    src = write_source("many.md", _many_sections_doc(10))

    with pytest.raises(EmbeddingOversizedInputError) as exc_info:
        pipeline.index(str(src))

    # The failure is on the *second* embed_batch call (batch index 1), which
    # covers global chunks [batch_size, 2*batch_size); local index 1 within
    # it is global index `batch_size + 1` -- never the raw local index 1.
    assert exc_info.value.oversized_indices == [batch_size + 1]
    assert fake.calls == 2  # never attempted a third batch after the failure


# ---------------------------------------------------------------------- #
# Part 2: progress callback contract

def test_progress_callback_is_monotonic_per_stage_and_survives_raising(pipeline, pipeline_config, write_source):
    pipeline_config.embedding.batch_size = 3
    src = write_source("many.md", _many_sections_doc(10))

    events = []

    def flaky_progress(stage, payload):
        events.append((stage, dict(payload)))
        raise RuntimeError("progress callbacks must never break an ingest")

    result = pipeline.index(str(src), progress=flaky_progress)

    assert result.index_path  # the ingest completed despite every callback raising
    assert events  # the callback was actually invoked

    by_stage = {}
    for stage, payload in events:
        if "current" not in payload:
            continue
        by_stage.setdefault(stage, []).append(payload["current"])
    for stage, currents in by_stage.items():
        assert currents == sorted(currents), f"{stage!r} current values were not monotonic: {currents}"

    assert "embedding" in by_stage
    assert by_stage["embedding"][-1] == len(result.chunking_result.chunks)


def test_progress_callback_default_none_is_unaffected(pipeline, write_source):
    """Strictly additive: no progress= at all reproduces plain index()."""
    src = write_source("doc.md", "# Title\n\nSome content.\n")
    result = pipeline.index(str(src))
    assert result.index_path

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from document_converter.config import PipelineConfig
from document_converter.embedding_client import EmbeddingModelInfo
from document_converter.pipeline import DocumentPipeline

FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbeddingClient:
    """Deterministic, offline stand-in for EmbeddingClient used by pipeline/orchestrator tests."""

    def __init__(self, dimension: int = 8, max_sequence_tokens: int = 512):
        self.dimension = dimension
        self.max_sequence_tokens = max_sequence_tokens
        self.embed_calls: list = []
        self.release_calls: int = 0

    def health(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            provider="fake", model="fake-model", dimension=self.dimension,
            max_sequence_tokens=self.max_sequence_tokens, normalized_embeddings=True,
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vector_for(self, text: str) -> list:
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in h[: self.dimension]]

    def embed_batch(self, texts, reject_oversized: bool = True, input_type: str = "document"):
        self.embed_calls.append(list(texts))
        return [self._vector_for(t) for t in texts]

    def embed_one(self, text: str, input_type: str = "document") -> list:
        return self.embed_batch([text], input_type=input_type)[0]

    def release(self) -> bool:
        self.release_calls += 1
        return True


@pytest.fixture
def pipeline_config(tmp_path) -> PipelineConfig:
    config = copy.deepcopy(PipelineConfig())
    config.paths.source_dir = str(tmp_path / "sources")
    config.paths.processed_dir = str(tmp_path / "processed")
    config.paths.report_dir = str(tmp_path / "reports")
    config.paths.index_dir = str(tmp_path / "index")
    config.paths.legacy_docs_dir = str(tmp_path / "legacy_docs")
    config.paths.ensure()
    config.semantic.mode = "off"
    return config


@pytest.fixture
def pipeline(pipeline_config) -> DocumentPipeline:
    p = DocumentPipeline(pipeline_config)
    p._embedding_client = FakeEmbeddingClient()
    return p


@pytest.fixture
def write_source(tmp_path):
    def _write(name: str, content, binary: bool = False):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if binary:
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path
    return _write

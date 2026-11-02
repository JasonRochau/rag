"""
document_converter: format-agnostic ingestion pipeline.

Any supported source file (markdown, txt, html, docx, pdf) is converted to
a canonical internal `Document` (model.py), deterministically normalized
and rendered to Markdown, chunked with `MarkdownChunker`, validated, and
optionally embedded and written to a versioned index. `DocumentPipeline`
(pipeline.py) is the single entry point that ties these stages together;
everything else (ingestion.py, orchestrator.py, cli.py) calls into it
rather than re-implementing any stage.
"""
from .config import PipelineConfig, load_config, DEFAULT_CONFIG
from .pipeline import DocumentPipeline, ProcessResult, PipelineValidationError, stable_document_id

__all__ = [
    "PipelineConfig", "load_config", "DEFAULT_CONFIG",
    "DocumentPipeline", "ProcessResult", "PipelineValidationError", "stable_document_id",
]

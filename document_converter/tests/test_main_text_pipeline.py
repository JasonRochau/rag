"""
Integration tests for the MAIN_TEXT/FULL_FIDELITY profile split through the
real `DocumentPipeline` (spec "MAIN_TEXT vs FULL_FIDELITY ingestion
profiles"). Uses the `pipeline`/`write_source` fixtures from conftest.py
(markdown source, `FakeEmbeddingClient`) so these run fast and
deterministically -- real-PDF/live-embedding-service validation for the
regression-analysis and OpenCL papers is covered separately (see
DELIVERABLES.md's real-document validation table), not duplicated here as
a slow unit test.
"""
from __future__ import annotations

from document_converter.config import MAIN_TEXT_PROFILE, FULL_FIDELITY_PROFILE

_DOC_WITH_FRONT_MATTER_AND_FIGURE = """# A Study of Something Interesting

Jane Doe, John Smith

Institute of Somewhere

jane@example.com

Copyright 2024 Example Corp. All rights reserved.

## Introduction

This is the first real paragraph of the introduction, written with enough words that it does not look like a stray author byline or a two-word fragment, and it explains what the paper is about in ordinary prose.

Fig. 1. A diagram of the system architecture.

## Conclusion

This is the concluding paragraph, again written as ordinary prose with several full sentences describing what was learned.
"""


def test_main_text_is_the_default_profile(pipeline, write_source):
    src = write_source("doc.md", "# Title\n\nSome ordinary body prose here.\n")
    result = pipeline.preview(str(src))
    assert result.profile == MAIN_TEXT_PROFILE


def test_main_text_chunk_contains_no_excluded_block_types(pipeline, write_source):
    src = write_source("doc.md", _DOC_WITH_FRONT_MATTER_AND_FIGURE)
    result = pipeline.preview(str(src))
    assert result.profile == MAIN_TEXT_PROFILE
    for c in result.chunking_result.chunks:
        assert "AuthorBlock" not in c.content_types
        assert "PublicationMetadataBlock" not in c.content_types
        assert "CaptionBlock" not in c.content_types
        full_text = c.text.lower()
        assert "jane doe" not in full_text
        assert "copyright" not in full_text
        assert "fig. 1" not in full_text


def test_main_text_chunk_has_minimal_required_metadata(pipeline, write_source):
    src = write_source("doc.md", _DOC_WITH_FRONT_MATTER_AND_FIGURE)
    result = pipeline.preview(str(src))
    assert result.title
    assert result.source_path
    for c in result.chunking_result.chunks:
        assert c.chunk_id
        assert c.header_path or c.heading  # section/heading context
        assert c.provenance  # stable citation string (filename + heading path)


def test_main_text_chunks_fit_configured_tokenizer_budget(pipeline, write_source):
    """Uses the fixture's FakeEmbeddingClient (word-count tokenizer,
    max_sequence_tokens=512) as a deterministic stand-in for the real BGE
    tokenizer -- the live-service check against the real model happens in
    the real-document validation pass, not in this fast unit test."""
    src = write_source("doc.md", _DOC_WITH_FRONT_MATTER_AND_FIGURE)
    result = pipeline.index(str(src))
    assert result.profile == MAIN_TEXT_PROFILE
    max_tokens = pipeline._embedding_client.max_sequence_tokens
    for c in result.chunking_result.chunks:
        assert c.token_count <= max_tokens


def test_main_text_deterministic_chunk_ids_and_boundaries(pipeline_config, write_source):
    from document_converter.pipeline import DocumentPipeline
    from document_converter.tests.conftest import FakeEmbeddingClient

    src = write_source("doc.md", _DOC_WITH_FRONT_MATTER_AND_FIGURE)
    p1 = DocumentPipeline(pipeline_config)
    p1._embedding_client = FakeEmbeddingClient()
    p2 = DocumentPipeline(pipeline_config)
    p2._embedding_client = FakeEmbeddingClient()

    r1 = p1.preview(str(src))
    r2 = p2.preview(str(src))
    ids1 = [c.chunk_id for c in r1.chunking_result.chunks]
    ids2 = [c.chunk_id for c in r2.chunking_result.chunks]
    assert ids1 == ids2
    texts1 = [c.text for c in r1.chunking_result.chunks]
    texts2 = [c.text for c in r2.chunking_result.chunks]
    assert texts1 == texts2


def test_emergency_split_reconstructs_original_paragraph_text(pipeline_config, write_source):
    """A legitimate long paragraph (spec Phase 6) must be split on sentence
    boundaries with every character preserved and sentence order intact --
    joining the resulting pieces back together must reproduce the original
    text exactly."""
    from document_converter.pipeline import DocumentPipeline
    from document_converter.tests.conftest import FakeEmbeddingClient

    sentences = [f"This is sentence number {i} of a very long paragraph about testing." for i in range(60)]
    long_paragraph = " ".join(sentences)
    text = f"# Title\n\n## Section\n\n{long_paragraph}\n"
    src = write_source("doc.md", text)

    cfg = pipeline_config
    cfg.chunking.hard_max_words = 40
    cfg.chunking.target_words = 30
    p = DocumentPipeline(cfg)
    p._embedding_client = FakeEmbeddingClient()

    result = p.preview(str(src))
    assert result.validation_report.retrieval_ready is True
    # Every forced-fallback chunk's body should be a verbatim run of
    # sentences from the original paragraph -- concatenating them (minus
    # the heading/provenance wrapper) must reconstruct the source text.
    fallback_chunks = [c for c in result.chunking_result.chunks if c.forced_fallback]
    assert fallback_chunks, "expected the oversized paragraph to trigger the emergency split path"
    reconstructed = " ".join(c.body.split("\n\n", 1)[1].strip() for c in fallback_chunks)
    assert reconstructed.replace("  ", " ") == long_paragraph.replace("  ", " ")


def test_full_fidelity_preserves_author_and_figure_blocks(pipeline, write_source):
    src = write_source("doc.md", _DOC_WITH_FRONT_MATTER_AND_FIGURE)
    result = pipeline.preview(str(src), profile=FULL_FIDELITY_PROFILE)
    assert result.profile == FULL_FIDELITY_PROFILE
    kinds = {type(b).__name__ for b in result.document.blocks}
    assert "AuthorBlock" in kinds
    assert "CaptionBlock" in kinds or any(
        "fig. 1" in (getattr(b, "text", "") or "").lower() for b in result.document.blocks
    )


def test_full_fidelity_never_applies_the_main_text_overlay(pipeline, write_source):
    """The MAIN_TEXT exemption/readiness overlay (retrieval_readiness.py)
    must only ever run for the MAIN_TEXT profile -- structurally verified
    here by checking the overlay's own stats keys are absent under
    FULL_FIDELITY (see `test_front_matter_normalization.py::
    test_genuinely_interleaved_front_matter_still_fails` for proof that
    genuine front-matter interleaving is still caught at the validation.py
    level, unaffected by this profile split)."""
    src = write_source("doc.md", "# Title\n\nOrdinary prose paragraph.\n")
    full = pipeline.preview(str(src), profile=FULL_FIDELITY_PROFILE)
    assert "main_text_title" not in full.validation_report.stats
    assert full.validation_report.stats["errors"] == full.validation_report.stats.get(
        "main_text_blocking_error_count", full.validation_report.stats["errors"]
    )

    main_text = pipeline.preview(str(src), profile=MAIN_TEXT_PROFILE)
    assert "main_text_title" in main_text.validation_report.stats


def test_renderer_and_chunker_share_the_same_active_document(pipeline, write_source, monkeypatch):
    """The exact bug shape reported against the exported Markdown: the
    Markdown renderer, the chunker, and (transitively) `preview()`/
    `process()`'s output and `index()`'s indexed chunks must all trace back
    to the identical `active_document` object the MAIN_TEXT profile
    selected -- not two separately-built-but-hopefully-equal ones. Spies on
    `render_markdown` and `MarkdownChunker.chunk` to capture the exact
    `Document` object each receives and asserts they're literally `is` the
    same object (stronger than equal content), for both the offline
    `preview()` path and the real-tokenizer `index()` path."""
    import document_converter.pipeline as pipeline_module
    from document_converter.chunker import MarkdownChunker

    rendered_docs = []
    chunked_docs = []

    real_render = pipeline_module.render_markdown
    def spy_render(document):
        rendered_docs.append(document)
        return real_render(document)
    monkeypatch.setattr(pipeline_module, "render_markdown", spy_render)

    real_chunk = MarkdownChunker.chunk
    def spy_chunk(self, document, **kwargs):
        chunked_docs.append(document)
        return real_chunk(self, document, **kwargs)
    monkeypatch.setattr(MarkdownChunker, "chunk", spy_chunk)

    src = write_source("doc.md", _DOC_WITH_FRONT_MATTER_AND_FIGURE)

    pipeline.preview(str(src))
    assert len(rendered_docs) == 1 and len(chunked_docs) == 1
    assert rendered_docs[0] is chunked_docs[0]

    rendered_docs.clear()
    chunked_docs.clear()
    result = pipeline.index(str(src), force=True)
    # index() runs _prepare_for_profile (chunk) twice -- once inside the
    # offline process() pass, once with the real tokenizer -- but the
    # *final* chunk call (the one whose output actually gets embedded and
    # indexed) must be paired with the *final* render call (the one whose
    # output is actually written to the downloadable .md file).
    assert rendered_docs[-1] is chunked_docs[-1]
    assert result.markdown == pipeline_module.render_markdown(chunked_docs[-1])

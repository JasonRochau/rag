import os
import shutil
import subprocess

import pytest
from docx import Document as DocxDocument

from document_converter.pipeline import DocumentPipeline, stable_document_id
from document_converter.config import ChunkingConfig
from document_converter.index_schema import load_index

HAS_SOFFICE = shutil.which("soffice") is not None or shutil.which("libreoffice") is not None


def test_stable_document_id_is_deterministic_and_path_based():
    a = stable_document_id("sub/dir/doc.md")
    b = stable_document_id("sub/dir/doc.md")
    c = stable_document_id("sub/dir/other.md")
    assert a == b
    assert a != c
    assert a.startswith("d-")


def test_preview_does_not_touch_disk(pipeline, write_source, pipeline_config):
    src = write_source("doc.md", "# Title\n\nSome content.\n")
    result = pipeline.preview(str(src))
    assert result.markdown_path is None
    assert result.index_path is None
    assert os.listdir(pipeline_config.paths.processed_dir) == []
    assert os.listdir(pipeline_config.paths.index_dir) == []
    assert result.chunking_result.chunks


def test_process_writes_markdown_and_report(pipeline, write_source):
    src = write_source("doc.md", "# Title\n\nSome content.\n")
    result = pipeline.process(str(src))
    assert os.path.exists(result.markdown_path)
    assert os.path.exists(result.report_path)
    assert "Title" in open(result.markdown_path).read()


def test_index_embeds_and_writes_versioned_index(pipeline, write_source):
    src = write_source("doc.md", "# Title\n\nSome content for indexing.\n")
    result = pipeline.index(str(src))
    assert result.index_path and os.path.exists(result.index_path)
    payload = load_index(result.index_path)
    assert payload["schema_version"] == 2
    assert len(payload["chunks"]) == len(result.chunking_result.chunks)
    for c in payload["chunks"]:
        assert c["embedding"]  # every chunk was actually embedded
    assert pipeline._embedding_client.embed_calls  # fake client was used


def test_index_is_idempotent_and_skips_unchanged(pipeline, write_source):
    src = write_source("doc.md", "# Title\n\nContent.\n")
    r1 = pipeline.index(str(src))
    calls_after_first = len(pipeline._embedding_client.embed_calls)
    r2 = pipeline.index(str(src))
    assert r2.skipped_unchanged is True
    assert len(pipeline._embedding_client.embed_calls) == calls_after_first  # no re-embedding


def test_index_reprocesses_when_source_changes(pipeline, write_source):
    src = write_source("doc.md", "# Title\n\nOriginal.\n")
    pipeline.index(str(src))
    src.write_text("# Title\n\nChanged content now.\n", encoding="utf-8")
    r2 = pipeline.index(str(src))
    assert r2.skipped_unchanged is False


def test_index_force_reindexes_even_when_unchanged(pipeline, write_source):
    src = write_source("doc.md", "# Title\n\nContent.\n")
    pipeline.index(str(src))
    r2 = pipeline.index(str(src), force=True)
    assert r2.skipped_unchanged is False


def test_strict_mode_force_splits_oversized_paragraph_instead_of_raising(pipeline, write_source):
    # Policy change (2026-09-23): the strict/FULL_FIDELITY "never split a
    # paragraph" error+drop path is retired -- an oversized paragraph routes
    # through the emergency sentence-boundary split in EVERY profile, so one
    # legitimate long paragraph no longer fails the whole upload; it indexes
    # with a forced_fallback_split WARNING instead of raising
    # PipelineValidationError.
    pipeline.config.chunking = ChunkingConfig(hard_max_words=20, target_words=15, strict=True)
    pipeline.config.strict = True
    src = write_source("doc.md", "# Title\n\n" + ("word " * 200))
    result = pipeline.index(str(src), profile="full_fidelity")
    assert result.chunking_result.chunks
    assert any(
        f.code == "forced_fallback_split" for f in result.chunking_result.findings.findings
    )


def test_delete_removes_index_markdown_and_report(pipeline, write_source):
    src = write_source("doc.md", "# Title\n\nContent.\n")
    result = pipeline.index(str(src))
    pipeline.delete(result.document_id)
    assert not os.path.exists(result.index_path)
    assert not os.path.exists(result.markdown_path)
    assert not os.path.exists(result.report_path)


def test_unsupported_extension_raises(pipeline, write_source):
    src = write_source("doc.exe", "not supported")
    with pytest.raises(ValueError):
        pipeline.preview(str(src))


def test_scan_and_index_walks_directory_and_is_idempotent(pipeline, tmp_path):
    root = tmp_path / "scanroot"
    (root / "a.md").parent.mkdir(parents=True, exist_ok=True)
    # Ordinary prose, not "Content A." -- a bare 2-word Title-Case fragment
    # right after the title shape-matches the generic author-name-line
    # heuristic (front_matter.py's `_looks_like_author_line`, correctly
    # tuned for real short author bylines) and would otherwise become a
    # spurious AuthorBlock, excluded by MAIN_TEXT and hollowing out this
    # already-minimal fixture. Real prose sentences don't hit this shape.
    (root / "a.md").write_text("# A\n\nSome ordinary content for document A.\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "b.md").write_text("# B\n\nSome ordinary content for document B.\n", encoding="utf-8")

    results = pipeline.scan_and_index(str(root))
    assert {r["status"] for r in results} == {"indexed"}
    assert len(results) == 2

    results2 = pipeline.scan_and_index(str(root))
    assert {r["status"] for r in results2} == {"skipped"}


def test_scan_and_index_skips_pipelines_own_managed_dirs(pipeline, pipeline_config):
    # source_dir is one of the pipeline's managed dirs; a file placed inside a
    # parent that also contains it should not be double-indexed via both paths.
    os.makedirs(pipeline_config.paths.source_dir, exist_ok=True)
    with open(os.path.join(pipeline_config.paths.source_dir, "inner.md"), "w") as f:
        f.write("# Inner\n\ncontent\n")
    parent = os.path.dirname(pipeline_config.paths.source_dir)
    results = pipeline.scan_and_index(parent)
    assert results == []


def test_completion_state_reflects_indexed_status(pipeline, write_source):
    src = write_source("doc.md", "# Title\n\nContent.\n")

    previewed = pipeline.preview(str(src))
    assert previewed.completion_state["indexed"] is False

    indexed = pipeline.index(str(src))
    assert indexed.completion_state["indexed"] is True
    assert indexed.completion_state["retrieval_ready"] is True

    unchanged = pipeline.index(str(src))
    assert unchanged.skipped_unchanged is True
    assert unchanged.completion_state["indexed"] is True


def test_index_persists_findings_and_stats_for_diagnostics_ui(pipeline, write_source):
    """document_converter.html's "View Details" panel reads
    `validation.findings`/`validation.stats` back off the written index
    payload (via GET /docs/{doc_id}) -- index() must actually persist the
    same FindingLog/stats validate() produced, not just the warnings/errors
    counts that were already stored before this."""
    src = write_source("doc.md", "# Title\n\n" + ("word " * 200) + "\n")
    result = pipeline.index(str(src))
    payload = load_index(result.index_path)

    assert payload["validation"]["stats"] == result.validation_report.stats
    assert payload["validation"]["findings"] == result.validation_report.findings.to_list()
    # The long paragraph is a real, expected WARNING for this fixture --
    # confirms the persisted list isn't just an empty placeholder.
    codes = [f["code"] for f in payload["validation"]["findings"]]
    assert "long_paragraph" in codes
    assert payload["validation"]["stats"]["page_count"] is not None or payload["validation"]["stats"]["heading_count"] >= 1


@pytest.mark.skipif(not HAS_SOFFICE, reason="LibreOffice not installed")
def test_legacy_doc_flows_through_full_pipeline(pipeline, tmp_path):
    src_docx = tmp_path / "legacy.docx"
    d = DocxDocument()
    d.add_heading("Legacy Pipeline Test", level=1)
    d.add_paragraph("Body content that should survive the .doc round trip.")
    d.save(str(src_docx))

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    subprocess.run(
        [soffice, "--headless", "--norestore", "--convert-to", "doc", "--outdir", str(tmp_path), str(src_docx)],
        capture_output=True, timeout=60, check=True,
    )
    src_doc = tmp_path / "legacy.doc"
    assert src_doc.exists()

    result = pipeline.index(str(src_doc))
    assert result.chunking_result.chunks
    assert result.validation_report.retrieval_ready is True
    payload = load_index(result.index_path)
    assert payload["pipeline"]["extractor"] == "doc"
    assert payload["chunks"][0]["embedding"]

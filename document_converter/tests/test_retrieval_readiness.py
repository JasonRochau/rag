"""
Unit tests for retrieval_readiness.py's MAIN_TEXT gating policy (spec
"MAIN_TEXT vs FULL_FIDELITY ingestion profiles", Phase 4). Builds
`ValidationReport` objects directly rather than running the full pipeline,
so these isolate the policy layer from extraction/chunking.
"""
from __future__ import annotations

from document_converter.findings import FindingLog
from document_converter.validation import ValidationReport
from document_converter.retrieval_readiness import evaluate_main_text_readiness, apply_main_text_overlay


def _report(error_codes=(), stats=None):
    log = FindingLog()
    for code in error_codes:
        log.error(code, f"{code} happened")
    base_stats = {"chunk_count": 1, "empty_chunk_count": 0}
    base_stats.update(stats or {})
    return ValidationReport(findings=log, stats=base_stats, retrieval_ready=False)


# ---------------------------------------------------------------------- #
# Non-blocking diagnostic findings (spec items 13-15)

def test_misordered_author_metadata_does_not_block():
    report = _report(["author_metadata_interleaved_with_body"])
    readiness = evaluate_main_text_readiness(report, "Title", "metadata", has_body_content=True)
    assert readiness.ready is True
    assert readiness.blocking_findings == []


def test_misordered_publication_metadata_does_not_block():
    report = _report(["publication_metadata_interleaved_with_body"])
    readiness = evaluate_main_text_readiness(report, "Title", "metadata", has_body_content=True)
    assert readiness.ready is True


def test_figure_caption_misclassification_does_not_block():
    report = _report(["caption_promoted_to_heading", "empty_section", "excessive_tiny_sections"])
    readiness = evaluate_main_text_readiness(report, "Title", "metadata", has_body_content=True)
    assert readiness.ready is True
    assert len(readiness.exempted_findings) == 3


def test_title_structural_codes_do_not_block_when_title_otherwise_resolved():
    report = _report(["invalid_document_title", "root_title_not_rendered_as_h1", "abstract_promoted_to_h1"])
    readiness = evaluate_main_text_readiness(report, "Real Title", "metadata", has_body_content=True)
    assert readiness.ready is True


# ---------------------------------------------------------------------- #
# Retrieval-critical blockers (spec items 16-20)

def test_missing_source_identity_still_blocks():
    report = _report(["missing_source_block"])
    readiness = evaluate_main_text_readiness(report, "Title", "metadata", has_body_content=True)
    assert readiness.ready is False
    assert "missing_source_block" in {f.code for f in readiness.blocking_findings}


def test_duplicate_source_block_still_blocks():
    report = _report(["duplicate_source_block"])
    readiness = evaluate_main_text_readiness(report, "Title", "metadata", has_body_content=True)
    assert readiness.ready is False


def test_oversized_paragraph_still_blocks():
    report = _report(["oversized_paragraph"])
    readiness = evaluate_main_text_readiness(report, "Title", "metadata", has_body_content=True)
    assert readiness.ready is False


def test_chunk_exceeds_token_limit_still_blocks():
    report = _report(["chunk_exceeds_token_limit"])
    readiness = evaluate_main_text_readiness(report, "Title", "metadata", has_body_content=True)
    assert readiness.ready is False


def test_missing_title_blocks_even_with_no_errors():
    report = _report([])
    readiness = evaluate_main_text_readiness(report, None, "unresolved", has_body_content=True)
    assert readiness.ready is False
    assert any("title" in r for r in readiness.reasons)


def test_empty_main_body_blocks():
    report = _report([])
    readiness = evaluate_main_text_readiness(report, "Title", "metadata", has_body_content=False)
    assert readiness.ready is False


def test_zero_chunks_blocks():
    report = _report([], stats={"chunk_count": 0})
    readiness = evaluate_main_text_readiness(report, "Title", "metadata", has_body_content=True)
    assert readiness.ready is False


def test_empty_chunk_blocks():
    report = _report([], stats={"empty_chunk_count": 1})
    readiness = evaluate_main_text_readiness(report, "Title", "metadata", has_body_content=True)
    assert readiness.ready is False


def test_clean_document_is_ready():
    report = _report([])
    readiness = evaluate_main_text_readiness(report, "Title", "metadata", has_body_content=True)
    assert readiness.ready is True
    assert readiness.blocking_findings == []
    assert readiness.exempted_findings == []


# ---------------------------------------------------------------------- #
# apply_main_text_overlay mutates the report consistently

def test_overlay_updates_retrieval_ready_and_stats_without_hiding_total_errors():
    report = _report(["author_metadata_interleaved_with_body", "oversized_paragraph"])
    readiness = evaluate_main_text_readiness(report, "Title", "metadata", has_body_content=True)
    apply_main_text_overlay(report, readiness)
    assert report.retrieval_ready is False  # oversized_paragraph still blocks
    assert report.stats["retrieval_ready"] is False
    assert report.stats["main_text_blocking_error_count"] == 1
    assert report.stats["main_text_exempted_error_count"] == 1
    # Total finding count (validate()'s own accounting) is untouched -- the
    # overlay only changes what *gates* indexing, never removes or hides a
    # finding from the underlying log.
    assert len(report.findings.findings) == 2

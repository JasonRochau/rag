"""Universal quality-report and completion-state schema (spec sections 8 and 24)."""
from document_converter.md_parser import parse_markdown
from document_converter.normalize import normalize_document
from document_converter.chunker import MarkdownChunker
from document_converter.config import ChunkingConfig
from document_converter.validation import validate, completion_state
from document_converter.findings import FindingLog


def _validated(text, config=None):
    config = config or ChunkingConfig()
    doc, extract_log = parse_markdown(text, source_path="x.md")
    norm_log = normalize_document(doc)
    upstream = FindingLog()
    upstream.extend(extract_log)
    upstream.extend(norm_log)
    result = MarkdownChunker().chunk(doc, "x.md", config, document_id="d1")
    report = validate(doc, result, config, "x.md", "x_normalized.md", "markdown", False, upstream)
    return doc, result, report


REQUIRED_SECTION_24_KEYS = {
    "source_format", "page_count", "pages_accounted_for", "native_text_pages", "ocr_pages",
    "failed_pages", "source_blocks", "represented_source_blocks", "excluded_source_blocks",
    "duplicate_source_blocks", "heading_count", "heading_hierarchy_valid", "unique_header_paths",
    "paragraph_count", "long_paragraph_count", "oversized_paragraph_count", "list_count",
    "table_count", "code_block_count", "oversized_atomic_block_count", "chunk_count",
    "empty_chunk_count", "forced_fallback_split_count", "max_chunk_words", "max_chunk_tokens",
    "embedding_truncation_count", "structural_readiness", "retrieval_ready",
}


def test_stats_contains_every_universal_quality_report_field():
    _, _, report = _validated("# T\n\nParagraph one.\n\n## Sub\n\n- a\n- b\n")
    missing = REQUIRED_SECTION_24_KEYS - set(report.stats.keys())
    assert not missing, f"missing quality-report fields: {missing}"


def test_zero_counts_are_valid_for_absent_element_types():
    """A document with no tables/code/lists must report 0, not omit the field
    or error -- 'counts may legitimately be zero' (section 7/24)."""
    _, _, report = _validated("# T\n\nJust one short paragraph, nothing else.\n")
    assert report.stats["table_count"] == 0
    assert report.stats["code_block_count"] == 0
    assert report.stats["list_count"] == 0
    assert report.stats["retrieval_ready"] is True


def test_short_flat_document_is_structurally_valid_with_no_h2():
    """A short, genuinely flat document is not penalized for lacking H2s
    (spec section 2/9's 'no absolute heading-count requirement')."""
    _, _, report = _validated("# Maintenance Window Notice\n\nService will be briefly unavailable.\n")
    assert report.stats["heading_hierarchy_valid"] is True
    assert report.stats["structural_readiness"] is True
    assert report.stats["retrieval_ready"] is True


def test_source_block_coverage_counts_match_chunking_result():
    _, result, report = _validated("# T\n\nPara one.\n\nPara two.\n\nPara three.\n")
    assert report.stats["source_blocks"] == report.stats["represented_source_blocks"]
    assert report.stats["excluded_source_blocks"] == 0
    assert report.stats["duplicate_source_blocks"] == 0


def test_oversized_paragraph_is_force_split_and_no_longer_blocks():
    """Since the 2026-09-23 policy change an oversized paragraph is
    force-split with a WARNING in every profile (strict included) --
    `oversized_paragraph` is no longer an error code, so the document
    stays structurally ready."""
    text = "# T\n\n" + ("word " * 200)
    config = ChunkingConfig(hard_max_words=50, target_words=40, strict=True)
    _, result, report = _validated(text, config)
    assert report.stats["oversized_paragraph_count"] == 0
    assert any(f.code == "forced_fallback_split" for f in result.findings.findings)
    assert report.stats["structural_readiness"] is True
    assert report.stats["retrieval_ready"] is True


def test_completion_state_matches_section_8_schema():
    _, _, report = _validated("# T\n\nA clean short paragraph.\n")
    state = completion_state(report, indexed=True)
    expected_keys = {
        "processing_complete", "source_coverage_complete", "structural_readiness",
        "retrieval_ready", "blocking_errors", "forced_fallback_splits",
        "embedding_truncations", "indexed",
        # Section 14's structural-readiness gate additions (title/front-matter/
        # reading-order/heading-audit/list validity).
        "document_title_valid", "h1_count", "h1_is_first_meaningful_block",
        "front_matter_grouped", "reading_order_valid", "paragraph_continuity_valid",
        "high_confidence_headings_preserved", "heading_hierarchy_valid",
        "publication_metadata_interleaved_with_body", "author_metadata_interleaved_with_body",
        "unresolved_blocking_list_errors",
    }
    assert set(state.keys()) == expected_keys
    assert state["processing_complete"] is True
    assert state["retrieval_ready"] is True
    assert state["indexed"] is True
    assert state["blocking_errors"] == 0
    assert state["document_title_valid"] is True
    assert state["h1_count"] == 1
    assert state["front_matter_grouped"] is True
    assert state["reading_order_valid"] is True
    assert state["unresolved_blocking_list_errors"] == 0


def test_completion_state_indexed_false_for_preview():
    # Preview never reports indexed=True, even for a fully clean document.
    # (The old fixture used a strict-mode oversized paragraph to force
    # blocking errors; that error branch no longer exists -- oversized
    # paragraphs are force-split with a warning instead.)
    _, _, report = _validated("# T\n\nA clean short paragraph.\n")
    state = completion_state(report, indexed=False)
    assert state["indexed"] is False
    assert state["retrieval_ready"] is True
    assert state["blocking_errors"] == 0

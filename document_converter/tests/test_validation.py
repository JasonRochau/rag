from document_converter.md_parser import parse_markdown
from document_converter.normalize import normalize_document
from document_converter.chunker import MarkdownChunker
from document_converter.config import ChunkingConfig
from document_converter.validation import validate, find_html_remnants
from document_converter.findings import FindingLog


def _validated(text, config=None):
    config = config or ChunkingConfig()
    doc, extract_log = parse_markdown(text, source_path="x.md")
    norm_log = normalize_document(doc)
    upstream = FindingLog()
    upstream.extend(extract_log)
    upstream.extend(norm_log)
    result = MarkdownChunker().chunk(doc, "x.md", config, document_id="d1")
    return validate(doc, result, config, "x.md", "x_normalized.md", "markdown", False, upstream)


def test_html_remnant_detection_ignores_inline_code():
    assert find_html_remnants("plain <tag> text") == ["<tag>"]
    assert find_html_remnants("code `<tag>` is fine") == []


def test_retrieval_ready_true_for_clean_doc():
    report = _validated("# T\n\nA short clean paragraph.\n")
    assert report.retrieval_ready is True
    assert report.stats["errors"] == 0


def test_long_paragraph_flagged_as_warning():
    text = "# T\n\n" + ("word " * 200)
    report = _validated(text, ChunkingConfig(preferred_paragraph_words=50, hard_max_words=1000, target_words=900))
    assert any(f.code == "long_paragraph" for f in report.findings.findings)


def test_every_source_block_used_exactly_once():
    report = _validated("# T\n\nPara one.\n\nPara two.\n\nPara three.\n")
    codes = [f.code for f in report.findings.findings]
    assert "missing_source_block" not in codes
    assert "duplicate_source_block" not in codes


def test_html_remnants_outside_code_produce_warning():
    report = _validated("# T\n\nSome text with a <weird_tag> in it.\n")
    assert any(f.code == "remaining_html" for f in report.findings.findings)


def test_strict_oversized_paragraph_splits_and_stays_retrieval_ready():
    # Retired behavior: this exact fixture used to produce an
    # `oversized_paragraph` ERROR and a dropped block under strict mode. Per
    # the 2026-09-23 drop-warn-succeed policy the paragraph is force-split
    # with a WARNING instead, so nothing blocks indexing anymore.
    text = "# T\n\n" + ("word " * 200)
    config = ChunkingConfig(hard_max_words=50, target_words=40, strict=True)
    report = _validated(text, config)
    assert report.stats["errors"] == 0
    assert report.retrieval_ready is True

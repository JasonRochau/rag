"""
Manual title-override capability (document_converter/title_overrides.py +
pipeline.py's `_prepare_for_profile` wiring, orchestrator.py's
`/docs/{id}/title` endpoints, document_converter.html's inline [edit]
control).

Reproduces the real bug this feature exists for: an export tool's own
"Untitled 1" default document-title metadata survives `extractors.
clean_title`'s junk-title filter (its regex matches "Untitled"/"Untitled1"
but not "Untitled 1" -- a space before the digit) and becomes the resolved
MAIN_TEXT title verbatim. Verified against a real corpus PDF
(`Create, configure, and manage identities.pdf`, LibreOffice-exported
`/Title` = "Untitled 1") in the real-document acceptance run
(artifacts/title_override_validation/); this file covers the same
end-to-end behavior with a fast, offline DOCX fixture that reproduces the
identical metadata shape.
"""
import hashlib
import os

import pytest
from docx import Document as DocxDocument

from document_converter.index_schema import load_index
from document_converter.title_overrides import (
    clear_title_override, get_title_override, load_title_overrides, set_title_override,
)

BAD_AUTO_TITLE = "Untitled 1"
GOOD_TITLE = "Create, configure, and manage identities"


def _write_untitled_docx(path, heading="Introduction", body="Body content that should survive title resolution."):
    d = DocxDocument()
    d.core_properties.title = BAD_AUTO_TITLE
    d.add_heading(heading, level=1)
    d.add_paragraph(body)
    d.save(str(path))
    return path


# ------------------------------------------------------------------ #
# title_overrides.py -- the sidecar store, in isolation

def test_title_overrides_roundtrip_and_survives_reload(tmp_path):
    index_dir = str(tmp_path / "index")
    assert get_title_override(index_dir, "d-1") is None

    stored = set_title_override(index_dir, "d-1", "  " + GOOD_TITLE + "  ")
    assert stored == GOOD_TITLE  # whitespace trimmed
    assert get_title_override(index_dir, "d-1") == GOOD_TITLE

    # A fresh read from disk (not the same in-process dict) sees it too.
    assert load_title_overrides(index_dir) == {"d-1": GOOD_TITLE}

    assert clear_title_override(index_dir, "d-1") is True
    assert get_title_override(index_dir, "d-1") is None
    # Clearing something already absent is a reported no-op, not an error.
    assert clear_title_override(index_dir, "d-1") is False


def test_title_overrides_rejects_empty_title(tmp_path):
    with pytest.raises(ValueError):
        set_title_override(str(tmp_path / "index"), "d-1", "   ")


def test_title_overrides_sidecar_is_not_picked_up_as_an_index_file(tmp_path):
    """Regression: the sidecar must not land directly in `index_dir` as a
    top-level *.json file -- orchestrator.py's DocStore.reload() and
    eval_retrieval.py both glob `index_dir/*.json` and treat every match as
    a versioned per-document index payload; a sidecar there would be
    misreported as a corrupt/legacy index needing `cli.py rebuild`."""
    import glob
    index_dir = str(tmp_path / "index")
    set_title_override(index_dir, "d-1", GOOD_TITLE)
    assert glob.glob(os.path.join(index_dir, "*.json")) == []


# ------------------------------------------------------------------ #
# pipeline.py wiring -- acceptance tests 1-4, 6-7, 9

def test_automatic_resolution_reproduces_the_untitled_bug(pipeline, tmp_path):
    """Baseline/sanity: confirms the fixture actually reproduces the real
    defect (metadata title survives the junk filter) before testing the fix."""
    src = _write_untitled_docx(tmp_path / "doc.docx")
    result = pipeline.index(str(src))
    assert result.title == BAD_AUTO_TITLE
    assert result.title_source == "metadata"
    assert open(result.markdown_path, encoding="utf-8").readline().strip() == f"# {BAD_AUTO_TITLE}"


def test_manual_override_corrects_title_end_to_end(pipeline, tmp_path):
    """Acceptance 1, 3, 4: overriding "Untitled 1" -> the real title reaches
    MAIN_TEXT's H1 and the index/document metadata together."""
    src = _write_untitled_docx(tmp_path / "doc.docx")
    baseline = pipeline.index(str(src))
    assert baseline.title == BAD_AUTO_TITLE

    result = pipeline.set_title_override(str(src), GOOD_TITLE)

    assert result.title == GOOD_TITLE
    assert result.title_source == "manual_override"

    # MAIN_TEXT Markdown H1
    md = open(result.markdown_path, encoding="utf-8").read()
    assert md.splitlines()[0].strip() == f"# {GOOD_TITLE}"

    # Index/document metadata (what orchestrator.py's DocStore and the UI
    # listing/citations read from)
    payload = load_index(result.index_path)
    assert payload["document"]["title"] == GOOD_TITLE
    assert payload["document"]["title_source"] == "manual_override"
    assert payload["chunks"], "chunks must still be produced/embedded after an override"

    # Chunking wasn't silently skipped/degraded by the override
    assert len(payload["chunks"]) == len(baseline.chunking_result.chunks)


def test_override_does_not_touch_the_source_file(pipeline, tmp_path):
    """Acceptance 2: source filename/path/hash/mtime are untouched -- this
    is a title override, never a source-file rename."""
    src = _write_untitled_docx(tmp_path / "doc.docx")
    original_bytes = src.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    original_mtime = os.path.getmtime(src)
    original_path = str(src)

    baseline = pipeline.index(str(src))
    baseline_source_hash = load_index(baseline.index_path)["document"]["source_hash"]

    result = pipeline.set_title_override(str(src), GOOD_TITLE)

    assert src.read_bytes() == original_bytes
    assert hashlib.sha256(src.read_bytes()).hexdigest() == original_hash
    assert os.path.getmtime(src) == original_mtime
    assert result.source_path == original_path

    payload = load_index(result.index_path)
    assert payload["document"]["source_path"] == original_path
    assert payload["document"]["source_hash"] == baseline_source_hash


def test_reprocessing_preserves_the_override(pipeline, tmp_path):
    """Acceptance 6: the override survives reprocessing/reconversion of the
    same source, both via an explicit forced reprocess and via the ordinary
    idempotent "unchanged" re-index path."""
    src = _write_untitled_docx(tmp_path / "doc.docx")
    pipeline.index(str(src))
    result = pipeline.set_title_override(str(src), GOOD_TITLE)
    doc_id = result.document_id

    # Explicit reprocessing (e.g. re-running the pipeline over the same
    # source, as ingestion.py's rescan or a forced reconversion would).
    reprocessed = pipeline.index(str(src), document_id=doc_id, force=True)
    assert reprocessed.title == GOOD_TITLE
    assert reprocessed.title_source == "manual_override"

    # The ordinary "nothing changed, skip re-embedding" path also resolves
    # through the same override -- title resolution isn't only correct on
    # the write path that happens to run _prepare_for_profile explicitly.
    unchanged = pipeline.index(str(src), document_id=doc_id)
    assert unchanged.skipped_unchanged is True
    assert unchanged.title == GOOD_TITLE


def test_clearing_the_override_restores_automatic_resolution(pipeline, tmp_path):
    """Acceptance 7: removing the override falls back to normal automatic
    title resolution -- not a stuck/cached override, and not a permanently
    broken document."""
    src = _write_untitled_docx(tmp_path / "doc.docx")
    pipeline.index(str(src))
    overridden = pipeline.set_title_override(str(src), GOOD_TITLE)
    doc_id = overridden.document_id
    assert get_title_override(pipeline.config.paths.index_dir, doc_id) == GOOD_TITLE

    result = pipeline.clear_title_override(str(src), document_id=doc_id)

    assert get_title_override(pipeline.config.paths.index_dir, doc_id) is None
    assert result.title == BAD_AUTO_TITLE
    assert result.title_source == "metadata"
    md = open(result.markdown_path, encoding="utf-8").read()
    assert md.splitlines()[0].strip() == f"# {BAD_AUTO_TITLE}"
    payload = load_index(result.index_path)
    assert payload["document"]["title"] == BAD_AUTO_TITLE
    assert payload["document"]["title_source"] == "metadata"


def test_override_only_affects_the_targeted_document(pipeline, tmp_path):
    """Acceptance guardrail: "Do not alter unrelated documents." Overriding
    one document's title must not touch a second, unrelated document's
    resolved title or index file."""
    src_a = _write_untitled_docx(tmp_path / "a.docx", heading="Alpha Section")
    src_b = _write_untitled_docx(tmp_path / "b.docx", heading="Beta Section")
    result_a = pipeline.index(str(src_a))
    result_b = pipeline.index(str(src_b))
    b_index_mtime_before = os.path.getmtime(result_b.index_path)

    pipeline.set_title_override(str(src_a), GOOD_TITLE)

    payload_b = load_index(result_b.index_path)
    assert payload_b["document"]["title"] == BAD_AUTO_TITLE
    assert payload_b["document"]["title_source"] == "metadata"
    assert os.path.getmtime(result_b.index_path) == b_index_mtime_before

    payload_a = load_index(result_a.index_path)
    assert payload_a["document"]["title"] == GOOD_TITLE

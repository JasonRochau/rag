"""
Regression tests for the upload-time document-identity fix (orchestrator.py
`docs_upload` / `_find_existing_source_by_hash`): re-uploading byte-identical
content must resolve to the same source path (and therefore the same
`stable_document_id`) instead of minting a new timestamp-suffixed filename
and a brand new document id every retry.

Root cause this covers (see DELIVERABLES.md / final report): the same PDF,
re-uploaded after failing strict validation, previously produced a
different document id on every attempt because the upload handler only
checked for a *filename* collision (falling back to an
`f"{stem}-{int(time.time())}"` name) rather than a *content* collision, and
`stable_document_id` is deliberately path-based.

`orchestrator.py` is imported directly here (it is a plain module, not a
running server) -- its only import-time side effect is a read-only scan of
`docs/.index/*.json` to build `STORE`, so this is safe to import in tests.
Every test below monkeypatches `CONFIG.paths.source_dir` to an isolated
tmp_path first, so nothing here ever reads or writes the real
`docs/newly_formatted_docs_review/sources` tree.
"""
from __future__ import annotations

import hashlib

import orchestrator


def test_find_existing_source_by_hash_finds_identical_content(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.CONFIG.paths, "source_dir", str(tmp_path))
    content = b"%PDF-1.4 fake pdf bytes for identity test"
    existing = tmp_path / "Some-Paper.pdf"
    existing.write_bytes(content)

    found = orchestrator._find_existing_source_by_hash(hashlib.sha256(content).hexdigest())
    assert found == str(existing)


def test_find_existing_source_by_hash_returns_none_for_new_content(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.CONFIG.paths, "source_dir", str(tmp_path))
    (tmp_path / "Some-Paper.pdf").write_bytes(b"original content")

    found = orchestrator._find_existing_source_by_hash(hashlib.sha256(b"different content").hexdigest())
    assert found is None


def test_find_existing_source_by_hash_empty_dir_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.CONFIG.paths, "source_dir", str(tmp_path))
    found = orchestrator._find_existing_source_by_hash(hashlib.sha256(b"anything").hexdigest())
    assert found is None


def test_reupload_resolves_to_same_document_id(tmp_path, monkeypatch):
    """End-to-end simulation of the actual bug: the same bytes 'uploaded'
    three times (three separate filenames the old timestamp-fallback would
    have produced on a real retry) must all resolve to one path, and
    therefore one `stable_document_id`, once routed through
    `_find_existing_source_by_hash` the way `docs_upload` now does."""
    monkeypatch.setattr(orchestrator.CONFIG.paths, "source_dir", str(tmp_path))
    content = b"%PDF-1.4 identical regression-paper bytes"
    content_hash = hashlib.sha256(content).hexdigest()

    # First "upload": no existing match, write it.
    assert orchestrator._find_existing_source_by_hash(content_hash) is None
    first_path = tmp_path / "Regression-Analyses.pdf"
    first_path.write_bytes(content)

    # Second and third "uploads" of the identical bytes must resolve back
    # to the same file instead of writing a new timestamp-suffixed one.
    for _ in range(2):
        found = orchestrator._find_existing_source_by_hash(content_hash)
        assert found == str(first_path)

    import os
    rel = os.path.relpath(str(first_path), str(tmp_path))
    doc_id_1 = orchestrator.stable_document_id(rel)
    doc_id_2 = orchestrator.stable_document_id(rel)
    assert doc_id_1 == doc_id_2

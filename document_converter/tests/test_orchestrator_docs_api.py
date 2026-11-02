"""
End-to-end tests for the orchestrator.py document-conversion/diagnostics API
that document_converter.html's "View Details" panel and summary cards
consume (see DELIVERABLES.md's document-processing-diagnostics task).

`orchestrator.py` is imported directly (same convention as
test_upload_identity.py) rather than run as a subprocess -- its only
import-time side effect is a read-only scan of the real `docs/.index/*.json`
tree to build the module-level `STORE`. Every test here monkeypatches
`orchestrator.CONFIG.paths` to an isolated `tmp_path` *before* touching any
endpoint that writes to disk, and the real `STORE.documents`/`STORE.chunks`
snapshot is saved/restored around each test, so nothing here ever reads or
writes the real `docs/` tree or interferes with a live orchestrator.py
process that might already be running against it.
"""
from __future__ import annotations

import io
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

import fitz

import orchestrator
import document_converter.split_ingest as split_ingest_mod
from document_converter.tests.conftest import FakeEmbeddingClient


def _post_and_wait(client, path, files, **kwargs):
    """`/docs/upload` and `/docs/preview` now return 202 with a job id
    immediately instead of blocking for the whole ingest (see PART 3 of the
    ingestion-observability task) -- this POSTs, polls GET /docs/jobs/{id}
    to a terminal state, and returns an object shaped like the old
    synchronous response (`.status_code`/`.json()`-compatible), so every
    existing test below only had to change its call site, not its
    assertions. Raises on a timeout rather than hanging a test forever if a
    job is stuck."""
    r = client.post(path, files=files, **kwargs)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        jr = client.get(f"/docs/jobs/{job_id}")
        assert jr.status_code == 200
        job = jr.json()
        if job["state"] == "succeeded":
            return _FakeResponse(200, job["result"])
        if job["state"] == "failed":
            # `job["validation"]` already carries the exact 422 payload shape
            # (error/document_id/validation/completion_state/docs) for a
            # PipelineValidationError; a non-validation failure only has
            # `job["error"]`.
            body = job["validation"] if job["validation"] is not None else {"error": job["error"]}
            return _FakeResponse(422 if job["validation"] is not None else 400, body)
        time.sleep(0.05)
    raise AssertionError(f"job {job_id!r} did not reach a terminal state within 10s (last state: {job})")


def _upload_and_wait(client, files, **kwargs):
    return _post_and_wait(client, "/docs/upload", files, **kwargs)


def _preview_and_wait(client, files, **kwargs):
    return _post_and_wait(client, "/docs/preview", files, **kwargs)


class _FakeResponse:
    """Minimal stand-in for `httpx.Response`, just enough for the assertions
    below (`.status_code`, `.json()`)."""

    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body

    def json(self) -> dict:
        return self._body


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.CONFIG.paths, "source_dir", str(tmp_path / "sources"))
    monkeypatch.setattr(orchestrator.CONFIG.paths, "processed_dir", str(tmp_path / "processed"))
    monkeypatch.setattr(orchestrator.CONFIG.paths, "report_dir", str(tmp_path / "reports"))
    monkeypatch.setattr(orchestrator.CONFIG.paths, "index_dir", str(tmp_path / "index"))
    orchestrator.CONFIG.paths.ensure()
    monkeypatch.setattr(orchestrator.PIPELINE, "_embedding_client", FakeEmbeddingClient())

    saved_documents, saved_chunks = orchestrator.STORE.documents, orchestrator.STORE.chunks
    orchestrator.STORE.documents, orchestrator.STORE.chunks = {}, []
    try:
        yield TestClient(orchestrator.app)
    finally:
        orchestrator.STORE.documents, orchestrator.STORE.chunks = saved_documents, saved_chunks


def _md_file(text: str, name: str = "doc.md"):
    return {"file": (name, io.BytesIO(text.encode("utf-8")), "text/markdown")}


CLEAN_DOC = "# A Clean Document\n\nA short, unremarkable paragraph.\n"
# Default ChunkingConfig.preferred_paragraph_words is 150 and hard_max_words
# is 512 -- 200 words trips the WARNING-only `long_paragraph` finding
# without tripping any ERROR, so this document must remain retrieval_ready.
WARNING_ONLY_DOC = "# Warns But Fine\n\n" + ("word " * 200) + "\n"


def test_docs_preview_returns_title_source_and_diagnostics(client):
    r = _preview_and_wait(client, _md_file(CLEAN_DOC))
    assert r.status_code == 200
    d = r.json()
    assert d["title"] == "A Clean Document"
    assert "title_source" in d
    assert d["source_format"] == "markdown"
    assert d["ocr_used"] is False
    assert d["validation"]["stats"]["page_count"] is not None or "heading_count" in d["validation"]["stats"]
    assert d["validation"]["stats"]["heading_count"] >= 1
    assert isinstance(d["validation"]["findings"], list)
    assert d["completion_state"]["indexed"] is False


def test_docs_preview_does_not_write_to_disk_or_index(client, tmp_path):
    _preview_and_wait(client, _md_file(CLEAN_DOC))
    import os
    assert os.listdir(orchestrator.CONFIG.paths.source_dir) == []
    assert os.listdir(orchestrator.CONFIG.paths.index_dir) == []


def test_docs_upload_success_reports_summary_and_stages(client):
    r = _upload_and_wait(client, _md_file(CLEAN_DOC))
    assert r.status_code == 200
    d = r.json()
    assert d["title"] == "A Clean Document"
    assert d["title_source"]
    assert d["chunks"] >= 1
    assert d["stages"]["indexed"] is True
    assert d["stages"]["embedded"] is True
    assert d["validation"]["retrieval_ready"] is True
    assert d["completion_state"]["indexed"] is True
    assert any(doc["name"] == "A Clean Document" for doc in d["docs"])


def test_docs_list_stays_lean_no_findings_or_stats_leak_into_row(client):
    """The KB table's normal row must only show counts (warnings/errors),
    never the full findings/stats payload -- that belongs behind View
    Details (GET /docs/{id}) only."""
    _upload_and_wait(client, _md_file(CLEAN_DOC))
    r = client.get("/docs")
    assert r.status_code == 200
    docs = r.json()["docs"]
    assert len(docs) == 1
    row = docs[0]
    assert "warnings" in row and "errors" in row
    assert "findings" not in row
    assert "stats" not in row


def test_docs_detail_exposes_findings_and_stats_for_view_details(client):
    upload = _upload_and_wait(client, _md_file(WARNING_ONLY_DOC))
    doc_id = upload.json()["document_id"]

    r = client.get(f"/docs/{doc_id}")
    assert r.status_code == 200
    payload = r.json()
    assert payload["validation"]["warnings"] >= 1
    assert payload["validation"]["errors"] == 0
    codes = [f["code"] for f in payload["validation"]["findings"]]
    assert "long_paragraph" in codes
    assert payload["validation"]["stats"]["heading_count"] >= 1


def test_warning_only_document_is_not_reported_as_failed(client):
    """An ERROR (or here, even a WARNING) must never be reinterpreted as
    "document failed" independent of what the backend's own
    retrieval_ready/completion_state say."""
    r = _upload_and_wait(client, _md_file(WARNING_ONLY_DOC))
    d = r.json()
    assert d["validation"]["retrieval_ready"] is True
    assert d["completion_state"]["retrieval_ready"] is True
    assert d["completion_state"]["indexed"] is True


def test_docs_detail_404_for_unknown_document(client):
    r = client.get("/docs/d-doesnotexist")
    assert r.status_code == 404
    assert "error" in r.json()


def test_title_edit_still_works_and_refreshes_persisted_diagnostics(client):
    upload = _upload_and_wait(client, _md_file(CLEAN_DOC))
    doc_id = upload.json()["document_id"]

    r = client.put(f"/docs/{doc_id}/title", json={"title": "Renamed Title"})
    assert r.status_code == 200
    d = r.json()
    assert d["title"] == "Renamed Title"
    assert d["title_source"] == "manual_override"
    assert any(doc["id"] == doc_id and doc["name"] == "Renamed Title" and doc["has_title_override"]
               for doc in d["docs"])

    detail = client.get(f"/docs/{doc_id}").json()
    assert detail["document"]["title"] == "Renamed Title"
    assert detail["document"]["title_source"] == "manual_override"
    # Reprocessing on override still refreshes the persisted diagnostics
    # block (not left stale from the original upload).
    assert "stats" in detail["validation"]

    clear = client.delete(f"/docs/{doc_id}/title")
    assert clear.status_code == 200
    assert clear.json()["title_source"] != "manual_override"


def test_zero_findings_document_has_empty_findings_list(client):
    r = _upload_and_wait(client, _md_file(CLEAN_DOC))
    d = r.json()
    doc_id = d["document_id"]
    detail = client.get(f"/docs/{doc_id}").json()
    # A clean document has zero findings -- the API must report an empty
    # list, not omit the key or fabricate a finding.
    assert detail["validation"]["findings"] == []


# ---------------------------------------------------------------------- #
# Query-time GPU lifecycle (RAG_BUILD_SOURCE_OF_TRUTH.md, "strict
# sequential GPU model lifecycle" phase): /docs/search must explicitly
# release the embedding model's GPU memory once the query has been
# embedded and retrieval is complete, so a subsequent client-side
# generation call (index.html -> Ollama directly) never has to contend
# with it for VRAM.

def test_docs_search_releases_embedding_model_after_query_embedding(client):
    r = _upload_and_wait(client, _md_file(CLEAN_DOC))
    assert r.status_code == 200

    # Ingestion itself already releases at the end of its own embedding
    # stage (see test_embedding_client_released_at_end_of_ingestion) --
    # baseline after upload, then assert the search call adds exactly one
    # more release.
    fake_client = orchestrator.PIPELINE._embedding_client
    before = fake_client.release_calls

    r = client.post("/docs/search", json={"query": "unremarkable paragraph", "k": 4})
    assert r.status_code == 200
    assert r.json()["hits"]

    assert fake_client.release_calls == before + 1


def test_docs_search_releases_embedding_model_even_with_no_hits(client):
    # release() must not be conditioned on finding a match -- it belongs to
    # "the query was embedded", not "retrieval succeeded".
    r = _upload_and_wait(client, _md_file(CLEAN_DOC))
    assert r.status_code == 200
    fake_client = orchestrator.PIPELINE._embedding_client
    before = fake_client.release_calls

    r = client.post("/docs/search", json={"query": "completely unrelated gibberish zzy", "k": 4})
    assert r.status_code == 200
    assert fake_client.release_calls == before + 1


# ---------------------------------------------------------------------- #
# Non-blocking ingestion (PART 3 of the ingestion-observability task):
# `/docs/upload` used to be an `async def` handler calling `PIPELINE.index()`
# synchronously -- a blocking call inside an async handler blocks the whole
# event loop, so `/health` (and chat/search) stopped responding for the
# entire ingest. This is the regression lock on that fix.

def test_health_stays_responsive_while_upload_job_runs(client, monkeypatch):
    started = threading.Event()
    original_index = orchestrator.PIPELINE.index

    def slow_index(*a, **kw):
        started.set()
        time.sleep(1.5)
        return original_index(*a, **kw)

    monkeypatch.setattr(orchestrator.PIPELINE, "index", slow_index)

    r = client.post("/docs/upload", files=_md_file(CLEAN_DOC))
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    assert started.wait(timeout=5), "background ingest job never started running"

    start = time.monotonic()
    hr = client.get("/health")
    elapsed = time.monotonic() - start
    assert hr.status_code == 200
    assert elapsed < 1.0, f"/health took {elapsed:.2f}s while an ingest job was running"

    # Drain the job so it doesn't leak (mid-flight) into a later test.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if client.get(f"/docs/jobs/{job_id}").json()["state"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)


def test_docs_upload_returns_202_with_job_id_immediately(client):
    r = client.post("/docs/upload", files=_md_file(CLEAN_DOC))
    assert r.status_code == 202
    d = r.json()
    assert d["job_id"]
    assert d["filename"] == "doc.md"
    assert d["document_id"].startswith("d-")

    # Drain the job before this test (and its `client` fixture's monkeypatch
    # of CONFIG.paths) tears down -- otherwise the background worker thread
    # outlives the tmp_path redirection and writes its processed/report/
    # index files under whatever CONFIG.paths reverts to, which is the real
    # docs/ tree outside tests.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if client.get(f"/docs/jobs/{d['job_id']}").json()["state"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)


def test_docs_job_get_returns_404_for_unknown_job(client):
    r = client.get("/docs/jobs/j-doesnotexist")
    assert r.status_code == 404


def test_docs_jobs_list_route_is_not_shadowed_by_doc_id_route(client):
    """Regression lock: `/docs/{doc_id}` is registered elsewhere in this
    router too, and FastAPI matches routes in registration order -- an
    untyped single-segment path parameter would otherwise greedily swallow
    a literal `/docs/jobs` request (doc_id="jobs") before it ever reached
    the jobs-list route."""
    r = _upload_and_wait(client, _md_file(CLEAN_DOC))
    assert r.status_code == 200

    r = client.get("/docs/jobs")
    assert r.status_code == 200
    d = r.json()
    assert isinstance(d["jobs"], list)
    assert any(j["kind"] == "index" for j in d["jobs"])


def test_docs_job_stream_emits_done_event_with_result(client):
    r = client.post("/docs/preview", files=_md_file(CLEAN_DOC))
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    events = []
    with client.stream("GET", f"/docs/jobs/{job_id}/stream") as resp:
        assert resp.status_code == 200
        current_event = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current_event:
                events.append((current_event, line.split(":", 1)[1].strip()))
                current_event = None

    kinds = [e for e, _ in events]
    assert "done" in kinds
    done_payload = json.loads(events[kinds.index("done")][1])
    assert done_payload["result"]["title"] == "A Clean Document"


# ===== SPLIT-INGEST THRESHOLD (H6) =====

def _build_multi_section_pdf(path, sections=6, pages_per_section=10):
    """Same fitz+set_toc fixture pattern as test_split_ingest.py: one
    level-1 outline entry per section, so split_pdf resolves boundaries
    from the outline tree alone. Sized so the whole document exceeds a
    small part target while each individual section fits under it."""
    d = fitz.open()
    toc = []
    page = 0
    for sec in range(sections):
        toc.append([1, f"Section {sec + 1}", page + 1])
        for _ in range(pages_per_section):
            pg = d.new_page()
            pg.insert_text((72, 60), f"Section {sec + 1}", fontsize=18)
            for i in range(20):
                pg.insert_text((72, 95 + i * 16), f"Body line {i} section {sec + 1}.", fontsize=11)
            page += 1
    d.set_toc(toc)
    d.save(str(path))
    d.close()


def _pdf_upload(path):
    return {"file": (path.name, path.read_bytes(), "application/pdf")}


def test_oversized_pdf_goes_through_split_ingest(client, tmp_path, monkeypatch):
    """Threshold of 1 byte forces every PDF down the split path; a 20 KB
    target_bytes sits between one section (~7 KB assembled) and the whole
    fixture (~41 KB assembled), yielding 3 parts (orchestrator imports
    split_and_ingest lazily inside _run_index_job, so patching the module
    attribute is seen by the job thread)."""
    monkeypatch.setattr(orchestrator.CONFIG, "split_ingest_threshold_bytes", 1)
    real_split_and_ingest = split_ingest_mod.split_and_ingest

    def small_target_split(pipeline, source_path, **kwargs):
        kwargs.setdefault("target_bytes", 20_000)
        return real_split_and_ingest(pipeline, source_path, **kwargs)

    monkeypatch.setattr(split_ingest_mod, "split_and_ingest", small_target_split)

    pdf = tmp_path / "big.pdf"
    _build_multi_section_pdf(pdf)
    r = _upload_and_wait(client, _pdf_upload(pdf))
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["split"] is True
    assert len(body["parts"]) > 1
    assert all(p["status"] == "indexed" for p in body["parts"])
    assert len(body["docs"]) == len(body["parts"])


def test_pdf_under_threshold_takes_normal_single_document_path(client, tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.CONFIG, "split_ingest_threshold_bytes", 100 * 1024 * 1024)
    pdf = tmp_path / "small.pdf"
    _build_multi_section_pdf(pdf, sections=2, pages_per_section=1)
    r = _upload_and_wait(client, _pdf_upload(pdf))
    assert r.status_code == 200, r.json()
    body = r.json()
    assert "split" not in body
    assert body["document_id"]
    assert body["stages"]["indexed"] is True

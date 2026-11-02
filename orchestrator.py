"""
LOCAL AI ASSISTANT - ORCHESTRATOR
FastAPI server routing, state management, document/search endpoints.
Delegates document processing to document_converter.DocumentPipeline,
ranking to control.py, and legacy markdown rescans to ingestion.py.
"""
import glob
import hashlib
import os
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import logging
from typing import Callable, List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from ingestion import DocIngester
from control import HybridRanker
from document_converter.config import load_config
from document_converter.pipeline import DocumentPipeline, PipelineValidationError, stable_document_id
from document_converter.extractors import SUPPORTED_EXTENSIONS
from document_converter.index_schema import load_index, is_legacy_schema, sha256_hex, check_embedding_compatibility
from document_converter.validation import completion_state
from document_converter import gpu_lifecycle
import job_runner

# Strict sequential GPU model lifecycle (RAG_BUILD_SOURCE_OF_TRUTH.md,
# "strict sequential GPU model lifecycle" phase): one process-wide lock
# around every GPU-heavy stage this orchestrator initiates directly --
# ingestion (VLM + embedding, inside `PIPELINE.index()`) and query-time
# embedding/retrieval (inside `DocStore.search()`). FastAPI/Starlette runs
# ordinary `def` endpoints in a thread pool, so two concurrent requests
# (e.g. a document upload and a search) could otherwise both reach the
# embedding service/Ollama at once and recreate the very VRAM contention
# this phase exists to remove; this keeps them from interleaving. Generation
# (`/chat/stream`) also runs in this process now, but deliberately outside
# this lock -- it only needs the embedding model to have already released
# VRAM (enforced by `DocStore.search()`'s own barrier, still taken under this
# lock), not the lock itself, and holding it for the full duration of a
# multi-minute generation would needlessly serialize an unrelated search or
# upload behind it. Scoped to this process only -- an accepted limitation
# for this single-user local desktop app rather than a full cross-process
# scheduler.
GPU_STAGE_LOCK = threading.Lock()

# Job registry (observability for long-running ingestion, see PART 3 of the
# ingestion-observability task): /docs/upload and /docs/preview hand a job
# off to a background thread and return immediately instead of blocking the
# event loop for the entire extract->chunk->embed run -- see
# `_run_index_job`/`_run_preview_job` near those endpoints below. Bounded to
# the 50 most recent jobs (oldest *terminal* job evicted first -- a queued
# or running job is never evicted) so this can never grow unbounded across a
# long-lived process. Plain dict, not keyed by anything else, so insertion
# order doubles as "oldest first" for eviction.
JOBS: dict = {}
JOBS_LOCK = threading.Lock()
MAX_JOBS = 50

# Remaining-time estimation (replaces a bare elapsed-seconds status line with
# a real ETA -- see `_job_progress_cb`/`_progress_tick` below). Every stage
# already reports (stage, current, total) through that one callback -- see
# pdf_extractor.py's three labeled page passes ("extracting",
# "scanning_images", "detecting_tables", each a full page sweep) and
# pipeline.py's per-stage `_emit` calls (each block-iterating stage reports
# against the document's own block count). This turns that stream into one
# work-unit total per job:
#   - the three extractor passes each sweep every page once, so together
#     they cost 3x page_count, not page_count once.
#   - each block-iterating pipeline stage below costs one block_count
#     (unknown until extraction finishes -- the total is recomputed the
#     first time a block-iterating stage reports its own total, and that
#     total is carried through the rest of the run).
#   - "structuring" (semantic_structuring.py's single Ollama call) has no
#     knowable denominator -- it is listed here only for ordering, excluded
#     from `_BLOCK_STAGES`, and any job that has passed through it has its
#     estimate flagged as a lower bound from then on.
_EXTRACTION_STAGE_ORDER = ("extracting", "scanning_images", "detecting_tables")
_PIPELINE_STAGE_ORDER = (
    "normalizing", "repairing_running_headers", "structuring", "repairing_headings",
    "auditing_heading_siblings", "auditing_section_numbering", "repairing_paragraph_continuity",
    "chunking", "validating", "rendering",
)
_BLOCK_STAGES = tuple(s for s in _PIPELINE_STAGE_ORDER if s != "structuring")
_UNESTIMABLE_STAGES = ("structuring",)
# Trailing window (in seconds -- a unit already reported in the status line's
# elapsed-time display) that the units-completed/elapsed rate is averaged
# over, so a slow stage's rate isn't masked by a fast earlier one.
_RATE_WINDOW_SECONDS = 60.0

# Per-job (timestamp, overall_units_done) samples for the rate above. Kept
# out of `JOBS` itself (that dict is returned verbatim to API callers) and
# evicted alongside its job in `_evict_terminal_jobs_locked`.
_JOB_PROGRESS_HISTORY: dict = {}

# ===== CHAT MODEL (llama.cpp llama-server) =====
# The single chat model behind the dropdown, session summarization, and the
# Job tab's build agent. Served by llama-server's OpenAI-compatible
# /v1/chat/completions endpoint, launched on demand with a fixed 32768-token
# context and --jinja. The embedding model (CONFIG.embedding, a separate
# sentence-transformers service) and the vision model (CONFIG.visual, via
# Ollama) are part of the RAG pipeline, are not chat options, and never
# appear in /api/models.
CHAT_MODEL_ID = "qwen3.6-35b-a3b"
CHAT_MODEL_NAME = "Qwen3.6-35B-A3B"
CHAT_MODEL_PATH = os.environ.get(
    "CHAT_MODEL_PATH",
    "/mnt/hdd1/ai-models/llm/shennguyen/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-GGUF/"
    "huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated-q4_k_m.gguf")
CHAT_LLM_HOST = os.environ.get("CHAT_LLM_HOST", "http://127.0.0.1:8080")
CHAT_LLM_PORT = int(os.environ.get("CHAT_LLM_PORT", "8080"))
CHAT_NUM_CTX = 32768
LLAMA_SERVER_BIN = os.environ.get(
    "LLAMA_SERVER_BIN", os.path.expanduser("~/llama.cpp/build/bin/llama-server"))
# A 21 GB GGUF paged from disk can take minutes to load -- startup waits
# this long for /health before giving up.
CHAT_SERVER_START_TIMEOUT_S = 900
_chat_server_proc = None
_chat_server_lock = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("ORCH_PORT", "11436"))
STATE_PATH = os.path.join(BASE_DIR, "state.json")
SESSIONS = os.path.join(BASE_DIR, "sessions")
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")
THREADS = os.path.join(BASE_DIR, "threads")

DEFAULT_STATE = {
    "goal": "",
    "current_task": "",
    "constraints": [],
    "completed": [],
    "open_tasks": [],
    "design_decisions": [],
    "known_bugs": [],
    "files": [],
    "questions": [],
    "last_summary": "",
    "updated_at": "",
}

for d in (SESSIONS, ARTIFACTS, THREADS):
    os.makedirs(d, exist_ok=True)

# ===== LOCAL AGENT: user-granted directories =====
# The security boundary for the localAgent tool (see /localagent endpoints
# near _sse()). Unlike the old inbox/managed-grant model, directories here
# come ONLY from user input via POST /localagent/directories -- there is no
# hardcoded path and the running agent can never name a new path that
# becomes granted, it can only operate within one already added by the
# user. Persisted to LOCAL_AGENT_DIRS_PATH so the list and each directory's
# read/write/execute toggles survive a restart.
LOCAL_AGENT_DIRS_PATH = os.path.join(BASE_DIR, "local_agent_dirs.json")
LOCAL_AGENT_DIRS: dict = {}  # id -> {"id","path","read","write","execute","created_at"}


def _load_local_agent_dirs() -> None:
    global LOCAL_AGENT_DIRS
    try:
        with open(LOCAL_AGENT_DIRS_PATH, "r", encoding="utf-8") as f:
            rows = json.load(f)
        LOCAL_AGENT_DIRS = {r["id"]: r for r in rows}
    except (FileNotFoundError, json.JSONDecodeError):
        LOCAL_AGENT_DIRS = {}


def _save_local_agent_dirs() -> None:
    with open(LOCAL_AGENT_DIRS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(LOCAL_AGENT_DIRS.values()), f, indent=2, ensure_ascii=False)


_load_local_agent_dirs()

CONFIG = load_config()
PIPELINE = DocumentPipeline(CONFIG)


def load_state() -> dict:
    """Load session state."""
    if not os.path.exists(STATE_PATH):
        save_state(dict(DEFAULT_STATE))
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        s = json.load(f)
    for k, v in DEFAULT_STATE.items():
        s.setdefault(k, v if not isinstance(v, list) else list(v))
    return s


def save_state(state: dict) -> dict:
    """Save session state."""
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    return state


# ===== THREAD STORE =====
# One JSON file per conversation thread under THREADS/, mirroring the
# state.json/sessions/ pattern above. Unlike `messages` in the old
# browser-only chat (a disposable in-memory array), a thread's messages --
# including each assistant message's retrieved sources -- are durable, so
# the sidebar's history list and the "click an older message to see its
# sources" interaction both survive a reload (spec section 2.3).

def _thread_path(thread_id: str) -> str:
    return os.path.join(THREADS, f"{thread_id}.json")


def new_thread_id() -> str:
    return "t-" + uuid.uuid4().hex[:16]


def new_message_id() -> str:
    return "m-" + uuid.uuid4().hex[:16]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def create_thread(title: str = "") -> dict:
    now = _now()
    thread = {
        "id": new_thread_id(),
        "title": title or "New chat",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    save_thread(thread)
    return thread


def load_thread(thread_id: str) -> Optional[dict]:
    path = _thread_path(thread_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_thread(thread: dict) -> dict:
    thread["updated_at"] = _now()
    with open(_thread_path(thread["id"]), "w", encoding="utf-8") as f:
        json.dump(thread, f, indent=2, ensure_ascii=False)
    return thread


def delete_thread(thread_id: str) -> bool:
    path = _thread_path(thread_id)
    if not os.path.exists(path):
        return False
    os.remove(path)
    return True


def list_threads() -> list:
    """Summaries only (no messages/sources) -- a small personal corpus of
    threads, so a directory scan per call (like `_find_existing_source_by_hash`
    above) is simple and always correct rather than maintaining a separate
    index file that could drift from the per-thread files it summarizes."""
    out = []
    for name in os.listdir(THREADS):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(THREADS, name), "r", encoding="utf-8") as f:
                t = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "id": t.get("id"),
            "title": t.get("title") or "New chat",
            "created_at": t.get("created_at"),
            "updated_at": t.get("updated_at"),
            "message_count": len(t.get("messages") or []),
        })
    out.sort(key=lambda t: t.get("updated_at") or "", reverse=True)
    return out


def chat_server_health() -> bool:
    """True when llama-server answers /health with the model fully loaded
    (llama-server returns 503 while the model is still loading)."""
    try:
        r = requests.get(CHAT_LLM_HOST + "/health", timeout=5)
        return r.ok
    except Exception:
        return False


def ensure_chat_server() -> "tuple[bool, Optional[str]]":
    """Make sure the chat model is being served. Reuses an already-running
    healthy llama-server on CHAT_LLM_HOST; otherwise launches one with the
    configured GGUF at the fixed context (--jinja enables the model's chat
    template) and waits for it to finish loading. Returns (ok, error)."""
    global _chat_server_proc
    if chat_server_health():
        return True, None
    with _chat_server_lock:
        if chat_server_health():
            return True, None
        if not os.path.exists(LLAMA_SERVER_BIN):
            return False, f"llama-server binary not found at {LLAMA_SERVER_BIN}"
        if not os.path.exists(CHAT_MODEL_PATH):
            return False, f"chat model file not found at {CHAT_MODEL_PATH}"
        logger.info(f"[chat] starting llama-server for {CHAT_MODEL_NAME} on port {CHAT_LLM_PORT}")
        log_path = os.path.join(BASE_DIR, "llama_server.log")
        log_file = open(log_path, "ab")
        _chat_server_proc = subprocess.Popen(
            [LLAMA_SERVER_BIN, "-m", CHAT_MODEL_PATH,
             "--host", "127.0.0.1", "--port", str(CHAT_LLM_PORT),
             "-c", str(CHAT_NUM_CTX), "--jinja"],
            stdout=log_file, stderr=subprocess.STDOUT, cwd=BASE_DIR,
            start_new_session=True)
        deadline = time.time() + CHAT_SERVER_START_TIMEOUT_S
        while time.time() < deadline:
            if _chat_server_proc.poll() is not None:
                return False, ("llama-server exited during startup "
                               f"(code {_chat_server_proc.returncode}) -- see llama_server.log")
            if chat_server_health():
                logger.info("[chat] llama-server is ready")
                return True, None
            time.sleep(3)
        return False, f"timed out after {CHAT_SERVER_START_TIMEOUT_S}s waiting for llama-server to load the model"


def chat_completion(messages, want_json=False, temperature=0.2, max_tokens: Optional[int] = None,
                    timeout: float = 1800.0) -> str:
    """Non-streaming chat completion against the llama-server serving the
    chat model (session summarization and the Job tab's agent loop; the
    interactive chat path streams instead -- see /chat/stream)."""
    ok, err = ensure_chat_server()
    if not ok:
        raise RuntimeError(f"chat model unavailable: {err}")
    body = {
        "model": CHAT_MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    if want_json:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(CHAT_LLM_HOST + "/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    return ((choices[0].get("message") or {}).get("content", "") if choices else "") or ""


def safe_json(raw: str) -> dict:
    """Safely parse JSON from string."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        a, b = raw.find("{"), raw.rfind("}")
        if a != -1 and b != -1 and b > a:
            try:
                return json.loads(raw[a:b + 1])
            except Exception:
                pass
    return {}


def _merge_list(existing: list, additions, cap=200):
    """Merge additions into existing list, removing duplicates."""
    seen = {str(x).strip().lower() for x in existing}
    for item in additions or []:
        item = str(item).strip()
        if item and item.lower() not in seen:
            existing.append(item)
            seen.add(item.lower())
    return existing[-cap:]


# Matches `stable_document_id`'s output shape (document_converter/
# pipeline.py: "d-" + 16 hex chars) but intentionally a bit looser
# ([A-Za-z0-9]+ rather than exactly 16 hex chars) so a legitimate id never
# 400s just because the id-minting scheme changes length/alphabet later.
# The real security boundary here is "no path-traversal/separator
# characters can survive this check", not exact format equality.
DOC_ID_RE = re.compile(r"^d-[A-Za-z0-9]+$")


def _validate_doc_id(doc_id: str) -> Optional[JSONResponse]:
    """First line of defense for any route that turns `doc_id` into a path
    component: rejects anything that isn't a plausible document id shape
    (see DOC_ID_RE) before it ever reaches a filesystem call, so an encoded
    `..` segment in the URL path never gets as far as `os.path.join`."""
    if not DOC_ID_RE.match(doc_id or ""):
        return JSONResponse(status_code=400, content={"error": f"Malformed document id {doc_id!r}"})
    return None


# Same reasoning as DOC_ID_RE/_validate_doc_id, for the thread store below --
# every /threads/{thread_id} route turns thread_id into a filename before
# ever reading it back.
THREAD_ID_RE = re.compile(r"^t-[A-Za-z0-9]+$")


def _validate_thread_id(thread_id: str) -> Optional[JSONResponse]:
    if not THREAD_ID_RE.match(thread_id or ""):
        return JSONResponse(status_code=400, content={"error": f"Malformed thread id {thread_id!r}"})
    return None


def _resolve_within(path: str, root: str) -> "tuple[str | None, JSONResponse | None]":
    """Second line of defense, for paths that come out of an index JSON
    file rather than the URL itself (`document.source_path`/
    `document.markdown_path`) -- that JSON is data on disk, not something
    either new route controls, so a malformed or hand-edited index record
    must not be able to hand back an arbitrary file (e.g. /etc/passwd).
    Resolves both `path` and `root` with `os.path.realpath` (following
    symlinks, collapsing `..`) and confirms the former is inside the
    latter. Returns `(resolved_path, None)` on success or `(None,
    error_response)` on failure."""
    resolved = os.path.realpath(path)
    resolved_root = os.path.realpath(root)
    if resolved != resolved_root and not resolved.startswith(resolved_root + os.sep):
        return None, JSONResponse(status_code=403, content={"error": "Resolved path escapes the expected directory"})
    return resolved, None


def _safe_stem(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "-", stem).strip("-") or "upload"
    return stem[:80]


def _find_existing_source_by_hash(content_hash: str) -> "str | None":
    """Scans `CONFIG.paths.source_dir` for a file whose content hash already
    matches -- used by `docs_upload` to recognize "this is the same PDF
    again" *before* minting a new filename/document id for it. A plain
    directory scan (not an index lookup) on purpose: a source that failed
    strict validation is never written to docs/.index at all, so its hash
    would otherwise be unrecorded anywhere and every retry would look like
    a brand new document. Source directories in this system are small
    (a personal document corpus, not a bulk ingestion pipeline), so an
    O(n) re-hash per upload is an acceptable, simple, always-correct
    tradeoff over maintaining a separate hash cache."""
    source_dir = CONFIG.paths.source_dir
    if not os.path.isdir(source_dir):
        return None
    for name in os.listdir(source_dir):
        path = os.path.join(source_dir, name)
        if not os.path.isfile(path) or os.path.splitext(name)[1].lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            with open(path, "rb") as f:
                if sha256_hex(f.read()) == content_hash:
                    return path
        except OSError:
            continue
    return None


def _relative_source_for(path: str) -> str:
    """Best-effort recovery of the `relative_source` an already-indexed
    document was originally indexed with -- used only by the title-override
    endpoints below when reprocessing an *existing* document, to keep the
    filename-derived fallback title (pipeline.py's `clean_filename_title`)
    consistent with what the original `/docs/upload` or `/ingest` scan would
    have produced, in case an override is cleared and resolution falls back
    to it. Never affects `document_id` (already fixed, passed explicitly by
    every caller here) or whether the override itself applies."""
    for root in (CONFIG.paths.source_dir, CONFIG.paths.legacy_docs_dir):
        try:
            rel = os.path.relpath(path, root)
        except ValueError:
            continue
        if not rel.startswith(".."):
            return rel
    return os.path.basename(path)


class DocStore:
    """
    In-memory document store built from document_converter's schema_version 2
    index files (document_converter/index_schema.py). Each index file already
    contains fully-embedded chunks, so loading is a single pass over
    docs/.index/*.json with no parallel arrays to keep in sync.
    """

    def __init__(self, config, pipeline: DocumentPipeline):
        self.config = config
        self.pipeline = pipeline
        self.chunks = []
        self.documents = {}
        self.reload()

    def reload(self):
        index_dir = self.config.paths.index_dir
        os.makedirs(index_dir, exist_ok=True)
        print(f"\n[store] Loading indices from {index_dir}...")
        self.chunks = []
        self.documents = {}

        # Live embedding-service metadata, fetched once per reload -- an
        # index whose vectors were embedded under a different model/
        # revision/dimension/chunk-token configuration than what's actually
        # running now must never be silently loaded into the same search
        # space (BGE vectors and Qwen vectors must never mix). Best-effort:
        # if the embedding service isn't reachable yet, document metadata
        # still loads (listing docs shouldn't require a live model), but
        # the compatibility check itself is skipped with a clear warning
        # rather than failing reload() outright.
        live_embedding_info = None
        try:
            live_embedding_info = self.pipeline.embedding_client.health()
        except Exception as e:
            print(f"[store] ⚠ Could not reach the embedding service to verify index compatibility: {e}")

        index_files = sorted(glob.glob(os.path.join(index_dir, "*.json")))
        legacy_count = 0
        incompatible_count = 0
        for index_file in index_files:
            try:
                payload = load_index(index_file)
            except Exception as e:
                print(f"[store] ✗ Failed to load {index_file}: {e}")
                continue

            if is_legacy_schema(payload):
                legacy_count += 1
                continue

            if live_embedding_info is not None:
                mismatch = check_embedding_compatibility(
                    payload, live_embedding_info,
                    self.config.chunking.target_tokens, self.config.chunking.max_tokens,
                )
                if mismatch:
                    incompatible_count += 1
                    print(f"[store] ✗ Skipping {index_file}: embedding-incompatible with the active "
                          f"configuration ({mismatch})")
                    continue

            doc = payload["document"]
            pipeline_meta = payload["pipeline"]
            validation = payload["validation"]
            doc_id = doc["document_id"]
            chunks = payload.get("chunks", [])

            self.documents[doc_id] = {
                "title": doc["title"],
                "title_source": doc.get("title_source") or "",
                "source_path": doc["source_path"],
                "source_format": pipeline_meta["extractor"],
                "ocr_used": pipeline_meta["ocr_used"],
                "created_at": doc["created_at"],
                "updated_at": doc["updated_at"],
                "chunk_count": len(chunks),
                "retrieval_ready": validation["retrieval_ready"],
                "warnings": validation["warnings"],
                "errors": validation["errors"],
            }

            n_embedded = 0
            for c in chunks:
                if not c.get("embedding"):
                    continue
                n_embedded += 1
                record = {
                    "doc_id": doc_id,
                    "doc_title": doc["title"],
                    "source_path": doc["source_path"],
                    "source_format": pipeline_meta["extractor"],
                    "chunk_idx": c["chunk_index"],
                    "chunk_id": c["chunk_id"],
                    "heading": c["heading"],
                    "header_path": c["header_path"],
                    "text": c["text"],
                    "embedding": c["embedding"],
                    "source_pages": c.get("source_pages", []),
                    # "main_text" for every chunk written before this field
                    # existed too -- see index_schema.CONTENT_KIND_MAIN_TEXT.
                    "content_kind": c.get("content_kind", "main_text"),
                }
                if record["content_kind"] == "visual_analysis":
                    record.update({
                        "visual_id": c.get("visual_id"),
                        "page": c.get("page"),
                        "bbox": c.get("bbox"),
                        "visual_type": c.get("visual_type"),
                        "image_path": c.get("image_path"),
                        "linked_chunk_ids": c.get("linked_chunk_ids", []),
                        "analysis_model": c.get("analysis_model"),
                        "analysis_version": c.get("analysis_version"),
                    })
                self.chunks.append(record)

            n_visual = sum(1 for c in chunks if c.get("content_kind") == "visual_analysis")
            status = "✓" if n_embedded == len(chunks) else f"✗ {n_embedded}/{len(chunks)} embedded"
            extra = f", {n_visual} visual" if n_visual else ""
            print(f"[store] {doc['title']}: {len(chunks) - n_visual} chunks{extra} {status}")

        if legacy_count:
            print(f"[store] {legacy_count} legacy-schema index file(s) found in {index_dir}; "
                  f"run `python -m document_converter.cli rebuild` to migrate them")
        if incompatible_count:
            print(f"[store] {incompatible_count} index file(s) skipped: embedded under a different "
                  f"model/configuration than the active embedding service")

        # Lookup structures for targeted context expansion (spec Phase 9) --
        # built once per reload, not per query. `_chunk_by_key` resolves a
        # visual's `linked_chunk_ids` back to the actual MAIN_TEXT chunk
        # record (scoped by (doc_id, chunk_id) since chunk ids are only
        # unique within a document); `_visuals_by_linked_chunk` is the
        # reverse direction, so a strongly-ranked MAIN_TEXT chunk can also
        # surface the visual(s) that reference it.
        self._chunk_by_key = {(c["doc_id"], c["chunk_id"]): c for c in self.chunks}
        self._visuals_by_linked_chunk = {}
        for c in self.chunks:
            if c.get("content_kind") != "visual_analysis":
                continue
            for linked_id in c.get("linked_chunk_ids", []):
                self._visuals_by_linked_chunk.setdefault((c["doc_id"], linked_id), []).append(c["chunk_id"])

        n_visual_total = sum(1 for c in self.chunks if c.get("content_kind") == "visual_analysis")
        extra = f" ({n_visual_total} visual)" if n_visual_total else ""
        print(f"[store] ✓ Loaded {len(self.documents)} document(s), {len(self.chunks)} chunk(s){extra}\n")

    # Context-expansion budget (spec Phase 9: "targeted... not document
    # reconstruction") -- at most this many extra linked chunks/visuals are
    # folded into one search() response, on top of the `k` ranked hits.
    MAX_LINKED_CONTEXT = 4

    def _hit_dict(self, c: dict, score: Optional[float] = None, linked_from: Optional[str] = None) -> dict:
        d = {
            "doc_id": c["doc_id"],
            "chunk_id": c["chunk_id"],
            "name": c["doc_title"],
            "chunk": c["chunk_idx"],
            "heading": c["heading"],
            "text": c["text"],
            "score": round(score if score is not None else c.get("score", 0), 4),
            "source_path": c["source_path"],
            "source_format": c.get("source_format"),
            "source_pages": c.get("source_pages", []),
            "content_kind": c.get("content_kind", "main_text"),
        }
        if linked_from:
            # Present but not separately ranked -- pulled in as targeted
            # context for `linked_from` (a chunk_id or visual_id already in
            # the top results), not a hybrid-ranked hit of its own.
            d["linked_from"] = linked_from
        if d["content_kind"] == "visual_analysis":
            d.update({
                "visual_id": c.get("visual_id"),
                "page": c.get("page"),
                "visual_type": c.get("visual_type"),
                "image_path": c.get("image_path"),
                "linked_chunk_ids": c.get("linked_chunk_ids", []),
            })
        return d

    def search(self, query: str, k=4, alpha=0.7, expand_context: bool = True):
        """Search chunks using hybrid (semantic + BM25) ranking. Results mix
        `content_kind == "main_text"` and `"visual_analysis"` records
        indistinguishably for ranking purposes (both are just text+embedding
        to `HybridRanker`) but are labeled by `content_kind` in the
        response. When `expand_context` is true (the default -- the normal
        question-answering path), a ranked visual's linked MAIN_TEXT chunks
        (and vice versa) are appended after the ranked hits, each tagged
        `linked_from`, up to `MAX_LINKED_CONTEXT` total -- targeted context
        expansion, never a full document dump (spec Phase 9)."""
        logger.info(f"[search] Query: '{query}', k={k}, alpha={alpha}")

        if not self.chunks:
            logger.error("[search] No chunks loaded!")
            raise RuntimeError("No chunks loaded")

        # QUERY LIFECYCLE (spec "QUERY / ANSWER LIFECYCLE"): embed the query,
        # then explicitly release the embedding model's GPU memory *before*
        # returning to the caller -- retrieval below is pure CPU (BM25 +
        # cosine over already-loaded vectors), and the next step after this
        # response is chat-model generation (llama-server, see
        # /chat/stream), which must not have to contend with this process's
        # embedding model for VRAM. Best-effort and non-fatal: a release
        # that can't be measured/confirmed is logged, never blocks the
        # response (this endpoint's job is retrieval correctness, not GPU
        # bookkeeping).
        baseline_mib = gpu_lifecycle.read_gpu_memory_used_mib()
        query_embedding = self.pipeline.embedding_client.embed_one(query, input_type="query")
        released = self.pipeline.embedding_client.release()
        barrier = None
        if baseline_mib is not None:
            barrier = gpu_lifecycle.wait_for_vram_release(baseline_mib, tolerance_mib=1024, timeout_s=10.0)
            if not barrier.released:
                logger.warning(f"[search] embedding-model VRAM barrier did not confirm release within "
                                f"{barrier.elapsed_s:.1f}s (baseline={baseline_mib} MiB, final={barrier.final_mib} MiB)")
        logger.info(f"[search] query embedded; embedding-model release={'ok' if released else 'unreachable'}"
                    f"{' (VRAM confirmed at/near baseline)' if barrier and barrier.released else ''}")

        ranker = HybridRanker(alpha=alpha)
        top_results = ranker.rank(query, self.chunks, query_embedding, k=k, debug=True)

        logger.info(f"[search] Got {len(top_results)} results")
        hits = [self._hit_dict(c) for c in top_results]

        if expand_context:
            already = {(c["doc_id"], c["chunk_id"]) for c in top_results}
            added = 0
            for c in top_results:
                if added >= self.MAX_LINKED_CONTEXT:
                    break
                doc_id = c["doc_id"]
                extra_ids: List[str] = []
                if c.get("content_kind") == "visual_analysis":
                    extra_ids = list(c.get("linked_chunk_ids", []))
                else:
                    extra_ids = list(self._visuals_by_linked_chunk.get((doc_id, c["chunk_id"]), []))
                for extra_id in extra_ids:
                    if added >= self.MAX_LINKED_CONTEXT:
                        break
                    key = (doc_id, extra_id)
                    if key in already:
                        continue
                    extra_chunk = self._chunk_by_key.get(key)
                    if extra_chunk is None:
                        continue
                    already.add(key)
                    hits.append(self._hit_dict(extra_chunk, score=c.get("score", 0), linked_from=c["chunk_id"]))
                    added += 1

        return hits

    def list_docs(self):
        """List all indexed documents."""
        out = []
        for doc_id, meta in sorted(self.documents.items(), key=lambda x: x[1]["title"]):
            out.append({
                "id": doc_id,
                "name": meta["title"],
                "title_source": meta["title_source"],
                "has_title_override": meta["title_source"] == "manual_override",
                "source_format": meta["source_format"],
                "source_path": meta["source_path"],
                "chunks": meta["chunk_count"],
                "created_at": meta["created_at"],
                "updated_at": meta["updated_at"],
                "retrieval_ready": meta["retrieval_ready"],
                "warnings": meta["warnings"],
                "errors": meta["errors"],
                "ocr_used": meta["ocr_used"],
            })
        return out


# ===== SOURCE MAPPING / GROUNDING =====
# Shared by /docs/search's browser-facing shape (unchanged, `_hit_dict`
# above) and /chat/stream's Source objects below -- both start from the same
# `DocStore.search()` hits.

# Relevance buckets, derived from control.py's hybrid formula
# (0.7*cosine + 0.3*bm25_norm, empirically ~0.2-0.8 in practice for this
# corpus/embedding model). Tune here if that formula or its weights change.
RELEVANCE_HIGH = 0.55
RELEVANCE_MEDIUM = 0.35


def relevance_bucket(score) -> str:
    if score is None:
        return "low"
    if score >= RELEVANCE_HIGH:
        return "high"
    if score >= RELEVANCE_MEDIUM:
        return "medium"
    return "low"


def hit_to_source(h: dict) -> dict:
    """Maps one DocStore.search() hit to the Source object schema the
    frontend's sources panel renders. `content` is the chunk text exactly as
    it will be fed to the model -- verbatim, untruncated (never the
    snippet-sliced preview the old inline `renderSources` used)."""
    return {
        "id": h.get("chunk_id"),
        "title": h.get("name"),
        "type": "document",  # this pipeline has no web/code/database sources today
        "content": h.get("text", ""),
        "score": h.get("score"),
        "document_id": h.get("doc_id"),
        "metadata": {
            "page": h.get("page"),
            "chunk_index": h.get("chunk"),
            "url": None,
            "retrieved_at": _now(),
        },
    }


def _src_label(h: dict) -> str:
    """Python port of index.html's srcLabel() -- must stay in sync with it."""
    base = h["name"] + (" › " + h["heading"] if h.get("heading") else " · chunk " + str(h.get("chunk", "?")))
    if h.get("content_kind") == "visual_analysis":
        page = ("p." + str(h["page"])) if h.get("page") is not None else (
            "p." + str(h["source_pages"][0]) if h.get("source_pages") else "")
        vt = (h.get("visual_type") or "visual").replace("_", " ")
        return base + " · visual (" + vt + (", " + page if page else "") + ")"
    if h.get("linked_from"):
        return base + " · linked context"
    return base


def ground_last_user_turn(out_msgs: list, question: str, hits: Optional[list]) -> None:
    """Python port of index.html's groundLastUserTurn() -- rewrites the last
    user message in `out_msgs` in place. Must stay in sync with it (same
    three cases, same wording) since it replaces that client-side step now
    that generation is orchestrator-mediated."""
    def rewrite_last_user(content: str) -> None:
        for i in range(len(out_msgs) - 1, -1, -1):
            if out_msgs[i]["role"] == "user":
                out_msgs[i] = {"role": "user", "content": content}
                return

    if hits is None:
        rewrite_last_user(
            "[Knowledge base unreachable — no sources could be retrieved for this "
            "question. Answer only if you are certain, and state that the knowledge "
            "base was offline.]\n\nQuestion: " + question)
        return
    if not hits:
        rewrite_last_user(
            "The knowledge base returned no matching sources for the question below. "
            "Say plainly that the indexed documents do not cover it. Do NOT answer from "
            "outside knowledge.\n\nQuestion: " + question)
        return
    ctx = "\n\n---\n\n".join(f"[Source {i + 1}] {_src_label(h)}\n{h['text']}" for i, h in enumerate(hits))
    rewrite_last_user(
        "Answer the question using ONLY the sources below. Cite each source you use "
        "inline like [Source 1]. If the sources do not fully cover the question, say "
        "exactly what is missing — do NOT fill gaps with outside knowledge.\n\n"
        "If the sources describe an ordered, numbered procedure and the question asks "
        "how to do something, answer as an incrementing numbered list that follows the "
        "source order exactly — one source step per answer step, never merged or "
        "renumbered, and never invented. Include every source step needed to answer the "
        "question, and only those. For any other kind of question, answer in normal "
        "prose.\n\n"
        "=== SOURCES ===\n" + ctx + "\n\n=== QUESTION ===\n" + question)


# GET /api/models no longer proxies a model server's tag list: there is
# exactly one chat model now (CHAT_MODEL_* above), so the endpoint below
# returns it as a static entry.

STORE = DocStore(CONFIG, PIPELINE)

# ===== JOB TAB: build-job runner =====
# The Job tab's agent (job_runner.py) drives the same llama-server chat model
# as interactive chat, one job at a time, each job building inside its own
# jobs/<id>/output directory. KB retrieval for a job goes through the same
# DocStore.search + GPU_STAGE_LOCK path as /chat/stream (the search releases
# the embedding model's VRAM itself before the agent loop starts generating).
JOBS_ROOT = os.path.join(BASE_DIR, "jobs")
os.makedirs(JOBS_ROOT, exist_ok=True)


def _job_kb_retrieve(query: str) -> list:
    """KB snippets for a build job's briefing: the top chunks for the job
    description, as plain text. Empty when nothing is indexed yet."""
    if not STORE.chunks:
        return []
    with GPU_STAGE_LOCK:
        hits = STORE.search(query, k=4)
    return [h.get("text", "") for h in hits if h.get("text")]


JOB_RUNNER = job_runner.JobRunner(
    JOBS_ROOT, chat_completion, retrieve=_job_kb_retrieve,
    model_name=CHAT_MODEL_NAME)

_JOB_ID_RE = re.compile(r"^j-[0-9a-f]{12}$")


# ===== FASTAPI SETUP =====

app = FastAPI(docs_url="/api-docs", redoc_url=None)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ===== MODELS =====

class SearchReq(BaseModel):
    query: str
    k: int = 4


class TitleOverrideReq(BaseModel):
    title: str


class Msg(BaseModel):
    role: str
    content: str


class SessionEndReq(BaseModel):
    messages: List[Msg]
    # Legacy request fields, still accepted so older clients don't 422 --
    # summarization always runs on the one configured chat model
    # (CHAT_MODEL_*, served by llama-server), so all three are ignored.
    model: str = ""
    ollama_host: str = ""
    num_ctx: int = 0


class ThreadCreateReq(BaseModel):
    title: str = ""


class ThreadRenameReq(BaseModel):
    title: str


class ChatStreamReq(BaseModel):
    thread_id: str
    # Informational only: generation always runs on the one configured chat
    # model (CHAT_MODEL_*). Kept so older clients don't 422.
    model: str = ""
    text: str
    system_prompt: str = ""
    project_context: str = ""
    # Drives the frontend's token monitor only -- the server context is
    # fixed at CHAT_NUM_CTX when llama-server launches.
    num_ctx: int = CHAT_NUM_CTX
    use_kb: bool = True
    kb_k: int = 4


# ===== ENDPOINTS =====

@app.get("/health")
def health():
    """Health check."""
    result = {"ok": True, "dir": BASE_DIR, "chunks": len(STORE.chunks), "documents": len(STORE.documents)}
    return result


@app.get("/api/models")
def api_models():
    """The chat model selector's source. Exactly one chat model exists now
    (CHAT_MODEL_*, served by llama-server); the embedding model (separate
    sentence-transformers service) and the vision model (Ollama) are RAG
    pipeline components, not chat options, and are never listed here.
    `available` reports whether the model is being served right now or can
    be launched on demand (binary + GGUF present) -- the frontend renders
    unavailable entries disabled rather than hiding them."""
    available = chat_server_health() or (
        os.path.exists(LLAMA_SERVER_BIN) and os.path.exists(CHAT_MODEL_PATH))
    return {"models": [{"id": CHAT_MODEL_ID, "name": CHAT_MODEL_NAME, "available": available}]}


@app.get("/gpu/status")
def gpu_status():
    """Read-only GPU/model-residency diagnostic (strict sequential GPU
    model lifecycle phase): real `nvidia-smi` memory usage, every
    GPU-compute process nvidia-smi reports (so an *unrelated* GPU user --
    e.g. a game -- is visible rather than silently confused for this
    project's own usage, spec "EXTERNAL GPU USERS"), Ollama's own
    `/api/ps` (which models it currently has resident), and this
    process's embedding-service `/health` (model id + current device).
    Not used by index.html/document_converter.html today (spec: "do not
    redesign the UI during this task") -- exists for operational
    visibility and for the acceptance-test evidence this phase's source-
    of-truth entry cites."""
    embedding_info = None
    try:
        embedding_info = {
            "model": CONFIG.embedding.model, "host": CONFIG.embedding.host,
            **{k: v for k, v in vars(PIPELINE.embedding_client.health(force=True)).items()},
        }
    except Exception as e:
        embedding_info = {"error": str(e)}
    return {
        "nvidia_smi_available": gpu_lifecycle.nvidia_smi_available(),
        "gpu_memory_used_mib": gpu_lifecycle.read_gpu_memory_used_mib(),
        "gpu_compute_processes": gpu_lifecycle.list_gpu_compute_processes(),
        "ollama_ps": gpu_lifecycle.ollama_ps("http://127.0.0.1:11434"),
        "embedding_service": embedding_info,
    }


@app.get("/state")
def get_state():
    """Get current session state."""
    return load_state()


@app.put("/state")
def put_state(state: dict):
    """Update session state."""
    keep = load_state()
    for k in DEFAULT_STATE:
        if k in state:
            keep[k] = state[k]
    return save_state(keep)


@app.post("/ingest")
def ingest():
    """Rescan the legacy docs/ markdown tree and index anything new or changed."""
    logger.info("[ingest endpoint] Ingestion triggered via API")
    try:
        ingester = DocIngester(config=CONFIG)
        with GPU_STAGE_LOCK:
            report = ingester.ingest()
        if report["new"]:
            logger.info(f"[ingest endpoint] Reloading DocStore ({len(report['new'])} new/changed docs)")
            STORE.reload()
        logger.info(f"[ingest endpoint] Done: {len(report['new'])} indexed, {len(report['failed'])} failed")
        return {"report": report, "docs": STORE.list_docs()}
    except Exception as e:
        logger.error(f"[ingest endpoint] Ingestion failed: {e}")
        logger.exception("Full traceback:")
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/docs")
def docs_list():
    """List all indexed documents."""
    return {"docs": STORE.list_docs(), "supported_extensions": sorted(SUPPORTED_EXTENSIONS)}


@app.get("/docs/jobs")
def docs_jobs_list():
    with JOBS_LOCK:
        return {"jobs": [dict(j) for j in JOBS.values()]}


@app.get("/docs/jobs/{job_id}")
def docs_job_get(job_id: str):
    snapshot = _job_snapshot(job_id)
    if snapshot is None:
        return JSONResponse(status_code=404, content={"error": f"No job with id {job_id!r}"})
    return snapshot


JOB_POLL_INTERVAL_S = 0.5
JOB_KEEPALIVE_INTERVAL_S = 15.0


@app.get("/docs/jobs/{job_id}/stream")
def docs_job_stream(job_id: str):
    """SSE progress feed for one job (reuses `_sse`, the same event shape
    `/chat/stream`/localAgent already use). Polls the job record rather than
    holding JOBS_LOCK for the life of the connection -- `_job_snapshot`
    takes and releases the lock per read, so the generator below never
    holds it while yielding (or while blocked in `time.sleep`)."""
    if _job_snapshot(job_id) is None:
        return JSONResponse(status_code=404, content={"error": f"No job with id {job_id!r}"})

    def gen():
        last_progress_key = None
        last_sent_at = time.time()
        while True:
            snap = _job_snapshot(job_id)
            if snap is None:
                yield _sse("error", {"error": f"Job {job_id!r} no longer exists"})
                return
            state = snap["state"]
            if state in ("queued", "running"):
                key = (state, snap.get("stage"), snap.get("current"), snap.get("total"))
                if key != last_progress_key:
                    yield _sse("progress", {
                        "state": state, "stage": snap.get("stage"),
                        "current": snap.get("current"), "total": snap.get("total"),
                        "remaining_seconds": snap.get("remaining_seconds"),
                        "eta_lower_bound": snap.get("eta_lower_bound"),
                    })
                    last_progress_key = key
                    last_sent_at = time.time()
                elif time.time() - last_sent_at >= JOB_KEEPALIVE_INTERVAL_S:
                    yield ": keepalive\n\n"
                    last_sent_at = time.time()
                time.sleep(JOB_POLL_INTERVAL_S)
                continue
            if state == "succeeded":
                yield _sse("done", {"result": snap.get("result")})
            else:
                yield _sse("error", {"error": snap.get("error"), "validation": snap.get("validation")})
            return

    return StreamingResponse(gen(), media_type="text/event-stream")


# NOTE: registered *before* `/docs/{doc_id}` below on purpose -- FastAPI/
# Starlette matches routes in registration order, and `/docs/{doc_id}` is an
# untyped path parameter that would otherwise greedily swallow a literal
# `/docs/jobs` request (doc_id="jobs") before it ever reached the routes
# above.
@app.get("/docs/{doc_id}")
def docs_detail(doc_id: str):
    """Full index payload for one document: metadata, pipeline/embedding info, and every chunk."""
    id_error = _validate_doc_id(doc_id)
    if id_error:
        return id_error
    index_path = os.path.join(CONFIG.paths.index_dir, f"{doc_id}.json")
    if not os.path.exists(index_path):
        return JSONResponse(status_code=404, content={"error": f"No document with id {doc_id!r}"})
    try:
        payload = load_index(index_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to read index for {doc_id!r}: {e}"})
    return payload


# Maps `document.source_format`/`pipeline.extractor` (the recorded extractor
# id, e.g. "pdf"/"docx" -- see document_converter/model.py's
# `DocumentMetadata.source_format`) to a media type for `/docs/{doc_id}/raw`.
# Deliberately keyed off this recorded field rather than re-sniffing the
# source file's extension -- the recorded value is what the pipeline
# actually parsed the file as, which is what a viewer should trust.
SOURCE_FORMAT_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "html": "text/html",
    "htm": "text/html",
    "txt": "text/plain",
    "md": "text/markdown",
    "markdown": "text/markdown",
}


def _load_doc_index_payload(doc_id: str):
    """Shared by /docs/{doc_id}/raw and /docs/{doc_id}/markdown: validate the
    id, then load {index_dir}/{doc_id}.json. Returns `(payload, None)` or
    `(None, error_response)`."""
    id_error = _validate_doc_id(doc_id)
    if id_error:
        return None, id_error
    index_path = os.path.join(CONFIG.paths.index_dir, f"{doc_id}.json")
    if not os.path.exists(index_path):
        return None, JSONResponse(status_code=404, content={"error": f"No document with id {doc_id!r}"})
    try:
        payload = load_index(index_path)
    except Exception as e:
        return None, JSONResponse(status_code=500, content={"error": f"Failed to read index for {doc_id!r}: {e}"})
    return payload, None


@app.get("/docs/{doc_id}/raw")
def docs_raw(doc_id: str):
    """Return the original uploaded file's bytes, for inline display in an
    embedded viewer (no Content-Disposition: attachment -- that would force
    a download instead of rendering in place)."""
    payload, error = _load_doc_index_payload(doc_id)
    if error:
        return error

    source_path = (payload.get("document", {}) or {}).get("source_path") or ""
    if not source_path:
        return JSONResponse(status_code=404, content={"error": f"No source_path recorded for {doc_id!r}"})

    resolved, escape_error = _resolve_within(source_path, CONFIG.paths.source_dir)
    if escape_error:
        return escape_error

    if not os.path.isfile(resolved):
        return JSONResponse(status_code=404, content={"error": f"Source file for {doc_id!r} no longer exists on disk"})

    source_format = (payload.get("pipeline", {}) or {}).get("extractor") or ""
    media_type = SOURCE_FORMAT_MEDIA_TYPES.get(source_format.lower(), "application/octet-stream")
    return FileResponse(resolved, media_type=media_type)


@app.get("/docs/{doc_id}/markdown")
def docs_markdown(doc_id: str):
    """Return the pipeline's already-rendered markdown for a document --
    the fallback view for formats a browser can't render natively (DOCX).
    Reads and returns the file as-is; never reconstructs text by
    concatenating chunks (chunking is lossy at section boundaries)."""
    payload, error = _load_doc_index_payload(doc_id)
    if error:
        return error

    markdown_path = (payload.get("document", {}) or {}).get("markdown_path") or ""
    if not markdown_path:
        return JSONResponse(status_code=404, content={"error": f"No markdown_path recorded for {doc_id!r}"})

    resolved, escape_error = _resolve_within(markdown_path, CONFIG.paths.processed_dir)
    if escape_error:
        return escape_error

    if not os.path.isfile(resolved):
        return JSONResponse(status_code=404, content={"error": f"Markdown file for {doc_id!r} no longer exists on disk"})

    return FileResponse(resolved, media_type="text/markdown; charset=utf-8")


@app.post("/docs/search")
def docs_search(req: SearchReq):
    """Search indexed documents using hybrid ranking."""
    logger.info(f"[search endpoint] Search query: '{req.query}', k={req.k}")
    try:
        with GPU_STAGE_LOCK:
            hits = STORE.search(req.query, k=req.k)
        logger.info(f"[search endpoint] Returned {len(hits)} results")
        return {"hits": hits}
    except Exception as e:
        logger.error(f"[search endpoint] Search failed: {e}")
        logger.exception("Full traceback:")
        return JSONResponse(status_code=400, content={"error": str(e)})


# ===== JOB REGISTRY (see JOBS/JOBS_LOCK near GPU_STAGE_LOCK) =====
#
# `/docs/upload` and `/docs/preview` used to run the whole extract->chunk->
# embed pipeline synchronously inside an `async def` handler -- a blocking
# call in an async handler blocks the *entire* event loop, so chat, search,
# and even `/health` all stopped responding for the duration of one ingest.
# Both endpoints now stream the upload to disk, hand the actual pipeline
# call off to a background thread, and return 202 with a job id immediately;
# the caller polls/streams progress via the endpoints below instead of
# waiting on the original request.

def _new_job(kind: str, filename: str, document_id: str) -> dict:
    job = {
        "job_id": "j-" + uuid.uuid4().hex[:16],
        "kind": kind,
        "filename": filename,
        "document_id": document_id,
        "state": "queued",
        "stage": None,
        "current": None,
        "total": None,
        "started_at": _now(),
        "updated_at": _now(),
        "result": None,
        "error": None,
        "validation": None,
        # Remaining-time estimate (see `_progress_tick`): `page_count`/
        # `block_count` are learned opportunistically from the first
        # extraction/block-stage tick that reports one, `remaining_seconds`
        # is recomputed on every tick, and `eta_lower_bound` latches true
        # the first time this job passes through an unestimable stage.
        "page_count": None,
        "block_count": None,
        "remaining_seconds": None,
        "eta_lower_bound": False,
    }
    with JOBS_LOCK:
        JOBS[job["job_id"]] = job
        _evict_terminal_jobs_locked()
    return dict(job)


def _evict_terminal_jobs_locked() -> None:
    """Caller must hold JOBS_LOCK. Keeps at most MAX_JOBS jobs, evicting the
    oldest *terminal* (succeeded/failed) job first -- a queued or running
    job is never evicted, no matter how many of those there currently are.
    `JOBS` is a plain dict, so insertion order already doubles as "oldest
    first"."""
    if len(JOBS) <= MAX_JOBS:
        return
    for job_id, job in list(JOBS.items()):
        if len(JOBS) <= MAX_JOBS:
            break
        if job["state"] in ("succeeded", "failed"):
            JOBS.pop(job_id, None)
            _JOB_PROGRESS_HISTORY.pop(job_id, None)


def _update_job(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.update(fields)
        job["updated_at"] = _now()


def _job_snapshot(job_id: str) -> Optional[dict]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job is not None else None


def _job_total_units(page_count: Optional[int], block_count: Optional[int]) -> Optional[float]:
    """Overall work-unit total for a job at its current state of knowledge --
    extraction-only until `block_count` is known, then recomputed to include
    every block-iterating pipeline stage once it is (see the module-level
    comment above `_EXTRACTION_STAGE_ORDER`)."""
    if page_count is None:
        return None
    total = float(len(_EXTRACTION_STAGE_ORDER) * page_count)
    if block_count is not None:
        total += len(_BLOCK_STAGES) * block_count
    return total


def _job_overall_current(stage: Optional[str], current: Optional[int],
                          page_count: Optional[int], block_count: Optional[int]) -> Optional[float]:
    """Converts one stage-local (stage, current) tick into overall units
    completed against `_job_total_units`'s denominator; `None` when this
    stage falls outside the modeled extract/normalize/repair/chunk/validate
    run (e.g. visual_analysis/embedding/writing_index/done) or too little is
    known yet to place it at all."""
    if stage is None or page_count is None:
        return None
    if stage in _EXTRACTION_STAGE_ORDER:
        idx = _EXTRACTION_STAGE_ORDER.index(stage)
        return float(idx * page_count + (current or 0))
    if stage not in _PIPELINE_STAGE_ORDER:
        return None
    base = float(len(_EXTRACTION_STAGE_ORDER) * page_count)
    if block_count is None:
        return base
    completed_block_stages = sum(
        1 for s in _PIPELINE_STAGE_ORDER[:_PIPELINE_STAGE_ORDER.index(stage)] if s in _BLOCK_STAGES
    )
    base += completed_block_stages * block_count
    if stage in _BLOCK_STAGES:
        base += (current or 0)
    return base


def _progress_tick(job_id: str, stage: str, payload: dict) -> None:
    """The one place a (stage, current, total) tick becomes both the job's
    plain stage/current/total fields (unchanged contract -- reset, not
    merely left alone, on every call; see the old `_job_progress_cb`
    docstring this absorbed) and its `remaining_seconds`/`eta_lower_bound`
    estimate, recomputed on every tick rather than derived once so a slow
    stage is never masked by an earlier fast one."""
    current = payload.get("current")
    total = payload.get("total")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["stage"] = stage
        job["current"] = current
        job["total"] = total
        job["updated_at"] = _now()

        if stage in _EXTRACTION_STAGE_ORDER and total:
            job["page_count"] = total
        elif stage in _BLOCK_STAGES and total:
            job["block_count"] = total
        if stage in _UNESTIMABLE_STAGES:
            job["eta_lower_bound"] = True
        if stage == "done":
            job["remaining_seconds"] = 0.0
            _JOB_PROGRESS_HISTORY.pop(job_id, None)
            return

        overall_total = _job_total_units(job["page_count"], job["block_count"])
        overall_current = _job_overall_current(stage, current, job["page_count"], job["block_count"])
        if overall_total is None or overall_current is None:
            job["remaining_seconds"] = None
            return

        now = time.time()
        samples = _JOB_PROGRESS_HISTORY.setdefault(job_id, [])
        samples.append((now, overall_current))
        cutoff = now - _RATE_WINDOW_SECONDS * 2
        while len(samples) > 1 and samples[0][0] < cutoff:
            samples.pop(0)

        window_start = now - _RATE_WINDOW_SECONDS
        oldest = next((s for s in samples if s[0] >= window_start), samples[0])
        elapsed = now - oldest[0]
        done_delta = overall_current - oldest[1]
        if elapsed > 0 and done_delta > 0:
            rate = done_delta / elapsed
            job["remaining_seconds"] = max(0.0, (overall_total - overall_current) / rate)
        # else: not enough signal yet from this tick alone -- leave whatever
        # estimate was already showing rather than flashing it to unknown.


def _job_progress_cb(job_id: str) -> Callable[[str, dict], None]:
    """Built once per job and handed to `PIPELINE.preview()`/`index()` as
    `progress=` -- see `_progress_tick` for what happens on every call."""
    def cb(stage: str, payload: dict) -> None:
        _progress_tick(job_id, stage, payload)
    return cb


def _run_preview_job(job_id: str, tmp_dir: str, tmp_path: str, filename: str) -> None:
    _update_job(job_id, state="running")
    try:
        result = PIPELINE.preview(tmp_path, document_id="preview", relative_source=filename,
                                   progress=_job_progress_cb(job_id))
    except Exception as e:
        logger.error(f"[preview] failed for {filename}: {e}")
        _update_job(job_id, state="failed", error=str(e))
        return
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _update_job(job_id, state="succeeded", result={
        "title": result.title,
        # Where `title` came from ("metadata" | "front_matter_zone" |
        # "first_body_heading" | "unresolved" | ...) -- already computed by
        # `_prepare_for_profile`, just not previously threaded through this
        # response; lets the pre-index preview card show the same
        # automatic/override distinction the indexed knowledge-base table
        # already shows via `title_source`/`has_title_override`.
        "title_source": result.title_source,
        "source_format": result.document.metadata.source_format,
        "ocr_used": result.document.metadata.ocr_used,
        "markdown": result.markdown,
        "chunks": [{
            "index": c.chunk_index,
            "heading": c.heading,
            "provenance": c.provenance,
            "text": c.text,
            "word_count": c.word_count,
            "token_count": c.token_count,
        } for c in result.chunking_result.chunks],
        "validation": result.validation_report.to_dict(),
        "completion_state": result.completion_state,
    })


def _run_index_job(job_id: str, dest_path: str, doc_id: str, rel: str, filename: str, title_override: str) -> None:
    """`state` is set to "running" only once GPU_STAGE_LOCK is actually
    acquired -- while a job sits at "queued" past that point, the UI can
    tell the difference between "waiting for another ingest/search to
    release the GPU" and "stalled", which a single boolean couldn't."""
    from document_converter.split_ingest import SplitError, split_and_ingest
    try:
        with GPU_STAGE_LOCK:
            _update_job(job_id, state="running")
            cb = _job_progress_cb(job_id)
            # Oversized PDFs go through split_ingest instead of one monolithic
            # PIPELINE.index call. Parts land in a `.split_parts` subdirectory
            # of source_dir, so _find_existing_source_by_hash's os.listdir
            # scan never mistakes them for top-level sources.
            if (not title_override and dest_path.lower().endswith(".pdf")
                    and os.path.getsize(dest_path) > CONFIG.split_ingest_threshold_bytes):
                parts_dir = os.path.join(CONFIG.paths.source_dir, ".split_parts", _safe_stem(filename))
                results = split_and_ingest(PIPELINE, dest_path, work_dir=parts_dir, progress=cb)
                STORE.reload()
                failed_parts = [p for p in results if p["status"] == "failed"]
                if failed_parts:
                    _update_job(job_id, state="failed",
                                error=f"{len(failed_parts)} of {len(results)} split parts failed to index",
                                result={"split": True, "parts": results, "failed_parts": failed_parts,
                                        "docs": STORE.list_docs()})
                else:
                    _update_job(job_id, state="succeeded",
                                result={"split": True, "parts": results, "docs": STORE.list_docs()})
                return
            if title_override:
                result = PIPELINE.set_title_override(dest_path, title_override, document_id=doc_id,
                                                      relative_source=rel, progress=cb)
            else:
                result = PIPELINE.index(dest_path, document_id=doc_id, relative_source=rel, progress=cb)
    except PipelineValidationError as e:
        logger.warning(f"[upload] {filename} failed strict validation: {e}")
        # Exactly the payload shape the old synchronous 422 branch returned,
        # so the UI's existing failure rendering (built against that shape)
        # keeps working unchanged against `job["validation"]`.
        _update_job(job_id, state="failed", error=str(e), validation={
            "error": str(e),
            "document_id": doc_id,
            "validation": e.report.to_dict(),
            "completion_state": completion_state(e.report, indexed=False),
            "docs": STORE.list_docs(),
        })
        return
    except SplitError as e:
        logger.error(f"[upload] split ingest failed for {filename}: {e}")
        logger.exception("Full traceback:")
        _update_job(job_id, state="failed", error=str(e))
        return
    except Exception as e:
        logger.error(f"[upload] pipeline failed for {filename}: {e}")
        logger.exception("Full traceback:")
        _update_job(job_id, state="failed", error=str(e))
        return

    STORE.reload()
    _update_job(job_id, state="succeeded", result={
        "document_id": result.document_id,
        "title": result.title,
        "title_source": result.title_source,
        "chunks": len(result.chunking_result.chunks),
        "stages": {
            "extracted": True,
            "normalized": True,
            "semantic_structuring_used": result.structuring_used,
            "chunked": True,
            "validated": True,
            "embedded": not result.skipped_unchanged,
            "indexed": True,
        },
        "validation": result.validation_report.to_dict(),
        "completion_state": result.completion_state,
        "docs": STORE.list_docs(),
    })


@app.post("/docs/preview")
async def docs_preview(file: UploadFile = File(...)):
    """Extract/normalize/chunk/validate an uploaded file without writing or
    indexing anything. Streams the upload to disk and hands the actual
    extraction off to a background thread (see JOB REGISTRY above), returning
    202 with a job id immediately -- the UI calls this endpoint *before*
    "Add to knowledge base", so a 31,000-page PDF blocking here would mean
    the user never even reaches the upload step."""
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return JSONResponse(status_code=400, content={
            "error": f"Unsupported file type {ext!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        })

    # Keep the real filename on disk (in a unique per-request temp dir) rather
    # than a random name -- document.metadata.source_path feeds both the
    # provenance prefix baked into every chunk's text and (when no H1 is
    # found) the synthesized title, so a random temp name would leak into
    # both instead of the file the user actually uploaded. Streamed in
    # chunks -- never buffers the whole upload in memory -- with no size cap.
    tmp_dir = tempfile.mkdtemp(prefix="docpreview-")
    tmp_path = os.path.join(tmp_dir, _safe_stem(filename) + ext)
    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return JSONResponse(status_code=400, content={"error": str(e)})

    job = _new_job(kind="preview", filename=filename, document_id="preview")
    threading.Thread(target=_run_preview_job, args=(job["job_id"], tmp_dir, tmp_path, filename), daemon=True).start()
    return JSONResponse(status_code=202, content={
        "job_id": job["job_id"], "document_id": "preview", "filename": filename,
    })


@app.post("/docs/upload")
async def docs_upload(file: UploadFile = File(...), title_override: Optional[str] = Form(None)):
    """Upload a document and run it through the full pipeline: extract, normalize,
    optionally semantic-structure, chunk, validate, embed, and index. Streams
    the upload to disk and hands the actual pipeline run off to a background
    thread (see JOB REGISTRY above), returning 202 with a job id immediately
    instead of blocking the event loop for the whole run.

    `title_override`, when non-blank, is the document-conversion UI's inline
    title edit (made on the pre-index preview card, before "Add to knowledge
    base" is clicked) -- recorded as this document's manual title override
    (document_converter/title_overrides.py) and applied on this very first
    index pass, so the corrected title is what MAIN_TEXT/chunks/citations
    see from the start instead of "Untitled 1" briefly landing in the index
    and needing a second edit afterward."""
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return JSONResponse(status_code=400, content={
            "error": f"Unsupported file type {ext!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
            "docs": STORE.list_docs(),
        })

    os.makedirs(CONFIG.paths.source_dir, exist_ok=True)

    # Streamed straight to a temp file next to the eventual destination (same
    # filesystem, so the move below is a cheap rename), hashing as it goes --
    # never buffers the whole upload in memory, and with no size cap (this
    # pipeline's whole reason for existing is 750MB-1GB PDFs).
    tmp_fd, tmp_upload_path = tempfile.mkstemp(prefix="docupload-", dir=CONFIG.paths.source_dir)
    hasher = hashlib.sha256()
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                hasher.update(chunk)
                out.write(chunk)
    except Exception as e:
        os.remove(tmp_upload_path)
        return JSONResponse(status_code=400, content={"error": str(e), "docs": STORE.list_docs()})
    content_hash = hasher.hexdigest()

    # `stable_document_id` is deliberately path-based (see pipeline.py) so
    # re-processing "the same source" always overwrites the same index
    # entry instead of orphaning it under a fresh content-hash id. That
    # invariant only holds if re-uploading identical *content* lands back
    # on the same path -- the old name-collision fallback below (appending
    # a Unix timestamp) instead minted a brand new filename, and therefore
    # a brand new document id, every time the same PDF was re-uploaded
    # (e.g. retried after a strict-validation failure), silently defeating
    # the "one id per source" design and leaving every failed attempt's
    # artifacts behind. Checking content hash against already-present
    # sources first, before falling back to the timestamp-suffixed name for
    # genuinely different content, restores that invariant.
    existing_path = _find_existing_source_by_hash(content_hash)
    if existing_path is not None:
        dest_path = existing_path
        os.remove(tmp_upload_path)
        logger.info(f"[upload] {filename} content matches existing source "
                     f"{os.path.basename(dest_path)} (sha256={content_hash[:12]}...); reusing it instead of "
                     f"writing a duplicate")
    else:
        dest_name = f"{_safe_stem(filename)}{ext}"
        dest_path = os.path.join(CONFIG.paths.source_dir, dest_name)
        if os.path.exists(dest_path):
            base, e = os.path.splitext(dest_name)
            dest_path = os.path.join(CONFIG.paths.source_dir, f"{base}-{int(time.time())}{e}")
        os.replace(tmp_upload_path, dest_path)

    rel = os.path.relpath(dest_path, CONFIG.paths.source_dir)
    doc_id = stable_document_id(rel)
    title_override = (title_override or "").strip()

    job = _new_job(kind="index", filename=filename, document_id=doc_id)
    threading.Thread(
        target=_run_index_job, args=(job["job_id"], dest_path, doc_id, rel, filename, title_override), daemon=True,
    ).start()
    return JSONResponse(status_code=202, content={"job_id": job["job_id"], "document_id": doc_id, "filename": filename})


@app.put("/docs/{doc_id}/title")
def docs_set_title(doc_id: str, req: TitleOverrideReq):
    """Set a manual title override for an already-indexed document (the
    document-conversion UI's [edit] control) and reprocess just this one
    document so MAIN_TEXT/chunks/citations pick up the corrected title too
    -- never renames, moves, or otherwise touches the source file itself."""
    meta = STORE.documents.get(doc_id)
    if meta is None:
        return JSONResponse(status_code=404, content={"error": f"No document with id {doc_id!r}"})
    title = (req.title or "").strip()
    if not title:
        return JSONResponse(status_code=400, content={"error": "title must not be empty"})

    source_path = meta["source_path"]
    if not os.path.exists(source_path):
        return JSONResponse(status_code=409, content={
            "error": f"Source file for {doc_id!r} no longer exists on disk: {source_path}",
        })

    try:
        with GPU_STAGE_LOCK:
            result = PIPELINE.set_title_override(
                source_path, title, document_id=doc_id, relative_source=_relative_source_for(source_path),
            )
    except PipelineValidationError as e:
        logger.warning(f"[title] {doc_id} failed strict validation after override: {e}")
        return JSONResponse(status_code=422, content={
            "error": str(e), "document_id": doc_id,
            "validation": e.report.to_dict(), "docs": STORE.list_docs(),
        })

    STORE.reload()
    return {"document_id": doc_id, "title": result.title, "title_source": result.title_source, "docs": STORE.list_docs()}


@app.delete("/docs/{doc_id}/title")
def docs_clear_title(doc_id: str):
    """Remove a document's manual title override and reprocess it so it
    falls back to normal automatic title resolution."""
    meta = STORE.documents.get(doc_id)
    if meta is None:
        return JSONResponse(status_code=404, content={"error": f"No document with id {doc_id!r}"})

    source_path = meta["source_path"]
    if not os.path.exists(source_path):
        return JSONResponse(status_code=409, content={
            "error": f"Source file for {doc_id!r} no longer exists on disk: {source_path}",
        })

    try:
        with GPU_STAGE_LOCK:
            result = PIPELINE.clear_title_override(
                source_path, document_id=doc_id, relative_source=_relative_source_for(source_path),
            )
    except PipelineValidationError as e:
        logger.warning(f"[title] {doc_id} failed strict validation after clearing override: {e}")
        return JSONResponse(status_code=422, content={
            "error": str(e), "document_id": doc_id,
            "validation": e.report.to_dict(), "docs": STORE.list_docs(),
        })

    STORE.reload()
    return {"document_id": doc_id, "title": result.title, "title_source": result.title_source, "docs": STORE.list_docs()}


@app.delete("/docs/{doc_id}")
def docs_delete(doc_id: str):
    """Delete a document's index entry, normalized markdown, and report."""
    meta = STORE.documents.get(doc_id)
    source_path = None
    if meta is not None:
        source_path = os.path.join(CONFIG.paths.source_dir, os.path.basename(meta["source_path"]))
    PIPELINE.delete(doc_id, source_path=source_path)
    STORE.reload()
    return {"docs": STORE.list_docs()}


@app.post("/session/end")
def session_end(req: SessionEndReq):
    """
    End session: summarize conversation and extract state deltas.
    """
    convo = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in req.messages)
    ts = time.strftime("%Y%m%d-%H%M%S")

    # Save raw session
    with open(os.path.join(SESSIONS, f"session-{ts}.json"), "w", encoding="utf-8") as f:
        json.dump([m.dict() for m in req.messages], f, indent=2, ensure_ascii=False)

    summary, delta = "", {}

    SUMMARY_PROMPT = (
        "Summarize the following work session into concise markdown notes. Capture "
        "what was attempted, what was decided, what got finished, what is still open, "
        "and any commands, file paths, or errors that matter for continuing later. "
        "Be factual and brief. Do not invent anything not present.\n\nSESSION:\n"
    )

    EXTRACT_PROMPT = (
        "From the following work session, extract durable project facts as JSON with "
        "exactly these keys: completed (array of tasks clearly finished this session), "
        "current_task (string, the single most relevant next thing, or \"\"), "
        "open_tasks (array), known_bugs (array of unresolved problems), "
        "design_decisions (array of choices made and why), questions (array of open "
        "questions). Only include items clearly established in the conversation; use "
        "empty arrays when unsure. Output JSON only.\n\nSESSION:\n"
    )

    try:
        summary = chat_completion(
            [{"role": "user", "content": SUMMARY_PROMPT + convo}]).strip()
    except Exception as e:
        summary = f"(summary failed: {e})"

    try:
        delta = safe_json(chat_completion(
            [{"role": "user", "content": EXTRACT_PROMPT + convo}], want_json=True))
    except Exception:
        delta = {}

    if summary:
        with open(os.path.join(ARTIFACTS, f"session-{ts}.md"), "w", encoding="utf-8") as f:
            f.write(summary + "\n")

    state = load_state()
    _merge_list(state["completed"], delta.get("completed"))
    _merge_list(state["open_tasks"], delta.get("open_tasks"))
    _merge_list(state["known_bugs"], delta.get("known_bugs"))
    _merge_list(state["design_decisions"], delta.get("design_decisions"))
    _merge_list(state["questions"], delta.get("questions"))
    ct = (delta.get("current_task") or "").strip()
    if ct:
        state["current_task"] = ct
    state["last_summary"] = summary
    save_state(state)

    return {"state": state, "summary": summary, "archived": f"session-{ts}"}


# ===== THREADS (persisted chat history + per-message sources) =====

@app.get("/threads")
def threads_list():
    return {"threads": list_threads()}


@app.post("/threads")
def threads_create(req: ThreadCreateReq):
    return create_thread(req.title.strip())


@app.get("/threads/{thread_id}")
def threads_get(thread_id: str):
    id_error = _validate_thread_id(thread_id)
    if id_error:
        return id_error
    thread = load_thread(thread_id)
    if thread is None:
        return JSONResponse(status_code=404, content={"error": f"No thread with id {thread_id!r}"})
    return thread


@app.put("/threads/{thread_id}")
def threads_rename(thread_id: str, req: ThreadRenameReq):
    id_error = _validate_thread_id(thread_id)
    if id_error:
        return id_error
    thread = load_thread(thread_id)
    if thread is None:
        return JSONResponse(status_code=404, content={"error": f"No thread with id {thread_id!r}"})
    title = req.title.strip()
    if title:
        thread["title"] = title
    return save_thread(thread)


@app.delete("/threads/{thread_id}")
def threads_delete(thread_id: str):
    id_error = _validate_thread_id(thread_id)
    if id_error:
        return id_error
    delete_thread(thread_id)
    return {"ok": True}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ===== LOCAL AGENT =====
# The only component that can crawl, write into, or execute files on the
# local filesystem. Directories are user-entered only (see
# LOCAL_AGENT_DIRS above) -- the model never names a new path that becomes
# granted, it only ever operates inside one the user already added, and
# only within whichever of read/write/execute the user has switched on for
# it (each defaults OFF). Streams progress with `_sse()`, reusing the exact
# SSE shape /chat/stream uses (event: <name>\ndata: <json>\n\n) so the same
# fetch-stream reader on the frontend drives all of it.

def _real(path: str) -> str:
    return os.path.realpath(path)


def _is_contained(root: str, path: str) -> bool:
    """True iff `path` resolves -- symlinks and all -- to somewhere inside
    `root`. A plain string-prefix check is not enough: a symlink inside a
    granted directory that points outside it must be caught, so both sides
    are always realpath()'d before comparing. Used at every localAgent
    filesystem boundary: crawl, write, and exec all re-check this."""
    root_r = _real(root)
    path_r = _real(path)
    return path_r == root_r or path_r.startswith(root_r + os.sep)


def _err_stream(message: str):
    yield _sse("error", {"message": message})
    yield _sse("done", {"summary": message})


def _get_dir_or_error(dir_id: str, capability: str) -> dict:
    """Looks up a granted directory and enforces one capability against it.
    Every localAgent op (crawl/write/exec) calls this first -- a capability
    the user hasn't switched on is refused here, server-side, not merely
    hidden by the UI. Re-reads LOCAL_AGENT_DIRS fresh every call (never
    cached in a task object) so a capability flipped off mid-task, or the
    directory being removed, takes effect on the very next check."""
    d = LOCAL_AGENT_DIRS.get(dir_id)
    if d is None:
        raise ValueError("no such granted directory")
    if not os.path.isdir(d["path"]):
        raise ValueError("granted directory no longer exists on disk")
    if not d.get(capability):
        raise ValueError(f"'{capability}' is not enabled for this directory")
    return d


def _command_for(path: str) -> list:
    """The exact command that will run, shown to the human before the
    execute-confirm gate. Installer-class files are almost always a shell
    script or a self-extracting .run; anything else is executed directly
    (made executable first, see the exec step below)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".sh":
        return ["bash", path]
    if ext == ".run":
        return ["sh", path]
    return [path]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


APPROVAL_WAIT_SECONDS = 1800  # bounded so a forgotten confirmation can't pin a worker thread forever
EXEC_TIMEOUT_SECONDS = 600
MAX_TREE_ENTRIES = 2000
MAX_TREE_DEPTH = 6
DOC_EXTENSIONS = {".md", ".txt"}
INSTALLER_EXTENSIONS = {".sh", ".run", ".deb", ".rpm", ".appimage", ".bin", ".iso", ".img"}

LOCAL_AGENT_LOCK = threading.Lock()


def _wait_gate(evt_yes: threading.Event, evt_cancel: threading.Event, total_timeout: float) -> str:
    """Blocks in short ticks until either event fires or `total_timeout`
    elapses, so a task waiting on a human click still notices interruption
    (the toggle-off path) within about a second rather than hanging a
    worker thread forever. Returns 'approved' | 'cancelled' | 'timeout'."""
    waited = 0.0
    tick = 1.0
    while waited < total_timeout:
        if evt_cancel.is_set():
            return "cancelled"
        if evt_yes.wait(tick):
            return "approved"
        waited += tick
    return "timeout"


# ----- directories: user-entered grant list, with per-directory capability toggles -----

class LocalAgentDirAddReq(BaseModel):
    path: str


class LocalAgentDirPatchReq(BaseModel):
    read: Optional[bool] = None
    write: Optional[bool] = None
    execute: Optional[bool] = None


@app.get("/localagent/directories")
def localagent_dirs_list():
    return {"directories": list(LOCAL_AGENT_DIRS.values())}


@app.post("/localagent/directories")
def localagent_dirs_add(req: LocalAgentDirAddReq):
    """The only way a path enters the grant set. `path` is exactly what the
    user typed -- never a model-chosen or model-modified path -- validated
    as an absolute, existing directory and then realpath()'d so every later
    containment check has a stable root."""
    raw = (req.path or "").strip()
    if not raw or not os.path.isabs(raw):
        return JSONResponse(status_code=400, content={"error": "enter an absolute directory path"})
    real = os.path.realpath(raw)
    if not os.path.isdir(real):
        return JSONResponse(status_code=400, content={"error": "no such directory"})
    with LOCAL_AGENT_LOCK:
        for d in LOCAL_AGENT_DIRS.values():
            if d["path"] == real:
                return d
        entry = {
            "id": "ld-" + uuid.uuid4().hex[:16], "path": real,
            "read": False, "write": False, "execute": False, "created_at": _now(),
        }
        LOCAL_AGENT_DIRS[entry["id"]] = entry
        _save_local_agent_dirs()
    return entry


@app.patch("/localagent/directories/{dir_id}")
def localagent_dirs_patch(dir_id: str, req: LocalAgentDirPatchReq):
    with LOCAL_AGENT_LOCK:
        d = LOCAL_AGENT_DIRS.get(dir_id)
        if d is None:
            return JSONResponse(status_code=404, content={"error": "no such directory"})
        if req.read is not None:
            d["read"] = bool(req.read)
        if req.write is not None:
            d["write"] = bool(req.write)
        if req.execute is not None:
            d["execute"] = bool(req.execute)
        _save_local_agent_dirs()
        return d


@app.delete("/localagent/directories/{dir_id}")
def localagent_dirs_delete(dir_id: str):
    with LOCAL_AGENT_LOCK:
        if LOCAL_AGENT_DIRS.pop(dir_id, None) is not None:
            _save_local_agent_dirs()
    return {"ok": True}


# ----- read: crawl a granted directory and report its structure -----

def _build_tree(root: str) -> dict:
    count = [0]

    def walk(path: str, depth: int) -> dict:
        node = {"name": os.path.basename(path) or path, "type": "dir", "children": []}
        if depth >= MAX_TREE_DEPTH:
            node["truncated"] = True
            return node
        try:
            entries = sorted(os.listdir(path))
        except OSError:
            node["error"] = "unreadable"
            return node
        for name in entries:
            if count[0] >= MAX_TREE_ENTRIES:
                node["truncated"] = True
                break
            full = os.path.join(path, name)
            count[0] += 1
            if not _is_contained(root, full):
                node["children"].append({"name": name, "type": "rejected", "reason": "escapes granted root (symlink)"})
                continue
            if os.path.isdir(full) and not os.path.islink(full):
                node["children"].append(walk(full, depth + 1))
            else:
                node["children"].append({"name": name, "type": "file"})
        return node

    return walk(root, 0)


class LocalAgentDirIdReq(BaseModel):
    dir_id: str


@app.post("/localagent/crawl/stream")
def localagent_crawl_stream(req: LocalAgentDirIdReq):
    try:
        d = _get_dir_or_error(req.dir_id, "read")
    except ValueError as e:
        return StreamingResponse(_err_stream(str(e)), media_type="text/event-stream")

    def gen():
        yield _sse("crawling", {"directory": d["path"]})
        tree = _build_tree(d["path"])
        yield _sse("structure", {"tree": tree})
        yield _sse("done", {"summary": f"crawled {d['path']}"})

    return StreamingResponse(gen(), media_type="text/event-stream")


# ----- write: create a subdirectory or a document (never code) in a granted directory -----

class LocalAgentWriteReq(BaseModel):
    dir_id: str
    kind: str  # "dir" | "doc"
    name: str
    content: str = ""


@app.post("/localagent/write/stream")
def localagent_write_stream(req: LocalAgentWriteReq):
    try:
        d = _get_dir_or_error(req.dir_id, "write")
    except ValueError as e:
        return StreamingResponse(_err_stream(str(e)), media_type="text/event-stream")

    kind = (req.kind or "").strip()
    name = (req.name or "").strip().strip("/")
    if kind not in ("dir", "doc"):
        return StreamingResponse(_err_stream("kind must be 'dir' or 'doc'"), media_type="text/event-stream")
    if not name or "\\" in name or "\x00" in name or ".." in name.split("/"):
        return StreamingResponse(_err_stream("invalid name"), media_type="text/event-stream")
    if kind == "doc":
        ext = os.path.splitext(name)[1].lower()
        if ext not in DOC_EXTENSIONS:
            return StreamingResponse(_err_stream(
                "localAgent only creates documents (.md/.txt) -- code files are out of scope for this tool"
            ), media_type="text/event-stream")

    full = os.path.join(d["path"], name)
    if not _is_contained(d["path"], full):
        return StreamingResponse(_err_stream("path escapes the granted directory"), media_type="text/event-stream")

    def gen():
        yield _sse("writing", {"path": full, "kind": kind})
        try:
            if kind == "dir":
                os.makedirs(full, exist_ok=False)
            else:
                parent = os.path.dirname(full)
                os.makedirs(parent, exist_ok=True)
                if not _is_contained(d["path"], parent):
                    raise ValueError("path escapes the granted directory")
                with open(full, "x", encoding="utf-8") as f:
                    f.write(req.content or "")
        except FileExistsError:
            yield from _err_stream("already exists: " + full)
            return
        except Exception as e:
            yield from _err_stream(str(e))
            return
        yield _sse("wrote", {"path": full})
        yield _sse("done", {"summary": f"created {full}"})

    return StreamingResponse(gen(), media_type="text/event-stream")


# ----- execute: propose an installer-class file, run only on explicit human yes -----

def _pick_installer_candidate(root: str) -> Optional[str]:
    """Deterministic, rule-based candidate pick -- not a model call -- same
    posture the old grant-proposal logic used: the tool may propose, it
    never decides on its own that something runs (that's the confirm gate
    in localagent_exec_stream below, the actual non-negotiable linchpin)."""
    best, best_mtime = None, -1.0
    try:
        names = os.listdir(root)
    except OSError:
        return None
    for name in names:
        full = os.path.join(root, name)
        if not _is_contained(root, full) or not os.path.isfile(full):
            continue
        if os.path.splitext(name)[1].lower() not in INSTALLER_EXTENSIONS:
            continue
        try:
            mtime = os.stat(full).st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best, best_mtime = full, mtime
    return best


class LocalAgentExecTask:
    def __init__(self, task_id: str, dir_id: str, dir_path: str, file_path: str, sha256: str, command: list):
        self.id = task_id
        self.dir_id = dir_id
        self.dir_path = dir_path
        self.file = file_path
        self.sha256 = sha256
        self.command = command
        self.exit_code = None
        self.state = "proposed"
        self.confirmed = threading.Event()
        self.cancelled = threading.Event()


LOCAL_AGENT_EXEC_CURRENT: Optional[LocalAgentExecTask] = None
LOCAL_AGENT_EXEC_TERMINAL = ("done", "error", "cancelled")


class LocalAgentConfirmReq(BaseModel):
    yes: bool


@app.post("/localagent/exec/stream")
def localagent_exec_stream(req: LocalAgentDirIdReq):
    """The only endpoint that can execute a file, and only ever in-place in
    a granted directory (no move step). Proposing a candidate is not
    authorization: the stream halts at awaiting_exec_confirm and nothing
    runs until POST .../confirm {yes: true} -- see localagent_exec_confirm
    below. There is no other code path into the subprocess.run() call."""
    global LOCAL_AGENT_EXEC_CURRENT
    try:
        d = _get_dir_or_error(req.dir_id, "execute")
    except ValueError as e:
        return StreamingResponse(_err_stream(str(e)), media_type="text/event-stream")

    with LOCAL_AGENT_LOCK:
        if LOCAL_AGENT_EXEC_CURRENT is not None and LOCAL_AGENT_EXEC_CURRENT.state not in LOCAL_AGENT_EXEC_TERMINAL:
            return StreamingResponse(_err_stream("an execution task is already in flight"), media_type="text/event-stream")
        candidate = _pick_installer_candidate(d["path"])
        if candidate is None:
            return StreamingResponse(_err_stream("no installer-class file found in this directory"), media_type="text/event-stream")
        task = LocalAgentExecTask(
            new_message_id(), req.dir_id, d["path"], candidate, _sha256_file(candidate), _command_for(candidate))
        LOCAL_AGENT_EXEC_CURRENT = task

    def gen():
        global LOCAL_AGENT_EXEC_CURRENT
        try:
            command_str = " ".join(shlex.quote(a) for a in task.command)
            yield _sse("proposed_exec", {"task_id": task.id, "file": task.file, "sha256": task.sha256, "command": command_str})
            yield _sse("awaiting_exec_confirm", {"task_id": task.id})

            # THE GATE: nothing below this line runs until a human clicks yes
            # in localagent_exec_confirm -- the model's candidate pick above
            # is a proposal only, never a trigger.
            outcome = _wait_gate(task.confirmed, task.cancelled, APPROVAL_WAIT_SECONDS)
            if outcome != "approved":
                task.state = "cancelled" if outcome == "cancelled" else "error"
                yield from _err_stream("execution not confirmed (" + outcome + ")")
                return

            # re-check capability, containment, and hash at the moment of truth --
            # a toggle flipped off or the file touched out of band since the
            # proposal must still be caught here, not just at proposal time.
            d_now = LOCAL_AGENT_DIRS.get(task.dir_id)
            if d_now is None or not d_now.get("execute") or not _is_contained(task.dir_path, task.file):
                task.state = "error"
                yield from _err_stream("execution refused: directory access or execute permission changed")
                return
            if not os.path.isfile(task.file) or _sha256_file(task.file) != task.sha256:
                task.state = "error"
                yield from _err_stream("execution refused: file hash changed since it was shown for approval (touched out of band)")
                return

            try:
                os.chmod(task.file, os.stat(task.file).st_mode | 0o111)
            except OSError:
                pass

            yield _sse("executing", {"command": command_str, "cwd": task.dir_path})
            try:
                result = subprocess.run(task.command, cwd=task.dir_path, capture_output=True, timeout=EXEC_TIMEOUT_SECONDS)
                task.exit_code = result.returncode
            except subprocess.TimeoutExpired:
                task.exit_code = -1
            except Exception as e:
                logger.error(f"[localagent] execution failed: {e}")
                task.exit_code = -1
            yield _sse("exec_result", {"exit_code": task.exit_code})

            if task.exit_code == 0:
                task.state = "done"
                yield _sse("done", {"summary": "installed"})
            else:
                task.state = "error"
                summary = f"install failed (exit {task.exit_code})"
                yield _sse("error", {"message": summary})
                yield _sse("done", {"summary": summary})
        finally:
            with LOCAL_AGENT_LOCK:
                if LOCAL_AGENT_EXEC_CURRENT is task:
                    LOCAL_AGENT_EXEC_CURRENT = None

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/localagent/exec/{task_id}/confirm")
def localagent_exec_confirm(task_id: str, req: LocalAgentConfirmReq):
    """The explicit human yes/no this tool's execute gate is built on. yes
    sets `confirmed` (the stream proceeds to hash re-check + subprocess.run);
    no sets `cancelled` (the stream exits without running anything). This is
    the only way `task.confirmed` is ever set -- no model or agent code path
    reaches it."""
    t = LOCAL_AGENT_EXEC_CURRENT
    if t is None or t.id != task_id or t.state != "proposed":
        return JSONResponse(status_code=404, content={"error": "no execution awaiting confirmation"})
    if req.yes:
        t.confirmed.set()
    else:
        t.cancelled.set()
    return {"ok": True}


@app.post("/localagent/exec/{task_id}/interrupt")
def localagent_exec_interrupt(task_id: str):
    """Called by the client the instant the master or localAgent toggle
    goes off while an execution is mid-flight (awaiting confirmation, or
    running). Signals the same `cancelled` event a "no" click would -- no
    grant to revoke any more, directory access stays exactly as the user
    set it; this just stops the task cleanly."""
    t = LOCAL_AGENT_EXEC_CURRENT
    if t is not None and t.id == task_id:
        t.cancelled.set()
    return {"ok": True}


# ===== JOB TAB ENDPOINTS =====
# Thin HTTP shell over JOB_RUNNER (job_runner.py). The frontend polls
# GET /jobs/{id} for live progress; there is no SSE stream here because the
# snapshot already carries the full stage/item/event state and the agent
# loop's cadence (one model turn at a time) doesn't need push.

class JobCreateReq(BaseModel):
    description: str
    checklist: str
    system_prompt: str = ""
    title: str = ""


def _validate_job_id(job_id: str) -> Optional[JSONResponse]:
    if not _JOB_ID_RE.match(job_id or ""):
        return JSONResponse(status_code=400, content={"error": "invalid job id"})
    return None


@app.post("/jobs")
def jobs_create(req: JobCreateReq):
    """Create a build job and start its agent loop. One job runs at a time:
    while another is running this returns 409 and the new job stays on disk
    as queued (delete it or retry once the runner is free)."""
    job, err = JOB_RUNNER.create_job(req.description, req.checklist,
                                     req.system_prompt, req.title)
    if err:
        return JSONResponse(status_code=400, content={"error": err})
    ok, err = JOB_RUNNER.start(job["id"])
    if not ok:
        return JSONResponse(status_code=409, content={"error": err, "job_id": job["id"]})
    return {"job_id": job["id"], "status": "running"}


@app.get("/jobs")
def jobs_list():
    return {"jobs": JOB_RUNNER.list_jobs()}


@app.get("/jobs/{job_id}")
def jobs_get(job_id: str):
    id_error = _validate_job_id(job_id)
    if id_error:
        return id_error
    snap = JOB_RUNNER.snapshot(job_id)
    if snap is None:
        return JSONResponse(status_code=404, content={"error": "no such job"})
    return snap


@app.post("/jobs/{job_id}/cancel")
def jobs_cancel(job_id: str):
    id_error = _validate_job_id(job_id)
    if id_error:
        return id_error
    ok, err = JOB_RUNNER.cancel(job_id)
    if not ok:
        return JSONResponse(status_code=409, content={"error": err})
    return {"ok": True}


@app.delete("/jobs/{job_id}")
def jobs_delete(job_id: str):
    id_error = _validate_job_id(job_id)
    if id_error:
        return id_error
    ok, err = JOB_RUNNER.delete(job_id)
    if not ok:
        return JSONResponse(status_code=409, content={"error": err})
    return {"ok": True}


@app.post("/chat/stream")
def chat_stream(req: ChatStreamReq):
    """SSE chat endpoint (spec section 3.2): the orchestrator now sits
    between the browser and the llama-server chat model so it can emit
    state transitions -- `status` (retrieving / loading_model / generating),
    `sources`, `token`, `done` -- not just tokens. Sources are always
    retrieved and emitted *before* generation starts (section 2.1), and the
    assistant message (with its sources) is persisted to the thread store on
    completion so History and per-message source switching (section 2.3)
    work."""
    id_error = _validate_thread_id(req.thread_id)
    if id_error:
        return id_error
    thread = load_thread(req.thread_id)
    if thread is None:
        return JSONResponse(status_code=404, content={"error": f"No thread with id {req.thread_id!r}"})

    def gen():
        user_msg = {
            "id": new_message_id(), "role": "user", "content": req.text,
            "sources": None, "created_at": _now(),
        }
        thread["messages"].append(user_msg)
        save_thread(thread)

        out_msgs = []
        if req.system_prompt.strip():
            out_msgs.append({"role": "system", "content": req.system_prompt.strip()})
        if req.project_context.strip():
            out_msgs.append({"role": "system", "content": req.project_context.strip()})
        for m in thread["messages"]:
            out_msgs.append({"role": m["role"], "content": m["content"]})

        sources = None
        if req.use_kb:
            yield _sse("status", {"state": "retrieving"})
            if not STORE.chunks:
                hits = []  # legitimately no documents indexed yet, not an outage
            else:
                try:
                    with GPU_STAGE_LOCK:
                        hits = STORE.search(req.text, k=req.kb_k)
                except Exception as e:
                    logger.error(f"[chat/stream] KB search failed: {e}")
                    hits = None
            sources = [hit_to_source(h) for h in hits] if hits is not None else []
            yield _sse("sources", {"sources": sources})
            ground_last_user_turn(out_msgs, req.text, hits)

        # Residency = llama-server's /health (503 while the model loads).
        # If it isn't serving yet, announce the load and start it on demand
        # -- a cold 21 GB load takes minutes, hence the long status window.
        if not chat_server_health():
            yield _sse("status", {"state": "loading_model", "model": CHAT_MODEL_NAME})
            ok, err = ensure_chat_server()
            if not ok:
                yield _sse("error", {"message": f"chat model unavailable: {err}"})
                return

        # Context size is fixed server-side at CHAT_NUM_CTX (32768) when
        # llama-server launches; req.num_ctx only drives the frontend's
        # token monitor. KB turns keep the embedding model's VRAM free by
        # having DocStore.search() release it before we get here; with the
        # generator in a separate long-lived process there is no
        # keep_alive/unload step on this side anymore.
        body = {
            "model": CHAT_MODEL_ID,
            "messages": out_msgs,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if req.use_kb:
            body["chat_template_kwargs"] = {"enable_thinking": False}

        message_id = new_message_id()
        raw, stats, generating_announced, persisted = "", None, False, False
        usage = None
        first_token_at, last_token_at = None, None
        upstream = None
        try:
            upstream = requests.post(CHAT_LLM_HOST + "/v1/chat/completions", json=body,
                                     stream=True, timeout=1800)
            upstream.raise_for_status()
            for line in upstream.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    o = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if o.get("usage"):
                    usage = o["usage"]
                choices = o.get("choices") or []
                if not choices:
                    continue
                content = ((choices[0].get("delta") or {}).get("content")
                           or (choices[0].get("delta") or {}).get("reasoning_content"))
                if content:
                    now = time.time()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    if not generating_announced:
                        yield _sse("status", {"state": "generating"})
                        generating_announced = True
                    raw += content
                    yield _sse("token", {"text": content})
            if usage and usage.get("completion_tokens") and first_token_at and last_token_at:
                # Same {eval_count, eval_duration(ns)} shape the frontend's
                # tok/s readout consumed from Ollama, derived from llama-
                # server's usage + measured token window.
                stats = {"eval_count": usage["completion_tokens"],
                         "eval_duration": max(1e-6, last_token_at - first_token_at) * 1e9}
            thread["messages"].append({
                "id": message_id, "role": "assistant", "content": raw, "model": CHAT_MODEL_NAME,
                "sources": sources, "stats": stats, "created_at": _now(),
            })
            save_thread(thread)
            persisted = True
            yield _sse("done", {"message_id": message_id})
        except Exception as e:
            logger.error(f"[chat/stream] chat-model call failed: {e}")
            yield _sse("error", {"message": str(e)})
        finally:
            if upstream is not None:
                try:
                    upstream.close()
                except Exception:
                    pass
            # Covers both a mid-stream error above and a client abort (Stop
            # button closes the fetch, which throws GeneratorExit in here at
            # whatever yield this generator is suspended on) -- either way,
            # whatever was generated so far is worth keeping, same as the
            # old client-side AbortError handling.
            if not persisted and raw:
                thread["messages"].append({
                    "id": message_id, "role": "assistant", "content": raw, "model": CHAT_MODEL_NAME,
                    "sources": sources, "stats": None, "created_at": _now(), "stopped": True,
                })
                save_thread(thread)

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    print("=" * 70)
    print(f"[orchestrator] LOCAL AI ASSISTANT - HYBRID RAG")
    print(f"[orchestrator] Source dir: {CONFIG.paths.source_dir}")
    print(f"[orchestrator] Index dir: {CONFIG.paths.index_dir}")
    print(f"[orchestrator] Embedding host: {CONFIG.embedding.host}  model: {CONFIG.embedding.model}")
    print(f"[orchestrator] Listening on http://localhost:{PORT}")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=PORT)

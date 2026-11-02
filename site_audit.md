# Audit of /home/jrochau/Projects/site: facts for rewriting the project instructions

- **Audit date:** 2026-09-26
- **Scope:** read-only. All paths are relative to `/home/jrochau/Projects/site/` unless written in full.
- **Evidence format:** `file:line`, command output, or `NOT FOUND (searched: …)`.
- **Excluded from searches unless stated:** `.venv/`, `.venv-docling/`, `.venv-xberg/`, `__pycache__/`.

---

## 1. Layout

### 1.1 Directory tree (depth 3, with file counts)

There is no `data/` directory (`ls data`: "No such file or directory"). Data lives under `docs/`, `threads/`, `sessions/` and `state.json` (see 3.3).

```
/home/jrochau/Projects/site/                [20 files at top level]
├── app.css, index.html, document_converter.html          (frontend)
├── orchestrator.py, control.py, ingestion.py              (orchestrator + ranking + legacy rescan)
├── embedding_server.py, embedding_backend.py              (embedding service)
├── whisper_server.py                                      (speech-to-text service)
├── state.json, manifest.json
├── orchestrator.log (238,896,256 bytes), ingestion.log, orch_debug.log
├── runCommands, "commands and directories"                (plain-text notes, no extension)
├── HANDOFF.md, DELIVERABLES.md, RAG_BUILD_SOURCE_OF_TRUTH.md, TABLE_SPLIT_TOKEN_ACCOUNTING_ANALYSIS.md
├── .claude/                                [0 files]
├── artifacts/                              [1 file: session-20260629-123746.md]
│   ├── gpu_lifecycle_validation/           [4 .py acceptance scripts]
│   ├── heading_classifier_step1/           [16 files] + figures/ [70]
│   ├── layout_poc/                         [2 files]; adapter/ [1], environment/ [3], reports/ [7],
│   │                                       shadow_index/ [5], shadow_index_strict/ [8], source/ [8],
│   │                                       validation/ [25]; chunks/, comparator/, docling/, docling_v2/,
│   │                                       legacy/, normalized/, visualizations/, xberg/ [0 each]
│   ├── qwen_multimodal_validation/         [9 files] + logs/ [3]
│   └── title_override_validation/          [3 files]
├── docs/
│   ├── .index/                             [389 d-*.json] + _meta/ [title_overrides.json]
│   ├── .index-qwen3-4b-shadow/             [18 d-*.json]
│   ├── azure/                              [28 topic subdirs, 365 files total]
│   ├── networking/                         [6 .md]
│   └── newly_formatted_docs_review/
│       ├── sources/                        [220 files: 151 md, 63 pdf, 6 docx]
│       ├── processed/                      [451 files]
│       └── reports/                        [451 files]
├── document_converter/                     [35 files: pipeline package + 2 empty log files]
│   ├── extractors/                         [9 .py]
│   └── tests/                              [46 files: conftest.py, __init__.py, 44 test_*.py]
│       └── fixtures/                       [0 files]
├── sessions/                               [0 files]
└── threads/                                [9 t-*.json]
```

### 1.2 Python file count and lines of code

| Scope | .py files | Lines (wc -l) |
|---|---|---|
| Everything except venvs and `__pycache__` | 111 | 29,801 |
| Same, also excluding `artifacts/` | 94 | 27,228 |

Largest files: `orchestrator.py` (2,288 lines), `document_converter/extractors/pdf_extractor.py` (1,641), `document_converter/atomic_split.py` (1,004), `document_converter/validation.py` (984), `document_converter/pipeline.py` (947).

### 1.3 Entry points

These are files containing `if __name__ == "__main__"` or `uvicorn.run`.

| File:line | What it runs |
|---|---|
| `orchestrator.py:2280-2288` | `uvicorn.run(app, host="127.0.0.1", port=PORT)` |
| `embedding_server.py:371-372` | `app.run(host='127.0.0.1', port=PORT)` (Flask, not uvicorn) |
| `whisper_server.py:88-89` | `uvicorn.run(app, host="127.0.0.1", port=PORT)` |
| `ingestion.py:65` | `DocIngester().ingest()` (legacy rescan, run once) |
| `document_converter/cli.py:190` | CLI with `process`, `validate`, `preview`, `index`, `scan` and `rebuild` subcommands (`cli.py:144-173`) |
| `document_converter/split_ingest.py:623` | Pre-ingest PDF splitter |
| `document_converter/eval_retrieval.py:117` | Retrieval smoke evaluation |
| `artifacts/**` (13 files) | One-off validation/acceptance scripts: `gpu_lifecycle_validation/*` (4), `qwen_multimodal_validation/*` (3), `layout_poc/validation/*` (7 incl. `run_phase9_retrieval.py`), `layout_poc/adapter/docling_adapter.py` |

---

## 2. Runtime and services

### 2.1 Services

| Service | File | Host:port | Start command (per repo) | Depends on |
|---|---|---|---|---|
| Orchestrator (FastAPI) | `orchestrator.py` | `127.0.0.1:11436` (`orchestrator.py:120`, env `ORCH_PORT`; bind at `:2288`) | `python orchestrator.py` (`runCommands`, `HANDOFF.md:435`) | Embedding server (checked at `DocStore.reload`, `orchestrator.py:458-462`; used for search, upload and index); Ollama `http://127.0.0.1:11434` (hardcoded, `orchestrator.py:117`) for `/api/models`, `/chat/stream` and `/gpu/status`; `nvidia-smi` (optional, `gpu_lifecycle.py:29`) |
| Embedding server (Flask) | `embedding_server.py` | `127.0.0.1:11437` (`embedding_server.py:39`, env `EMBEDDING_PORT`; bind at `:372`) | `python embedding_server.py` (`runCommands`, `HANDOFF.md:434`) | sentence-transformers model loaded at import (`embedding_server.py:87`); torch/CUDA |
| Ollama | external, not in repo | `127.0.0.1:11434` (`orchestrator.py:117`; `config.py:78,194`) | `ollama serve` (`HANDOFF.md:61`) | — |
| Whisper STT (FastAPI) | `whisper_server.py` | `127.0.0.1:11435` (`whisper_server.py:35`, env `WHISPER_PORT`; bind at `:89`) | `python whisper_server.py` (`HANDOFF.md:436`) | faster-whisper, CPU by default (`whisper_server.py:33`) |
| Static frontend | `index.html`, `document_converter.html` | `127.0.0.1:8000` | `python -m http.server 8000 --bind 127.0.0.1` (`runCommands`) | Orchestrator; whisper (`index.html:221`); crawler (`index.html:418`) |
| webCrawler UI | not in this repo | `http://127.0.0.1:8420` (default URL field, `index.html:418`) | `python webui_server.py` from `../webCrawler/` (`HANDOFF.md:65,146,440`); nothing in this repo starts it | Loaded as a JS-created `<iframe>` (`index.html:830-843`) |

How this compares with the ports you expected:
- **Confirmed:** 11436, 11437, 11434, 8420.
- **Whisper server:** port **11435** (`whisper_server.py:35`).
- **Also present:** a static file server on port **8000** (`runCommands`).

### 2.2 Start scripts, Makefile, systemd units, README

| Item | Status |
|---|---|
| systemd unit files | NOT FOUND (searched `find` for `*.service`) |
| Shell scripts | NOT FOUND (`*.sh`) |
| Makefile | NOT FOUND |
| Dockerfile | NOT FOUND |
| `.env` files | NOT FOUND |
| README | NOT FOUND at repo root. There is only `artifacts/heading_classifier_step1/README.md`. |

Start instructions that do exist:
- `runCommands`, verbatim:
  ```
  cd ~/Projects/site
  source .venv/bin/activate
  python embedding_server.py
  python orchestrator.py
  python -m http.server 8000 --bind 127.0.0.1
  ```
- `HANDOFF.md:57-65` (services table), `HANDOFF.md:434-440` (start commands), `DELIVERABLES.md:68-86`.

### 2.3 `grep -rni "docker\|compose\|container"` (excluding venvs, `__pycache__`, `*.log`)

No hit is Docker/Compose infrastructure. Hits outside the document corpus:

| Location | What it is |
|---|---|
| `app.css:1622,1631` | "composer" CSS class |
| `index.html:248` | `class="composer"` |
| `index.html:368` | placeholder text "No Docker / Runs offline / Uses Ollama" |
| `document_converter.html:763-795` | JS variable named `container` |
| `commands and directories:18-19` | `touch azure/04_compute/configure_azure_container_*.md` |
| `document_converter/pdf_extractor.py:584`, `heading_repair.py:22,23,114`, `tests/test_structural_validation.py:3,68` | "container" as a heading/section concept |
| `artifacts/`: 6 files (`heading_classifier_step1/D4.md`, `D5.md`, `README.md`, `D1_all_docs.csv`, `layout_poc/adapter/docling_adapter.py`, `layout_poc/reports/phase5_adapter_mapping.md`) | Callout-container analysis and corpus titles |
| `docs/`: 239 files outside `docs/.index`, 102 files in `docs/.index` | Corpus content (e.g. `docs/networking/network_fundamentals.md:897-915` "Containers"/"Docker") |

---

## 3. Configuration

### 3.1 Environment variables read in code

Tests and `artifacts/` are excluded. Nine `artifacts/` scripts also read env vars (`grep -l`).

| Name | Default | file:line | Purpose |
|---|---|---|---|
| `ORCH_PORT` | `11436` | `orchestrator.py:120` | Orchestrator port |
| `EMBEDDING_PORT` | `11437` | `embedding_server.py:39` | Embedding server port |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | `embedding_server.py:38`; `config.py:62,443` | Model the server loads, and the model name the pipeline records and fingerprints |
| `EMBEDDING_SERVER_BATCH_SIZE` | falls back to `EMBEDDING_BATCH_SIZE`, then 2 for Qwen3-Embedding or 32 otherwise | `embedding_server.py:53-56`; `embedding_backend.py:89-95` | Encode sub-batch size inside one `/api/embed` call |
| `EMBEDDING_BATCH_SIZE` | `64` | `config.py:68,444` (also a fallback at `embedding_server.py:55`) | Client side: chunks per HTTP embed request (`pipeline.py:556`) |
| `EMBEDDING_HOST` | `http://127.0.0.1:11437` | `config.py:61,442` | Embedding service URL |
| `EMBEDDING_CLIENT_RETRIES` | `3` | `embedding_client.py:29` | Client retry count |
| `EMBEDDING_CLIENT_RETRY_BASE_DELAY` | `5.0` | `embedding_client.py:30` | Client retry backoff base (seconds) |
| `EMBEDDING_CHUNK_TARGET_TOKENS` | unset (`None`) | `config.py:123,452` | Token-native chunk target |
| `EMBEDDING_CHUNK_MAX_TOKENS` | unset (`None`) | `config.py:124,453` | Token ceiling, capped by the service's `max_sequence_tokens` (`chunker.py:238`) |
| `SEMANTIC_HOST` | `http://localhost:11434` | `config.py:78,445` | Ollama URL for semantic structuring |
| `SEMANTIC_MODEL` | `""` | `config.py:79,446` | Structuring model. Empty means disabled (`config.py:91-93`). |
| `SEMANTIC_MODE` | `auto` (field default, no env default) | `config.py:77,447` | `off`, `auto` or `always` |
| `SEMANTIC_BATCH_CHAR_BUDGET` | `24000` | `config.py:89,448` | Batch size cap (characters) for structuring |
| `CHUNK_HARD_MAX_WORDS` | `512` | `config.py:98,449` | Word hard max |
| `CHUNK_TARGET_WORDS` | `425` | `config.py:99,450` | Word target |
| `CHUNK_PARAGRAPH_WARN_WORDS` | `150` | `config.py:100,451` | Warning threshold for long paragraphs |
| `CHUNK_OVERSIZED_ROW_POLICY` | `split` | `config.py:150,454` | `split` or `error` for a table row too big for the budget |
| `OCR_MODE` | `auto` | `config.py:158,455` | `off`, `auto` or `on` (see 6.4) |
| `VISUAL_ANALYSIS_ENABLED` | `0` (disabled) | `config.py:193` | Turns on the visual-analysis pipeline (VLM) |
| `VISUAL_OLLAMA_HOST` | `http://127.0.0.1:11434` | `config.py:194` | Ollama URL for the VLM |
| `VISUAL_MODEL` | `qwen3-vl:8b-instruct-q4_K_M` | `config.py:197` | VLM model |
| `VISUAL_TIMEOUT_SECONDS` | `180` | `config.py:198` | VLM call timeout |
| `VISUAL_KEEP_ALIVE` | `5m` | `config.py:349` | Ollama `keep_alive` during a visual batch |
| `VISUAL_VRAM_BARRIER_TIMEOUT_S` | `45` | `config.py:362` | VRAM barrier timeout |
| `VISUAL_VRAM_BARRIER_TOLERANCE_MIB` | `1024` | `config.py:363` | VRAM barrier slack |
| `DOCPIPE_STRICT` | `1` (on) | `config.py:390,458` | Strict validation gate (see 6.3) |
| `DOCPIPE_PROFILE` | `main_text` | `config.py:394,459` | Ingestion profile (see 6.3) |
| `DOCPIPE_CONFIG` | unset | `config.py:473` | Path to a YAML/JSON config file |
| `UPLOAD_MAX_BYTES` | `52428800` | `config.py:391,456` | Defined only. It is never read outside `config.py` (searched `upload_max_bytes`). |
| `RAG_CONTEXT_BUDGET_TOKENS` | `3000` | `config.py:392,457` | Defined only. It is never read outside `config.py` (searched `retrieval_context_budget_tokens\|RAG_CONTEXT_BUDGET`). |
| `SOURCE_DIR`, `PROCESSED_DIR`, `REPORT_DIR`, `INDEX_DIR`, `MANIFEST_PATH` | see 3.3 | `config.py:437-441` | Path overrides |
| `PARSER` | `legacy` | `extractors/__init__.py:713` | `docling` puts DoclingExtractor ahead of PdfExtractor for `.pdf` |
| `WHISPER_MODEL` | `small.en` | `whisper_server.py:29` | Whisper model |
| `WHISPER_DEVICE` | `cpu` | `whisper_server.py:33` | Whisper device |
| `WHISPER_COMPUTE` | `int8` on CPU, `float16` otherwise | `whisper_server.py:34` | Whisper compute type |
| `WHISPER_PORT` | `11435` | `whisper_server.py:35` | Whisper port |
| `HF_HUB_OFFLINE` | set to `1` via `setdefault` | `tests/test_atomic_split.py:29` | Tests only: offline tokenizer load |

### 3.2 DOCPIPE_CONFIG and config files

Load order (`config.py:470-496`):
1. Dataclass defaults.
2. A config file: `DOCPIPE_CONFIG`, or else `REPO_ROOT/docpipe.yaml`, `docpipe.yml` or `docpipe.json`, where `REPO_ROOT` is `/home/jrochau/Projects/site` (`config.py:41,474-479`).
3. Environment variables from `_ENV_OVERRIDES`, which always win.

How the file is applied:
- **Format:** YAML (needs PyYAML) or JSON (`config.py:414-422`). Nested sections `paths`, `embedding`, `semantic`, `chunking`, `ocr` and `visual` are applied (`config.py:500-502`).
- **Top-level keys actually applied:** only `strict`, `upload_max_bytes`, `retrieval_context_budget_tokens` and `delete_source_on_doc_delete` (`config.py:503-505`). A top-level `profile` key in the file is **not** applied.
- **Side effect:** `load_config()` calls `cfg.paths.ensure()`, which creates the source, processed, report and index directories (`config.py:53-55,495`). `DEFAULT_CONFIG = load_config()` runs at import time (`config.py:508`).

Config files present in the repo: NOT FOUND (searched `docpipe.*` at the repo root and in `document_converter/`, plus `find` for `*.yaml`, `*.yml`, `*.toml`, `*.ini` and `*.cfg`).

### 3.3 Data, DB and index paths

There is no SQLite DB. NOT FOUND (searched `sqlite|fts5|fts4|faiss|chromadb|lancedb|qdrant|hnsw` in `*.py`; `find` for `*.db` and `*.sqlite*`). The only SQLite use is a throwaway `/tmp` FTS5 test in `artifacts/layout_poc/validation/run_phase9_retrieval.py:4,44,91`.

| Path | Read/written by | Evidence |
|---|---|---|
| `docs/newly_formatted_docs_review/sources/` | Uploads are written here; hash-dedupe scans it | `config.py:46`; `orchestrator.py:1480-1526,381-405` |
| `docs/newly_formatted_docs_review/processed/<doc_id>.md` | Rendered markdown | `config.py:47`; `pipeline.py:443` |
| `docs/newly_formatted_docs_review/reports/<doc_id>.report.md` | Validation report | `config.py:48`; `pipeline.py:444` |
| `docs/.index/<doc_id>.json` | Index (JSON, schema v2), written atomically via `.<id>.json.tmp` + `os.replace` | `config.py:49`; `index_schema.py:385-394` |
| `docs/.index/_meta/title_overrides.json` | Manual title overrides | `title_overrides.py:39,49,53` |
| `docs/.index/_visuals/<hash>.png` and `<hash>__<tag>__v<ver>.json` | VLM crops and analysis cache. Directory does not currently exist. | `visual_extraction.py:663-665`; `visual_cache.py:36-37` |
| `docs/` (legacy root) | `/ingest` and `ingestion.py` scan every supported file here, skipping managed dirs | `config.py:51`; `pipeline.py:904-930`; `ingestion.py:42-44` |
| `docs/.index-qwen3-4b-shadow/` | Shadow index. Not referenced by any non-artifact code path (default `index_dir` is `docs/.index`). | `config.py:49` |
| `manifest.json` | `paths.manifest_path`. No reader or writer outside `config.py` (searched `manifest_path`). | `config.py:50,441` |
| `state.json` | `/state` GET/PUT and `/session/end` | `orchestrator.py:121,176-192` |
| `threads/t-*.json` | Thread store | `orchestrator.py:124,203-277` |
| `sessions/session-<ts>.json` | Raw `/session/end` transcripts | `orchestrator.py:122,1625` |
| `artifacts/session-<ts>.md` | `/session/end` summaries | `orchestrator.py:123,1662` |
| `local_agent_dirs.json` | localAgent grants. File not currently present. | `orchestrator.py:151-167` |
| Temp dirs `docpreview-*` (system tmp) and `docupload-*` (in `source_dir`) | Upload staging | `orchestrator.py:1437,1486` |
| `.venv-docling/bin/docling` | Docling CLI subprocess | `extractors/docling_extractor.py:68` |

---

## 4. API surface

The embedding server uses Flask (`embedding_server.py:27,214`). The other two servers use FastAPI.

### 4.1 Orchestrator (`orchestrator.py`, :11436)

Swagger UI is at `/api-docs` (`orchestrator.py:817`).

| Method | Path | Handler | file:line | Purpose |
|---|---|---|---|---|
| GET | `/health` | `health` | orchestrator.py:867 | ok flag, base dir, chunk and document counts |
| GET | `/api/models` | `api_models` | :874 | Proxies Ollama `/api/tags`, filtered by `is_generation_model` |
| GET | `/gpu/status` | `gpu_status` | :899 | nvidia-smi memory/processes, Ollama `/api/ps`, embedding `/health` |
| GET | `/state` | `get_state` | :929 | Reads `state.json` |
| PUT | `/state` | `put_state` | :935 | Merges known keys into `state.json` |
| POST | `/ingest` | `ingest` | :945 | Legacy `docs/` rescan via `DocIngester`, under `GPU_STAGE_LOCK` |
| GET | `/docs` | `docs_list` | :964 | Document list plus supported extensions |
| GET | `/docs/jobs` | `docs_jobs_list` | :970 | All job records |
| GET | `/docs/jobs/{job_id}` | `docs_job_get` | :976 | One job snapshot |
| GET | `/docs/jobs/{job_id}/stream` | `docs_job_stream` | :988 | SSE progress (0.5 s poll, 15 s keepalive) |
| GET | `/docs/{doc_id}` | `docs_detail` | :1037 | Full index payload |
| GET | `/docs/{doc_id}/raw` | `docs_raw` | :1088 | Original source bytes (path-confined to `source_dir`) |
| GET | `/docs/{doc_id}/markdown` | `docs_markdown` | :1113 | Rendered .md (confined to `processed_dir`) |
| POST | `/docs/search` | `docs_search` | :1137 | Hybrid search, `k` defaults to 4 (`:827`) |
| POST | `/docs/preview` | `docs_preview` | :1416 | Background preview job, returns 202 |
| POST | `/docs/upload` | `docs_upload` | :1457 | Background index job, returns 202; optional `title_override` form field |
| PUT | `/docs/{doc_id}/title` | `docs_set_title` | :1539 | Sets title override and reindexes (sync) |
| DELETE | `/docs/{doc_id}/title` | `docs_clear_title` | :1574 | Clears override and reindexes |
| DELETE | `/docs/{doc_id}` | `docs_delete` | :1604 | Removes index/.md/report (source kept: `delete_source_on_doc_delete=False`, `config.py:393`) |
| POST | `/session/end` | `session_end` | :1616 | Summarizes the chat via Ollama and merges deltas into `state.json` |
| GET | `/threads` | `threads_list` | :1682 | Thread summaries |
| POST | `/threads` | `threads_create` | :1687 | New thread |
| GET | `/threads/{thread_id}` | `threads_get` | :1692 | Full thread |
| PUT | `/threads/{thread_id}` | `threads_rename` | :1703 | Rename |
| DELETE | `/threads/{thread_id}` | `threads_delete` | :1717 | Delete |
| GET | `/localagent/directories` | `localagent_dirs_list` | :1836 | Granted dirs |
| POST | `/localagent/directories` | `localagent_dirs_add` | :1841 | Add a dir (absolute path; read/write/execute all default off) |
| PATCH | `/localagent/directories/{dir_id}` | `localagent_dirs_patch` | :1866 | Toggle read/write/execute |
| DELETE | `/localagent/directories/{dir_id}` | `localagent_dirs_delete` | :1882 | Remove grant |
| POST | `/localagent/crawl/stream` | `localagent_crawl_stream` | :1927 | SSE tree of a granted dir (max depth 6, max 2000 entries, `:1800-1801`) |
| POST | `/localagent/write/stream` | `localagent_write_stream` | :1952 | Creates a subdir or a `.md`/`.txt` doc |
| POST | `/localagent/exec/stream` | `localagent_exec_stream` | :2049 | Proposes the newest installer-type file, waits for confirmation, runs it (600 s timeout) |
| POST | `/localagent/exec/{task_id}/confirm` | `localagent_exec_confirm` | :2133 | Human yes/no |
| POST | `/localagent/exec/{task_id}/interrupt` | `localagent_exec_interrupt` | :2150 | Cancel |
| POST | `/chat/stream` | `chat_stream` | :2163 | SSE chat: retrieve, emit sources, stream from Ollama, persist to thread |

### 4.2 Embedding server (`embedding_server.py`, :11437)

| Method | Path | Handler | file:line | Purpose |
|---|---|---|---|---|
| GET | `/health` | `health` | embedding_server.py:235 | Model id, revision, dimension, max_sequence_tokens, device |
| POST | `/api/release` | `release` | :255 | Moves the model off CUDA and empties the cache |
| POST | `/api/tokenize` | `tokenize` | :270 | Token counts |
| POST | `/api/embed` | `embed` | :282 | Embeds `input` with `input_type` document/query. Rejects oversized inputs with 413 by default (`:303-318`). Retries on OOM by halving the batch (`:149-183`). |

### 4.3 Whisper server (`whisper_server.py`, :11435)

| Method | Path | Handler | file:line | Purpose |
|---|---|---|---|---|
| GET | `/health` | `health` | whisper_server.py:54 | Model and device |
| POST | `/transcribe` | `transcribe` | :59 | Uploaded audio in, text out (beam 5, VAD filter) |

---

## 5. Models

### 5.1 Model names in code and config

| Role | Name | file:line |
|---|---|---|
| Embedding (default) | `BAAI/bge-large-en-v1.5` | `config.py:62`; `embedding_server.py:38`; `tests/test_atomic_split.py:31` |
| Embedding (alternate family) | `Qwen/Qwen3-Embedding*` (prefix match); example `Qwen/Qwen3-Embedding-4B` | `embedding_backend.py:18,21` |
| Visual (VLM, Ollama) | `qwen3-vl:8b-instruct-q4_K_M` | `config.py:197` |
| Semantic structuring (Ollama) | `""` (none, so disabled) | `config.py:79,93` |
| Generation (Ollama) | Not configured in code. Chosen by the user from `/api/models` (see 5.2). `qwen3:14b` appears only in comments and docstrings. | comments at `orchestrator.py:629`, `gpu_lifecycle.py:107`, `embedding_backend.py:66`, `embedding_server.py:100`, `whisper_server.py:17`, test `test_gpu_lifecycle.py:125`; recorded in `threads/*.json` (`"model": "qwen3:14b"`) |
| Whisper | `small.en` (default); `base.en` and `medium.en` mentioned in the docstring | `whisper_server.py:29`; `:16` |
| `/session/end` summarizer | `req.model`, from the client | `orchestrator.py:1648-1657` |

### 5.2 How the generation model is selected

1. `GET /api/models` fetches Ollama `/api/tags` and keeps names where `is_generation_model(name)` is true (`orchestrator.py:884-896`).
2. `is_generation_model` excludes an empty name, excludes `CONFIG.visual.model`, and excludes names matching the regex `vision|vl|llava|scout|clip|embed|bge|nomic|gte-|e5-|minilm` (case-insensitive) (`orchestrator.py:801-810`). Every returned model has `available: True` (`:892`).
3. `index.html` fills `#modelSelect` from that list. It restores the saved choice from browser storage key `model` if still present, otherwise picks the first option (`index.html:508-527`), and saves on change (`index.html:1676`).
4. The chosen value is sent as `model` in the `/chat/stream` body (`index.html:1518,1545-1552`). The orchestrator passes it straight to Ollama `/api/chat` (`orchestrator.py:2223,2231`).

`state.json` stores no model.

### 5.3 `document_converter/gpu_lifecycle.py` (198 lines)

It has no torch dependency and uses best-effort calls that never raise (`gpu_lifecycle.py:1-15`).

- **`read_gpu_memory_used_mib`** (`:36-55`): runs `nvidia-smi --query-gpu=memory.used`. Returns `None` if unavailable.
- **`list_gpu_compute_processes`** (`:58-85`): runs `nvidia-smi --query-compute-apps`. Reports external GPU users; does not manage them.
- **`ollama_ps` / `is_ollama_model_loaded`** (`:88-100`): Ollama `/api/ps`.
- **`unload_ollama_model`** (`:103-118`): POSTs `/api/generate` with `{"model", "prompt": "", "keep_alive": 0}`.
- **`wait_for_ollama_model_unload`** (`:121-134`): polls `/api/ps` every 1 s until the model is gone or the timeout (default 30 s) passes.
- **`wait_for_vram_release`** (`:160-198`), the VRAM barrier: polls `nvidia-smi` every 1 s until `used <= baseline + tolerance` (default tolerance 512 MiB, timeout 45 s). If the baseline is `None` or nvidia-smi is missing, it returns immediately with `measured=False, released=False`. Result type is `VramBarrierResult` (`:137-157`).

Where it is used:

**Ingestion** (`pipeline.py`), when visual analysis is enabled:
1. Baseline memory is read before the VLM runs (`:713`).
2. After the visual batch, in a `finally` block, the VLM is unloaded (`visual_analysis.unload_model`), then the code waits for Ollama to drop it, then runs the VRAM barrier. Timeout and tolerance come from `VISUAL_VRAM_BARRIER_*`, defaults 45 s and 1024 MiB (`:768-783`).
3. After embedding: `client.release()` is called (embedding `/api/release`), then a VRAM barrier runs against the embedding baseline taken at `:552` (`:611-622`).

**Query path** (`orchestrator.py` `DocStore.search`):
1. Reads baseline memory, embeds the query, calls `embedding_client.release()`.
2. Runs the barrier with tolerance 1024 MiB and timeout 10 s. It only logs a warning if release isn't confirmed (`:634-644`).

**keep_alive handling:**
- VLM calls send `keep_alive = VISUAL_KEEP_ALIVE` (default `5m`) (`visual_analysis.py:198`; `config.py:349`).
- `/chat/stream` sends `keep_alive: 0` only when `use_kb` is true (`orchestrator.py:2223-2225`). When `use_kb` is false, `keep_alive` is omitted (Ollama's default applies).
- `/chat/stream` emits `status: loading_model` when `is_ollama_model_loaded` is false (`orchestrator.py:2211-2216`).

**Locking:** `GPU_STAGE_LOCK` is a process-local `threading.Lock` (`orchestrator.py:66`). It is held for `/ingest`, `/docs/search`, index jobs, title set/clear and the KB search inside `/chat/stream`. It is **not** held during generation (`orchestrator.py:58-63,2202-2203`).

---

## 6. Ingestion pipeline (docpipe)

### 6.1 Formats and extractors

`SUPPORTED_EXTENSIONS = {.pdf, .docx, .doc, .html, .htm, .txt, .md, .markdown}` (`extractors/__init__.py:727`). Extractor selection is first-match over the order in `get_extractors()` (`extractors/__init__.py:697-724`).

| Extension | Module / class | Library | Evidence |
|---|---|---|---|
| `.md`, `.markdown` | `markdown_extractor.MarkdownExtractor` → `md_parser.parse_markdown` | none (regex) | `markdown_extractor.py:13-22` |
| `.txt` (also `.text`, but `.text` is not in `SUPPORTED_EXTENSIONS`) | `txt_extractor.TxtExtractor` | — | `txt_extractor.py:24-25` |
| `.html`, `.htm` | `html_extractor.HtmlExtractor` | BeautifulSoup | `html_extractor.py:7,37-38` |
| `.docx` | `docx_extractor.DocxExtractor` | python-docx | `docx_extractor.py:8-11,107-108` |
| `.doc` | `doc_extractor.DocExtractor` | converts via a `soffice`/`libreoffice` subprocess | `doc_extractor.py:30-61` |
| `.pdf` (default) | `pdf_extractor.PdfExtractor` | pymupdf; pytesseract for OCR | `pdf_extractor.py:922-928` |
| `.pdf` when `PARSER=docling` | `docling_extractor.DoclingExtractor` | subprocess to `.venv-docling/bin/docling` | `extractors/__init__.py:713-715`; `docling_extractor.py:8-11,68` |

### 6.2 Stage order

The core is `DocumentPipeline._build_document` (`pipeline.py:185-268`), followed by `_prepare_for_profile` (`:274-379`) and then `index()` (`:453-643`).

| # | Stage (progress label) | Function | pipeline.py line |
|---|---|---|---|
| 1 | extracting | `select_extractor(path).extract(...)`. On exception it logs an `extraction_failed` error and returns an empty doc. | 189-210 |
| 2 | normalizing | `normalize_document` | 211-213 |
| 3 | repairing_running_headers | `repair_running_headers` | 214-217 |
| 4 | structuring (only if semantic is enabled) | `apply_semantic_structuring` | 218-221 |
| 5 | repairing_headings | `repair_bodyless_heading_chains` | 222-225 |
| 6 | demoting_false_headings | `demote_false_headings` | 252-255 |
| 7 | auditing_heading_siblings | `audit_numbered_sibling_headings` | 256-259 |
| 8 | auditing_section_numbering | `audit_repeated_top_level_numbering` | 260-263 |
| 9 | repairing_paragraph_continuity | `repair_interrupted_paragraphs` | 264-267 |
| 10 | (profile view) | MAIN_TEXT: `build_main_text_document`, `resolve_main_text_title` or the manual override, then `apply_resolved_title` | 297-349 |
| 11 | chunking | `MarkdownChunker().chunk` | 362-366 |
| 12 | validating | `validate`, plus the MAIN_TEXT readiness overlay (`evaluate_main_text_readiness`, `apply_main_text_overlay`) | 367-377 |
| 13 | rendering | `render_markdown` | 400-401 |
| 14 | (process) | writes `processed/<id>.md` and `reports/<id>.report.md` | 435-451 |
| 15 | (index) | re-chunks and re-validates with the live embedding tokenizer and `max_sequence_tokens`, re-renders, and rewrites the .md | 477-511 |
| 16 | (strict gate) | raises `PipelineValidationError` if `config.strict` and the document is blocked | 513-518 |
| 17 | visual_analysis (PDF, MAIN_TEXT, visual enabled) | `_analyze_visuals`, then VLM unload and barrier | 539-544, 645-785 |
| 18 | embedding | `client.embed_batch` in slices of `embedding.batch_size`. Oversized chunks produce `chunk_exceeds_token_limit` and a raise. The chunk count must equal the embedding count. | 552-596 |
| 19 | embedding (visual) | `_embed_visuals` (best-effort) | 604-609, 787-845 |
| 20 | (release) | `client.release()` and VRAM barrier | 611-622 |
| 21 | writing_index | `build_index_payload`, then `write_index_atomic` | 624-639 |
| 22 | done | — | 642 |

`index()` skips all work if the index exists and `is_up_to_date(source_hash, config_fingerprint)` is true, unless `force=True` (`pipeline.py:461-473`; `index_schema.py:406-411`).

### 6.3 Strict mode and profiles

**Strict mode:**
- `config.strict` defaults on (`DOCPIPE_STRICT`, `config.py:390`). It is enforced in exactly one place, `pipeline.py:517-518`, inside `index()` only. `preview()` and `process()` never raise on validation.
- Blocking condition (`pipeline.py:513-516`):
  - **`main_text`:** the document is blocked when `not validation_report.retrieval_ready`. `retrieval_ready` is false if any of these hold (`retrieval_readiness.py:100-117`):
    - any ERROR finding whose code is not in `MAIN_TEXT_EXEMPT_ERROR_CODES` (`retrieval_readiness.py:45-66`: `author_metadata_interleaved_with_body`, `publication_metadata_interleaved_with_body`, `paragraph_interrupted_by_front_matter`, `invalid_document_title`, `root_title_not_rendered_as_h1`, `abstract_promoted_to_h1`, `empty_section`, `excessive_tiny_sections`, `excessive_suspicious_headings`, `caption_promoted_to_heading`, `page_number_promoted_to_heading`, `unsupported_ocr_heading`)
    - no resolved title
    - no main body content
    - zero chunks
    - any empty chunk
  - **`full_fidelity`:** the document is blocked when `findings.has_errors()`, i.e. any ERROR.
- Error codes named as still blocking (`retrieval_readiness.py:36-44`): `chunk_exceeds_token_limit`, `missing_source_block`, `duplicate_source_block`, and code/equation/fragment heading errors. Others observed in code: `extraction_failed` (`pipeline.py:199`), `unclosed_fence` (`md_parser.py:147`), `ocr_unavailable` (`pdf_extractor.py:1011`).
- In `main_text`, chunking always runs with `strict=False` (`pipeline.py:312`).
- `/docs/upload` failures are recorded in the job with the full validation payload (`orchestrator.py:1376-1388`). Title endpoints return 422 (`:1563-1568`, `:1593-1598`).

**Profiles** (`config.py:366-379`; `DOCPIPE_PROFILE`; an unknown value falls back to `main_text`, `pipeline.py:270-272`):

| Profile | What it does |
|---|---|
| `main_text` (default) | Chunks the filtered retrieval view (`retrieval_view.build_main_text_document`), which drops front-matter/metadata block types (`retrieval_view.py:43-47`). Title comes from `resolve_main_text_title` and a synthesized H1. Non-strict chunking. Readiness overlay gate. Visual analysis is allowed only in this profile (`pipeline.py:539`). |
| `full_fidelity` | Indexes the complete document. Title comes from `document.metadata.title` or the override. Configured chunking (strict as configured). Gate is any ERROR. |

### 6.4 OCR and visual analysis

**OCR** (PDF only, `pdf_extractor.py:977-1015`):
- A page is OCR'd when its `page.get_text().strip()` has fewer than `min_text_chars_per_page` (40) characters (`config.py:160`).
- `OCR_MODE=off`: the page is skipped with warning `page_looks_scanned_ocr_disabled`.
- Any other mode value (`auto` or `on`) OCRs the page: render at 200 dpi (`config.py:161`), then pytesseract with language `eng`. The code only checks for `mode == "off"` (`pdf_extractor.py:981`); `on` and `auto` behave the same (searched `ocr_config.mode`).
- If tesseract is missing: `ocr_unavailable` ERROR (`ocr.py:29-37`; `pdf_extractor.py:1010-1013`).

**Visual analysis:**
- Off by default (`VISUAL_ANALYSIS_ENABLED=0`, `config.py:193`). Runs only for `.pdf` under `main_text` inside `index()` (`pipeline.py:539`).
- Raster eligibility: at least 0.6% of page area and at least 36 pt per side (`config.py:203-204`). Vector-cluster rules are at `config.py:224-342`. At most 24 visuals per document (`config.py:210`).
- Results are cached by image hash + model + `analysis_version` "2" (`config.py:182`; `pipeline.py:722`). Records are appended to `chunks` with `content_kind: "visual_analysis"` (`index_schema.py:231`).

---

## 7. Retrieval

### 7.1 Index schema (`document_converter/index_schema.py`)

- **Storage:** one JSON file per document, `docs/.index/<doc_id>.json`, `schema_version` 2 (`config.py:38`; `index_schema.py:245`).
- **Tables / FTS5:** NOT FOUND (searched `sqlite|fts5|fts4`).
- **Vector storage:** each chunk dict holds an `"embedding"` float list inline (`index_schema.py:222`). There is no ANN index. Vectors are held in memory by `DocStore.chunks` (`orchestrator.py:509-541`).

Payload keys (`index_schema.py:244-315`):
- `document`: `document_id`, `title`, `title_source`, `source_path`, `markdown_path`, `report_path`, `source_hash`, `normalized_hash`, `created_at`, `updated_at`, `profile`
- `pipeline`: `pipeline_version`, `configuration_hash`, `extractor`, `ocr_used`, `semantic_structuring_used`, `chunker_version`
- `embedding`: `provider`, `model`, `revision`, `dimension`, `max_sequence_tokens`, `normalized_embeddings`, `query_prompt_name`, `tokenizer_id`, `chunk_target_tokens`, `chunk_max_tokens`, `config_fingerprint`
- `validation`: `retrieval_ready`, `warnings`, `errors`, `findings`, `stats`
- `visual`: counts
- `chunks[]`: each has `content_kind`, `chunk_id`, `chunk_index`, `header_path`, `heading`, `provenance`, `text`, `body`, `word_count`, `token_count`, `source_block_ids`, `source_pages`, `content_types`, `forced_fallback`, `split_provenance`, `embedding`

At load time, `DocStore.reload` skips:
- legacy-schema files (`orchestrator.py:474-476`)
- files that fail `check_embedding_compatibility` against the live embedding `/health` (`orchestrator.py:478-487`; `index_schema.py:75`)
- chunks without an embedding (`orchestrator.py:511`)

### 7.2 Ranking formula (`control.py`, `HybridRanker.rank`)

Called from `DocStore.search(query, k=4, alpha=0.7)` (`orchestrator.py:608,646-647`).

1. **Query cleaning:** remove punctuation, then lowercase (`control.py:15-17,138`).
2. **Cosine similarity** between the query embedding and **every** chunk embedding (0 if a chunk has none). Sort and keep the top 50 (`control.py:149-155`).
3. **BM25** over those 50 chunks only:
   - `k1 = 1.5`, `b = 0.75` (`control.py:33-34`)
   - `idf = ln(1 + (N - df + 0.5)/(df + 0.5))`, with N = 50 (`control.py:43-44`)
   - chunk tokens are `text.lower().split()` (`control.py:161`); query tokens are the cleaned query split on whitespace (`control.py:162`)
4. **Normalization:** `bm25_norm = bm25 / max(bm25 over the 50)`, using 1 if the max is 0 (`control.py:167-175`). The cosine score is used raw, not normalized (`control.py:174`).
5. **Score:** `hybrid = alpha * cosine + (1 - alpha) * bm25_norm`, i.e. `0.7*cos + 0.3*bm25_norm` (`control.py:176`). Sort descending and return the top `k` (`control.py:185-194`).
6. **Linked-context expansion:** after ranking, up to `MAX_LINKED_CONTEXT = 4` extra linked visual or main-text chunks are appended. Each carries the linking hit's score and `linked_from` (`orchestrator.py:577,652-675`).

`k` sources: `/docs/search` uses `SearchReq.k` (default 4, `orchestrator.py:827`). `/chat/stream` uses `kb_k` (default 4, `orchestrator.py:862`) from the UI `#kbK` field (`index.html:315,1545`).

### 7.3 Relevance buckets

These live in `orchestrator.py:709-720`:
- `RELEVANCE_HIGH = 0.55`, `RELEVANCE_MEDIUM = 0.35`
- `score >= 0.55` → `high`; `>= 0.35` → `medium`; otherwise (or `None`) → `low`

`relevance_bucket` is not called anywhere in `orchestrator.py` (only its definition matches). Scores go to the client raw in `hit_to_source` (`:733`).

### 7.4 How retrieved context reaches the model

`/chat/stream` (`orchestrator.py:2163-2277`):
1. Builds `out_msgs`: optional `system_prompt` as a system message, optional `project_context` as a second system message, then **every message in the thread** (`:2187-2193`).
2. If `use_kb`: emits `status: retrieving`, then runs `STORE.search(req.text, k=kb_k)` under `GPU_STAGE_LOCK`. If nothing is indexed the hit list is `[]`; if the search throws it is `None` (`:2196-2206`).
3. Builds sources with `hit_to_source` (`:723-741`, fields `id`, `title`, `type="document"`, full chunk text as `content`, `score`, `document_id`, `metadata.page/chunk_index/retrieved_at`) and emits the `sources` SSE event **before** generation (`:2207-2208`).
4. `ground_last_user_turn(out_msgs, question, hits)` (`:757-791`) rewrites the last user message in place:
   - `None` → "[Knowledge base unreachable…]" plus the question
   - `[]` → "returned no matching sources… Do NOT answer from outside knowledge"
   - otherwise → an instruction block (answer only from sources, cite inline as `[Source N]`, follow numbered-procedure order), then `=== SOURCES ===` with `[Source i] <_src_label>\n<full chunk text>` joined by `---`, then `=== QUESTION ===`
   - labels come from `_src_label` (`:744-754`)
5. Posts to Ollama `/api/chat` with `stream: true`, `options.num_ctx` (default 8192, `:860`), and `keep_alive: 0` if `use_kb` (`:2223-2231`). Streams `token` events and saves the assistant message with `sources` and `stats` to the thread. A partial answer is saved with `stopped: True` if the client aborts (`:2249-2275`).

Context budget: `RAG_CONTEXT_BUDGET_TOKENS` (default 3000) is **not applied**. Hits are joined untruncated (`orchestrator.py:780`), and the config value has no reader (see 3.1).

### 7.5 Agent routing, intent classification, multi-agent dispatch

NOT FOUND. Searched `intent|classif(y|ier)_?(query|intent)|agent|router|dispatch` in non-test `.py` and `.html`, plus the per-file lists for `route`, `intent` and `agent`.

What the hits actually are:
- **`route`:** FastAPI/Flask route decorators; the word "routers" in an eval question (`eval_retrieval.py:47`); comments.
- **`intent`:** the word "intentional" in comments, and one UI label, "Knowledge base — indexed by question intent" (`index.html:301`). No code implements intent indexing.
- **`agent`:** only **localAgent**, a user-driven filesystem tool with crawl/write/exec endpoints (`orchestrator.py:143-170,1730-2160`) and the `index.html` panel (`index.html:191,428-433,845+`). It is invoked by UI buttons, not by a model. Execution candidates are picked by a fixed rule, not a model call (`orchestrator.py:2002-2024`).
- **`dispatch`:** `embedding_backend.py:48` (model-id dispatch for load kwargs).

Every chat request follows the single path in 7.4.

---

## 8. Frontend

### 8.1 `index.html` (Knowledge Assistant, 86,758 bytes)

Sidebar modes (`index.html:173-191`): New Chat (`chat`), Knowledge Base (`kb`), Settings (`settings`), webCrawler (`crawler`, with an enable toggle), localAgent (`localagent`, with an enable toggle). Other UI:
- history list (`#historyToggle`, `:206`)
- model selector (`:200`)
- sources panel (`#sourcesPanel`, `:439`)
- mic dictation (`#micBtn`, `:261`)
- "think" collapsible (`:559`)
- link to `document_converter.html` ("Manage Documents")

| Feature | Endpoint called | index.html line |
|---|---|---|
| Model list | GET `/api/models` | 511 |
| Threads | POST/GET `/threads`, GET/PUT `/threads/{id}` | 734, 743, 759, 785, 797 |
| localAgent dirs | GET/POST `/localagent/directories`, PATCH/DELETE `/localagent/directories/{id}` | 921, 963, 975, 984 |
| localAgent ops | POST `/localagent/crawl/stream`, `/write/stream`, `/exec/stream` (via `fetch(orchHost()+url)`) | 1043 (urls at 1074, 1088, 1098) |
| localAgent confirm/interrupt | POST `/localagent/exec/{id}/confirm`, `/interrupt` | 1128, 1144 |
| Project state | GET/PUT `/state` | 1238, 1263 |
| KB list | GET `/docs` | 1291 |
| Document viewer | GET `/docs/{id}/raw`, `/docs/{id}/markdown` (via `fetchDocOk`) | 1348, 1357, 1370 |
| KB upload | POST `/docs/upload` | 1423 |
| KB delete | DELETE `/docs/{id}` | 1437 |
| End session | POST `/session/end` | 1490 |
| Chat | POST `/chat/stream` (SSE via fetch reader) | 1549 |
| Speech-to-text | POST `{whisper}/transcribe` | 1649 |
| webCrawler | JS-created `<iframe src="{crawlerHost}/">`, no fetch | 830-843 |

`index.html` never calls Ollama directly. NOT FOUND: `11434`, `/api/chat`, `keep_alive` in `index.html`.

### 8.2 `document_converter.html` (Converter / library, 45,884 bytes)

Sections: "Convert a document to markdown" (`:141-142`), "Knowledge base" (`:152-154`), "Test search" (`:169-170`), and a details view with Document, Extraction/Conversion, Chunking and Findings (`:259-306`).

| Feature | Endpoint | line |
|---|---|---|
| Health | GET `/health` | 353 |
| Doc list | GET `/docs` | 370 |
| Doc details | GET `/docs/{id}` | 402 |
| Title edit/clear | PUT/DELETE `/docs/{id}/title` | 430, 446 |
| Delete | DELETE `/docs/{id}` | 581 |
| Job progress | EventSource `/docs/jobs/{id}/stream`, falling back to polling GET `/docs/jobs/{id}` | 681, 667 |
| Upload | POST `/docs/upload` | 846 |
| Preview | POST `/docs/preview` | 883 |
| Test search | POST `/docs/search` | 902 |
| Legacy rescan | POST `/ingest` | 934 |

### 8.3 `app.css`

41,462 bytes of shared styles for both pages (`HANDOFF.md:312-313`). No endpoints.

### 8.4 Routes no frontend calls

Searched both HTML files.
- GET `/gpu/status` (its docstring says so: `orchestrator.py:908`)
- GET `/docs/jobs` (the list)
- DELETE `/threads/{id}`
- GET `/health` from `index.html`; only the comment at `:832` mentions it

---

## 9. Monitoring and logging

**Prometheus/Grafana/metrics:** NOT FOUND (searched `prometheus|grafana|/metrics|metrics` in non-test `.py`, `.html` and `.md` outside `docs/` and `artifacts/`). Hits are unrelated uses of the word "metrics": `control.py:218` `RankingMetrics` (MRR/nDCG/P@k helpers) and docstrings saying "metrics" in `heading_repair.py:142`, `heading_sibling_audit.py:76`, `section_numbering_audit.py:86`, `paragraph_continuity_repair.py:82`, `text_repair.py:118` and `validation.py:355`. `state.json`'s `last_summary` mentions Prometheus and Grafana, but that is saved chat-session text, not configuration.

Logging:

| Component | Config | Destination |
|---|---|---|
| orchestrator | `logging.basicConfig(level=DEBUG, …)` with `FileHandler('orchestrator.log')` and `StreamHandler` (`orchestrator.py:30-37`) | `orchestrator.log`, relative to the **working directory** of the process. Current `/home/jrochau/Projects/site/orchestrator.log` is 238,896,256 bytes, last modified Sep 26 12:14. |
| ingestion.py | `basicConfig(level=INFO)` with `FileHandler('ingestion.log')` (`ingestion.py:21-28`). This is a no-op for the root logger when imported after the orchestrator's `basicConfig`. | `ingestion.log` (cwd) |
| embedding_server | `basicConfig(level=INFO)`, stream only (`embedding_server.py:35`) | stdout/stderr |
| whisper_server | `print(...)` only (`whisper_server.py:37-43,66,79`) | stdout |
| Library modules | `logging.getLogger(__name__)` (`pipeline.py:56`, `atomic_split.py:36`, `gpu_lifecycle.py:27`) | inherit root |
| control.py | `print` debug output of the ranking (`control.py:141,158,188-192`); `debug=True` is passed on every search (`orchestrator.py:647`) | stdout |

Other log files present:
- `/home/jrochau/Projects/site/ingestion.log` (305,577 bytes, Jul 15)
- `/home/jrochau/Projects/site/orch_debug.log` (352 bytes, Jul 15; no code writes this name)
- `/home/jrochau/Projects/site/document_converter/orchestrator.log` and `/home/jrochau/Projects/site/document_converter/ingestion.log` (0 bytes each, Sep 23)

---

## 10. Tests

- **Test files:** 44 `test_*.py` in `document_converter/tests/`, plus `conftest.py` and `__init__.py`.
- **Test functions:** 546 (sum of `grep -c '^\s*(async )?def test_'`).
- **Test config:** no `pytest.ini`, `setup.cfg`, `tox.ini`, `pyproject.toml` or root `conftest.py` (all NOT FOUND).
- **Fixtures:** `document_converter/tests/fixtures/` is empty (0 files). `conftest.py:12` defines `FIXTURES` but no test file references it (searched `FIXTURES`).
- **Pytest run: NOT RUN, by your decision during this audit.** Two test files (`test_orchestrator_docs_api.py:25`, `test_upload_identity.py:26`) `import orchestrator`. That import opens `orchestrator.log` in the working directory at DEBUG level, which would have appended to `/home/jrochau/Projects/site/orchestrator.log` and broken the no-modification rule. Pass/fail/skip counts and failing test names were not collected.

Tests that need live services, GPU or external tools:

| Requirement | Tests | Evidence |
|---|---|---|
| Real HF tokenizer `BAAI/bge-large-en-v1.5` in the local HF cache (offline) | 8 tests marked `@requires_real_tokenizer` in `test_atomic_split.py` (lines 222, 238, 262, 285, 314, 333, 366, 400) | skipif at `test_atomic_split.py:29-39` |
| LibreOffice `soffice` | `test_doc_extractor.py` (skip at 22, skipif at 40); `test_pipeline.py:176` | skipif |
| tesseract | all of `test_ocr_heading_scoring.py` (module-level skipif, line 24); `test_pdf_advanced.py:203,224` | skipif |
| Embedding server / Ollama / GPU | None found. Pipeline and orchestrator tests use `FakeEmbeddingClient` (`conftest.py:15-47`). `test_gpu_lifecycle.py` monkeypatches nvidia-smi and requests (`test_gpu_lifecycle.py:3`). `test_embedding_server_oom.py` stubs `sentence_transformers` before import (`:4-5,53-69`). `test_semantic_structuring.py:32` points at localhost:11434 with `model="fake-model"`. | — |
| Import-time reads of the real repo | `import orchestrator` loads config, runs `DocStore` over the real `docs/.index`, and attempts an embedding `/health` call (`orchestrator.py:172-173,442-462,813`) | — |

---

## 11. Dependencies

`requirements*.txt`, `pyproject.toml`, `setup.py`, `setup.cfg`: NOT FOUND (searched with `find` across the repo, venvs excluded). There is no dependency manifest to quote verbatim.

Python version: `/home/jrochau/Projects/site/.venv`, `/home/jrochau/Projects/site/.venv-docling` and `/home/jrochau/Projects/site/.venv-xberg` all report **Python 3.12.3**.

Installed versions in `/home/jrochau/Projects/site/.venv` (`pip list`, selected packages imported by code):

| Package | Version | | Package | Version |
|---|---|---|---|---|
| fastapi | 0.138.2 | | uvicorn | 0.49.0 |
| Flask | 3.1.3 | | pydantic | 2.13.4 |
| requests | 2.34.2 | | httpx | 0.28.1 |
| numpy | 2.5.0 | | torch | 2.13.0 |
| sentence-transformers | 5.6.0 | | transformers | 5.14.0 |
| pymupdf | 1.28.0 | | python-docx | 1.2.0 |
| beautifulsoup4 | 4.15.0 | | pytesseract | 0.3.13 |
| pillow | 12.3.0 | | faster-whisper | 1.2.1 |
| PyYAML | 6.0.3 | | python-multipart | 0.0.32 |
| pytest | 9.1.1 | | | |

Docling is not in `.venv`; it runs from `.venv-docling` as a subprocess (`docling_extractor.py:8-11`). External binaries: `nvidia-smi` (optional), `tesseract` (OCR), `soffice`/`libreoffice` (.doc), `ollama`.

---

## 12. Open defects

### a. CalloutBlock and U+FF17 (`７`) promotion to HeadingBlock: **PARTIAL**

- **CalloutBlock type:** NOT FOUND (searched `CalloutBlock|Callout` in `document_converter/`, tests excluded). `model.py` block classes are listed at `model.py:29-280`; none is a callout.
- **Numbering:** fixed so `７` is not parsed as a section number. `_NUMBERING_RE` is ASCII-only `[0-9]` (`extractors/__init__.py:88-100`), and so is `_LEADING_NUMBER_RE` (`section_numbering_audit.py:36-40`). Tests: `test_heading_classifier.py:376-382`, `test_generalization_repairs.py:222-247`.
- **Heading promotion is not prevented:**
  - `pdf_extractor.py:1494-1504`: a line is promoted to `HeadingBlock` by `_is_heading_line` (`:1161-1184`, font-size and candidate rules), with no `７`/callout check. With ASCII-only numbering, `numbering_depth("７ Note")` returns `None`, so the heading gets the default level rather than depth 1 (`:1259-1261` and the comment at `:1216-1219`).
  - `md_parser.py:165,184,195`: ATX, setext H1 and setext H2 lines always become `HeadingBlock`, with no `７` check (searched `７|７|FF17` in `md_parser.py`: NOT FOUND).
  - The test fixture keeps `"７ Note"` as a `HeadingBlock` after the audit and asserts the text is untouched (`test_generalization_repairs.py:233-247`).
- **Pattern mismatch still in place:**
  - `_NUMBERING_WORD_RE` still uses `\d+` after chapter/section/appendix (`extractors/__init__.py:100`).
  - The PDF list-bullet regex `_BULLET_RE` uses `\d{1,3}[.)]` (`pdf_extractor.py:42`), so `７.` or `７)` would match as an ordered bullet.
- `heading_demotion.demote_false_headings` demotes verbatim-repeated heading text at or above `_REPEAT_THRESHOLD` (`heading_demotion.py:350`). Whether that catches repeated "７ Note" depends on the count. Not verified by running.

### b. `_split_table_block` token estimator vs. rendered markdown table size: **FIXED on the index path; PARTIAL on preview/process**

- `atomic_split.py:595-639`: every candidate part is measured as the full rendered markdown (`_render_table_text` → `render_table`, `:437-445`) and tokenized whole. The header cost is only used as a floor (`:606-613`). The comment at `:629-634` says per-row token summing was "the bug this replaces".
- Per-part final gate: `context_header + render_block(part)` must be `<= max_tokens`, otherwise `AtomicSplitError` (`atomic_split.py:965-976`). The table reserve comes from a sample header "(repeated header)" (`:896-902`). Oversized-row fragments use their own per-fragment header estimate (`:658-678`).
- Estimator choice (`chunker.py:397-398`):
  - **index path:** the live embedding tokenizer on the rendered table
  - **preview/process:** `_approx_token_count`, i.e. `round(words * 1.3)` (`chunker.py:53-56`), where "words" are whitespace-split tokens of the rendered markdown including `|` and `---` separators. Budget is `round(effective_hard * 1.3)`.
- `oversized_atomic_block` is now a **WARNING** and the block is **dropped** from the index (`chunker.py:405-422`). It no longer fails the document (`retrieval_readiness.py:40-44`).
- Background analysis document: `TABLE_SPLIT_TOKEN_ACCOUNTING_ANALYSIS.md`.

### c. `max_sequence_tokens` short-circuit bypassing the "chunks exceeding embedding limit" check: **PRESENT**

- `chunker.py:275`: `if max_sequence_tokens and token_count > max_sequence_tokens:` → `chunk_exceeds_token_limit`. When `max_sequence_tokens` is `None`, the check is skipped entirely. That is always the case for `preview()`, `process()` and the first `_run` inside `index()`, which pass `embedding_client=None` (`pipeline.py:387-392,426,439`).
- The check compares against `max_sequence_tokens`, not the configured `max_tokens` ceiling (`chunker.py:238` vs `:275`).
- During `index()` the re-chunk passes the live `info.max_sequence_tokens` (`pipeline.py:485-488`), so the check runs there. The embedding server's 413 is the final guard (`embedding_server.py:303-318`; `pipeline.py:566-585`).
- Validation reads the finding: `validation.py:785` counts `chunk_exceeds_token_limit` codes.

### d. Cell-split on pipe boundaries shredding prose (chunker / chunker_ref): **PRESENT in the markdown parser; chunker_ref NOT FOUND**

- `chunker_ref`: NOT FOUND (searched the filename and `chunker_ref` in `.py`/`.md` under `/home/jrochau/Projects/site`, venvs excluded).
- `md_parser.py:231-253`: a table starts whenever a line contains `|` and the **next** line passes `_looks_like_table_delimiter` (`md_parser.py:89-96`). Following lines are absorbed as rows while they contain `|` (`:236`), and each is split on every unescaped `|` (`_split_table_row`, `:64-86`). There is no check that the rows are table-shaped. Only a warning (`complex_table`) is logged when column counts differ (`:239-250`).
- The paragraph loop breaks at any line containing `|` whose next line looks like a delimiter (`md_parser.py:266`).
- `_looks_like_table_delimiter` accepts any line whose non-empty `|`-cells all match `:?-{1,}:?`. A lone `---` line passes this regex, but setext-H2 detection (`md_parser.py:175-202`) only runs when the current line contains no `|`.
- `atomic_split.py:715,737` rebuilds `row_texts` with `" | ".join(row)` for the reconstruction check only.

### e. Nav link-dump filter in the indexing path: **NOT FOUND**

Searched `nav(igation)?_?link|link_dump|link-dump|linkdump|link_density|is_nav` in non-test `document_converter/` .py files, and `link|url|href|menu|breadcrumb` in `retrieval_view.py`, `html_extractor.py` and `normalize.py`.
- `html_extractor.py:29-31` renders `<a>` tags as `[text](href)`, with no filtering.
- `retrieval_view.py` excludes block types, shell headings, junk headings, arXiv stamps and reference entries (`retrieval_view.py:43,89,148,175,391`), but has no link-list rule.
- `tests/test_navigation_links.py` tests `index.html` ↔ `document_converter.html` hyperlinks, not content filtering (`test_navigation_links.py:1-9`).

---

## 13. Repo state

`/home/jrochau/Projects/site` is **not a git repository**. `git status --short`, `git log --oneline -20` and `git branch --show-current` all returned:

```
fatal: not a git repository (or any of the parent directories): .git
```

Current branch: N/A (no `.git`).

Because there is no git, change detection used a file snapshot instead (see Verification).

---

## Contradictions (docs/comments vs code)

1. **Whisper defaults.** The `whisper_server.py:17-18` docstring says `WHISPER_DEVICE` defaults to `'cuda'` and `WHISPER_COMPUTE` to `'float16'`. The code defaults to `"cpu"` and `"int8"` on CPU (`whisper_server.py:33-34`).
2. **Who calls Ollama for generation.**
   - `orchestrator.py:626-630` (`DocStore.search` comment) says the browser's next step is "a direct call to Ollama for `qwen3:14b` generation (index.html)". In code, generation is orchestrator-mediated (`/chat/stream`, `orchestrator.py:2231`), and `index.html` contains no `11434`, `/api/chat` or `keep_alive`.
   - `HANDOFF.md:318-319` says `index.html` "Sends `keep_alive: 0` on generation". It does not; the orchestrator sends it, and only when `use_kb` is true (`orchestrator.py:2224-2225`).
3. **"Must stay in sync" with functions that don't exist.** `_src_label` and `ground_last_user_turn` are documented as Python ports of `index.html`'s `srcLabel()` and `groundLastUserTurn()` that "must stay in sync" (`orchestrator.py:745,758-761`). Neither function exists in `index.html` (searched `function srcLabel|function groundLastUserTurn`: NOT FOUND).
4. **Context budget.** The `config.py:10-11` docstring lists "retrieval budget" as defined there. `RAG_CONTEXT_BUDGET_TOKENS` / `retrieval_context_budget_tokens` has no consumer, and `ground_last_user_turn` concatenates full chunk texts with no budget (`orchestrator.py:780`).
5. **Upload limits.**
   - The `config.py:10` docstring lists "upload limits". `upload_max_bytes` has no consumer.
   - `orchestrator.py:1436,1484-1485` comments say "no size cap". That matches the code but contradicts the presence of `UPLOAD_MAX_BYTES`.
6. **Profile in config file.**
   - `DELIVERABLES.md:104-105` says the profile can be selected via the config file's "top-level `profile` key". `_apply_nested` applies only `strict`, `upload_max_bytes`, `retrieval_context_budget_tokens` and `delete_source_on_doc_delete` at the top level (`config.py:503-505`), so a file `profile` key is ignored.
   - The `config.py:11-13` docstring says "ingestion.py, orchestrator.py and control.py all import this module". `control.py` imports only `math`, `string`, `numpy`, `typing` and `collections` (`control.py:8-12`).
7. **Config file location.** The `config.py:6-7` docstring says `./docpipe.yaml` / `./docpipe.json` "next to this package". Code looks in `REPO_ROOT`, the package's parent directory (`config.py:41,475-477`).
8. **What `ingestion.py` scans.** The `ingestion.py:4-6` docstring says it scans "docs/**/*.md". It calls `scan_and_index`, which indexes every `SUPPORTED_EXTENSIONS` file (pdf, docx, doc, html, htm, txt, md, markdown) under `docs/` (`pipeline.py:925-927`). `HANDOFF.md:292,333` also say `docs/**/*.md`.
9. **Title overrides location.** The `title_overrides.py:21-22` docstring says the file is `docs/.index/title_overrides.json`. Code uses `docs/.index/_meta/title_overrides.json` (`title_overrides.py:49,53`). `HANDOFF.md:332` matches the code.
10. **Hybrid ranking description.**
    - `HANDOFF.md:220-221` and the `control.py:4` docstring say "BM25 + cosine hybrid". Code scores cosine on all chunks and BM25 only on the top 50 by cosine (`control.py:149-165`).
    - The `control.py:78` docstring says cosine is in the range [0, 1]. The function returns a raw cosine that can be negative (`control.py:96`).
    - The `orchestrator.py:707` comment describes the formula as `0.7*cosine + 0.3*bm25_norm`, which matches `control.py:176`.
11. **ETA stage list.** `_PIPELINE_STAGE_ORDER` (`orchestrator.py:100-104`), used for job ETA, omits `demoting_false_headings`, which `pipeline.py:252-255` emits. `_job_overall_current` returns `None` for that stage (`orchestrator.py:1250-1251`), so `remaining_seconds` is set to `None` during it (`:1295-1297`).
12. **index.html upload vs the 202 response.** `/docs/upload` returns 202 with `{job_id, document_id, filename}` only (`orchestrator.py:1536`). `index.html:1423-1431` reads `d.chunks` and `d.validation` from that response and reports "indexed" without polling the job. `document_converter.html` does poll (`:637-681`).
13. **"No iframe" vs a runtime iframe.** The `tests/test_navigation_links.py:7` docstring and `test_neither_page_uses_an_iframe` (`:40-42`) assert no iframe, by checking only the literal string `<iframe`. `index.html:839-843` creates an iframe at runtime with `document.createElement("iframe")`.
14. **Orchestrator import side effects.** `tests/test_orchestrator_docs_api.py:7-9` and `tests/test_upload_identity.py:15-17` say importing `orchestrator` has "only" a read-only side effect. Import also:
    - attaches a DEBUG `FileHandler('orchestrator.log')` (`orchestrator.py:30-37`)
    - runs `os.makedirs` for `sessions/`, `artifacts/` and `threads/` (`:140-141`)
    - runs `load_config()` → `paths.ensure()` (`config.py:495`)
    - calls the embedding `/health` (`orchestrator.py:460`)
15. **Profile override in chunking.** The `chunker.py:10-11` docstring says the emergency paragraph split applies "in EVERY profile, strict included", which matches `chunker.py:344-351`. The comment at `pipeline.py:306-312` still describes strict mode's "never split a paragraph" rule as a FULL_FIDELITY guarantee and forces `strict=False` for MAIN_TEXT. The oversized-paragraph path no longer branches on `config.strict` in the chunker (`chunker.py:340-394`).
16. **Relevance buckets.** `orchestrator.py:706-708` says the buckets are used to tune relevance. `relevance_bucket()` is never called (see 7.3).
17. **Supported extensions.** `TxtExtractor.supports` accepts `.text` (`txt_extractor.py:25`), but `SUPPORTED_EXTENSIONS` omits it (`extractors/__init__.py:727`), so `.text` uploads are rejected at `orchestrator.py:1474` and `pipeline.py:414`.
18. **OCR modes.** The `config.py:158` comment lists `off | auto | on`. Code distinguishes only `off` (`pdf_extractor.py:981`); `on` does not force OCR on pages with text.

---

## Verification

- **Change check:** there is no git, so a file list with sizes and mtimes of `/home/jrochau/Projects/site` (2,336 files, venvs and `orchestrator.log` excluded) was captured before the audit and compared afterwards with `diff`. No existing file changed. Two files were added:
  - `/home/jrochau/Projects/site/site_audit.md` (this report)
  - `/home/jrochau/Projects/site/.claude/settings.local.json` (239 bytes). Claude Code wrote this when a permission prompt was approved with "always allow". It contains two allow rules for `.venv/bin/python --version` and a pytest version check. The audit's commands did not write it.
- `/home/jrochau/Projects/site/orchestrator.log` is unchanged: 238,896,256 bytes, mtime Sep 26 12:14, the same as before the audit.
- **Sections:** 1–13 are all present. Empty findings are marked NOT FOUND with the search used.
- **Not collected:** pytest results (Section 10), skipped at your instruction.

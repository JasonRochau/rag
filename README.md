# rag — Local-First Retrieval-Augmented Generation

A fully local RAG knowledge assistant: upload documents, index them through a
strict conversion-and-validation pipeline, and chat with a local LLM that
answers only from your sources. No cloud, no Docker, no external API calls.

Built around three small services and a static web UI:

```
┌────────────────────────────────────────────────────────────┐
│  Browser — index.html (Knowledge Assistant, port 8000)     │
└──────────────┬───────────────────┬────────────┬────────────┘
               │                   │            │
        ┌──────▼───────┐   ┌───────▼──────┐  ┌──▼───────────┐
        │ orchestrator │   │  embedding   │  │   whisper    │
        │   FastAPI    │   │    Flask     │  │   FastAPI    │
        │    :11436    │   │   :11437     │  │   :11435     │
        └──┬────┬────┬─┘   └──────────────┘  └──────────────┘
           │    │    │
   ┌───────▼┐ ┌─▼────▼──────┐ ┌────────────┐
   │llama.cpp│ │ docs/.index │ │   Ollama   │
   │ :8080   │ │ per-doc JSON│ │ :11434     │
   │(chat LLM)│ │  + vectors  │ │(vision, opt)│
   └─────────┘ └─────────────┘ └────────────┘
```

## Features

**Document ingestion pipeline** (`document_converter/`)
- Extractors for PDF (PyMuPDF, with Tesseract OCR fallback for scanned
  pages), DOCX, DOC (via LibreOffice), HTML, Markdown, and plain text
- Normalization and repair passes: running-header removal, heading repair,
  false-heading demotion, numbered-sibling and section-numbering audits,
  interrupted-paragraph repair
- Token-aware chunking measured with the live embedding tokenizer, with
  atomic splitting for oversized tables and code blocks
- Strict validation gate: every document is scored for retrieval readiness
  and blocked with actionable findings if it wouldn't retrieve well
- Oversized PDFs are automatically split and ingested as parts
- Optional visual analysis: figures and diagrams are extracted and described
  by a local vision model (Qwen3-VL via Ollama), cached by content hash
- Manual title overrides, per-document validation reports, and rendered
  markdown for every indexed file

**Retrieval**
- Per-document JSON index (`docs/.index/`) with embedding fingerprints —
  stale or incompatible indexes are skipped automatically at load
- Hybrid ranking: cosine similarity over all chunks, then BM25 re-ranking of
  the top 50, blended `0.7 * cosine + 0.3 * BM25`
- Linked-context expansion pulls in related chunks (e.g. a figure's analysis
  alongside the text that references it)

**Chat**
- Streaming chat (SSE) backed by a local llama.cpp server — the orchestrator
  launches and manages it automatically
- Sources are retrieved, shown to the user, and injected into a grounding
  prompt before generation; the model is instructed to answer only from them
- Persistent threads, project state, and end-of-session summaries

**Agent build jobs**
- Submit build jobs (`/jobs`) that run an agent loop grounded in the
  knowledge base, each producing output in its own `jobs/<id>/output/`
  directory with live progress over SSE

**GPU lifecycle management**
- VRAM barriers around every GPU stage: the embedding model is released and
  memory is verified free before the chat model generates
- `nvidia-smi` and Ollama status surfaced through `/gpu/status`

**localAgent**
- User-granted directory access from the UI: crawl, write markdown notes,
  and run installers — every execution requires explicit confirmation

## Quickstart

Prerequisites:

- Python 3.12
- A llama.cpp `llama-server` binary and a GGUF chat model
  (default: Qwen3.6-35B-A3B — set `CHAT_MODEL_PATH` to your `.gguf`)
- Optional: Ollama (visual analysis), Tesseract (scanned-PDF OCR),
  LibreOffice (`.doc` files), `nvidia-smi` (GPU management)

```bash
git clone <repo-url> rag && cd rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python embedding_server.py &     # :11437
python whisper_server.py &       # :11435 (optional, mic dictation)
python orchestrator.py &         # :11436
python -m http.server 8000 --bind 127.0.0.1
```

Open http://127.0.0.1:8000/index.html, switch to the Knowledge Base tab,
and upload a document.

## Configuration

Everything runs on localhost with sane defaults. Common environment
variables:

| Variable | Default | Purpose |
|---|---|---|
| `ORCH_PORT` | `11436` | Orchestrator port |
| `EMBEDDING_PORT` | `11437` | Embedding server port |
| `WHISPER_PORT` | `11435` | Whisper server port |
| `CHAT_LLM_PORT` | `8080` | llama.cpp chat server port |
| `CHAT_MODEL_PATH` | — | Path to the chat GGUF model |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `DOCPIPE_STRICT` | `1` | Block documents that fail validation |
| `DOCPIPE_PROFILE` | `main_text` | `main_text` (retrieval view) or `full_fidelity` |
| `VISUAL_ANALYSIS_ENABLED` | `0` | Vision-model analysis of figures/diagrams |

A `docpipe.yaml` / `docpipe.json` at the repo root can set paths and
pipeline options; environment variables always win.

## Testing

```bash
python -m pytest document_converter/tests/
```

~550 tests covering the pipeline stages, chunking, validation, ranking, and
the orchestrator API. External tools (Tesseract, LibreOffice, real
tokenizers) are auto-skipped when unavailable.

## Project layout

```
orchestrator.py          FastAPI app: KB API, chat, threads, jobs, localAgent
job_runner.py            Agent build-job runner (jobs/<id>/output)
embedding_server.py      Flask embedding service (sentence-transformers)
embedding_backend.py     Model loading and batching for the embedding server
whisper_server.py        faster-whisper speech-to-text service
ingestion.py             Legacy full-rescan entry point
control.py               Hybrid ranking (cosine + BM25)
index.html, app.css      Web UI
document_converter/      Ingestion pipeline package
  extractors/            pdf / docx / doc / html / md / txt extractors
  pipeline.py            Stage orchestration, strict gate, indexing
  chunker.py             Token-aware chunking
  atomic_split.py        Atomic splitting of oversized blocks
  validation.py          Findings and retrieval-readiness scoring
  tests/                 pytest suite
docs/                    Document corpus and generated index
```

## License

MIT
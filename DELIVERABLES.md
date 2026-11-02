# Document Pipeline Rewrite — Deliverables

Rewrite of the knowledge-base ingestion pipeline: a single format-agnostic
`document_converter` package now sits between raw source files (markdown,
txt, html, docx, doc, pdf) and the search index, replacing the old
markdown-only, ollama-embed-direct `ingestion.py`.

## Architecture / data flow

```
 upload / legacy scan
        |
        v
 select_extractor(path)              document_converter/extractors/*
        |  -> (Document, FindingLog)
        v
 normalize_document(document)        document_converter/normalize.py
        |  (single/first H1 rule, heading-level repair, both as FIXED findings)
        v
 apply_semantic_structuring(...)     document_converter/semantic_structuring.py
        |  (optional local Ollama JSON "plan": headings + existing block ids only,
        |   validated for coverage/order before being trusted; off by default)
        v
 render_markdown(document)           document_converter/render.py
        |  -> canonical Markdown text (also what MarkdownChunker parses)
        v
 MarkdownChunker.chunk(...)          document_converter/chunker.py
        |  -> ChunkRecord[] (provenance-prefixed, atomic blocks never split
        |     in strict mode, packed to a word AND real-token budget)
        v
 validate(...)                       document_converter/validation.py
        |  -> ValidationReport (typed findings + stats + retrieval_ready)
        v
 EmbeddingClient.embed_batch(...)    document_converter/embedding_client.py
        |  -> one vector per chunk, from the single local embedding_server.py
        v
 write_index_atomic(...)             document_converter/index_schema.py
        -> docs/.index/<document_id>.json   (schema_version 2, one object
           per chunk, atomic temp-file + os.replace so a crash never leaves
           a half-written index)
```

`document_converter/pipeline.py` (`DocumentPipeline`) is the only thing
that calls the stages above, in this order, and is the single entry point
used by:

- `document_converter/cli.py` — `process | validate | preview | index | scan | rebuild`
- `ingestion.py` — thin wrapper that scans the legacy `docs/**/*.md` tree
- `orchestrator.py` — `/docs/upload`, `/docs/preview`, `/docs/{id}`, `/docs/search`, `/ingest`

`DocumentPipeline.preview()`/`.process()` never touch the embedding
service (word-count packing + an offline token estimator), so validation
and chunk-preview work with no services running. `.index()` fetches the
embedding service's real `dimension`/`max_sequence_tokens` once
(`EmbeddingClient.health()`) and re-chunks using the model's *actual*
tokenizer to decide where to split — see "token vs. word budgets" below.

Document identity is `stable_document_id(relative_source_path)` — a hash
of the file's path relative to its scan root, not its content. Re-running
`index()`/`scan_and_index()` on an edited file overwrites the same index
entry (atomically) instead of orphaning the old one, which is what the
old ollama-based `ingestion.py` did by keying on content hash.

## Commands

```bash
# one file
python -m document_converter.cli preview  path/to/file.pdf     # print chunks, write nothing
python -m document_converter.cli validate  path/to/file.docx    # findings report only
python -m document_converter.cli process   path/to/file.html    # + writes normalized .md + report
python -m document_converter.cli index     path/to/file.md      # + embeds + writes docs/.index/<id>.json

# directories
python -m document_converter.cli scan    <dir> [--force]        # index every supported file under <dir>
python -m document_converter.cli rebuild [--force]               # re-index paths.source_dir + legacy_docs_dir

# services (both must be running for /docs/upload, /docs/search, and `cli index`)
python embedding_server.py     # local sentence-transformers model, port 11437
python orchestrator.py         # FastAPI app, port 11436

# tests
python -m pytest document_converter/tests -q

# lightweight retrieval smoke-eval against whatever's currently indexed
# (requires embedding_server.py running; see "retrieval smoke-eval" below)
python -m document_converter.eval_retrieval
```

Two GUIs, both static files opened directly in a browser (no server of
their own): `index.html` (chat assistant, with an embedded KB panel) and
`document_converter.html` (dedicated converter/library page — drop a file,
see its chunk count, open the converted markdown in a separate window,
optionally commit it to the knowledge base).

Configuration is centralized in `document_converter/config.py`
(`PipelineConfig`) — one dataclass tree covering paths, embedding
host/model, chunk budgets, OCR mode, and strict mode, loadable from
`docpipe.yaml`/`docpipe.json` or environment variables (see the module
docstring for the full variable list). `ingestion.py`, `orchestrator.py`
and `document_converter.cli` all load the same config.

## MAIN_TEXT vs FULL_FIDELITY ingestion profiles

Two ingestion profiles, selected via `--profile`, the `DOCPIPE_PROFILE` env
var, `docpipe.yaml`/`.json`'s top-level `profile` key, or a `profile=`
keyword on any `DocumentPipeline` method (`config.py`'s `MAIN_TEXT_PROFILE`
/ `FULL_FIDELITY_PROFILE` / `INGESTION_PROFILES`). Both profiles reuse the
exact same extraction/normalization result -- there is no second parser
pipeline, only a filter (`retrieval_view.py`) and a gating policy
(`retrieval_readiness.py`) layered on top.

**MAIN_TEXT** (default -- what `orchestrator.py`'s `/docs/upload`/`/docs/search`
and a bare `cli index`/`process`/`validate` use unless told otherwise):
intended for the active RAG index. Chunks and indexes only the filtered
main-text retrieval view: title, headings, body paragraphs/lists/code,
existing tables -- author/affiliation/contact, copyright/permission/
conference/publication-history metadata, figures/images/captions/diagram
labels, table-of-contents entries, and References/Bibliography content are
excluded from the view (never embedded, never a retrievable chunk).
Gates indexing on retrieval-critical findings only (missing title, no main
body text, a chunk that can't be made to fit the real embedding tokenizer,
broken source/page/heading-path provenance, empty/duplicate/missing
chunks) -- front-matter placement, figure-caption-shaped headings, and the
full-document single-H1 gate no longer block indexing, since MAIN_TEXT has
already excluded that content from what actually gets searched. A
legitimately long body paragraph is split on sentence boundaries (real
tokenizer, when available) rather than dropped, since strict mode's
"never split a paragraph" guarantee is a FULL_FIDELITY diagnostic
promise, not a MAIN_TEXT one.

**FULL_FIDELITY**: today's original, unmodified behavior. Chunks and
indexes the complete document -- every block, every front-matter/figure/
citation region -- and gates on full structural validation (`validation.py`'s
`errors == 0`, unchanged). Intended for parser diagnostics and any future
document-reconstruction work; a developer can always inspect the complete
extraction result this way, even for a document MAIN_TEXT successfully
indexed off a filtered view of.

```bash
# MAIN_TEXT (default) -- what actually gets embedded/indexed for search
python -m document_converter.cli index    path/to/paper.pdf
python -m document_converter.cli process  path/to/paper.pdf --profile main_text

# FULL_FIDELITY -- diagnostics on the complete, unfiltered document
python -m document_converter.cli validate path/to/paper.pdf --profile full_fidelity --json
python -m document_converter.cli process  path/to/paper.pdf --profile full_fidelity
```

Every indexed chunk carries a `content_types` list (the distinct block
classes that made it up, e.g. `["ParagraphBlock"]`) alongside the
existing `chunk_id`/`header_path`/`source_pages`/`provenance` fields, and
the index's `document` object records which `profile` produced it.

## Migration performed

Ran `python -m document_converter.cli rebuild` against the existing
`docs/**/*.md` corpus (17 files, previously indexed by the old
content-hash-keyed, schema-less `docs/.index/*.json` format). Result:
**13 of 17 migrated cleanly** to schema_version 2 with real embeddings
(634 chunks, `BAAI/bge-large-en-v1.5`, verified end to end with
`/docs/search` returning sensibly ranked hits). The old schema-less index
files for those 13 sources were replaced; `DocStore.reload()` detects and
warns about any remaining legacy-schema files via
`index_schema.is_legacy_schema()`, so a partially-migrated index dir is
never silently misread as v2.

4 files did **not** migrate, and were left out of the index rather than
silently truncated:

| file | issue |
|---|---|
| `azure/identities_and_governance/azure_policy_initiatives.md` | one table block is 487 tokens on its own |
| `networking/automation_and_programmability.md` | two list blocks (501, 1281 tokens) |
| `networking/ip_connectivity.md` | one list block (516 tokens) |
| `networking/security_fundamentals.md` | one list block (949 tokens, "Domain 5 Rapid-Fire Review") |

These are single Markdown list/table blocks whose *own* content exceeds
the embedding model's 512-token limit. Chunking treats lists and tables
as atomic (never splits mid-item/mid-row, since that would silently
corrupt structure) and strict mode refuses to index a document it can't
faithfully chunk — so these `ERROR`-level findings correctly blocked
indexing rather than being silently dropped or mangled. Fix at the
source (break the oversized list/table into smaller subsections) and
re-run `rebuild`, or set `chunking.strict: false` to allow the
sentence-boundary emergency fallback (paragraphs only, not lists/tables).

## Key bug found and fixed during migration

The chunker packed blocks against a **word**-count budget
(`hard_max_words`/`target_words`) but the embedding service enforces a
**token** limit. `token_counter`/`max_sequence_tokens` were already
threaded through `MarkdownChunker.chunk()`, but only used for a post-hoc
check after packing was already decided — so technical content (CLI
syntax, hyphenated jargon, IP addresses) that tokenizes at up to ~2x the
naive 1.3-tokens/word estimate routinely produced chunks the embedding
service would reject. First `rebuild` run: 9/17 files failed on
`chunk_exceeds_token_limit`.

Fixed in `document_converter/chunker.py`: `_pack_blocks` now tracks a running
*token* total (one `token_counter` call per block, summed — a conservative
upper bound on the group's real tokenized length) alongside the word
total, and flushes on whichever budget would be exceeded first, whenever
a real token counter is available (i.e. during `index()`, not
`preview()`/`process()`, which stay offline). This dropped the failure
count from 9 to 4 — the remaining 4 are genuine oversized-atomic-block
content issues described above, not a packing bug.

## Universal document formatting corrective pass

A later round of work targeted format-agnostic gaps rather than one
document at a time: fixes and additions below apply across the whole
supported-format population, not to any one publisher/title/vocabulary.

**Implemented:**

- **PDF compound-block code detection** (`pdf_extractor.py`). PDFs that
  render code in the same font as body text (common for print-to-PDF
  sources that only vary background color) previously left code as plain
  paragraph text, which then tripped the HTML-remnant warning on syntax
  like `#include <iostream>`. A content-based fallback (regex patterns for
  unambiguous code syntax, requiring either one strong signal or two
  weaker ones in a row) now reclassifies these runs as code even with no
  font signal at all. Regression-tested in `test_pdf_advanced.py`.
- **PDF multi-column reading order.** Text blocks are now clustered into
  left/right column bands (when the page clearly has two of them) and
  read left-column-top-to-bottom then right-column-top-to-bottom, instead
  of whatever order the PDF's content stream happened to emit them in.
  Falls back to natural top-to-bottom order for ordinary single-column
  pages (the large majority).
- **Flat-OCR-transcript fix.** OCR'd text previously had no font-size or
  gap signal at all, so every line on a scanned page collapsed into one
  giant paragraph regardless of its actual structure (this is spec
  section 1's explicitly named "flat OCR transcripts" failure mode).
  OCR text is now split using tesseract's own blank-line paragraph breaks
  (real gaps, not zero) plus a content-based heading heuristic (an
  isolated short unpunctuated line between blank lines) for scanned
  headings that have no font-size signal to lean on.
- **Encrypted/password-protected PDFs** now produce a clean
  `pdf_password_protected` error (after trying an empty password, since
  some PDFs only restrict printing/editing, not viewing) instead of
  crashing or hanging downstream.
- **Corrupt/unreadable files** (any format) are now caught at
  `DocumentPipeline._build_document()` and turned into a normal
  `extraction_failed` `ERROR` finding with the same typed
  `ValidationReport` shape as any other failure, instead of an unhandled
  exception/raw traceback bubbling up through the API.
- **DOCX headers/footers** were previously not read at all — silently
  absent, not logged. `docx_extractor.py` now collects distinct
  header/footer text across all sections and logs it as an intentional
  exclusion (`docx_header_footer_excluded`, `INFO`) rather than leaving no
  trace that it existed. Still never added as body content (it's running
  boilerplate, not content).
- **Legacy `.doc` support** (`extractors/doc_extractor.py`): converts to
  `.docx` locally via LibreOffice headless mode (`soffice --headless
  --convert-to docx`), then runs the existing `DocxExtractor` on the
  result. A missing `soffice`/`libreoffice` binary produces a clean
  `doc_conversion_unavailable` dependency error (same pattern as the
  Tesseract-missing error), not a crash. `.doc` files are never modified;
  conversion happens in a throwaway temp directory. Runs fully locally.
- **Universal quality-report fields** (`validation.py::compute_stats`)
  now include the full field set from the spec's section 24 schema:
  page/OCR accounting (`native_text_pages`, `ocr_pages`, `failed_pages`,
  `pages_accounted_for`), source-block coverage
  (`source_blocks`/`represented_source_blocks`/`excluded_source_blocks`/`duplicate_source_blocks`),
  heading validity (`heading_count`, `heading_hierarchy_valid`,
  `unique_header_paths`), per-type element counts (`list_count`,
  `table_count`, `code_block_count`), and a `structural_readiness` flag
  distinct from `retrieval_ready` (structural = no blocking errors;
  retrieval-ready = structural *and* at least one chunk was actually
  produced). All fields report `0`/`None` rather than being omitted when
  the corresponding element type is absent from a given source, per the
  spec's explicit "counts may legitimately be zero" rule.
- **Per-document completion-state object** (`validation.py::completion_state`,
  exposed as `ProcessResult.completion_state` and in the `/docs/preview`
  and `/docs/upload` API responses) matching the spec's section 8 schema
  exactly: `processing_complete`, `source_coverage_complete`,
  `structural_readiness`, `retrieval_ready`, `blocking_errors`,
  `forced_fallback_splits`, `embedding_truncations`, `indexed`.
- **Lightweight retrieval smoke-eval** (`document_converter/eval_retrieval.py`,
  `python -m document_converter.eval_retrieval`): 10 hand-written
  questions run against whatever's currently in `docs/.index/` through the
  same `HybridRanker` used in production, reporting top-1/top-3/top-5 hit
  rates against an expected source-document match. Measured against the
  real 13-document/634-chunk corpus: **80% top-1, 100% top-3, 100% top-5**.
- 23 new tests covering all of the above (`test_pdf_advanced.py`,
  `test_docx_headers_footers.py`, `test_doc_extractor.py`,
  `test_corrupt_files.py`, `test_quality_report.py`, plus additions to
  `test_pipeline.py`), all run against real PyMuPDF/python-docx/tesseract/
  LibreOffice — not mocked. 81 tests total, all passing.

**Explicitly deferred, and why** (the full spec's scope is roughly 10x
what's implemented here — see the corrective-implementation-prompt
conversation for the complete ask):

- **Per-block bounding-box / OCR-confidence tracking across every
  extractor.** The `Block` model doesn't carry per-block geometry beyond
  what the PDF path already needs internally, and nothing downstream
  (chunker, validator, UI) consumes it — building it out fully would be a
  large mechanical change with no current consumer.
- **Form-field and footnote/endnote extraction.** Not present in this
  project's actual corpus (CCNA/Azure/DS&A study material); would be
  speculative work against an untested use case.
- **The full retrieval-evaluation harness** with a curated gold question
  set across all 19 fixture types and formal precision@1/3/5 targets.
  What's shipped (`eval_retrieval.py`) is a real, working smoke check
  against the actual production corpus, not a mock — but it's 10
  hand-written questions against one corpus, not the full multi-format
  curated harness the spec describes; scaling it up is its own
  multi-day deliverable.
- **Iterative repair-loop convergence** (spec section 22's "re-attempt
  classification/decomposition/rendering until it passes or attempts are
  exhausted"). The pipeline is a single deterministic pass; the
  content-based code detection and OCR paragraph reconstruction above are
  one-shot decomposition passes, not a retry loop. An oversized atomic
  block still fails cleanly rather than being retried with progressively
  different strategies.
- **Password-protected DOCX/legacy-`.doc`** — only encrypted **PDF** got
  explicit handling; an encrypted `.docx` currently surfaces via the
  generic `extraction_failed` catch-all (still a clean error, just not a
  format-specific `docx_password_protected` code).

## Assumptions

- **Document identity is path-based, not content-based.** Re-processing
  the same relative path always overwrites the same index entry. Two
  different source files that happen to produce byte-identical content
  are treated as two separate documents.
- **`strict: true` by default** (`PipelineConfig.strict`,
  `ChunkingConfig.strict`). A document with any `ERROR`-severity finding
  is not indexed at all — no partial/degraded index is ever written. This
  trades "some content is better than none" for "never silently return
  content the pipeline can't vouch for."
- **Embedding and semantic-structuring services are local and
  optional-at-different-layers.** `embedding_server.py` is required for
  `.index()`/`/docs/upload`/`/docs/search`, but not for
  `.preview()`/`.process()`/`/docs/preview`. Semantic structuring
  (`SemanticStructuringConfig`) is off unless a model is explicitly
  configured (`SEMANTIC_MODEL` env var), and any malformed/invalid plan
  from the model is rejected wholesale in favor of the deterministic
  structure — the model can never edit or drop content, only propose
  headings/grouping over blocks it was given by id.
- **OCR is local-only** (pytesseract + local `tesseract` binary) and off
  by default (`OCR_MODE=auto`|`on`|`off`); nothing is sent to a remote
  OCR service.
- **Legacy `.doc` support depends on a local LibreOffice install**
  (`soffice`/`libreoffice` on PATH). Without it, `.doc` uploads produce a
  clean `doc_conversion_unavailable` error rather than being silently
  unsupported; `.docx`/`.pdf`/etc. are entirely unaffected either way.
- **HTML-remnant detection is a heuristic**, not a real HTML parser pass
  over prose — it flags `<...>`-shaped substrings outside inline code as
  `WARNING`s for human review. This produces some false positives on
  Markdown that legitimately uses angle brackets as CLI placeholders
  (e.g. `<inside_local>` in a NAT config example) — 15 such warnings on
  `ip_services.md`, none of them blocking (`WARNING`, not `ERROR`).
- **`docs/.index/` is shared** between the legacy scan and the pipeline's
  own working directories; `scan_and_index()` explicitly excludes its own
  managed dirs (`index_dir`, `processed_dir`, `report_dir`, `source_dir`)
  when walking a parent directory, so scanning `docs/` doesn't also
  re-walk `docs/newly_formatted_docs_review/sources/` under a different
  relative path (which would have produced a second `document_id` for
  the same file).

## Known limitations

- **No live progress stream during upload.** `/docs/upload` runs the
  full extract→normalize→chunk→validate→embed→index pipeline
  synchronously and returns one response with a `stages` object; the UI
  shows a single status line, not a per-stage progress bar. Streaming
  (SSE/WebSocket) would need actual multi-stage checkpoints exposed by
  `DocumentPipeline`, which was out of scope here.
- **PDF/DOCX heading detection is heuristic** (font-size buckets for PDF;
  paragraph style name for DOCX), since neither format carries the kind
  of explicit heading semantics Markdown/HTML do. Arbitrary real-world
  PDFs with inconsistent typography will misclassify some headings as
  body text or vice versa — validation will still catch most structural
  fallout (e.g. skipped heading levels get repaired/logged by
  `normalize.py`), but it is not a guarantee of correct structure.
- **`HybridRanker.rank()` was mutating the shared `DocStore.chunks`
  dicts in place** (writing `score`/`vector_score` onto the same dict
  objects held long-term by the store). Under concurrent `/docs/search`
  requests this is a real race — one request's in-flight scores were
  visible to, and clobbered by, a concurrent request. Fixed to build
  separate score records and return shallow copies; regression-tested in
  `document_converter/tests/test_control.py::test_ranker_does_not_mutate_input_chunk_dicts`.
- **Oversized atomic blocks are a hard stop, not an auto-fix.** There is
  no automatic "split a table by rows" or "split a list into two
  sub-lists" repair — see the 4 unmigrated files above. This is a
  deliberate choice (structural correctness over completeness) but does
  mean some legitimately-useful content needs manual source cleanup
  before it can be indexed.
- **Semantic structuring's cost/latency is unbounded by this code** — a
  large document with hundreds of blocks sends its entire block list
  (id + kind + 200-char preview each) in one Ollama chat call. No
  batching/chunked-planning was implemented; for very large documents
  this may exceed the semantic model's context window before the JSON
  plan validation even gets a chance to reject it.
- **Upload directory naming collisions** are resolved with a
  timestamp suffix (`orchestrator.py`'s `docs_upload`), not a full
  content-addressed scheme — two uploads of files with the same stem
  within the same second (unlikely but not impossible) would still
  collide.
- **Multi-column PDF detection is a two-band heuristic** (clean left/right
  x-position clustering with at least 2 blocks per side). Three-or-more
  column layouts, ragged/ irregular column widths, or a two-column layout
  where one column is mostly images will not reliably trigger it and fall
  back to natural top-to-bottom order (which is what every PDF got before
  this fix, so this is strictly an improvement, not a regression risk).
- **The retrieval smoke-eval is exactly that — a smoke check.** 10
  hand-written questions against the real corpus (80%/100%/100% top-1/3/5
  at time of writing) is enough to catch a gross retrieval regression, not
  a statistically rigorous evaluation. Treat a failing run as a real
  signal; treat a passing run as "nothing obviously broke," not proof of
  retrieval quality.

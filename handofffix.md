# HANDOFFFIX — Restore the 122B model stack after bulk file deletion

Date: 2026-11-03
Project: /home/jrochau/Projects/site (RAG knowledge assistant) + /mnt/hdd1/Live_Projects/llm-ui (122B model server)

## What happened

1. A previous session produced a stale audit and then a list of "safe to delete" files
   in ~/Projects/site. The user deleted files based on that list.
2. After the deletion, the user reports the 122B model "can't run".
3. Verified TODAY before the app exit: the 122B model itself is FINE.
   - Model files intact at:
     /mnt/hdd1/ai-models/llm/huihui-ai/Huihui-Qwen3.5-122B-A10B-abliterated-GGUF/
     Q4_K-GGUF/Q4_K-GGUF-0000{1..8}-of-00008.gguf + mmproj-model-f16.gguf
   - `cd /mnt/hdd1/Live_Projects/llm-ui && LLM_PROFILE=huihui-122b python3 llm-server.py`
     launched cleanly: llama-server listened on 127.0.0.1:8080, llm-server UI on 0.0.0.0:8001,
     and a real /v1/chat/completions call returned a valid completion.
   - Load takes ~80 seconds with NO progress output while mmap'ing tensors.
     It sits at "load_tensors: loading model tensors, this can take a while..."
     Do not Ctrl-C during this window — it looks hung but is not.
4. The user's launch command (from bash history):
     cd /mnt/hdd1/Live_Projects/llm-ui && LLM_PROFILE=huihui-122b python3 llm-server.py
   Profile config: /mnt/hdd1/Live_Projects/llm-ui/model_profiles.py (model_path, mmproj_path,
   launch args: --n-gpu-layers 99 --cpu-moe --parallel 1 --load-mode mmap --jinja).

## Files the previous session said were safe to delete (site project)

- capture_regression_headings.py, extract_numbering_context.py,
  extract_original_headings.py, run_anthology_headings.py (ad-hoc scripts)
- document_converter/tests/test_navigation_links.py (broken test; read the removed
  document_converter.html)
- document_converter/extractors/docling_extractor.py (needed .venv-docling, already gone)
- __pycache__/ dirs, zero-byte logs under document_converter/
- .claude/, site_audit.md
- Runtime state: jobs/, threads/, artifacts/, state.json, sessions/
  (orchestrator.py recreates sessions/, artifacts/, threads/ at startup — see
  orchestrator.py ~line 163; state.json is rewritten by /state and /session/end)

Files that MUST exist for the stack to run (verify first):
- site: orchestrator.py, control.py, ingestion.py, job_runner.py,
  embedding_server.py, embedding_backend.py, whisper_server.py, index.html, app.css,
  manifest.json, requirements.txt, .venv/, document_converter/ (whole package)
- llm-ui: llm-server.py, model_profiles.py
- model shards on /mnt/hdd1 (see paths above)

## Suspected breakage points (check in this order)

1. **Stale processes / port conflicts.** A previous llama-server or llm-server may still
   hold 8080/8001, or a zombie may hold VRAM. Check:
     ps aux | grep -E 'llm-server|llama-server' | grep -v grep
     nvidia-smi
   Kill leftovers, then relaunch.
2. **Impatience during load.** The 80-second silent tensor load is the #1 historical
   cause of "it won't run" (repeated launches in bash history). Wait at least 3 minutes
   before concluding failure.
3. **Deleted runtime state in site/.** state.json / threads/ / jobs/ deletion is
   non-fatal (recreated), but if the orchestrator was mid-write it can leave a truncated
   state.json that breaks JSON parsing. Symptom: orchestrator crashes on startup with
   json.JSONDecodeError. Fix: `echo '{}' > state.json` (loses saved state only).
4. **Deleted file still imported.** If anything beyond the approved list was deleted
   (e.g. a document_converter/ module), imports will fail with ModuleNotFoundError.
   Check: `cd ~/Projects/site && .venv/bin/python -c "import orchestrator"` and
   `.venv/bin/python -c "import embedding_server"`. Restore any missing module from git
   history (the repo was initialized AFTER deletions, so only pre-deletion commits —
   if any — contain them) or from backups under /mnt/hdd1 (site-backup,
   siteBackupVerisons, backup-storage-2026-09-18).
5. **VRAM contention.** The 3060 has 12 GB; the launcher expects ~11.3 GiB free. Desktop
   apps holding VRAM (> ~1 GiB) can make llama.cpp's fit step fail. Check nvidia-smi,
   close GPU-heavy apps, retry.

## Fix procedure

1. `ps aux | grep -E 'llm-server|llama-server|orchestrator|embedding_server' | grep -v grep`
   — kill anything stale.
2. Verify model shards:
   `ls -la /mnt/hdd1/ai-models/llm/huihui-ai/Huihui-Qwen3.5-122B-A10B-abliterated-GGUF/Q4_K-GGUF/`
   must show 8 shards + ../mmproj-model-f16.gguf.
3. Launch: `cd /mnt/hdd1/Live_Projects/llm-ui && LLM_PROFILE=huihui-122b python3 llm-server.py`
   Wait 3 full minutes. Success markers in output:
   - "llama_server: model loaded"
   - "llama_server: listening on http://127.0.0.1:8080"
   - "Uvicorn running on http://0.0.0.0:8001"
4. Verify generation:
   `curl -s -m 300 http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":16}'`
5. If the SITE stack is also wanted:
   `cd ~/Projects/site && source .venv/bin/activate && python embedding_server.py &`
   then `python orchestrator.py` (port 11436). Embedding model BAAI/bge-large-en-v1.5
   is in the local HF cache (~/.cache/huggingface) — unaffected by deletions.
6. If a needed code file turns out to be deleted, restore from (in order of preference):
   - git history in ~/Projects/site/.git (if it was committed before deletion):
     `git log --all --oneline -- <path>` then `git checkout <commit> -- <path>`
   - /mnt/hdd1/site-backup, /mnt/hdd1/siteBackupVerisons, /mnt/hdd1/backup-storage-2026-09-18
   - pip/HF caches for anything reinstallable (.venv: `pip install -r requirements.txt`)

## Known non-issues (do not chase)

- docs/ corpus and docs/.index/ are data; deleting them does not stop the model from running.
- jobs/, threads/, sessions/, artifacts/ deletion only loses history; orchestrator recreates them.
- The pytest collection error from test_navigation_links.py is cosmetic; it does not
  affect runtime.
- Ollama models (qwen3:14b etc.) are separate from the 122B llama.cpp stack and were
  verified working today.

## Verification record (2026-11-03, this session)

- ollama serve: active, `ollama run qwen3:14b` generated successfully.
- torch 2.13.0+cu130, CUDA available on RTX 3060 (12 GB).
- llm-ui 122B launch: model loaded in ~81 s, chat completion returned "OK".
- RAM: 93 GiB total, ~80 GiB available; swap 15 GiB. No OOM kills in journal.

---

# HANDOFFFIX — FORENSIC RESTORE REPORT (2026-11-03, empirical session)

## Verdict

**No file in the 122B launch path was ever deleted or changed.** The stack works
end-to-end on the untouched baseline config. The CUDA illegal-memory-access crash is
NOT REPRODUCIBLE (5/5 real generations passed, incl. image chat). Nothing was restored
because nothing was missing; no config was changed; no build flags were touched.

## Three-bucket diff (site = only place deletions occurred; llm-ui/llama.cpp/model untouched)

Deletion evidence: `~/.local/share/Trash/info/*.trashinfo` (all deletions 2026-11-02
08:36–08:40, ALL under `/home/jrochau/Projects/site`), bash history lines 970–975
(all `rm` commands scoped to `~/Projects/site`), `/mnt/hdd1/.Trash-1000` (only 16 empty
`lost+found` dirs). No llm-ui file appears in any Trash, history command, or manifest diff.

### Bucket 1 — missing AND on the approved list (expected, ignored)
- capture_regression_headings.py, extract_numbering_context.py,
  extract_original_headings.py, run_anthology_baseline.py  (all 4 confirmed in Trash)
- NAME QUESTION RESOLVED: the deleted file was `run_anthology_baseline.py`
  (Trash + .trashinfo proof). `run_anthology_headings.py` was never in the deletion set.
- jobs/, threads/, sessions/, fixtures (document_converter/tests/fixtures), __pycache__ dirs,
  zero-byte logs (orchestrator.log, ingestion.log, orch_debug.log), .claude/,
  run_anthology_*.out/.err, extract_numbering_context.out, regression_headings_after.json
- NOTE: `document_converter/extractors/docling_extractor.py` and
  `document_converter/tests/test_navigation_links.py` were on the approved list but were
  NOT actually deleted — both still present on disk.

### Bucket 2 — missing and NOT on the approved list (prime suspects)
- llm-ui: NONE. llama.cpp: NONE (git status clean at 6fdd0ac89, build/bin untouched
  since Aug 27). Model dir: NONE (8 shards + mmproj intact).
- site: NO code files. Only files that exist in the old `/mnt/hdd1/site-backup` snapshot
  but not today: RAG_BUILD_SOURCE_OF_TRUTH.md, TABLE_SPLIT_TOKEN_ACCOUNTING_ANALYSIS.md,
  document_converter.html, "commands and directories", "runCommands", *.bak/*.orig scratch
  — all predate the Nov 2 event (not in Trash, not in the Nov 2 history window); none are
  imported by anything. `import orchestrator, embedding_server, control, ingestion,
  job_runner` all pass on the site .venv.

### Bucket 3 — modified in the launch path (suspects: changed configs)
- `llm-server.py` + `index.html` had sed edits (max_tokens default 4096 -> 32768) from an
  earlier session. RULED OUT as breakage: `__pycache__/llm-server.cpython-312.pyc` header
  (mtime=1793138916, size=47046) exactly matches the current file, proving llm-server.py is
  byte-identical to the Oct 27 16:08 state that MANIFEST.md records as verified working
  (real tool-call chats at 18:07 same day). model_profiles.py unchanged since Oct 19
  (pyc match). index.html mtime Oct 26 predates the working period.
- NVIDIA driver 580.173.02: installed 2026-07-15 (apt history), i.e. BEFORE the Oct 27
  verified-working state. Ruled out.
- apt: no relevant package changes Oct 27 – Nov 3.

## Root cause

Not a file. All launch-path artifacts are provably identical to the last known-good state,
and the identical configuration passes every test today. The crash the user hit was
environmental/transient. Top suspect: stale-process/VRAM contention — this session began
with a zombie `llm-server.py` (PID 168381) polling a dead llama-server backend, matching
the user's own "address already in use" incident; the first real decode after such a
half-dead relaunch (desktop already holding ~832 MiB of 12 GiB) is when an illegal memory
access surfaces. The fix was process hygiene: kill leftovers, cold-launch clean.

## What was done

1. Killed stale `llm-server.py` (PID 168381). No llama-server was running.
2. Cold launch: `cd /mnt/hdd1/Live_Projects/llm-ui && LLM_PROFILE=huihui-122b python3 llm-server.py`
   (nohup, log: /tmp/forensics/launch1.log). Baseline profile args UNCHANGED:
   `--n-gpu-layers 99 --cpu-moe --parallel 1 --load-mode mmap --jinja --ctx-size 131072`.
3. Ran the full empirical matrix below. No bisect step was needed (baseline passed
   everything), so model_profiles.py was NOT modified.

## Verification (verbatim outputs)

Baseline repro + full verification, all against the running stack:

```
- baseline direct-8080 repro (115 tok prompt, max_tokens 32768) | PASS | 747 tok generated, 16.36 t/s, no CUDA error
    slot print_timing: prompt eval 4015 ms / 115 tokens; eval 47117 ms / 772 tokens (16.36 t/s)
- baseline proxy /api/chat repro (exact UI payload)             | PASS | ~1500 tok, server alive, 0 CUDA errors
- image chat via proxy (mmproj vision, 640x480 test PNG)        | PASS | 357 chunks; reply correctly described
    "deep navy blue background, red rectangle upper-left, green ellipse" — vision confirmed working
- VERIFY direct-8080 2000-token generation                      | PASS |
    COMPLETION TOKENS: 2000 | PROMPT: 58 | finish: length
    content: "# The Architecture of Scalability: A Deep Dive into Mixture-of-Experts Transformers ..."
- VERIFY UI chat 2 via /api/chat (>=256 tok, second in a row)   | PASS |
    usage: {"completion_tokens":2062,"prompt_tokens":37,"total_tokens":2099}
    timings: predicted_per_second 16.41; finish_reason stop
- CUDA errors across entire session: 0 (grep -cE 'CUDA error|illegal memory' launch1.log -> 0)
- Process sanity: exactly one llm-server.py (PID 169845) + one llama-server (PID 169848)
- nvidia-smi: 10231 MiB / 12288 MiB used, util 34% during generation — sane
- 8080 health after every run: {"status":"ok"}
```

## Server left RUNNING for the user (per process rules, stated explicitly)

- `llm-server.py` PID **169845** (UI on http://0.0.0.0:8001)
- `llama-server`  PID **169848** (backend on http://127.0.0.1:8080)
- To stop: `kill 169845 169848`  (killing 169845 alone also reaps llama-server, its child)
- Logs: /tmp/forensics/launch1.log ; bisect log: /tmp/forensics/bisect-log.md

## Operational notes (prevent recurrence)

- Before every launch: `ps aux | grep -E 'llama-server|llm-server' | grep -v grep` — kill
  leftovers by PID first. A zombie llm-server.py holding port 8001 while its llama-server
  child is dead produces exactly the "proxy 200 OK then crash" symptom.
- The ~80 s silent mmap load is normal (warm page cache loads in ~30 s). Never Ctrl-C.
- The "tensor overrides to CPU are used with mmap enabled" warning is informational;
  the verified-working config has always run this way. Do not "fix" it.
- Free VRAM before launch: desktop uses ~0.8 GiB; llama.cpp takes ~9.4 GiB on this profile.

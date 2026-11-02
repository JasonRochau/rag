"""job_runner.py -- build-job agent behind the Job tab.

A job = a description, a fixed build checklist (stages with sub-stage items,
each item optionally carrying a check command), and a job-specific system
prompt. On submit the runner drives the chat model in an agent loop: one JSON
action per turn (write_file / run_command / check_item / ...). The checklist
is the validation list -- the model can never add, remove, reorder, or reword
items; the only way an item gets checked off is the validation gate, which
re-runs the item's check command (or verifies named evidence files) in the
job's output directory. When the agent claims done, final validation
independently re-runs every check command and every discoverable test in the
output directory; the job is marked complete only if everything passes.

The runner is model- and index-agnostic: the orchestrator injects
`chat_completion` (llama-server chat model) and `retrieve` (KB search), so
this module never imports the serving or RAG stack and stays unit-testable
with fakes.

Job layout on disk (JOBS_ROOT/<job_id>/):
    job.json   durable job state (checklist, events, files, validation)
    output/    the only directory the agent can write to / run commands in
"""

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid

logger = logging.getLogger(__name__)

MAX_TURNS = 150                 # agent-loop iterations before the job errors out
ACTION_TIMEOUT_S = 300          # one run_command / check command
TEST_TIMEOUT_S = 900            # one discovered test suite during final validation
OUTPUT_TAIL_CHARS = 6000        # captured stdout+stderr fed back to the agent
READ_BACK_CHARS = 8000          # read_file result fed back to the agent
MAX_EVENTS = 400                # ring buffer for the live-progress event log
CONTEXT_BUDGET_CHARS = 90000    # rough char budget for the conversation (~22k tok)

AGENT_PROTOCOL = """\
You are the build agent for this job. You build the project inside the job's \
output directory, ONE action per turn. Every reply MUST be exactly one JSON \
object and nothing else -- no prose, no markdown fences.

Actions:
{"action": "write_file", "path": "rel/path", "content": "..."}  create or overwrite a file
{"action": "append_file", "path": "rel/path", "content": "..."} append to a file
{"action": "read_file", "path": "rel/path"}                     read a file back
{"action": "list_files"}                                        list everything in the output directory
{"action": "run_command", "command": "..."}                     run a shell command in the output directory
{"action": "check_item", "item_id": "s1.i2", "evidence": ["rel/path"]}  request an item be checked off
{"action": "message", "text": "..."}                            progress note shown to the user
{"action": "finish"}                                            claim the build is done

Rules:
- The checklist is fixed. You cannot add, remove, reorder, or reword items. \
Work through it in order, one stage at a time.
- An item is checked off ONLY by the system: for an item with a check \
command, the system runs that command and checks the item off only when it \
exits 0. For an item without a check command, name the files you produced in \
"evidence"; they must exist in the output directory.
- Before requesting check_item, run the item's check command yourself with \
run_command and fix whatever it reports.
- Use "evidence" on check_item even for items with check commands when you \
produced files -- it is shown in the final report.
- All paths are relative to the output directory. Absolute paths and ".." are \
rejected.
- Reply with "finish" only when every item is checked off. The system then \
re-runs every check command and every test independently; if anything fails, \
the job is marked failed.
"""

CHECK_PREFIXES = ("::", "check:")


# ---------------------------------------------------------------- checklist

def parse_checklist(text):
    """Parse the checklist text the user enters on the Job tab.

    Format (line based):
        ## Stage title            -> starts a stage
        - item text               -> sub-stage item
        - item text :: command    -> item whose check command is `command`
        - item text check: command

    Items listed before the first "##" header fall into a stage named
    "Stage 1". Blank lines and everything else are ignored. Returns a list of
    stages: {"id", "title", "items": [{"id", "title", "check", ...}]}.
    """
    stages = []
    current = None
    stage_no, item_no = 0, 0

    def ensure_stage():
        nonlocal current, stage_no
        if current is None:
            stage_no += 1
            current = {"id": f"s{stage_no}", "title": f"Stage {stage_no}", "items": []}
            stages.append(current)
        return current

    for raw in (text or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("##"):
            stage_no += 1
            item_no = 0
            title = stripped.lstrip("#").strip() or f"Stage {stage_no}"
            current = {"id": f"s{stage_no}", "title": title, "items": []}
            stages.append(current)
            continue
        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if not m:
            continue
        body = m.group(1).strip()
        title, check = body, ""
        for prefix in CHECK_PREFIXES:
            if prefix in body:
                title, _, check = body.partition(prefix)
                title, check = title.strip(), check.strip()
                break
        stage = ensure_stage()
        item_no += 1
        stage["items"].append({
            "id": f"{stage['id']}.i{item_no}",
            "title": title,
            "check": check,
            # pending -> checked (gate passed) | failed (gate/final validation failed)
            "status": "pending",
            "evidence": [],
            "last_check": None,   # {command, exit_code, output, passed, at}
        })
    return stages


def find_item(stages, item_id):
    for stage in stages:
        for item in stage["items"]:
            if item["id"] == item_id:
                return stage, item
    return None, None


def stage_status(stage):
    items = stage["items"]
    if not items:
        return "complete"
    if all(i["status"] == "checked" for i in items):
        return "complete"
    if any(i["status"] == "failed" for i in items):
        return "failed"
    if any(i["status"] == "checked" for i in items):
        return "in_progress"
    return "pending"


def checklist_progress(stages):
    total = sum(len(s["items"]) for s in stages)
    done = sum(1 for s in stages for i in s["items"] if i["status"] == "checked")
    return done, total


# ---------------------------------------------------------------- runner

class JobRunner:
    """Owns job state and the background agent loop. One job runs at a time
    (a single chat model on a single GPU is the bottleneck anyway)."""

    def __init__(self, jobs_root, chat_completion, retrieve=None,
                 model_name="", kb_k=4, max_turns=MAX_TURNS):
        self.jobs_root = jobs_root
        self.chat_completion = chat_completion
        self.retrieve = retrieve          # fn(query) -> list[str] of KB snippets, or None
        self.model_name = model_name
        self.kb_k = kb_k
        self.max_turns = max_turns
        self._lock = threading.Lock()     # guards _running / job dict mutations
        self._running = None              # job_id of the active job
        self._cancel = {}                 # job_id -> threading.Event
        os.makedirs(jobs_root, exist_ok=True)

    # ---- paths / persistence ----

    def _job_dir(self, job_id):
        return os.path.join(self.jobs_root, job_id)

    def _output_dir(self, job_id):
        return os.path.join(self.jobs_root, job_id, "output")

    def _state_path(self, job_id):
        return os.path.join(self.jobs_root, job_id, "job.json")

    def _save(self, job):
        job["updated_at"] = _now()
        path = self._state_path(job["id"])
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    def _load(self, job_id):
        path = self._state_path(job_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    # ---- public API (called by the orchestrator's /jobs endpoints) ----

    def create_job(self, description, checklist_text, system_prompt="", title=""):
        description = (description or "").strip()
        stages = parse_checklist(checklist_text or "")
        if not description:
            return None, "description is required"
        if not stages or not any(s["items"] for s in stages):
            return None, ("checklist is empty -- add at least one stage (## title) "
                          "with at least one item (- item)")
        job_id = "j-" + uuid.uuid4().hex[:12]
        os.makedirs(self._output_dir(job_id), exist_ok=True)
        job = {
            "id": job_id,
            "title": (title or "").strip() or description.splitlines()[0][:80],
            "description": description,
            "system_prompt": (system_prompt or "").strip(),
            "stages": stages,
            "model": self.model_name,
            # queued -> running -> validating -> complete | failed | error | cancelled
            "status": "queued",
            "created_at": _now(),
            "updated_at": _now(),
            "files": [],            # relative paths the agent wrote, in order
            "events": [],           # [{at, kind, text}]
            "turns": 0,
            "validation": None,     # final-validation report
            "error": None,
        }
        self._save(job)
        return job, None

    def start(self, job_id):
        """Launch the agent loop for a queued job in a daemon thread."""
        with self._lock:
            if self._running is not None:
                return False, f"job {self._running} is still running -- one job at a time"
            job = self._load(job_id)
            if job is None:
                return False, "no such job"
            if job["status"] not in ("queued",):
                return False, f"job is {job['status']}, not queued"
            self._running = job_id
            self._cancel[job_id] = threading.Event()
        t = threading.Thread(target=self._run_guarded, args=(job_id,), daemon=True,
                             name=f"job-{job_id}")
        t.start()
        return True, None

    def cancel(self, job_id):
        ev = self._cancel.get(job_id)
        if ev is None:
            return False, "job is not running"
        ev.set()
        return True, None

    def delete(self, job_id):
        with self._lock:
            if self._running == job_id:
                return False, "job is running -- cancel it first"
        shutil.rmtree(self._job_dir(job_id), ignore_errors=True)
        return True, None

    def get(self, job_id):
        return self._load(job_id)

    def list_jobs(self):
        out = []
        if not os.path.isdir(self.jobs_root):
            return out
        for name in os.listdir(self.jobs_root):
            job = self._load(name)
            if job is None:
                continue
            done, total = checklist_progress(job["stages"])
            out.append({
                "id": job["id"], "title": job["title"], "status": job["status"],
                "created_at": job["created_at"], "updated_at": job["updated_at"],
                "items_done": done, "items_total": total,
            })
        out.sort(key=lambda j: j.get("created_at") or "", reverse=True)
        return out

    def snapshot(self, job_id):
        """Full live state for the Job tab's progress view."""
        job = self._load(job_id)
        if job is None:
            return None
        stages = []
        for s in job["stages"]:
            stages.append({
                "id": s["id"], "title": s["title"], "status": stage_status(s),
                "items": [{
                    "id": i["id"], "title": i["title"], "check": i["check"],
                    "status": i["status"], "evidence": i.get("evidence") or [],
                    "last_check": i.get("last_check"),
                } for i in s["items"]],
            })
        done, total = checklist_progress(job["stages"])
        current = None
        for s in stages:
            for i in s["items"]:
                if i["status"] == "pending":
                    current = {"stage": s["title"], "item": i["title"], "item_id": i["id"]}
                    break
            if current:
                break
        return {
            "id": job["id"], "title": job["title"], "status": job["status"],
            "model": job.get("model"), "error": job.get("error"),
            "created_at": job["created_at"], "updated_at": job["updated_at"],
            "description": job["description"], "system_prompt": job["system_prompt"],
            "stages": stages, "current": current,
            "items_done": done, "items_total": total,
            "files": job.get("files") or [],
            "events": (job.get("events") or [])[-120:],
            "turns": job.get("turns", 0),
            "validation": job.get("validation"),
            "output_dir": self._output_dir(job_id),
        }

    # ---- agent loop ----

    def _run_guarded(self, job_id):
        try:
            self._run(job_id)
        except Exception as e:
            logger.exception(f"[job {job_id}] agent loop crashed")
            job = self._load(job_id)
            if job is not None:
                job["status"] = "error"
                job["error"] = str(e)
                self._event(job, "error", f"agent loop crashed: {e}")
                self._save(job)
        finally:
            with self._lock:
                if self._running == job_id:
                    self._running = None
                self._cancel.pop(job_id, None)

    def _event(self, job, kind, text):
        job.setdefault("events", []).append({"at": _now(), "kind": kind, "text": text})
        if len(job["events"]) > MAX_EVENTS:
            del job["events"][:-MAX_EVENTS]

    def _cancelled(self, job_id):
        ev = self._cancel.get(job_id)
        return ev is not None and ev.is_set()

    def _run(self, job_id):
        job = self._load(job_id)
        job["status"] = "running"
        self._event(job, "status", f"job started with {self.model_name or 'chat model'}")
        self._save(job)

        # Knowledge base retrieval: the job builds with retrieval plus the
        # model's own knowledge. One retrieval pass over the description (the
        # most specific statement of intent) up front.
        kb_snippets = []
        if self.retrieve is not None:
            try:
                kb_snippets = self.retrieve(job["description"]) or []
            except Exception as e:
                logger.error(f"[job {job_id}] KB retrieval failed: {e}")
                self._event(job, "warning", f"knowledge base retrieval failed: {e}")
        if kb_snippets:
            self._event(job, "kb", f"retrieved {len(kb_snippets)} knowledge base snippets")

        # The job's system prompt leads every model call, ahead of the
        # description and checklist, exactly as the user entered it.
        system = job["system_prompt"] + "\n\n" + AGENT_PROTOCOL if job["system_prompt"] \
            else AGENT_PROTOCOL
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": self._briefing(job, kb_snippets)}]

        for turn in range(1, self.max_turns + 1):
            if self._cancelled(job_id):
                job["status"] = "cancelled"
                self._event(job, "status", "cancelled by user")
                self._save(job)
                return
            job["turns"] = turn
            self._save(job)

            try:
                raw = self.chat_completion(messages, want_json=True,
                                           temperature=0.2, max_tokens=8192)
            except Exception as e:
                job["status"] = "error"
                job["error"] = str(e)
                self._event(job, "error", f"chat model call failed: {e}")
                self._save(job)
                return

            action, parse_err = _parse_action(raw)
            messages.append({"role": "assistant", "content": raw})
            if action is None:
                result = {"ok": False, "error": f"could not parse one JSON action: {parse_err}. "
                                                "Reply with exactly one JSON object."}
                self._event(job, "warning", "unparseable model reply; asked for one JSON action")
            else:
                result = self._dispatch(job, action)
                if result.pop("_finish", False):
                    messages.append({"role": "user", "content": json.dumps(result)})
                    break
            messages.append({"role": "user", "content": json.dumps(result)})
            self._trim_context(messages, job)
            self._save(job)
        else:
            job["status"] = "error"
            job["error"] = f"hit the {self.max_turns}-turn limit without finishing"
            self._event(job, "error", job["error"])
            self._save(job)
            return

        self._final_validation(job_id)

    def _briefing(self, job, kb_snippets):
        lines = ["# Job description", job["description"], "", "# Build checklist (fixed)"]
        for s in job["stages"]:
            lines.append(f"## {s['id']}: {s['title']}")
            for i in s["items"]:
                suffix = f"  :: {i['check']}" if i["check"] else ""
                lines.append(f"- [{i['id']}] {i['title']}{suffix}")
        lines.append("")
        lines.append(f"Your output directory is empty. Build everything there. "
                     f"Check items off in order with check_item; finish when all "
                     f"{sum(len(s['items']) for s in job['stages'])} items are checked.")
        if kb_snippets:
            lines.append("")
            lines.append("# Knowledge base context (may be relevant)")
            for n, snip in enumerate(kb_snippets, 1):
                lines.append(f"[{n}] {snip}")
        return "\n".join(lines)

    def _trim_context(self, messages, job):
        """Keep the conversation inside the model's context: once over budget,
        collapse the oldest action/result pairs into a compact state recap."""
        total = sum(len(m["content"]) for m in messages)
        if total <= CONTEXT_BUDGET_CHARS or len(messages) <= 8:
            return
        recap = {"role": "user", "content":
                 "[state recap] Older turns were dropped to fit the context window.\n"
                 "Checklist state:\n" + self._checklist_state_text(job) +
                 "\nFiles in output directory:\n" + self._list_files(job)}
        messages[2:4] = [recap]
        while sum(len(m["content"]) for m in messages) > CONTEXT_BUDGET_CHARS and len(messages) > 8:
            del messages[3:5]

    def _checklist_state_text(self, job):
        lines = []
        for s in job["stages"]:
            lines.append(f"## {s['id']}: {s['title']} [{stage_status(s)}]")
            for i in s["items"]:
                lines.append(f"- [{i['id']}] ({i['status']}) {i['title']}")
        return "\n".join(lines)

    # ---- actions ----

    def _resolve_output_path(self, job, rel):
        """Contain an agent-supplied relative path inside the output dir."""
        rel = str(rel or "").strip().lstrip("/")
        if not rel or rel.startswith(".."):
            return None
        root = os.path.realpath(self._output_dir(job["id"]))
        path = os.path.realpath(os.path.join(root, rel))
        if path != root and not path.startswith(root + os.sep):
            return None
        return path

    def _dispatch(self, job, action):
        name = action.get("action")
        if name == "write_file":
            return self._act_write(job, action, append=False)
        if name == "append_file":
            return self._act_write(job, action, append=True)
        if name == "read_file":
            return self._act_read(job, action)
        if name == "list_files":
            return {"ok": True, "files": self._list_files(job)}
        if name == "run_command":
            return self._act_run(job, action)
        if name == "check_item":
            return self._act_check_item(job, action)
        if name == "message":
            text = str(action.get("text", "")).strip()
            if text:
                self._event(job, "note", text)
            return {"ok": True}
        if name == "finish":
            self._event(job, "status", "agent reports the build done -- verifying independently")
            return {"ok": True, "note": "final validation starting", "_finish": True}
        return {"ok": False, "error": f"unknown action {name!r}"}

    def _act_write(self, job, action, append):
        path = self._resolve_output_path(job, action.get("path"))
        if path is None:
            return {"ok": False, "error": "path must be relative and stay inside the output directory"}
        content = action.get("content")
        if not isinstance(content, str):
            return {"ok": False, "error": "content must be a string"}
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        rel = os.path.relpath(path, self._output_dir(job["id"]))
        if rel not in job["files"]:
            job["files"].append(rel)
        self._event(job, "file", f"{'updated' if append else 'wrote'} {rel} ({len(content)} chars)")
        return {"ok": True, "path": rel, "bytes": len(content.encode("utf-8"))}

    def _act_read(self, job, action):
        path = self._resolve_output_path(job, action.get("path"))
        if path is None or not os.path.isfile(path):
            return {"ok": False, "error": "no such file in the output directory"}
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(READ_BACK_CHARS + 1)
        truncated = len(content) > READ_BACK_CHARS
        return {"ok": True, "path": action.get("path"),
                "content": content[:READ_BACK_CHARS], "truncated": truncated}

    def _list_files(self, job):
        root = self._output_dir(job["id"])
        out = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in sorted(filenames):
                out.append(os.path.relpath(os.path.join(dirpath, name), root))
        return "\n".join(out) if out else "(empty)"

    def _run_shell(self, job, command, timeout):
        """Run one shell command inside the job's output directory."""
        try:
            p = subprocess.run(command, shell=True, cwd=self._output_dir(job["id"]),
                               capture_output=True, text=True, timeout=timeout,
                               errors="replace")
            out = (p.stdout or "") + (p.stderr or "")
            return p.returncode, out
        except subprocess.TimeoutExpired:
            return -1, f"(timed out after {timeout}s)"
        except Exception as e:
            return -1, f"(failed to run: {e})"

    def _act_run(self, job, action):
        command = str(action.get("command", "")).strip()
        if not command:
            return {"ok": False, "error": "command is empty"}
        code, out = self._run_shell(job, command, ACTION_TIMEOUT_S)
        tail = out[-OUTPUT_TAIL_CHARS:] if out else ""
        self._event(job, "cmd", f"$ {command} -> exit {code}")
        return {"ok": code == 0, "exit_code": code, "output": tail}

    # ---- the validation gate: the ONLY way an item gets checked off ----

    def _act_check_item(self, job, action):
        item_id = str(action.get("item_id", "")).strip()
        stage, item = find_item(job["stages"], item_id)
        if item is None:
            return {"ok": False, "error": f"no checklist item with id {item_id!r}"}
        if item["status"] == "checked":
            return {"ok": True, "item_id": item_id, "status": "checked",
                    "note": "already checked off"}

        evidence = action.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence = [str(e) for e in evidence][:20]

        if item["check"]:
            code, out = self._run_shell(job, item["check"], ACTION_TIMEOUT_S)
            passed = code == 0
            item["last_check"] = {"command": item["check"], "exit_code": code,
                                  "output": (out or "")[-OUTPUT_TAIL_CHARS:],
                                  "passed": passed, "at": _now()}
            if not passed:
                item["status"] = "failed"
                self._event(job, "check", f"FAIL {item_id}: {item['check']} (exit {code})")
                self._save(job)
                return {"ok": False, "item_id": item_id, "status": "failed",
                        "exit_code": code, "output": item["last_check"]["output"],
                        "error": "check command failed -- fix it and request again"}
        else:
            # No check command: the gate verifies the agent's named evidence
            # files actually exist in the output directory.
            if not evidence:
                return {"ok": False, "item_id": item_id,
                        "error": ("this item has no check command -- pass "
                                  "\"evidence\": [files you produced] and they must exist")}
            missing = [e for e in evidence
                       if self._resolve_output_path(job, e) is None
                       or not os.path.exists(self._resolve_output_path(job, e))]
            if missing:
                item["status"] = "failed"
                item["last_check"] = {"command": "(evidence check)", "exit_code": 1,
                                      "output": "missing: " + ", ".join(missing),
                                      "passed": False, "at": _now()}
                self._event(job, "check", f"FAIL {item_id}: evidence missing ({', '.join(missing)})")
                self._save(job)
                return {"ok": False, "item_id": item_id, "status": "failed",
                        "error": f"evidence files missing: {', '.join(missing)}"}
            item["last_check"] = {"command": "(evidence check)", "exit_code": 0,
                                  "output": "exists: " + ", ".join(evidence),
                                  "passed": True, "at": _now()}

        item["status"] = "checked"
        item["evidence"] = evidence
        self._event(job, "check", f"PASS {item_id}: {item['title']}")
        s_done, s_total = checklist_progress(job["stages"])
        self._event(job, "progress", f"{s_done}/{s_total} items checked off")
        self._save(job)
        return {"ok": True, "item_id": item_id, "status": "checked",
                "items_done": s_done, "items_total": s_total}

    # ---- final validation: trust nothing the agent claimed ----

    def _final_validation(self, job_id):
        job = self._load(job_id)
        job["status"] = "validating"
        self._event(job, "status", "final validation: re-running every check and test")
        self._save(job)

        report = {"at": _now(), "items": [], "tests": [],
                  "flagged": [],   # items the agent marked ready that failed re-verification
                  "passed": False}
        all_ok = True

        for s in job["stages"]:
            for item in s["items"]:
                entry = {"item_id": item["id"], "title": item["title"],
                         "stage": s["title"], "claimed": item["status"] == "checked"}
                if item["check"]:
                    code, out = self._run_shell(job, item["check"], ACTION_TIMEOUT_S)
                    entry.update({"command": item["check"], "exit_code": code,
                                  "output": (out or "")[-OUTPUT_TAIL_CHARS:],
                                  "passed": code == 0})
                else:
                    # Re-verify the evidence the gate accepted earlier.
                    ev = item.get("evidence") or []
                    missing = [e for e in ev
                               if self._resolve_output_path(job, e) is None
                               or not os.path.exists(self._resolve_output_path(job, e))]
                    entry.update({"command": "(evidence check)",
                                  "exit_code": 0 if ev and not missing else 1,
                                  "output": ("exists: " + ", ".join(ev)) if ev and not missing
                                            else ("missing: " + ", ".join(missing) if ev
                                                  else "no evidence recorded"),
                                  "passed": bool(ev) and not missing})
                if entry["claimed"] and not entry["passed"]:
                    report["flagged"].append(item["id"])
                    item["status"] = "failed"
                    item["last_check"] = {"command": entry["command"],
                                          "exit_code": entry["exit_code"],
                                          "output": entry["output"],
                                          "passed": False, "at": _now()}
                if not entry["passed"]:
                    all_ok = False
                report["items"].append(entry)
                self._event(job, "verify",
                            f"{'PASS' if entry['passed'] else 'FAIL'} {item['id']}: {item['title']}")

        # Every test in the job's output directory, discovered independently.
        for test in self._discover_tests(job):
            entry = {"name": test["name"], "command": test["command"]}
            code, out = self._run_shell(job, test["command"], TEST_TIMEOUT_S)
            entry.update({"exit_code": code,
                          "output": (out or "")[-OUTPUT_TAIL_CHARS:],
                          "passed": code == 0})
            if code != 0:
                all_ok = False
            report["tests"].append(entry)
            self._event(job, "verify",
                        f"{'PASS' if entry['passed'] else 'FAIL'} test: {test['name']} (exit {code})")

        report["passed"] = all_ok
        job["validation"] = report
        job["status"] = "complete" if all_ok else "failed"
        if all_ok:
            self._event(job, "status", "job complete -- every check and test passed")
        else:
            failing = [e["item_id"] for e in report["items"] if not e["passed"]]
            failing += [f"test:{t['name']}" for t in report["tests"] if not t["passed"]]
            self._event(job, "status",
                        "job failed validation -- failing: " + ", ".join(failing))
        self._save(job)

    def _discover_tests(self, job):
        """Find runnable test suites in the output directory. Independent of
        anything the agent said: pure filesystem discovery."""
        root = self._output_dir(job["id"])
        tests = []
        run_sh = os.path.join(root, "run_tests.sh")
        if os.path.isfile(run_sh):
            tests.append({"name": "run_tests.sh", "command": "bash run_tests.sh"})
        has_py_tests = False
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("node_modules", ".git", ".venv")]
            if any(re.match(r"^(test_.*\.py|.*_test\.py)$", f) for f in filenames):
                has_py_tests = True
                break
        if has_py_tests and shutil.which("pytest"):
            tests.append({"name": "pytest", "command": "pytest -q"})
        pkg = os.path.join(root, "package.json")
        if os.path.isfile(pkg) and shutil.which("npm"):
            try:
                with open(pkg, "r", encoding="utf-8") as f:
                    scripts = (json.load(f).get("scripts") or {})
                if "test" in scripts:
                    tests.append({"name": "npm test", "command": "npm test --silent"})
            except (OSError, json.JSONDecodeError):
                pass
        return tests


def _parse_action(raw):
    """Extract the single JSON action object from a model reply. Tolerates
    prose around the object; rejects replies with no parseable object."""
    raw = (raw or "").strip()
    candidates = [raw]
    a, b = raw.find("{"), raw.rfind("}")
    if 0 <= a < b:
        candidates.append(raw[a:b + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "action" in obj:
            return obj, None
        return None, "JSON object has no \"action\" key"
    return None, "no JSON object found in reply"


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

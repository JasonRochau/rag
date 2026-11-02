"""
GPU/VRAM lifecycle primitives shared by ingestion (pipeline.py) and the
query path (orchestrator.py): reading real GPU state (nvidia-smi, Ollama's
`/api/ps`), explicitly unloading an Ollama-resident model, and a VRAM
barrier that waits for real evidence of release instead of an arbitrary
sleep (spec "VRAM BARRIER" / "DO NOT ... use arbitrary sleeps as proof of
VRAM release").

Pure process/HTTP calls, no GPU/ML framework dependency of its own (no
`torch` import here) so this module stays importable and unit-testable
without a GPU present -- every external call is best-effort and never
raises: a machine with no NVIDIA GPU, an `nvidia-smi` not on PATH, or
Ollama unreachable must degrade to "couldn't measure", never break the
pipeline it exists to protect.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_NVIDIA_SMI = shutil.which("nvidia-smi")


def nvidia_smi_available() -> bool:
    return _NVIDIA_SMI is not None


def read_gpu_memory_used_mib(gpu_index: int = 0) -> Optional[int]:
    """Real, current GPU memory usage in MiB from `nvidia-smi` -- the same
    tool the acceptance tests measure with, so pipeline-reported numbers
    and externally-verified numbers can never silently disagree. `None`
    when nvidia-smi isn't available (no NVIDIA GPU, not on PATH, or the
    call failed) -- callers must treat that as "couldn't measure", never
    silently treat it as 0."""
    if not _NVIDIA_SMI:
        return None
    try:
        out = subprocess.run(
            [_NVIDIA_SMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu_index)],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return int(out.stdout.strip().splitlines()[0].strip())
    except Exception as e:
        logger.warning(f"[gpu_lifecycle] nvidia-smi memory query failed: {e}")
        return None


def list_gpu_compute_processes() -> List[Dict[str, Any]]:
    """Every process currently holding GPU memory, per `nvidia-smi
    --query-compute-apps` -- used to report *external* GPU users (spec
    "EXTERNAL GPU USERS": identify and report them, never manage or kill
    them -- e.g. a Steam/Battle.net game sharing the same card)."""
    if not _NVIDIA_SMI:
        return []
    try:
        out = subprocess.run(
            [_NVIDIA_SMI, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
        procs: List[Dict[str, Any]] = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                pid, name, mib = int(parts[0]), parts[1], int(parts[2])
            except ValueError:
                continue
            procs.append({"pid": pid, "process_name": name, "used_memory_mib": mib})
        return procs
    except Exception as e:
        logger.warning(f"[gpu_lifecycle] nvidia-smi compute-apps query failed: {e}")
        return []


def ollama_ps(ollama_host: str, timeout: float = 10.0) -> List[Dict[str, Any]]:
    """Currently GPU/CPU-resident Ollama models, per `/api/ps` -- best
    effort, `[]` (never raises) if Ollama is unreachable."""
    try:
        resp = requests.get(ollama_host.rstrip("/") + "/api/ps", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception:
        return []


def is_ollama_model_loaded(ollama_host: str, model: str) -> bool:
    return any(m.get("model") == model or m.get("name") == model for m in ollama_ps(ollama_host))


def unload_ollama_model(ollama_host: str, model: str, timeout: float = 30.0) -> bool:
    """Explicitly releases `model` from Ollama's GPU/CPU residency
    (`keep_alive: 0`) -- the same mechanism already proven for Qwen3-VL
    (originally `visual_analysis.unload_model`), generalized here so the
    answer-generation model (`qwen3:14b`) uses the identical,
    already-validated primitive rather than a second implementation.
    Best-effort: returns False (never raises) if Ollama is unreachable."""
    try:
        resp = requests.post(
            ollama_host.rstrip("/") + "/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=timeout,
        )
        return resp.ok
    except requests.exceptions.RequestException:
        return False


def wait_for_ollama_model_unload(
    ollama_host: str, model: str, timeout_s: float = 30.0, poll_interval_s: float = 1.0,
) -> bool:
    """Polls Ollama's own `/api/ps` (the authoritative "is this model
    actually still resident" signal, not a guess) until `model` no longer
    appears or `timeout_s` elapses. Returns whether it was confirmed gone.
    Never a fixed sleep-and-hope."""
    t0 = time.monotonic()
    while True:
        if not is_ollama_model_loaded(ollama_host, model):
            return True
        if time.monotonic() - t0 >= timeout_s:
            return False
        time.sleep(poll_interval_s)


@dataclass
class VramBarrierResult:
    """Evidence-based proof (or disproof) that a GPU stage's model actually
    released VRAM before the next stage is allowed to start -- never an
    arbitrary sleep (spec "VRAM BARRIER"). `samples` is the full
    memory.used-MiB trace collected while polling, so a failed barrier can
    be diagnosed after the fact instead of just asserted."""
    measured: bool                # False when nvidia-smi wasn't available at all
    released: bool                # True once used_mib <= baseline_mib + tolerance_mib
    baseline_mib: Optional[int]
    tolerance_mib: int
    final_mib: Optional[int]
    elapsed_s: float
    samples: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measured": self.measured, "released": self.released, "baseline_mib": self.baseline_mib,
            "tolerance_mib": self.tolerance_mib, "final_mib": self.final_mib,
            "elapsed_s": round(self.elapsed_s, 2), "samples": self.samples,
        }


def wait_for_vram_release(
    baseline_mib: Optional[int], tolerance_mib: int = 512,
    timeout_s: float = 45.0, poll_interval_s: float = 1.0, gpu_index: int = 0,
) -> VramBarrierResult:
    """Polls real `nvidia-smi` memory.used (never a fixed sleep) until it
    returns to within `tolerance_mib` of `baseline_mib`, or `timeout_s`
    elapses. `baseline_mib=None`/no nvidia-smi (couldn't be measured when
    the stage started) short-circuits to an unmeasured, non-blocking
    result -- a barrier that can't be measured must never silently block
    the pipeline forever, and must never be misreported as "released"
    either; `measured=False` makes that distinction explicit to callers and
    to the final report (spec "DO NOT DEMAND ZERO VRAM" / evidence-based
    tolerance, established from a real measured baseline, never
    hardcoded)."""
    t0 = time.monotonic()
    if baseline_mib is None or not nvidia_smi_available():
        return VramBarrierResult(
            measured=False, released=False, baseline_mib=baseline_mib,
            tolerance_mib=tolerance_mib, final_mib=None, elapsed_s=0.0,
        )
    samples: List[int] = []
    final: Optional[int] = None
    while True:
        used = read_gpu_memory_used_mib(gpu_index)
        elapsed = time.monotonic() - t0
        if used is not None:
            samples.append(used)
            final = used
            if used <= baseline_mib + tolerance_mib:
                return VramBarrierResult(
                    measured=True, released=True, baseline_mib=baseline_mib,
                    tolerance_mib=tolerance_mib, final_mib=final, elapsed_s=elapsed, samples=samples,
                )
        if elapsed >= timeout_s:
            return VramBarrierResult(
                measured=True, released=False, baseline_mib=baseline_mib,
                tolerance_mib=tolerance_mib, final_mib=final, elapsed_s=elapsed, samples=samples,
            )
        time.sleep(poll_interval_s)

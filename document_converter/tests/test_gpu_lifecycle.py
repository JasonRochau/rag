"""
Unit tests for gpu_lifecycle.py (strict sequential GPU model lifecycle
phase). Monkeypatches `subprocess.run`/`requests` -- no real nvidia-smi, no
real Ollama, so this suite runs identically on a machine with no NVIDIA GPU
at all.
"""
from __future__ import annotations

import subprocess

import pytest

from document_converter import gpu_lifecycle


class _FakeCompleted:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


class _FakeResponse:
    def __init__(self, json_data, ok=True, status_code=200):
        self._json = json_data
        self.ok = ok
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


# ---------------------------------------------------------------------- #
# read_gpu_memory_used_mib / nvidia_smi_available

def test_read_gpu_memory_used_mib_parses_nvidia_smi_output(monkeypatch):
    monkeypatch.setattr(gpu_lifecycle, "_NVIDIA_SMI", "/usr/bin/nvidia-smi")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompleted(stdout="3122\n"))
    assert gpu_lifecycle.read_gpu_memory_used_mib() == 3122


def test_read_gpu_memory_used_mib_returns_none_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(gpu_lifecycle, "_NVIDIA_SMI", None)
    assert gpu_lifecycle.read_gpu_memory_used_mib() is None
    assert gpu_lifecycle.nvidia_smi_available() is False


def test_read_gpu_memory_used_mib_returns_none_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(gpu_lifecycle, "_NVIDIA_SMI", "/usr/bin/nvidia-smi")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompleted(stdout="", returncode=1))
    assert gpu_lifecycle.read_gpu_memory_used_mib() is None


def test_read_gpu_memory_used_mib_never_raises_on_unexpected_error(monkeypatch):
    monkeypatch.setattr(gpu_lifecycle, "_NVIDIA_SMI", "/usr/bin/nvidia-smi")

    def boom(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr(subprocess, "run", boom)
    assert gpu_lifecycle.read_gpu_memory_used_mib() is None


# ---------------------------------------------------------------------- #
# list_gpu_compute_processes -- external GPU users report (never managed)

def test_list_gpu_compute_processes_parses_csv_rows(monkeypatch):
    monkeypatch.setattr(gpu_lifecycle, "_NVIDIA_SMI", "/usr/bin/nvidia-smi")
    out = "6596, WoWClassic.exe, 1681\n3798, cosmic-app-library, 70\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompleted(stdout=out))
    procs = gpu_lifecycle.list_gpu_compute_processes()
    assert procs == [
        {"pid": 6596, "process_name": "WoWClassic.exe", "used_memory_mib": 1681},
        {"pid": 3798, "process_name": "cosmic-app-library", "used_memory_mib": 70},
    ]


def test_list_gpu_compute_processes_empty_when_unavailable(monkeypatch):
    monkeypatch.setattr(gpu_lifecycle, "_NVIDIA_SMI", None)
    assert gpu_lifecycle.list_gpu_compute_processes() == []


# ---------------------------------------------------------------------- #
# ollama_ps / is_ollama_model_loaded / unload_ollama_model

def test_ollama_ps_returns_models_list(monkeypatch):
    monkeypatch.setattr(
        gpu_lifecycle.requests, "get",
        lambda url, timeout=None: _FakeResponse({"models": [{"model": "qwen3:14b"}]}),
    )
    assert gpu_lifecycle.ollama_ps("http://x") == [{"model": "qwen3:14b"}]


def test_ollama_ps_returns_empty_list_when_unreachable(monkeypatch):
    def boom(url, timeout=None):
        raise ConnectionError("down")

    monkeypatch.setattr(gpu_lifecycle.requests, "get", boom)
    assert gpu_lifecycle.ollama_ps("http://x") == []


def test_is_ollama_model_loaded_matches_model_or_name_field(monkeypatch):
    monkeypatch.setattr(
        gpu_lifecycle, "ollama_ps",
        lambda host, timeout=10.0: [{"model": "qwen3-vl:8b-instruct-q4_K_M"}],
    )
    assert gpu_lifecycle.is_ollama_model_loaded("http://x", "qwen3-vl:8b-instruct-q4_K_M") is True
    assert gpu_lifecycle.is_ollama_model_loaded("http://x", "qwen3:14b") is False


def test_unload_ollama_model_posts_keep_alive_zero(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse({})

    monkeypatch.setattr(gpu_lifecycle.requests, "post", fake_post)
    assert gpu_lifecycle.unload_ollama_model("http://x", "qwen3:14b") is True
    assert captured["url"] == "http://x/api/generate"
    assert captured["json"] == {"model": "qwen3:14b", "prompt": "", "keep_alive": 0}


def test_unload_ollama_model_best_effort_false_when_unreachable(monkeypatch):
    import requests as _requests

    def boom(url, json=None, timeout=None):
        raise _requests.exceptions.RequestException("down")

    monkeypatch.setattr(gpu_lifecycle.requests, "post", boom)
    assert gpu_lifecycle.unload_ollama_model("http://x", "qwen3:14b") is False


def test_wait_for_ollama_model_unload_returns_true_immediately_when_already_gone(monkeypatch):
    monkeypatch.setattr(gpu_lifecycle, "is_ollama_model_loaded", lambda host, model: False)
    assert gpu_lifecycle.wait_for_ollama_model_unload("http://x", "qwen3:14b", timeout_s=5, poll_interval_s=0.01) is True


def test_wait_for_ollama_model_unload_times_out_when_still_loaded(monkeypatch):
    monkeypatch.setattr(gpu_lifecycle, "is_ollama_model_loaded", lambda host, model: True)
    assert gpu_lifecycle.wait_for_ollama_model_unload("http://x", "qwen3:14b", timeout_s=0.05, poll_interval_s=0.01) is False


# ---------------------------------------------------------------------- #
# wait_for_vram_release -- the VRAM barrier itself

def test_wait_for_vram_release_unmeasured_when_baseline_is_none():
    result = gpu_lifecycle.wait_for_vram_release(None)
    assert result.measured is False
    assert result.released is False
    assert result.final_mib is None


def test_wait_for_vram_release_unmeasured_when_nvidia_smi_unavailable(monkeypatch):
    monkeypatch.setattr(gpu_lifecycle, "nvidia_smi_available", lambda: False)
    result = gpu_lifecycle.wait_for_vram_release(3000)
    assert result.measured is False
    assert result.released is False


def test_wait_for_vram_release_succeeds_on_first_poll_within_tolerance(monkeypatch):
    monkeypatch.setattr(gpu_lifecycle, "nvidia_smi_available", lambda: True)
    monkeypatch.setattr(gpu_lifecycle, "read_gpu_memory_used_mib", lambda gpu_index=0: 3100)
    result = gpu_lifecycle.wait_for_vram_release(3000, tolerance_mib=512, timeout_s=5, poll_interval_s=0.01)
    assert result.measured is True
    assert result.released is True
    assert result.final_mib == 3100
    assert result.samples == [3100]


def test_wait_for_vram_release_polls_until_released(monkeypatch):
    readings = iter([9600, 5200, 3050])
    monkeypatch.setattr(gpu_lifecycle, "nvidia_smi_available", lambda: True)
    monkeypatch.setattr(gpu_lifecycle, "read_gpu_memory_used_mib", lambda gpu_index=0: next(readings))
    result = gpu_lifecycle.wait_for_vram_release(3000, tolerance_mib=100, timeout_s=5, poll_interval_s=0.001)
    assert result.released is True
    assert result.samples == [9600, 5200, 3050]


def test_wait_for_vram_release_times_out_and_reports_not_released(monkeypatch):
    monkeypatch.setattr(gpu_lifecycle, "nvidia_smi_available", lambda: True)
    monkeypatch.setattr(gpu_lifecycle, "read_gpu_memory_used_mib", lambda gpu_index=0: 9700)
    result = gpu_lifecycle.wait_for_vram_release(3000, tolerance_mib=100, timeout_s=0.03, poll_interval_s=0.01)
    assert result.measured is True
    assert result.released is False
    assert result.final_mib == 9700


def test_vram_barrier_result_to_dict_round_trips_fields():
    result = gpu_lifecycle.wait_for_vram_release(None)
    d = result.to_dict()
    assert d["measured"] is False
    assert d["baseline_mib"] is None
    assert "samples" in d

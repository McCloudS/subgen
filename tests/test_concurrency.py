"""whisper.cpp thread sizing: WHISPER_THREADS wins; otherwise cores are shared between concurrent jobs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

import subgen


@pytest.mark.parametrize("cpus,concurrent,expected", [
    (24, 1, 16),   # capped at 16
    (24, 2, 12),
    (24, 3, 8),
    (8, 2, 4),     # never below 4
    (2, 1, 4),
])
def test_auto_threads_share_cores(monkeypatch, cpus, concurrent, expected):
    monkeypatch.setattr(subgen, "_whisper_threads_env", "")
    monkeypatch.setattr(subgen, "_available_cpus", lambda: cpus)
    monkeypatch.setattr(subgen, "concurrent_transcriptions", concurrent)
    assert subgen.whispercpp_threads() == expected


def test_whisper_threads_env_wins(monkeypatch):
    monkeypatch.setattr(subgen, "_whisper_threads_env", "6")
    monkeypatch.setattr(subgen, "_available_cpus", lambda: 24)
    assert subgen.whispercpp_threads() == 6


def test_whispercpp_command_uses_thread_count(monkeypatch):
    monkeypatch.setattr(subgen, "whispercpp_threads", lambda: 11)
    cmd = subgen._whispercpp_base_cmd("/m.bin", "/a.wav", "/tmp/o")
    assert cmd[cmd.index("-t") + 1] == "11"


def test_faster_whisper_on_gpu_preloads_torch_cuda_libraries(monkeypatch):
    # Without this, faster-whisper on CUDA fails with "libcublas.so.12 is not found" in images
    # where cuBLAS only ships inside torch's wheels (seen with dev code in mccloud/subgen:latest).
    imported = []
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *a, **k):
        if name == "torch":
            imported.append(name)
            return object()
        return real_import(name, *a, **k)
    monkeypatch.setattr("builtins.__import__", fake_import)
    for device, expected in (("cuda", ["torch"]), ("cpu", [])):
        imported.clear()
        monkeypatch.setattr(subgen, "transcribe_device", device)
        subgen._preload_cuda_libraries()
        assert imported == expected


def test_faster_whisper_language_detection_handles_both_return_shapes():
    # faster-whisper 1.2 returns (lang, prob, all_probs); older releases return (lang, prob).
    assert subgen._fw_language(("de", 0.9, [("de", 0.9)])) == "de"
    assert subgen._fw_language(("fr", 0.8)) == "fr"

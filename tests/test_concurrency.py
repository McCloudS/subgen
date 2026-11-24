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

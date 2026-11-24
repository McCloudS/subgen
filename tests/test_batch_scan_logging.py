"""
Tests for issue #353: /batch scans must report what they found instead of going
silent when a folder is missing or unreadable.
"""
import logging
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

import subgen
from subgen import transcribe_existing


@pytest.fixture(autouse=True)
def _no_monitor(monkeypatch):
    monkeypatch.setattr(subgen, "monitor", False)
    monkeypatch.setattr(subgen, "use_path_mapping", False)


def test_missing_folder_warns_and_reports(tmp_path, caplog):
    missing = str(tmp_path / "nope")
    with caplog.at_level(logging.INFO):
        summary = transcribe_existing(missing)
    assert summary["errors"] == 1
    assert summary["folders"][0]["exists"] is False
    assert "does not exist inside the container" in caplog.text


def test_summary_counts_found_and_queued(tmp_path, caplog):
    (tmp_path / "a.mkv").touch()
    (tmp_path / "b.mkv").touch()
    (tmp_path / "notes.txt").touch()

    def fake_queue(path, *args, **kwargs):
        return path.endswith("a.mkv")

    with patch.object(subgen, "gen_subtitles_queue", side_effect=fake_queue), caplog.at_level(logging.INFO):
        summary = transcribe_existing(str(tmp_path))
    assert summary["files_found"] == 3
    assert summary["queued"] == 1
    assert summary["errors"] == 0
    assert "3 files found, 1 queued" in caplog.text


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="root can read any folder")
def test_unreadable_subfolder_warns(tmp_path, caplog):
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "a.mkv").touch()
    locked.chmod(0)
    try:
        with patch.object(subgen, "gen_subtitles_queue", return_value=False), caplog.at_level(logging.INFO):
            summary = transcribe_existing(str(tmp_path))
    finally:
        locked.chmod(0o755)
    assert summary["errors"] == 1
    assert "Check that PUID/PGID can read this folder" in caplog.text


def test_unreadable_folder_warns_via_onerror(tmp_path, caplog):
    """Simulate a permission error from os.walk regardless of the test user."""
    def walk_with_error(path, onerror=None, **kwargs):
        err = PermissionError(13, "Permission denied", str(path))
        if onerror:
            onerror(err)
        return iter(())

    with patch.object(subgen.os, "walk", side_effect=walk_with_error), caplog.at_level(logging.INFO):
        summary = transcribe_existing(str(tmp_path))
    assert summary["errors"] == 1
    assert "Permission denied" in caplog.text

"""
Integration tests: full request → queue flow (Whisper still mocked).

These tests verify that a webhook POST actually results in a task appearing
in the task_queue, and that path mapping is applied end-to-end.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
import subgen
from subgen import app, task_queue, DeduplicatedQueue
from language_code import LanguageCode


@pytest.fixture(autouse=True)
def reset_queue(monkeypatch):
    """Give each test a clean queue so tasks from one test don't affect another."""
    fresh = DeduplicatedQueue()
    monkeypatch.setattr(subgen, "task_queue", fresh)
    yield fresh


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestTautulliQueuesTask:
    def test_webhook_results_in_queued_task(self, client, monkeypatch, reset_queue):
        monkeypatch.setattr(subgen, "procaddedmedia", True)
        # Prevent has_audio() from touching the file system
        monkeypatch.setattr(subgen, "has_audio", lambda path: True)
        monkeypatch.setattr(subgen, "should_skip_file", lambda path, lang: False)
        monkeypatch.setattr(subgen, "choose_transcribe_language", lambda path, lang: lang)
        monkeypatch.setattr(subgen, "should_whiser_detect_audio_language", False)

        client.post(
            "/tautulli",
            headers={"source": "Tautulli"},
            json={"event": "added", "file": "/media/show.mkv"},
        )

        # Give the synchronous handler a moment then check queue
        queued = reset_queue.get_queued_tasks()
        assert len(queued) == 1
        assert "/media/show.mkv" in queued[0]


class TestPathMappingApplied:
    def test_container_path_is_remapped(self, client, monkeypatch, reset_queue):
        """Container path /tv → host path /Volumes/TV must be applied before queuing."""
        monkeypatch.setattr(subgen, "procaddedmedia", True)
        monkeypatch.setattr(subgen, "use_path_mapping", True)
        monkeypatch.setattr(subgen, "path_mapping_from", "/tv")
        monkeypatch.setattr(subgen, "path_mapping_to", "/Volumes/TV")
        monkeypatch.setattr(subgen, "has_audio", lambda path: True)
        monkeypatch.setattr(subgen, "should_skip_file", lambda path, lang: False)
        monkeypatch.setattr(subgen, "choose_transcribe_language", lambda path, lang: lang)
        monkeypatch.setattr(subgen, "should_whiser_detect_audio_language", False)

        client.post(
            "/tautulli",
            headers={"source": "Tautulli"},
            json={"event": "added", "file": "/tv/show.mkv"},
        )

        queued = reset_queue.get_queued_tasks()
        assert len(queued) == 1
        assert "/Volumes/TV/show.mkv" in queued[0]


class TestEmbyQueuesTask:
    def test_emby_library_new_adds_to_queue(self, client, monkeypatch, reset_queue):
        monkeypatch.setattr(subgen, "procaddedmedia", True)
        monkeypatch.setattr(subgen, "has_audio", lambda path: True)
        monkeypatch.setattr(subgen, "should_skip_file", lambda path, lang: False)
        monkeypatch.setattr(subgen, "choose_transcribe_language", lambda path, lang: lang)
        monkeypatch.setattr(subgen, "should_whiser_detect_audio_language", False)

        data = {"Event": "library.new", "Item": {"Path": "/media/movie.mkv"}}
        client.post("/emby", data={"data": json.dumps(data)})

        queued = reset_queue.get_queued_tasks()
        assert len(queued) == 1
        assert "/media/movie.mkv" in queued[0]

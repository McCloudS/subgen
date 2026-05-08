"""Tests for DeduplicatedQueue."""
import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

# conftest.py has already patched sys.modules; safe to import subgen now
import subgen
from subgen import DeduplicatedQueue


@pytest.fixture()
def q():
    return DeduplicatedQueue()


def _task(path, task_type="transcribe"):
    return {"path": path, "type": task_type, "transcribe_or_translate": "transcribe", "force_language": None}


class TestPut:
    def test_new_item_returns_true(self, q):
        assert q.put(_task("/a/b.mkv")) is True

    def test_duplicate_while_queued_returns_false(self, q):
        q.put(_task("/a/b.mkv"))
        assert q.put(_task("/a/b.mkv")) is False

    def test_different_paths_both_accepted(self, q):
        assert q.put(_task("/a/b.mkv")) is True
        assert q.put(_task("/a/c.mkv")) is True

    def test_duplicate_while_processing_returns_false(self, q):
        q.put(_task("/a/b.mkv"))
        q.get()  # moves to _processing
        assert q.put(_task("/a/b.mkv")) is False


class TestPriority:
    def test_detect_language_before_transcribe(self, q):
        q.put({"path": "/a.mkv", "type": "transcribe", "transcribe_or_translate": "transcribe", "force_language": None})
        q.put({"path": "/b.mkv", "type": "detect_language"})
        first = q.get()
        assert first["type"] == "detect_language"

    def test_asr_before_transcribe(self, q):
        q.put({"path": "/a.mkv", "type": "transcribe", "transcribe_or_translate": "transcribe", "force_language": None})
        q.put({"path": "asr-abc", "type": "asr"})
        first = q.get()
        assert first["type"] == "asr"

    def test_full_priority_order(self, q):
        q.put({"path": "/c.mkv", "type": "transcribe", "transcribe_or_translate": "transcribe", "force_language": None})
        q.put({"path": "asr-xyz", "type": "asr"})
        q.put({"path": "/d.mkv", "type": "detect_language"})
        assert q.get()["type"] == "detect_language"
        assert q.get()["type"] == "asr"
        assert q.get()["type"] == "transcribe"


class TestMarkDone:
    def test_mark_done_allows_requeue(self, q):
        task = _task("/a/b.mkv")
        q.put(task)
        popped = q.get()
        q.mark_done(popped)
        assert q.put(task) is True

    def test_without_mark_done_blocks_requeue(self, q):
        task = _task("/a/b.mkv")
        q.put(task)
        q.get()  # now _processing, but not marked done
        assert q.put(task) is False


class TestIsActive:
    def test_active_while_queued(self, q):
        q.put(_task("/a/b.mkv"))
        assert q.is_active("/a/b.mkv") is True

    def test_active_while_processing(self, q):
        q.put(_task("/a/b.mkv"))
        q.get()
        assert q.is_active("/a/b.mkv") is True

    def test_not_active_when_absent(self, q):
        assert q.is_active("/a/b.mkv") is False

    def test_not_active_after_done(self, q):
        task = _task("/a/b.mkv")
        q.put(task)
        popped = q.get()
        q.mark_done(popped)
        assert q.is_active("/a/b.mkv") is False


class TestStatusSnapshots:
    def test_get_queued_tasks_returns_list(self, q):
        q.put(_task("/a.mkv"))
        q.put(_task("/b.mkv"))
        assert len(q.get_queued_tasks()) == 2

    def test_get_processing_tasks_returns_list(self, q):
        q.put(_task("/a.mkv"))
        q.get()
        assert len(q.get_processing_tasks()) == 1


class TestConcurrency:
    def test_concurrent_puts_only_one_wins(self, q):
        successes = []
        lock = threading.Lock()

        def try_put():
            result = q.put(_task("/shared.mkv"))
            with lock:
                successes.append(result)

        threads = [threading.Thread(target=try_put) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(successes) == 1, "Exactly one put should succeed for the same path"

#!/usr/bin/env python3
"""
Direct test for LEAK #4: DeduplicatedQueue Internal Sets

This standalone test imports only the DeduplicatedQueue class logic
to test memory behavior without requiring all subgen dependencies.
"""

import sys
import time
from threading import Lock
import queue


# Mock the DeduplicatedQueue class directly here
class DeduplicatedQueue(queue.PriorityQueue):
    """Queue that prevents duplicates, handles priority, and tracks status."""

    def __init__(self):
        super().__init__()
        self._queued = set()  # Tracks task IDs waiting in queue
        self._processing = set()  # Tracks task IDs currently being handled
        self._lock = Lock()

    def put(self, item, block=True, timeout=None):
        with self._lock:
            task_id = item["path"]
            if task_id not in self._queued and task_id not in self._processing:
                # Priority: 0 (Detect), 1 (ASR), 2 (Transcribe)
                task_type = item.get("type", "transcribe")
                priority = (
                    0
                    if task_type == "detect_language"
                    else (1 if task_type == "asr" else 2)
                )

                # PriorityQueue requires a tuple: (priority, tie_breaker, item)
                super().put((priority, time.time(), item), block, timeout)
                self._queued.add(task_id)
                return True
            return False

    def get(self, block=True, timeout=None):
        # PriorityQueue returns the tuple, we want just the item
        priority, timestamp, item = super().get(block, timeout)
        with self._lock:
            task_id = item["path"]
            self._queued.discard(task_id)
            self._processing.add(task_id)
        return item

    def mark_done(self, item):
        with self._lock:
            task_id = item["path"]
            self._processing.discard(task_id)

    def is_idle(self):
        with self._lock:
            return self.empty() and len(self._processing) == 0

    def is_active(self, task_id):
        """Checks if a task_id is currently queued or processing."""
        with self._lock:
            return task_id in self._queued or task_id in self._processing

    def get_queued_tasks(self):
        with self._lock:
            return list(self._queued)

    def get_processing_tasks(self):
        with self._lock:
            return list(self._processing)


def test_internal_sets_growth():
    """
    Test if internal sets grow unbounded after processing many tasks.
    """
    import tracemalloc

    tracemalloc.start()
    baseline = tracemalloc.take_snapshot()

    q = DeduplicatedQueue()

    # Process 1000 tasks
    for i in range(1000):
        task = {
            "path": f"/media/test_file_{i}.mp4",
            "type": "transcribe",
            "transcribe_or_translate": "transcribe",
            "force_language": None,
        }
        q.put(task)
        item = q.get()
        q.mark_done(item)

    # Check internal state
    queued_size = len(q._queued)
    processing_size = len(q._processing)

    current = tracemalloc.take_snapshot()
    stats = current.compare_to(baseline, "lineno")
    total_memory = sum(stat.size_diff for stat in stats) / 1024 / 1024

    print(f"\n{'=' * 60}")
    print(f"LEAK #4 TEST: DeduplicatedQueue Internal Sets")
    print(f"{'=' * 60}")
    print(f"Tasks processed: 1000")
    print(f"Queued set size: {queued_size}")
    print(f"Processing set size: {processing_size}")
    print(f"Queue empty: {q.empty()}")
    print(f"Memory growth: {total_memory:.2f} MB")
    print(f"{'=' * 60}")

    # Evaluate results
    if queued_size == 0 and processing_size == 0:
        print(f"✅ PASS: Internal sets properly cleaned (no leak detected)")
        return True
    else:
        print(f"❌ FAIL: Internal sets not cleaned - LEAK CONFIRMED!")
        print(f"   - {queued_size} items in _queued")
        print(f"   - {processing_size} items in _processing")
        return False


def test_mark_done_not_called():
    """
    Test what happens if mark_done() is not called (exception scenario).
    """
    q = DeduplicatedQueue()

    # Add tasks but don't mark them done
    for i in range(100):
        task = {
            "path": f"/media/test_file_{i}.mp4",
            "type": "transcribe",
            "transcribe_or_translate": "transcribe",
            "force_language": None,
        }
        q.put(task)
        item = q.get()
        # Deliberately NOT calling mark_done()

    processing_size = len(q._processing)

    print(f"\n{'=' * 60}")
    print(f"LEAK #4 TEST: mark_done() Not Called Edge Case")
    print(f"{'=' * 60}")
    print(f"Tasks processed: 100")
    print(f"Processing set size: {processing_size}")
    print(f"Expected: 100 items stuck in _processing")
    print(f"{'=' * 60}")

    if processing_size == 100:
        print(f"❌ LEAK CONFIRMED: All 100 tasks stuck in _processing set")
        return False
    else:
        print(f"✅ Unexpected: _processing was cleaned somehow")
        return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING LEAK #4: DeduplicatedQueue Internal Sets")
    print("=" * 60)

    test1_pass = test_internal_sets_growth()
    test2_pass = test_mark_done_not_called()

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    if not test1_pass or not test2_pass:
        print(f"❌ LEAK #4 IS REAL - Memory leak confirmed")
        print(f"   Fix is required!")
        sys.exit(1)
    else:
        print(f"✅ NO LEAK DETECTED - May be theoretical issue")
        sys.exit(0)

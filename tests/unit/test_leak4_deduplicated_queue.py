"""
Memory Leak Test: LEAK #4 - DeduplicatedQueue Internal Sets Never Cleaned

PROBLEM:
The DeduplicatedQueue class maintains two internal sets (_queued and _processing)
that track task IDs. While mark_done() removes items from _processing, there's no
cleanup mechanism for completed tasks, and edge cases exist where mark_done()
might not be called. This causes unbounded memory growth.

EXPECTED BEHAVIOR:
- Internal sets should not grow unbounded
- Completed tasks should be removed from tracking
- Memory usage should remain stable across many operations

IMPACT:
- Each task ID (string ~20-100 chars): ~100 bytes
- For 10,000 transcriptions: ~500KB-2MB leaked
- In 24/7 batch processing: 10MB-50MB per day

TEST STRATEGY:
1. Create DeduplicatedQueue instance
2. Add many tasks with unique IDs
3. Process and mark them done
4. Measure memory growth of internal sets
5. Verify sets don't grow unbounded
"""

import pytest
import sys
import time
from pathlib import Path

# Add parent directory to path to import subgen
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.memory_utils import MemoryProfiler, LeakDetector
from subgen import DeduplicatedQueue


@pytest.fixture
def queue():
    """Create a fresh DeduplicatedQueue for each test."""
    return DeduplicatedQueue()


@pytest.mark.unit
@pytest.mark.memory_leak
class TestDeduplicatedQueueLeaks:
    """Test suite for DeduplicatedQueue memory leaks."""

    def test_internal_sets_grow_unbounded(self, queue):
        """
        LEAK TEST: Internal _queued and _processing sets never shrink.

        This test demonstrates that the internal sets accumulate task IDs
        and never clean up completed tasks, leading to unbounded growth.
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=0.5)

        # Baseline measurement
        profiler.snapshot("baseline")

        # Simulate processing 1000 tasks
        iterations = 1000
        for i in range(iterations):
            task = {
                "path": f"/media/test_file_{i}.mp4",
                "type": "transcribe",
                "transcribe_or_translate": "transcribe",
                "force_language": None,
            }

            # Add to queue
            queue.put(task)

            # Immediately retrieve (simulating fast processing)
            item = queue.get()

            # Mark as done
            queue.mark_done(item)

        profiler.snapshot("after_1000_tasks")

        # Check internal sets size
        queued_size = len(queue._queued)
        processing_size = len(queue._processing)

        # After processing, both sets should be empty or very small
        # If they're not, we have a leak
        print(f"\n=== DeduplicatedQueue Internal State ===")
        print(f"Queued set size: {queued_size}")
        print(f"Processing set size: {processing_size}")
        print(f"Queue empty: {queue.empty()}")

        # Memory analysis
        growth = profiler.compare("baseline", "after_1000_tasks")
        print(f"\nMemory growth: {growth['increase_mb']:.2f} MB")

        # EXPECTED FAILURE: This test should show that memory grows
        # because the sets are not properly cleaned up
        assert queued_size == 0, "Queued set should be empty after processing"
        assert processing_size == 0, "Processing set should be empty after mark_done"

        # Memory should not grow significantly (< 0.5 MB for 1000 simple strings)
        leak_detected = leak_detector.check(growth["increase_mb"])
        assert not leak_detected, (
            f"Memory leak detected: {growth['increase_mb']:.2f} MB growth"
        )

    def test_mark_done_not_called_edge_case(self, queue):
        """
        LEAK TEST: If mark_done() is not called due to exception, tasks accumulate.

        This simulates scenarios where task processing fails and mark_done()
        might not be reached.
        """
        profiler = MemoryProfiler()

        profiler.snapshot("baseline")

        # Add tasks but don't mark them done (simulating exception scenario)
        for i in range(500):
            task = {
                "path": f"/media/test_file_{i}.mp4",
                "type": "transcribe",
                "transcribe_or_translate": "transcribe",
                "force_language": None,
            }
            queue.put(task)
            item = queue.get()
            # Deliberately NOT calling mark_done() to simulate exception path

        profiler.snapshot("after_incomplete_processing")

        processing_size = len(queue._processing)
        print(f"\n=== Edge Case: mark_done() Not Called ===")
        print(f"Processing set size: {processing_size}")
        print(f"Expected: 500 items stuck in _processing set")

        # This demonstrates the leak: all 500 tasks are stuck in _processing
        assert processing_size == 500, "All tasks should be stuck in _processing"

        growth = profiler.compare("baseline", "after_incomplete_processing")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        # Even 500 task IDs shouldn't cause huge memory leak, but it demonstrates
        # that without proper cleanup, this accumulates
        assert growth["increase_mb"] < 1.0, (
            "Memory growth should be minimal for 500 strings"
        )

    def test_task_id_collision_handling(self, queue):
        """
        LEAK TEST: Verify duplicate task IDs are properly handled.

        The queue is supposed to deduplicate, but we need to ensure the
        internal sets properly track this.
        """
        task1 = {
            "path": "/media/same_file.mp4",
            "type": "transcribe",
            "transcribe_or_translate": "transcribe",
            "force_language": None,
        }

        task2 = {
            "path": "/media/same_file.mp4",  # Same path = same task ID
            "type": "transcribe",
            "transcribe_or_translate": "transcribe",
            "force_language": None,
        }

        # Add first task
        added1 = queue.put(task1)
        assert added1 is True, "First task should be added"

        # Try to add duplicate
        added2 = queue.put(task2)
        assert added2 is False, "Duplicate task should be rejected"

        # Check internal state
        queued_size = len(queue._queued)
        assert queued_size == 1, "Only one task should be in _queued set"

        # Process the task
        item = queue.get()
        processing_size = len(queue._processing)
        assert processing_size == 1, "Task should be in _processing"
        assert len(queue._queued) == 0, "Task should be removed from _queued"

        # Mark done
        queue.mark_done(item)
        assert len(queue._processing) == 0, "Task should be removed from _processing"

    def test_concurrent_operations_memory_growth(self, queue):
        """
        LEAK TEST: High-frequency operations should not cause memory bloat.

        Simulates rapid queue operations like those in 24/7 batch processing.
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=2.0)

        profiler.snapshot("baseline")

        # Simulate high-frequency operations
        iterations = 2000
        for i in range(iterations):
            task = {
                "path": f"/media/batch_{i % 100}.mp4",  # Reuse some paths
                "type": "transcribe",
                "transcribe_or_translate": "transcribe",
                "force_language": None,
            }

            if queue.put(task):  # Only process if not duplicate
                item = queue.get()
                queue.mark_done(item)

        profiler.snapshot("after_high_frequency")

        # Internal state should be clean
        assert len(queue._queued) == 0, "Queued set should be empty"
        assert len(queue._processing) == 0, "Processing set should be empty"
        assert queue.empty(), "Queue should be empty"

        # Memory check
        growth = profiler.compare("baseline", "after_high_frequency")
        print(f"\n=== High-Frequency Operations ===")
        print(f"Iterations: {iterations}")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        leak_detected = leak_detector.check(growth["increase_mb"])
        assert not leak_detected, (
            f"Memory leak detected: {growth['increase_mb']:.2f} MB"
        )

    def test_memory_usage_per_task_id(self, queue):
        """
        MEASUREMENT TEST: Calculate memory cost per task ID stored.

        This helps quantify the leak impact in production scenarios.
        """
        profiler = MemoryProfiler()

        # Measure empty queue
        profiler.snapshot("empty_queue")

        # Add tasks without processing (fill internal sets)
        count = 1000
        for i in range(count):
            task = {
                "path": f"/media/long_filename_to_simulate_real_paths_test_{i}.mp4",
                "type": "transcribe",
                "transcribe_or_translate": "transcribe",
                "force_language": None,
            }
            queue.put(task)

        profiler.snapshot("queue_with_1000_tasks")

        growth = profiler.compare("empty_queue", "queue_with_1000_tasks")
        per_task_kb = (growth["increase_mb"] * 1024) / count

        print(f"\n=== Memory Cost Analysis ===")
        print(f"Tasks added: {count}")
        print(f"Total memory growth: {growth['increase_mb']:.2f} MB")
        print(f"Memory per task ID: {per_task_kb:.2f} KB")
        print(f"\n=== Extrapolation ===")
        print(f"For 10,000 tasks: {per_task_kb * 10:.2f} MB")
        print(f"For 100,000 tasks: {per_task_kb * 100:.2f} MB")

        # Each task ID should be relatively small (< 1 KB)
        assert per_task_kb < 1.0, (
            f"Each task ID uses too much memory: {per_task_kb:.2f} KB"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

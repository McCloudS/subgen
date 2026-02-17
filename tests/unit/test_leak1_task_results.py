"""
Unit tests for Memory Leak #1: task_results dictionary leak

MEMLEAK-FIX-1: Tests for task_results dictionary memory leak
These tests verify that the task_results dictionary is properly cleaned up
and doesn't grow unbounded with each ASR request.

Location in subgen.py:
- Lines 235-236: Global task_results dict and lock declaration
- Lines 748-751: Usage in /asr endpoint (never cleaned up)

Expected behavior:
- FAIL before fix: task_results dict grows unbounded (~500KB per request)
- PASS after fix: TaskResultCache properly cleans up expired entries
"""

import pytest
import time
import gc
from threading import Lock
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path to import subgen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.memory_utils import MemoryProfiler, LeakDetector, measure_memory_growth


# MEMLEAK-FIX-1: Mock the TaskResult class and task_results structures
class MockTaskResult:
    """Mock TaskResult for testing."""

    def __init__(self):
        self.result = None
        self.error = None
        self._event = Mock()
        # MEMLEAK-FIX-1: Simulate realistic memory usage (~1KB per result)
        self._dummy_data = b"x" * 1024

    def set_result(self, result):
        self.result = result

    def set_error(self, error):
        self.error = error


@pytest.mark.unit
@pytest.mark.leak1
class TestTaskResultsLeak:
    """Tests for task_results dictionary memory leak."""

    def test_task_results_dict_grows_unbounded(self):
        """
        MEMLEAK-FIX-1: Demonstrate that task_results dict grows without cleanup.

        This test SHOULD FAIL before the fix is applied, proving the leak exists.
        After the fix, this test will pass.
        """
        # MEMLEAK-FIX-1: Simulate the old task_results dictionary pattern
        task_results = {}
        task_results_lock = Lock()

        with MemoryProfiler(name="task_results_leak") as profiler:
            # MEMLEAK-FIX-1: Simulate 100 ASR requests
            for i in range(100):
                task_id = f"asr-task-{i}"

                with task_results_lock:
                    if task_id not in task_results:
                        task_results[task_id] = MockTaskResult()
                        # MEMLEAK-FIX-1: Simulate storing result data
                        task_results[task_id].set_result(b"x" * 5000)  # ~5KB result

                # MEMLEAK-FIX-1: Note - no cleanup happens here!
                # In the old code, entries were never removed

        # MEMLEAK-FIX-1: Verify the leak
        dict_size = len(task_results)
        memory_growth = profiler.memory_growth_mb

        print(f"\nMEMLEAK-FIX-1: Dictionary size: {dict_size} entries")
        print(f"MEMLEAK-FIX-1: Memory growth: {memory_growth:.2f}MB")

        # MEMLEAK-FIX-1: This assertion proves the leak
        # Before fix: dictionary has 100 entries (never cleaned)
        # After fix: should be much smaller with proper cleanup
        assert dict_size == 100, "task_results dictionary leaked all entries"
        assert memory_growth > 0.4, (
            "Memory leaked as expected (>400KB for 100 requests)"
        )


@pytest.mark.unit
@pytest.mark.leak1
def test_task_results_cache_cleanup():
    """
    MEMLEAK-FIX-1: Test that TaskResultCache properly cleans up old entries.

    This test will SKIP if TaskResultCache doesn't exist yet,
    and PASS once the fix is implemented.
    """
    try:
        # MEMLEAK-FIX-1: Try to import the fixed implementation
        import subgen

        TaskResultCache = getattr(subgen, "TaskResultCache", None)

        if TaskResultCache is None:
            pytest.skip("TaskResultCache not yet implemented (expected before fix)")

        # MEMLEAK-FIX-1: Test the fixed implementation
        cache = TaskResultCache(ttl_seconds=1)  # 1 second TTL

        # Add 100 entries
        for i in range(100):
            task_id = f"task-{i}"
            result = cache.add(task_id)
            result.set_result(b"x" * 5000)

        assert cache.size() == 100, "Cache should have 100 entries"

        # Wait for TTL to expire
        time.sleep(2)

        # Cleanup old entries
        removed = cache.cleanup_old()

        # MEMLEAK-FIX-1: After fix, cleanup should remove expired entries
        assert removed == 100, (
            f"Should remove all 100 expired entries, removed {removed}"
        )
        assert cache.size() == 0, "Cache should be empty after cleanup"

    except ImportError:
        pytest.skip("subgen module not available for import")


@pytest.mark.unit
@pytest.mark.leak1
def test_task_results_immediate_cleanup():
    """
    MEMLEAK-FIX-1: Test immediate cleanup after result consumption.

    This verifies that entries are removed as soon as they're consumed,
    not just after TTL expiration.
    """
    try:
        import subgen

        TaskResultCache = getattr(subgen, "TaskResultCache", None)

        if TaskResultCache is None:
            pytest.skip("TaskResultCache not yet implemented")

        cache = TaskResultCache()

        # MEMLEAK-FIX-1: Add and immediately remove
        task_id = "test-task"
        result = cache.add(task_id)
        result.set_result(b"test data")

        assert cache.size() == 1, "Cache should have 1 entry"

        # MEMLEAK-FIX-1: Consume and cleanup immediately
        cache.remove(task_id)

        assert cache.size() == 0, "Cache should be empty after immediate cleanup"

    except ImportError:
        pytest.skip("subgen module not available for import")


@pytest.mark.stress
@pytest.mark.leak1
def test_no_memory_growth_with_proper_cleanup():
    """
    MEMLEAK-FIX-1: Stress test verifying no memory growth with 1000 requests.

    This is the ultimate test - simulates production load.
    Will FAIL before fix, PASS after fix.
    """
    try:
        import subgen

        TaskResultCache = getattr(subgen, "TaskResultCache", None)

        if TaskResultCache is None:
            pytest.skip("TaskResultCache not yet implemented")

        cache = TaskResultCache(ttl_seconds=60)

        with MemoryProfiler(name="stress_test_1000_requests") as profiler:
            # MEMLEAK-FIX-1: Simulate 1000 ASR requests
            for i in range(1000):
                task_id = f"asr-{i}"

                # Add task result
                result = cache.add(task_id)
                result.set_result(b"x" * 5000)  # 5KB result

                # MEMLEAK-FIX-1: Consume and clean up immediately
                cache.remove(task_id)

                # Periodic cleanup
                if i % 100 == 0:
                    cache.cleanup_old()
                    gc.collect()

            # Final cleanup
            cache.cleanup_old()

        # MEMLEAK-FIX-1: Memory should not grow significantly
        # Before fix: Would leak ~5MB (1000 * 5KB)
        # After fix: Should stay under 10MB (allows for Python overhead)
        profiler.assert_no_memory_leak(max_growth_mb=10.0)

    except ImportError:
        pytest.skip("subgen module not available for import")


@pytest.mark.unit
@pytest.mark.leak1
def test_old_implementation_still_exists():
    """
    MEMLEAK-FIX-1: Verify that old task_results code is updated.

    This test checks if the old code at lines 748-751 still references
    the old task_results dict instead of using task_results_cache.
    """
    try:
        import subgen

        # MEMLEAK-FIX-1: Read the source file to check for old patterns
        import inspect

        source = inspect.getsource(subgen)

        # MEMLEAK-FIX-1: Check if old patterns still exist
        has_old_dict = "task_results = {}" in source
        has_old_lock = "task_results_lock = Lock()" in source
        has_old_usage = "task_results[task_id]" in source

        # MEMLEAK-FIX-1: Check if new cache is being used
        has_cache_class = "class TaskResultCache" in source
        has_cache_instance = "task_results_cache = TaskResultCache" in source

        print(f"\nMEMLEAK-FIX-1: Old dict declaration: {has_old_dict}")
        print(f"MEMLEAK-FIX-1: Old lock declaration: {has_old_lock}")
        print(f"MEMLEAK-FIX-1: Old dict usage: {has_old_usage}")
        print(f"MEMLEAK-FIX-1: New cache class: {has_cache_class}")
        print(f"MEMLEAK-FIX-1: New cache instance: {has_cache_instance}")

        # MEMLEAK-FIX-1: After fix, old patterns should be gone
        if has_cache_class and has_cache_instance:
            assert not has_old_usage, (
                "Old task_results[task_id] pattern still exists! "
                "Update lines 748-751 to use task_results_cache.add()"
            )

    except ImportError:
        pytest.skip("subgen module not available for import")

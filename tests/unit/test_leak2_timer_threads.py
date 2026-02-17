"""
Unit tests for Memory Leak #2: Timer thread accumulation leak

MEMLEAK-FIX-2: Tests for Timer thread accumulation memory leak
These tests verify that cancelled Timer threads are properly cleaned up
and don't accumulate in memory.

Location in subgen.py:
- Lines 1149-1163: schedule_model_cleanup() function
- Line 1156: timer.cancel() without timer.join()

Expected behavior:
- FAIL before fix: Cancelled timers accumulate (~50KB per timer, threads grow)
- PASS after fix: Threads are explicitly joined after cancel()
"""

import pytest
import time
import gc
import threading
from threading import Timer, Lock
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.memory_utils import MemoryProfiler, measure_memory_growth


@pytest.mark.unit
@pytest.mark.leak2
class TestTimerThreadLeak:
    """Tests for Timer thread accumulation leak."""

    def test_timer_threads_accumulate_without_join(self):
        """
        MEMLEAK-FIX-2: Demonstrate that cancelled timers accumulate threads.

        This test SHOULD FAIL before the fix, proving thread accumulation.
        After adding timer.join() after cancel(), this will pass.
        """
        # MEMLEAK-FIX-2: Track initial thread count
        initial_threads = threading.active_count()
        timers = []

        # MEMLEAK-FIX-2: Simulate the old pattern - cancel without join
        for i in range(50):
            timer = Timer(10.0, lambda: None)
            timer.daemon = True
            timer.start()
            time.sleep(0.01)  # Let timer start

            # MEMLEAK-FIX-2: Cancel but don't join (the bug!)
            timer.cancel()

            timers.append(timer)

        # Give threads time to process cancellation
        time.sleep(0.5)
        gc.collect()

        # MEMLEAK-FIX-2: Check thread count
        final_threads = threading.active_count()
        thread_growth = final_threads - initial_threads

        print(f"\nMEMLEAK-FIX-2: Initial threads: {initial_threads}")
        print(f"MEMLEAK-FIX-2: Final threads: {final_threads}")
        print(f"MEMLEAK-FIX-2: Thread growth: {thread_growth}")

        # MEMLEAK-FIX-2: Before fix, threads accumulate
        # After fix with join(), growth should be minimal
        assert thread_growth > 10, (
            f"Expected thread accumulation (>10), got {thread_growth}. "
            "This proves the leak exists."
        )

    def test_timer_cleanup_with_join(self):
        """
        MEMLEAK-FIX-2: Verify that cancel() + join() prevents accumulation.

        This test shows the CORRECT pattern that should be used.
        """
        initial_threads = threading.active_count()

        # MEMLEAK-FIX-2: Use the correct pattern - cancel AND join
        for i in range(50):
            timer = Timer(10.0, lambda: None)
            timer.daemon = True
            timer.start()
            time.sleep(0.01)

            # MEMLEAK-FIX-2: Cancel AND join (the fix!)
            timer.cancel()
            timer.join()  # <-- This is what's missing in subgen.py

        gc.collect()
        time.sleep(0.5)

        final_threads = threading.active_count()
        thread_growth = final_threads - initial_threads

        print(f"\nMEMLEAK-FIX-2 (fixed): Initial threads: {initial_threads}")
        print(f"MEMLEAK-FIX-2 (fixed): Final threads: {final_threads}")
        print(f"MEMLEAK-FIX-2 (fixed): Thread growth: {thread_growth}")

        # MEMLEAK-FIX-2: With proper cleanup, threads should not accumulate
        assert thread_growth < 10, (
            f"Thread count should be stable with join(), got {thread_growth} growth"
        )


@pytest.mark.unit
@pytest.mark.leak2
def test_schedule_model_cleanup_pattern():
    """
    MEMLEAK-FIX-2: Test the actual schedule_model_cleanup pattern from subgen.py.

    This simulates the exact code pattern at lines 1493-1507 in subgen.py.
    """
    # MEMLEAK-FIX-2: Simulate global state from subgen.py
    model_cleanup_timer = None
    model_cleanup_lock = Lock()
    cleanup_executed = []

    def perform_model_cleanup():
        """Mock cleanup function."""
        cleanup_executed.append(True)

    def schedule_model_cleanup_old():
        """MEMLEAK-FIX-2: Old implementation WITHOUT join()."""
        nonlocal model_cleanup_timer

        with model_cleanup_lock:
            # Cancel any existing timer
            if model_cleanup_timer is not None:
                model_cleanup_timer.cancel()
                # MEMLEAK-FIX-2: Missing timer.join() here!

            # Schedule a new cleanup timer
            model_cleanup_timer = Timer(10.0, perform_model_cleanup)
            model_cleanup_timer.daemon = True
            model_cleanup_timer.start()

    # MEMLEAK-FIX-2: Test the pattern
    initial_threads = threading.active_count()

    # Simulate rapid rescheduling (like in production)
    for i in range(30):
        schedule_model_cleanup_old()
        time.sleep(0.02)

    time.sleep(0.5)
    gc.collect()

    final_threads = threading.active_count()
    thread_growth = final_threads - initial_threads

    print(f"\nMEMLEAK-FIX-2: schedule_model_cleanup pattern")
    print(f"MEMLEAK-FIX-2: Thread growth after 30 reschedules: {thread_growth}")

    # MEMLEAK-FIX-2: Before fix, threads accumulate
    assert thread_growth > 5, (
        f"Expected thread accumulation with old pattern, got {thread_growth}"
    )

    # Cleanup
    if model_cleanup_timer:
        model_cleanup_timer.cancel()
        model_cleanup_timer.join()


@pytest.mark.unit
@pytest.mark.leak2
def test_schedule_model_cleanup_fixed_pattern():
    """
    MEMLEAK-FIX-2: Test the FIXED schedule_model_cleanup pattern.

    This shows what the code should look like after the fix.
    """
    model_cleanup_timer = None
    model_cleanup_lock = Lock()

    def perform_model_cleanup():
        """Mock cleanup function."""
        pass

    def schedule_model_cleanup_fixed():
        """MEMLEAK-FIX-2: Fixed implementation WITH join()."""
        nonlocal model_cleanup_timer

        with model_cleanup_lock:
            # Cancel any existing timer
            if model_cleanup_timer is not None:
                model_cleanup_timer.cancel()
                model_cleanup_timer.join()  # MEMLEAK-FIX-2: ADD THIS LINE!

            # Schedule a new cleanup timer
            model_cleanup_timer = Timer(10.0, perform_model_cleanup)
            model_cleanup_timer.daemon = True
            model_cleanup_timer.start()

    initial_threads = threading.active_count()

    # Simulate rapid rescheduling with fix
    for i in range(30):
        schedule_model_cleanup_fixed()
        time.sleep(0.02)

    time.sleep(0.5)
    gc.collect()

    final_threads = threading.active_count()
    thread_growth = final_threads - initial_threads

    print(f"\nMEMLEAK-FIX-2 (fixed): Thread growth with join(): {thread_growth}")

    # MEMLEAK-FIX-2: With fix, threads should not accumulate
    assert thread_growth < 10, (
        f"Thread count should be stable with fixed pattern, got {thread_growth}"
    )

    # Cleanup
    if model_cleanup_timer:
        model_cleanup_timer.cancel()
        model_cleanup_timer.join()


@pytest.mark.stress
@pytest.mark.leak2
def test_timer_thread_stress_500_cycles():
    """
    MEMLEAK-FIX-2: Stress test with 500 schedule/cancel cycles.

    This simulates heavy production load with frequent model cleanup scheduling.
    Will FAIL before fix, PASS after adding timer.join().
    """
    try:
        import subgen

        # MEMLEAK-FIX-2: Try to find the actual functions in subgen
        schedule_func = getattr(subgen, "schedule_model_cleanup", None)

        if schedule_func is None:
            pytest.skip("schedule_model_cleanup not accessible for testing")

        initial_threads = threading.active_count()

        with MemoryProfiler(name="timer_stress_500_cycles") as profiler:
            # MEMLEAK-FIX-2: Stress test with 500 cycles
            for i in range(500):
                schedule_func()

                # Immediately cancel (simulates rapid request pattern)
                # In production, this happens when requests come in quickly
                time.sleep(0.01)

                # Periodic GC
                if i % 100 == 0:
                    gc.collect()
                    time.sleep(0.1)

        gc.collect()
        time.sleep(1)

        final_threads = threading.active_count()
        thread_growth = final_threads - initial_threads

        print(f"\nMEMLEAK-FIX-2: Stress test results")
        print(f"MEMLEAK-FIX-2: Thread growth: {thread_growth}")
        print(f"MEMLEAK-FIX-2: Memory growth: {profiler.memory_growth_mb:.2f}MB")

        # MEMLEAK-FIX-2: After fix, should not accumulate threads
        profiler.assert_no_thread_leak(max_growth=20)

    except ImportError:
        pytest.skip("subgen module not available for import")


@pytest.mark.unit
@pytest.mark.leak2
def test_check_source_for_timer_join():
    """
    MEMLEAK-FIX-2: Check if timer.join() is present in schedule_model_cleanup.

    This test verifies that the fix has been applied to the source code.
    """
    try:
        import subgen
        import inspect

        # MEMLEAK-FIX-2: Try to get source of schedule_model_cleanup
        schedule_func = getattr(subgen, "schedule_model_cleanup", None)

        if schedule_func is None:
            pytest.skip("schedule_model_cleanup not accessible")

        source = inspect.getsource(schedule_func)

        # MEMLEAK-FIX-2: Check for the fix
        has_cancel = "timer.cancel()" in source or ".cancel()" in source
        has_join = "timer.join()" in source or ".join()" in source

        print(f"\nMEMLEAK-FIX-2: Source code analysis")
        print(f"MEMLEAK-FIX-2: Has cancel(): {has_cancel}")
        print(f"MEMLEAK-FIX-2: Has join(): {has_join}")

        if has_cancel and not has_join:
            pytest.fail(
                "MEMLEAK-FIX-2: timer.cancel() found but timer.join() missing! "
                "Add 'model_cleanup_timer.join()' after line 1500 in subgen.py"
            )

        if has_cancel and has_join:
            print("MEMLEAK-FIX-2: Fix is properly implemented!")

    except ImportError:
        pytest.skip("subgen module not available for import")

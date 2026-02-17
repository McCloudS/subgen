"""
Memory profiling utilities for detecting memory leaks in subgen.

MEMLEAK-FIX-1: Tests for task_results dictionary leak
MEMLEAK-FIX-2: Tests for Timer thread accumulation leak
MEMLEAK-FIX-3: Tests for BytesIO context manager leak

These utilities use tracemalloc to track memory allocations and detect leaks.
"""

import gc
import sys
import time
import tracemalloc
import threading
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""

    current_bytes: int
    peak_bytes: int
    timestamp: float
    active_threads: int

    @property
    def current_mb(self) -> float:
        """Current memory usage in MB."""
        return self.current_bytes / 1024 / 1024

    @property
    def peak_mb(self) -> float:
        """Peak memory usage in MB."""
        return self.peak_bytes / 1024 / 1024

    def __repr__(self) -> str:
        return (
            f"MemorySnapshot(current={self.current_mb:.2f}MB, "
            f"peak={self.peak_mb:.2f}MB, threads={self.active_threads})"
        )


class MemoryProfiler:
    """
    Context manager for tracking memory usage and detecting leaks.

    Example usage:
        with MemoryProfiler() as profiler:
            # Run code that might leak
            for i in range(1000):
                create_objects()

        growth = profiler.memory_growth_mb
        assert growth < 10, f"Memory leak: {growth}MB growth"
    """

    def __init__(self, name: str = "test", enable_gc: bool = True):
        """
        Initialize memory profiler.

        Args:
            name: Name for this profiling session (for logging)
            enable_gc: Whether to force garbage collection at start/end
        """
        self.name = name
        self.enable_gc = enable_gc
        self.baseline: Optional[MemorySnapshot] = None
        self.final: Optional[MemorySnapshot] = None

    def __enter__(self) -> "MemoryProfiler":
        """Start memory profiling."""
        if self.enable_gc:
            gc.collect()
            time.sleep(0.1)  # Let GC settle

        tracemalloc.start()
        self.baseline = self._take_snapshot()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop memory profiling."""
        if self.enable_gc:
            gc.collect()
            time.sleep(0.1)  # Let GC settle

        self.final = self._take_snapshot()
        tracemalloc.stop()

    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        current, peak = tracemalloc.get_traced_memory()
        return MemorySnapshot(
            current_bytes=current,
            peak_bytes=peak,
            timestamp=time.time(),
            active_threads=threading.active_count(),
        )

    @property
    def memory_growth_bytes(self) -> int:
        """Memory growth in bytes."""
        if not self.baseline or not self.final:
            raise RuntimeError("Profiler must be used as context manager")
        return self.final.current_bytes - self.baseline.current_bytes

    @property
    def memory_growth_mb(self) -> float:
        """Memory growth in MB."""
        return self.memory_growth_bytes / 1024 / 1024

    @property
    def thread_growth(self) -> int:
        """Number of threads added."""
        if not self.baseline or not self.final:
            raise RuntimeError("Profiler must be used as context manager")
        return self.final.active_threads - self.baseline.active_threads

    def assert_no_memory_leak(self, max_growth_mb: float = 10.0):
        """
        Assert that memory growth is within acceptable bounds.

        Args:
            max_growth_mb: Maximum allowed memory growth in MB

        Raises:
            AssertionError: If memory growth exceeds threshold
        """
        growth = self.memory_growth_mb
        assert growth < max_growth_mb, (
            f"Memory leak detected in '{self.name}': "
            f"{growth:.2f}MB growth (max: {max_growth_mb}MB)\n"
            f"Baseline: {self.baseline}\n"
            f"Final: {self.final}"
        )

    def assert_no_thread_leak(self, max_growth: int = 5):
        """
        Assert that thread count is within acceptable bounds.

        Args:
            max_growth: Maximum allowed thread count increase

        Raises:
            AssertionError: If thread growth exceeds threshold
        """
        growth = self.thread_growth
        assert growth <= max_growth, (
            f"Thread leak detected in '{self.name}': "
            f"{growth} threads added (max: {max_growth})\n"
            f"Baseline: {self.baseline.active_threads} threads\n"
            f"Final: {self.final.active_threads} threads"
        )


def measure_memory_growth(
    func, iterations: int = 100, cleanup_interval: int = 10
) -> Tuple[float, int]:
    """
    Measure memory and thread growth over multiple iterations.

    Args:
        func: Function to test (should be callable with no args)
        iterations: Number of times to call the function
        cleanup_interval: How often to force garbage collection

    Returns:
        Tuple of (memory_growth_mb, thread_growth)
    """
    with MemoryProfiler(name=func.__name__) as profiler:
        for i in range(iterations):
            func()

            # Periodic cleanup to give GC a chance
            if (i + 1) % cleanup_interval == 0:
                gc.collect()
                time.sleep(0.01)

    return profiler.memory_growth_mb, profiler.thread_growth


def get_object_count(obj_type: type) -> int:
    """
    Count instances of a specific type in memory.

    Useful for detecting object leaks.

    Args:
        obj_type: Type to count (e.g., BytesIO, Timer)

    Returns:
        Number of instances found
    """
    gc.collect()
    return sum(1 for obj in gc.get_objects() if isinstance(obj, obj_type))


class LeakDetector:
    """
    Helper for detecting specific types of leaks.

    Example:
        detector = LeakDetector(BytesIO)
        detector.start()

        # Run code that might leak BytesIO
        for i in range(100):
            buffer = BytesIO(b"data")
            # Oops, didn't close it!

        leaked_count = detector.leaked_count()
        assert leaked_count < 10, f"{leaked_count} BytesIO objects leaked"
    """

    def __init__(self, obj_type: type):
        """
        Initialize leak detector.

        Args:
            obj_type: Type to track (e.g., BytesIO, Timer)
        """
        self.obj_type = obj_type
        self.baseline_count: Optional[int] = None

    def start(self):
        """Record baseline object count."""
        gc.collect()
        time.sleep(0.1)
        self.baseline_count = get_object_count(self.obj_type)

    def leaked_count(self) -> int:
        """
        Get number of leaked objects.

        Returns:
            Number of objects that weren't cleaned up
        """
        if self.baseline_count is None:
            raise RuntimeError("Must call start() before leaked_count()")

        gc.collect()
        time.sleep(0.1)
        current_count = get_object_count(self.obj_type)
        return max(0, current_count - self.baseline_count)

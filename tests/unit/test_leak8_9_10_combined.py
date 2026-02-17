"""
Memory Leak Tests: LEAK #8, #9, #10 - Combined Tests for Smaller Leaks

LEAK #8: File System Observer Never Stopped
LEAK #9: Whisper Model Segments Accumulate
LEAK #10: FFmpeg Process Pipes Not Flushed

These are lower severity leaks but still important for 24/7 operation.
"""

import pytest
import sys
import os
import tempfile
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.memory_utils import MemoryProfiler, LeakDetector


# ============================================================================
# LEAK #8: File System Observer
# ============================================================================


@pytest.mark.unit
@pytest.mark.memory_leak
class TestFileObserverLeaks:
    """Test suite for watchdog Observer resource management."""

    @pytest.mark.skipif(
        not hasattr(__import__("watchdog", fromlist=["observers"]), "observers"),
        reason="watchdog not installed",
    )
    def test_observer_never_stopped(self):
        """
        LEAK #8: Observer threads and polling structures accumulate.

        The code starts Observer but never stops it, leaving threads running.
        """
        from watchdog.observers.polling import PollingObserver
        from watchdog.events import FileSystemEventHandler

        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=5.0)

        profiler.snapshot("baseline")

        # Create temporary directory to watch
        with tempfile.TemporaryDirectory() as tmpdir:
            # Start multiple observers without stopping (simulating leak)
            observers = []
            for i in range(5):
                observer = PollingObserver()
                handler = FileSystemEventHandler()
                observer.schedule(handler, tmpdir, recursive=True)
                observer.start()
                observers.append(observer)
                time.sleep(0.1)  # Let observer initialize

            profiler.snapshot("after_starting_observers")

            # Keep them running briefly
            time.sleep(1)

            profiler.snapshot("observers_running")

            growth = profiler.compare("baseline", "observers_running")

            print(f"\n=== File Observer Memory Usage ===")
            print(f"Observers started: {len(observers)}")
            print(f"Memory growth: {growth['increase_mb']:.2f} MB")
            print(f"Per observer: {growth['increase_mb'] / len(observers):.2f} MB")

            # Now stop them properly
            profiler.snapshot("before_stop")
            for observer in observers:
                observer.stop()
                observer.join(timeout=2)

            profiler.snapshot("after_stop")

            freed = profiler.compare("before_stop", "after_stop")
            print(f"Memory freed by stop(): {abs(freed['increase_mb']):.2f} MB")

            # Should show memory growth from observers
            assert growth["increase_mb"] > 0.5, "Expected memory from observer threads"

    @pytest.mark.skipif(
        not hasattr(__import__("watchdog", fromlist=["observers"]), "observers"),
        reason="watchdog not installed",
    )
    def test_observer_proper_cleanup_pattern(self):
        """
        COMPARISON TEST: Proper observer cleanup prevents leak.
        """
        from watchdog.observers.polling import PollingObserver
        from watchdog.events import FileSystemEventHandler

        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=3.0)

        profiler.snapshot("baseline")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Start and properly stop observers
            for i in range(5):
                observer = PollingObserver()
                handler = FileSystemEventHandler()
                observer.schedule(handler, tmpdir, recursive=True)
                observer.start()
                time.sleep(0.2)
                observer.stop()
                observer.join(timeout=2)

        profiler.snapshot("after_proper_cleanup")

        growth = profiler.compare("baseline", "after_proper_cleanup")

        print(f"\n=== Observer With Proper Cleanup ===")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        leak_detected = leak_detector.check(growth["increase_mb"])
        assert not leak_detected, (
            f"Memory leak despite cleanup: {growth['increase_mb']:.2f} MB"
        )


# ============================================================================
# LEAK #9: Whisper Model Segments
# ============================================================================


@pytest.mark.unit
@pytest.mark.memory_leak
class TestWhisperSegmentLeaks:
    """Test suite for Whisper transcription result memory management."""

    def test_segment_accumulation(self):
        """
        LEAK #9: Whisper result segments accumulate when appended.

        The appendLine() function modifies result.segments in place,
        potentially keeping references longer than needed.
        """

        # Simulate Whisper result structure
        class MockSegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text
                self.id = 0
                self.words = []

        class MockResult:
            def __init__(self):
                self.segments = []

        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=5.0)

        profiler.snapshot("baseline")

        # Create many results with segments
        results = []
        for i in range(1000):
            result = MockResult()
            # Add segments (typical transcription has 50-200 segments)
            for j in range(50):
                segment = MockSegment(
                    start=j * 2.0,
                    end=(j + 1) * 2.0,
                    text=f"This is test segment {j} with some content " * 5,
                )
                result.segments.append(segment)
            results.append(result)

        profiler.snapshot("after_1000_results")

        growth = profiler.compare("baseline", "after_1000_results")

        print(f"\n=== Whisper Segment Accumulation ===")
        print(f"Results created: 1000")
        print(f"Segments per result: 50")
        print(f"Total segments: 50,000")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")
        print(f"Per segment: {(growth['increase_mb'] * 1024) / 50000:.3f} KB")

        leak_detected = leak_detector.check(growth["increase_mb"])
        # This is expected to grow, but should be bounded
        assert growth["increase_mb"] < 20.0, (
            f"Segments using too much memory: {growth['increase_mb']:.2f} MB"
        )

        # Clear results
        results.clear()

    def test_append_line_pattern(self):
        """
        LEAK #9: Test the appendLine() modification pattern.

        Simulates appending attribution segment as done in the code.
        """

        class MockSegment:
            def __init__(self, start, end, text, segment_id=0):
                self.start = start
                self.end = end
                self.text = text
                self.id = segment_id
                self.words = []

        class MockResult:
            def __init__(self):
                self.segments = []

        profiler = MemoryProfiler()

        profiler.snapshot("baseline")

        # Simulate processing 500 files
        results = []
        for i in range(500):
            result = MockResult()

            # Add normal segments
            for j in range(30):
                result.segments.append(MockSegment(j, j + 1, f"Segment {j}"))

            # Simulate appendLine() - adds attribution
            if len(result.segments) > 0:
                last = result.segments[-1]
                new_segment = MockSegment(
                    last.end + 5,
                    last.end + 10,
                    "Transcribed by whisperAI with faster-whisper (medium) on 01 Jan 2024",
                    last.id + 1,
                )
                result.segments.append(new_segment)

            results.append(result)

        profiler.snapshot("after_append_line")

        growth = profiler.compare("baseline", "after_append_line")

        print(f"\n=== appendLine() Pattern ===")
        print(f"Results: 500")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        results.clear()


# ============================================================================
# LEAK #10: FFmpeg Process Pipes
# ============================================================================


@pytest.mark.unit
@pytest.mark.memory_leak
class TestFFmpegPipeLeaks:
    """Test suite for FFmpeg subprocess pipe management."""

    def test_ffmpeg_pipe_accumulation(self):
        """
        LEAK #10: FFmpeg subprocess pipes may not flush completely.

        Tests if subprocess communication leaves buffers unflushed.
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=5.0)

        # Create test audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            test_file = f.name

        try:
            # Generate test audio
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1000:duration=1",
                    "-y",
                    test_file,
                ],
                capture_output=True,
                check=True,
            )

            profiler.snapshot("baseline")

            # Run FFmpeg operations repeatedly
            iterations = 50
            for i in range(iterations):
                result = subprocess.run(
                    ["ffmpeg", "-i", test_file, "-f", "wav", "-ar", "16000", "-"],
                    capture_output=True,
                    check=False,
                )

                # Capture but don't explicitly flush
                _ = result.stdout
                _ = result.stderr

            profiler.snapshot("after_ffmpeg_operations")

            growth = profiler.compare("baseline", "after_ffmpeg_operations")

            print(f"\n=== FFmpeg Pipe Operations ===")
            print(f"Iterations: {iterations}")
            print(f"Memory growth: {growth['increase_mb']:.2f} MB")
            print(f"Per operation: {growth['increase_mb'] / iterations:.3f} MB")

            leak_detected = leak_detector.check(growth["increase_mb"])
            assert not leak_detected, (
                f"Memory leak in FFmpeg pipes: {growth['increase_mb']:.2f} MB"
            )

        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)

    def test_ffmpeg_error_handling_pipes(self):
        """
        LEAK #10: FFmpeg errors may leave stderr buffers full.

        Large error output can accumulate in pipes.
        """
        profiler = MemoryProfiler()

        profiler.snapshot("baseline")

        # Run FFmpeg with intentional errors
        stderr_outputs = []
        for i in range(30):
            result = subprocess.run(
                ["ffmpeg", "-i", "/nonexistent/file.mp4", "-f", "wav", "-"],
                capture_output=True,
                check=False,
            )

            # Keep stderr (simulating not cleaning up error output)
            stderr_outputs.append(result.stderr)

        profiler.snapshot("after_errors")

        growth = profiler.compare("baseline", "after_errors")

        print(f"\n=== FFmpeg Error Pipe Accumulation ===")
        print(f"Errors generated: 30")
        print(f"Stderr outputs held: {len(stderr_outputs)}")
        print(f"Total stderr size: {sum(len(s) for s in stderr_outputs)} bytes")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        # Clear error outputs
        stderr_outputs.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

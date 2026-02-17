"""
Memory Leak Test: LEAK #5 - PyAV Container Objects Not Explicitly Closed

PROBLEM:
While Python's 'with' statement closes PyAV containers, FFmpeg resources
(decoder contexts, buffers) wrapped by PyAV may not be immediately released.
In high-frequency scenarios, FFmpeg decoder contexts accumulate, causing
memory growth that's only cleared when garbage collection runs.

EXPECTED BEHAVIOR:
- PyAV containers should release FFmpeg resources immediately
- No accumulation of decoder contexts
- Memory should be stable across repeated file operations

IMPACT:
- Each unclosed FFmpeg context: ~1-5MB
- If GC is delayed: 10-50MB per 100 files
- Critical for 24/7 batch processing

LOCATIONS IN CODE:
- Line 1665: get_subtitle_languages() - opens container
- Line 1720: has_subtitle_language_in_file() - opens container
- Line 2047: has_audio() - opens container

TEST STRATEGY:
1. Create test video file
2. Open with PyAV repeatedly
3. Monitor memory growth
4. Verify FFmpeg resources are released
"""

import pytest
import sys
import os
import tempfile
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.memory_utils import MemoryProfiler, LeakDetector
import av


@pytest.fixture
def test_video_file():
    """Create a minimal test video file using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name

    # Create a 1-second test video with audio
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-y",
            video_path,
        ],
        capture_output=True,
        check=True,
    )

    yield video_path

    # Cleanup
    if os.path.exists(video_path):
        os.unlink(video_path)


@pytest.mark.unit
@pytest.mark.memory_leak
class TestPyAVContainerLeaks:
    """Test suite for PyAV container resource management."""

    def test_repeated_container_open_closes_leak(self, test_video_file):
        """
        LEAK TEST: Opening containers repeatedly without explicit cleanup.

        Tests if PyAV containers properly release FFmpeg resources when
        using Python's context manager (with statement).
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=10.0)

        profiler.snapshot("baseline")

        # Open the same file many times
        iterations = 100
        for i in range(iterations):
            with av.open(test_video_file) as container:
                # Read some streams to allocate decoder contexts
                for stream in container.streams:
                    _ = stream.type
                    _ = stream.codec_context

        profiler.snapshot("after_100_opens")

        growth = profiler.compare("baseline", "after_100_opens")

        print(f"\n=== PyAV Container Repeated Opens ===")
        print(f"Iterations: {iterations}")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")
        print(f"Per operation: {growth['increase_mb'] / iterations:.3f} MB")

        # Should not accumulate significant memory
        leak_detected = leak_detector.check(growth["increase_mb"])
        assert not leak_detected, (
            f"Memory leak detected: {growth['increase_mb']:.2f} MB for {iterations} operations"
        )

    def test_container_without_context_manager(self, test_video_file):
        """
        LEAK TEST: Opening containers without 'with' statement.

        This tests the leak when containers are opened but not explicitly
        closed, relying on garbage collection.
        """
        profiler = MemoryProfiler()

        profiler.snapshot("baseline")

        containers = []
        for i in range(50):
            # Open WITHOUT context manager - simulating the leak
            container = av.open(test_video_file)
            # Access streams to allocate resources
            for stream in container.streams:
                _ = stream.codec_context
            containers.append(container)

        profiler.snapshot("after_50_unclosed")

        growth = profiler.compare("baseline", "after_50_unclosed")

        print(f"\n=== Unclosed PyAV Containers ===")
        print(f"Containers opened: 50")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        # Should show significant memory growth
        assert growth["increase_mb"] > 5.0, (
            "Expected memory growth from unclosed containers"
        )

        # Now close them all
        profiler.snapshot("before_cleanup")
        for container in containers:
            container.close()
        containers.clear()

        profiler.snapshot("after_cleanup")

        cleanup_diff = profiler.compare("before_cleanup", "after_cleanup")
        print(f"Memory freed after close(): {abs(cleanup_diff['increase_mb']):.2f} MB")

    def test_subtitle_stream_enumeration_leak(self, test_video_file):
        """
        LEAK TEST: Simulating get_subtitle_languages() pattern.

        This mimics the code at line 1665 which opens containers to read
        subtitle stream languages.
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=5.0)

        profiler.snapshot("baseline")

        # Simulate the pattern from get_subtitle_languages()
        iterations = 200
        for i in range(iterations):
            with av.open(test_video_file) as container:
                subtitle_languages = []
                for stream in container.streams.subtitles:
                    lang_code = stream.metadata.get("language")
                    subtitle_languages.append(lang_code if lang_code else "und")

        profiler.snapshot("after_subtitle_enumeration")

        growth = profiler.compare("baseline", "after_subtitle_enumeration")

        print(f"\n=== Subtitle Stream Enumeration ===")
        print(f"Iterations: {iterations}")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        leak_detected = leak_detector.check(growth["increase_mb"])
        assert not leak_detected, (
            f"Memory leak in subtitle enumeration: {growth['increase_mb']:.2f} MB"
        )

    def test_audio_stream_check_leak(self, test_video_file):
        """
        LEAK TEST: Simulating has_audio() pattern.

        This mimics the code at line 2047 which opens containers to check
        for audio streams.
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=5.0)

        profiler.snapshot("baseline")

        # Simulate the pattern from has_audio()
        iterations = 200
        for i in range(iterations):
            has_audio = False
            try:
                with av.open(test_video_file) as container:
                    for stream in container.streams:
                        if stream.type == "audio":
                            if (
                                stream.codec_context
                                and stream.codec_context.name != "none"
                            ):
                                has_audio = True
                                break
            except Exception:
                pass

        profiler.snapshot("after_audio_checks")

        growth = profiler.compare("baseline", "after_audio_checks")

        print(f"\n=== Audio Stream Checks ===")
        print(f"Iterations: {iterations}")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        leak_detected = leak_detector.check(growth["increase_mb"])
        assert not leak_detected, (
            f"Memory leak in audio checks: {growth['increase_mb']:.2f} MB"
        )

    def test_ffmpeg_decoder_context_accumulation(self, test_video_file):
        """
        LEAK TEST: Measure decoder context memory usage.

        Each codec_context accessed creates FFmpeg decoder state.
        Verify these are properly cleaned up.
        """
        profiler = MemoryProfiler()

        profiler.snapshot("baseline")

        # Access codec contexts to trigger decoder allocation
        for i in range(100):
            with av.open(test_video_file) as container:
                for stream in container.streams:
                    ctx = stream.codec_context
                    # Access codec properties to ensure context is initialized
                    if ctx:
                        _ = ctx.name
                        _ = ctx.type

        profiler.snapshot("after_decoder_contexts")

        growth = profiler.compare("baseline", "after_decoder_contexts")

        print(f"\n=== Decoder Context Allocation ===")
        print(f"Iterations: 100")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")
        print(f"Per context: {growth['increase_mb'] / 100:.3f} MB")

        # Each decoder context should be small when properly cleaned
        per_context_mb = growth["increase_mb"] / 100
        assert per_context_mb < 0.5, (
            f"Decoder contexts using too much memory: {per_context_mb:.3f} MB each"
        )

    def test_multiple_files_sequential_processing(self):
        """
        LEAK TEST: Processing multiple files sequentially.

        Simulates batch processing scenario where many different files
        are processed one after another.
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=20.0)

        # Create multiple test files
        test_files = []
        for i in range(5):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                video_path = f.name
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    f"testsrc=duration=1:size=320x240:rate=1",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1000:duration=1",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-y",
                    video_path,
                ],
                capture_output=True,
                check=True,
            )
            test_files.append(video_path)

        try:
            profiler.snapshot("baseline")

            # Process each file multiple times
            for iteration in range(20):
                for file_path in test_files:
                    with av.open(file_path) as container:
                        # Simulate typical processing
                        for stream in container.streams:
                            _ = stream.type
                            _ = stream.codec_context

            profiler.snapshot("after_batch_processing")

            growth = profiler.compare("baseline", "after_batch_processing")

            print(f"\n=== Multiple Files Sequential Processing ===")
            print(f"Files: {len(test_files)}")
            print(f"Iterations per file: 20")
            print(f"Total operations: {len(test_files) * 20}")
            print(f"Memory growth: {growth['increase_mb']:.2f} MB")

            leak_detected = leak_detector.check(growth["increase_mb"])
            assert not leak_detected, (
                f"Memory leak in batch processing: {growth['increase_mb']:.2f} MB"
            )

        finally:
            # Cleanup test files
            for file_path in test_files:
                if os.path.exists(file_path):
                    os.unlink(file_path)

    def test_container_close_explicitly(self, test_video_file):
        """
        COMPARISON TEST: Explicit close() vs context manager.

        Compare memory behavior between manual close and with statement.
        """
        profiler = MemoryProfiler()

        # Test with explicit close
        profiler.snapshot("baseline_explicit")
        for i in range(50):
            container = av.open(test_video_file)
            for stream in container.streams:
                _ = stream.type
            container.close()
        profiler.snapshot("after_explicit_close")

        # Test with context manager
        profiler.snapshot("baseline_context")
        for i in range(50):
            with av.open(test_video_file) as container:
                for stream in container.streams:
                    _ = stream.type
        profiler.snapshot("after_context_manager")

        explicit_growth = profiler.compare("baseline_explicit", "after_explicit_close")
        context_growth = profiler.compare("baseline_context", "after_context_manager")

        print(f"\n=== Explicit close() vs Context Manager ===")
        print(
            f"Explicit close() memory growth: {explicit_growth['increase_mb']:.2f} MB"
        )
        print(f"Context manager memory growth: {context_growth['increase_mb']:.2f} MB")
        print(
            f"Difference: {abs(explicit_growth['increase_mb'] - context_growth['increase_mb']):.2f} MB"
        )

        # Both methods should result in similar memory behavior
        assert (
            abs(explicit_growth["increase_mb"] - context_growth["increase_mb"]) < 2.0
        ), "Significant difference between close methods"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

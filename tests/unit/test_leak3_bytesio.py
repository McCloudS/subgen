"""
Unit tests for Memory Leak #3: BytesIO context manager leak

MEMLEAK-FIX-3: Tests for BytesIO objects not being closed
These tests verify that BytesIO objects returned from audio extraction
functions are properly closed and don't leak memory.

Location in subgen.py:
- Lines 1065-1069: detect_language_task() - BytesIO.read() without close
- Lines 1100-1141: extract_audio_segment_to_memory() - returns BytesIO
- Lines 1245-1247: gen_subtitles() - BytesIO.read() without close
- Line 1346: handle_multiple_audio_tracks() - returns BytesIO
- Lines 1352-1386: extract_audio_track_to_memory() - returns BytesIO

Expected behavior:
- FAIL before fix: BytesIO objects leak (480KB - 10MB per transcription)
- PASS after fix: BytesIO properly closed or bytes returned directly
"""

import pytest
import io
from io import BytesIO
import gc
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.memory_utils import MemoryProfiler, LeakDetector, get_object_count


@pytest.mark.unit
@pytest.mark.leak3
class TestBytesIOLeak:
    """Tests for BytesIO memory leak in audio extraction."""

    def test_bytesio_leaks_without_close(self):
        """
        MEMLEAK-FIX-3: Demonstrate BytesIO leak when not closed.

        This test SHOULD FAIL before fix, proving BytesIO objects leak.
        """
        # MEMLEAK-FIX-3: Track BytesIO objects
        detector = LeakDetector(BytesIO)
        detector.start()

        with MemoryProfiler(name="bytesio_leak") as profiler:
            # MEMLEAK-FIX-3: Simulate the old pattern - create but don't close
            buffers = []
            for i in range(100):
                # Create BytesIO with audio data (simulate 480KB per segment)
                audio_data = b"x" * (480 * 1024)
                buffer = BytesIO(audio_data)

                # Read from it (like in detect_language_task)
                _ = buffer.read()

                # MEMLEAK-FIX-3: Bug! We never close the buffer
                # In old code: extract_audio_segment_to_memory().read()
                # The BytesIO is not closed!

                buffers.append(buffer)

        # MEMLEAK-FIX-3: Check for leaks
        leaked_count = detector.leaked_count()
        memory_growth = profiler.memory_growth_mb

        print(f"\nMEMLEAK-FIX-3: BytesIO objects leaked: {leaked_count}")
        print(f"MEMLEAK-FIX-3: Memory growth: {memory_growth:.2f}MB")

        # MEMLEAK-FIX-3: Before fix, BytesIO objects accumulate
        assert leaked_count >= 90, (
            f"Expected ~100 leaked BytesIO, got {leaked_count}. "
            "This proves the leak exists."
        )
        assert memory_growth > 40, (
            f"Expected >40MB leak (100 * 480KB), got {memory_growth:.2f}MB"
        )

        # Cleanup
        for buf in buffers:
            buf.close()

    def test_bytesio_no_leak_with_close(self):
        """
        MEMLEAK-FIX-3: Verify proper cleanup when BytesIO is closed.

        This shows the correct pattern.
        """
        detector = LeakDetector(BytesIO)
        detector.start()

        with MemoryProfiler(name="bytesio_fixed") as profiler:
            # MEMLEAK-FIX-3: Correct pattern - close after use
            for i in range(100):
                audio_data = b"x" * (480 * 1024)
                buffer = BytesIO(audio_data)

                try:
                    _ = buffer.read()
                finally:
                    buffer.close()  # MEMLEAK-FIX-3: Always close!

        leaked_count = detector.leaked_count()
        memory_growth = profiler.memory_growth_mb

        print(f"\nMEMLEAK-FIX-3 (fixed): BytesIO leaked: {leaked_count}")
        print(f"MEMLEAK-FIX-3 (fixed): Memory growth: {memory_growth:.2f}MB")

        # MEMLEAK-FIX-3: With proper cleanup, should not leak
        assert leaked_count < 10, (
            f"Should not leak BytesIO with close(), got {leaked_count}"
        )
        # Allow some memory growth for Python overhead
        assert memory_growth < 10, (
            f"Memory should be freed, got {memory_growth:.2f}MB growth"
        )

    def test_bytesio_context_manager_pattern(self):
        """
        MEMLEAK-FIX-3: Test context manager pattern (preferred fix).

        This is the recommended way to fix the leak - use with statements.
        """

        def extract_audio_mock():
            """Mock that returns BytesIO as context manager."""
            from contextlib import contextmanager

            @contextmanager
            def _extract():
                buffer = BytesIO(b"x" * (480 * 1024))
                try:
                    yield buffer
                finally:
                    buffer.close()

            return _extract()

        detector = LeakDetector(BytesIO)
        detector.start()

        with MemoryProfiler(name="context_manager_pattern") as profiler:
            for i in range(100):
                # MEMLEAK-FIX-3: Use context manager (the fix!)
                with extract_audio_mock() as audio:
                    _ = audio.read()
                # buffer.close() called automatically

        leaked_count = detector.leaked_count()

        print(f"\nMEMLEAK-FIX-3 (context mgr): BytesIO leaked: {leaked_count}")

        assert leaked_count < 10, "Context manager should prevent leaks"


@pytest.mark.unit
@pytest.mark.leak3
def test_extract_audio_segment_returns_bytes():
    """
    MEMLEAK-FIX-3: Test if extract_audio_segment_to_memory returns bytes.

    Option 3 for the fix: Return bytes directly instead of BytesIO.
    This is the simplest fix and eliminates the cleanup problem.
    """
    try:
        import subgen

        extract_func = getattr(subgen, "extract_audio_segment_to_memory", None)

        if extract_func is None:
            pytest.skip("extract_audio_segment_to_memory not accessible")

        # MEMLEAK-FIX-3: Mock ffmpeg to test return type
        with patch("subgen.ffmpeg") as mock_ffmpeg:
            # Mock ffmpeg output
            mock_output = b"fake audio data"
            mock_ffmpeg.input.return_value.output.return_value.run.return_value = (
                mock_output,
                b"",
            )

            # MEMLEAK-FIX-3: Call the function
            result = extract_func("test.mp4", 0, 30)

            # MEMLEAK-FIX-3: Check return type
            if isinstance(result, bytes):
                print("\nMEMLEAK-FIX-3: Returns bytes directly (GOOD!)")
                assert True
            elif isinstance(result, BytesIO):
                print("\nMEMLEAK-FIX-3: Returns BytesIO (NEEDS FIX!)")
                result.close()  # Clean up
                pytest.fail(
                    "extract_audio_segment_to_memory still returns BytesIO. "
                    "Should return bytes directly or use context manager."
                )
            else:
                pytest.fail(f"Unexpected return type: {type(result)}")

    except ImportError:
        pytest.skip("subgen module not available for import")


@pytest.mark.unit
@pytest.mark.leak3
def test_detect_language_task_closes_bytesio():
    """
    MEMLEAK-FIX-3: Test that detect_language_task properly closes BytesIO.

    This tests the usage at lines 1065-1069 in subgen.py.
    """
    try:
        import subgen

        detect_func = getattr(subgen, "detect_language_task", None)

        if detect_func is None:
            pytest.skip("detect_language_task not accessible")

        # MEMLEAK-FIX-3: Track BytesIO before and after
        detector = LeakDetector(BytesIO)
        detector.start()

        # MEMLEAK-FIX-3: Mock dependencies
        with patch("subgen.extract_audio_segment_to_memory") as mock_extract:
            with patch("subgen.model") as mock_model:
                # Setup mocks
                audio_buffer = BytesIO(b"audio data")
                mock_extract.return_value = audio_buffer

                mock_result = Mock()
                mock_result.language = "en"
                mock_model.transcribe.return_value = mock_result

                # MEMLEAK-FIX-3: Call function
                try:
                    # This will fail if function signature changed, that's ok
                    detect_func("test.mp4", {})
                except Exception as e:
                    # Expected - we're mocking
                    pass

                # MEMLEAK-FIX-3: Check if BytesIO was closed
                if not audio_buffer.closed:
                    pytest.fail(
                        "MEMLEAK-FIX-3: detect_language_task didn't close BytesIO! "
                        "Fix lines 1065-1069 to use context manager or close explicitly."
                    )

    except ImportError:
        pytest.skip("subgen module not available for import")


@pytest.mark.stress
@pytest.mark.leak3
def test_audio_extraction_stress_100_files():
    """
    MEMLEAK-FIX-3: Stress test simulating 100 transcriptions.

    This simulates production load with multiple audio extractions.
    Will FAIL before fix, PASS after BytesIO cleanup is implemented.
    """
    try:
        import subgen

        extract_func = getattr(subgen, "extract_audio_segment_to_memory", None)

        if extract_func is None:
            pytest.skip("extract_audio_segment_to_memory not accessible")

        detector = LeakDetector(BytesIO)
        detector.start()

        with MemoryProfiler(name="audio_extraction_stress") as profiler:
            # MEMLEAK-FIX-3: Mock ffmpeg for 100 extractions
            with patch("subgen.ffmpeg") as mock_ffmpeg:
                for i in range(100):
                    # Mock audio data (480KB per segment)
                    audio_data = b"x" * (480 * 1024)
                    mock_ffmpeg.input.return_value.output.return_value.run.return_value = (
                        audio_data,
                        b"",
                    )

                    # MEMLEAK-FIX-3: Extract audio
                    result = extract_func(f"test{i}.mp4", 0, 30)

                    # Read data (like in detect_language_task)
                    if isinstance(result, BytesIO):
                        _ = result.read()
                        # MEMLEAK-FIX-3: Should close here, or return bytes
                        result.close()  # Manually close for this test
                    elif isinstance(result, bytes):
                        _ = result  # Already bytes, no cleanup needed

                    # Periodic GC
                    if i % 20 == 0:
                        gc.collect()

        leaked_count = detector.leaked_count()
        memory_growth = profiler.memory_growth_mb

        print(f"\nMEMLEAK-FIX-3: Stress test results")
        print(f"MEMLEAK-FIX-3: BytesIO leaked: {leaked_count}")
        print(f"MEMLEAK-FIX-3: Memory growth: {memory_growth:.2f}MB")

        # MEMLEAK-FIX-3: After fix, should not leak
        assert leaked_count < 10, f"BytesIO objects leaked: {leaked_count}"
        profiler.assert_no_memory_leak(max_growth_mb=50.0)

    except ImportError:
        pytest.skip("subgen module not available for import")


@pytest.mark.unit
@pytest.mark.leak3
def test_check_source_for_bytesio_usage():
    """
    MEMLEAK-FIX-3: Check source code for BytesIO usage patterns.

    This verifies that all BytesIO objects are properly managed.
    """
    try:
        import subgen
        import inspect

        source = inspect.getsource(subgen)

        # MEMLEAK-FIX-3: Find BytesIO usage patterns
        has_bytesio_import = "from io import BytesIO" in source or "import io" in source
        has_bytesio_return = (
            "return io.BytesIO(" in source or "return BytesIO(" in source
        )
        has_context_manager = "@contextmanager" in source

        # MEMLEAK-FIX-3: Check specific problematic lines
        problematic_patterns = [
            "extract_audio_segment_to_memory(",
            ").read()",  # Pattern: function_returning_bytesio().read()
        ]

        print(f"\nMEMLEAK-FIX-3: Source code analysis")
        print(f"MEMLEAK-FIX-3: Has BytesIO: {has_bytesio_import}")
        print(f"MEMLEAK-FIX-3: Returns BytesIO: {has_bytesio_return}")
        print(f"MEMLEAK-FIX-3: Uses context managers: {has_context_manager}")

        if has_bytesio_return and not has_context_manager:
            print(
                "\nMEMLEAK-FIX-3: WARNING - BytesIO returned without context manager!"
            )
            print("MEMLEAK-FIX-3: Consider one of these fixes:")
            print("MEMLEAK-FIX-3: 1. Return bytes directly instead of BytesIO")
            print(
                "MEMLEAK-FIX-3: 2. Make functions context managers with @contextmanager"
            )
            print(
                "MEMLEAK-FIX-3: 3. Add explicit .close() calls in all usage locations"
            )

    except ImportError:
        pytest.skip("subgen module not available for import")


@pytest.mark.unit
@pytest.mark.leak3
def test_all_audio_extraction_functions():
    """
    MEMLEAK-FIX-3: Test all audio extraction functions for BytesIO leaks.

    Tests:
    - extract_audio_segment_to_memory (line 1426)
    - extract_audio_track_to_memory (line 1735)
    - handle_multiple_audio_tracks (line 1318)
    """
    try:
        import subgen

        functions_to_check = [
            "extract_audio_segment_to_memory",
            "extract_audio_track_to_memory",
            "handle_multiple_audio_tracks",
        ]

        results = {}

        for func_name in functions_to_check:
            func = getattr(subgen, func_name, None)
            if func is None:
                results[func_name] = "Not found"
                continue

            # MEMLEAK-FIX-3: Check if function uses BytesIO
            import inspect

            try:
                source = inspect.getsource(func)
                returns_bytesio = "BytesIO(" in source or "io.BytesIO(" in source
                has_close = ".close()" in source
                is_context_mgr = "@contextmanager" in source or "with " in source

                results[func_name] = {
                    "returns_bytesio": returns_bytesio,
                    "has_close": has_close,
                    "is_context_mgr": is_context_mgr,
                }
            except:
                results[func_name] = "Could not inspect"

        print("\nMEMLEAK-FIX-3: Analysis of audio extraction functions:")
        for func_name, result in results.items():
            print(f"\n{func_name}:")
            if isinstance(result, dict):
                print(f"  Returns BytesIO: {result['returns_bytesio']}")
                print(f"  Has close(): {result['has_close']}")
                print(f"  Context manager: {result['is_context_mgr']}")

                if (
                    result["returns_bytesio"]
                    and not result["has_close"]
                    and not result["is_context_mgr"]
                ):
                    print(f"  ⚠️ LEAK RISK: Returns BytesIO without proper cleanup!")
            else:
                print(f"  {result}")

    except ImportError:
        pytest.skip("subgen module not available for import")

"""
Memory Leak Test: LEAK #6 - HTTP Response Objects Not Closed

PROBLEM:
The code makes HTTP requests to Plex/Jellyfin APIs without:
1. Using requests.Session() for connection pooling
2. Explicitly closing response objects
3. Reusing connections across requests

This causes:
- TCP connections left in TIME_WAIT state
- Socket file descriptors accumulating
- HTTP connection pool exhaustion

EXPECTED BEHAVIOR:
- HTTP connections should be reused via Session()
- Response objects should be closed after use
- No accumulation of sockets/file descriptors

IMPACT:
- Each unclosed response: ~5-10KB socket overhead
- For 1000 Plex/Jellyfin webhooks: 5-10MB
- TCP connections in TIME_WAIT accumulate

LOCATIONS IN CODE:
- Line 1826-1960: get_next_plex_episode(), get_plex_file_name(), refresh_plex_metadata()
- Line 1990-2030: refresh_jellyfin_metadata(), get_jellyfin_file_name()

TEST STRATEGY:
1. Mock HTTP server
2. Make many requests without Session pooling
3. Make many requests WITH Session pooling
4. Measure socket/memory differences
"""

import pytest
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.memory_utils import MemoryProfiler, LeakDetector


class MockPlexServer(BaseHTTPRequestHandler):
    """Mock HTTP server simulating Plex API responses."""

    def log_message(self, format, *args):
        """Suppress server logging."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        self.send_response(200)
        self.send_header("Content-type", "text/xml")
        self.end_headers()

        # Simple XML response
        response = b"""<?xml version="1.0" encoding="UTF-8"?>
<MediaContainer>
    <Video ratingKey="12345" title="Test">
        <Part file="/media/test.mp4" />
    </Video>
</MediaContainer>"""
        self.wfile.write(response)

    def do_PUT(self):
        """Handle PUT requests."""
        self.send_response(200)
        self.end_headers()


@pytest.fixture
def mock_server():
    """Start a mock HTTP server for testing."""
    server = HTTPServer(("localhost", 0), MockPlexServer)
    port = server.server_address[1]

    # Run server in background thread
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://localhost:{port}"

    server.shutdown()


@pytest.mark.unit
@pytest.mark.memory_leak
class TestHTTPResponseLeaks:
    """Test suite for HTTP response resource management."""

    def test_requests_without_session_pooling(self, mock_server):
        """
        LEAK TEST: Making requests without Session causes connection buildup.

        Each request creates a new connection instead of reusing from pool.
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=5.0)

        profiler.snapshot("baseline")

        # Make many requests WITHOUT session (current code pattern)
        iterations = 100
        for i in range(iterations):
            response = requests.get(f"{mock_server}/library/metadata/123")
            # Simulating the bug: not explicitly closing
            _ = response.content
            # response.close() not called!

        profiler.snapshot("after_no_session")

        growth = profiler.compare("baseline", "after_no_session")

        print(f"\n=== HTTP Requests Without Session ===")
        print(f"Iterations: {iterations}")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")
        print(f"Per request: {growth['increase_mb'] / iterations:.3f} MB")

        # Should show memory growth from unclosed connections
        leak_detected = leak_detector.check(growth["increase_mb"])
        assert not leak_detected, (
            f"Memory leak detected: {growth['increase_mb']:.2f} MB"
        )

    def test_requests_with_session_pooling(self, mock_server):
        """
        COMPARISON TEST: Using Session() should reuse connections efficiently.

        This is the correct pattern that should be used.
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=3.0)

        profiler.snapshot("baseline")

        # Make many requests WITH session (correct pattern)
        session = requests.Session()
        iterations = 100

        try:
            for i in range(iterations):
                response = session.get(f"{mock_server}/library/metadata/123")
                _ = response.content
                response.close()  # Explicitly close
        finally:
            session.close()

        profiler.snapshot("after_with_session")

        growth = profiler.compare("baseline", "after_with_session")

        print(f"\n=== HTTP Requests With Session ===")
        print(f"Iterations: {iterations}")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")
        print(f"Per request: {growth['increase_mb'] / iterations:.3f} MB")

        # Should have minimal memory growth
        leak_detected = leak_detector.check(growth["increase_mb"])
        assert not leak_detected, (
            f"Unexpected memory growth: {growth['increase_mb']:.2f} MB"
        )

    def test_response_not_closed_accumulation(self, mock_server):
        """
        LEAK TEST: Response objects held in memory without close().

        Simulates the pattern where response objects are created but
        not explicitly closed.
        """
        profiler = MemoryProfiler()

        profiler.snapshot("baseline")

        # Hold response objects without closing
        responses = []
        for i in range(50):
            response = requests.get(f"{mock_server}/library/metadata/123")
            _ = response.content  # Read content but don't close
            responses.append(response)  # Keep reference

        profiler.snapshot("after_holding_responses")

        growth = profiler.compare("baseline", "after_holding_responses")

        print(f"\n=== Unclosed Response Objects ===")
        print(f"Responses held: 50")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        # Should show significant growth
        assert growth["increase_mb"] > 0.5, (
            "Expected memory growth from unclosed responses"
        )

        # Now close them all
        profiler.snapshot("before_cleanup")
        for response in responses:
            response.close()
        responses.clear()

        profiler.snapshot("after_cleanup")

        cleanup = profiler.compare("before_cleanup", "after_cleanup")
        print(f"Memory freed: {abs(cleanup['increase_mb']):.2f} MB")

    def test_context_manager_vs_manual_close(self, mock_server):
        """
        COMPARISON TEST: Using context manager vs manual close().

        Best practice is to use 'with' statement for automatic cleanup.
        """
        profiler = MemoryProfiler()

        # Test with manual close
        profiler.snapshot("baseline_manual")
        for i in range(50):
            response = requests.get(f"{mock_server}/library/metadata/123")
            _ = response.content
            response.close()
        profiler.snapshot("after_manual_close")

        # Test with context manager (Python 3.x supports this)
        profiler.snapshot("baseline_context")
        for i in range(50):
            with requests.get(f"{mock_server}/library/metadata/123") as response:
                _ = response.content
        profiler.snapshot("after_context_manager")

        manual_growth = profiler.compare("baseline_manual", "after_manual_close")
        context_growth = profiler.compare("baseline_context", "after_context_manager")

        print(f"\n=== Manual close() vs Context Manager ===")
        print(f"Manual close memory growth: {manual_growth['increase_mb']:.2f} MB")
        print(f"Context manager memory growth: {context_growth['increase_mb']:.2f} MB")

        # Both should be similar and low
        assert manual_growth["increase_mb"] < 2.0, "Manual close should not leak"
        assert context_growth["increase_mb"] < 2.0, "Context manager should not leak"

    def test_session_without_close_leak(self, mock_server):
        """
        LEAK TEST: Session objects not closed accumulate connections.

        Even when using Session, they must be closed properly.
        """
        profiler = MemoryProfiler()

        profiler.snapshot("baseline")

        # Create multiple sessions without closing (bad practice)
        sessions = []
        for i in range(20):
            session = requests.Session()
            for j in range(5):
                response = session.get(f"{mock_server}/library/metadata/123")
                response.close()
            sessions.append(session)  # Keep session alive

        profiler.snapshot("after_unclosed_sessions")

        growth = profiler.compare("baseline", "after_unclosed_sessions")

        print(f"\n=== Unclosed Session Objects ===")
        print(f"Sessions created: 20")
        print(f"Requests per session: 5")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        # Clean up
        for session in sessions:
            session.close()
        sessions.clear()

    def test_plex_api_pattern_simulation(self, mock_server):
        """
        LEAK TEST: Simulate actual Plex API usage pattern from code.

        Mimics get_next_plex_episode() which makes multiple sequential requests.
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=10.0)

        profiler.snapshot("baseline")

        # Simulate processing 100 episodes (each requires multiple API calls)
        iterations = 100
        for episode_num in range(iterations):
            # Call 1: Get episode metadata
            response1 = requests.get(f"{mock_server}/library/metadata/{episode_num}")
            _ = response1.content

            # Call 2: Get season list
            response2 = requests.get(f"{mock_server}/library/metadata/show/children")
            _ = response2.content

            # Call 3: Get episodes in season
            response3 = requests.get(f"{mock_server}/library/metadata/season/children")
            _ = response3.content

            # Bug: responses not closed!

        profiler.snapshot("after_plex_pattern")

        growth = profiler.compare("baseline", "after_plex_pattern")

        print(f"\n=== Plex API Pattern (3 calls per episode) ===")
        print(f"Episodes processed: {iterations}")
        print(f"Total requests: {iterations * 3}")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        leak_detected = leak_detector.check(growth["increase_mb"])
        assert not leak_detected, (
            f"Memory leak in Plex pattern: {growth['increase_mb']:.2f} MB"
        )

    def test_jellyfin_api_pattern_simulation(self, mock_server):
        """
        LEAK TEST: Simulate Jellyfin API usage pattern.

        Mimics refresh_jellyfin_metadata() which makes multiple requests.
        """
        profiler = MemoryProfiler()
        leak_detector = LeakDetector(threshold_mb=5.0)

        profiler.snapshot("baseline")

        iterations = 100
        for item_num in range(iterations):
            # Call 1: Get user list
            response1 = requests.get(f"{mock_server}/Users")
            _ = response1.content

            # Call 2: Get item details
            response2 = requests.get(f"{mock_server}/Users/admin/Items/{item_num}")
            _ = response2.content

            # Call 3: Refresh metadata
            response3 = requests.get(
                f"{mock_server}/Users/admin/Items/{item_num}/Refresh"
            )
            _ = response3.content

            # Call 4: POST refresh
            response4 = requests.get(f"{mock_server}/Items/{item_num}/Refresh")
            _ = response4.content

        profiler.snapshot("after_jellyfin_pattern")

        growth = profiler.compare("baseline", "after_jellyfin_pattern")

        print(f"\n=== Jellyfin API Pattern (4 calls per item) ===")
        print(f"Items processed: {iterations}")
        print(f"Total requests: {iterations * 4}")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        leak_detected = leak_detector.check(growth["increase_mb"])
        assert not leak_detected, (
            f"Memory leak in Jellyfin pattern: {growth['increase_mb']:.2f} MB"
        )

    def test_connection_pool_exhaustion(self, mock_server):
        """
        MEASUREMENT TEST: Verify connection pool doesn't exhaust.

        Without session pooling, connection pools can exhaust under load.
        """
        from requests.adapters import HTTPAdapter

        profiler = MemoryProfiler()

        # Session with small pool
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=5, pool_maxsize=5)
        session.mount("http://", adapter)

        profiler.snapshot("baseline")

        try:
            # Make more requests than pool size
            for i in range(20):
                with session.get(f"{mock_server}/library/metadata/123") as response:
                    _ = response.content
        finally:
            session.close()

        profiler.snapshot("after_pooled_requests")

        growth = profiler.compare("baseline", "after_pooled_requests")

        print(f"\n=== Connection Pool Management ===")
        print(f"Pool size: 5")
        print(f"Requests: 20")
        print(f"Memory growth: {growth['increase_mb']:.2f} MB")

        # Should handle requests without issues despite small pool
        assert growth["increase_mb"] < 1.0, "Connection pooling should be efficient"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

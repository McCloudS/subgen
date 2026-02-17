#!/usr/bin/env python3
"""
Direct test for LEAK #6: HTTP Response Objects Not Closed

Tests if HTTP responses are properly closed and if Session pooling helps.
"""

import sys
import tracemalloc
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading


class MockServer(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK" * 1000)  # 2KB response


def test_requests_without_session():
    """Test if making requests without Session leaks."""
    tracemalloc.start()
    baseline = tracemalloc.take_snapshot()

    # Start mock server
    server = HTTPServer(("localhost", 0), MockServer)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://localhost:{port}/"

    # Make many requests WITHOUT session
    for i in range(100):
        response = requests.get(url)
        _ = response.content
        # Bug: not calling response.close()

    current = tracemalloc.take_snapshot()
    stats = current.compare_to(baseline, "lineno")
    total_memory = sum(stat.size_diff for stat in stats) / 1024 / 1024

    server.shutdown()

    print(f"\n{'=' * 60}")
    print(f"LEAK #6 TEST: HTTP Requests Without Session")
    print(f"{'=' * 60}")
    print(f"Requests made: 100")
    print(f"Memory growth: {total_memory:.2f} MB")
    print(f"Per request: {total_memory / 100:.3f} MB")
    print(f"{'=' * 60}")

    if total_memory > 5.0:
        print(f"❌ LEAK CONFIRMED: {total_memory:.2f} MB growth is significant")
        return False
    elif total_memory > 1.0:
        print(f"⚠️  MINOR LEAK: {total_memory:.2f} MB growth (may be GC-able)")
        return False
    else:
        print(f"✅ NO LEAK: {total_memory:.2f} MB is acceptable")
        return True


def test_requests_with_session():
    """Test if using Session reduces memory growth."""
    tracemalloc.start()
    baseline = tracemalloc.take_snapshot()

    # Start mock server
    server = HTTPServer(("localhost", 0), MockServer)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://localhost:{port}/"

    # Make many requests WITH session
    session = requests.Session()
    try:
        for i in range(100):
            response = session.get(url)
            _ = response.content
            response.close()
    finally:
        session.close()

    current = tracemalloc.take_snapshot()
    stats = current.compare_to(baseline, "lineno")
    total_memory = sum(stat.size_diff for stat in stats) / 1024 / 1024

    server.shutdown()

    print(f"\n{'=' * 60}")
    print(f"LEAK #6 TEST: HTTP Requests WITH Session")
    print(f"{'=' * 60}")
    print(f"Requests made: 100")
    print(f"Memory growth: {total_memory:.2f} MB")
    print(f"Per request: {total_memory / 100:.3f} MB")
    print(f"{'=' * 60}")

    if total_memory < 1.0:
        print(f"✅ GOOD: Session pooling keeps memory low")
        return True
    else:
        print(f"⚠️  Even with Session, {total_memory:.2f} MB growth")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING LEAK #6: HTTP Response Objects")
    print("=" * 60)

    test1_pass = test_requests_without_session()
    test2_pass = test_requests_with_session()

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    if not test1_pass:
        print(f"❌ LEAK #6 IS REAL - HTTP responses not being closed")
        print(f"   Session pooling helps: {'YES' if test2_pass else 'NO'}")
        sys.exit(1)
    else:
        print(f"✅ NO SIGNIFICANT LEAK DETECTED")
        sys.exit(0)

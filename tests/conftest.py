"""
conftest.py — patches heavy ML/AV dependencies before any test module imports subgen.

The mock setup must happen here, at collection time, before subgen.py is imported.
subgen.py starts worker threads at import time; they're daemon threads and are harmless.
"""
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy dependencies that are not installed in CI
# ---------------------------------------------------------------------------
_MOCKED_MODULES = [
    "faster_whisper",
    "torch",
    "av",
    "ffmpeg",
    "watchdog",
    "watchdog.observers",
    "watchdog.observers.polling",
    "watchdog.events",
    "numpy",
]

for _mod in _MOCKED_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Give the version mock a usable string so /status doesn't blow up
sys.modules["faster_whisper"].__version__ = "1.0.0"

# Ensure watchdog attribute imports work
# e.g. `from watchdog.observers.polling import PollingObserver as Observer`
sys.modules["watchdog.observers.polling"].PollingObserver = MagicMock()
# e.g. `from watchdog.events import FileSystemEventHandler`
sys.modules["watchdog.events"].FileSystemEventHandler = object

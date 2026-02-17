# Memory Leak Tests for Subgen

This directory contains test-driven development (TDD) tests for identifying and fixing memory leaks in subgen.

## Overview

Three critical memory leaks have been identified in subgen:

1. **MEMLEAK-FIX-1**: `task_results` dictionary leak (~500KB per ASR request)
2. **MEMLEAK-FIX-2**: Timer thread accumulation (~50KB per timer, thread count grows)
3. **MEMLEAK-FIX-3**: BytesIO objects not closed (480KB - 10MB per transcription)

## Test Structure

```
tests/
├── __init__.py
├── README.md                          # This file
├── memory_utils.py                    # Memory profiling utilities
├── unit/
│   ├── test_leak1_task_results.py    # Tests for task_results dictionary leak
│   ├── test_leak2_timer_threads.py   # Tests for Timer thread accumulation
│   └── test_leak3_bytesio.py         # Tests for BytesIO context manager leak
├── integration/                       # Integration tests (future)
└── stress/                            # Stress tests (future)
```

## Running Tests

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or `./venv/bin/activate` on Linux

# Install dependencies
pip install pytest pytest-asyncio pytest-timeout
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run Tests by Memory Leak

```bash
# Test task_results dictionary leak (MEMLEAK-FIX-1)
pytest tests/ -v -m leak1

# Test Timer thread accumulation (MEMLEAK-FIX-2)
pytest tests/ -v -m leak2

# Test BytesIO leaks (MEMLEAK-FIX-3)
pytest tests/ -v -m leak3
```

### Run Tests by Type

```bash
# Unit tests only (fast)
pytest tests/ -v -m unit

# Integration tests (slower)
pytest tests/ -v -m integration

# Stress tests (slow, high memory usage)
pytest tests/ -v -m stress
```

### Run Specific Test File

```bash
# Test task_results leak
pytest tests/unit/test_leak1_task_results.py -v

# Test Timer threads
pytest tests/unit/test_leak2_timer_threads.py -v

# Test BytesIO leak
pytest tests/unit/test_leak3_bytesio.py -v
```

## Test Markers

Tests are marked with pytest markers for easy filtering:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests requiring subgen components
- `@pytest.mark.stress` - Stress tests (slow, high memory usage)
- `@pytest.mark.leak1` - Tests for task_results dictionary leak
- `@pytest.mark.leak2` - Tests for Timer thread accumulation
- `@pytest.mark.leak3` - Tests for BytesIO leak

## Test-Driven Development Approach

### Phase 1: Baseline (Current State)

**Status**: Tests demonstrate leaks exist

- Tests are designed to **FAIL** before fixes are applied
- Failures prove that memory leaks exist
- Baseline metrics captured for comparison

### Phase 2: Fix Implementation

**Status**: In progress

- Fix code based on failing tests
- Tests should **PASS** after fixes applied
- Memory usage should be significantly reduced

### Phase 3: Validation

**Status**: Pending

- All tests pass
- Memory usage stable over time
- Production-ready

## Expected Test Results

### Before Fixes Applied

```
MEMLEAK-FIX-1: task_results dictionary leak
  ✗ test_task_results_dict_grows_unbounded - FAILS (expected)
  ⊘ test_task_results_cache_cleanup - SKIPS (not implemented yet)
  ⊘ test_no_memory_growth_with_proper_cleanup - SKIPS

MEMLEAK-FIX-2: Timer thread accumulation
  ✗ test_timer_threads_accumulate_without_join - FAILS (expected)
  ✓ test_timer_cleanup_with_join - PASSES (shows correct pattern)
  ✗ test_schedule_model_cleanup_pattern - FAILS (old pattern)

MEMLEAK-FIX-3: BytesIO objects not closed
  ✗ test_bytesio_leaks_without_close - FAILS (expected)
  ✓ test_bytesio_no_leak_with_close - PASSES (shows correct pattern)
  ✓ test_bytesio_context_manager_pattern - PASSES (recommended fix)
```

### After Fixes Applied

```
All tests should PASS:

MEMLEAK-FIX-1: task_results dictionary leak
  ✓ test_task_results_dict_grows_unbounded - PASSES
  ✓ test_task_results_cache_cleanup - PASSES
  ✓ test_no_memory_growth_with_proper_cleanup - PASSES

MEMLEAK-FIX-2: Timer thread accumulation
  ✓ test_timer_threads_accumulate_without_join - PASSES (with join)
  ✓ test_timer_cleanup_with_join - PASSES
  ✓ test_schedule_model_cleanup_pattern - PASSES (fixed pattern)

MEMLEAK-FIX-3: BytesIO objects not closed
  ✓ test_bytesio_leaks_without_close - PASSES (with close)
  ✓ test_bytesio_no_leak_with_close - PASSES
  ✓ test_bytesio_context_manager_pattern - PASSES
```

## Memory Profiling Utilities

### MemoryProfiler

Context manager for tracking memory usage:

```python
from tests.memory_utils import MemoryProfiler

with MemoryProfiler(name="my_test") as profiler:
    # Run code that might leak
    for i in range(1000):
        create_objects()

# Check results
print(f"Memory growth: {profiler.memory_growth_mb:.2f}MB")
print(f"Thread growth: {profiler.thread_growth}")

# Assert no leaks
profiler.assert_no_memory_leak(max_growth_mb=10.0)
profiler.assert_no_thread_leak(max_growth=5)
```

### LeakDetector

Track specific object types:

```python
from tests.memory_utils import LeakDetector
from io import BytesIO

detector = LeakDetector(BytesIO)
detector.start()

# Run code that might leak BytesIO
for i in range(100):
    buffer = BytesIO(b"data")
    # Oops, didn't close it!

# Check for leaks
leaked = detector.leaked_count()
assert leaked < 10, f"{leaked} BytesIO objects leaked"
```

### measure_memory_growth

Measure memory/thread growth over iterations:

```python
from tests.memory_utils import measure_memory_growth

def my_function():
    # Function to test
    pass

memory_mb, threads = measure_memory_growth(
    my_function,
    iterations=100,
    cleanup_interval=10
)

print(f"Memory growth: {memory_mb:.2f}MB over 100 iterations")
print(f"Thread growth: {threads} threads")
```

## Greppable Tags

All test comments are tagged with greppable identifiers:

```bash
# Find all task_results leak comments
grep -r "MEMLEAK-FIX-1" tests/

# Find all Timer thread leak comments
grep -r "MEMLEAK-FIX-2" tests/

# Find all BytesIO leak comments
grep -r "MEMLEAK-FIX-3" tests/

# Find all memory leak comments
grep -r "MEMLEAK-FIX" tests/
```

This makes it easy to:
- Track all comments related to a specific leak
- Remove comments later if desired
- Generate documentation
- Review changes by leak type

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Memory Leak Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install pytest pytest-asyncio pytest-timeout
      - name: Run unit tests
        run: pytest tests/ -v -m unit
      - name: Run stress tests
        run: pytest tests/ -v -m stress
        continue-on-error: true  # Stress tests may be flaky
```

## Memory Leak Details

### Leak #1: task_results Dictionary

**Location**: `subgen.py` lines 748-751 (task_results usage in /asr endpoint)  
**Impact**: ~500KB per ASR request  
**Fix**: Complete TaskResultCache refactoring

```python
# Before (leaks - line 750):
with task_results_lock:
    if task_id not in task_results:
        task_results[task_id] = TaskResult()
    task_result = task_results[task_id]

# After (fixed):
task_result = task_results_cache.add(task_id)
```

### Leak #2: Timer Thread Accumulation

**Location**: `subgen.py` lines 1149-1163 (schedule_model_cleanup function)  
**Impact**: ~50KB per timer, thread count grows  
**Fix**: Add `timer.join()` after `timer.cancel()`

```python
# Before (leaks - line 1156):
if model_cleanup_timer is not None:
    model_cleanup_timer.cancel()

# After (fixed):
if model_cleanup_timer is not None:
    model_cleanup_timer.cancel()
    model_cleanup_timer.join()  # MEMLEAK-FIX-2: Add this line
```

### Leak #3: BytesIO Not Closed

**Location**: `subgen.py` multiple locations:
- Lines 1065-1069 (detect_language_task)
- Lines 1100-1141 (extract_audio_segment_to_memory)
- Lines 1245-1247 (gen_subtitles)
- Line 1346 (handle_multiple_audio_tracks)  
- Lines 1352-1386 (extract_audio_track_to_memory)

**Impact**: 480KB - 10MB per transcription  
**Fix**: Return bytes directly or use context managers

```python
# Before (leaks - lines 1065-1069):
audio_segment = extract_audio_segment_to_memory(
    path, 
    detect_language_offset, 
    int(detect_language_length)
).read()  # BytesIO never closed!

# Option 1 - Return bytes directly (simplest):
def extract_audio_segment_to_memory(...):
    # ... ffmpeg processing ...
    return out  # Return bytes directly

# Option 2 - Context manager:
@contextmanager
def extract_audio_segment_to_memory(...):
    buffer = io.BytesIO(out)
    try:
        yield buffer
    finally:
        buffer.close()
```

## Contributing

When adding new tests:

1. Use appropriate markers (`@pytest.mark.unit`, `@pytest.mark.leak1`, etc.)
2. Add `MEMLEAK-FIX-N:` tags to all comments
3. Write docstrings explaining what the test validates
4. Include expected behavior before and after fix
5. Update this README with new test information

## References

- [Python tracemalloc documentation](https://docs.python.org/3/library/tracemalloc.html)
- [pytest markers](https://docs.pytest.org/en/stable/how-to/mark.html)
- [Context managers](https://docs.python.org/3/library/contextlib.html)
- [Memory profiling in Python](https://docs.python.org/3/library/gc.html)

## Additional Memory Leaks (Phase 2)

Seven additional memory leaks have been identified for remediation:

4. **LEAK #4**: `DeduplicatedQueue` internal sets never cleaned (~5-20MB per 1000 ops)
5. **LEAK #5**: PyAV container objects not explicitly closed (~10-50MB per 100 files)
6. **LEAK #6**: HTTP response objects not closed (~5-10MB per 1000 requests)
7. **LEAK #7**: XML ElementTree objects not cleared (~10-50MB per 1000 parses)
8. **LEAK #8**: File Observer never stopped (~2-5MB, thread leak)
9. **LEAK #9**: Whisper model segments accumulate (~1-5MB per 1000 transcriptions)
10. **LEAK #10**: FFmpeg process pipes not flushed (~0.5-2MB, edge cases)

### Extended Test Structure

```
tests/
├── unit/
│   ├── test_leak1_task_results.py        # MEMLEAK-FIX-1 (FIXED)
│   ├── test_leak2_timer_threads.py       # MEMLEAK-FIX-2 (FIXED)
│   ├── test_leak3_bytesio.py             # MEMLEAK-FIX-3 (FIXED)
│   ├── test_leak4_deduplicated_queue.py  # LEAK #4 (NEW)
│   ├── test_leak5_pyav_containers.py     # LEAK #5 (NEW)
│   ├── test_leak6_http_responses.py      # LEAK #6 (NEW)
│   ├── test_leak7_xml_elementtree.py     # LEAK #7 (NEW)
│   └── test_leak8_9_10_combined.py       # LEAKS #8, #9, #10 (NEW)
```

### Running Tests for New Leaks

```bash
# All new leak tests
pytest tests/unit/test_leak*.py -v

# Individual leak tests
pytest tests/unit/test_leak4_deduplicated_queue.py -v
pytest tests/unit/test_leak5_pyav_containers.py -v
pytest tests/unit/test_leak6_http_responses.py -v
pytest tests/unit/test_leak7_xml_elementtree.py -v
pytest tests/unit/test_leak8_9_10_combined.py -v

# All memory leak tests (including original 3)
pytest tests/ -v -m memory_leak
```

### Priority Order for Fixes

**Priority 1 (Critical)**:
- LEAK #4: DeduplicatedQueue - unbounded growth in high-volume scenarios
- LEAK #5: PyAV containers - FFmpeg resource accumulation

**Priority 2 (Important)**:
- LEAK #6: HTTP responses - affects Plex/Jellyfin integration
- LEAK #8: File Observer - thread and memory leak in monitor mode

**Priority 3 (Nice-to-Have)**:
- LEAK #7: XML ElementTree - GC eventually cleans, but optimization worthwhile
- LEAK #9: Whisper segments - small but accumulates
- LEAK #10: FFmpeg pipes - edge case, low frequency


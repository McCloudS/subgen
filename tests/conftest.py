import importlib
import sys
import types

import pytest


def _install_stub(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)


def _build_fastapi_stub():
    fastapi = types.ModuleType("fastapi")

    class FastAPI:
        def get(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def post(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    def _param(*args, **kwargs):
        return None

    fastapi.FastAPI = FastAPI
    fastapi.File = _param
    fastapi.UploadFile = type("UploadFile", (), {})
    fastapi.Query = _param
    fastapi.Header = _param
    fastapi.Body = _param
    fastapi.Form = _param
    fastapi.Request = type("Request", (), {})

    responses = types.ModuleType("fastapi.responses")

    class StreamingResponse:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    responses.StreamingResponse = StreamingResponse

    return fastapi, responses


def _build_watchdog_stub():
    watchdog = types.ModuleType("watchdog")
    observers = types.ModuleType("watchdog.observers")
    polling = types.ModuleType("watchdog.observers.polling")
    events = types.ModuleType("watchdog.events")

    class PollingObserver:
        def schedule(self, *args, **kwargs):
            return None

        def start(self):
            return None

    class FileSystemEventHandler:
        pass

    polling.PollingObserver = PollingObserver
    observers.polling = polling
    events.FileSystemEventHandler = FileSystemEventHandler
    watchdog.observers = observers
    watchdog.events = events

    return watchdog, observers, polling, events


def _build_stable_whisper_stub():
    stable_whisper = types.ModuleType("stable_whisper")

    class Segment:
        def __init__(self, start, end, text, words=None, id=0):
            self.start = start
            self.end = end
            self.text = text
            self.words = words or []
            self.id = id

    class DummyResult:
        def __init__(self, language="en"):
            self.language = language
            self.segments = [Segment(0, 1, "hello", [], 0)]
            self.to_srt_vtt_calls = []

        def to_srt_vtt(self, filepath=None, word_level=False):
            self.to_srt_vtt_calls.append((filepath, word_level))
            return ["dummy"]

    class DummyModel:
        def __init__(self, language="en"):
            self.language = language
            self.model = types.SimpleNamespace(unload_model=lambda: None)
            self.transcribe_calls = []

        def transcribe(self, *args, **kwargs):
            self.transcribe_calls.append((args, kwargs))
            return DummyResult(language=self.language)

    def load_faster_whisper(*args, **kwargs):
        return DummyModel()

    stable_whisper.Segment = Segment
    stable_whisper.DummyResult = DummyResult
    stable_whisper.DummyModel = DummyModel
    stable_whisper.load_faster_whisper = load_faster_whisper
    stable_whisper.__version__ = "0.0"

    return stable_whisper


def _build_ffmpeg_stub():
    ffmpeg = types.ModuleType("ffmpeg")

    class Error(Exception):
        def __init__(self, message="", stderr=b""):
            super().__init__(message)
            self.stderr = stderr

    class DummyInput:
        def output(self, *args, **kwargs):
            return self

        def run(self, *args, **kwargs):
            return b"", b""

    def input_stub(*args, **kwargs):
        return DummyInput()

    def probe_stub(*args, **kwargs):
        return {"streams": []}

    ffmpeg.Error = Error
    ffmpeg.input = input_stub
    ffmpeg.probe = probe_stub

    return ffmpeg


def _build_av_stub():
    av = types.ModuleType("av")

    class FFmpegError(Exception):
        pass

    def open_stub(*args, **kwargs):
        raise FFmpegError("av.open stub not configured")

    av.FFmpegError = FFmpegError
    av.open = open_stub

    return av


def _build_requests_stub():
    requests = types.ModuleType("requests")

    class RequestException(Exception):
        pass

    class Exceptions:
        pass

    Exceptions.RequestException = RequestException

    def _not_implemented(*args, **kwargs):
        raise NotImplementedError("requests stub not configured")

    requests.get = _not_implemented
    requests.post = _not_implemented
    requests.put = _not_implemented
    requests.exceptions = Exceptions

    return requests


def _build_torch_stub():
    torch = types.ModuleType("torch")

    class DummyCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

    torch.cuda = DummyCuda
    return torch


@pytest.fixture
def subgen_module(monkeypatch):
    monkeypatch.setenv("CONCURRENT_TRANSCRIPTIONS", "0")
    monkeypatch.setenv("MONITOR", "False")

    fastapi, fastapi_responses = _build_fastapi_stub()
    watchdog, observers, polling, events = _build_watchdog_stub()
    stable_whisper = _build_stable_whisper_stub()
    ffmpeg = _build_ffmpeg_stub()
    av = _build_av_stub()
    requests = _build_requests_stub()
    torch = _build_torch_stub()

    _install_stub(monkeypatch, "fastapi", fastapi)
    _install_stub(monkeypatch, "fastapi.responses", fastapi_responses)
    _install_stub(monkeypatch, "watchdog", watchdog)
    _install_stub(monkeypatch, "watchdog.observers", observers)
    _install_stub(monkeypatch, "watchdog.observers.polling", polling)
    _install_stub(monkeypatch, "watchdog.events", events)
    _install_stub(monkeypatch, "stable_whisper", stable_whisper)
    _install_stub(monkeypatch, "faster_whisper", types.ModuleType("faster_whisper"))
    _install_stub(monkeypatch, "whisper", types.ModuleType("whisper"))
    _install_stub(monkeypatch, "ffmpeg", ffmpeg)
    _install_stub(monkeypatch, "av", av)
    _install_stub(monkeypatch, "requests", requests)
    _install_stub(monkeypatch, "torch", torch)

    sys.modules["faster_whisper"].__version__ = "0.0"

    if "subgen" in sys.modules:
        del sys.modules["subgen"]
    module = importlib.import_module("subgen")
    return module

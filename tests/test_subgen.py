import asyncio
import io
import json
import os
import types

import numpy as np

from language_code import LanguageCode


class DummyUploadFile:
    def __init__(self, data=b"audio"):
        self.file = io.BytesIO(data)

    async def close(self):
        return None


class AsyncFile:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    async def seek(self, pos):
        self.pos = pos

    async def read(self, length):
        chunk = self.data[self.pos : self.pos + length]
        self.pos += length
        return chunk


class DummyResponse:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class DummyStream:
    def __init__(self, metadata=None, stream_type="audio"):
        self.metadata = metadata or {}
        self.type = stream_type
        self.codec_context = types.SimpleNamespace(name="aac")


class DummyContainer:
    def __init__(self, streams):
        self.streams = streams

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyStreams:
    def __init__(self, subtitles=None, streams=None):
        self.subtitles = subtitles or []
        self._streams = streams if streams is not None else []

    def __iter__(self):
        return iter(self._streams)


def _configure_no_skip(subgen_module):
    subgen_module.transcribe_or_translate = "transcribe"
    subgen_module.lrc_for_audio_files = False
    subgen_module.skip_unknown_language = False
    subgen_module.skip_if_to_transcribe_sub_already_exist = False
    subgen_module.skipifinternalsublang = LanguageCode.NONE
    subgen_module.skipifexternalsub = False
    subgen_module.namesublang = ""
    subgen_module.skip_lang_codes_list = []
    subgen_module.limit_to_preferred_audio_languages = False
    subgen_module.preferred_audio_languages = []
    subgen_module.skip_if_audio_track_is_in_list = []
    subgen_module.only_skip_if_subgen_subtitle = False

    subgen_module.has_subtitle_language = lambda *args, **kwargs: False
    subgen_module.has_subtitle_language_in_file = lambda *args, **kwargs: False
    subgen_module.has_subtitle_of_language_in_folder = lambda *args, **kwargs: False
    subgen_module.get_subtitle_languages = lambda *args, **kwargs: []
    subgen_module.get_audio_languages = lambda *args, **kwargs: []


def test_convert_to_bool(subgen_module):
    convert = subgen_module.convert_to_bool
    assert convert("true")
    assert convert("On")
    assert convert("1")
    assert convert("y")
    assert convert("yes")
    assert not convert("false")
    assert not convert("0")
    assert not convert(None)


def test_get_env_with_fallback(monkeypatch, subgen_module):
    monkeypatch.setenv("NEW_NAME", "123")
    monkeypatch.setenv("OLD_NAME", "456")
    value = subgen_module.get_env_with_fallback(
        "NEW_NAME",
        "OLD_NAME",
        default_value="0",
        convert_func=int,
    )
    assert value == 123

    monkeypatch.delenv("NEW_NAME", raising=False)
    value = subgen_module.get_env_with_fallback("NEW_NAME", "OLD_NAME", default_value="0")
    assert value == "456"

    monkeypatch.delenv("OLD_NAME", raising=False)
    value = subgen_module.get_env_with_fallback("NEW_NAME", "OLD_NAME", default_value="fallback")
    assert value == "fallback"


def test_deduplicated_queue_tracks_processing(subgen_module):
    queue = subgen_module.DeduplicatedQueue()
    queue.put({"path": "one"})
    queue.put({"path": "one"})
    assert queue.qsize() == 1

    item = queue.get()
    assert item["path"] == "one"
    assert queue.get_processing_tasks() == ["one"]
    queue.task_done()
    assert queue.get_processing_tasks() == []


def test_progress_updates_timestamp(subgen_module):
    subgen_module.docker_status = "Docker"
    subgen_module.last_print_time = None
    subgen_module.progress(1, 10)
    assert subgen_module.last_print_time is not None


def test_appendLine_adds_segment(subgen_module):
    subgen_module.append = True
    result = subgen_module.stable_whisper.DummyResult(language="English")
    original_len = len(result.segments)
    subgen_module.appendLine(result)
    assert len(result.segments) == original_len + 1
    assert "Transcribed by whisperAI" in result.segments[-1].text


def test_handle_get_request(subgen_module):
    response = subgen_module.handle_get_request(None)
    assert isinstance(response, set)
    assert any("incorrectly via a GET request" in item for item in response)


def test_webui_and_status(subgen_module):
    response = subgen_module.webui()
    assert isinstance(response, set)
    assert any("webui for configuration was removed" in item.lower() for item in response)

    status = subgen_module.status()
    assert "Subgen" in status["version"]


def test_receive_tautulli_webhook_triggers_queue(monkeypatch, subgen_module):
    calls = []
    subgen_module.procaddedmedia = True
    subgen_module.procmediaonplay = False
    monkeypatch.setattr(subgen_module, "gen_subtitles_queue", lambda path, mode: calls.append((path, mode)))
    monkeypatch.setattr(subgen_module, "path_mapping", lambda path: f"mapped:{path}")

    result = subgen_module.receive_tautulli_webhook(
        source="Tautulli",
        event="added",
        file="/media/show.mkv",
    )
    assert result == ""
    assert calls == [("mapped:/media/show.mkv", subgen_module.transcribe_or_translate)]


def test_receive_tautulli_webhook_rejects_invalid(subgen_module):
    response = subgen_module.receive_tautulli_webhook(source="Other", event="added", file="/media/show.mkv")
    assert "properly configured" in response["message"]


def test_receive_plex_webhook(monkeypatch, subgen_module):
    calls = []
    refresh_calls = []
    subgen_module.procaddedmedia = True
    subgen_module.procmediaonplay = False
    subgen_module.plex_queue_next_episode = False
    subgen_module.plex_queue_season = False
    subgen_module.plex_queue_series = False

    monkeypatch.setattr(subgen_module, "get_plex_file_name", lambda *args, **kwargs: "/media/plex.mkv")
    monkeypatch.setattr(subgen_module, "refresh_plex_metadata", lambda *args, **kwargs: refresh_calls.append(args))
    monkeypatch.setattr(subgen_module, "gen_subtitles_queue", lambda path, mode: calls.append((path, mode)))
    monkeypatch.setattr(subgen_module, "path_mapping", lambda path: path)

    payload = {"event": "library.new", "Metadata": {"ratingKey": "1"}}
    result = subgen_module.receive_plex_webhook(user_agent="PlexMediaServer", payload=json.dumps(payload))
    assert result == ""
    assert calls == [("/media/plex.mkv", subgen_module.transcribe_or_translate)]
    assert refresh_calls


def test_receive_plex_webhook_rejects_invalid(subgen_module):
    payload = {"event": "library.new", "Metadata": {"ratingKey": "1"}}
    response = subgen_module.receive_plex_webhook(user_agent="Other", payload=json.dumps(payload))
    assert "properly configured Plex webhook" in response["message"]


def test_receive_jellyfin_webhook(monkeypatch, subgen_module):
    calls = []
    subgen_module.procaddedmedia = True
    subgen_module.procmediaonplay = False
    monkeypatch.setattr(subgen_module, "get_jellyfin_file_name", lambda *args, **kwargs: "/media/jellyfin.mkv")
    monkeypatch.setattr(subgen_module, "refresh_jellyfin_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(subgen_module, "gen_subtitles_queue", lambda path, mode: calls.append((path, mode)))
    monkeypatch.setattr(subgen_module, "path_mapping", lambda path: path)

    result = subgen_module.receive_jellyfin_webhook(
        user_agent="Jellyfin-Server",
        NotificationType="ItemAdded",
        file=None,
        ItemId="abc",
    )
    assert result == ""
    assert calls == [("/media/jellyfin.mkv", subgen_module.transcribe_or_translate)]


def test_receive_jellyfin_webhook_rejects_invalid(subgen_module):
    response = subgen_module.receive_jellyfin_webhook(
        user_agent="Other",
        NotificationType="ItemAdded",
        file=None,
        ItemId="abc",
    )
    assert "properly configured Jellyfin webhook" in response["message"]


def test_receive_emby_webhook(monkeypatch, subgen_module):
    calls = []
    subgen_module.procaddedmedia = True
    subgen_module.procmediaonplay = False
    monkeypatch.setattr(subgen_module, "gen_subtitles_queue", lambda path, mode: calls.append((path, mode)))
    monkeypatch.setattr(subgen_module, "path_mapping", lambda path: path)

    test_payload = {"Event": "system.notificationtest"}
    response = subgen_module.receive_emby_webhook(user_agent=None, data=json.dumps(test_payload))
    assert "Notification test received" in response["message"]

    payload = {"Event": "library.new", "Item": {"Path": "/media/emby.mkv"}}
    result = subgen_module.receive_emby_webhook(user_agent=None, data=json.dumps(payload))
    assert result == ""
    assert calls == [("/media/emby.mkv", subgen_module.transcribe_or_translate)]


def test_batch_calls_transcribe_existing(monkeypatch, subgen_module):
    calls = []
    monkeypatch.setattr(subgen_module, "transcribe_existing", lambda path, lang: calls.append((path, lang)))
    subgen_module.batch(directory="/media", forceLanguage="eng")
    assert calls == [("/media", LanguageCode.ENGLISH)]


def test_asr_success(monkeypatch, subgen_module):
    monkeypatch.setattr(subgen_module, "delete_model", lambda: None)
    subgen_module.append = False
    upload = DummyUploadFile(b"audio")
    response = asyncio.run(subgen_module.asr(task="transcribe", audio_file=upload, encode=True, output="srt"))
    assert isinstance(response, subgen_module.StreamingResponse)
    assert "Source" in response.kwargs["headers"]


def test_detect_language_forced(subgen_module):
    subgen_module.force_detected_language_to = LanguageCode.ENGLISH
    upload = DummyUploadFile(b"audio")
    result = asyncio.run(subgen_module.detect_language(audio_file=upload, encode=True))
    assert result["language_code"] == "en"


def test_detect_language_updates_window(monkeypatch, subgen_module):
    subgen_module.force_detected_language_to = LanguageCode.NONE
    subgen_module.model = subgen_module.stable_whisper.DummyModel(language="English")
    monkeypatch.setattr(subgen_module, "start_model", lambda: None)
    monkeypatch.setattr(subgen_module, "delete_model", lambda: None)
    monkeypatch.setattr(subgen_module, "extract_audio_segment_to_memory", lambda *args, **kwargs: io.BytesIO(b"data"))
    upload = DummyUploadFile(b"audio")

    result = asyncio.run(
        subgen_module.detect_language(
            audio_file=upload,
            encode=True,
            detect_lang_length=10,
            detect_lang_offset=2,
        )
    )
    assert result["language_code"] == "en"
    assert subgen_module.detect_language_length == 10
    assert subgen_module.detect_language_offset == 2


def test_get_audio_chunk_returns_audio(subgen_module):
    data = np.array([0, 1, 2, 3], dtype=np.int16).tobytes()
    file_obj = AsyncFile(data)
    result = asyncio.run(subgen_module.get_audio_chunk(file_obj, offset=1, length=1, sample_rate=2))
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 2


def test_detect_language_task_queues(monkeypatch, subgen_module):
    queued = []
    subgen_module.model = subgen_module.stable_whisper.DummyModel(language="English")
    monkeypatch.setattr(subgen_module, "start_model", lambda: None)
    monkeypatch.setattr(subgen_module, "delete_model", lambda: None)
    monkeypatch.setattr(subgen_module, "extract_audio_segment_to_memory", lambda *args, **kwargs: io.BytesIO(b"data"))
    monkeypatch.setattr(subgen_module.task_queue, "task_done", lambda: None)
    monkeypatch.setattr(subgen_module.task_queue, "put", lambda item: queued.append(item))

    subgen_module.detect_language_task("/media/file.mkv")
    assert queued
    assert queued[-1]["path"] == "/media/file.mkv"
    assert queued[-1]["force_language"] == LanguageCode.ENGLISH


def test_extract_audio_segment_to_memory_errors(monkeypatch, subgen_module):
    assert subgen_module.extract_audio_segment_to_memory(123, 0, 1) is None

    class DummyInput:
        def output(self, *args, **kwargs):
            return self

        def run(self, *args, **kwargs):
            return b"data", b""

    monkeypatch.setattr(subgen_module.ffmpeg, "input", lambda *args, **kwargs: DummyInput())
    result = subgen_module.extract_audio_segment_to_memory("/media/file.mkv", 0, 1)
    assert isinstance(result, io.BytesIO)


def test_start_model_and_delete_model(monkeypatch, subgen_module):
    dummy_model = subgen_module.stable_whisper.DummyModel(language="English")
    monkeypatch.setattr(subgen_module.stable_whisper, "load_faster_whisper", lambda *args, **kwargs: dummy_model)
    monkeypatch.setattr(subgen_module.task_queue, "is_idle", lambda: True)
    monkeypatch.setattr(subgen_module.gc, "collect", lambda: None)
    monkeypatch.setattr(subgen_module.ctypes.util, "find_library", lambda name: "c")
    monkeypatch.setattr(subgen_module.ctypes, "CDLL", lambda name: types.SimpleNamespace(malloc_trim=lambda x: None))
    subgen_module.transcribe_device = "cpu"
    subgen_module.model = None

    subgen_module.start_model()
    assert subgen_module.model is dummy_model

    subgen_module.delete_model()
    assert subgen_module.model is None


def test_isAudioFileExtension_and_extensions(subgen_module):
    assert subgen_module.isAudioFileExtension(".mp3")
    assert subgen_module.has_audio_extension("track.m4a")
    assert subgen_module.has_video_extension("movie.mkv")
    assert not subgen_module.has_video_extension("movie.txt")


def test_write_lrc(tmp_path, subgen_module):
    result = subgen_module.stable_whisper.DummyResult(language="English")
    result.segments = [subgen_module.stable_whisper.Segment(65.12, 70.0, "Hello\nWorld", [], 0)]
    path = tmp_path / "track.lrc"
    subgen_module.write_lrc(result, str(path))
    content = path.read_text()
    assert "[01:05.12]HelloWorld" in content


def test_gen_subtitles_audio_path(monkeypatch, subgen_module):
    calls = []
    subgen_module.append = False
    subgen_module.lrc_for_audio_files = True
    monkeypatch.setattr(subgen_module, "start_model", lambda: None)
    subgen_module.model = subgen_module.stable_whisper.DummyModel(language="English")
    monkeypatch.setattr(subgen_module, "handle_multiple_audio_tracks", lambda *args, **kwargs: None)
    monkeypatch.setattr(subgen_module, "isAudioFileExtension", lambda ext: True)
    monkeypatch.setattr(subgen_module, "write_lrc", lambda result, path: calls.append(path))
    monkeypatch.setattr(subgen_module, "delete_model", lambda: None)

    subgen_module.gen_subtitles("/media/track.mp3", "transcribe", LanguageCode.NONE)
    assert calls == ["/media/track.lrc"]


def test_gen_subtitles_video_path(monkeypatch, subgen_module):
    subgen_module.append = False
    subgen_module.lrc_for_audio_files = False
    monkeypatch.setattr(subgen_module, "start_model", lambda: None)
    model = subgen_module.stable_whisper.DummyModel(language="English")
    subgen_module.model = model
    monkeypatch.setattr(subgen_module, "handle_multiple_audio_tracks", lambda *args, **kwargs: io.BytesIO(b"data"))
    monkeypatch.setattr(subgen_module, "isAudioFileExtension", lambda ext: False)
    monkeypatch.setattr(subgen_module, "name_subtitle", lambda path, lang: "/media/track.en.srt")
    monkeypatch.setattr(subgen_module, "delete_model", lambda: None)

    subgen_module.gen_subtitles("/media/track.mkv", "transcribe", LanguageCode.NONE)
    assert model.transcribe_calls
    assert model.transcribe_calls[0][1]["task"] == "transcribe"


def test_define_subtitle_language_naming(subgen_module):
    subgen_module.namesublang = "custom"
    assert subgen_module.define_subtitle_language_naming(LanguageCode.FRENCH, "ISO_639_1") == "custom"

    subgen_module.namesublang = ""
    subgen_module.transcribe_or_translate = "translate"
    assert subgen_module.define_subtitle_language_naming(LanguageCode.FRENCH, "ISO_639_1") == "fr"


def test_name_subtitle(subgen_module):
    subgen_module.show_in_subname_subgen = True
    subgen_module.show_in_subname_model = True
    subgen_module.whisper_model = "tiny"
    subgen_module.subtitle_language_naming_type = "ISO_639_1"
    name = subgen_module.name_subtitle("/media/show.mkv", LanguageCode.ENGLISH)
    assert name.endswith(".subgen.tiny.en.srt")


def test_handle_multiple_audio_tracks(monkeypatch, subgen_module):
    tracks = [
        {"index": 0, "language": LanguageCode.FRENCH, "codec": "aac", "default": True},
        {"index": 1, "language": LanguageCode.ENGLISH, "codec": "aac", "default": False},
    ]
    monkeypatch.setattr(subgen_module, "get_audio_tracks", lambda path: tracks)
    monkeypatch.setattr(subgen_module, "get_audio_track_by_language", lambda tracks, language: tracks[1])
    monkeypatch.setattr(subgen_module, "extract_audio_track_to_memory", lambda *args, **kwargs: io.BytesIO(b"data"))
    result = subgen_module.handle_multiple_audio_tracks("/media/show.mkv", LanguageCode.ENGLISH)
    assert isinstance(result, io.BytesIO)

    monkeypatch.setattr(subgen_module, "get_audio_tracks", lambda path: tracks[:1])
    result = subgen_module.handle_multiple_audio_tracks("/media/show.mkv", LanguageCode.ENGLISH)
    assert result is None


def test_extract_audio_track_to_memory(subgen_module, monkeypatch):
    assert subgen_module.extract_audio_track_to_memory("/media/show.mkv", None) is None

    def raise_error(*args, **kwargs):
        raise subgen_module.ffmpeg.Error("ffmpeg failed")

    monkeypatch.setattr(subgen_module.ffmpeg, "input", raise_error)
    assert subgen_module.extract_audio_track_to_memory("/media/show.mkv", 0) is None


def test_get_audio_track_by_language(subgen_module):
    tracks = [
        {"language": LanguageCode.FRENCH},
        {"language": LanguageCode.ENGLISH},
    ]
    assert subgen_module.get_audio_track_by_language(tracks, LanguageCode.ENGLISH) == tracks[1]
    assert subgen_module.get_audio_track_by_language(tracks, LanguageCode.SPANISH) is None


def test_choose_transcribe_language(monkeypatch, subgen_module):
    forced = subgen_module.choose_transcribe_language("/media/file.mkv", LanguageCode.FRENCH)
    assert forced == LanguageCode.FRENCH

    subgen_module.force_detected_language_to = LanguageCode.ENGLISH
    result = subgen_module.choose_transcribe_language("/media/file.mkv", LanguageCode.NONE)
    assert result == LanguageCode.ENGLISH

    subgen_module.force_detected_language_to = LanguageCode.NONE
    monkeypatch.setattr(subgen_module, "get_audio_tracks", lambda path: [])
    monkeypatch.setattr(subgen_module, "find_language_audio_track", lambda tracks, languages: LanguageCode.SPANISH)
    subgen_module.preferred_audio_languages = [LanguageCode.SPANISH]
    result = subgen_module.choose_transcribe_language("/media/file.mkv", LanguageCode.NONE)
    assert result == LanguageCode.SPANISH


def test_get_audio_tracks(monkeypatch, subgen_module):
    def probe_stub(*args, **kwargs):
        return {
            "streams": [
                {
                    "index": 0,
                    "codec_name": "aac",
                    "channels": 2,
                    "tags": {"language": "eng", "title": "English"},
                    "disposition": {"default": 1, "forced": 0, "original": 1},
                },
                {
                    "index": 1,
                    "codec_name": "aac",
                    "channels": 2,
                    "tags": {"language": "fra", "title": "French commentary"},
                    "disposition": {"default": 0, "forced": 0, "original": 0},
                },
            ]
        }

    monkeypatch.setattr(subgen_module.ffmpeg, "probe", probe_stub)
    tracks = subgen_module.get_audio_tracks("/media/file.mkv")
    assert tracks[0]["language"] == LanguageCode.ENGLISH
    assert tracks[1]["commentary"] is True


def test_find_language_audio_track(subgen_module):
    tracks = [{"language": LanguageCode.ENGLISH}]
    result = subgen_module.find_language_audio_track(tracks, [LanguageCode.SPANISH, LanguageCode.ENGLISH])
    assert result == LanguageCode.ENGLISH


def test_find_default_audio_track_language(subgen_module):
    tracks = [
        {"language": LanguageCode.ENGLISH, "default": False},
        {"language": LanguageCode.FRENCH, "default": True},
    ]
    assert subgen_module.find_default_audio_track_language(tracks) == LanguageCode.FRENCH


def test_gen_subtitles_queue_skips(monkeypatch, subgen_module):
    monkeypatch.setattr(subgen_module, "has_audio", lambda path: False)
    monkeypatch.setattr(subgen_module.task_queue, "put", lambda item: None)
    subgen_module.gen_subtitles_queue("/media/file.mkv", "transcribe", LanguageCode.NONE)


def test_gen_subtitles_queue_detect_language(monkeypatch, subgen_module):
    queued = []
    monkeypatch.setattr(subgen_module, "has_audio", lambda path: True)
    monkeypatch.setattr(subgen_module, "choose_transcribe_language", lambda path, language: LanguageCode.NONE)
    monkeypatch.setattr(subgen_module, "should_skip_file", lambda *args, **kwargs: False)
    subgen_module.should_whiser_detect_audio_language = True
    monkeypatch.setattr(subgen_module.task_queue, "put", lambda item: queued.append(item))

    subgen_module.gen_subtitles_queue("/media/file.mkv", "transcribe", LanguageCode.NONE)
    assert queued[0]["type"] == "detect_language"


def test_gen_subtitles_queue_normal(monkeypatch, subgen_module):
    queued = []
    monkeypatch.setattr(subgen_module, "has_audio", lambda path: True)
    monkeypatch.setattr(subgen_module, "choose_transcribe_language", lambda path, language: LanguageCode.ENGLISH)
    monkeypatch.setattr(subgen_module, "should_skip_file", lambda *args, **kwargs: False)
    subgen_module.should_whiser_detect_audio_language = False
    monkeypatch.setattr(subgen_module.task_queue, "put", lambda item: queued.append(item))

    subgen_module.gen_subtitles_queue("/media/file.mkv", "transcribe", LanguageCode.NONE)
    assert queued[0]["force_language"] == LanguageCode.ENGLISH


def test_should_skip_file_lrc_exists(tmp_path, subgen_module):
    _configure_no_skip(subgen_module)
    subgen_module.lrc_for_audio_files = True
    subgen_module.isAudioFileExtension = lambda ext: True
    lrc_path = tmp_path / "track.lrc"
    lrc_path.write_text("test")
    result = subgen_module.should_skip_file(str(tmp_path / "track.mp3"), LanguageCode.ENGLISH)
    assert result is True


def test_should_skip_file_unknown_language(subgen_module):
    _configure_no_skip(subgen_module)
    subgen_module.skip_unknown_language = True
    assert subgen_module.should_skip_file("/media/file.mkv", LanguageCode.NONE) is True


def test_should_skip_file_existing_subtitles(monkeypatch, subgen_module):
    _configure_no_skip(subgen_module)
    subgen_module.skip_if_to_transcribe_sub_already_exist = True
    monkeypatch.setattr(subgen_module, "has_subtitle_language", lambda *args, **kwargs: True)
    assert subgen_module.should_skip_file("/media/file.mkv", LanguageCode.ENGLISH) is True


def test_should_skip_file_internal_subtitles(monkeypatch, subgen_module):
    _configure_no_skip(subgen_module)
    subgen_module.skipifinternalsublang = LanguageCode.ENGLISH
    monkeypatch.setattr(subgen_module, "has_subtitle_language_in_file", lambda *args, **kwargs: True)
    assert subgen_module.should_skip_file("/media/file.mkv", LanguageCode.FRENCH) is True


def test_should_skip_file_external_subtitles(monkeypatch, subgen_module):
    _configure_no_skip(subgen_module)
    subgen_module.skipifexternalsub = True
    subgen_module.namesublang = "eng"
    monkeypatch.setattr(subgen_module, "has_subtitle_of_language_in_folder", lambda *args, **kwargs: True)
    assert subgen_module.should_skip_file("/media/file.mkv", LanguageCode.FRENCH) is True


def test_should_skip_file_skip_lang_codes(monkeypatch, subgen_module):
    _configure_no_skip(subgen_module)
    subgen_module.skip_lang_codes_list = [LanguageCode.ENGLISH]
    monkeypatch.setattr(subgen_module, "get_subtitle_languages", lambda *args, **kwargs: [LanguageCode.ENGLISH])
    assert subgen_module.should_skip_file("/media/file.mkv", LanguageCode.FRENCH) is True


def test_should_skip_file_preferred_audio(monkeypatch, subgen_module):
    _configure_no_skip(subgen_module)
    subgen_module.limit_to_preferred_audio_languages = True
    subgen_module.preferred_audio_languages = [LanguageCode.ENGLISH]
    monkeypatch.setattr(subgen_module, "get_audio_languages", lambda *args, **kwargs: [LanguageCode.FRENCH])
    assert subgen_module.should_skip_file("/media/file.mkv", LanguageCode.FRENCH) is True


def test_should_skip_file_skip_audio(monkeypatch, subgen_module):
    _configure_no_skip(subgen_module)
    subgen_module.skip_if_audio_track_is_in_list = [LanguageCode.ENGLISH]
    monkeypatch.setattr(subgen_module, "get_audio_languages", lambda *args, **kwargs: [LanguageCode.ENGLISH])
    assert subgen_module.should_skip_file("/media/file.mkv", LanguageCode.FRENCH) is True


def test_should_skip_file_none(monkeypatch, subgen_module):
    _configure_no_skip(subgen_module)
    monkeypatch.setattr(subgen_module, "get_audio_languages", lambda *args, **kwargs: [])
    assert subgen_module.should_skip_file("/media/file.mkv", LanguageCode.FRENCH) is False


def test_get_subtitle_languages(monkeypatch, subgen_module):
    subtitles = [
        DummyStream(metadata={"language": "eng"}, stream_type="subtitle"),
        DummyStream(metadata={}, stream_type="subtitle"),
    ]
    streams = DummyStreams(subtitles=subtitles)
    monkeypatch.setattr(subgen_module.av, "open", lambda *args, **kwargs: DummyContainer(streams))
    languages = subgen_module.get_subtitle_languages("/media/file.mkv")
    assert languages == [LanguageCode.ENGLISH, LanguageCode.NONE]


def test_has_subtitle_language_in_file(monkeypatch, subgen_module):
    subtitles = [DummyStream(metadata={"language": "eng"}, stream_type="subtitle")]
    streams = DummyStreams(streams=subtitles)
    monkeypatch.setattr(subgen_module.av, "open", lambda *args, **kwargs: DummyContainer(streams))

    subgen_module.skip_if_language_is_not_set_but_subtitles_exist = True
    assert subgen_module.has_subtitle_language_in_file("/media/file.mkv", LanguageCode.NONE) is True

    subgen_module.skip_if_language_is_not_set_but_subtitles_exist = False
    subgen_module.only_skip_if_subgen_subtitle = True
    assert subgen_module.has_subtitle_language_in_file("/media/file.mkv", LanguageCode.NONE) is False

    subgen_module.only_skip_if_subgen_subtitle = False
    assert subgen_module.has_subtitle_language_in_file("/media/file.mkv", LanguageCode.ENGLISH) is True


def test_has_subtitle_of_language_in_folder(tmp_path, subgen_module):
    video_path = tmp_path / "movie.mkv"
    video_path.write_text("video")

    (tmp_path / "movie.subgen.srt").write_text("sub")
    assert subgen_module.has_subtitle_of_language_in_folder(
        str(video_path),
        LanguageCode.NONE,
        recursion=False,
        only_skip_if_subgen_subtitle=True,
    ) is False

    (tmp_path / "movie.en.srt").write_text("sub")
    assert subgen_module.has_subtitle_of_language_in_folder(
        str(video_path),
        LanguageCode.ENGLISH,
        recursion=False,
        only_skip_if_subgen_subtitle=True,
    ) is False

    (tmp_path / "movie.subgen.en.srt").write_text("sub")
    assert subgen_module.has_subtitle_of_language_in_folder(
        str(video_path),
        LanguageCode.ENGLISH,
        recursion=False,
        only_skip_if_subgen_subtitle=True,
    ) is True


def test_is_valid_subtitle_language(subgen_module):
    assert subgen_module.is_valid_subtitle_language(["eng"], LanguageCode.ENGLISH)
    assert not subgen_module.is_valid_subtitle_language(["fra"], LanguageCode.ENGLISH)


def test_get_next_plex_episode(monkeypatch, subgen_module):
    metadata_xml = b"""
        <MediaContainer>
            <Video grandparentRatingKey="10" parentRatingKey="20" ratingKey="1" index="1" parentIndex="1" />
        </MediaContainer>
    """
    seasons_xml = b"""
        <MediaContainer>
            <Directory type="season" index="1" ratingKey="20" />
        </MediaContainer>
    """
    episodes_xml = b"""
        <MediaContainer>
            <Video ratingKey="1" index="1" parentIndex="1" />
            <Video ratingKey="2" index="2" parentIndex="1" />
        </MediaContainer>
    """

    responses = [
        DummyResponse(200, metadata_xml),
        DummyResponse(200, seasons_xml),
        DummyResponse(200, episodes_xml),
    ]

    def get_stub(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(subgen_module.requests, "get", get_stub)
    result = subgen_module.get_next_plex_episode("1", stay_in_season=True)
    assert result == "2"


def test_get_plex_file_name_and_refresh(monkeypatch, subgen_module):
    xml = b"""<MediaContainer><Part file="/media/show.mkv" /></MediaContainer>"""
    monkeypatch.setattr(subgen_module.requests, "get", lambda *args, **kwargs: DummyResponse(200, xml))
    file_name = subgen_module.get_plex_file_name("1", "http://plex", "token")
    assert file_name == "/media/show.mkv"

    monkeypatch.setattr(subgen_module.requests, "put", lambda *args, **kwargs: DummyResponse(200, b""))
    subgen_module.refresh_plex_metadata("1", "http://plex", "token")


def test_refresh_jellyfin_and_get_file_name(monkeypatch, subgen_module):
    users = [{"Id": "admin", "Policy": {"IsAdministrator": True}}]
    users_payload = json.dumps(users).encode("utf-8")
    file_payload = b"{\"Path\": \"/media/jellyfin.mkv\"}"

    def get_stub(url, *args, **kwargs):
        if url.endswith("/Users"):
            return DummyResponse(200, users_payload)
        return DummyResponse(200, file_payload)

    monkeypatch.setattr(subgen_module.requests, "get", get_stub)
    monkeypatch.setattr(subgen_module.requests, "post", lambda *args, **kwargs: DummyResponse(204, b""))
    subgen_module.refresh_jellyfin_metadata("1", "http://jellyfin", "token")

    file_name = subgen_module.get_jellyfin_file_name("1", "http://jellyfin", "token")
    assert file_name == "/media/jellyfin.mkv"


def test_get_jellyfin_admin(subgen_module):
    users = [
        {"Id": "user", "Policy": {"IsAdministrator": False}},
        {"Id": "admin", "Policy": {"IsAdministrator": True}},
    ]
    assert subgen_module.get_jellyfin_admin(users) == "admin"


def test_has_audio(monkeypatch, subgen_module, tmp_path):
    file_path = tmp_path / "audio.mkv"
    file_path.write_text("data")
    subgen_module.is_valid_path = lambda path: True
    monkeypatch.setattr(subgen_module, "has_video_extension", lambda path: True)
    monkeypatch.setattr(subgen_module, "has_audio_extension", lambda path: False)
    streams = DummyStreams(streams=[DummyStream(stream_type="audio")])
    monkeypatch.setattr(subgen_module.av, "open", lambda *args, **kwargs: DummyContainer(streams))
    assert subgen_module.has_audio(str(file_path)) is True


def test_is_valid_path(tmp_path, subgen_module):
    file_path = tmp_path / "file.txt"
    file_path.write_text("data")
    assert subgen_module.is_valid_path(str(file_path)) is True
    assert subgen_module.is_valid_path(str(tmp_path)) is False
    assert subgen_module.is_valid_path(str(tmp_path / "missing.txt")) is False


def test_path_mapping(subgen_module):
    subgen_module.use_path_mapping = True
    subgen_module.path_mapping_from = "/tv"
    subgen_module.path_mapping_to = "/media/tv"
    assert subgen_module.path_mapping("/tv/show.mkv") == "/media/tv/show.mkv"

    subgen_module.use_path_mapping = False
    assert subgen_module.path_mapping("/tv/show.mkv") == "/tv/show.mkv"


def test_is_file_stable(tmp_path, subgen_module):
    file_path = tmp_path / "stable.txt"
    file_path.write_text("data")
    assert subgen_module.is_file_stable(str(file_path), wait_time=0, check_intervals=2) is True
    assert subgen_module.is_file_stable(str(tmp_path / "missing.txt"), wait_time=0, check_intervals=1) is False


def test_transcribe_existing(monkeypatch, subgen_module, tmp_path):
    file_path = tmp_path / "audio.mkv"
    file_path.write_text("data")
    calls = []
    monkeypatch.setattr(subgen_module, "has_audio", lambda path: True)
    monkeypatch.setattr(subgen_module, "gen_subtitles_queue", lambda path, mode, lang=None: calls.append((path, mode, lang)))

    subgen_module.transcribe_existing(str(file_path), LanguageCode.ENGLISH)
    assert calls

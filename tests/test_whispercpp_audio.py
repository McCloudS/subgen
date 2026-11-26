"""
/asr audio handling for the whisper.cpp backend.

Issue #361 follow-up: Bazarr's Whisper provider sends raw headerless 16 kHz mono s16le
PCM with encode=false, which whisper-cli can't read ("error: 1 input file(s) failed").
"""
import io
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

import subgen


# ---------------------------------------------------------------------------
# /asr audio for whisper.cpp (issue #361 follow-up: Bazarr sends raw PCM)
# ---------------------------------------------------------------------------
class TestAudioForWhisperCpp:
    def test_raw_pcm_gets_wav_header(self):
        pcm = b"\x01\x00\x02\x00" * 1000
        out = subgen._audio_for_whispercpp(pcm, encode=False)
        assert out[:4] == b"RIFF" and out[8:12] == b"WAVE"
        with wave.open(io.BytesIO(out)) as wf:
            assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, 16000)
            assert wf.readframes(wf.getnframes()) == pcm

    @pytest.mark.parametrize("head", [b"RIFF\x00\x00\x00\x00WAVEfmt ", b"fLaC\x00\x00\x00\x22", b"OggS\x00\x02", b"ID3\x04\x00"])
    def test_formats_whisper_cli_reads_pass_through(self, head):
        data = head + b"\x00" * 64
        assert subgen._audio_for_whispercpp(data, encode=True) == data

    def test_asr_worker_wraps_pcm_for_whispercpp(self, monkeypatch):
        monkeypatch.setattr(subgen, "transcribe_backend", "whispercpp")
        monkeypatch.setattr(subgen, "start_model", lambda: None)
        seen = {}

        def fake_transcribe(audio, task, language, name):
            seen["audio"] = audio
            return subgen.TranscriptionResult(segments=[{"start": 0, "end": 1, "text": "hi"}], language="en")

        monkeypatch.setattr(subgen, "_transcribe_whispercpp", fake_transcribe)
        subgen.asr_task_worker({"task": "transcribe", "language": "en", "audio_content": b"\x00\x00" * 800,
                                "encode": False, "path": "x"})
        assert seen["audio"][:4] == b"RIFF"


# ---------------------------------------------------------------------------
# Files from webhooks, /batch and folder scans (whisper-cli can't read MKV/MP4)
# ---------------------------------------------------------------------------
class TestFileAudioForWhisperCpp:
    def _run_gen_subtitles(self, monkeypatch, tmp_path, extracted, converted):
        video = tmp_path / "show.mkv"
        video.write_bytes(b"\x1aE\xdf\xa3 not audio")
        seen = {}
        monkeypatch.setattr(subgen, "transcribe_backend", "whispercpp")
        monkeypatch.setattr(subgen, "start_model", lambda: None)
        monkeypatch.setattr(subgen, "delete_model", lambda: None)
        monkeypatch.setattr(subgen, "handle_multiple_audio_tracks", lambda *a, **k: extracted)
        monkeypatch.setattr(subgen, "_media_file_to_wav", lambda path: seen.setdefault("converted_path", path) and converted)
        monkeypatch.setattr(subgen, "send_completion_webhook", lambda *a, **k: None)

        def fake_transcribe(audio, *a, **k):
            seen["audio"] = audio
            return subgen.TranscriptionResult(segments=[{"start": 0.0, "end": 1.0, "text": "Hi."}], language="en")
        monkeypatch.setattr(subgen, "_transcribe_whispercpp", fake_transcribe)
        subgen.gen_subtitles(str(video), "transcribe", subgen.LanguageCode.from_string("en"))
        return video, seen

    def test_single_track_video_is_converted_not_read_whole(self, monkeypatch, tmp_path):
        video, seen = self._run_gen_subtitles(monkeypatch, tmp_path, extracted=None, converted=b"RIFF-converted")
        assert seen["converted_path"] == str(video)
        assert seen["audio"] == b"RIFF-converted"
        assert list(tmp_path.glob("show*.srt")), "subtitle written"

    def test_chosen_track_bytes_used_as_is(self, monkeypatch, tmp_path):
        _, seen = self._run_gen_subtitles(monkeypatch, tmp_path, extracted=b"RIFF-track2", converted=b"unused")
        assert seen["audio"] == b"RIFF-track2"
        assert "converted_path" not in seen

    def test_extraction_failure_writes_no_subtitle(self, monkeypatch, tmp_path):
        _, seen = self._run_gen_subtitles(monkeypatch, tmp_path, extracted=None, converted=None)
        assert "audio" not in seen
        assert not list(tmp_path.glob("*.srt"))

    def test_asr_video_upload_converted_from_a_file(self, monkeypatch):
        paths = []
        monkeypatch.setattr(subgen, "_media_file_to_wav", lambda path: paths.append(path) or b"RIFFwav")
        assert subgen._audio_for_whispercpp(b"\x00\x00\x00\x18ftypmp42 moov later", encode=True) == b"RIFFwav"
        assert paths and os.path.basename(paths[0]).startswith("subgen_upload_")


def test_detect_language_clip_is_downmixed_to_mono(monkeypatch):
    # faster-whisper reads these samples directly; stereo would be read as garbled mono
    # (an English MKV upload was detected as Spanish).
    seen = {}

    class FakeStream:
        def __init__(self, kwargs):
            seen.update(kwargs)

        def run(self, **k):
            return b"RIFF", b""

    class FakeInput:
        def output(self, *a, **kwargs):
            return FakeStream(kwargs)

    def fake_input(src, **k):
        seen["src"] = src
        return FakeInput()
    monkeypatch.setattr(subgen.ffmpeg, "input", fake_input)
    subgen.extract_audio_segment_from_content(b"mkv", 0, 30)
    assert seen.get("ac") == 1 and seen.get("ar") == 16000
    # from a temp file, not stdin (a pipe returned ~25 s of a 30 s request)
    assert seen["src"] != "pipe:0" and os.path.basename(seen["src"]).startswith("subgen_detect_")

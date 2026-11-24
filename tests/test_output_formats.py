"""
/asr and OpenAI output formats.

/asr used to ignore its `output` parameter and always return SRT (the endpoint stored the
format as 'output' but the worker read 'output_format'). The worker now hands back the
transcription and each request renders the format it asked for, so duplicate requests for
the same audio in different formats also get the right answer.
"""
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient

import subgen


@pytest.fixture(scope="module")
def client():
    return TestClient(subgen.app)

RESULT = subgen.TranscriptionResult(
    segments=[
        {"start": 0.0, "end": 1.5, "text": "Hello there,\nGeneral Kenobi."},
        {"start": 2.0, "end": 3.25, "text": "You are a bold one."},
    ],
    language="en",
)


class TestFormatTranscription:
    def test_srt(self):
        out = subgen.format_transcription(RESULT, "srt")
        assert out.startswith("1\n00:00:00,000 --> 00:00:01,500\nHello there,\nGeneral Kenobi.\n")

    def test_vtt(self):
        out = subgen.format_transcription(RESULT, "vtt")
        assert out.startswith("WEBVTT\n")
        assert "00:00:02.000 --> 00:00:03.250\nYou are a bold one." in out

    def test_txt_is_one_line_per_cue(self):
        assert subgen.format_transcription(RESULT, "txt") == "Hello there, General Kenobi.\nYou are a bold one.\n"

    def test_tsv_in_milliseconds(self):
        lines = subgen.format_transcription(RESULT, "tsv").splitlines()
        assert lines[0] == "start\tend\ttext"
        assert lines[2] == "2000\t3250\tYou are a bold one."

    def test_asr_json_has_segments(self):
        body = json.loads(subgen.format_transcription(RESULT, "asr_json"))
        assert body["language"] == "en"
        assert body["segments"][1] == {"id": 1, "start": 2.0, "end": 3.25, "text": "You are a bold one."}
        assert body["text"].startswith("Hello there, General Kenobi.")

    def test_openai_json_and_text(self):
        assert json.loads(subgen.format_transcription(RESULT, "json")) == {
            "text": "Hello there, General Kenobi. You are a bold one."}
        assert subgen.format_transcription(RESULT, "text") == "Hello there, General Kenobi. You are a bold one."

    def test_openai_verbose_json(self):
        body = json.loads(subgen.format_transcription(RESULT, "verbose_json", "translate"))
        assert body["task"] == "translate" and body["duration"] == 3.25 and len(body["segments"]) == 2


def _finish_immediately(item, *a, **k):
    """Stand-in for task_queue.put: complete the task at once with RESULT."""
    item["result_container"].set_result(RESULT)
    return True


class TestAsrEndpointFormats:
    @pytest.mark.parametrize("output,check", [
        ("srt", lambda t: t.startswith("1\n00:00:00,000 -->")),
        ("vtt", lambda t: t.startswith("WEBVTT")),
        ("txt", lambda t: t.startswith("Hello there, General Kenobi.\n")),
        ("tsv", lambda t: t.startswith("start\tend\ttext")),
        ("json", lambda t: json.loads(t)["segments"][0]["start"] == 0.0),
    ])
    def test_asr_returns_requested_format(self, client, output, check):
        with patch.object(subgen.task_queue, "put", side_effect=_finish_immediately):
            resp = client.post(f"/asr?output={output}", files={"audio_file": ("a.wav", b"RIFF0000", "audio/wav")})
        assert resp.status_code == 200
        assert check(resp.text), resp.text

    def test_unknown_output_falls_back_to_srt(self, client):
        with patch.object(subgen.task_queue, "put", side_effect=_finish_immediately):
            resp = client.post("/asr?output=docx", files={"audio_file": ("a.wav", b"RIFF0000", "audio/wav")})
        assert resp.text.startswith("1\n00:00:00,000 -->")

    def test_unknown_task_is_transcribed(self, client):
        seen = {}

        def put(item, *a, **k):
            seen["task"] = item["task"]
            return _finish_immediately(item)
        with patch.object(subgen.task_queue, "put", side_effect=put):
            client.post("/asr?task=bogus", files={"audio_file": ("a.wav", b"RIFF0000", "audio/wav")})
        assert seen["task"] == "transcribe"

    @pytest.mark.parametrize("fmt", ["json", "text", "srt", "vtt", "verbose_json"])
    def test_openai_transcriptions_formats(self, client, fmt):
        with patch.object(subgen.task_queue, "put", side_effect=_finish_immediately):
            resp = client.post("/v1/audio/transcriptions", data={"response_format": fmt},
                               files={"file": ("a.wav", b"RIFF0000", "audio/wav")})
        assert resp.status_code == 200
        assert resp.text == subgen.format_transcription(RESULT, fmt, "transcribe")


class TestWorkerHandsBackResult:
    def test_asr_worker_sets_transcription_object(self, monkeypatch):
        monkeypatch.setattr(subgen, "transcribe_backend", "whispercpp")
        monkeypatch.setattr(subgen, "start_model", lambda: None)
        monkeypatch.setattr(subgen, "delete_model", lambda: None)
        monkeypatch.setattr(subgen, "_transcribe_whispercpp", lambda *a: subgen.TranscriptionResult(
            segments=list(RESULT.segments), language="en"))
        box = subgen.TaskResult()
        subgen.asr_task_worker({"path": "asr-x", "task": "transcribe", "language": "en", "audio_content": b"RIFF....WAVE",
                                "encode": True, "result_container": box})
        assert box.error is None
        assert isinstance(box.result, subgen.TranscriptionResult)


class TestFasterWhisperDetectLanguage:
    def test_detect_language_with_three_value_return(self, client, monkeypatch):
        from unittest.mock import MagicMock
        fake_model = MagicMock()
        fake_model.detect_language.return_value = ("de", 0.97, [("de", 0.97), ("en", 0.02)])
        monkeypatch.setattr(subgen, "transcribe_backend", "faster-whisper")
        monkeypatch.setattr(subgen, "force_detected_language_to", None)
        monkeypatch.setattr(subgen, "start_model", lambda: None)
        monkeypatch.setattr(subgen, "model", fake_model)
        monkeypatch.setattr(subgen, "extract_audio_segment_from_content", lambda *a: b"RIFFxxxxWAVE")
        monkeypatch.setattr(subgen, "wav_bytes_to_numpy", lambda b: [0.0])
        resp = client.post("/detect-language", files={"audio_file": ("a.wav", b"RIFF0000", "audio/wav")})
        assert resp.json()["language_code"] == "de", resp.json()

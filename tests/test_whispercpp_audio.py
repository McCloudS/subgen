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

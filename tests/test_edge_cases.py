"""Round 3 edge cases: hallucinated cues over silence, commentary tracks, files with no speech."""
import io
import math
import os
import struct
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import subgen
from language_code import LanguageCode


def wav(seconds_spec, rate=16000):
    """seconds_spec: list of (seconds, amplitude) runs of a 440 Hz tone (amplitude 0 = silence)."""
    frames = bytearray()
    for secs, amp in seconds_spec:
        for i in range(int(secs * rate)):
            frames += struct.pack("<h", int(amp * 32767 * math.sin(2 * math.pi * 440 * i / rate)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


class TestSilentCues:
    def test_cues_over_digital_silence_are_dropped(self):
        audio = wav([(5, 0.0), (5, 0.2), (5, 0.0)])
        segs = [{"start": 0.5, "end": 4.5, "text": "Thank you."},
                {"start": 5.5, "end": 9.5, "text": "Real dialogue."},
                {"start": 10.5, "end": 14.5, "text": "you"}]
        assert [s["text"] for s in subgen._drop_silent_segments(segs, audio)] == ["Real dialogue."]

    def test_quiet_but_audible_speech_is_kept(self):
        audio = wav([(4, 0.01)])  # about -43 dBFS: quiet dialogue, well above the -55 dBFS cut
        segs = [{"start": 0.0, "end": 4.0, "text": "whispering"}]
        assert subgen._drop_silent_segments(segs, audio) == segs

    def test_non_wav_input_is_left_alone(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "hi"}]
        assert subgen._drop_silent_segments(segs, b"ID3\x04mp3data") == segs


class TestCommentaryTracks:
    TRACKS = [
        {"index": 0, "language": LanguageCode.ENGLISH, "commentary": True, "codec": "ac3", "default": True},
        {"index": 1, "language": LanguageCode.ENGLISH, "commentary": False, "codec": "eac3", "default": False},
    ]

    def test_language_match_prefers_main_track(self):
        assert subgen.get_audio_track_by_language(self.TRACKS, LanguageCode.ENGLISH)["index"] == 1

    def test_commentary_used_when_it_is_the_only_match(self):
        assert subgen.get_audio_track_by_language(self.TRACKS[:1], LanguageCode.ENGLISH)["index"] == 0

    def test_fallback_skips_commentary(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(subgen, "extract_audio_track_to_memory", lambda path, idx: seen.setdefault("idx", idx) is not None and b"RIFF")
        subgen.handle_multiple_audio_tracks("/x.mkv", LanguageCode.GERMAN, audio_tracks=self.TRACKS)
        assert seen["idx"] == 1


def test_no_speech_writes_no_subtitle(monkeypatch, tmp_path):
    video = tmp_path / "silent.mkv"
    video.write_bytes(b"x")
    monkeypatch.setattr(subgen, "transcribe_backend", "whispercpp")
    monkeypatch.setattr(subgen, "start_model", lambda: None)
    monkeypatch.setattr(subgen, "delete_model", lambda: None)
    monkeypatch.setattr(subgen, "handle_multiple_audio_tracks", lambda *a, **k: None)
    monkeypatch.setattr(subgen, "_media_file_to_wav", lambda p: b"RIFF")
    monkeypatch.setattr(subgen, "_transcribe_whispercpp", lambda *a, **k: subgen.TranscriptionResult(segments=[], language="en"))
    hooks = []
    monkeypatch.setattr(subgen, "send_completion_webhook", lambda *a, **k: hooks.append(a))
    subgen.gen_subtitles(str(video), "transcribe", LanguageCode.ENGLISH)
    assert not list(tmp_path.glob("*.srt")) and not hooks

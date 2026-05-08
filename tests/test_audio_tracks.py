"""
Regression tests for handle_multiple_audio_tracks().

Fix 1 addressed: UnboundLocalError when language=None and len(audio_tracks) > 1.
Before the fix, `audio_track` was never initialized when `language is None`,
causing `if audio_track is None:` to raise UnboundLocalError.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch
import subgen
from subgen import handle_multiple_audio_tracks
from language_code import LanguageCode

FAKE_TRACK_ENG = {"index": 0, "codec": "aac", "language": LanguageCode.ENGLISH, "default": True}
FAKE_TRACK_FRA = {"index": 1, "codec": "ac3", "language": LanguageCode.FRENCH, "default": False}
FAKE_BYTES = b"fake audio data"


class TestSingleAudioTrack:
    def test_single_track_returns_none(self):
        """Single-track file: no extraction needed, return None."""
        with patch.object(subgen, "get_audio_tracks", return_value=[FAKE_TRACK_ENG]):
            result = handle_multiple_audio_tracks("/fake/movie.mkv")
        assert result is None


class TestMultipleAudioTracks:
    def test_language_none_uses_first_track_no_error(self):
        """
        Regression: language=None must NOT raise UnboundLocalError.
        Should fall back to the first audio track.
        """
        with (
            patch.object(subgen, "get_audio_tracks", return_value=[FAKE_TRACK_ENG, FAKE_TRACK_FRA]),
            patch.object(subgen, "extract_audio_track_to_memory", return_value=FAKE_BYTES) as mock_extract,
        ):
            result = handle_multiple_audio_tracks("/fake/movie.mkv", language=None)

        # Must not raise; must return bytes from the first track (index 0)
        assert result == FAKE_BYTES
        mock_extract.assert_called_once_with("/fake/movie.mkv", FAKE_TRACK_ENG["index"])

    def test_language_match_selects_correct_track(self):
        """Matching language selects the right track."""
        with (
            patch.object(subgen, "get_audio_tracks", return_value=[FAKE_TRACK_ENG, FAKE_TRACK_FRA]),
            patch.object(subgen, "extract_audio_track_to_memory", return_value=FAKE_BYTES) as mock_extract,
        ):
            result = handle_multiple_audio_tracks("/fake/movie.mkv", language=LanguageCode.FRENCH)

        assert result == FAKE_BYTES
        mock_extract.assert_called_once_with("/fake/movie.mkv", FAKE_TRACK_FRA["index"])

    def test_no_language_match_falls_back_to_first_track(self):
        """When no track matches the requested language, fall back to first track."""
        with (
            patch.object(subgen, "get_audio_tracks", return_value=[FAKE_TRACK_ENG, FAKE_TRACK_FRA]),
            patch.object(subgen, "extract_audio_track_to_memory", return_value=FAKE_BYTES) as mock_extract,
        ):
            result = handle_multiple_audio_tracks("/fake/movie.mkv", language=LanguageCode.GERMAN)

        assert result == FAKE_BYTES
        mock_extract.assert_called_once_with("/fake/movie.mkv", FAKE_TRACK_ENG["index"])

    def test_extraction_failure_returns_none(self):
        """If extraction returns None (ffmpeg error), propagate None."""
        with (
            patch.object(subgen, "get_audio_tracks", return_value=[FAKE_TRACK_ENG, FAKE_TRACK_FRA]),
            patch.object(subgen, "extract_audio_track_to_memory", return_value=None),
        ):
            result = handle_multiple_audio_tracks("/fake/movie.mkv", language=None)

        assert result is None

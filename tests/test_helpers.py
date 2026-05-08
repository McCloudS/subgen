"""Tests for pure helper functions in subgen.py."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import subgen


class TestConvertToBool:
    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "on", "ON", "1", "y", "yes", "YES"])
    def test_truthy_values(self, value):
        assert subgen.convert_to_bool(value) is True

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "off", "OFF", "0", "n", "no", "NO"])
    def test_falsy_values(self, value):
        assert subgen.convert_to_bool(value) is False

    def test_bool_true_input(self):
        assert subgen.convert_to_bool(True) is True

    def test_bool_false_input(self):
        assert subgen.convert_to_bool(False) is False


class TestGenerateAudioHash:
    def test_deterministic(self):
        data = b"fake audio bytes"
        h1 = subgen.generate_audio_hash(data, "transcribe", "en")
        h2 = subgen.generate_audio_hash(data, "transcribe", "en")
        assert h1 == h2

    def test_different_content_different_hash(self):
        assert (
            subgen.generate_audio_hash(b"audio1")
            != subgen.generate_audio_hash(b"audio2")
        )

    def test_different_task_different_hash(self):
        data = b"same audio"
        assert (
            subgen.generate_audio_hash(data, "transcribe")
            != subgen.generate_audio_hash(data, "translate")
        )

    def test_different_language_different_hash(self):
        data = b"same audio"
        assert (
            subgen.generate_audio_hash(data, language="en")
            != subgen.generate_audio_hash(data, language="fr")
        )

    def test_returns_16_char_string(self):
        h = subgen.generate_audio_hash(b"data")
        assert isinstance(h, str) and len(h) == 16

    def test_no_task_no_language(self):
        h = subgen.generate_audio_hash(b"data")
        assert isinstance(h, str) and len(h) == 16


class TestGetEnvWithFallback:
    def test_new_name_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("NEW_VAR", "new_value")
        monkeypatch.setenv("OLD_VAR", "old_value")
        result = subgen.get_env_with_fallback("NEW_VAR", "OLD_VAR", "default")
        assert result == "new_value"

    def test_falls_back_to_old_name(self, monkeypatch):
        monkeypatch.delenv("NEW_VAR", raising=False)
        monkeypatch.setenv("OLD_VAR", "old_value")
        result = subgen.get_env_with_fallback("NEW_VAR", "OLD_VAR", "default")
        assert result == "old_value"

    def test_uses_default_when_neither_set(self, monkeypatch):
        monkeypatch.delenv("NEW_VAR", raising=False)
        monkeypatch.delenv("OLD_VAR", raising=False)
        result = subgen.get_env_with_fallback("NEW_VAR", "OLD_VAR", "default")
        assert result == "default"

    def test_convert_func_applied(self, monkeypatch):
        monkeypatch.setenv("NEW_VAR", "true")
        result = subgen.get_env_with_fallback("NEW_VAR", "OLD_VAR", False, subgen.convert_to_bool)
        assert result is True

    def test_convert_func_not_applied_to_none_default(self, monkeypatch):
        monkeypatch.delenv("NEW_VAR", raising=False)
        monkeypatch.delenv("OLD_VAR", raising=False)
        result = subgen.get_env_with_fallback("NEW_VAR", "OLD_VAR", None, subgen.convert_to_bool)
        assert result is None


class TestPathMapping:
    def test_disabled_returns_original(self, monkeypatch):
        monkeypatch.setattr(subgen, "use_path_mapping", False)
        monkeypatch.setattr(subgen, "path_mapping_from", "/tv")
        monkeypatch.setattr(subgen, "path_mapping_to", "/Volumes/TV")
        assert subgen.path_mapping("/tv/show.mkv") == "/tv/show.mkv"

    def test_enabled_replaces_prefix(self, monkeypatch):
        monkeypatch.setattr(subgen, "use_path_mapping", True)
        monkeypatch.setattr(subgen, "path_mapping_from", "/tv")
        monkeypatch.setattr(subgen, "path_mapping_to", "/Volumes/TV")
        assert subgen.path_mapping("/tv/show.mkv") == "/Volumes/TV/show.mkv"

    def test_enabled_no_match_returns_original(self, monkeypatch):
        monkeypatch.setattr(subgen, "use_path_mapping", True)
        monkeypatch.setattr(subgen, "path_mapping_from", "/tv")
        monkeypatch.setattr(subgen, "path_mapping_to", "/Volumes/TV")
        assert subgen.path_mapping("/movies/film.mkv") == "/movies/film.mkv"


class TestFileExtensions:
    @pytest.mark.parametrize("fname", ["movie.mkv", "movie.mp4", "movie.avi", "movie.MOV"])
    def test_has_video_extension(self, fname):
        assert subgen.has_video_extension(fname) is True

    @pytest.mark.parametrize("fname", ["subtitle.srt", "text.txt", "audio.mp3"])
    def test_has_no_video_extension(self, fname):
        assert subgen.has_video_extension(fname) is False

    @pytest.mark.parametrize("fname", ["track.mp3", "track.flac", "track.WAV"])
    def test_has_audio_extension(self, fname):
        assert subgen.has_audio_extension(fname) is True

    @pytest.mark.parametrize("fname", ["movie.mkv", "subtitle.srt"])
    def test_has_no_audio_extension(self, fname):
        assert subgen.has_audio_extension(fname) is False

    def test_is_audio_file_extension_case_insensitive(self):
        assert subgen.isAudioFileExtension(".MP3") is True
        assert subgen.isAudioFileExtension(".mp3") is True
        assert subgen.isAudioFileExtension(".Flac") is True

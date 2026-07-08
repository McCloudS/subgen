"""
Tests for should_skip_file().

should_skip_file() reads many module-level globals (skip_unknown_language,
skip_if_target_subtitle_exists, etc.) and calls several functions
that touch the file system (subtitle_exists_in_language, get_subtitle_languages, ...).
Both are patched here with monkeypatch / unittest.mock.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock
import subgen
from subgen import should_skip_file
from language_code import LanguageCode


# ---------------------------------------------------------------------------
# Helper: set all the globals should_skip_file uses to safe, non-skipping defaults.
# Individual tests override whichever ones they need.
# ---------------------------------------------------------------------------
def _patch_defaults(monkeypatch):
    monkeypatch.setattr(subgen, "transcribe_or_translate", "transcribe")
    monkeypatch.setattr(subgen, "lrc_for_audio_files", True)
    monkeypatch.setattr(subgen, "skip_unknown_language", False)
    monkeypatch.setattr(subgen, "skip_if_target_subtitle_exists", False)
    monkeypatch.setattr(subgen, "skip_if_internal_sub_language", LanguageCode.NONE)
    monkeypatch.setattr(subgen, "skip_if_external_sub_exists", False)
    monkeypatch.setattr(subgen, "subtitle_language_name", "")
    monkeypatch.setattr(subgen, "skip_subtitle_languages", [])
    monkeypatch.setattr(subgen, "limit_to_preferred_audio_languages", False)
    monkeypatch.setattr(subgen, "preferred_audio_languages", [LanguageCode.ENGLISH])
    monkeypatch.setattr(subgen, "skip_audio_languages", [])
    monkeypatch.setattr(subgen, "only_match_subgen_subtitles", False)
    monkeypatch.setattr(subgen, "skip_if_no_audio_language_but_subtitles_exist", False)


class TestLrcSkip:
    def test_skip_when_lrc_exists(self, monkeypatch, tmp_path):
        _patch_defaults(monkeypatch)
        audio = tmp_path / "song.mp3"
        audio.touch()
        lrc = tmp_path / "song.lrc"
        lrc.touch()
        # isAudioFileExtension must return True for .mp3
        assert should_skip_file(str(audio), LanguageCode.ENGLISH) is True

    def test_no_skip_when_lrc_missing(self, monkeypatch, tmp_path):
        _patch_defaults(monkeypatch)
        # Mock all file-system checks to return False/[]
        with (
            patch.object(subgen, "get_subtitle_languages", return_value=[]),
            patch.object(subgen, "get_audio_languages", return_value=[]),
        ):
            audio = tmp_path / "song.mp3"
            audio.touch()
            # No .lrc file created
            assert should_skip_file(str(audio), LanguageCode.ENGLISH) is False

    def test_no_skip_lrc_when_lrc_for_audio_disabled(self, monkeypatch, tmp_path):
        _patch_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "lrc_for_audio_files", False)
        with (
            patch.object(subgen, "get_subtitle_languages", return_value=[]),
            patch.object(subgen, "get_audio_languages", return_value=[]),
        ):
            audio = tmp_path / "song.mp3"
            audio.touch()
            lrc = tmp_path / "song.lrc"
            lrc.touch()
            assert should_skip_file(str(audio), LanguageCode.ENGLISH) is False


class TestUnknownLanguageSkip:
    def test_skip_unknown_language_when_enabled(self, monkeypatch, tmp_path):
        _patch_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "skip_unknown_language", True)
        with (
            patch.object(subgen, "get_subtitle_languages", return_value=[]),
            patch.object(subgen, "get_audio_languages", return_value=[]),
        ):
            video = tmp_path / "file.mkv"
            video.touch()
            assert should_skip_file(str(video), LanguageCode.NONE) is True

    def test_no_skip_unknown_language_when_disabled(self, monkeypatch, tmp_path):
        _patch_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "skip_unknown_language", False)
        with (
            patch.object(subgen, "get_subtitle_languages", return_value=[]),
            patch.object(subgen, "get_audio_languages", return_value=[]),
        ):
            video = tmp_path / "file.mkv"
            video.touch()
            assert should_skip_file(str(video), LanguageCode.NONE) is False


class TestSubtitleExistsSkip:
    def test_skip_when_subtitle_exists(self, monkeypatch, tmp_path):
        _patch_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "skip_if_target_subtitle_exists", True)
        with (
            patch.object(subgen, "subtitle_exists_in_language", return_value=True),
            patch.object(subgen, "get_subtitle_languages", return_value=[]),
            patch.object(subgen, "get_audio_languages", return_value=[]),
        ):
            video = tmp_path / "file.mkv"
            video.touch()
            assert should_skip_file(str(video), LanguageCode.ENGLISH) is True

    def test_no_skip_when_subtitle_missing(self, monkeypatch, tmp_path):
        _patch_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "skip_if_target_subtitle_exists", True)
        with (
            patch.object(subgen, "subtitle_exists_in_language", return_value=False),
            patch.object(subgen, "get_subtitle_languages", return_value=[]),
            patch.object(subgen, "get_audio_languages", return_value=[]),
        ):
            video = tmp_path / "file.mkv"
            video.touch()
            assert should_skip_file(str(video), LanguageCode.ENGLISH) is False


class TestInternalSubtitleLanguageSkip:
    def test_skip_internal_subtitle_match(self, monkeypatch, tmp_path):
        _patch_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "skip_if_internal_sub_language", LanguageCode.ENGLISH)
        with (
            patch.object(subgen, "has_internal_subtitle_in_language", return_value=True),
            patch.object(subgen, "get_subtitle_languages", return_value=[]),
            patch.object(subgen, "get_audio_languages", return_value=[]),
        ):
            video = tmp_path / "file.mkv"
            video.touch()
            assert should_skip_file(str(video), LanguageCode.FRENCH) is True

    def test_no_skip_internal_subtitle_no_match(self, monkeypatch, tmp_path):
        _patch_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "skip_if_internal_sub_language", LanguageCode.ENGLISH)
        with (
            patch.object(subgen, "has_internal_subtitle_in_language", return_value=False),
            patch.object(subgen, "get_subtitle_languages", return_value=[]),
            patch.object(subgen, "get_audio_languages", return_value=[]),
        ):
            video = tmp_path / "file.mkv"
            video.touch()
            assert should_skip_file(str(video), LanguageCode.FRENCH) is False


class TestAudioTrackSkip:
    def test_skip_audio_in_skip_list(self, monkeypatch, tmp_path):
        _patch_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "skip_audio_languages", [LanguageCode.ENGLISH])
        with (
            patch.object(subgen, "get_subtitle_languages", return_value=[]),
            patch.object(subgen, "get_audio_languages", return_value=[LanguageCode.ENGLISH]),
        ):
            video = tmp_path / "file.mkv"
            video.touch()
            assert should_skip_file(str(video), LanguageCode.FRENCH) is True

    def test_skip_when_no_preferred_track_and_limit_enabled(self, monkeypatch, tmp_path):
        _patch_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "limit_to_preferred_audio_languages", True)
        monkeypatch.setattr(subgen, "preferred_audio_languages", [LanguageCode.ENGLISH])
        with (
            patch.object(subgen, "get_subtitle_languages", return_value=[]),
            patch.object(subgen, "get_audio_languages", return_value=[LanguageCode.FRENCH]),
        ):
            video = tmp_path / "file.mkv"
            video.touch()
            assert should_skip_file(str(video), LanguageCode.ENGLISH) is True

    def test_no_skip_when_preferred_track_found(self, monkeypatch, tmp_path):
        _patch_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "limit_to_preferred_audio_languages", True)
        monkeypatch.setattr(subgen, "preferred_audio_languages", [LanguageCode.ENGLISH])
        with (
            patch.object(subgen, "get_subtitle_languages", return_value=[]),
            patch.object(subgen, "get_audio_languages", return_value=[LanguageCode.ENGLISH]),
        ):
            video = tmp_path / "file.mkv"
            video.touch()
            assert should_skip_file(str(video), LanguageCode.ENGLISH) is False


class TestSubtitleLanguageListSkip:
    def test_skip_when_subtitle_lang_in_skip_list(self, monkeypatch, tmp_path):
        _patch_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "skip_subtitle_languages", [LanguageCode.ENGLISH])
        with (
            patch.object(subgen, "get_subtitle_languages", return_value=[LanguageCode.ENGLISH]),
            patch.object(subgen, "get_audio_languages", return_value=[]),
        ):
            video = tmp_path / "file.mkv"
            video.touch()
            assert should_skip_file(str(video), LanguageCode.FRENCH) is True


class TestTranslateForceTarget:
    def test_translate_sets_target_to_english(self, monkeypatch, tmp_path):
        """When transcribe_or_translate='translate', target is forced to ENGLISH.
        A file that already has an English subtitle should be skipped."""
        _patch_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "transcribe_or_translate", "translate")
        monkeypatch.setattr(subgen, "skip_if_target_subtitle_exists", True)
        with (
            patch.object(subgen, "subtitle_exists_in_language") as mock_has,
            patch.object(subgen, "get_subtitle_languages", return_value=[]),
            patch.object(subgen, "get_audio_languages", return_value=[]),
        ):
            # Return True only if called with ENGLISH (the forced target)
            mock_has.side_effect = lambda f, lang: lang == LanguageCode.ENGLISH
            video = tmp_path / "file.mkv"
            video.touch()
            # Even though we pass FRENCH, translate mode forces ENGLISH target
            assert should_skip_file(str(video), LanguageCode.FRENCH) is True


# ---------------------------------------------------------------------------
# .subgen_skip marker — transcribe_existing() directory pruning
# ---------------------------------------------------------------------------
class TestSubgenSkipMarker:
    def _patch_scan_defaults(self, monkeypatch):
        monkeypatch.setattr(subgen, "monitor", False)
        monkeypatch.setattr(subgen, "skip_startup_scan", False)
        monkeypatch.setattr(subgen, "use_path_mapping", False)
        monkeypatch.setattr(subgen, "transcribe_or_translate", "transcribe")

    def test_marker_skips_directory_and_subdirs(self, monkeypatch, tmp_path):
        """A .subgen_skip file causes the directory and all subdirectories to be skipped."""
        self._patch_scan_defaults(monkeypatch)

        show_dir = tmp_path / "The Simpsons"
        season1 = show_dir / "Season 1"
        season1.mkdir(parents=True)
        (show_dir / subgen.SKIP_MARKER).touch()
        (season1 / "S01E01.mkv").touch()

        queued = []
        monkeypatch.setattr(subgen, "gen_subtitles_queue", lambda p, t, fl: queued.append(p))

        subgen.transcribe_existing(str(tmp_path))

        assert not any("Simpsons" in p for p in queued), (
            "Files inside a directory with .subgen_skip must not be queued"
        )

    def test_marker_only_skips_marked_subtree(self, monkeypatch, tmp_path):
        """Only the marked subtree is skipped; sibling directories are still scanned."""
        self._patch_scan_defaults(monkeypatch)

        skipped_dir = tmp_path / "OldShow"
        active_dir = tmp_path / "NewShow"
        skipped_dir.mkdir()
        active_dir.mkdir()
        (skipped_dir / subgen.SKIP_MARKER).touch()
        (skipped_dir / "ep.mkv").touch()
        (active_dir / "ep.mkv").touch()

        queued = []
        monkeypatch.setattr(subgen, "gen_subtitles_queue", lambda p, t, fl: queued.append(p))

        subgen.transcribe_existing(str(tmp_path))

        assert not any("OldShow" in p for p in queued), "Marked directory must be skipped"
        assert any("NewShow" in p for p in queued), "Unmarked sibling must still be scanned"

    def test_no_marker_scans_normally(self, monkeypatch, tmp_path):
        """Without a marker file, directories are scanned as normal."""
        self._patch_scan_defaults(monkeypatch)

        video = tmp_path / "movie.mkv"
        video.touch()

        queued = []
        monkeypatch.setattr(subgen, "gen_subtitles_queue", lambda p, t, fl: queued.append(p))

        subgen.transcribe_existing(str(tmp_path))

        assert str(video) in queued


# ---------------------------------------------------------------------------
# SKIP_STARTUP_SCAN — transcribe_existing() scan bypass
# ---------------------------------------------------------------------------
class TestSkipStartupScan:
    def _patch_scan_defaults(self, monkeypatch):
        monkeypatch.setattr(subgen, "monitor", False)
        monkeypatch.setattr(subgen, "use_path_mapping", False)
        monkeypatch.setattr(subgen, "transcribe_or_translate", "transcribe")

    def test_skip_startup_scan_queues_nothing(self, monkeypatch, tmp_path):
        """With SKIP_STARTUP_SCAN=True, no existing files are queued on startup."""
        self._patch_scan_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "skip_startup_scan", True)

        (tmp_path / "movie.mkv").touch()

        queued = []
        monkeypatch.setattr(subgen, "gen_subtitles_queue", lambda p, t, fl: queued.append(p))

        subgen.transcribe_existing(str(tmp_path))

        assert queued == [], "SKIP_STARTUP_SCAN=True must not queue any existing files"

    def test_skip_startup_scan_false_scans_normally(self, monkeypatch, tmp_path):
        """With SKIP_STARTUP_SCAN=False (default), existing files are still queued."""
        self._patch_scan_defaults(monkeypatch)
        monkeypatch.setattr(subgen, "skip_startup_scan", False)

        video = tmp_path / "movie.mkv"
        video.touch()

        queued = []
        monkeypatch.setattr(subgen, "gen_subtitles_queue", lambda p, t, fl: queued.append(p))

        subgen.transcribe_existing(str(tmp_path))

        assert str(video) in queued

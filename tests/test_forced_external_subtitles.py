"""
Tests for issue #358: forced external subtitle files must not count as full
coverage when IGNORE_FORCED_SUBTITLES is enabled, matching how embedded forced
tracks are already treated.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

import subgen
from language_code import LanguageCode
from subgen import has_external_subtitle_in_language


def _make(tmp_path, *names):
    video = tmp_path / "Blue Planet II S01E01.mkv"
    video.touch()
    for name in names:
        (tmp_path / name).touch()
    return str(video)


@pytest.mark.parametrize("sidecar", [
    "Blue Planet II S01E01.en.forced.srt",
    "Blue Planet II S01E01.forced.en.srt",
    "Blue Planet II S01E01.en.FORCED.srt",
])
def test_forced_sidecar_is_not_coverage(monkeypatch, tmp_path, sidecar):
    monkeypatch.setattr(subgen, "ignore_forced_subtitles", True)
    video = _make(tmp_path, sidecar)
    assert has_external_subtitle_in_language(video, LanguageCode.ENGLISH) is False


def test_forced_sidecar_counts_when_setting_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(subgen, "ignore_forced_subtitles", False)
    video = _make(tmp_path, "Blue Planet II S01E01.en.forced.srt")
    assert has_external_subtitle_in_language(video, LanguageCode.ENGLISH) is True


def test_full_sidecar_still_counts_next_to_forced(monkeypatch, tmp_path):
    monkeypatch.setattr(subgen, "ignore_forced_subtitles", True)
    video = _make(tmp_path, "Blue Planet II S01E01.en.forced.srt", "Blue Planet II S01E01.en.srt")
    assert has_external_subtitle_in_language(video, LanguageCode.ENGLISH) is True


def test_forced_sidecar_ignored_when_audio_language_unknown(monkeypatch, tmp_path):
    """With an unknown target language, any subtitle normally counts; a forced one should not."""
    monkeypatch.setattr(subgen, "ignore_forced_subtitles", True)
    video = _make(tmp_path, "Blue Planet II S01E01.en.forced.srt")
    assert has_external_subtitle_in_language(video, LanguageCode.NONE) is False


def test_title_containing_forced_is_not_treated_as_forced(monkeypatch, tmp_path):
    """Only a standalone 'forced' part marks a forced sub, not a word inside the title."""
    monkeypatch.setattr(subgen, "ignore_forced_subtitles", True)
    video = tmp_path / "Forced Entry.mkv"
    video.touch()
    (tmp_path / "Forced Entry.en.srt").touch()
    assert has_external_subtitle_in_language(str(video), LanguageCode.ENGLISH) is True

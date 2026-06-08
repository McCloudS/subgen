"""
Regression tests for specific bug fixes.

Bugs covered:
1. get_subtitle_languages() has no try/except around av.open() — propagates exception
2. refresh_jellyfin_metadata() makes a dead extra GET request before the real POST
3. transcribe_existing() uses `path` after loop — scoping bug when multiple folders
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock, call
import subgen
from subgen import get_subtitle_languages, refresh_jellyfin_metadata, transcribe_existing
from language_code import LanguageCode


# ---------------------------------------------------------------------------
# Bug 1: get_subtitle_languages() unhandled exception
# ---------------------------------------------------------------------------
class TestGetSubtitleLanguagesException:
    def test_returns_empty_list_on_av_error(self, monkeypatch):
        """
        If av.open() raises (corrupt file, permissions, wrong format),
        get_subtitle_languages() must return [] instead of propagating.
        """
        import av as av_mock
        av_mock.open.side_effect = Exception("corrupt file")

        result = get_subtitle_languages("/fake/corrupt.mkv")
        assert result == [], (
            "get_subtitle_languages() should return [] on exception, not propagate"
        )

    def test_still_returns_languages_on_success(self, monkeypatch):
        """Happy path: still returns languages when av.open works."""
        stream = MagicMock()
        stream.metadata = {"language": "eng"}
        # Make stream.disposition & <anything> return 0 (falsy) so the forced-subtitle
        # check does not skip this stream. av is fully mocked in tests so we can't
        # use the real Disposition type — configure __and__ directly instead.
        stream.disposition = MagicMock()
        stream.disposition.__and__ = MagicMock(return_value=0)
        container = MagicMock()
        container.__enter__ = MagicMock(return_value=container)
        container.__exit__ = MagicMock(return_value=False)
        container.streams.subtitles = [stream]

        import av as av_mock
        av_mock.open.side_effect = None
        av_mock.open.return_value = container

        result = get_subtitle_languages("/fake/good.mkv")
        assert LanguageCode.ENGLISH in result


# ---------------------------------------------------------------------------
# Bug 2: refresh_jellyfin_metadata() dead GET
# ---------------------------------------------------------------------------
class TestRefreshJellyfinMetadataDeadGet:
    def test_dead_get_is_not_called(self, requests_mock):
        """
        The intermediate GET to /Users/{admin}/Items/{id}/Refresh is dead code —
        its response is immediately overwritten by the POST.
        It should be removed; calling it wastes bandwidth on every metadata refresh.
        """
        server = "http://jellyfin.local:8096"
        token = "test-token"
        item_id = "abc123"
        admin_id = "admin-user-id"

        # Mock the users list call
        requests_mock.get(
            f"{server}/Users",
            json=[{"Id": admin_id, "Policy": {"IsAdministrator": True}}],
        )

        # This is the dead GET that should be removed
        dead_get_url = f"{server}/Users/{admin_id}/Items/{item_id}/Refresh"

        # The real POST that should stay
        requests_mock.post(
            f"{server}/Items/{item_id}/Refresh?MetadataRefreshMode=FullRefresh",
            status_code=204,
        )

        refresh_jellyfin_metadata(item_id, server, token)

        # The dead GET URL must NOT have been called
        dead_get_called = any(
            req.url == dead_get_url and req.method == "GET"
            for req in requests_mock.request_history
        )
        assert not dead_get_called, (
            "The intermediate GET /Users/{admin}/Items/{id}/Refresh is dead code and must be removed"
        )

    def test_post_is_still_called(self, requests_mock):
        """Verify the actual POST refresh call is still made after removing the dead GET."""
        server = "http://jellyfin.local:8096"
        token = "test-token"
        item_id = "abc123"
        admin_id = "admin-user-id"

        requests_mock.get(
            f"{server}/Users",
            json=[{"Id": admin_id, "Policy": {"IsAdministrator": True}}],
        )
        requests_mock.post(
            f"{server}/Items/{item_id}/Refresh?MetadataRefreshMode=FullRefresh",
            status_code=204,
        )

        refresh_jellyfin_metadata(item_id, server, token)

        post_calls = [
            req for req in requests_mock.request_history
            if req.method == "POST"
        ]
        assert len(post_calls) == 1


# ---------------------------------------------------------------------------
# Bug 3: transcribe_existing() path scoping
# ---------------------------------------------------------------------------
class TestTranscribeExistingScoping:
    def test_single_file_path_is_processed(self, monkeypatch, tmp_path):
        """
        When transcribe_folders contains a single file path (not a directory),
        it should be processed. The fix is to check os.path.isfile(path) INSIDE
        the outer loop, not after it ends (where `path` holds only the last value).
        """
        video = tmp_path / "movie.mkv"
        video.touch()

        queued = []
        monkeypatch.setattr(subgen, "gen_subtitles_queue", lambda p, t, fl: queued.append(p))
        monkeypatch.setattr(subgen, "has_audio", lambda p: True)
        monkeypatch.setattr(subgen, "monitor", False)
        monkeypatch.setattr(subgen, "path_mapping_from", "")
        monkeypatch.setattr(subgen, "path_mapping_to", "")
        monkeypatch.setattr(subgen, "use_path_mapping", False)
        monkeypatch.setattr(subgen, "transcribe_or_translate", "transcribe")

        # Pass the file path directly (not a directory)
        transcribe_existing(str(video))

        assert str(video) in queued, (
            "A direct file path passed to transcribe_existing() should be queued. "
            "The scoping bug causes it to be missed when it's not the last item."
        )

    def test_first_path_file_processed_when_second_path_is_directory(self, monkeypatch, tmp_path):
        """
        Scoping bug: `path` after the outer `for path in transcribe_folders` loop holds
        only the LAST value. When the first entry is a direct file path and the second is
        a directory, the `os.path.isfile(path)` check runs on the directory (last value)
        and the first file is never processed.

        The fix: move `os.path.isfile(path)` check INSIDE the outer loop.
        """
        file_path = tmp_path / "movie.mkv"
        file_path.touch()
        folder = tmp_path / "subdir"
        folder.mkdir()

        queued = []
        monkeypatch.setattr(subgen, "gen_subtitles_queue", lambda p, t, fl: queued.append(p))
        monkeypatch.setattr(subgen, "has_audio", lambda p: True)
        monkeypatch.setattr(subgen, "monitor", False)
        monkeypatch.setattr(subgen, "use_path_mapping", False)
        monkeypatch.setattr(subgen, "transcribe_or_translate", "transcribe")

        # First entry is a FILE, second is a DIRECTORY (empty, so os.walk produces nothing)
        transcribe_existing(f"{file_path}|{folder}")

        assert str(file_path) in queued, (
            "Direct file path passed as first entry must be queued even when followed by a directory. "
            "The scoping bug causes it to be checked against the last path (the directory) instead."
        )

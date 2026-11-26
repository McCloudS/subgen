"""Language detection skips DETECT_LANGUAGE_OFFSET (default 90 s, past most opening themes)
but falls back for audio shorter than offset + length, e.g. a short Bazarr sample."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

import subgen


def test_default_offset_is_90():
    assert subgen.detect_language_offset == int(os.getenv("DETECT_LANGUAGE_OFFSET", 90))


@pytest.mark.parametrize("total,offset,length,expected", [
    (1320, 90, 30, 90),      # 22-minute episode: skip the intro
    (120, 90, 30, 90),       # exactly long enough
    (100, 90, 30, 70),       # too short for the offset: last 30 s
    (30, 90, 30, 0),         # Bazarr-style 30 s sample: all of it
    (10, 90, 30, 0),         # shorter than the window
    (None, 90, 30, 90),      # duration unknown: use the offset
    (1320, 0, 30, 0),        # offset turned off
])
def test_detect_start(total, offset, length, expected):
    assert subgen.detect_start(total, offset, length) == expected


def test_pcm_window_falls_back_to_the_start_of_a_short_clip():
    pcm = bytes(range(256)) * 250 * 15   # 30 s of 16 kHz mono s16le
    assert len(subgen._pcm_window(pcm, 90, 30)) == len(pcm)


def test_pcm_window_skips_the_offset_on_long_audio():
    second = 32000
    pcm = b"\x00\x00" * 16000 * 90 + b"\x01\x00" * 16000 * 60   # 90 s silence, then 60 s of signal
    window = subgen._pcm_window(pcm, 90, 30)
    assert len(window) == 30 * second
    assert window[:2] == b"\x01\x00"

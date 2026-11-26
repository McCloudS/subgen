"""HALLUCINATION_FILTER: collapse phrases Whisper loops over non-speech and drop invented credits.

Cases come from real dev output (soak and non-English tests, Nov 2026)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

import subgen
from tests.test_segmentation import words_from


@pytest.fixture(autouse=True)
def filter_on(monkeypatch):
    monkeypatch.setattr(subgen, "hallucination_filter", True)


def texts(segments):
    return [s["text"].replace("\n", " ") for s in segments]


def test_looped_phrase_keeps_one_copy_with_its_own_timing():
    # The Intouchables: 16 copies over ~30 s without dialogue, then real speech.
    loop = " ".join(["I'm sorry, I can't help it."] * 16)
    words = words_from(loop + " It's coming from there!", start=489.43, step=0.3)
    segs = subgen.split_segments(words)
    assert texts(segs) == ["I'm sorry, I can't help it.", "It's coming from there!"]
    assert abs(segs[0]["start"] - 489.43) < 1e-6
    assert segs[0]["end"] < 492  # first copy's timing, not stretched over the loop


def test_phrase_said_twice_is_kept():
    segs = subgen.split_segments(words_from("Get out of here. Get out of here."))
    assert " ".join(texts(segs)) == "Get out of here. Get out of here."


def test_short_repeats_are_real_speech():
    segs = subgen.split_segments(words_from("No! No! No! No! Stop it."))
    assert " ".join(texts(segs)) == "No! No! No! No! Stop it."


def test_repeats_separated_by_a_pause_are_kept():
    text = "Come back here. Come back here. Come back here."
    segs = subgen.split_segments(words_from(text, gaps={3: 5.0, 6: 5.0}))
    assert " ".join(texts(segs)) == text


def test_credit_split_over_two_cues_is_dropped():
    # My Little Pony: "Transcription" and "by CastingWords" came out as separate cues.
    words = words_from("And that's how it happened.", start=0) + \
        words_from("Transcription", start=1313.6) + words_from("by CastingWords", start=1315.8)
    assert texts(subgen.split_segments(words)) == ["And that's how it happened."]


@pytest.mark.parametrize("credit", [
    "Subtitles by the Amara.org community",
    "Thanks for watching!",
    "Thank you for watching.",
    "Untertitel im Auftrag des ZDF, 2017",
    "Sous-titres réalisés par la communauté d'Amara.org",
])
def test_known_credits_are_dropped(credit):
    words = words_from("We should go.", start=0) + words_from(credit, start=10)
    assert texts(subgen.split_segments(words)) == ["We should go."]


def test_credit_words_inside_real_dialogue_are_kept():
    text = "Thanks for watching the kids last night."
    assert " ".join(texts(subgen.split_segments(words_from(text)))) == text


def test_filter_off_keeps_everything(monkeypatch):
    monkeypatch.setattr(subgen, "hallucination_filter", False)
    loop = " ".join(["I'm sorry, I can't help it."] * 4)
    words = words_from(loop) + words_from("Thanks for watching!", start=60)
    assert " ".join(texts(subgen.split_segments(words))).count("help it") == 4
    assert texts(subgen.split_segments(words))[-1] == "Thanks for watching!"

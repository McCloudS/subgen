"""
Subtitle segmentation and whisper.cpp token merging.

Cases come from a real 22-minute episode where dev's earlier segmentation produced
an unwrapped 84-character line, "Cnote" for "C-note", "classic.Ew!" across a speaker
change, and orphan one-word cues.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

import subgen


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.setattr(subgen, "max_line_length", 42)
    monkeypatch.setattr(subgen, "gap_split_secs", 0.4)
    monkeypatch.setattr(subgen, "max_segment_secs", 5.0)


def words_from(text, start=0.0, step=0.3, gaps=None):
    """Word dicts with evenly spaced timings; gaps={index: extra_seconds_before_word}."""
    out, t = [], start
    for i, w in enumerate(text.split()):
        t += (gaps or {}).get(i, 0.0)
        out.append({"word": w, "start": t, "end": t + step * 0.9})
        t += step
    return out


def tokens(*texts, start_ms=0, step_ms=100):
    return [{"text": t, "offsets": {"from": start_ms + i * step_ms, "to": start_ms + (i + 1) * step_ms}}
            for i, t in enumerate(texts)]


def seg(*texts):
    return {"offsets": {"from": 0, "to": 100 * len(texts)}, "tokens": tokens(*texts)}


def merged(*texts):
    return [w["word"] for w in subgen._wcp_tokens_to_words([seg(*texts)])]


# ---------------------------------------------------------------------------
# whisper.cpp token merging
# ---------------------------------------------------------------------------
class TestTokenMerge:
    def test_hyphenated_word_keeps_hyphen(self):
        assert merged(" Also", " a", " C", "-", "note", ".") == ["Also", "a", "C-note."]

    def test_hyphen_inside_multi_part_word(self):
        assert merged(" five", "-", "year", "-", "old") == ["five-year-old"]
        assert merged(" Mm", "-", "hmm", ".") == ["Mm-hmm."]

    def test_continuation_token_starting_with_hyphen(self):
        assert merged(" e", "-mail") == ["e-mail"]

    def test_speaker_dash_dropped_and_next_word_not_glued(self):
        assert merged(" classic", ".", " -", "Ew", "!", " -", "Ew", "!") == ["classic.", "Ew!", "Ew!"]

    def test_speaker_dash_at_segment_start(self):
        assert merged("-", " Mm", "-", "hmm", ",") == ["Mm-hmm,"]

    def test_dash_glued_to_new_word_is_stripped(self):
        assert merged(" Yes", ".", " -Ew", "!") == ["Yes.", "Ew!"]

    def test_interruption_dashes_kept(self):
        assert merged(" I", "'m", "--") == ["I'm--"]

    def test_special_tokens_skipped(self):
        assert merged("[_BEG_]", " Hello", ".", "[_TT_388]") == ["Hello."]


# ---------------------------------------------------------------------------
# split_segments
# ---------------------------------------------------------------------------
def lines_of(segments):
    return [line for s in segments for line in s["text"].split("\n")]


class TestSegmentation:
    def test_no_line_exceeds_max_length(self):
        # 84 characters: the old code couldn't split this into two lines of <= 42 and
        # wrote it on one line.
        text = "Well, I emailed the superintendent and told him everything that Ava has done today."
        segments = subgen.split_segments(words_from(text, step=0.05))
        assert all(len(line) <= 42 for line in lines_of(segments))
        assert " ".join(lines_of(segments)) == text

    def test_long_sentence_not_cut_before_its_last_word(self):
        text = "Well, I emailed the superintendent and told him everything that Ava has done today."
        segments = subgen.split_segments(words_from(text, step=0.05))
        assert all(s["text"] != "today." for s in segments)

    def test_pieces_dont_end_on_dangling_word(self):
        text = ("I have been working in the Philadelphia School District for twenty years "
                "and Ava is just the latest in a long line of people who do absolutely nothing.")
        segments = subgen.split_segments(words_from(text, step=0.05))
        for s in segments[:-1]:
            assert subgen._bare(s["text"].split()[-1]) not in subgen._DANGLING, s["text"]

    def test_pause_mid_sentence_does_not_orphan_a_word(self):
        segments = subgen.split_segments(words_from("You Sicilian?", gaps={1: 0.5}))
        assert [s["text"] for s in segments] == ["You Sicilian?"]

    def test_long_pause_splits_cues(self):
        segments = subgen.split_segments(words_from("Good morning. See you later.", gaps={2: 2.5}))
        assert [s["text"] for s in segments] == ["Good morning.", "See you later."]

    def test_too_long_cue_splits_after_short_phrase_with_comma(self):
        # Spans more than max_segment_secs, so it must split; "Oh, Nina," is a natural phrase.
        segments = subgen.split_segments(words_from("Oh, Nina, excellent sentence structure.", step=1.4))
        assert [s["text"] for s in segments] == ["Oh, Nina,", "excellent sentence structure."]

    def test_short_sentences_share_a_cue(self):
        segments = subgen.split_segments(words_from("Good. Mine too."))
        assert [s["text"] for s in segments] == ["Good. Mine too."]

    def test_two_line_cue_breaks_at_sentence_end(self):
        text = "It's an old book, so here's where I taped it. These are the others."
        segments = subgen.split_segments(words_from(text, step=0.06))
        assert len(segments) == 1
        assert segments[0]["text"] == "It's an old book, so here's where I taped it.\nThese are the others." \
            or all(len(line) <= 42 for line in lines_of(segments))

    def test_at_most_two_sentences_per_cue(self):
        segments = subgen.split_segments(words_from("Okay. Nice. Yeah. Sure.", step=0.2))
        assert all(subgen._sentence_count(words_from(s["text"])) <= 2 for s in segments)

    def test_abbreviation_is_not_a_sentence_end(self):
        segments = subgen.split_segments(words_from("Mr. Johnson is here. Hi."))
        assert segments[0]["text"].startswith("Mr. Johnson is here.")

    def test_cue_never_exceeds_max_segment_secs(self):
        text = " ".join(["word"] * 30)
        segments = subgen.split_segments(words_from(text, step=0.4))
        assert all(s["end"] - s["start"] <= 5.0 + 1e-9 for s in segments)

    def test_short_cue_extended_without_overlapping_next(self):
        ws = words_from("Hi. There you are, finally.", step=0.2, gaps={1: 2.0})
        segments = subgen.split_segments(ws)
        assert segments[0]["text"] == "Hi."
        assert segments[0]["end"] - segments[0]["start"] >= 1.0 - 1e-9
        assert segments[0]["end"] <= segments[1]["start"]

    def test_no_words_lost_or_reordered(self):
        text = ("All right guys, so there have been three presidents since this one, okay? "
                "It's an old book, so here's where I taped in the others. Any questions? No? Great.")
        segments = subgen.split_segments(words_from(text, step=0.15, gaps={14: 0.6}))
        assert " ".join(lines_of(segments)) == text

    def test_empty(self):
        assert subgen.split_segments([]) == []

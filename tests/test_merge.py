"""Unit tests for the merge algorithm (ARCHITECTURE.md §7, test plan §15 #1).

Synthetic words/segments only — no models, no I/O, no network. Target: milliseconds.
"""

from __future__ import annotations

import pytest

from transcription247.config import MergeConfig
from transcription247.merge import (
    MergeError,
    assign_speakers,
    group_turns,
    merge,
    renumber,
    smooth,
)
from transcription247.models import SCHEMA_VERSION, AsrResult, SpeakerSegment, Word

CFG = MergeConfig()


def w(start: float, end: float, text: str = "palabra") -> Word:
    return Word(start=start, end=end, word=f" {text}", probability=0.99)


def seg(start: float, end: float, speaker: int) -> SpeakerSegment:
    return SpeakerSegment(start=start, end=end, speaker=speaker)


def asr(words: list[Word], **kw) -> AsrResult:
    return AsrResult(
        schema=SCHEMA_VERSION,
        job_id=kw.get("job_id", "01k5test"),
        language=kw.get("language", "es"),
        language_probability=0.99,
        language_forced=False,
        duration_s=kw.get("duration_s", 100.0),
        model="faster-whisper-large-v3-turbo-int8",
        segments=[],
        words=words,
    )


# --------------------------------------------------------------------------- 1. assignment


def test_max_overlap_wins_for_a_word_straddling_a_boundary():
    segments = [seg(0.0, 10.0, 0), seg(10.0, 20.0, 1)]
    # 0.2 s in speaker 0, 0.8 s in speaker 1.
    words = [w(9.8, 10.8)]
    assert assign_speakers(words, segments, CFG) == [1]
    # Mirror image: now the majority is speaker 0.
    assert assign_speakers([w(9.2, 10.2)], segments, CFG) == [0]


def test_overlap_ties_go_to_the_lower_segment_index():
    segments = [seg(0.0, 10.0, 7), seg(10.0, 20.0, 3)]
    assert assign_speakers([w(9.5, 10.5)], segments, CFG) == [7]


def test_word_outside_every_segment_takes_the_nearest_centre_within_max_gap():
    segments = [seg(0.0, 4.0, 0), seg(10.0, 14.0, 1)]  # centres at 2.0 and 12.0
    # Word centred at 11.5: 0.5 s from segment 1's centre -> speaker 1.
    assert assign_speakers([w(11.0, 12.0)], segments, CFG) == [1]


def test_word_far_from_every_centre_inherits_the_previous_word_then_the_first_segment():
    segments = [seg(0.0, 2.0, 5), seg(100.0, 102.0, 9)]  # centres 1.0 and 101.0
    # First word is far from both centres and has no predecessor -> first segment's speaker (5).
    # Second word overlaps segment 9; third is far from both centres -> inherits 9.
    speakers = assign_speakers([w(50.0, 50.5), w(100.5, 101.0), w(50.0, 50.5)], segments, CFG)
    assert speakers == [5, 9, 9]


def test_no_segments_at_all_assigns_zero():
    assert assign_speakers([w(0.0, 1.0), w(1.0, 2.0)], [], CFG) == [0, 0]


def test_empty_words_assigns_nothing():
    assert assign_speakers([], [seg(0.0, 5.0, 0)], CFG) == []


# --------------------------------------------------------------------------- 2. smoothing


def test_one_short_word_flip_between_identical_neighbours_is_absorbed():
    words = [w(0.0, 1.0), w(1.0, 2.0), w(2.0, 2.3), w(2.4, 3.4), w(3.5, 4.5)]
    labels = [0, 0, 1, 0, 0]  # a single 0.3 s interjection
    assert smooth(labels, words, CFG) == [0, 0, 0, 0, 0]


def test_a_long_enough_run_is_not_absorbed():
    words = [w(0.0, 1.0), w(1.0, 2.0), w(2.0, 3.0), w(3.0, 4.0), w(4.0, 5.0)]
    labels = [0, 0, 1, 0, 0]  # the flip lasts 1.0 s >= min_run_s (0.6)
    assert smooth(labels, words, CFG) == [0, 0, 1, 0, 0]


def test_a_run_with_different_neighbours_is_not_absorbed():
    words = [w(0.0, 1.0), w(1.0, 1.2), w(1.3, 2.3)]
    assert smooth([0, 2, 1], words, CFG) == [0, 2, 1]


def test_a_leading_or_trailing_short_run_has_no_two_neighbours_and_survives():
    words = [w(0.0, 0.2), w(0.3, 1.3), w(1.4, 2.4)]
    assert smooth([1, 0, 0], words, CFG) == [1, 0, 0]


def test_smooth_does_not_mutate_its_input():
    words = [w(0.0, 1.0), w(1.0, 1.2), w(1.3, 2.3)]
    labels = [0, 1, 0]
    smooth(labels, words, CFG)
    assert labels == [0, 1, 0]


# --------------------------------------------------------------------------- 3. turn grouping


def test_a_long_silence_starts_a_new_turn_for_the_same_speaker():
    words = [w(0.0, 1.0, "hola"), w(1.1, 2.0, "qué"), w(3.6, 4.5, "tal")]  # 1.6 s gap > 1.20
    turns = group_turns(words, [0, 0, 0], CFG)
    assert [t.text for t in turns] == ["hola qué", "tal"]


def test_a_speaker_change_starts_a_new_turn_and_text_is_assembled():
    words = [w(0.0, 1.0, "hola."), w(1.1, 2.0, "adiós"), w(2.1, 3.0, ",")]
    turns = group_turns(words, [0, 1, 1], CFG)
    assert [(t.speaker, t.text) for t in turns] == [(0, "hola."), (1, "adiós,")]
    assert turns[1].start == 1.1 and turns[1].end == 3.0


def test_a_long_monologue_only_splits_at_a_sentence_end():
    cfg = MergeConfig(max_turn_s=5.0)
    words = [
        w(0.0, 1.0, "uno"), w(1.1, 2.0, "dos"), w(2.1, 3.0, "tres"),
        w(3.1, 4.0, "cuatro"), w(4.1, 5.5, "cinco"),   # spans > 5 s but does not end a sentence
        w(5.6, 6.5, "seis."),                          # ends a sentence -> next word opens a turn
        w(6.6, 7.5, "siete"),
    ]
    turns = group_turns(words, [0] * len(words), cfg)
    assert [t.text for t in turns] == ["uno dos tres cuatro cinco seis.", "siete"]


def test_empty_words_group_to_no_turns():
    assert group_turns([], [], CFG) == []


# --------------------------------------------------------------------------- 5. renumbering


def test_renumbering_follows_first_appearance():
    words = [w(0.0, 1.0, "a"), w(1.1, 2.0, "b"), w(2.1, 3.0, "c")]
    turns = group_turns(words, [2, 0, 2], CFG)
    renumbered, participantes = renumber(turns)
    assert participantes == ["Speaker 1", "Speaker 2"]
    assert [(t.speaker, t.speaker_label) for t in renumbered] == [
        (2, "Speaker 1"),
        (0, "Speaker 2"),
        (2, "Speaker 1"),
    ]


# --------------------------------------------------------------------------- merge() end to end


def test_merge_end_to_end_two_speakers():
    words = [
        w(0.0, 1.0, "Hola."), w(1.1, 2.0, "Buenas"), w(2.1, 3.0, "tardes."),
        w(10.5, 11.0, "Sí,"), w(11.1, 12.0, "claro."),
    ]
    segments = [seg(0.0, 10.0, 4), seg(10.0, 20.0, 1)]
    result = merge(asr(words, duration_s=20.0), segments, CFG)
    assert result.schema == SCHEMA_VERSION
    assert result.job_id == "01k5test" and result.language == "es" and result.duration_s == 20.0
    assert result.participantes == ["Speaker 1", "Speaker 2"]
    assert [(t.speaker_label, t.text) for t in result.turns] == [
        ("Speaker 1", "Hola. Buenas tardes."),
        ("Speaker 2", "Sí, claro."),
    ]


def test_merge_without_diarization_yields_a_single_speaker():
    words = [w(0.0, 1.0, "solo"), w(1.1, 2.0, "yo")]
    result = merge(asr(words), [], CFG)
    assert result.participantes == ["Speaker 1"]
    assert len(result.turns) == 1


def test_merge_raises_on_empty_words():
    with pytest.raises(MergeError):
        merge(asr([]), [seg(0.0, 5.0, 0)], CFG)


def test_merge_is_deterministic():
    words = [w(float(i), i + 0.8, f"p{i}") for i in range(40)]
    segments = [seg(0.0, 15.0, 3), seg(15.0, 30.0, 8), seg(30.0, 45.0, 3)]
    first = merge(asr(words, duration_s=45.0), segments, CFG)
    second = merge(asr(words, duration_s=45.0), segments, CFG)
    assert first == second

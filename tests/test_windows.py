"""Window planning and per-window timestamp offsetting (ARCHITECTURE.md §3.4/§3.5 "Long audio").

No ML anywhere: the planner is pure signal processing over synthetic audio, and the ASR collector
is driven with hand-built stand-ins for faster-whisper's segments. The tests use a 1 kHz sample
rate so a "four-minute" recording is a few hundred kB — the planner is rate-agnostic, everything
it does is in frames.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")

from transcription247.asr import PARTIAL_NAME, Transcriber, _PartialState  # noqa: E402
from transcription247.config import AsrConfig, PipelineConfig  # noqa: E402
from transcription247.windows import (  # noqa: E402
    Window,
    WindowError,
    plan_from_config,
    plan_windows,
    read_window,
)

RATE = 1000


def _write(path, spans, *, rate=RATE):
    """Build a wav from `[(seconds, amplitude), ...]`; amplitude 0 is a pause."""
    parts = []
    t = 0.0
    for seconds, amp in spans:
        n = int(seconds * rate)
        if amp == 0.0:
            parts.append(np.zeros(n, dtype=np.float32))
        else:
            # A tone, not noise, so the RMS of every block is identical and the thresholds in
            # _longest_pause() are exercised at their worst case.
            idx = np.arange(n, dtype=np.float32) + t * rate
            parts.append((amp * np.sin(2 * np.pi * 40.0 * idx / rate)).astype(np.float32))
        t += seconds
    data = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    soundfile.write(str(path), data, rate, subtype="PCM_16")
    return data


# --------------------------------------------------------------------------- planning


def test_short_recording_is_one_window_covering_everything(tmp_path):
    wav = tmp_path / "short.wav"
    _write(wav, [(30.0, 0.3)])

    plan = plan_windows(wav, max_s=100.0, search_s=20.0, min_s=10.0)

    assert len(plan) == 1
    assert plan[0].start_frame == 0
    assert plan[0].end_frame == 30 * RATE
    assert plan[0].start_s == 0.0
    assert plan[0].duration_s == pytest.approx(30.0)


def test_cuts_land_in_the_pause_inside_the_search_region(tmp_path):
    wav = tmp_path / "pauses.wav"
    # Speech everywhere except two 4 s pauses placed inside the tail each window will search.
    _write(wav, [
        (88.0, 0.3), (4.0, 0.0),    # pause 88-92 s, inside [80, 100)
        (86.0, 0.3), (4.0, 0.0),    # pause 178-182 s, inside [170, 190)
        (68.0, 0.3),                # tail, shorter than max_s
    ])

    plan = plan_windows(wav, max_s=100.0, search_s=20.0, min_s=10.0)

    assert len(plan) == 3
    assert plan[0].end_s == pytest.approx(90.0, abs=1.0)
    assert plan[1].end_s == pytest.approx(180.0, abs=1.0)
    assert plan[2].end_s == pytest.approx(250.0)


def test_continuous_speech_falls_back_to_a_hard_cut(tmp_path):
    wav = tmp_path / "solid.wav"
    _write(wav, [(250.0, 0.3)])  # no pause anywhere

    plan = plan_windows(wav, max_s=100.0, search_s=20.0, min_s=10.0)

    assert [w.end_s for w in plan[:-1]] == [100.0, 200.0]  # exactly max_s, no drift
    assert plan[-1].end_s == pytest.approx(250.0)


def test_windows_are_contiguous_and_cover_the_file_exactly(tmp_path):
    wav = tmp_path / "mixed.wav"
    _write(wav, [(88.0, 0.3), (4.0, 0.0), (86.0, 0.3), (4.0, 0.0), (68.0, 0.3)])

    plan = plan_windows(wav, max_s=100.0, search_s=20.0, min_s=10.0)

    assert plan[0].start_frame == 0
    assert plan[-1].end_frame == 250 * RATE
    for a, b in zip(plan, plan[1:]):
        assert a.end_frame == b.start_frame  # no gap, no overlap
    assert [w.index for w in plan] == list(range(len(plan)))
    assert sum(w.frames for w in plan) == 250 * RATE


def test_no_window_is_shorter_than_min_s(tmp_path):
    wav = tmp_path / "tail.wav"
    # A pause right before the end would tempt the planner into a 3 s final window.
    _write(wav, [(100.0, 0.3), (4.0, 0.0), (3.0, 0.3)])

    plan = plan_windows(wav, max_s=100.0, search_s=20.0, min_s=10.0)

    assert all(w.duration_s >= 10.0 for w in plan)


def test_read_window_returns_exactly_that_slice(tmp_path):
    wav = tmp_path / "slice.wav"
    data = _write(wav, [(88.0, 0.3), (4.0, 0.0), (86.0, 0.3), (4.0, 0.0), (68.0, 0.3)])

    plan = plan_windows(wav, max_s=100.0, search_s=20.0, min_s=10.0)

    rebuilt = np.concatenate([read_window(wav, w) for w in plan])
    assert rebuilt.dtype == np.float32
    assert len(rebuilt) == len(data)
    assert np.allclose(rebuilt, data, atol=1e-4)  # PCM_16 round-trip
    for w in plan:
        assert len(read_window(wav, w)) == w.frames


def test_window_max_s_zero_disables_windowing(tmp_path):
    wav = tmp_path / "off.wav"
    _write(wav, [(250.0, 0.3)])

    plan = plan_from_config(wav, PipelineConfig(window_max_s=0))

    assert len(plan) == 1
    assert plan[0].frames == 250 * RATE


def test_plan_from_config_is_deterministic_across_stages(tmp_path):
    wav = tmp_path / "det.wav"
    _write(wav, [(88.0, 0.3), (4.0, 0.0), (86.0, 0.3), (4.0, 0.0), (68.0, 0.3)])
    pipeline = PipelineConfig(window_max_s=100.0, window_cut_search_s=20.0, window_min_s=10.0)

    first = plan_from_config(wav, pipeline)   # what the ASR stage would use
    second = plan_from_config(wav, pipeline)  # what the diarization stage would use

    assert [w.as_pair() for w in first] == [w.as_pair() for w in second]


def test_missing_wav_is_a_window_error(tmp_path):
    with pytest.raises(WindowError):
        plan_windows(tmp_path / "nope.wav", max_s=100.0)


# --------------------------------------------------------------------------- ASR offsetting


class _FakeWord:
    def __init__(self, start, end, word):
        self.start, self.end, self.word, self.probability = start, end, word, 0.9


class _FakeSegment:
    """Shaped like faster-whisper's Segment; timestamps are window-local, as the real one's are."""

    def __init__(self, start, end, text, words):
        self.start, self.end, self.text = start, end, text
        self.words = [_FakeWord(a, b, w) for a, b, w in words]
        self.avg_logprob, self.no_speech_prob = -0.2, 0.01


def _collector():
    t = Transcriber(AsrConfig(), pipeline=PipelineConfig())
    return t, _PartialState.empty()


def test_collect_shifts_every_timestamp_onto_the_global_timeline():
    t, state = _collector()
    window = Window(index=2, start_frame=1800 * 16000, end_frame=2700 * 16000, sample_rate=16000)

    t._collect(
        iter([_FakeSegment(5.0, 7.5, " hola", [(5.0, 6.0, " ho"), (6.0, 7.5, "la")])]),
        window, state, total_s=2700.0, t0=0.0,
    )

    assert state.segments[0].start == pytest.approx(1805.0)
    assert state.segments[0].end == pytest.approx(1807.5)
    assert [w.start for w in state.words] == pytest.approx([1805.0, 1806.0])
    assert [w.end for w in state.words] == pytest.approx([1806.0, 1807.5])


def test_segment_ids_stay_unique_and_ordered_across_windows():
    t, state = _collector()
    first = Window(index=0, start_frame=0, end_frame=900 * 16000, sample_rate=16000)
    second = Window(index=1, start_frame=900 * 16000, end_frame=1800 * 16000, sample_rate=16000)

    t._collect(iter([_FakeSegment(1.0, 2.0, "a", [(1.0, 2.0, "a")]),
                     _FakeSegment(3.0, 4.0, "b", [(3.0, 4.0, "b")])]),
               first, state, total_s=1800.0, t0=0.0)
    t._collect(iter([_FakeSegment(1.0, 2.0, "c", [(1.0, 2.0, "c")])]),
               second, state, total_s=1800.0, t0=0.0)

    assert [s.id for s in state.segments] == [0, 1, 2]
    assert [round(s.start, 1) for s in state.segments] == [1.0, 3.0, 901.0]
    assert [s.text for s in state.segments] == ["a", "b", "c"]


# --------------------------------------------------------------------------- partial resume


def _plan(n, seconds=900.0, rate=16000):
    return [
        Window(index=i, start_frame=int(i * seconds * rate),
               end_frame=int((i + 1) * seconds * rate), sample_rate=rate)
        for i in range(n)
    ]


def test_partial_state_round_trips(tmp_path):
    t, state = _collector()
    plan = _plan(3)
    t._collect(iter([_FakeSegment(1.0, 2.0, "a", [(1.0, 2.0, "a")])]), plan[0], state, 2700.0, 0.0)
    state.language, state.language_probability, state.done = "es", 0.98, 1
    path = tmp_path / PARTIAL_NAME

    state.save(path, plan)
    back = _PartialState.load(path, plan)

    assert back is not None
    assert back.done == 1
    assert back.language == "es"
    assert back.language_probability == pytest.approx(0.98)
    assert [s.text for s in back.segments] == ["a"]
    assert [w.start for w in back.words] == pytest.approx([1.0])


def test_partial_state_is_ignored_when_the_window_plan_changed(tmp_path):
    _, state = _collector()
    state.done = 1
    path = tmp_path / PARTIAL_NAME
    state.save(path, _plan(3))

    # Same job, different pipeline.window_max_s -> the old timeline must not be spliced in.
    assert _PartialState.load(path, _plan(3, seconds=600.0)) is None
    assert _PartialState.load(path, _plan(4)) is None


def test_partial_state_survives_a_corrupt_file(tmp_path):
    path = tmp_path / PARTIAL_NAME
    path.write_text("{not json", encoding="utf-8")
    assert _PartialState.load(path, _plan(2)) is None

    path.write_text(json.dumps({"schema": 999, "windows": [], "done": 1}), encoding="utf-8")
    assert _PartialState.load(path, _plan(2)) is None

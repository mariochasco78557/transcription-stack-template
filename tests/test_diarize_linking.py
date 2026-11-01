"""Global speaker linking across windows (ARCHITECTURE.md §3.5 "Long audio").

Pure numpy: the embeddings here are hand-built unit vectors standing in for eres2net centroids, so
these tests pin the *linking logic* — which is where a cross-window bug would hide — without
loading a model or touching audio.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from transcription247.config import DiarizationConfig
from transcription247.diarize import (
    _cut_k,
    _merge_adjacent,
    _pool,
    estimate_speakers,
    link_windows,
)
from transcription247.models import SpeakerSegment

CFG = DiarizationConfig()


def _voice(seed: int, dim: int = 192, jitter: float = 0.0, rng_seed: int = 0) -> np.ndarray:
    """A reproducible unit vector; `jitter` perturbs it the way a different window of the same
    speaker would.

    The perturbation is normalised before it is added, so `jitter` is the perturbation's length
    relative to the voice itself: the cosine distance to the unjittered vector is
    `1 - 1/sqrt(1 + jitter**2)` (0.4 -> 0.07, 0.5 -> 0.11), i.e. comfortably inside the shipped
    `merge_threshold` of 0.475, while two different seeds are near-orthogonal in 192 dimensions
    and land around 1.0. That is the same geometry real eres2net centroids show (bench/RESULTS.md).
    """
    base = np.random.default_rng(seed).normal(size=dim)
    base = base / np.linalg.norm(base)
    if jitter:
        noise = np.random.default_rng(1_000_000 * seed + rng_seed).normal(size=dim)
        base = base + jitter * (noise / np.linalg.norm(noise))
    return base / np.linalg.norm(base)


# --------------------------------------------------------------------------- link_windows


def test_same_speaker_in_different_windows_gets_one_global_id():
    alice = [_voice(1, jitter=0.4, rng_seed=w) for w in range(3)]
    bob = [_voice(2, jitter=0.4, rng_seed=w) for w in range(3)]
    keys = [(w, local) for w in range(3) for local in (0, 1)]
    centroids = [v for w in range(3) for v in (alice[w], bob[w])]
    speech = [60.0] * 6

    mapping = link_windows(keys, centroids, speech, CFG)

    assert len({mapping[(w, 0)] for w in range(3)}) == 1  # Alice is one person everywhere
    assert len({mapping[(w, 1)] for w in range(3)}) == 1  # so is Bob
    assert mapping[(0, 0)] != mapping[(0, 1)]             # and they are not each other
    assert len(set(mapping.values())) == 2


def test_local_ids_may_be_permuted_between_windows():
    """sherpa's cluster ids are per-call and arbitrary: local 0 in window 1 can be the person who
    was local 1 in window 0. Only the embeddings decide."""
    alice0, alice1 = _voice(1, jitter=0.4, rng_seed=0), _voice(1, jitter=0.4, rng_seed=1)
    bob0, bob1 = _voice(2, jitter=0.4, rng_seed=0), _voice(2, jitter=0.4, rng_seed=1)
    keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    centroids = [alice0, bob0, bob1, alice1]  # window 1 has them the other way round
    speech = [60.0] * 4

    mapping = link_windows(keys, centroids, speech, CFG)

    assert mapping[(0, 0)] == mapping[(1, 1)]  # Alice
    assert mapping[(0, 1)] == mapping[(1, 0)]  # Bob
    assert len(set(mapping.values())) == 2


def test_a_speaker_absent_from_a_window_does_not_invent_one():
    """Window 1 is a monologue. The global answer is still 2 speakers, not 3."""
    keys = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]
    centroids = [
        _voice(1, jitter=0.4, rng_seed=0), _voice(2, jitter=0.4, rng_seed=0),
        _voice(1, jitter=0.4, rng_seed=1),
        _voice(1, jitter=0.4, rng_seed=2), _voice(2, jitter=0.4, rng_seed=2),
    ]
    speech = [60.0, 60.0, 300.0, 60.0, 60.0]

    mapping = link_windows(keys, centroids, speech, CFG)

    assert len(set(mapping.values())) == 2
    assert mapping[(0, 0)] == mapping[(1, 0)] == mapping[(2, 0)]
    assert mapping[(0, 1)] == mapping[(2, 1)]


def test_tiny_clusters_are_judged_on_the_global_speech_share():
    """A 4 s blip in one window is 4 s out of the whole recording, not 4 s out of that window —
    so it is absorbed even though it would be 20% of a short window's speech."""
    keys = [(0, 0), (0, 1), (1, 0)]
    centroids = [_voice(1), _voice(2), _voice(3)]
    speech = [1200.0, 1200.0, 4.0]  # 4 s out of ~40 min is well under min_cluster_share

    mapping = link_windows(keys, centroids, speech, CFG)

    assert len(set(mapping.values())) == 2
    assert mapping[(1, 0)] in {mapping[(0, 0)], mapping[(0, 1)]}


def test_max_speakers_caps_the_global_count():
    cfg = dataclasses.replace(CFG, max_speakers=3)
    keys = [(w, 0) for w in range(6)]
    centroids = [_voice(10 + i) for i in range(6)]  # six mutually distant voices
    speech = [200.0] * 6

    mapping = link_windows(keys, centroids, speech, cfg)

    assert len(set(mapping.values())) <= 3


def test_link_windows_on_an_empty_plan_is_empty():
    assert link_windows([], [], [], CFG) == {}


# --------------------------------------------------------------------------- known count


def test_force_k_cuts_the_global_tree_at_exactly_k():
    keys = [(w, local) for w in range(2) for local in (0, 1, 2)]
    centroids = [
        _voice(1, jitter=0.4, rng_seed=0), _voice(2, jitter=0.4, rng_seed=0), _voice(1, jitter=0.5, rng_seed=5),
        _voice(1, jitter=0.4, rng_seed=1), _voice(2, jitter=0.4, rng_seed=1), _voice(2, jitter=0.5, rng_seed=6),
    ]
    speech = [100.0] * 6

    assert len(set(link_windows(keys, centroids, speech, CFG, force_k=2).values())) == 2
    assert len(set(link_windows(keys, centroids, speech, CFG, force_k=3).values())) == 3
    assert len(set(link_windows(keys, centroids, speech, CFG, force_k=4).values())) == 4


def test_force_k_ignores_the_tiny_cluster_rule_the_auto_path_applies():
    """The caller said there are 3 people; a quiet third one must not be absorbed away."""
    keys = [(0, 0), (0, 1), (1, 0)]
    centroids = [_voice(1), _voice(2), _voice(3)]
    speech = [1200.0, 1200.0, 4.0]

    auto = link_windows(keys, centroids, speech, CFG)
    known = link_windows(keys, centroids, speech, CFG, force_k=3)

    assert len(set(auto.values())) == 2
    assert len(set(known.values())) == 3


def test_cut_k_is_exact_for_every_k():
    history = [(0, 1, 0.1), (2, 3, 0.2), (0, 2, 0.6), (0, 4, 0.9)]
    for k in range(1, 6):
        assert len(set(_cut_k(history, 5, k).tolist())) == k


def test_cut_k_above_n_leaves_everything_apart():
    assert _cut_k([(0, 1, 0.1)], 2, 5).tolist() == [0, 1]


# --------------------------------------------------------------------------- pooling equivalence


def test_pooling_per_window_matches_pooling_inside_estimate_speakers():
    """The single-window regression guarantee in one assertion.

    `estimate_speakers()` pools per-segment embeddings by over-cluster internally. The windowed
    path pools them itself and hands over one row per cluster. Both must produce the same
    speakers, or a recording would be labelled differently depending on whether it happened to
    fit in one window.
    """
    rng = np.random.default_rng(7)
    vecs, durs, labels = [], [], []
    for local, seed in ((0, 1), (1, 2)):
        for _ in range(5):
            v = _voice(seed) + 0.12 * rng.normal(size=192)
            vecs.append((v / np.linalg.norm(v)).astype(np.float32))
            durs.append(6.0)
            labels.append(local)

    per_segment = estimate_speakers(
        np.stack(vecs), np.asarray(durs, dtype=np.float64), np.asarray(labels), CFG
    )
    pooled = _pool(vecs, durs, labels)
    windowed = link_windows(
        [(0, local) for local, _, _ in pooled],
        [c for _, c, _ in pooled],
        [s for _, _, s in pooled],
        CFG,
    )

    assert len(set(per_segment.values())) == len(set(windowed.values())) == 2
    assert (per_segment[0] == per_segment[1]) == (windowed[(0, 0)] == windowed[(0, 1)])


def test_pool_returns_unit_centroids_weighted_by_duration():
    a, b = _voice(1), _voice(2)
    pooled = dict((cid, (c, s)) for cid, c, s in _pool([a, a, b], [10.0, 30.0, 5.0], [0, 0, 1]))

    assert set(pooled) == {0, 1}
    assert pooled[0][1] == pytest.approx(40.0)
    assert pooled[1][1] == pytest.approx(5.0)
    assert np.linalg.norm(pooled[0][0]) == pytest.approx(1.0)
    assert np.allclose(pooled[0][0], a, atol=1e-9)  # same direction twice -> unchanged


# --------------------------------------------------------------------------- window joins


def test_same_speaker_across_a_window_join_becomes_one_turn():
    """Sentence straddling the cut at 900 s: after global relabelling it must be one segment."""
    segments = [
        SpeakerSegment(start=880.0, end=899.8, speaker=0),
        SpeakerSegment(start=900.0, end=920.0, speaker=0),
        SpeakerSegment(start=920.4, end=940.0, speaker=1),
    ]

    merged = _merge_adjacent(segments, CFG.min_duration_off)

    assert len(merged) == 2
    assert merged[0].start == pytest.approx(880.0)
    assert merged[0].end == pytest.approx(920.0)
    assert merged[1].speaker == 1


def test_different_speakers_across_a_join_are_not_glued():
    segments = [
        SpeakerSegment(start=880.0, end=899.9, speaker=0),
        SpeakerSegment(start=900.0, end=920.0, speaker=1),
    ]

    assert len(_merge_adjacent(segments, CFG.min_duration_off)) == 2


# --------------------------------------------------------------------------- windowed orchestration
# The whole `_diarize_windowed` path with both ONNX sessions replaced by stubs: this pins the
# wiring (per-window offsets, pooling, the global mapping, relabelling) that the pure-numpy tests
# above cannot see, without loading a model or reading real audio.

RATE = 1000
WIN_S = 60.0


class _FakeSeg:
    def __init__(self, start, end, speaker):
        self.start, self.end, self.speaker = start, end, speaker


class _FakeResult:
    def __init__(self, segs):
        self._segs = segs

    def sort_by_start_time(self):
        return sorted(self._segs, key=lambda s: s.start)


class _FakeImpl:
    """Stands in for `OfflineSpeakerDiarization`: returns the prescribed window-local segments."""

    sample_rate = RATE

    def __init__(self, per_window):
        self._per_window = list(per_window)
        self.calls = 0

    def process(self, samples, callback=None):
        segs = self._per_window[self.calls]
        self.calls += 1
        return _FakeResult([_FakeSeg(*s) for s in segs])


class _FakeStream:
    def accept_waveform(self, sample_rate, waveform):
        pass

    def input_finished(self):
        pass


class _FakeExtractor:
    """Returns the prescribed voice for each segment, in the order `_embed` visits them."""

    def __init__(self, identities):
        self._identities = list(identities)
        self.calls = 0

    def create_stream(self):
        return _FakeStream()

    def compute(self, stream):
        vec = _voice(self._identities[self.calls], jitter=0.4, rng_seed=self.calls)
        self.calls += 1
        return vec.astype(np.float32)


def _windowed(monkeypatch, tmp_path, per_window, identities, *, num_clusters=-1):
    """Drive SherpaOnnxDiarizer._diarize_windowed with stubbed models and a stubbed wav."""
    from transcription247 import diarize as dz

    n = len(per_window)
    plan = [
        dz.Window(index=i, start_frame=int(i * WIN_S * RATE),
                  end_frame=int((i + 1) * WIN_S * RATE), sample_rate=RATE)
        for i in range(n)
    ]
    wav = tmp_path / "fake.wav"
    wav.write_bytes(b"not really a wav")  # only `is_file()` is ever consulted

    impl = _FakeImpl(per_window)
    extractor = _FakeExtractor(identities)
    monkeypatch.setattr(dz, "plan_from_config", lambda *_a, **_k: plan)
    monkeypatch.setattr(dz, "read_window", lambda _w, window: np.zeros(window.frames, np.float32))
    monkeypatch.setattr(dz.SherpaOnnxDiarizer, "_build_impl", lambda self, *a: impl)
    monkeypatch.setattr(dz.SherpaOnnxDiarizer, "_build_extractor", lambda self: extractor)
    monkeypatch.setattr(dz.SherpaOnnxDiarizer, "_release_impl", lambda self: None)

    cfg = dataclasses.replace(CFG, num_clusters=num_clusters)
    d = dz.SherpaOnnxDiarizer(cfg)
    d._seg_model = tmp_path / "seg.onnx"  # skip __enter__'s model resolution
    d._emb_model = d._count_model = tmp_path / "emb.onnx"
    return d.diarize(wav, duration_s=n * WIN_S)


def test_windowed_diarization_offsets_and_links_speakers_globally(monkeypatch, tmp_path):
    # Two people. sherpa's local ids are permuted in window 1 and only one of them talks in
    # window 2 — all three things a naive per-window stitch gets wrong.
    per_window = [
        [(2.0, 25.0, 0), (26.0, 55.0, 1)],   # window 0: A then B
        [(1.0, 20.0, 0), (21.0, 58.0, 1)],   # window 1: B then A  (ids permuted)
        [(3.0, 50.0, 0)],                    # window 2: A only
    ]
    identities = [1, 2, 2, 1, 1]  # A, B, B, A, A in embedding order

    segments = _windowed(monkeypatch, tmp_path, per_window, identities)

    # Timestamps are global, not window-local.
    assert segments[0].start == pytest.approx(2.0)
    assert any(s.start == pytest.approx(61.0) for s in segments)    # window 1 + 60 s
    assert any(s.start == pytest.approx(123.0) for s in segments)   # window 2 + 120 s
    assert max(s.end for s in segments) == pytest.approx(170.0)

    # Exactly two people, and each local cluster mapped to the right one.
    assert len({s.speaker for s in segments}) == 2
    by_start = {round(s.start, 1): s.speaker for s in segments}
    assert by_start[2.0] == by_start[81.0] == by_start[123.0]  # A in all three windows
    assert by_start[26.0] == by_start[61.0]                    # B in windows 0 and 1
    assert by_start[2.0] != by_start[26.0]


def test_windowed_known_count_cuts_the_global_tree_at_k(monkeypatch, tmp_path):
    """Window 1 holds three local clusters but the caller said two people: the global cut decides,
    and the window is not forced to have exactly k of its own."""
    per_window = [
        [(2.0, 28.0, 0), (29.0, 55.0, 1)],
        [(1.0, 18.0, 0), (19.0, 36.0, 1), (37.0, 58.0, 2)],
        [(3.0, 50.0, 0)],
    ]
    identities = [1, 2, 1, 2, 1, 2]

    segments = _windowed(monkeypatch, tmp_path, per_window, identities, num_clusters=2)

    assert len({s.speaker for s in segments}) == 2


def test_windowed_merges_a_speaker_across_the_window_join(monkeypatch, tmp_path):
    """One person talking straight through the cut must come out as a single segment."""
    per_window = [[(10.0, 60.0, 0)], [(0.0, 40.0, 0)]]
    identities = [1, 1]

    segments = _windowed(monkeypatch, tmp_path, per_window, identities)

    assert len(segments) == 1
    assert segments[0].start == pytest.approx(10.0)
    assert segments[0].end == pytest.approx(100.0)

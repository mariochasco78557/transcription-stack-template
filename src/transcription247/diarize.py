"""Speaker diarization: `Diarizer` protocol + sherpa-onnx backend (ARCHITECTURE.md §3.5).

Stage 3: `transcribed -> diarized`. Produces `diarization.json`
(`dataclasses.asdict(DiarizationResult)` is exactly the §3.5 payload).

`cfg.num_clusters` is per-job and this module never knows about jobs: the worker passes
`dataclasses.replace(config.diarization, num_clusters=job.speakers_user)` when the count is known,
otherwise `config.diarization` unchanged (`-1` = auto).

Two clustering paths:

* **known count** (`num_clusters > 0`) — sherpa's `FastClustering` with a fixed k. Unchanged,
  reliable, and always preferred when the caller can supply the count.
* **auto count** (`num_clusters <= 0`) — `cfg.auto_count`:
  - `"overcluster"` (default): *over-cluster then merge*. sherpa runs with a deliberately low
    threshold so it over-splits into many small-but-pure clusters; each of those is re-embedded
    with `cfg.count_embedding_model` and pooled into one duration-weighted centroid; the
    centroids are merged bottom-up (cosine distance, average linkage) and the dendrogram is cut
    at `cfg.merge_threshold`; clusters holding less than `cfg.min_cluster_share` of the speech
    are folded into their nearest neighbour. Two things make this stable where a raw threshold
    is not: the decision is taken on tens of seconds of pooled speech per centroid instead of on
    ~1 s windows, and the pooling model is multilingual (the VoxCeleb/English embedding used for
    the clustering itself barely separates Spanish speakers — bench/RESULTS.md).
  - `"threshold"`: the previous behaviour, a single `cfg.cluster_threshold` on sherpa's own
    per-window embeddings. Kept as an escape hatch; unreliable (bench/RESULTS.md §16 D15).

**Long audio (§3.5 "Long audio").** sherpa's `process()` holds the whole recording plus its
per-frame segmentation output, ~54 MB per audio-minute, so a 2 h meeting needed ~6.6 GB. The stage
therefore runs per `windows.Window` (`pipeline.window_max_s`) and links the windows afterwards:

* every window is **over-clustered** on its own (sherpa with `num_clusters = -1`), producing
  window-local ids that mean nothing outside the window;
* each `(window, local id)` is pooled into one duration-weighted centroid with
  `cfg.count_embedding_model`;
* **one** global agglomerative merge over all those centroids (`link_windows()`) assigns the final
  speaker ids, so a label means the same person everywhere.

Only the centroids — a few hundred rows whatever the duration — survive a window; audio and
per-segment embeddings do not. A recording that fits in one window skips all of this and takes the
original code path unchanged.

Known count (`num_clusters = k`) follows the same route when there is more than one window, but
cuts the *global* tree at exactly k. Forcing k inside each window would be wrong: a speaker can be
silent for a whole window, so k local clusters would split someone else in two to fill the quota.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from transcription247.asr import release_arenas
from transcription247.config import DiarizationConfig, PipelineConfig, project_root
from transcription247.models import SCHEMA_VERSION, DiarizationResult, SpeakerSegment
from transcription247.windows import Window, plan_from_config, read_window

log = logging.getLogger(__name__)

#: Speaker-embedding models need roughly half a second of audio to produce anything useful.
_MIN_EMBED_S = 0.5
#: An over-cluster only *seeds* the speaker-count decision if it holds at least this much
#: speech; shorter ones are assigned to the nearest final cluster afterwards.
_MIN_POOL_S = 2.0


class DiarizeError(RuntimeError):
    """A diarization model is missing, unloadable, or the wav does not match the model's rate."""


@runtime_checkable
class Diarizer(Protocol):
    """Context manager that loads its models on entry and frees them on exit."""

    def __enter__(self) -> Diarizer: ...

    def __exit__(self, *exc) -> None: ...

    def diarize(self, wav: Path, duration_s: float) -> list[SpeakerSegment]: ...


def _resolve_model(raw: str, what: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = project_root() / path
    if not path.is_file():
        raise DiarizeError(f"{what} model not found: {path}")
    return path


# --------------------------------------------------------------------------- count estimation
# numpy only, no scipy/scikit-learn: the matrices here are at most a few hundred rows.


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-10)


def _agglomerative(dist: np.ndarray, weights: np.ndarray) -> list[tuple[int, int, float]]:
    """Bottom-up merge history over a cosine-distance matrix, weighted average linkage (UPGMA).

    Returns `[(kept, absorbed, distance), ...]` in merge order, with the *whole* history so a
    dendrogram cut at any distance is a cheap pass over the list.
    """
    n = dist.shape[0]
    cur = dist.astype(np.float64).copy()
    np.fill_diagonal(cur, np.inf)
    wt = weights.astype(np.float64).copy()
    active = list(range(n))
    history: list[tuple[int, int, float]] = []
    while len(active) > 1:
        sub = cur[np.ix_(active, active)]
        a, b = np.unravel_index(np.argmin(sub), sub.shape)
        i, j, d = active[a], active[b], float(sub[a, b])
        history.append((i, j, d))
        wi, wj = wt[i], wt[j]
        for k in active:
            if k not in (i, j):
                cur[i, k] = cur[k, i] = (wi * cur[i, k] + wj * cur[j, k]) / (wi + wj)
        wt[i] += wj
        active.remove(j)
        cur[j, :] = np.inf
        cur[:, j] = np.inf
    return history


def _replay(history: list[tuple[int, int, float]], n: int, take) -> np.ndarray:
    """Labels obtained by replaying the merge history while `take(step, distance)` holds."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for step, (i, j, d) in enumerate(history):
        if not take(step, d):
            break
        parent[find(j)] = find(i)
    seen: dict[int, int] = {}
    labels = np.empty(n, dtype=np.int32)
    for x in range(n):
        labels[x] = seen.setdefault(find(x), len(seen))
    return labels


def _cut(history: list[tuple[int, int, float]], n: int, threshold: float) -> np.ndarray:
    """Labels obtained by replaying the merge history up to `threshold` (union-find)."""
    return _replay(history, n, lambda _step, d: d <= threshold)


def _cut_k(history: list[tuple[int, int, float]], n: int, k: int) -> np.ndarray:
    """Labels obtained by cutting the dendrogram at exactly `k` clusters.

    The history holds `n - 1` merges, so replaying the first `n - k` of them — cheapest first,
    whatever their distance — leaves exactly `k`. This is the known-speaker-count path
    (`num_clusters = k`) applied to the *global* tree: forcing k inside every window would be
    wrong, because a speaker can be silent for a whole window.
    """
    if k >= n:
        return np.arange(n, dtype=np.int32)
    return _replay(history, n, lambda step, _d: step < n - k)


def _cluster_centroids(cent: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> dict:
    return {
        c: _l2norm((cent[labels == c] * weights[labels == c, None]).sum(axis=0))
        for c in np.unique(labels)
    }


def _merge_tiny(
    cent: np.ndarray, labels: np.ndarray, weights: np.ndarray, min_share: float
) -> np.ndarray:
    """Fold every cluster holding < `min_share` of the speech into its nearest bigger cluster.

    This is what makes slight over-splitting harmless: a spurious speaker made of a handful of
    seconds is absorbed by whoever it is closest to, while a genuine short speaker that clears
    the share stays.
    """
    labels = labels.copy()
    total = float(weights.sum()) or 1.0
    while True:
        ids = np.unique(labels)
        if len(ids) <= 1:
            return labels
        share = {int(c): float(weights[labels == c].sum()) / total for c in ids}
        small = [c for c in ids if share[int(c)] < min_share]
        if not small:
            return labels
        victim = min(small, key=lambda c: share[int(c)])
        cents = _cluster_centroids(cent, labels, weights)
        target = max(
            (c for c in ids if c != victim), key=lambda c: float(cents[victim] @ cents[c])
        )
        labels[labels == victim] = target


def _compact(labels: np.ndarray) -> np.ndarray:
    remap = {int(c): i for i, c in enumerate(np.unique(labels))}
    return np.asarray([remap[int(c)] for c in labels], dtype=np.int32)


def estimate_speakers(
    embeddings: np.ndarray,
    durations: np.ndarray,
    over_labels: np.ndarray,
    cfg: DiarizationConfig,
    *,
    force_k: int | None = None,
) -> dict[int, int]:
    """Map each over-cluster id to a final speaker id.

    `embeddings`/`durations`/`over_labels` are per *segment* (one row each). Segments of the same
    over-cluster are pooled into a single duration-weighted centroid before any distance is taken.

    The windowed caller pools per window first and passes one already-pooled row per
    `(window, local cluster)` with its speech seconds as the duration — which is arithmetically
    the same thing, because pooling a single unit vector and re-normalising is the identity. That
    is what makes one global pass over all windows' centroids produce consistent speaker ids, and
    why a single-window recording gets bit-identical results either way.

    `force_k` cuts the tree at exactly that many speakers instead of at `cfg.merge_threshold`,
    skipping the tiny-cluster absorption and the `max_speakers` cap (the caller asked for a count
    and gets it). `min_cluster_share` is deliberately evaluated on the GLOBAL speech totals, so a
    speaker who only talks in one window is judged on the whole recording, not on that window.
    """
    emb = _l2norm(embeddings.astype(np.float64))
    ids = np.unique(over_labels)
    cent = np.empty((len(ids), emb.shape[1]))
    speech = np.empty(len(ids))
    for row, cid in enumerate(ids):
        m = over_labels == cid
        speech[row] = durations[m].sum()
        cent[row] = (emb[m] * durations[m, None]).sum(axis=0) / max(speech[row], 1e-9)
    cent = _l2norm(cent)

    seed = speech >= _MIN_POOL_S
    if seed.sum() < 2:  # not enough material to argue about: trust the over-clustering
        seed = np.ones(len(ids), dtype=bool)
    if len(ids) == 1 or seed.sum() < 2:
        return {int(c): 0 for c in ids}

    csub, wsub = cent[seed], speech[seed]
    dist = np.clip(1.0 - (csub @ csub.T), 0.0, 2.0)
    history = _agglomerative(dist, wsub)
    if force_k is not None and force_k > 0:
        labels = _cut_k(history, len(csub), force_k)
    else:
        labels = _merge_tiny(
            csub, _cut(history, len(csub), cfg.merge_threshold), wsub, cfg.min_cluster_share
        )
        if len(np.unique(labels)) > cfg.max_speakers:  # keep merging until the cap is respected
            for t in sorted({d for _, _, d in history}):
                labels = _merge_tiny(csub, _cut(history, len(csub), t), wsub, cfg.min_cluster_share)
                if len(np.unique(labels)) <= cfg.max_speakers:
                    break
    labels = _compact(labels)

    mapping = {int(c): int(lab) for c, lab in zip(ids[seed], labels)}
    if (~seed).any():  # short over-clusters go to whichever final speaker they resemble most
        finals = _cluster_centroids(csub, labels, wsub)
        order = sorted(finals)
        stack = np.stack([finals[c] for c in order])
        for cid, vec in zip(ids[~seed], cent[~seed]):
            mapping[int(cid)] = int(order[int(np.argmax(stack @ vec))])
    return mapping


def _pool(vecs, durs, labels) -> list[tuple[int, np.ndarray, float]]:
    """`[(local cluster id, unit centroid, speech seconds), ...]` for one window.

    The duration-weighted mean of the L2-normalised segment embeddings, re-normalised — exactly
    what `estimate_speakers()` computes internally per over-cluster, so pooling here and pooling
    there give the same vectors (a scalar factor vanishes under the final normalisation).
    """
    if not vecs:
        return []
    emb = _l2norm(np.stack(vecs).astype(np.float64))
    weights = np.asarray(durs, dtype=np.float64)
    ids = np.asarray(labels)
    out = []
    for cid in np.unique(ids):
        m = ids == cid
        total = float(weights[m].sum())
        centroid = _l2norm((emb[m] * weights[m, None]).sum(axis=0))
        out.append((int(cid), centroid, total))
    return out


def link_windows(
    keys: list[tuple[int, int]],
    centroids: list[np.ndarray],
    speech: list[float],
    cfg: DiarizationConfig,
    *,
    force_k: int | None = None,
) -> dict[tuple[int, int], int]:
    """One global agglomerative merge over every window's pooled centroids (§3.5 "Long audio").

    `keys[i]` is the `(window index, window-local cluster id)` that `centroids[i]` was pooled
    from and `speech[i]` is how many seconds of audio went into it. The returned mapping sends
    every window-local id to a *global* speaker id, so the same person keeps one label from the
    first minute to the last.

    Clustering each window independently and then stitching by adjacency would be the obvious
    alternative and is much worse: it has no way to recognise a speaker who is silent across a
    join, and its errors compound window after window. Merging the centroids in one pass instead
    means the decision sees the whole recording, while the memory held between windows is a few
    hundred 192-float rows — flat in the duration.
    """
    if not keys:
        return {}
    cent = np.stack(centroids)
    durations = np.asarray(speech, dtype=np.float64)
    rows = np.arange(len(keys))
    mapping = estimate_speakers(cent, durations, rows, cfg, force_k=force_k)
    return {keys[i]: mapping[i] for i in rows}


def _merge_adjacent(segments: list[SpeakerSegment], max_gap: float) -> list[SpeakerSegment]:
    """Glue consecutive same-speaker segments separated by at most `max_gap` seconds.

    After relabelling, what used to be several over-clusters of one speaker is one turn again.
    """
    out: list[SpeakerSegment] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        if out and out[-1].speaker == seg.speaker and seg.start - out[-1].end <= max_gap:
            out[-1] = SpeakerSegment(
                start=out[-1].start, end=max(out[-1].end, seg.end), speaker=seg.speaker
            )
        else:
            out.append(seg)
    return out


class SherpaOnnxDiarizer:
    """`sherpa_onnx.OfflineSpeakerDiarization`: pyannote segmentation + a speaker embedding,
    clustered with FastClustering (~250 MB peak RSS, RTF ≈ 0.09 — D14).

    In `auto_count="overcluster"` mode a second `SpeakerEmbeddingExtractor`
    (`cfg.count_embedding_model`) is built *inside* `diarize()`, used to re-embed the
    over-clusters, and dropped again — it is the dominant term in that mode's peak RSS.
    """

    def __init__(self, cfg: DiarizationConfig, pipeline: PipelineConfig | None = None) -> None:
        self.cfg = cfg
        self.pipeline = pipeline or PipelineConfig()
        self._impl = None
        self._seg_model: Path | None = None
        self._emb_model: Path | None = None
        self._count_model: Path | None = None
        self._known_count = bool(cfg.num_clusters and cfg.num_clusters > 0)
        self._overcluster = not self._known_count and cfg.auto_count == "overcluster"
        self.last_plan: list[Window] = []  # filled by diarize(); persisted in diarization.json

    def __enter__(self) -> SherpaOnnxDiarizer:
        """Resolve every model path so a missing file fails before any audio is read.

        The ONNX sessions themselves are built inside `diarize()`, once the window plan is known:
        a windowed recording needs a different clustering configuration (always over-cluster, then
        link globally) than a single-window one, and building here would mean either the wrong
        configuration or two live sessions at once.
        """
        try:
            import sherpa_onnx  # noqa: F401  (imported for the clear error, used in _build_impl)
        except ImportError as exc:  # pragma: no cover - see §16 D9 (pinned sherpa-onnx-core)
            raise DiarizeError(f"sherpa_onnx is not importable: {exc}") from exc

        self._seg_model = _resolve_model(self.cfg.segmentation_model, "segmentation")
        self._emb_model = _resolve_model(self.cfg.embedding_model, "embedding")
        # Needed by the auto path always, and by the known-count path as soon as there is more
        # than one window (the global link runs on these embeddings).
        self._count_model = (
            _resolve_model(self.cfg.count_embedding_model, "count embedding")
            if self.cfg.count_embedding_model
            else self._emb_model
        )
        return self

    def __exit__(self, *exc) -> None:
        self._release_impl()
        self._seg_model = self._emb_model = self._count_model = None
        log.info("diarize: models released")

    # ----------------------------------------------------------------- model lifecycle

    def _build_impl(self, num_clusters: int, threshold: float):
        import sherpa_onnx

        t0 = time.perf_counter()
        log.info(
            "diarize: loading sherpa-onnx (num_clusters=%s, threshold=%.2f, mode=%s, %d threads)",
            num_clusters if num_clusters > 0 else "auto", threshold,
            "known" if self._known_count else self.cfg.auto_count, self.cfg.num_threads,
        )
        try:
            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=str(self._seg_model)
                    ),
                    num_threads=self.cfg.num_threads,
                    debug=False,
                    provider="cpu",
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(self._emb_model),
                    num_threads=self.cfg.num_threads,
                    debug=False,
                    provider="cpu",
                ),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=num_clusters,
                    threshold=threshold,
                ),
                min_duration_on=self.cfg.min_duration_on,
                min_duration_off=self.cfg.min_duration_off,
            )
            self._impl = sherpa_onnx.OfflineSpeakerDiarization(config)
        except Exception as exc:
            raise DiarizeError(f"could not initialise sherpa-onnx diarization: {exc}") from exc
        log.info("diarize: models loaded in %.1fs", time.perf_counter() - t0)
        return self._impl

    def _release_impl(self) -> None:
        impl, self._impl = self._impl, None
        del impl
        gc.collect()
        release_arenas()  # see asr.release_arenas(): glibc holds freed model memory otherwise

    def _build_extractor(self):
        """The `count_embedding_model` session. onnxruntime's arenas for this second model are the
        bulk of the auto path's extra RSS, so it is built as late and dropped as early as possible
        and never coexists with the segmentation session."""
        import sherpa_onnx

        return sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self._count_model),
                num_threads=self.cfg.num_threads,
                debug=False,
                provider="cpu",
            )
        )

    # ----------------------------------------------------------------- entry point

    def diarize(self, wav: Path, duration_s: float) -> list[SpeakerSegment]:
        if self._seg_model is None:
            raise DiarizeError("SherpaOnnxDiarizer.diarize() called outside its `with` block")
        wav = Path(wav)
        if not wav.is_file():
            raise DiarizeError(f"wav not found: {wav}")

        plan = plan_from_config(wav, self.pipeline)
        self.last_plan = plan
        t0 = time.perf_counter()
        if len(plan) == 1:
            segments = self._diarize_single(wav, plan[0], duration_s)
        else:
            segments = self._diarize_windowed(wav, plan, duration_s)
        total = duration_s or (plan[-1].end_s if plan else 0.0)
        elapsed = time.perf_counter() - t0
        log.info(
            "diarize: %d segments, %d speakers in %.1fs (RTF %.3f)",
            len(segments), len({s.speaker for s in segments}), elapsed,
            elapsed / total if total else 0.0,
        )
        return segments

    # ----------------------------------------------------------------- single window (unchanged)

    def _diarize_single(
        self, wav: Path, window: Window, duration_s: float
    ) -> list[SpeakerSegment]:
        """The pre-windowing code path, kept bit-for-bit: a recording shorter than
        `pipeline.window_max_s` must produce exactly the labels it always did (bench/diar_e2e.py).
        """
        num_clusters = self.cfg.num_clusters if self._known_count else -1
        threshold = (
            self.cfg.overcluster_threshold if self._overcluster else self.cfg.cluster_threshold
        )
        impl = self._build_impl(num_clusters, threshold)
        samples = None
        try:
            samples = read_window(wav, window)
            self._check_rate(wav, window.sample_rate, impl)
            total = duration_s or window.duration_s
            log.info("diarize: processing %.1f s of audio", total)
            segments = self._process(impl, wav, samples, total, offset=0.0)
            # Drop the segmentation session before the embedding one is built: the two never
            # coexist, which is what keeps this stage's peak at max(model), not the sum.
            self._release_impl()
            if not self._overcluster or len({s.speaker for s in segments}) < 2:
                return segments
            vecs, durs, labels = self._embed(segments, samples, window.sample_rate, offset=0.0)
        finally:
            self._release_impl()  # idempotent; covers the error paths above
            del samples
            gc.collect()
            release_arenas()
        return self._label(segments, vecs, durs, labels)

    def _label(self, segments, vecs, durs, labels) -> list[SpeakerSegment]:
        """Single-window auto path: per-segment embeddings -> final speakers."""
        t0 = time.perf_counter()
        over = len({s.speaker for s in segments})
        if len({*labels}) < 2:  # nothing long enough to judge: leave sherpa's answer alone
            log.info("diarize: auto-count fell back to sherpa's %d over-clusters", over)
            return segments

        mapping = estimate_speakers(
            np.stack(vecs), np.asarray(durs, dtype=np.float64), np.asarray(labels), self.cfg
        )
        default = max(set(mapping.values()), key=list(mapping.values()).count)
        relabelled = [
            SpeakerSegment(start=s.start, end=s.end, speaker=mapping.get(s.speaker, default))
            for s in segments
        ]
        merged = _merge_adjacent(relabelled, self.cfg.min_duration_off)
        log.info(
            "diarize: auto-count %d over-clusters (%d embedded) -> %d speakers in %.1fs",
            over, len(vecs), len({s.speaker for s in merged}), time.perf_counter() - t0,
        )
        return merged

    # ----------------------------------------------------------------- windowed

    def _diarize_windowed(
        self, wav: Path, plan: list[Window], duration_s: float
    ) -> list[SpeakerSegment]:
        """Per-window segmentation + over-clustering, then ONE global link over the centroids.

        Two passes over the file, never both models at once and never more than one window of
        audio in RAM:

        1. segmentation + sherpa clustering per window -> segments carrying a *window-local*
           cluster id, timestamps already shifted onto the global timeline;
        2. the count-embedding model per window -> one duration-weighted centroid per
           `(window, local id)`.

        The centroids (a few hundred float32 rows in total, regardless of the duration) are then
        merged globally, which is what makes "Speaker 2" mean the same person in minute 5 and in
        minute 115.
        """
        total = duration_s or plan[-1].end_s
        threshold = (
            self.cfg.cluster_threshold
            if (not self._known_count and self.cfg.auto_count == "threshold")
            else self.cfg.overcluster_threshold
        )
        if self._known_count:
            log.info(
                "diarize: %d windows with a known count of %d: over-clustering per window, "
                "cutting the global tree at %d (a speaker may be silent for a whole window)",
                len(plan), self.cfg.num_clusters, self.cfg.num_clusters,
            )

        # --- pass 1: segmentation ------------------------------------------------------------
        t0 = time.perf_counter()
        per_window: list[list[SpeakerSegment]] = []
        impl = self._build_impl(-1, threshold)
        try:
            self._check_rate(wav, plan[0].sample_rate, impl)
            for window in plan:
                samples = read_window(wav, window)
                try:
                    segs = self._process(impl, wav, samples, total, offset=window.start_s)
                finally:
                    del samples
                    gc.collect()
                    release_arenas()
                per_window.append(segs)
                log.info(
                    "diarize: window %d/%d [%s-%s] %d segments, %d local clusters (%.0fs elapsed)",
                    window.index + 1, len(plan), _hms(window.start_s), _hms(window.end_s),
                    len(segs), len({s.speaker for s in segs}), time.perf_counter() - t0,
                )
        finally:
            self._release_impl()

        # --- pass 2: one centroid per (window, local cluster) ---------------------------------
        keys: list[tuple[int, int]] = []
        centroids: list[np.ndarray] = []
        speech: list[float] = []
        extractor = self._build_extractor()
        try:
            for window, segs in zip(plan, per_window):
                if not segs:
                    continue
                samples = read_window(wav, window)
                try:
                    vecs, durs, labels = self._embed(
                        segs, samples, window.sample_rate, offset=window.start_s,
                        extractor=extractor,
                    )
                finally:
                    del samples
                    gc.collect()
                    release_arenas()
                for local, vec, dur in _pool(vecs, durs, labels):
                    keys.append((window.index, local))
                    centroids.append(vec)
                    speech.append(dur)
        finally:
            del extractor
            gc.collect()
            release_arenas()

        # --- global link ----------------------------------------------------------------------
        flat = [s for segs in per_window for s in segs]
        if len(centroids) < 2:
            # Nothing embeddable anywhere: one speaker is the only honest answer, and it at least
            # keeps the labels consistent (raw per-window ids would not be).
            log.warning("diarize: only %d embeddable cluster(s) across %d windows; "
                        "labelling the whole recording as one speaker", len(centroids), len(plan))
            return _merge_adjacent(
                [SpeakerSegment(start=s.start, end=s.end, speaker=0) for s in flat],
                self.cfg.min_duration_off,
            )

        mapping = link_windows(
            keys, centroids, speech, self.cfg,
            force_k=self.cfg.num_clusters if self._known_count else None,
        )
        default = max(set(mapping.values()), key=list(mapping.values()).count)
        relabelled = [
            SpeakerSegment(
                start=s.start, end=s.end,
                speaker=mapping.get((w.index, s.speaker), default),
            )
            for w, segs in zip(plan, per_window)
            for s in segs
        ]
        merged = _merge_adjacent(relabelled, self.cfg.min_duration_off)
        log.info(
            "diarize: %d windows, %d local clusters -> %d global speakers in %.1fs",
            len(plan), len(keys), len({s.speaker for s in merged}), time.perf_counter() - t0,
        )
        return merged

    # ----------------------------------------------------------------- shared helpers

    def _check_rate(self, wav: Path, sample_rate: int, impl) -> None:
        expected = impl.sample_rate
        if sample_rate != expected:
            raise DiarizeError(
                f"{wav.name} is {sample_rate} Hz but the diarization models expect {expected} Hz"
            )

    def _process(self, impl, wav: Path, samples, total: float, *, offset: float):
        t0 = time.perf_counter()
        try:
            result = impl.process(samples, callback=_make_callback(total, t0))
        except TypeError:  # older sherpa-onnx builds have no progress callback
            result = impl.process(samples)
        except Exception as exc:
            raise DiarizeError(f"diarization of {wav.name} failed: {exc}") from exc
        return [
            SpeakerSegment(
                start=float(s.start) + offset, end=float(s.end) + offset, speaker=int(s.speaker)
            )
            for s in result.sort_by_start_time()
        ]

    def _embed(self, segments, samples, sample_rate: int, *, offset: float, extractor=None):
        """Embed every segment long enough to mean something, with the count model.

        `offset` is subtracted again to index back into this window's samples, so the caller can
        keep the segments on the global timeline throughout.
        """
        own = extractor is None
        if own:
            extractor = self._build_extractor()
        min_samples = int(_MIN_EMBED_S * sample_rate)
        vecs, durs, labels = [], [], []
        try:
            for seg in segments:
                a = int(round((seg.start - offset) * sample_rate))
                b = int(round((seg.end - offset) * sample_rate))
                a, b = max(0, a), min(len(samples), b)
                if b - a < min_samples:
                    continue
                stream = extractor.create_stream()
                stream.accept_waveform(sample_rate=sample_rate, waveform=samples[a:b])
                stream.input_finished()
                vecs.append(np.asarray(extractor.compute(stream), dtype=np.float32))
                durs.append(seg.end - seg.start)
                labels.append(seg.speaker)
        finally:
            if own:
                del extractor
                gc.collect()
                release_arenas()
        return vecs, durs, labels


def _make_callback(total_s: float, t0: float):
    state = {"pct": 0}

    def callback(processed_chunks: int, total_chunks: int, arg=None) -> int:
        if total_chunks <= 0:
            return 0
        pct = int(processed_chunks * 100 / total_chunks)
        if pct >= state["pct"] + 20:
            state["pct"] = pct - pct % 20
            log.info(
                "diarize: %d%% of %.0f s, %.0fs elapsed", pct, total_s, time.perf_counter() - t0
            )
        return 0

    return callback


def _hms(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def get_diarizer(cfg: DiarizationConfig, pipeline: PipelineConfig | None = None) -> Diarizer:
    """Factory on cfg.backend ('sherpa_onnx' today, 'pyannote' reserved); else ValueError."""
    if cfg.backend == "sherpa_onnx":
        if cfg.auto_count not in ("overcluster", "threshold"):
            raise ValueError(f"unknown diarization auto_count {cfg.auto_count!r}")
        return SherpaOnnxDiarizer(cfg, pipeline)
    if cfg.backend == "pyannote":
        raise ValueError("diarization backend 'pyannote' is reserved and not implemented")
    raise ValueError(f"unknown diarization backend {cfg.backend!r}")


def run(
    wav: Path,
    cfg: DiarizationConfig,
    *,
    pipeline: PipelineConfig | None = None,
    duration_s: float = 0.0,
    job_id: str = "",
) -> DiarizationResult:
    """Load the models, diarize, release the models. The worker's single entry point."""
    d = get_diarizer(cfg, pipeline)
    with d:
        segments = d.diarize(wav, duration_s)
    return DiarizationResult(
        schema=SCHEMA_VERSION,
        job_id=job_id,
        backend=cfg.backend,
        num_speakers_detected=len({s.speaker for s in segments}),
        segments=segments,
        windows=[w.as_pair() for w in getattr(d, "last_plan", ()) or ()],
    )

#!/usr/bin/env python3
"""Diarization speaker-count sweep harness (unknown `num_clusters`).

Why this exists
---------------
`sherpa_onnx.OfflineSpeakerDiarization` with `num_clusters=-1` decides the speaker
count from a single cosine threshold on *short* per-window embeddings.  That is
very unstable (bench/RESULTS.md: 8-18 speakers at 0.5, 1 speaker at 0.9 on the
same 2-speaker clip).  This script exists to find a stable replacement and to do
it *cheaply*, by caching the two expensive stages per clip:

  stage `segs` : one sherpa run with a deliberately LOW clustering threshold, so the
                 result is a heavily over-split but *pure* partition of the speech
                 ("over-clusters").  Cached as JSON.
  stage `emb`  : per-segment speaker embeddings for those cached segments, for any
                 candidate embedding model.  Cached as .npz.
  stage `analyze` : pure numpy.  Pools segment embeddings into over-cluster centroids,
                 runs agglomerative clustering with cosine distance, cuts the
                 dendrogram at a distance threshold, merges tiny clusters, and
                 reports the predicted speaker count.  Costs nothing, so the
                 threshold / linkage / pooling grid can be swept exhaustively.

Everything heavy must be launched through the project's systemd-run cgroup wrapper
(see bench/RESULTS.md "Resource rules"); `analyze` is numpy-only and safe to run
directly.

Usage
-----
  diar_sweep.py --stage segs                       # sherpa over-split runs (heavy)
  diar_sweep.py --stage baseline                   # today's auto-threshold (heavy)
  diar_sweep.py --stage gt                         # forced-k diagnostics (heavy)
  diar_sweep.py --stage emb --models all           # embeddings per model (heavy)
  diar_sweep.py --stage analyze                    # grid search (cheap, numpy only)
  diar_sweep.py --stage report                     # final table for the chosen recipe
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "bench" / "audio"
CACHE = ROOT / "bench" / "cache"
EMB_DIR = ROOT / "models" / "sherpa-onnx-diarization" / "embedding"
SEG_MODEL = (
    ROOT / "models" / "sherpa-onnx-diarization"
    / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
)

# --- dev set -------------------------------------------------------------------
# `k` is the ground-truth speaker count *inside the extracted excerpt*.
# `certain` marks clips whose count follows from the format itself (solo reading,
# solo lecture, 1:1 interview); the rest are estimated from the source description
# and are flagged as such in bench/README.md.


@dataclass(frozen=True)
class Clip:
    name: str
    k: int
    lang: str
    certain: bool
    note: str


CLIPS: tuple[Clip, ...] = (
    Clip("en_monologo_1spk_4min", 1, "en", True, "LibriVox solo reading, studio"),
    Clip("es_conferencia_1spk_5min", 1, "es", True, "keynote lecture, room audio"),
    Clip("en_interview_2min", 2, "en", True, "1:1 radio interview"),
    Clip("es_entrevista_5min", 2, "es", True, "1:1 interview"),
    Clip("es_entrevista2_5min", 2, "es", True, "1:1 interview"),
    Clip("es_mesaredonda_5min", 2, "es", False, "TV programme, host + guest"),
    Clip("en_panel4_5min", 4, "en", False, "radio studio panel: host + 3 guests"),
    Clip("es_coloquio_6min", 5, "es", False, "room panel: moderator + 4 panellists"),
)

EMB_MODELS: dict[str, str] = {
    "wespeaker": "wespeaker_en_resnet34.onnx",
    "wespeaker_lm": "wespeaker_en_voxceleb_resnet34_LM.onnx",
    "campplus_zhen": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    "campplus_en": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    "eres2netv2": "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
    "eres2net": "eres2net_base_zh.onnx",
    "titanet": "nemo_en_titanet_large.onnx",
}

# stage-1 (over-split) settings.  SEG_EMB is the embedding sherpa itself uses; override with
# --seg-emb.  Note the useful threshold range depends on the model's cosine scale.
SEG_EMB = "wespeaker"
OVERSPLIT_THRESHOLDS = (0.05, 0.15, 0.30)
MIN_ON, MIN_OFF = 0.30, 0.50
THREADS = 4


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def wav_path(clip: str) -> Path:
    return AUDIO / f"{clip}.wav"


def read_wav(clip: str):
    import soundfile as sf

    samples, sr = sf.read(str(wav_path(clip)), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1, dtype="float32")
    return samples, sr


# ---------------------------------------------------------------- heavy stages


def _diarizer(emb_key: str, *, threshold: float, num_clusters: int = -1):
    import sherpa_onnx

    return sherpa_onnx.OfflineSpeakerDiarization(
        sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(SEG_MODEL)
                ),
                num_threads=THREADS,
                provider="cpu",
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(EMB_DIR / EMB_MODELS[emb_key]), num_threads=THREADS, provider="cpu"
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_clusters, threshold=threshold
            ),
            min_duration_on=MIN_ON,
            min_duration_off=MIN_OFF,
        )
    )


def segs_path(clip: str, thr: float) -> Path:
    return CACHE / f"segs__{clip}__{SEG_EMB}__{thr:.2f}.json"


def stage_segs(thresholds=OVERSPLIT_THRESHOLDS) -> None:
    """One sherpa run per (clip, over-split threshold); cache the raw segments."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        samples, sr = read_wav(clip.name)
        for thr in thresholds:
            out = segs_path(clip.name, thr)
            if out.exists():
                print(f"[segs] cached {out.name}")
                continue
            t0 = time.perf_counter()
            res = _diarizer(SEG_EMB, threshold=thr).process(samples)
            segs = [
                {"start": float(s.start), "end": float(s.end), "label": int(s.speaker)}
                for s in res.sort_by_start_time()
            ]
            el = time.perf_counter() - t0
            out.write_text(
                json.dumps(
                    {
                        "clip": clip.name,
                        "threshold": thr,
                        "seg_emb": SEG_EMB,
                        "duration_s": len(samples) / sr,
                        "process_s": el,
                        "segments": segs,
                    }
                )
            )
            print(
                f"[segs] {clip.name} thr={thr:.2f}: {len(segs)} segs, "
                f"{len({s['label'] for s in segs})} over-clusters, {el:.1f}s"
            )
    print(f"[segs] peak RSS {peak_rss_mb():.0f} MB")


def stage_baseline() -> None:
    """Today's behaviour: sherpa auto-threshold.  Heavy: one full run per point."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "baseline.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    for emb in ("wespeaker", "campplus_zhen"):
        for thr in (0.50, 0.70, 0.80, 0.90, 0.95):
            for clip in CLIPS:
                key = f"{clip.name}|{emb}|{thr:.2f}"
                if key in data:
                    continue
                samples, _ = read_wav(clip.name)
                t0 = time.perf_counter()
                res = _diarizer(emb, threshold=thr).process(samples)
                segs = res.sort_by_start_time()
                data[key] = {
                    "k": len({s.speaker for s in segs}),
                    "process_s": time.perf_counter() - t0,
                }
                print(f"[baseline] {key} -> {data[key]['k']} ({data[key]['process_s']:.0f}s)")
                out.write_text(json.dumps(data, indent=1))
    print(f"[baseline] peak RSS {peak_rss_mb():.0f} MB")


def stage_gt() -> None:
    """Forced-k diagnostics: per-speaker speech share for k=1..6, used to sanity-check
    the ground-truth count of the clips whose source description is ambiguous."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "gt_forced_k.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    for clip in CLIPS:
        if clip.certain:
            continue
        samples, _ = read_wav(clip.name)
        for k in range(1, 7):
            key = f"{clip.name}|{k}"
            if key in data:
                continue
            res = _diarizer(SEG_EMB, threshold=0.5, num_clusters=k).process(samples)
            per: dict[int, float] = {}
            for s in res.sort_by_start_time():
                per[s.speaker] = per.get(s.speaker, 0.0) + (s.end - s.start)
            tot = sum(per.values()) or 1.0
            shares = sorted((v / tot for v in per.values()), reverse=True)
            data[key] = [round(x, 4) for x in shares]
            print(f"[gt] {key}: {[f'{x:.1%}' for x in shares]}")
            out.write_text(json.dumps(data, indent=1))


def emb_path(clip: str, thr: float, model: str) -> Path:
    return CACHE / f"emb__{clip}__{SEG_EMB}__{thr:.2f}__{model}.npz"


def stage_emb(models: list[str], thresholds=OVERSPLIT_THRESHOLDS) -> None:
    """Per-segment embeddings for the cached over-split segments (one model at a time)."""
    import sherpa_onnx

    CACHE.mkdir(parents=True, exist_ok=True)
    for model in models:
        todo = [
            (c, t)
            for c in CLIPS
            for t in thresholds
            if segs_path(c.name, t).exists() and not emb_path(c.name, t, model).exists()
        ]
        if not todo:
            print(f"[emb] {model}: nothing to do")
            continue
        t_load = time.perf_counter()
        ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(EMB_DIR / EMB_MODELS[model]), num_threads=THREADS, provider="cpu"
            )
        )
        print(f"[emb] {model}: loaded in {time.perf_counter() - t_load:.1f}s, dim={ex.dim}")
        for clip, thr in todo:
            samples, sr = read_wav(clip.name)
            meta = json.loads(segs_path(clip.name, thr).read_text())
            vecs, starts, ends, labels = [], [], [], []
            t0 = time.perf_counter()
            for s in meta["segments"]:
                a, b = int(s["start"] * sr), int(s["end"] * sr)
                if b - a < int(0.5 * sr):  # embedding models need ~0.5 s of audio
                    continue
                st = ex.create_stream()
                st.accept_waveform(sample_rate=sr, waveform=samples[a:b])
                st.input_finished()
                vecs.append(np.asarray(ex.compute(st), dtype=np.float32))
                starts.append(s["start"])
                ends.append(s["end"])
                labels.append(s["label"])
            np.savez_compressed(
                emb_path(clip.name, thr, model),
                emb=np.stack(vecs) if vecs else np.zeros((0, 1), np.float32),
                start=np.asarray(starts, np.float32),
                end=np.asarray(ends, np.float32),
                label=np.asarray(labels, np.int32),
                embed_s=np.float32(time.perf_counter() - t0),
            )
            print(
                f"[emb] {model} {clip.name} thr={thr:.2f}: {len(vecs)} vecs "
                f"in {time.perf_counter() - t0:.1f}s"
            )
        del ex
    print(f"[emb] peak RSS {peak_rss_mb():.0f} MB")


# ------------------------------------------------------- cheap stage: analysis
# Everything below is pure numpy and is what gets ported into diarize.py.


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-10)


def pool_by_overcluster(
    emb: np.ndarray, start: np.ndarray, end: np.ndarray, label: np.ndarray, *, min_pool_s: float
):
    """Duration-weighted mean of the L2-normalised segment embeddings of each
    over-cluster.  Returns (centroids, weights_s, overcluster_ids, kept_mask_ids)."""
    e = l2norm(emb.astype(np.float64))
    dur = (end - start).astype(np.float64)
    ids = np.unique(label)
    cents, weights, keep = [], [], []
    for cid in ids:
        m = label == cid
        w = dur[m].sum()
        c = (e[m] * dur[m, None]).sum(axis=0) / max(w, 1e-9)
        cents.append(c)
        weights.append(w)
        keep.append(w >= min_pool_s)
    C = l2norm(np.stack(cents))
    return C, np.asarray(weights), ids, np.asarray(keep, dtype=bool)


def agglomerative(D: np.ndarray, w: np.ndarray, linkage: str = "average"):
    """Bottom-up clustering over the cosine-distance matrix D.

    Returns the merge history [(i, j, distance)] with cluster ids in 0..n-1 and
    the membership at every step, so a dendrogram cut at any distance is free.
    `average` = weighted UPGMA (weights = speech seconds), `complete` = max link.
    """
    n = D.shape[0]
    members = {i: [i] for i in range(n)}
    active = list(range(n))
    cur = D.astype(np.float64).copy()
    np.fill_diagonal(cur, np.inf)
    wt = w.astype(np.float64).copy()
    history = []
    while len(active) > 1:
        sub = cur[np.ix_(active, active)]
        idx = np.unravel_index(np.argmin(sub), sub.shape)
        i, j = active[idx[0]], active[idx[1]]
        d = sub[idx]
        history.append((i, j, float(d), [list(members[i]), list(members[j])]))
        # merge j into i
        if linkage == "average":
            wi, wj = wt[i], wt[j]
            for k in active:
                if k in (i, j):
                    continue
                cur[i, k] = cur[k, i] = (wi * cur[i, k] + wj * cur[j, k]) / (wi + wj)
        elif linkage == "complete":
            for k in active:
                if k in (i, j):
                    continue
                cur[i, k] = cur[k, i] = max(cur[i, k], cur[j, k])
        else:  # single
            for k in active:
                if k in (i, j):
                    continue
                cur[i, k] = cur[k, i] = min(cur[i, k], cur[j, k])
        wt[i] += wt[j]
        members[i] = members[i] + members[j]
        active.remove(j)
        cur[j, :] = np.inf
        cur[:, j] = np.inf
    return history


def cut(history, n: int, threshold: float) -> np.ndarray:
    """Labels obtained by stopping the merge history at `threshold`."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, d, _ in history:
        if d > threshold:
            break
        parent[find(j)] = find(i)
    roots = {}
    lab = np.empty(n, dtype=np.int32)
    for x in range(n):
        r = find(x)
        lab[x] = roots.setdefault(r, len(roots))
    return lab


def merge_tiny(C: np.ndarray, lab: np.ndarray, w: np.ndarray, *, min_share: float) -> np.ndarray:
    """Fold clusters holding < min_share of total speech into the nearest big cluster."""
    lab = lab.copy()
    total = w.sum()
    while True:
        ids = np.unique(lab)
        if len(ids) <= 1:
            return lab
        shares = {c: w[lab == c].sum() / total for c in ids}
        small = [c for c in ids if shares[c] < min_share]
        if not small:
            return lab
        victim = min(small, key=lambda c: shares[c])
        big = [c for c in ids if c != victim]
        cents = {c: l2norm((C[lab == c] * w[lab == c, None]).sum(0)) for c in ids}
        target = max(big, key=lambda c: float(cents[victim] @ cents[c]))
        lab[lab == victim] = target


def within_scale(emb: np.ndarray, start, end, label) -> float:
    """Median cosine distance between two segments of the *same* over-cluster.

    An over-cluster is (by construction) one speaker in one recording, so this is a per-recording
    estimate of "how far apart can the same voice look here" — it absorbs the channel, the room
    and the model's own dynamic range.  Used to turn the absolute merge threshold into a
    scale-free multiple, so one default can fit iPhone-in-a-room and studio audio alike.
    """
    e = l2norm(emb.astype(np.float64))
    vals = []
    for cid in np.unique(label):
        idx = np.flatnonzero(label == cid)
        if len(idx) < 2:
            continue
        sub = 1.0 - (e[idx] @ e[idx].T)
        vals.extend(sub[np.triu_indices(len(idx), k=1)].tolist())
    return float(np.median(vals)) if vals else float("nan")


def predict_k(
    emb, start, end, label, *, min_pool_s, threshold, linkage, min_share, max_k, scale="abs"
) -> tuple[int, np.ndarray, np.ndarray]:
    """The candidate recipe.  Returns (k, over-cluster ids, final label per over-cluster)."""
    if scale == "rel":
        s = within_scale(emb, start, end, label)
        threshold = threshold * s if s == s else threshold
    C, w, ids, keep = pool_by_overcluster(emb, start, end, label, min_pool_s=min_pool_s)
    if keep.sum() < 2:
        keep = np.ones_like(keep)
    Ck, wk = C[keep], w[keep]
    if len(Ck) == 1:
        lab_k = np.zeros(1, np.int32)
    else:
        D = 1.0 - (Ck @ Ck.T)
        np.clip(D, 0.0, 2.0, out=D)
        hist = agglomerative(D, wk, linkage)
        lab_k = cut(hist, len(Ck), threshold)
        lab_k = merge_tiny(Ck, lab_k, wk, min_share=min_share)
        if len(np.unique(lab_k)) > max_k:  # force down to max_k by continuing the merge
            lab_k = cut(hist, len(Ck), np.inf)
            for t in sorted({h[2] for h in hist}):
                lab_k = merge_tiny(Ck, cut(hist, len(Ck), t), wk, min_share=min_share)
                if len(np.unique(lab_k)) <= max_k:
                    break
    # relabel compactly and assign the dropped (too short) over-clusters
    uniq = {c: i for i, c in enumerate(np.unique(lab_k))}
    lab_k = np.asarray([uniq[c] for c in lab_k], np.int32)
    final = np.empty(len(C), np.int32)
    final[keep] = lab_k
    if (~keep).any():
        cents = np.stack(
            [l2norm((C[keep][lab_k == c] * wk[lab_k == c, None]).sum(0)) for c in range(len(uniq))]
        )
        final[~keep] = np.argmax(C[~keep] @ cents.T, axis=1)
    return len(uniq), ids, final


def load(clip: str, thr: float, model: str):
    z = np.load(emb_path(clip, thr, model))
    return z["emb"], z["start"], z["end"], z["label"]


def stage_analyze(models: list[str], thresholds, out_json: Path | None) -> None:
    # cosine-distance scale differs a lot between embedding models, so the grid has to be wide
    grid_t = [round(x, 3) for x in np.arange(0.05, 1.21, 0.025)]
    rows = []
    for model in models:
        for thr in thresholds:
            if not all(emb_path(c.name, thr, model).exists() for c in CLIPS):
                continue
            data = {c.name: load(c.name, thr, model) for c in CLIPS}
            for linkage in ("average", "complete"):
                for min_share in (0.0, 0.03, 0.05):
                    for t in grid_t:
                        preds, err, exact = {}, 0, 0
                        for c in CLIPS:
                            k, _, _ = predict_k(
                                *data[c.name], min_pool_s=2.0, threshold=t,
                                linkage=linkage, min_share=min_share, max_k=8,
                            )
                            preds[c.name] = k
                            err += abs(k - c.k)
                            exact += int(k == c.k)
                        rows.append(
                            {
                                "model": model, "overs_thr": thr, "linkage": linkage,
                                "min_share": min_share, "t": t, "exact": exact,
                                "abs_err": err,
                                "max_err": max(abs(preds[c.name] - c.k) for c in CLIPS),
                                "certain_exact": sum(
                                    preds[c.name] == c.k for c in CLIPS if c.certain
                                ),
                                "preds": preds,
                            }
                        )
    rows.sort(key=lambda r: (-r["exact"], r["abs_err"], r["max_err"]))
    print(f"{'model':<14}{'ovr':>5}{'link':>9}{'share':>7}{'t':>7}{'ok':>4}{'|err|':>6}{'max':>5}  preds")
    for r in rows[:40]:
        p = " ".join(str(r["preds"][c.name]) for c in CLIPS)
        print(
            f"{r['model']:<14}{r['overs_thr']:>5.2f}{r['linkage']:>9}{r['min_share']:>7.2f}"
            f"{r['t']:>7.3f}{r['exact']:>4}{r['abs_err']:>6}{r['max_err']:>5}  {p}"
        )
    print("\nclip order: " + " ".join(f"{c.name}(k={c.k})" for c in CLIPS))
    if out_json:
        out_json.write_text(json.dumps(rows, indent=1))
        print(f"\nfull grid -> {out_json}")


def stage_plateau(model: str, thr: float, linkage: str, min_share: float) -> None:
    """For the chosen recipe, print the predicted k per clip across the whole
    threshold grid, so the width of the 'all correct' plateau is visible."""
    data = {c.name: load(c.name, thr, model) for c in CLIPS}
    print(f"recipe: {model} over-split={thr} linkage={linkage} min_share={min_share}")
    print("t      " + "".join(f"{c.name[:12]:>14}" for c in CLIPS) + "   ok")
    print("true   " + "".join(f"{c.k:>14}" for c in CLIPS))
    for t in [round(x, 3) for x in np.arange(0.05, 1.21, 0.025)]:
        ks = []
        for c in CLIPS:
            k, _, _ = predict_k(
                *data[c.name], min_pool_s=2.0, threshold=t, linkage=linkage,
                min_share=min_share, max_k=8,
            )
            ks.append(k)
        ok = sum(k == c.k for k, c in zip(ks, CLIPS))
        print(f"{t:<7.3f}" + "".join(f"{k:>14}" for k in ks) + f"   {ok}/{len(CLIPS)}")


def main() -> int:
    global SEG_EMB
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["segs", "baseline", "gt", "emb", "analyze", "plateau"])
    ap.add_argument("--models", default="wespeaker")
    ap.add_argument("--seg-emb", default=SEG_EMB, help="embedding used by the sherpa over-split pass")
    ap.add_argument("--thresholds", default="")
    ap.add_argument("--linkage", default="average")
    ap.add_argument("--min-share", type=float, default=0.03)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    SEG_EMB = a.seg_emb
    models = list(EMB_MODELS) if a.models == "all" else a.models.split(",")
    thrs = (
        tuple(float(x) for x in a.thresholds.split(",")) if a.thresholds else OVERSPLIT_THRESHOLDS
    )

    if a.stage == "segs":
        stage_segs(thrs)
    elif a.stage == "baseline":
        stage_baseline()
    elif a.stage == "gt":
        stage_gt()
    elif a.stage == "emb":
        stage_emb(models, thrs)
    elif a.stage == "analyze":
        stage_analyze(models, thrs, Path(a.out) if a.out else None)
    elif a.stage == "plateau":
        stage_plateau(models[0], thrs[0], a.linkage, a.min_share)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Long-audio memory benchmark: the real stages, on 20/60/120-minute recordings.

Runs `normalize -> asr -> release -> diarize -> merge` exactly as `worker.py` does — same config,
same artifacts, same stage isolation — with no summary call and no Drive delivery, against roots
pointed at a scratch directory. What it measures per stage:

  wall     seconds spent in the stage
  RTF      wall / audio seconds
  VmHWM    the *process* high-water mark, sampled before and after the stage. Because glibc holds
           freed model memory in its arenas (see asr.release_arenas()), VmHWM only ever goes up,
           so the per-stage number reported is the increment this stage added to the high-water
           mark, and the final figure is the one that matters against MemoryHigh.

The cgroup's own `MemoryPeak` is printed by the systemd-run wrapper that must launch this:

  systemd-run --user --collect --wait --pipe -p Nice=19 -p CPUSchedulingPolicy=idle \
    -p IOSchedulingClass=idle -p CPUQuota=400% -p MemoryMax=3G -p MemorySwapMax=0 \
    -p OOMScoreAdjust=500 --working-directory="<project>" .venv/bin/python bench/long_audio.py ...

Never run two of these at once, and never outside the cgroup (bench/RESULTS.md, resource rules).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcription247 import asr, audio, diarize, merge, paths  # noqa: E402
from transcription247.config import load_config  # noqa: E402


def vmhwm_mb() -> float:
    """Peak resident set of this process, in MB, straight from /proc (never decreases)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / 1024.0
    return 0.0


def vmrss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return 0.0


class Stages:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def run(self, name: str, audio_s: float, fn):
        before_hwm, before_rss = vmhwm_mb(), vmrss_mb()
        t0 = time.perf_counter()
        out = fn()
        wall = time.perf_counter() - t0
        after_hwm, after_rss = vmhwm_mb(), vmrss_mb()
        row = {
            "stage": name,
            "wall_s": round(wall, 1),
            "rtf": round(wall / audio_s, 4) if audio_s else None,
            "vmhwm_before_mb": round(before_hwm),
            "vmhwm_after_mb": round(after_hwm),
            "added_to_peak_mb": round(max(0.0, after_hwm - before_hwm)),
            "rss_before_mb": round(before_rss),
            "rss_after_mb": round(after_rss),
        }
        self.rows.append(row)
        print(
            f"  {name:<10} {wall:7.1f}s  RTF {row['rtf']}  "
            f"VmHWM {before_hwm:.0f}->{after_hwm:.0f} MB  RSS {before_rss:.0f}->{after_rss:.0f} MB",
            flush=True,
        )
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="source audio (already 16 kHz mono or not)")
    ap.add_argument("--root", required=True, help="scratch directory for the job artifacts")
    ap.add_argument("--label", default="")
    ap.add_argument("--speakers", type=int, default=0, help="0 = auto count")
    ap.add_argument("--window-max-s", type=float, default=-1.0, help="-1 = config.toml value")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = load_config()
    if a.window_max_s >= 0:
        cfg = dataclasses.replace(
            cfg, pipeline=dataclasses.replace(cfg.pipeline, window_max_s=a.window_max_s)
        )
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    # Every root points inside the scratch directory: this benchmark must not be able to write to
    # Drive, to the inbox, or to the real data dir, whatever a stage decides to do.
    for sub in ("out", "inbox", "data"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    paths.set_roots(
        output_root=root / "out", inbox_root=root / "inbox", data_root=root / "data"
    )

    src = Path(a.wav)
    label = a.label or src.stem
    print(f"\n=== {label}: {src.name} ({src.stat().st_size / 2**20:.0f} MiB), "
          f"window_max_s={cfg.pipeline.window_max_s} ===", flush=True)

    stages = Stages()
    wav = root / "data" / "audio.wav"

    normalized = stages.run("normalize", 1.0, lambda: audio.normalize(
        src, wav, cfg=cfg.audio, job_id=label, source_name=src.name,
        source_sha256="0" * 64, recorded_at_user=None, tz=cfg.general.zona_horaria,
    ))
    audio_s = normalized.duration_s
    stages.rows[-1]["rtf"] = round(stages.rows[-1]["wall_s"] / audio_s, 4)
    print(f"  audio: {audio_s:.1f}s ({audio_s / 60:.1f} min)", flush=True)

    asr_result = stages.run("asr", audio_s, lambda: asr.run(
        Path(normalized.wav_path), cfg.asr, pipeline=cfg.pipeline, job_id=label,
        duration_s=audio_s,
        allowed_languages=tuple(cfg.general.idiomas_permitidos),
        fallback_language=cfg.general.idioma_fallback,
        partial_path=root / asr.PARTIAL_NAME,
    ))
    stages.run("release", audio_s, asr.release_arenas)

    diar_cfg = cfg.diarization
    if a.speakers > 0:
        diar_cfg = dataclasses.replace(diar_cfg, num_clusters=a.speakers)
    diar = stages.run("diarize", audio_s, lambda: diarize.run(
        Path(normalized.wav_path), diar_cfg, pipeline=cfg.pipeline,
        duration_s=audio_s, job_id=label,
    ))
    turns = stages.run("merge", audio_s, lambda: merge.merge(
        asr_result, diar.segments, cfg.merge
    ))

    summary = {
        "label": label,
        "wav": str(src),
        "audio_s": round(audio_s, 1),
        "audio_min": round(audio_s / 60.0, 1),
        "window_max_s": cfg.pipeline.window_max_s,
        "windows": asr_result.windows,
        "n_windows": len(asr_result.windows),
        "speakers_requested": a.speakers or "auto",
        "speakers_found": diar.num_speakers_detected,
        "asr_segments": len(asr_result.segments),
        "asr_words": len(asr_result.words),
        "language": asr_result.language,
        "diar_segments": len(diar.segments),
        "turns": len(turns.turns),
        "participantes": turns.participantes,
        "peak_vmhwm_mb": round(vmhwm_mb()),
        "total_wall_s": round(sum(r["wall_s"] for r in stages.rows), 1),
        "total_rtf": round(sum(r["wall_s"] for r in stages.rows) / audio_s, 4),
        "stages": stages.rows,
        "pid": os.getpid(),
    }
    # Which speaker holds which slice of the timeline, to eyeball consistency across joins.
    per_speaker: dict[int, float] = {}
    for s in diar.segments:
        per_speaker[s.speaker] = per_speaker.get(s.speaker, 0.0) + (s.end - s.start)
    summary["speech_per_speaker_s"] = {str(k): round(v, 1) for k, v in sorted(per_speaker.items())}
    # First and last appearance of every speaker: a label that only exists in one window is the
    # signature of a linking failure.
    span: dict[int, list[float]] = {}
    for s in diar.segments:
        cur = span.setdefault(s.speaker, [s.start, s.end])
        cur[0], cur[1] = min(cur[0], s.start), max(cur[1], s.end)
    summary["speaker_span_s"] = {str(k): [round(v[0], 1), round(v[1], 1)] for k, v in sorted(span.items())}

    print(json.dumps({k: v for k, v in summary.items() if k != "stages"}, indent=1), flush=True)
    print(f"\nTOTAL {summary['total_wall_s']:.0f}s for {audio_s / 60:.0f} min "
          f"(RTF {summary['total_rtf']}), peak VmHWM {summary['peak_vmhwm_mb']} MB", flush=True)

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

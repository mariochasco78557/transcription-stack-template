#!/usr/bin/env python3
"""End-to-end check of `transcription247.diarize.run()` on the dev set.

Runs the *shipped* code (not the sweep's re-implementation) over every clip in three
configurations and reports speaker count, wall time, RTF and peak RSS:

  known     num_clusters = ground-truth k         (the reference path, unchanged)
  auto-new  num_clusters = -1, auto_count="overcluster"   (the new default)
  auto-old  num_clusters = -1, auto_count="threshold"     (what config.toml shipped before)

One clip per subprocess is *not* used here: the models are released between clips by
`run()` itself, and `ru_maxrss` is reported as the high-water mark of the whole process,
so the number below is the worst case across the whole sweep, not a per-clip figure.

Must be launched through the systemd-run cgroup wrapper (bench/RESULTS.md, resource rules).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diar_sweep import CLIPS, wav_path  # noqa: E402

from transcription247.config import load_config  # noqa: E402
from transcription247.diarize import run as diarize_run  # noqa: E402


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="known,auto-new,auto-old")
    ap.add_argument("--out", default="bench/results/auto_count_e2e.json")
    a = ap.parse_args()

    base = load_config().diarization
    rows = []
    for mode in a.modes.split(","):
        for clip in CLIPS:
            if mode == "known":
                cfg = dataclasses.replace(base, num_clusters=clip.k)
            elif mode == "auto-new":
                cfg = dataclasses.replace(base, num_clusters=-1, auto_count="overcluster")
            else:
                cfg = dataclasses.replace(base, num_clusters=-1, auto_count="threshold")
            wav = wav_path(clip.name)
            rss0 = peak_rss_mb()
            t0 = time.perf_counter()
            res = diarize_run(wav, cfg, job_id="bench")
            el = time.perf_counter() - t0
            dur = max((s.end for s in res.segments), default=0.0)
            rows.append(
                {
                    "mode": mode, "clip": clip.name, "true_k": clip.k,
                    "pred_k": res.num_speakers_detected, "segments": len(res.segments),
                    "wall_s": round(el, 2), "rtf": round(el / dur, 4) if dur else None,
                    "peak_rss_mb_so_far": round(max(rss0, peak_rss_mb())),
                }
            )
            print(
                f"{mode:<9} {clip.name:<26} true={clip.k} pred={res.num_speakers_detected} "
                f"segs={len(res.segments):>3} {el:6.1f}s rss<={peak_rss_mb():.0f}MB",
                flush=True,
            )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"peak_rss_mb": round(peak_rss_mb()), "rows": rows}, indent=1))
    print(f"\npeak RSS overall: {peak_rss_mb():.0f} MB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

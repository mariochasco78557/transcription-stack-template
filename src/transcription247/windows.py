"""Window planning for long recordings (ARCHITECTURE.md §3.4/§3.5).

Both heavy stages used to hand the *whole* decoded recording to their backend, so peak RSS grew
linearly with the duration (faster-whisper's VAD ≈10 MB per audio-minute, sherpa ≈54 MB per
audio-minute — bench/RESULTS.md "Long audio"). This module cuts the normalized 16 kHz mono wav
into contiguous, non-overlapping windows of at most `pipeline.window_max_s`, so each backend only
ever sees one window and the peak becomes flat in the duration.

Two properties the rest of the pipeline relies on:

* **Deterministic.** `plan_windows()` is a pure function of (wav bytes, window settings). ASR and
  diarization plan independently and get the same cuts, and a resumed job re-derives the same plan.
* **Contiguous.** `windows[i].end_frame == windows[i + 1].start_frame`, the first starts at 0 and
  the last ends at the last frame, so offsetting every window-local timestamp by `start_s`
  reconstructs the original timeline exactly.

Cuts land on the longest low-energy pause found in the last `window_cut_search_s` seconds of the
window (a block-wise RMS scan that reads ~90 s at a time — never the whole file); when the tail is
continuous speech there is no good pause and the cut is hard, at exactly `window_max_s`.
A recording shorter than `window_max_s` yields exactly one window, which is how every short clip
keeps its pre-windowing behaviour bit for bit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: RMS is computed over blocks of this many milliseconds. 20 ms is short enough to resolve the
#: edges of a pause and long enough that the scan of a 90 s region is a few thousand rows.
_BLOCK_MS = 20.0
#: A candidate pause must be at least this long to be preferred over a hard cut.
_MIN_PAUSE_S = 0.30
#: Samples are read in chunks of at most this many seconds during the RMS scan.
_SCAN_CHUNK_S = 30.0


class WindowError(RuntimeError):
    """The wav could not be inspected or the window settings are nonsense."""


@dataclass(frozen=True, slots=True)
class Window:
    """One contiguous slice of the normalized wav. Frames are the unit of truth; the seconds are
    derived, so no rounding can ever make two windows overlap or leave a gap."""

    index: int
    start_frame: int
    end_frame: int
    sample_rate: int

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def start_s(self) -> float:
        return self.start_frame / float(self.sample_rate)

    @property
    def end_s(self) -> float:
        return self.end_frame / float(self.sample_rate)

    @property
    def duration_s(self) -> float:
        return self.frames / float(self.sample_rate)

    def as_pair(self) -> list[float]:
        """`[start_s, end_s]`, the form persisted in asr.json / diarization.json."""
        return [round(self.start_s, 3), round(self.end_s, 3)]


def plan_windows(
    wav: Path,
    *,
    max_s: float = 900.0,
    search_s: float = 90.0,
    min_s: float = 60.0,
    hard_cut_only: bool = False,
) -> list[Window]:
    """Split `wav` into windows of at most `max_s` seconds, cutting at pauses where possible.

    `min_s` is the shortest window the planner will produce: it bounds the pause search away from
    the start of the window and stops a trailing sliver from becoming its own window (the tail is
    absorbed into the previous one instead, which may therefore slightly exceed `max_s`).
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment problem
        raise WindowError(f"soundfile is not importable: {exc}") from exc

    wav = Path(wav)
    if not wav.is_file():
        raise WindowError(f"wav not found: {wav}")
    if max_s <= 0:
        raise WindowError(f"window_max_s must be > 0, got {max_s}")

    try:
        info = sf.info(str(wav))
    except Exception as exc:
        raise WindowError(f"could not read the header of {wav.name}: {exc}") from exc

    rate = int(info.samplerate)
    total = int(info.frames)
    if total <= 0:
        raise WindowError(f"{wav.name} has no frames")

    max_frames = max(1, int(round(max_s * rate)))
    min_frames = max(1, int(round(min(min_s, max_s) * rate)))
    search_frames = max(0, int(round(search_s * rate)))

    # The common case by a wide margin: one window, no scan, behaviour identical to pre-windowing.
    if total <= max_frames:
        return [Window(index=0, start_frame=0, end_frame=total, sample_rate=rate)]

    bounds: list[int] = [0]
    start = 0
    while total - start > max_frames:
        target = start + max_frames
        if total - target < min_frames:
            # Cutting here would leave a sliver of a final window, which is worse for both stages
            # (a few seconds of audio is not enough to detect a language or seed a speaker) than
            # one window slightly over `max_s`. Absorb the tail and stop.
            break
        # Leave at least `min_frames` of window before the earliest acceptable cut.
        lo = max(start + min_frames, target - search_frames)
        hi = target
        cut = target
        if not hard_cut_only and hi > lo:
            found = _longest_pause(sf, wav, lo, hi, rate)
            if found is not None:
                cut = found
            else:
                log.info(
                    "windows: no pause in %.0f-%.0f s; hard cut at %.0f s",
                    lo / rate, hi / rate, target / rate,
                )
        if cut <= start:  # defensive: never make an empty or backwards window
            cut = target
        bounds.append(cut)
        start = cut
    bounds.append(total)

    out = [
        Window(index=i, start_frame=a, end_frame=b, sample_rate=rate)
        for i, (a, b) in enumerate(zip(bounds, bounds[1:]))
    ]
    log.info(
        "windows: %s -> %d window(s) of %s (max %.0f s)",
        wav.name, len(out), ", ".join(f"{w.duration_s:.0f}s" for w in out), max_s,
    )
    return out


def plan_from_config(wav: Path, pipeline) -> list[Window]:
    """The single place both heavy stages turn `PipelineConfig` into a plan.

    Having one function guarantees ASR and diarization cut at exactly the same frames, which is
    what lets a resumed job re-derive the plan instead of having to trust a persisted copy.
    `window_max_s = 0` disables windowing entirely (one window over the whole file).
    """
    if pipeline is None or pipeline.window_max_s <= 0:
        return whole_file(wav)
    return plan_windows(
        wav,
        max_s=pipeline.window_max_s,
        search_s=pipeline.window_cut_search_s,
        min_s=pipeline.window_min_s,
    )


def whole_file(wav: Path) -> list[Window]:
    """A single window spanning the whole file (windowing disabled)."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment problem
        raise WindowError(f"soundfile is not importable: {exc}") from exc
    wav = Path(wav)
    if not wav.is_file():
        raise WindowError(f"wav not found: {wav}")
    info = sf.info(str(wav))
    return [
        Window(index=0, start_frame=0, end_frame=int(info.frames), sample_rate=int(info.samplerate))
    ]


def read_window(wav: Path, window: Window) -> np.ndarray:
    """Read exactly one window's samples as float32 mono. Never reads the rest of the file."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment problem
        raise WindowError(f"soundfile is not importable: {exc}") from exc

    samples, rate = sf.read(
        str(wav),
        start=window.start_frame,
        frames=window.frames,
        dtype="float32",
        always_2d=False,
    )
    if samples.ndim > 1:  # defensive: audio.normalize() always writes mono
        samples = samples.mean(axis=1, dtype="float32")
    if rate != window.sample_rate:  # pragma: no cover - same file the plan was built from
        raise WindowError(f"{wav.name} changed sample rate between planning and reading")
    return np.ascontiguousarray(samples, dtype=np.float32)


def _longest_pause(sf, wav: Path, lo: int, hi: int, rate: int) -> int | None:
    """Frame index of the middle of the longest low-energy run in `[lo, hi)`, or None.

    Energy is read in `_SCAN_CHUNK_S` chunks so the scan's own footprint is ~2 MB regardless of
    how wide the search region is.
    """
    block = max(1, int(round(_BLOCK_MS / 1000.0 * rate)))
    rms = _block_rms(sf, wav, lo, hi, block)
    if rms.size < 2:
        return None

    if float(rms.max()) <= 0.0:  # pure digital silence: cut in the middle of the region
        return lo + (hi - lo) // 2
    # A pause has to be *meaningfully quieter than the speech around it*, which is why the
    # threshold is bounded on both sides:
    #   - `floor * 2` adapts to the recording's own noise floor instead of an absolute level;
    #   - `loud * 0.10` is the lower bound, so a very clean floor still admits real pauses;
    #   - `loud * 0.35` is the upper bound, and it is the important one: without it a region of
    #     uniform, continuous speech would have floor ≈ loud, every block would land under the
    #     threshold, and the "longest pause" would be the whole region — cutting mid-word. With
    #     it, continuous speech yields no candidate at all and the caller falls back to a hard cut.
    loud = float(np.percentile(rms, 75))
    floor = float(np.percentile(rms, 5))
    threshold = max(min(max(floor * 2.0, loud * 0.10), loud * 0.35), 1e-7)
    quiet = rms < threshold
    best_len, best_start = _longest_run(quiet)
    if best_len * block < _MIN_PAUSE_S * rate:
        return None
    middle = lo + (best_start + best_len // 2) * block
    return int(min(max(middle, lo + 1), hi - 1))


def _block_rms(sf, wav: Path, lo: int, hi: int, block: int) -> np.ndarray:
    """Block-wise RMS of `[lo, hi)`, read in bounded chunks."""
    out: list[np.ndarray] = []
    chunk_frames = max(block, int(_SCAN_CHUNK_S * block * 1000.0 / _BLOCK_MS))
    pos = lo
    with sf.SoundFile(str(wav)) as fh:
        while pos < hi:
            fh.seek(pos)
            want = min(chunk_frames, hi - pos)
            data = fh.read(frames=want, dtype="float32", always_2d=False)
            if data.size == 0:  # pragma: no cover - short read at EOF
                break
            if data.ndim > 1:
                data = data.mean(axis=1, dtype="float32")
            usable = (data.size // block) * block
            if usable:
                rows = data[:usable].reshape(-1, block)
                out.append(np.sqrt(np.mean(np.square(rows, dtype=np.float32), axis=1)))
            pos += data.size
            del data
    if not out:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(out)


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(length, start index) of the longest run of True in `mask`; (0, 0) when there is none."""
    if not mask.any():
        return 0, 0
    # Run-length encode via the positions where the value changes.
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    starts = np.concatenate(([0], edges + 1))
    ends = np.concatenate((edges + 1, [mask.size]))
    keep = mask[starts]
    starts, ends = starts[keep], ends[keep]
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(lengths[best]), int(starts[best])

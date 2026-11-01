"""faster-whisper large-v3-turbo (int8, CPU) wrapper (ARCHITECTURE.md §3.4).

Stage 2: `normalized -> transcribed`. Produces `asr.json`
(`dataclasses.asdict(AsrResult)` is exactly the §3.4 payload).

Heaviest stage on the box: ~1.66 GB peak RSS, RTF ≈ 0.27 at beam=5 (bench/RESULTS.md, D14).
The model is loaded in `__enter__` and genuinely released in `__exit__` (`del` + `gc.collect()`)
so the diarizer never coexists with it — peak RAM is max(stage), not the sum (§1).

**Long audio (§3.4 "Long audio").** `WhisperModel.transcribe()` runs its VAD eagerly over the
*entire* array before decoding anything, at ~10 MB per audio-minute, on top of the decoded samples
themselves — so a 60-min wav was OOM-killed 17 s in. The model is therefore loaded once and fed one
`windows.Window` at a time (`pipeline.window_max_s`, default 15 min): each window's samples are
read from disk on demand, every timestamp is shifted by the window's start, and the buffers are
dropped before the next window. Peak RSS is flat in the recording's duration; `asr.json` is
unchanged apart from the additive `windows` field.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from transcription247.config import AsrConfig, PipelineConfig
from transcription247.models import SCHEMA_VERSION, AsrResult, AsrSegment, Word
from transcription247.windows import Window, plan_from_config, read_window

log = logging.getLogger(__name__)

#: Display name written to asr.json's `model` field, suffixed with the compute type.
MODEL_FAMILY = "faster-whisper-large-v3-turbo"

#: `error` value the worker stores when the ASR found no speech at all (§13 #7).
NO_SPEECH_ERROR = "no_speech"

#: Progress is logged at most once per this many seconds of *audio* consumed.
PROGRESS_EVERY_S = 60.0

#: Type of the optional progress callback: (processed_audio_s, total_audio_s, elapsed_wall_s).
#: The worker uses it to renew the job lease, so it is called at least once per window.
ProgressFn = Callable[[float, float, float], None]

#: Name of the per-window resume file written next to `asr.json` while the stage is running.
PARTIAL_NAME = "asr.partial.json"


class AsrError(RuntimeError):
    """The model could not be loaded or the transcription failed."""


def release_arenas() -> None:
    """Return freed heap back to the OS after a model is dropped.

    `del` + `gc.collect()` frees the ~1.6 GB the CTranslate2 model used, but glibc keeps it in its
    malloc arenas, so the process RSS stays high and the next stage looks far heavier than it is
    (measured: 1045 MB resident after the ASR model was released). `malloc_trim(0)` hands it back.
    Best-effort: a no-op on any libc that does not export it.
    """
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # pragma: no cover - platform dependent
        pass


class _PartialState:
    """Everything a windowed transcription carries from one window to the next.

    Persisted to `asr.partial.json` after each window when the caller asks for it, so a worker
    killed 90 minutes into a 2 h recording restarts at the next window rather than at zero. The
    file is only ever trusted when its window plan is byte-identical to the one just computed —
    a config change or a re-normalized wav therefore silently starts over instead of splicing
    two incompatible timelines together.
    """

    __slots__ = ("segments", "words", "language", "language_probability", "language_forced",
                 "done", "next_mark")

    def __init__(self) -> None:
        self.segments: list[AsrSegment] = []
        self.words: list[Word] = []
        self.language: str = ""
        self.language_probability: float = 0.0
        self.language_forced: bool = False
        self.done: int = 0
        self.next_mark: float = PROGRESS_EVERY_S

    @classmethod
    def empty(cls) -> _PartialState:
        return cls()

    @classmethod
    def load(cls, path: Path, plan: list[Window]) -> _PartialState | None:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("schema") != SCHEMA_VERSION:
            return None
        if data.get("windows") != [w.as_pair() for w in plan]:
            log.info("asr: ignoring %s, its window plan does not match", Path(path).name)
            return None
        done = int(data.get("done", 0))
        if not 0 <= done <= len(plan):
            return None
        try:
            state = cls()
            state.segments = [AsrSegment(**s) for s in data.get("segments", [])]
            state.words = [Word(**w) for w in data.get("words", [])]
        except TypeError:
            return None
        state.language = str(data.get("language", "") or "")
        state.language_probability = float(data.get("language_probability", 0.0) or 0.0)
        state.language_forced = bool(data.get("language_forced", False))
        state.done = done
        state.next_mark = float(plan[done - 1].end_s) if done else PROGRESS_EVERY_S
        return state

    def save(self, path: Path, plan: list[Window]) -> None:
        """Atomic rewrite. Best-effort: losing the resume file must never fail the stage."""
        payload = {
            "schema": SCHEMA_VERSION,
            "windows": [w.as_pair() for w in plan],
            "done": self.done,
            "language": self.language,
            "language_probability": self.language_probability,
            "language_forced": self.language_forced,
            "segments": [asdict(s) for s in self.segments],
            "words": [asdict(w) for w in self.words],
        }
        target = Path(path)
        tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, target)
        except OSError as exc:  # pragma: no cover - disk problem, not logic
            log.warning("asr: could not write %s: %s", target.name, exc)
            tmp.unlink(missing_ok=True)


class Transcriber:
    """Context manager owning exactly one `WhisperModel`.

    `__init__` does NOT load the model, so the worker can build the object cheaply (e.g. for a
    pre-flight RAM check) and only pay the ~1.7 GB when it enters the block.
    """

    def __init__(
        self,
        cfg: AsrConfig,
        *,
        pipeline: PipelineConfig | None = None,
        allowed_languages: tuple[str, ...] = ("es", "en"),
        fallback_language: str = "es",
        progress: ProgressFn | None = None,
    ) -> None:
        self.cfg = cfg
        self.pipeline = pipeline or PipelineConfig()
        self.allowed_languages = tuple(allowed_languages)
        self.fallback_language = fallback_language
        self.progress = progress
        self._model = None

    # ----------------------------------------------------------------- lifecycle

    def __enter__(self) -> Transcriber:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - environment problem, not logic
            raise AsrError(f"faster-whisper is not importable: {exc}") from exc

        model_path = Path(self.cfg.model_path)
        if not model_path.is_absolute():
            from transcription247.config import project_root

            model_path = project_root() / model_path
        if not model_path.is_dir():
            raise AsrError(f"ASR model directory not found: {model_path}")

        t0 = time.perf_counter()
        log.info("asr: loading %s (%s, %d threads)", MODEL_FAMILY, self.cfg.compute_type,
                 self.cfg.cpu_threads)
        try:
            self._model = WhisperModel(
                str(model_path),
                device="cpu",
                compute_type=self.cfg.compute_type,
                cpu_threads=self.cfg.cpu_threads,
                num_workers=1,
                local_files_only=True,
            )
        except Exception as exc:
            raise AsrError(f"could not load the ASR model from {model_path}: {exc}") from exc
        log.info("asr: model loaded in %.1fs", time.perf_counter() - t0)
        return self

    def __exit__(self, *exc) -> None:
        """Free the model before the diarizer loads (§1, §3.4)."""
        model, self._model = self._model, None
        del model
        gc.collect()
        release_arenas()
        log.info("asr: model released")

    # ----------------------------------------------------------------- transcription

    def _transcribe_once(self, audio: str | np.ndarray, language: str | None):
        """`audio` is the wav path (single-window case: faster-whisper decodes it itself) or an
        already-decoded float32 window."""
        assert self._model is not None
        return self._model.transcribe(
            audio,
            beam_size=self.cfg.beam_size,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": self.cfg.vad_min_silence_ms,
                "speech_pad_ms": self.cfg.vad_speech_pad_ms,
            },
            word_timestamps=True,
            condition_on_previous_text=False,
            language=language,
        )

    def plan(self, wav: Path) -> list[Window]:
        """Window plan for `wav`. Pure, deterministic, and cheap (header + one tail scan per cut)."""
        return plan_from_config(wav, self.pipeline)

    def transcribe(
        self,
        wav: Path,
        *,
        job_id: str = "",
        duration_s: float | None = None,
        partial_path: Path | None = None,
    ) -> AsrResult:
        """Transcribe a 16 kHz mono WAV. Returns an AsrResult; empty `words` means no speech.

        The recording is processed one window at a time (§3.4 "Long audio"); with a single window
        the wav path is handed straight to faster-whisper, exactly as before windowing existed.
        `partial_path` (optional) is rewritten after every window so an interrupted stage resumes
        at the next one instead of from the beginning.
        """
        if self._model is None:
            raise AsrError("Transcriber.transcribe() called outside its `with` block")
        wav = Path(wav)
        if not wav.is_file():
            raise AsrError(f"wav not found: {wav}")

        plan = self.plan(wav)
        single = len(plan) == 1
        total = float(duration_s or 0.0) or sum(w.duration_s for w in plan)
        requested = self.cfg.language.strip() or None

        state = _PartialState.load(partial_path, plan) if partial_path else None
        if state is None:
            state = _PartialState.empty()
        elif state.done:
            log.info("asr: resuming at window %d/%d (%d words already transcribed)",
                     state.done + 1, len(plan), len(state.words))

        pinned = requested
        if requested is None and state.language and state.done:
            pinned = state.language  # a resumed run keeps the language the first pass detected

        t0 = time.perf_counter()
        for window in plan[state.done:]:
            samples: np.ndarray | None = None
            try:
                if single:
                    audio: str | np.ndarray = str(wav)
                else:
                    samples = read_window(wav, window)
                    audio = samples
                self._transcribe_window(window, audio, pinned, requested, state, total, t0, len(plan))
                pinned = state.language or pinned
            except AsrError:
                raise
            except Exception as exc:
                raise AsrError(f"transcription of {wav.name} window {window.index} failed: {exc}") from exc
            finally:
                del samples
                if not single:
                    gc.collect()
                    release_arenas()  # give the window's VAD/decode buffers back before the next
            state.done = window.index + 1
            if partial_path is not None:
                state.save(partial_path, plan)

        elapsed = time.perf_counter() - t0
        state.words.sort(key=lambda w: (w.start, w.end))
        log.info(
            "asr: %d segments, %d words, language=%s (p=%.4f, forced=%s) over %d window(s) "
            "in %.1fs (RTF %.3f)",
            len(state.segments), len(state.words), state.language, state.language_probability,
            state.language_forced, len(plan), elapsed, elapsed / total if total else 0.0,
        )
        if not state.words:
            log.warning("asr: 0 speech detected in %s (§13 #7)", wav.name)

        return AsrResult(
            schema=SCHEMA_VERSION,
            job_id=job_id,
            language=state.language or self.fallback_language,
            language_probability=state.language_probability,
            language_forced=state.language_forced,
            duration_s=round(total, 3),
            model=f"{MODEL_FAMILY}-{self.cfg.compute_type}",
            segments=state.segments,
            words=state.words,
            windows=[w.as_pair() for w in plan],
        )

    def _transcribe_window(
        self,
        window: Window,
        audio: str | np.ndarray,
        pinned: str | None,
        requested: str | None,
        state: _PartialState,
        total_s: float,
        t0: float,
        n_windows: int,
    ) -> None:
        """Transcribe one window and append its (globally offset) segments/words to `state`."""
        wt0 = time.perf_counter()
        segments_iter, info = self._transcribe_once(audio, pinned)
        detected = getattr(info, "language", None) or ""
        forced_here = False
        if pinned is None and requested is None and detected not in self.allowed_languages:
            # §3.4: detection outside idiomas_permitidos -> re-run pinned to the fallback.
            log.warning(
                "asr: window %d detected language %r not in %s; re-running with language=%r",
                window.index, detected, self.allowed_languages, self.fallback_language,
            )
            del segments_iter
            segments_iter, info = self._transcribe_once(audio, self.fallback_language)
            detected = self.fallback_language
            forced_here = True

        before = len(state.words)
        self._collect(segments_iter, window, state, total_s, t0)
        produced = len(state.words) - before

        # The language is decided by the first window that actually contained speech and then
        # pinned for the rest: re-detecting per window can flip on a window of silence or noise.
        probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        if pinned is None and (produced or window.index == n_windows - 1):
            state.language = detected or self.fallback_language
            state.language_probability = probability
            state.language_forced = forced_here
            if n_windows > 1:
                log.info("asr: language pinned to %r from window %d (p=%.4f)",
                         state.language, window.index, state.language_probability)
        elif not state.language:
            # The caller pinned the language up front (`asr.language` / `job.language_user`), so
            # there is nothing to decide — but faster-whisper still reports a probability for it
            # and `asr.json` has always carried it.
            state.language = pinned or detected or self.fallback_language
            state.language_probability = probability

        wall = time.perf_counter() - wt0
        if n_windows > 1:
            log.info(
                "asr: window %d/%d [%s-%s] done in %.1fs (RTF %.3f), %d words, %d total",
                window.index + 1, n_windows, _hms(window.start_s), _hms(window.end_s), wall,
                wall / window.duration_s if window.duration_s else 0.0, produced, len(state.words),
            )
        if self.progress is not None:  # at least one heartbeat per window (worker renews the lease)
            self.progress(float(window.end_s), total_s, time.perf_counter() - t0)

    def _collect(
        self, segments_iter, window: Window, state: _PartialState, total_s: float, t0: float
    ) -> None:
        """Drain faster-whisper's lazy generator, shifting every timestamp into the global
        timeline and emitting progress for long recordings."""
        offset = window.start_s
        next_mark = state.next_mark
        for seg in segments_iter:
            start, end = float(seg.start) + offset, float(seg.end) + offset
            state.segments.append(
                AsrSegment(
                    id=len(state.segments),
                    start=start,
                    end=end,
                    text=seg.text,
                    avg_logprob=float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                    no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                )
            )
            for w in seg.words or ():
                if w.start is None or w.end is None:
                    continue
                state.words.append(
                    Word(
                        start=float(w.start) + offset,
                        end=float(w.end) + offset,
                        word=w.word,
                        probability=float(getattr(w, "probability", 0.0) or 0.0),
                    )
                )
            if end >= next_mark:
                elapsed = time.perf_counter() - t0
                pct = (end / total_s * 100.0) if total_s else 0.0
                log.info(
                    "asr: %s / %s audio (%.0f%%), %.0fs elapsed, %d words",
                    _hms(end), _hms(total_s), pct, elapsed, len(state.words),
                )
                if self.progress is not None:
                    self.progress(end, total_s, elapsed)
                while next_mark <= end:
                    next_mark += PROGRESS_EVERY_S
        state.next_mark = max(next_mark, window.end_s)


def run(
    wav: Path,
    cfg: AsrConfig,
    *,
    pipeline: PipelineConfig | None = None,
    job_id: str = "",
    duration_s: float | None = None,
    allowed_languages: tuple[str, ...] = ("es", "en"),
    fallback_language: str = "es",
    progress: ProgressFn | None = None,
    partial_path: Path | None = None,
) -> AsrResult:
    """Load the model, transcribe, release the model. The worker's single entry point."""
    with Transcriber(
        cfg,
        pipeline=pipeline,
        allowed_languages=allowed_languages,
        fallback_language=fallback_language,
        progress=progress,
    ) as t:
        return t.transcribe(wav, job_id=job_id, duration_s=duration_s, partial_path=partial_path)


def ensure_speech(result: AsrResult) -> None:
    """Raise AsrError(NO_SPEECH_ERROR) when the recording contains no speech (§13 #7).

    Kept separate from `run()` so the worker can still persist `asr.json` (making the failure
    inspectable and the retry cheap) before deciding to fail the job.
    """
    if not result.words:
        raise AsrError(NO_SPEECH_ERROR)


def _hms(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:d}:{total % 3600 // 60:02d}:{total % 60:02d}"

"""Worker daemon: stage orchestration, pre-flight, resume, delivery (ARCHITECTURE.md §1, §4, §13).

One process, one job at a time, strictly serial (§1). Each stage loads its model, writes its JSON
artifact under `data/jobs/<id>/`, and releases the model before the next one loads, so peak RAM is
`max(stage)` and never the sum. The two heavy stages additionally process the recording one window
at a time (§3.4/§3.5 "Long audio"), so that `max(stage)` is also flat in the recording's *duration*
— which is what makes a 2 h job fit in the same budget as a 5-minute one. The state column is the
*last completed stage*, so a crash at any point resumes from the artifact that already exists —
that is the whole recovery story (§13 #10), now with a second, finer level inside the ASR stage
(`asr.partial.json`, one checkpoint per window).

Also hosts the four small entry points `cli.py` imports lazily: `enqueue_file`, `print_status`,
`retry` and `resummarize` (§14 P1).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import signal
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from transcription247 import (
    asr,
    audio,
    diarize,
    ids,
    inbox,
    indices,
    merge,
    paths,
    render,
    summarize,
)
from transcription247.config import Config
from transcription247.models import (
    SCHEMA_VERSION,
    AsrResult,
    AsrSegment,
    DiarizationResult,
    Job,
    MasDetalleItem,
    NormalizedAudio,
    PuntoClave,
    SpeakerSegment,
    SummaryHints,
    SummaryResult,
    Tarea,
    Turn,
    TurnsResult,
    Word,
)
from transcription247.notify import format_failure_message, format_success_message, get_notifier
from transcription247.store import DuplicateJobError, Store, StoreError

log = logging.getLogger(__name__)

#: Artifact filenames per stage (§4 "Stage skip / resume"). `received` and `delivered` have none.
ARTIFACTS = {
    "normalized": "normalized.json",
    "transcribed": "asr.json",
    "diarized": "diarization.json",
    "merged": "turns.json",
    "summarized": "summary.json",
}

#: How often the idle loop re-runs the `summary_pending` retry pass (§8.4, D4). Deliberately not a
#: config knob: it costs one real `claude` call per pending job, so it stays conservative.
SUMMARY_RETRY_INTERVAL_S = 1800

#: After this many automatic re-summary attempts a job is left alone until `t247 resummarize`.
SUMMARY_RETRY_MAX_AUTO = 5

#: Artifact pruning (§13 #9) runs at startup and then at this interval.
PRUNE_INTERVAL_S = 6 * 3600

#: "notify at most hourly, not per retry" for a Drive mount that is down (§13 #1).
MOUNT_NOTIFY_INTERVAL_S = 3600

#: How long delivery is held when the Drive mount is unavailable (§13 #1).
MOUNT_RETRY_BACKOFF_S = 300


class StageError(RuntimeError):
    """A stage failed in a retryable way: attempts += 1 and back off (§4 Retries)."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class FatalStageError(StageError):
    """A stage failed in a way retrying cannot fix — corrupt audio (§13 #6), no speech (§13 #7).
    The job goes straight to `failed` with no backoff."""


class HoldJob(Exception):
    """Not a failure: the job cannot proceed *right now* (Drive mount down, §13 #1; not enough
    free RAM, §13 #5). The lease is released and the job stays queued at its last good state, with
    `attempts` untouched."""

    def __init__(self, reason: str, retry_after_s: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after_s = retry_after_s


# --------------------------------------------------------------------------------- small helpers


def _now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def _iso_in(seconds: int) -> str:
    return (_now() + timedelta(seconds=seconds)).isoformat(timespec="seconds")


def mem_available_mb() -> int:
    """`MemAvailable` from /proc/meminfo, in MiB. 0 when unreadable (non-Linux/test sandbox)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return 0


def db_path(cfg: Config) -> Path:
    return cfg.project_root / cfg.paths.data_dir / "t247.db"


def jobs_dir(cfg: Config) -> Path:
    return cfg.project_root / cfg.paths.data_dir / "jobs"


def job_dir(cfg: Config, job_id: str) -> Path:
    return jobs_dir(cfg) / job_id


def _drive_up(cfg: Config) -> bool:
    """`archivo.raiz` present and writable (§13 #1). A down rclone FUSE mount usually reports the
    mountpoint as a plain missing/empty directory, so `is_dir()` + `os.access(W_OK)` is enough."""
    root = Path(cfg.archivo.raiz).expanduser()
    try:
        return root.is_dir() and os.access(root, os.W_OK)
    except OSError:
        return False


# ------------------------------------------------------------------------------ artifact plumbing


def write_artifact(path: Path, obj) -> None:
    """UTF-8 JSON, `ensure_ascii=False`, atomic temp + `os.replace`, through `guard_write` (§3)."""
    target = paths.guard_write(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dataclasses.asdict(obj), ensure_ascii=False)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:  # pragma: no cover - local disk, not FUSE
            pass
    os.replace(tmp, target)


def read_artifact(path: Path) -> dict | None:
    """Return the parsed artifact, or None when it is absent, unparseable, or carries an unknown
    `schema` — in which case it is deleted so its stage re-runs (§4 "Stage skip / resume")."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("artifact %s is corrupt; deleting so the stage re-runs", path)
        path.unlink(missing_ok=True)
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA_VERSION:
        log.warning("artifact %s has schema %r (expected %d); deleting", path,
                    (data or {}).get("schema") if isinstance(data, dict) else None, SCHEMA_VERSION)
        path.unlink(missing_ok=True)
        return None
    return data


def _load(path: Path, builder):
    """`read_artifact` + a dataclass builder, treating a shape mismatch as corruption."""
    data = read_artifact(path)
    if data is None:
        return None
    try:
        return builder(data)
    except (TypeError, KeyError, ValueError) as exc:
        log.warning("artifact %s does not match its dataclass (%s); deleting", path, exc)
        path.unlink(missing_ok=True)
        return None


def _build_asr(d: dict) -> AsrResult:
    return AsrResult(
        **{
            **d,
            "segments": [AsrSegment(**s) for s in d.get("segments", [])],
            "words": [Word(**w) for w in d.get("words", [])],
        }
    )


def _build_diarization(d: dict) -> DiarizationResult:
    return DiarizationResult(**{**d, "segments": [SpeakerSegment(**s) for s in d.get("segments", [])]})


def _build_turns(d: dict) -> TurnsResult:
    return TurnsResult(**{**d, "turns": [Turn(**t) for t in d.get("turns", [])]})


def _build_summary(d: dict) -> SummaryResult:
    return SummaryResult(
        **{
            **d,
            "mas_detalle": [MasDetalleItem(**m) for m in d.get("mas_detalle", [])],
            "puntos_clave": [PuntoClave(**p) for p in d.get("puntos_clave", [])],
            "tareas": [Tarea(**t) for t in d.get("tareas", [])],
        }
    )


_BUILDERS = {
    "normalized.json": lambda d: NormalizedAudio(**d),
    "asr.json": _build_asr,
    "diarization.json": _build_diarization,
    "turns.json": _build_turns,
    "summary.json": _build_summary,
}


def load_artifact(cfg: Config, job_id: str, name: str):
    return _load(job_dir(cfg, job_id) / name, _BUILDERS[name])


# ------------------------------------------------------------------------------------- shutdown


class Shutdown:
    """SIGTERM/SIGINT -> one event the loop and the stage sequence both poll (§11 graceful stop).

    A stage in flight is never interrupted mid-way: the worker finishes it, persists its artifact,
    releases the lease and exits, so the next start resumes from exactly there. `TimeoutStopSec`
    in the unit is the backstop for a stage longer than systemd is willing to wait — a SIGKILL
    there costs at most the current stage, never a completed one.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def install(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handle)
            except ValueError:  # pragma: no cover - not on the main thread (tests)
                pass

    def _handle(self, signum, _frame) -> None:
        log.info("worker: received signal %s; finishing the current stage and stopping", signum)
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        self._event.set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


# ---------------------------------------------------------------------------------- the worker


class Worker:
    """The daemon. Built as a class purely so the integration tests can drive one iteration
    (`run_once`) or one job (`process_job`) without a real loop, a real signal or a real model."""

    def __init__(self, cfg: Config, store: Store, *, shutdown: Shutdown | None = None, notifier=None) -> None:
        self.cfg = cfg
        self.store = store
        self.shutdown = shutdown or Shutdown()
        self.notifier = notifier if notifier is not None else get_notifier(cfg)
        self.summary_runtime = summarize.runtime_from_config(cfg)
        self._last_summary_retry = 0.0
        self._last_prune = 0.0
        self._last_mount_notice = 0.0
        #: jobs whose delivered note was edited by hand — never auto-overwritten again (§9.6).
        self._hand_edited: set[str] = set()

    # ------------------------------------------------------------------ main loop

    def run(self) -> None:
        """Block until SIGTERM/SIGINT. Never raises: a bug in one job must not stop the daemon."""
        requeued = self.store.requeue_stale()
        if requeued:
            log.info("worker: requeued %d stale job(s) from a previous run (§4)", requeued)
        self.prune_artifacts()

        while not self.shutdown.is_set():
            try:
                worked = self.run_once()
            except Exception:  # noqa: BLE001 - the daemon must survive anything
                log.exception("worker: unexpected error in the main loop")
                worked = False
            if not worked and not self.shutdown.is_set():
                self.shutdown.wait(max(1, self.cfg.worker.idle_poll_segundos))
        log.info("worker: stopped")

    def run_once(self) -> bool:
        """One iteration: claim and run a job if there is one. Returns True when work was done."""
        job = self.store.claim_next(self.cfg.worker.lease_minutes)
        if job is None:
            self.periodic()
            return False
        try:
            self.process_job(job)
        except Exception:  # noqa: BLE001
            log.exception("worker: job %s crashed outside stage handling", job.id)
            self._release(job.id)
        return True

    def periodic(self) -> None:
        """Idle-time housekeeping: pending-summary retries and artifact pruning."""
        now = time.monotonic()
        if now - self._last_summary_retry >= SUMMARY_RETRY_INTERVAL_S:
            self._last_summary_retry = now
            try:
                self.retry_pending_summaries()
            except Exception:  # noqa: BLE001
                log.exception("worker: summary_pending retry pass failed")
        if now - self._last_prune >= PRUNE_INTERVAL_S:
            self._last_prune = now
            try:
                self.prune_artifacts()
            except Exception:  # noqa: BLE001
                log.exception("worker: artifact pruning failed")

    # ------------------------------------------------------------------ one job

    def process_job(self, job: Job) -> None:
        """Run every stage after `job.state`, then deliver. One job at a time, strictly serial."""
        log.info("worker: job %s (%s) starting at state=%s", job.id, job.source_name, job.state)
        started = time.monotonic()
        try:
            if not self._wait_for_memory(f"job {job.id}"):
                raise HoldJob("not enough free memory", self.cfg.worker.mem_espera_segundos)

            normalized = self._stage_normalize(job)
            job = self._reload(job)
            if self._checkpoint(job, "normalized"):
                return

            asr_result = self._stage_transcribe(job, normalized)
            job = self._reload(job)
            if self._checkpoint(job, "transcribed"):
                return

            diarization = self._stage_diarize(job, normalized)
            job = self._reload(job)
            if self._checkpoint(job, "diarized"):
                return

            turns = self._stage_merge(job, asr_result, diarization)
            job = self._reload(job)
            if self._checkpoint(job, "merged"):
                return

            summary = self._stage_summarize(job, turns)
            job = self._reload(job)
            # No shutdown checkpoint before delivery: the note is cheap to render and expensive to
            # postpone (it is what the user is waiting for), so we always finish the job here.

            self._deliver(job, normalized, turns, summary)
            log.info("worker: job %s done in %.1fs", job.id, time.monotonic() - started)
        except HoldJob as hold:
            log.warning("worker: holding job %s: %s (retry in %ds)", job.id, hold.reason, hold.retry_after_s)
            self.store.add_event(job.id, "warn", job.state, f"held: {hold.reason}")
            self._hold(job, hold.retry_after_s)
        except FatalStageError as exc:
            self._fail_terminal(job, exc.stage, str(exc))
        except StageError as exc:
            self._fail_retryable(job, exc.stage, str(exc))

    def _reload(self, job: Job) -> Job:
        fresh = self.store.get_job(job.id)
        if fresh is None:  # pragma: no cover - the row cannot vanish under us
            raise StoreError(f"job {job.id} disappeared mid-pipeline")
        return fresh

    def _checkpoint(self, job: Job, stage: str) -> bool:
        """After a finished stage: renew the lease, or — if a stop was requested — release it and
        leave the job exactly where it is, so the next start resumes from this artifact."""
        if self.shutdown.is_set():
            log.info("worker: stopping after stage %s of job %s (artifact kept)", stage, job.id)
            self.store.add_event(job.id, "info", stage, "checkpointed on shutdown")
            self._release(job.id)
            return True
        self.store.renew_lease(job.id, self.cfg.worker.lease_minutes)
        return False

    def _release(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job is None or job.state in ("delivered", "failed", "summary_pending"):
            return
        self.store.advance_stage(job_id, job.state, lease_until=None)

    def _hold(self, job: Job, retry_after_s: int) -> None:
        fresh = self.store.get_job(job.id)
        state = fresh.state if fresh else job.state
        if state in ("delivered", "failed", "summary_pending"):
            return
        self.store.advance_stage(job.id, state, lease_until=None, next_attempt_at=_iso_in(retry_after_s))

    # ------------------------------------------------------------------ stages

    def _wait_for_memory(self, what: str) -> bool:
        """§13 #5: check `MemAvailable` before the job and before each heavy stage; sleep and
        re-check up to `worker.mem_max_esperas` times. Returns False when it never recovered."""
        need = self.cfg.worker.mem_disponible_minima_mb
        if need <= 0:
            return True
        for attempt in range(self.cfg.worker.mem_max_esperas + 1):
            available = mem_available_mb()
            if available >= need or available == 0:  # 0 = /proc/meminfo unreadable: don't block
                return True
            if attempt == 0:
                log.warning(
                    "worker: %s needs %d MB but only %d MB is available; waiting", what, need, available
                )
            if self.shutdown.wait(max(1, self.cfg.worker.mem_espera_segundos)):
                return False
        self._notify(
            format_failure_message(
                job_id=what, source_name="-", stage="preflight",
                error=f"MemAvailable stayed below {need} MB",
            )
        )
        return False

    def _stage_normalize(self, job: Job) -> NormalizedAudio:
        existing = load_artifact(self.cfg, job.id, "normalized.json")
        if existing is not None:
            return existing

        if not job.source_path or not Path(job.source_path).is_file():
            raise FatalStageError("normalized", f"source file is missing: {job.source_path}")

        src = Path(job.source_path)
        wav = job_dir(self.cfg, job.id) / "audio.wav"
        try:
            result = audio.normalize(
                src,
                wav,
                cfg=self.cfg.audio,
                job_id=job.id,
                source_name=job.source_name,
                source_sha256=job.audio_sha256,
                recorded_at_user=job.recorded_at_user,
                tz=self.cfg.general.zona_horaria,
            )
        except audio.AudioError as exc:
            raise FatalStageError("normalized", str(exc)) from exc  # §13 #6: no retries

        write_artifact(job_dir(self.cfg, job.id) / "normalized.json", result)
        self.store.advance_stage(
            job.id, "normalized",
            duration_s=result.duration_s,
            recorded_at=result.creation_time,
            start_source=result.start_source,
        )
        self.store.add_event(
            job.id, "info", "normalized", f"{result.duration_s:.1f}s, start from {result.start_source}"
        )
        return result

    def _lease_heartbeat(self, job_id: str, stage: str):
        """Progress callback that renews the job lease from *inside* a long stage.

        The lease is normally renewed between stages (`_after_stage`), which was enough when no
        single stage could outlast `worker.lease_minutes`. A 2 h recording spends ~35-45 min in
        the ASR alone, so the stage now heartbeats once per window: a reclaim mid-transcription
        would hand the same job to a second worker and waste the whole run. Failures are swallowed
        — a missed renewal must never abort a stage that is otherwise going fine.
        """

        def beat(processed_s: float, total_s: float, elapsed_s: float) -> None:
            try:
                self.store.renew_lease(job_id, self.cfg.worker.lease_minutes)
            except Exception:  # pragma: no cover - sqlite hiccup, not logic
                log.warning("worker: could not renew the lease of job %s during %s", job_id, stage)

        return beat

    def _stage_transcribe(self, job: Job, normalized: NormalizedAudio) -> AsrResult:
        existing = load_artifact(self.cfg, job.id, "asr.json")
        if existing is not None:
            return existing

        if not self._wait_for_memory(f"ASR of job {job.id}"):
            raise HoldJob("not enough free memory for the ASR stage", self.cfg.worker.mem_espera_segundos)

        asr_cfg = self.cfg.asr
        if job.language_user:
            asr_cfg = dataclasses.replace(asr_cfg, language=job.language_user)
        partial = (
            job_dir(self.cfg, job.id) / asr.PARTIAL_NAME
            if self.cfg.pipeline.asr_partial_resume
            else None
        )
        try:
            result = asr.run(
                Path(normalized.wav_path),
                asr_cfg,
                pipeline=self.cfg.pipeline,
                job_id=job.id,
                duration_s=normalized.duration_s,
                allowed_languages=tuple(self.cfg.general.idiomas_permitidos),
                fallback_language=self.cfg.general.idioma_fallback,
                progress=self._lease_heartbeat(job.id, "transcribed"),
                partial_path=partial,
            )
        except asr.AsrError as exc:
            raise StageError("transcribed", str(exc)) from exc
        finally:
            asr.release_arenas()  # stage isolation (§1): hand the model's heap back before the next

        write_artifact(job_dir(self.cfg, job.id) / "asr.json", result)
        if partial is not None:  # asr.json is the durable artifact now; the resume file is spent
            partial.unlink(missing_ok=True)
        self.store.advance_stage(job.id, "transcribed", language=result.language)
        self.store.add_event(
            job.id, "info", "transcribed",
            f"{len(result.segments)} segments, {len(result.words)} words, language={result.language}",
        )
        if not result.words:  # §13 #7
            raise FatalStageError("transcribed", asr.NO_SPEECH_ERROR)
        return result

    def _stage_diarize(self, job: Job, normalized: NormalizedAudio) -> DiarizationResult:
        existing = load_artifact(self.cfg, job.id, "diarization.json")
        if existing is not None:
            return existing

        if not self._wait_for_memory(f"diarization of job {job.id}"):
            raise HoldJob(
                "not enough free memory for the diarization stage", self.cfg.worker.mem_espera_segundos
            )

        diar_cfg = self.cfg.diarization
        if job.speakers_user:  # §3.5/D15: a known speaker count always wins over the threshold
            diar_cfg = dataclasses.replace(diar_cfg, num_clusters=job.speakers_user)
        try:
            result = diarize.run(
                Path(normalized.wav_path),
                diar_cfg,
                pipeline=self.cfg.pipeline,
                duration_s=normalized.duration_s,
                job_id=job.id,
            )
        except (diarize.DiarizeError, ValueError) as exc:
            raise StageError("diarized", str(exc)) from exc
        finally:
            asr.release_arenas()

        write_artifact(job_dir(self.cfg, job.id) / "diarization.json", result)
        self.store.advance_stage(job.id, "diarized")
        self.store.add_event(
            job.id, "info", "diarized",
            f"{len(result.segments)} segments, {result.num_speakers_detected} speakers",
        )
        return result

    def _stage_merge(self, job: Job, asr_result: AsrResult, diarization: DiarizationResult) -> TurnsResult:
        existing = load_artifact(self.cfg, job.id, "turns.json")
        if existing is not None:
            return existing

        if not diarization.segments:
            self.store.add_event(job.id, "warn", "merged", "no diarization segments; single speaker (§7)")
        try:
            result = merge.merge(asr_result, diarization.segments, self.cfg.merge)
        except merge.MergeError as exc:
            raise FatalStageError("merged", str(exc)) from exc

        write_artifact(job_dir(self.cfg, job.id) / "turns.json", result)
        self.store.advance_stage(job.id, "merged")
        self.store.add_event(
            job.id, "info", "merged", f"{len(result.turns)} turns, {len(result.participantes)} speakers"
        )
        return result

    def _hints(self, job: Job) -> SummaryHints:
        cliente = self.cfg.archivo.clientes.get(job.client_user or "")
        return SummaryHints(
            titulo_usuario=job.title_user,
            cliente_usuario=cliente.nombre if cliente else None,
            clientes_catalogo=[
                {"key": c.key, "nombre": c.nombre} for c in self.cfg.archivo.clientes.values()
            ],
            categorias_catalogo=[
                {"key": k, "nombre": v} for k, v in self.cfg.archivo.categorias.items()
            ],
        )

    def _stage_summarize(self, job: Job, turns: TurnsResult) -> SummaryResult:
        """Never fails a job: a summary that cannot be produced becomes a `pending` artifact and
        the note ships with the `Resumen pendiente` marker (§8.4)."""
        existing = load_artifact(self.cfg, job.id, "summary.json")
        if existing is not None:
            return existing

        try:
            result = summarize.summarize(turns, self._hints(job), self.cfg.summary, self.summary_runtime)
        except summarize.SummaryError as exc:
            log.warning("worker: summary failed for job %s: %s", job.id, exc)
            self.store.fail_summary(job.id, str(exc))
            self.store.add_event(job.id, "warn", "summarized", f"summary pending: {exc}")
            result = SummaryResult(
                schema=SCHEMA_VERSION, job_id=job.id, pending=True, chunks=0,
                model=self.cfg.summary.modelo, idioma=turns.language, error=str(exc),
            )

        write_artifact(job_dir(self.cfg, job.id) / "summary.json", result)
        self.store.advance_stage(job.id, "summarized")
        if not result.pending:
            self.store.add_event(
                job.id, "info", "summarized",
                f"{len(result.puntos_clave)} puntos clave, {len(result.tareas)} tareas, "
                f"{result.chunks} chunk(s)",
            )
        return result

    # ------------------------------------------------------------------ delivery

    def _deliver(
        self, job: Job, normalized: NormalizedAudio, turns: TurnsResult, summary: SummaryResult
    ) -> None:
        if not _drive_up(self.cfg):
            self._notify_mount_down()
            raise HoldJob("Drive mount is not available", MOUNT_RETRY_BACKOFF_S)

        cliente = render.resolve_client(
            self.cfg, client_field=job.client_user, titulo_usuario=job.title_user, summary=summary
        )
        inicio = self._start_time(job, normalized)
        titulo, markdown = render.render_markdown(
            self.cfg, turns, summary,
            job_id=job.id,
            titulo_usuario=job.title_user,
            inicio=inicio,
            audio_sha256=job.audio_sha256,
            cliente=cliente,
        )
        try:
            target = paths.note_path(cliente.carpeta, inicio.astimezone(ZoneInfo(self.cfg.general.zona_horaria)), titulo)
            written = render.write_note(target, markdown)
        except OSError as exc:  # the mount can drop between the check and the write (§13 #1)
            self._notify_mount_down()
            raise HoldJob(f"could not write the note: {exc}", MOUNT_RETRY_BACKOFF_S) from exc

        if summary.pending:
            self.store.mark_summary_pending(job.id, str(written), summary.error or "summary pending")
        else:
            self.store.mark_delivered(job.id, str(written))
        self._set_client_columns(job.id, cliente)
        self.store.add_event(job.id, "info", "delivered", f"note written: {written.name}")
        log.info("worker: job %s delivered to %s", job.id, written)

        self._after_delivery(job, written, turns, summary, cliente)

    def _set_client_columns(self, job_id: str, cliente) -> None:
        """`client_key`/`client_source` are plain columns; `advance_stage` refuses the terminal
        states, so they are written through the same private updater the store uses."""
        self.store._update(job_id, client_key=cliente.key, client_source=cliente.source)

    def _after_delivery(self, job: Job, written: Path, turns, summary, cliente) -> None:
        """Index rebuild, inbox handling, local-audio cleanup and notification — each independent
        and each best-effort: a failure here must never undo a delivered note."""
        try:
            indices.rebuild_all(self.cfg)
        except (OSError, RuntimeError) as exc:
            log.warning("worker: index rebuild failed after job %s: %s", job.id, exc)

        if job.origin == "inbox":
            inbox.after_success(self.cfg, job)

        if self.cfg.worker.borrar_audio_tras_entrega:
            self._delete_local_audio(job)

        note_rel = self._relative_note(written)
        self._notify(
            format_success_message(
                title=paths.sanitize_title(job.title_user or summary.titulo or ""),
                duration_s=turns.duration_s,
                client_name=cliente.nombre,
                resumen_corto=summary.resumen_corto or "(resumen pendiente)",
                note_path=note_rel,
            )
        )

    def _relative_note(self, written: Path) -> str:
        root = paths.OUTPUT_ROOT or Path(self.cfg.archivo.raiz)
        try:
            return str(written.relative_to(root))
        except ValueError:
            return str(written)

    def _start_time(self, job: Job, normalized: NormalizedAudio) -> datetime:
        tz = ZoneInfo(self.cfg.general.zona_horaria)
        for candidate in (job.recorded_at, normalized.creation_time):
            if not candidate:
                continue
            try:
                return datetime.fromisoformat(candidate).astimezone(tz)
            except ValueError:
                continue
        return _now().astimezone(tz)

    def _delete_local_audio(self, job: Job) -> None:
        """Drop the normalized wav and the original upload once the note exists (§5
        `borrar_audio_tras_entrega`). The JSON artifacts stay until pruning, so `t247 resummarize`
        and `t247 retry` from `merged` still work without the audio."""
        directory = job_dir(self.cfg, job.id)
        for pattern in ("audio.wav", "source.*"):
            for path in directory.glob(pattern):
                try:
                    paths.guard_write(path).unlink(missing_ok=True)
                except (OSError, paths.PathGuardError) as exc:
                    log.warning("worker: could not delete %s: %s", path, exc)

    # ------------------------------------------------------------------ failure paths

    def _fail_retryable(self, job: Job, stage: str, error: str) -> None:
        updated = self.store.fail_stage(
            job.id, error, self.cfg.worker.max_attempts, self.cfg.worker.retry_backoff_s
        )
        self.store.add_event(job.id, "error", stage, error)
        log.warning(
            "worker: job %s failed at %s (attempt %d/%d): %s",
            job.id, stage, updated.attempts, self.cfg.worker.max_attempts, error,
        )
        if updated.state == "failed":
            self._on_terminal_failure(job, stage, error)

    def _fail_terminal(self, job: Job, stage: str, error: str) -> None:
        """§13 #6/#7: corrupt audio and no-speech never retry."""
        self.store.fail_stage(job.id, error, max_attempts=1, retry_backoff_s=(0,))
        self.store.add_event(job.id, "error", stage, error)
        log.warning("worker: job %s failed terminally at %s: %s", job.id, stage, error)
        self._on_terminal_failure(job, stage, error)

    def _on_terminal_failure(self, job: Job, stage: str, error: str) -> None:
        if job.origin == "inbox":
            inbox.after_failure(self.cfg, job, f"{stage}: {error}")
        self._notify(
            format_failure_message(
                job_id=job.id, source_name=job.source_name, stage=stage, error=error
            )
        )

    def _notify(self, text: str) -> None:
        try:
            self.notifier.send(text)
        except Exception:  # noqa: BLE001 - a notifier must never affect job state (§3.8)
            log.warning("worker: notification failed", exc_info=True)

    def _notify_mount_down(self) -> None:
        now = time.monotonic()
        if now - self._last_mount_notice < MOUNT_NOTIFY_INTERVAL_S:
            return
        self._last_mount_notice = now
        log.warning("worker: Drive root %s is unavailable; holding delivery", self.cfg.archivo.raiz)
        self._notify(
            format_failure_message(
                job_id="-", source_name=self.cfg.archivo.raiz, stage="delivery",
                error="Drive mount unavailable; deliveries are on hold",
            )
        )

    # ------------------------------------------------------------------ pending summaries

    def retry_pending_summaries(self) -> int:
        """Re-run the summary for every `summary_pending` job and re-render its note (§8.4, D4).

        The note is only overwritten while it is still byte-for-byte the one we wrote: if its body
        no longer hashes to its own `hash_contenido` the user edited it by hand, so it is left
        alone (§9.6) and never retried automatically again — `t247 resummarize <id>` is then the
        explicit way to ask for a sibling.
        """
        if not _drive_up(self.cfg):
            return 0
        done = 0
        for job in self.store.list_jobs(state="summary_pending", limit=50):
            if self.shutdown.is_set():
                break
            if job.id in self._hand_edited or job.summary_attempts >= SUMMARY_RETRY_MAX_AUTO:
                continue
            try:
                if self.resummarize_job(job.id, manual=False):
                    done += 1
            except Exception:  # noqa: BLE001
                log.exception("worker: re-summary of job %s failed", job.id)
        if done:
            log.info("worker: %d pending summary(ies) completed", done)
        return done

    def resummarize_job(self, job_id: str, *, manual: bool) -> bool:
        """Re-summarize one job from its `turns.json` and rewrite the note. Returns True on a
        completed summary. `manual=True` (from `t247 resummarize`) also accepts a hand-edited note,
        where `render.write_note` writes a `(2)` sibling rather than destroying the edit."""
        job = self.store.get_job(job_id)
        if job is None:
            raise StoreError(f"no such job: {job_id}")
        if job.state != "summary_pending":
            # §4: `resummarize` is the fix for *delivered without a summary*. Re-running it on a
            # finished job costs a real `claude` call and can rename the note (a new `titulo` is a
            # new path), so it is refused rather than silently redone.
            raise StoreError(
                f"job {job_id} is in state '{job.state}', not 'summary_pending'; "
                "nothing to re-summarize (use `t247 retry` to re-run a failed job)"
            )
        turns = load_artifact(self.cfg, job_id, "turns.json")
        if turns is None:
            raise StoreError(f"job {job_id} has no usable turns.json; re-run `t247 retry {job_id}`")
        normalized = load_artifact(self.cfg, job_id, "normalized.json")

        if job.note_path and not manual and not self._note_is_ours(Path(job.note_path)):
            log.warning(
                "worker: note for job %s was edited by hand; not overwriting it automatically", job_id
            )
            self.store.add_event(job_id, "warn", "summarized", "note edited by hand; auto-retry disabled")
            self._hand_edited.add(job_id)
            return False

        try:
            summary = summarize.summarize(turns, self._hints(job), self.cfg.summary, self.summary_runtime)
        except summarize.SummaryError as exc:
            self.store.fail_summary(job_id, str(exc))
            self.store.add_event(job_id, "warn", "summarized", f"re-summary failed: {exc}")
            log.warning("worker: re-summary of job %s failed: %s", job_id, exc)
            return False

        write_artifact(job_dir(self.cfg, job_id) / "summary.json", summary)

        cliente = render.resolve_client(
            self.cfg, client_field=job.client_user, titulo_usuario=job.title_user, summary=summary
        )
        inicio = self._start_time(job, normalized) if normalized else _now()
        titulo, markdown = render.render_markdown(
            self.cfg, turns, summary,
            job_id=job_id, titulo_usuario=job.title_user, inicio=inicio,
            audio_sha256=job.audio_sha256, cliente=cliente,
        )
        target = paths.note_path(cliente.carpeta, inicio, titulo)
        written = render.write_note(target, markdown)

        # The client (and therefore the folder) can change now that a real summary exists: drop the
        # old note if it moved, so the index does not show the meeting twice.
        if job.note_path and Path(job.note_path) != written and Path(job.note_path).exists():
            if self._note_is_ours(Path(job.note_path)):
                try:
                    paths.guard_write(Path(job.note_path)).unlink()
                except (OSError, paths.PathGuardError) as exc:
                    log.warning("worker: could not remove the superseded note: %s", exc)

        self.store.mark_delivered(job_id, str(written))
        self._set_client_columns(job_id, cliente)
        self.store.add_event(job_id, "info", "delivered", "summary completed, note re-rendered")
        self._hand_edited.discard(job_id)
        try:
            indices.rebuild_all(self.cfg)
        except (OSError, RuntimeError) as exc:
            log.warning("worker: index rebuild failed after re-summary: %s", exc)
        log.info("worker: job %s re-summarized -> %s", job_id, written)
        return True

    @staticmethod
    def _note_is_ours(path: Path) -> bool:
        """True when the file is gone (nothing to protect) or its body still hashes to its own
        stored `hash_contenido` — i.e. exactly what we last wrote (§9.6)."""
        if not path.exists():
            return True
        return render._body_matches_stored_hash(path)

    # ------------------------------------------------------------------ pruning

    def prune_artifacts(self) -> int:
        """Delete `data/jobs/<id>/` for finished jobs older than `worker.conservar_artefactos_dias`
        (§13 #9). `summary_pending` jobs are kept: their `turns.json` is what `resummarize` needs."""
        days = self.cfg.worker.conservar_artefactos_dias
        if days <= 0:
            return 0
        root = jobs_dir(self.cfg)
        if not root.is_dir():
            return 0
        cutoff = time.time() - days * 86400
        removed = 0
        for directory in sorted(root.iterdir()):
            if not directory.is_dir():
                continue
            try:
                if directory.stat().st_mtime > cutoff:
                    continue
                job = self.store.get_job(directory.name)
                if job is not None and job.state not in ("delivered", "failed"):
                    continue
                paths.guard_write(directory)
                shutil.rmtree(directory)
                removed += 1
            except (OSError, paths.PathGuardError) as exc:
                log.warning("worker: could not prune %s: %s", directory, exc)
        if removed:
            log.info("worker: pruned %d job artifact director(ies) older than %d days", removed, days)
        return removed


# --------------------------------------------------------------------------------- cli entry points


def run_worker(cfg: Config) -> None:
    """`t247 worker` (§14 P1 -> `worker.run_worker`). Starts the inbox poller thread (§14 D3) and
    then the serial job loop, until SIGTERM."""
    paths.configure(cfg)
    for sub in ("jobs", "tmp", "logs"):
        (cfg.project_root / cfg.paths.data_dir / sub).mkdir(parents=True, exist_ok=True)

    store = Store.open(db_path(cfg))
    shutdown = Shutdown()
    shutdown.install()
    worker = Worker(cfg, store, shutdown=shutdown)

    if cfg.inbox.habilitado:
        inbox.start_poller(cfg, store)
        log.info("worker: inbox poller started on %s", cfg.inbox.ruta)

    log.info(
        "worker: ready (one job at a time, min free RAM %d MB, lease %d min)",
        cfg.worker.mem_disponible_minima_mb, cfg.worker.lease_minutes,
    )
    try:
        worker.run()
    finally:
        store.close()


def enqueue_file(
    path: Path,
    cfg: Config,
    *,
    title: str | None = None,
    language: str | None = None,
    client: str | None = None,
    recorded_at: str | None = None,
    speakers: int | None = None,
) -> Job:
    """`t247 enqueue` (§14 P1). Same hints as `POST /upload`, including the `[Np]` filename hint."""
    paths.configure(cfg)
    src = Path(path).expanduser()
    if not src.is_file():
        raise StoreError(f"no such file: {src}")
    ext = src.suffix.lower().lstrip(".")
    if ext not in cfg.audio.extensiones:
        raise StoreError(f"unsupported extension .{ext} (config audio.extensiones)")
    if client and client not in cfg.archivo.clientes:
        raise StoreError(f"unknown client {client!r} (config archivo.clientes)")
    if language and language not in cfg.general.idiomas_permitidos:
        raise StoreError(f"unknown language {language!r} (config general.idiomas_permitidos)")
    if recorded_at:
        try:
            datetime.fromisoformat(recorded_at)
        except ValueError as exc:
            raise StoreError("--recorded-at must be ISO 8601") from exc

    hint_name, speakers_from_name = paths.extract_speaker_hint(src.name)
    source_name = paths.sanitize_title(hint_name)
    sha256 = ids.sha256_file(src)

    store = Store.open(db_path(cfg))
    try:
        try:
            job = store.create_job(
                source_name=source_name,
                origin="upload",
                audio_sha256=sha256,
                title_user=title,
                client_user=client,
                language_user=language,
                recorded_at_user=recorded_at,
                speakers_user=speakers if speakers is not None else speakers_from_name,
            )
        except DuplicateJobError as exc:
            log.info("enqueue: %s is a duplicate of job %s", src.name, exc.existing.id)
            return exc.existing
        dest = paths.guard_write(job_dir(cfg, job.id) / f"source.{ext}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        job = store.advance_stage(job.id, "received", source_path=str(dest))
        store.add_event(job.id, "info", "received", f"enqueued from {src}")
        return job
    finally:
        store.close()


def print_status(cfg: Config) -> None:
    """`t247 status` (§14 P1): queue counts plus the most recent jobs."""
    paths.configure(cfg)
    store = Store.open(db_path(cfg))
    try:
        counts = store.counts_by_state()
        total = sum(counts.values())
        print(f"Cola: {total} trabajo(s) — " + (
            ", ".join(f"{state}: {n}" for state, n in sorted(counts.items())) or "vacía"
        ))
        print(f"MemAvailable: {mem_available_mb()} MB · Drive: {'ok' if _drive_up(cfg) else 'NO DISPONIBLE'}")
        jobs = store.list_jobs(limit=15)
        if not jobs:
            return
        print()
        print(f"{'id':26}  {'estado':15}  {'creado':25}  fichero")
        for job in jobs:
            print(f"{job.id:26}  {job.state:15}  {job.created_at:25}  {job.source_name}")
            if job.error:
                print(f"{'':26}  └─ {job.error}")
    finally:
        store.close()


def retry(job_id: str, cfg: Config) -> Job:
    """`t247 retry` (§14 P1): rewind a failed job to its last valid artifact and requeue it."""
    paths.configure(cfg)
    store = Store.open(db_path(cfg))
    try:
        job = store.retry(job_id, jobs_dir(cfg))
        store.add_event(job_id, "info", job.state, "retry requested; attempts reset")
        print(f"{job_id}: reencolado en estado '{job.state}'")
        return job
    finally:
        store.close()


def resummarize(job_id: str, cfg: Config) -> bool:
    """`t247 resummarize` (§14 P1): re-run the summary of a `summary_pending` job and rewrite its
    note. Explicit, so a hand-edited note gets a `(2)` sibling rather than being skipped."""
    paths.configure(cfg)
    store = Store.open(db_path(cfg))
    try:
        worker = Worker(cfg, store)
        try:
            ok = worker.resummarize_job(job_id, manual=True)
        except StoreError as exc:
            print(f"{job_id}: {exc}")
            return False
        print(f"{job_id}: {'resumen regenerado' if ok else 'el resumen sigue pendiente'}")
        return ok
    finally:
        store.close()

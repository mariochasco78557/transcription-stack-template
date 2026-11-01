"""Drive inbox poller: `_INBOX_AUDIO/` (rclone FUSE mount) -> queue rows (ARCHITECTURE.md §6, §13,
§14 P5).

Runs as a daemon thread started by the worker process (`start_poller`), never by the API process.
Exposes three entry points the worker calls: `poll_once` (one scan pass), `after_success` (called
once a job that originated here reaches `delivered`, or `summary_pending` — the note is already
written either way, D4) and `after_failure` (called once a job that originated here reaches
`failed`). Must tolerate the mount being down: every filesystem call that touches `inbox.ruta` is
wrapped so an `OSError` is logged and skipped, never raised into the worker/poller loop.

Data-safety rule: an inbox original is never touched before its job is terminal
(`store.TERMINAL_STATES`). `poll_once` checks the DB — not just its in-memory `tracker` — for an
active job at a candidate's exact path before doing anything else with it, so the rule holds even
right after a worker restart, when `tracker` starts out empty (§13 #8).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from transcription247 import paths
from transcription247.config import Config
from transcription247.models import Job
from transcription247.store import TERMINAL_STATES, DuplicateJobError, Store

log = logging.getLogger(__name__)

_CHUNK_BYTES = 1024 * 1024  # 1 MiB
_IGNORED_SUFFIXES = {".part", ".partial", ".tmp", ".crdownload", ".download"}


@dataclass(slots=True)
class _Snapshot:
    size: int
    mtime: float
    first_seen: float  # monotonic time this (size, mtime) pair was first observed


#: path (str) -> last-seen snapshot; passed explicitly so tests control it and start_poller can
#: keep one alive across polls without module-global state.
Tracker = dict[str, _Snapshot]


def start_poller(cfg: Config, store: Store) -> threading.Thread:
    """Daemon thread: poll every `inbox.poll_segundos`, back off to 60 s after a mount-down
    poll (§13 #2). The worker starts this; the API process never does (§14)."""
    tracker: Tracker = {}

    def _loop() -> None:
        while True:
            ok = poll_once(cfg, store, tracker)
            time.sleep(cfg.inbox.poll_segundos if ok else 60)

    thread = threading.Thread(target=_loop, name="inbox-poller", daemon=True)
    thread.start()
    return thread


def poll_once(cfg: Config, store: Store, tracker: Tracker) -> bool:
    """One scan pass over `inbox.ruta`. Returns False if the mount itself was unreachable (caller
    should back off), True otherwise (including when individual files failed — those are handled
    per-file and never abort the pass)."""
    if not cfg.inbox.habilitado:
        return True

    inbox_root = Path(cfg.inbox.ruta).expanduser()
    try:
        entries = list(os.scandir(inbox_root))
    except OSError as exc:
        log.warning("inbox: mount unreachable at %s: %s", inbox_root, exc)
        return False

    skip_dirs = {cfg.inbox.carpeta_procesados, cfg.inbox.carpeta_error}
    now = time.monotonic()

    for entry in entries:
        try:
            if entry.is_dir():
                continue
            name = entry.name
            if name.startswith(".") or name.startswith("~"):
                continue
            if Path(name).suffix.lower() in _IGNORED_SUFFIXES:
                continue
            if name in skip_dirs:
                continue
            _handle_candidate(cfg, store, tracker, Path(entry.path), now)
        except OSError as exc:
            log.warning("inbox: could not stat/process %s: %s", entry.path, exc)
        except Exception:  # never let one bad file take the poller down
            log.exception("inbox: unexpected error handling %s", entry.path)
    return True


def _handle_candidate(cfg: Config, store: Store, tracker: Tracker, src: Path, now: float) -> None:
    key = str(src)
    if store.find_active_by_origin_ref(key) is not None:
        # A job created from this exact inbox path is still in flight (not yet delivered/failed/
        # summary_pending): never touch or re-hash the original while its job is running (the
        # data-safety rule this module exists to enforce). Derived from the DB, not `tracker`, so
        # this still holds right after a worker restart even though `tracker` then starts empty.
        tracker.pop(key, None)
        return

    st = src.stat()
    prev = tracker.get(key)

    if prev is not None and prev.size == st.st_size and prev.mtime == st.st_mtime:
        if now - prev.first_seen < cfg.inbox.estabilidad_segundos:
            return  # unchanged, but not stable long enough yet
        # stable: two (or more) readings, size+mtime unchanged, >= estabilidad_segundos apart
        del tracker[key]
    else:
        tracker[key] = _Snapshot(size=st.st_size, mtime=st.st_mtime, first_seen=now)
        return  # first sighting, or the file is still changing: wait for the next poll

    _ingest_stable_file(cfg, store, src, st.st_size)


def _ingest_stable_file(cfg: Config, store: Store, src: Path, size: int) -> None:
    ext = src.suffix.lower().lstrip(".")
    if ext not in cfg.audio.extensiones:
        log.warning("inbox: %s has an unsupported extension, moving to error", src)
        _move_to_error(cfg, src, f"unsupported_media_type: .{ext}")
        return
    if size > cfg.audio.max_bytes:
        log.warning("inbox: %s exceeds max_bytes (%d > %d)", src, size, cfg.audio.max_bytes)
        _move_to_error(cfg, src, f"file exceeds max_bytes limit ({cfg.audio.max_bytes})")
        return

    raw_name = Path(src.name).name
    hint_name, speakers_hint = paths.extract_speaker_hint(raw_name)
    source_name = paths.sanitize_title(hint_name)

    tmp_dir = cfg.project_root / cfg.paths.data_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = paths.guard_write(tmp_dir / f"inbox-{uuid.uuid4().hex}.part")

    try:
        sha256 = _copy_chunked_with_hash(src, tmp_path)
    except OSError as exc:
        log.warning("inbox: could not copy %s: %s", src, exc)
        tmp_path.unlink(missing_ok=True)
        return

    try:
        job = store.create_job(
            source_name=source_name,
            origin="inbox",
            audio_sha256=sha256,
            origin_ref=str(src),
            speakers_user=speakers_hint,
        )
    except DuplicateJobError as exc:
        existing = exc.existing
        tmp_path.unlink(missing_ok=True)
        if existing.state not in TERMINAL_STATES:
            # A *different* in-flight job already covers this exact content (the same recording
            # dropped twice under different names). Leave this copy alone — deleting or moving it
            # now would be the same data-safety violation this module exists to prevent. Once the
            # other job reaches a terminal state, a later poll re-hashes this file (it isn't the
            # in-flight job's own path, so the early check above doesn't shortcut it) and resolves
            # it then (§13 #8).
            log.info(
                "inbox: %s duplicates in-flight job %s (state=%s); leaving until it finishes",
                src, existing.id, existing.state,
            )
            return
        if existing.state == "failed":
            # Same content as a permanently-failed job: it would fail the same way and the
            # UNIQUE(audio_sha256) index means it can never be enqueued while that row exists.
            # Move it to _error/ instead of leaving it to be re-hashed on every future poll.
            log.info("inbox: %s duplicates failed job %s; moving to error", src, existing.id)
            _move_to_error(cfg, src, f"duplicate of failed job {existing.id}")
            return
        # existing.state in {"delivered", "summary_pending"}: a true duplicate upload of content
        # already delivered (§13 #8) — apply inbox.after_success on THIS copy, never enqueueing.
        log.info(
            "inbox: %s is a duplicate of job %s (already delivered to %s)",
            src, existing.id, existing.note_path,
        )
        after_success(cfg, existing, original_override=src)
        return

    dest = paths.guard_write(
        cfg.project_root / cfg.paths.data_dir / "jobs" / job.id / f"source.{ext}"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path, dest)
    store.advance_stage(job.id, "received", source_path=str(dest))
    store.add_event(job.id, "info", "received", f"ingested from inbox: {src.name}")
    log.info("inbox: enqueued job %s from %s", job.id, src)


def _copy_chunked_with_hash(src: Path, dst: Path) -> str:
    hasher = hashlib.sha256()
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while chunk := fin.read(_CHUNK_BYTES):
            hasher.update(chunk)
            fout.write(chunk)
    return hasher.hexdigest()


# --------------------------------------------------------------------------- post-delivery hooks


def _guard_inbox_path(cfg: Config, candidate: Path, *, context: str) -> Path | None:
    """Every delete/move target in this module must go through here first (project rule, §12: no
    filesystem write/delete/move under Drive skips the guard).

    Two checks, both required: (1) `candidate` must resolve inside the *configured inbox root*
    specifically — `paths.guard_write()` alone would also accept a path resolving into OUTPUT_ROOT
    or DATA_ROOT, which is too permissive for an inbox original/target; (2) it must then also pass
    `paths.guard_write()` itself (denies symlink escapes, the `_Protected_Archive`/`_Protected_Private`
    segments, and raises if the roots were never configured). On either failure, logs and returns
    None — callers must treat that as "refuse to touch this path", never raise into the poller."""
    inbox_root = Path(cfg.inbox.ruta).expanduser()
    try:
        paths.ensure_inside(inbox_root, candidate)
    except paths.PathGuardError as exc:
        log.warning("inbox: refusing %s outside inbox root %s: %s", context, inbox_root, exc)
        return None
    try:
        return paths.guard_write(candidate)
    except paths.PathGuardError as exc:
        log.warning("inbox: refusing %s: %s", context, exc)
        return None


def after_success(cfg: Config, job: Job, *, original_override: Path | None = None) -> None:
    """Called once a job that originated in the inbox (`job.origin == "inbox"`) reaches
    `delivered` (or, for a duplicate, immediately — §13 #8). Applies `inbox.after_success`:
    `"delete"` removes the original Drive file; `"archive"` moves it under `_procesados/`.
    Never deletes anything before this is called. Tolerates the mount being down: logs and
    returns instead of raising. Every path touched is routed through `_guard_inbox_path` first
    (§12); a `job.origin_ref` that does not resolve inside the configured inbox root is refused,
    not touched."""
    raw_original = original_override or (Path(job.origin_ref) if job.origin_ref else None)
    if raw_original is None:
        return  # not an inbox-origin job; nothing to do
    original = _guard_inbox_path(cfg, raw_original, context="after_success original")
    if original is None:
        return
    try:
        if not original.exists():
            return  # already handled (idempotent re-call)
        if cfg.inbox.after_success == "archive":
            procesados_dir = Path(cfg.inbox.ruta).expanduser() / cfg.inbox.carpeta_procesados
            procesados = _guard_inbox_path(
                cfg, procesados_dir / original.name, context="after_success archive target"
            )
            if procesados is None:
                return
            procesados.parent.mkdir(parents=True, exist_ok=True)
            os.replace(original, procesados)
        else:
            original.unlink()
    except OSError as exc:
        log.warning("inbox: after_success could not handle %s: %s", original, exc)


def after_failure(cfg: Config, job: Job, reason: str) -> None:
    """Called once a job that originated in the inbox reaches `failed`. Moves the original into
    `_error/` and writes a sibling `<name>.txt` with `reason`. Tolerates the mount being down."""
    _move_to_error(cfg, Path(job.origin_ref) if job.origin_ref else None, reason)


def _move_to_error(cfg: Config, raw_original: Path | None, reason: str) -> None:
    if raw_original is None:
        return
    original = _guard_inbox_path(cfg, raw_original, context="move_to_error original")
    if original is None:
        return
    try:
        if not original.exists():
            return
        error_dir = Path(cfg.inbox.ruta).expanduser() / cfg.inbox.carpeta_error
        dest = _guard_inbox_path(cfg, error_dir / original.name, context="move_to_error target")
        if dest is None:
            return
        reason_path = _guard_inbox_path(
            cfg, error_dir / f"{original.name}.txt", context="move_to_error reason file"
        )
        if reason_path is None:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(original, dest)
        reason_path.write_text(reason, encoding="utf-8")
    except OSError as exc:
        log.warning("inbox: could not move %s to error dir: %s", original, exc)

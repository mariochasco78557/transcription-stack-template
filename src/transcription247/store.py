"""SQLite job queue: schema, state machine, leases, retry/resume, dedup (ARCHITECTURE.md §4).

Single-writer, single-reader-many usage (one worker, occasional CLI/API reads), so WAL + a short
busy_timeout is enough as far as *processes* go.

**Threads.** The worker process shares one `Store` between its main loop and the inbox poller
thread (§14, D3), so the connection is opened with `check_same_thread=False` and every public
method holds `Store._lock` (an `RLock`) for the whole operation. The lock is what makes
`claim_next()`'s `BEGIN IMMEDIATE … COMMIT` atomic with respect to the other thread as well: a
single `sqlite3.Connection` has exactly one transaction, so two threads interleaving statements on
it would otherwise commit each other's work. Cross-*process* safety still comes from WAL +
`busy_timeout` + the `BEGIN IMMEDIATE` claim (§13 #13).
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from transcription247.ids import new_job_id
from transcription247.models import STAGE_ORDER, Job, JobEvent

#: stage (excluding "received" and "delivered", which have no data/jobs/<id>/*.json artifact) ->
#: artifact filename, in pipeline order. Used by retry() to find where to rewind to.
_STAGE_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("normalized", "normalized.json"),
    ("transcribed", "asr.json"),
    ("diarized", "diarization.json"),
    ("merged", "turns.json"),
    ("summarized", "summary.json"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  state TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  source_name TEXT NOT NULL,
  source_path TEXT,
  origin TEXT NOT NULL,
  origin_ref TEXT,
  audio_sha256 TEXT NOT NULL,
  duration_s REAL,
  title_user TEXT, client_user TEXT, language_user TEXT, recorded_at_user TEXT,
  speakers_user INTEGER,
  recorded_at TEXT,
  start_source TEXT,
  language TEXT,
  client_key TEXT,
  client_source TEXT,
  note_path TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  summary_attempts INTEGER NOT NULL DEFAULT 0,
  next_attempt_at TEXT,
  lease_until TEXT,
  error TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_sha ON jobs(audio_sha256);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, created_at);

CREATE TABLE IF NOT EXISTS job_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  ts TEXT NOT NULL, level TEXT NOT NULL,
  stage TEXT NOT NULL, message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_job ON job_events(job_id, id);

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


#: Terminal `jobs.state` values: a job here is done (one way or another) and will never be
#: claimed again (§4, `claim_next`/`requeue_stale`). `summary_pending` counts as terminal-success
#: for delivery purposes — the note is already written, only the summary re-run is pending (D4).
TERMINAL_STATES: tuple[str, ...] = ("delivered", "failed", "summary_pending")


class StoreError(RuntimeError):
    """A store operation failed in a way the caller must handle (not necessarily a bug)."""


class DuplicateJobError(StoreError):
    """create_job() found an existing job with the same audio_sha256; carries that Job."""

    def __init__(self, existing: Job) -> None:
        super().__init__(f"duplicate audio_sha256: job {existing.id} already exists")
        self.existing = existing


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _locked(method):
    """Serialize a `Store` method on `self._lock`. The lock is reentrant, so methods that call
    other locked methods (create_job -> find_by_sha256 -> get_job) nest without deadlocking."""
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def connect(db_path: Path) -> sqlite3.Connection:
    """Open db_path with the pragmas §4 requires; creates the schema if absent.

    `check_same_thread=False` because the worker shares one connection between its loop and the
    inbox poller thread; `Store` serializes every use of it behind an `RLock` (module docstring).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        db_path, isolation_level=None, check_same_thread=False
    )  # autocommit; explicit BEGIN for txns
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    return conn


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(**{k: row[k] for k in row.keys()})


class Store:
    """Thin wrapper around the jobs/job_events tables. One instance per process is enough, and it
    is safe to share between threads: every public method takes `self._lock` first."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.RLock()

    @classmethod
    def open(cls, db_path: Path) -> Store:
        return cls(connect(db_path))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----------------------------------------------------------------- creation / lookup / dedup

    @_locked
    def create_job(
        self,
        *,
        source_name: str,
        origin: str,
        audio_sha256: str,
        source_path: str | None = None,
        origin_ref: str | None = None,
        duration_s: float | None = None,
        title_user: str | None = None,
        client_user: str | None = None,
        language_user: str | None = None,
        recorded_at_user: str | None = None,
        speakers_user: int | None = None,
    ) -> Job:
        """Insert a new job at state='received'. Raises DuplicateJobError (never inserts) when
        audio_sha256 already exists — the UNIQUE index is the authoritative backstop against a
        race between the pre-check and the insert."""
        existing = self.find_by_sha256(audio_sha256)
        if existing is not None:
            raise DuplicateJobError(existing)
        job_id = new_job_id()
        now = _now_iso()
        try:
            self._conn.execute(
                "INSERT INTO jobs (id, state, created_at, updated_at, source_name, source_path, "
                "origin, origin_ref, audio_sha256, duration_s, title_user, client_user, "
                "language_user, recorded_at_user, speakers_user, attempts, summary_attempts) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0)",
                (
                    job_id, "received", now, now, source_name, source_path, origin, origin_ref,
                    audio_sha256, duration_s, title_user, client_user, language_user,
                    recorded_at_user, speakers_user,
                ),
            )
        except sqlite3.IntegrityError as exc:
            existing = self.find_by_sha256(audio_sha256)
            if existing is not None:
                raise DuplicateJobError(existing) from exc
            raise
        job = self.get_job(job_id)
        assert job is not None
        return job

    @_locked
    def find_by_sha256(self, sha256: str) -> Job | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE audio_sha256 = ?", (sha256,)
        ).fetchone()
        return _row_to_job(row) if row else None

    @_locked
    def get_job(self, job_id: str) -> Job | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    @_locked
    def find_active_by_origin_ref(self, origin_ref: str) -> Job | None:
        """The most recent non-terminal job whose `origin_ref` is this inbox path, if any.

        Lets `inbox.py` recognize "this file's job is still in flight" without re-hashing the
        file on every poll, and — because it is derived from the DB rather than the poller's
        in-memory tracker — the answer is correct even right after a worker restart, when the
        tracker starts out empty (§13 #8)."""
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE origin_ref = ? AND state NOT IN "
            "('delivered','failed','summary_pending') ORDER BY created_at DESC LIMIT 1",
            (origin_ref,),
        ).fetchone()
        return _row_to_job(row) if row else None

    @_locked
    def list_jobs(self, *, state: str | None = None, limit: int = 50) -> list[Job]:
        if state:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE state = ? ORDER BY created_at DESC LIMIT ?",
                (state, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    @_locked
    def counts_by_state(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT state, COUNT(*) c FROM jobs GROUP BY state").fetchall()
        return {r["state"]: r["c"] for r in rows}

    # ----------------------------------------------------------------------------------- events

    @_locked
    def add_event(self, job_id: str, level: str, stage: str, message: str) -> None:
        self._conn.execute(
            "INSERT INTO job_events (job_id, ts, level, stage, message) VALUES (?,?,?,?,?)",
            (job_id, _now_iso(), level, stage, message),
        )

    @_locked
    def events(self, job_id: str, limit: int = 50) -> list[JobEvent]:
        rows = self._conn.execute(
            "SELECT * FROM job_events WHERE job_id = ? ORDER BY id DESC LIMIT ?",
            (job_id, limit),
        ).fetchall()
        return [
            JobEvent(
                id=r["id"], job_id=r["job_id"], ts=r["ts"], level=r["level"],
                stage=r["stage"], message=r["message"],
            )
            for r in reversed(rows)
        ]

    # ------------------------------------------------------------------------------ claim/lease

    @_locked
    def claim_next(self, lease_minutes: int) -> Job | None:
        """Select the oldest eligible job and claim its lease in one transaction (§4 Selection +
        Claim/lease). Returns None if nothing is eligible, or if another worker claimed it first
        between the SELECT and the UPDATE (0 rows affected)."""
        now = _now_iso()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT id FROM jobs WHERE state NOT IN ('delivered','failed','summary_pending') "
                "AND (next_attempt_at IS NULL OR next_attempt_at <= ?) "
                "AND (lease_until IS NULL OR lease_until < ?) "
                "ORDER BY created_at ASC LIMIT 1",
                (now, now),
            ).fetchone()
            if row is None:
                self._conn.execute("COMMIT")
                return None
            job_id = row["id"]
            lease_until = _plus_minutes(lease_minutes)
            cur = self._conn.execute(
                "UPDATE jobs SET lease_until = ?, updated_at = ? "
                "WHERE id = ? AND (lease_until IS NULL OR lease_until < ?)",
                (lease_until, now, job_id, now),
            )
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        if cur.rowcount == 0:
            return None
        return self.get_job(job_id)

    @_locked
    def renew_lease(self, job_id: str, lease_minutes: int) -> None:
        self._update(job_id, lease_until=_plus_minutes(lease_minutes))

    @_locked
    def requeue_stale(self) -> int:
        """Startup requeue (§4): any non-terminal job with a past lease_until gets lease_until =
        NULL. Returns the number of jobs requeued."""
        now = _now_iso()
        cur = self._conn.execute(
            "UPDATE jobs SET lease_until = NULL, updated_at = ? "
            "WHERE lease_until IS NOT NULL AND lease_until < ? "
            "AND state NOT IN ('delivered','failed','summary_pending')",
            (now, now),
        )
        return cur.rowcount

    # ------------------------------------------------------------------------- stage transitions

    @_locked
    def advance_stage(self, job_id: str, new_state: str, **fields) -> Job:
        """Move a job to `new_state` (must be a pipeline stage, §4) plus any extra column values
        (e.g. duration_s after 'normalized', language after 'transcribed')."""
        if new_state not in STAGE_ORDER:
            raise StoreError(f"not a stage state: {new_state!r}")
        self._update(job_id, state=new_state, **fields)
        job = self.get_job(job_id)
        assert job is not None
        return job

    @_locked
    def mark_delivered(self, job_id: str, note_path: str) -> Job:
        return self.advance_stage(
            job_id, "delivered", note_path=note_path, lease_until=None, next_attempt_at=None,
            error=None,
        )

    @_locked
    def mark_summary_pending(self, job_id: str, note_path: str, error: str) -> Job:
        self._update(
            job_id, state="summary_pending", note_path=note_path, lease_until=None,
            next_attempt_at=None, error=error,
        )
        job = self.get_job(job_id)
        assert job is not None
        return job

    @_locked
    def fail_stage(self, job_id: str, error: str, max_attempts: int, retry_backoff_s: list[int] | tuple[int, ...]) -> Job:
        """On stage failure: attempts += 1; below max_attempts, set next_attempt_at from the
        backoff table and leave `state` at the last good stage; at the cap, state='failed' (§4)."""
        job = self.get_job(job_id)
        if job is None:
            raise StoreError(f"no such job: {job_id}")
        attempts = job.attempts + 1
        if attempts >= max_attempts:
            self._update(
                job_id, state="failed", attempts=attempts, error=error,
                lease_until=None, next_attempt_at=None,
            )
        else:
            backoff_s = retry_backoff_s[min(attempts - 1, len(retry_backoff_s) - 1)]
            next_attempt_at = (
                datetime.now(timezone.utc).astimezone() + timedelta(seconds=backoff_s)
            ).isoformat(timespec="seconds")
            self._update(
                job_id, attempts=attempts, error=error,
                lease_until=None, next_attempt_at=next_attempt_at,
            )
        result = self.get_job(job_id)
        assert result is not None
        return result

    @_locked
    def fail_summary(self, job_id: str, error: str) -> Job:
        """Summary retries are counted separately and never push a job to 'failed' (§4)."""
        job = self.get_job(job_id)
        if job is None:
            raise StoreError(f"no such job: {job_id}")
        self._update(job_id, summary_attempts=job.summary_attempts + 1, error=error)
        result = self.get_job(job_id)
        assert result is not None
        return result

    @_locked
    def retry(self, job_id: str, jobs_dir: Path) -> Job:
        """`t247 retry`: zero attempts/error and rewind state to the last stage whose artifact is
        present and parseable (§4). Only valid on a job currently at state='failed'."""
        job = self.get_job(job_id)
        if job is None:
            raise StoreError(f"no such job: {job_id}")
        if job.state != "failed":
            raise StoreError(f"job {job_id} is not failed (state={job.state})")
        new_state = _last_valid_stage(jobs_dir / job_id)
        self._update(
            job_id, state=new_state, attempts=0, error=None,
            next_attempt_at=None, lease_until=None,
        )
        result = self.get_job(job_id)
        assert result is not None
        return result

    # ------------------------------------------------------------------------------------ misc

    @_locked
    def _update(self, job_id: str, **fields) -> None:
        fields["updated_at"] = _now_iso()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = [*fields.values(), job_id]
        self._conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)


def _plus_minutes(minutes: int) -> str:
    return (datetime.now(timezone.utc).astimezone() + timedelta(minutes=minutes)).isoformat(
        timespec="seconds"
    )


def _last_valid_stage(job_dir: Path) -> str:
    """Walk the artifact chain backwards; the first artifact that exists and parses as JSON with
    a 'schema' key wins. No valid artifact at all -> 'received'."""
    import json

    for stage, filename in reversed(_STAGE_ARTIFACTS):
        artifact = job_dir / filename
        if not artifact.is_file():
            continue
        try:
            data = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "schema" in data:
            return stage
    return "received"

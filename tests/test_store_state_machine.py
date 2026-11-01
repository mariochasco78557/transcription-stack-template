"""Store state machine: resume/requeue, artifact-based retry rewind, dedup, retry cap
(ARCHITECTURE.md §4). Uses a tmp SQLite file + tmp jobs dir — never the real data/ dir.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from transcription247.store import DuplicateJobError, Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "t247.db")
    yield s
    s.close()


def _new_job(store: Store, sha: str = "a" * 64) -> str:
    job = store.create_job(source_name="reunion.m4a", origin="upload", audio_sha256=sha)
    assert job.state == "received"
    assert job.attempts == 0
    return job.id


def test_dedup_by_sha256_returns_existing_job(store: Store):
    first_id = _new_job(store, sha="b" * 64)
    with pytest.raises(DuplicateJobError) as exc_info:
        store.create_job(source_name="otra_copia.m4a", origin="upload", audio_sha256="b" * 64)
    assert exc_info.value.existing.id == first_id
    # no second row was inserted
    assert len(store.list_jobs(limit=10)) == 1


def test_requeue_stale_clears_lease_and_stays_at_last_completed_stage(store: Store, tmp_path: Path):
    job_id = _new_job(store)
    store.advance_stage(job_id, "transcribed")
    # simulate a worker that crashed mid-'diarized' stage: lease claimed, then never renewed
    conn: sqlite3.Connection = store._conn  # type: ignore[attr-defined]
    conn.execute(
        "UPDATE jobs SET lease_until = '2000-01-01T00:00:00+00:00' WHERE id = ?", (job_id,)
    )

    requeued = store.requeue_stale()
    assert requeued == 1

    job = store.get_job(job_id)
    assert job is not None
    assert job.lease_until is None
    assert job.state == "transcribed"  # unchanged: worker resumes at the next stage, 'diarized'

    # and it is claimable again
    claimed = store.claim_next(lease_minutes=120)
    assert claimed is not None
    assert claimed.id == job_id


def test_requeue_stale_ignores_terminal_states(store: Store):
    job_id = _new_job(store)
    store.mark_delivered(job_id, note_path="Cliente/2026-09/nota.md")
    conn: sqlite3.Connection = store._conn  # type: ignore[attr-defined]
    conn.execute(
        "UPDATE jobs SET lease_until = '2000-01-01T00:00:00+00:00' WHERE id = ?", (job_id,)
    )
    assert store.requeue_stale() == 0


def test_retry_rewinds_to_last_valid_artifact(store: Store, tmp_path: Path):
    job_id = _new_job(store)
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "normalized.json").write_text(json.dumps({"schema": 1, "ok": True}))
    (job_dir / "asr.json").write_text(json.dumps({"schema": 1, "ok": True}))
    # diarization.json deliberately absent/corrupt -> retry must stop at 'transcribed'
    (job_dir / "diarization.json").write_text("not valid json{")

    # drive the job to 'failed' via three stage failures (retry cap)
    store.fail_stage(job_id, "boom", max_attempts=3, retry_backoff_s=[60, 300, 1800])
    store.fail_stage(job_id, "boom", max_attempts=3, retry_backoff_s=[60, 300, 1800])
    job = store.fail_stage(job_id, "boom", max_attempts=3, retry_backoff_s=[60, 300, 1800])
    assert job.state == "failed"

    retried = store.retry(job_id, jobs_dir)
    assert retried.state == "transcribed"
    assert retried.attempts == 0
    assert retried.error is None


def test_retry_with_no_artifacts_rewinds_to_received(store: Store, tmp_path: Path):
    job_id = _new_job(store)
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / job_id).mkdir(parents=True)
    for _ in range(3):
        job = store.fail_stage(job_id, "boom", max_attempts=3, retry_backoff_s=[60, 300, 1800])
    assert job.state == "failed"

    retried = store.retry(job_id, jobs_dir)
    assert retried.state == "received"
    assert retried.attempts == 0


def test_retry_cap_reaches_failed_with_documented_backoffs(store: Store):
    job_id = _new_job(store)
    backoff = [60, 300, 1800]

    job = store.fail_stage(job_id, "err1", max_attempts=3, retry_backoff_s=backoff)
    assert job.attempts == 1
    assert job.state == "received"  # last good stage, unchanged
    assert job.next_attempt_at is not None

    job = store.fail_stage(job_id, "err2", max_attempts=3, retry_backoff_s=backoff)
    assert job.attempts == 2
    assert job.state == "received"
    assert job.next_attempt_at is not None

    job = store.fail_stage(job_id, "err3", max_attempts=3, retry_backoff_s=backoff)
    assert job.attempts == 3
    assert job.state == "failed"
    assert job.next_attempt_at is None
    assert job.lease_until is None


def test_summary_failure_never_reaches_failed(store: Store):
    job_id = _new_job(store)
    store.advance_stage(job_id, "merged")
    for _ in range(5):
        job = store.fail_summary(job_id, "summary broke")
    assert job.summary_attempts == 5
    assert job.state == "merged"  # never pushed to 'failed' by summary retries

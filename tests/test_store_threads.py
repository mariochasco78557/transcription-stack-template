"""One `Store` shared by the worker loop and the inbox poller thread (ARCHITECTURE.md §14, D3).

Before the fix, `store.connect()` used sqlite3's default `check_same_thread=True`, so the very
first call the poller thread made on the worker's `Store` raised `ProgrammingError`. These tests
fail loudly if that regresses, and also cover the subtler half of the problem: two threads driving
`claim_next()`'s `BEGIN IMMEDIATE … COMMIT` on the *same* connection must not interleave into
each other's transaction, and a job must never be claimed twice.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from transcription247.store import DuplicateJobError, Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "t247.db")
    yield s
    s.close()


def _run_threads(targets: list) -> list[BaseException]:
    errors: list[BaseException] = []
    lock = threading.Lock()

    def wrap(fn):
        def runner():
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread below
                with lock:
                    errors.append(exc)

        return runner

    threads = [threading.Thread(target=wrap(t)) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a store thread deadlocked"
    return errors


def test_store_is_usable_from_another_thread(store: Store):
    """The poller thread's exact usage pattern: create a job and add an event on a Store that was
    opened on the main thread."""
    created: list[str] = []

    def poller_like():
        job = store.create_job(source_name="desde-hilo.m4a", origin="inbox", audio_sha256="c" * 64)
        store.add_event(job.id, "info", "received", "ingested from inbox")
        created.append(job.id)

    errors = _run_threads([poller_like])
    assert not errors, f"store raised on a second thread: {errors}"
    assert len(created) == 1
    assert store.get_job(created[0]) is not None  # and it is visible back on the main thread


def test_concurrent_writers_do_not_corrupt_the_queue(store: Store):
    """Two threads inserting 20 jobs each, while a third claims: every insert lands exactly once
    and nothing raises. `create_job` is the one path with a read-then-insert race, guarded by the
    UNIQUE index (§4)."""

    def inserter(prefix: str):
        def run():
            for i in range(20):
                store.create_job(
                    source_name=f"{prefix}-{i}.m4a", origin="upload", audio_sha256=f"{prefix}{i:062d}"
                )

        return run

    def claimer():
        for _ in range(20):
            store.claim_next(lease_minutes=5)

    errors = _run_threads([inserter("a"), inserter("b"), claimer])
    assert not errors, f"concurrent store use raised: {errors}"
    assert sum(store.counts_by_state().values()) == 40


def test_same_sha_from_two_threads_yields_one_job(store: Store):
    """Both ingestion paths can see the same file at once (§13 #8): exactly one wins."""
    outcomes: list[str] = []
    lock = threading.Lock()

    def create():
        try:
            job = store.create_job(source_name="misma.m4a", origin="upload", audio_sha256="d" * 64)
            with lock:
                outcomes.append(f"created:{job.id}")
        except DuplicateJobError as exc:
            with lock:
                outcomes.append(f"duplicate:{exc.existing.id}")

    errors = _run_threads([create, create])
    assert not errors, f"duplicate handling raised: {errors}"
    assert len(store.list_jobs(limit=10)) == 1
    assert sum(1 for o in outcomes if o.startswith("created")) == 1


def test_claim_next_hands_a_job_to_exactly_one_thread(store: Store):
    """The lease is the serialization point (§4, §13 #13): one job, five claimers, one winner."""
    store.create_job(source_name="unico.m4a", origin="upload", audio_sha256="e" * 64)
    claimed: list[str] = []
    lock = threading.Lock()

    def claim():
        job = store.claim_next(lease_minutes=60)
        if job is not None:
            with lock:
                claimed.append(job.id)

    errors = _run_threads([claim] * 5)
    assert not errors, f"claim_next raised: {errors}"
    assert len(claimed) == 1

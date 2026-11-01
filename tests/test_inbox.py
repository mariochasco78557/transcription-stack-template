"""P5 inbox.py: stability check, dedup, speaker hint, ignore rules, mount-missing tolerance,
after_success/after_failure (ARCHITECTURE.md §6, §13, §14 P5). No real Drive/FUSE mount — every
test points `inbox.ruta` at a tmp dir.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from transcription247 import inbox, paths
from transcription247.config import (
    ApiConfig,
    ArchivoConfig,
    AsrConfig,
    AudioConfig,
    Config,
    DiarizationConfig,
    GeneralConfig,
    InboxConfig,
    LoggingConfig,
    MergeConfig,
    PathsConfig,
    PipelineConfig,
    SummaryConfig,
    WorkerConfig,
)
from transcription247.models import Job
from transcription247.store import Store


def _make_config(tmp_path: Path, *, estabilidad_segundos: int = 0, after_success: str = "delete") -> Config:
    output_root = tmp_path / "drive" / "salida"
    inbox_root = tmp_path / "drive" / "_INBOX_AUDIO"
    output_root.mkdir(parents=True)
    inbox_root.mkdir(parents=True)
    return Config(
        general=GeneralConfig(),
        paths=PathsConfig(data_dir="data"),
        audio=AudioConfig(max_bytes=2000),
        asr=AsrConfig(),
        diarization=DiarizationConfig(),
        merge=MergeConfig(),
        pipeline=PipelineConfig(),
        summary=SummaryConfig(),
        archivo=ArchivoConfig(raiz=str(output_root)),
        inbox=InboxConfig(
            ruta=str(inbox_root),
            estabilidad_segundos=estabilidad_segundos,
            after_success=after_success,
        ),
        api=ApiConfig(),
        worker=WorkerConfig(),
        logging=LoggingConfig(),
        project_root=tmp_path,
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "data" / "t247.db")
    yield s
    s.close()


def _cfg_and_roots(tmp_path: Path, **kwargs) -> Config:
    cfg = _make_config(tmp_path, **kwargs)
    paths.set_roots(
        output_root=Path(cfg.archivo.raiz), inbox_root=Path(cfg.inbox.ruta), data_root=tmp_path / "data"
    )
    return cfg


# --------------------------------------------------------------------------------- stability


def test_stability_requires_two_unchanged_polls(tmp_path: Path, store: Store):
    cfg = _cfg_and_roots(tmp_path, estabilidad_segundos=0)
    audio = Path(cfg.inbox.ruta) / "reunion.wav"
    audio.write_bytes(b"a" * 20)
    tracker: inbox.Tracker = {}

    assert inbox.poll_once(cfg, store, tracker) is True
    assert store.list_jobs(limit=10) == []  # first sighting only: not enqueued yet

    assert inbox.poll_once(cfg, store, tracker) is True
    jobs = store.list_jobs(limit=10)
    assert len(jobs) == 1
    assert jobs[0].origin == "inbox"
    assert jobs[0].origin_ref == str(audio)
    assert audio.exists()  # original never deleted before delivery


def test_still_changing_file_resets_stability_clock(tmp_path: Path, store: Store):
    cfg = _cfg_and_roots(tmp_path, estabilidad_segundos=0)
    audio = Path(cfg.inbox.ruta) / "reunion.wav"
    audio.write_bytes(b"a" * 20)
    tracker: inbox.Tracker = {}

    inbox.poll_once(cfg, store, tracker)  # first sighting
    audio.write_bytes(b"a" * 999)  # still being written: size changes
    inbox.poll_once(cfg, store, tracker)  # sees the change -> resets, still not enqueued
    assert store.list_jobs(limit=10) == []

    inbox.poll_once(cfg, store, tracker)  # unchanged since the reset -> now stable
    assert len(store.list_jobs(limit=10)) == 1


def test_not_stable_before_configured_seconds_elapse(tmp_path: Path, store: Store):
    cfg = _cfg_and_roots(tmp_path, estabilidad_segundos=9999)
    audio = Path(cfg.inbox.ruta) / "reunion.wav"
    audio.write_bytes(b"a" * 20)
    tracker: inbox.Tracker = {}

    for _ in range(5):
        inbox.poll_once(cfg, store, tracker)
    assert store.list_jobs(limit=10) == []  # would need ~9999s of stability first


# ------------------------------------------------------------------------------------ dedup


def test_dedup_deletes_duplicate_of_delivered_job(tmp_path: Path, store: Store):
    """A NEW inbox file whose sha256 matches a job already `delivered` is a true duplicate:
    it is removed per `inbox.after_success` policy without ever being enqueued (§13 #8)."""
    cfg = _cfg_and_roots(tmp_path, estabilidad_segundos=0, after_success="delete")
    inbox_root = Path(cfg.inbox.ruta)
    first = inbox_root / "primero.wav"
    second = inbox_root / "segundo.wav"
    first.write_bytes(b"contenido-identico" * 3)
    tracker: inbox.Tracker = {}
    inbox.poll_once(cfg, store, tracker)
    inbox.poll_once(cfg, store, tracker)
    jobs = store.list_jobs(limit=10)
    assert len(jobs) == 1
    store.mark_delivered(jobs[0].id, note_path=str(tmp_path / "note.md"))

    second.write_bytes(b"contenido-identico" * 3)  # same sha256, different name
    inbox.poll_once(cfg, store, tracker)
    inbox.poll_once(cfg, store, tracker)

    assert len(store.list_jobs(limit=10)) == 1  # still one job, no duplicate row
    assert not second.exists()  # duplicate original was deleted (job was already delivered)


def test_dedup_of_in_flight_job_is_left_alone(tmp_path: Path, store: Store):
    """A duplicate of a job that is NOT yet terminal must be left untouched (not deleted, not
    enqueued) until the original job finishes — deleting it now would be the same data-safety
    violation as deleting the in-flight original itself."""
    cfg = _cfg_and_roots(tmp_path, estabilidad_segundos=0, after_success="delete")
    inbox_root = Path(cfg.inbox.ruta)
    first = inbox_root / "primero.wav"
    second = inbox_root / "segundo.wav"
    first.write_bytes(b"contenido-identico" * 3)
    tracker: inbox.Tracker = {}
    inbox.poll_once(cfg, store, tracker)
    inbox.poll_once(cfg, store, tracker)
    jobs = store.list_jobs(limit=10)
    assert len(jobs) == 1
    assert jobs[0].state != "delivered"  # still in flight ("received")

    second.write_bytes(b"contenido-identico" * 3)  # same sha256, different name
    for _ in range(3):
        inbox.poll_once(cfg, store, tracker)

    assert len(store.list_jobs(limit=10)) == 1  # no duplicate row was enqueued
    assert second.exists()  # left alone: the other job hasn't reached a terminal state yet


# ----------------------------------------------------------------- in-flight originals (data safety)


def test_in_flight_original_untouched_across_several_polls(tmp_path: Path, store: Store):
    """The bug this module must not regress: while a job is in flight, later polls that see its
    original again must leave it alone — no delete, no second job row — no matter how many times
    the poller runs before the job reaches a terminal state."""
    cfg = _cfg_and_roots(tmp_path, estabilidad_segundos=0)
    audio = Path(cfg.inbox.ruta) / "reunion.wav"
    audio.write_bytes(b"contenido" * 5)
    tracker: inbox.Tracker = {}
    inbox.poll_once(cfg, store, tracker)
    inbox.poll_once(cfg, store, tracker)
    jobs = store.list_jobs(limit=10)
    assert len(jobs) == 1
    job_id = jobs[0].id
    assert jobs[0].state == "received"

    for _ in range(6):
        assert inbox.poll_once(cfg, store, tracker) is True
        assert audio.exists()
        assert len(store.list_jobs(limit=10)) == 1
        assert store.get_job(job_id).state == "received"


def test_original_removed_only_after_delivered(tmp_path: Path, store: Store):
    cfg = _cfg_and_roots(tmp_path, estabilidad_segundos=0, after_success="delete")
    audio = Path(cfg.inbox.ruta) / "reunion.wav"
    audio.write_bytes(b"contenido" * 5)
    tracker: inbox.Tracker = {}
    inbox.poll_once(cfg, store, tracker)
    inbox.poll_once(cfg, store, tracker)
    job = store.list_jobs(limit=10)[0]

    # several more in-flight polls: still there
    for _ in range(3):
        inbox.poll_once(cfg, store, tracker)
    assert audio.exists()

    store.mark_delivered(job.id, note_path=str(tmp_path / "note.md"))

    # the poller needs to see it stable again (its own tracker entry was never touched while
    # in flight, so it re-runs the normal two-poll stability check) before it acts on it
    inbox.poll_once(cfg, store, tracker)
    inbox.poll_once(cfg, store, tracker)

    assert not audio.exists()  # removed now that the job is delivered
    assert len(store.list_jobs(limit=10)) == 1  # no duplicate row created along the way


def test_restart_mid_job_does_not_delete_or_double_enqueue(tmp_path: Path, store: Store):
    """A fresh `Tracker` (as after a worker restart) must not reintroduce the bug: the guard
    against touching an in-flight original is derived from the DB, not from `tracker`."""
    cfg = _cfg_and_roots(tmp_path, estabilidad_segundos=0)
    audio = Path(cfg.inbox.ruta) / "reunion.wav"
    audio.write_bytes(b"contenido" * 5)
    tracker: inbox.Tracker = {}
    inbox.poll_once(cfg, store, tracker)
    inbox.poll_once(cfg, store, tracker)
    assert len(store.list_jobs(limit=10)) == 1

    fresh_tracker: inbox.Tracker = {}  # simulates the worker process restarting
    for _ in range(4):
        inbox.poll_once(cfg, store, fresh_tracker)

    assert audio.exists()  # never deleted
    assert len(store.list_jobs(limit=10)) == 1  # never double-enqueued


def test_failure_moves_original_to_error_and_stops_reenqueueing(tmp_path: Path, store: Store):
    cfg = _cfg_and_roots(tmp_path, estabilidad_segundos=0)
    audio = Path(cfg.inbox.ruta) / "reunion.wav"
    audio.write_bytes(b"contenido" * 5)
    tracker: inbox.Tracker = {}
    inbox.poll_once(cfg, store, tracker)
    inbox.poll_once(cfg, store, tracker)
    job = store.list_jobs(limit=10)[0]

    updated = store.fail_stage(job.id, "no_speech", max_attempts=1, retry_backoff_s=(0,))
    assert updated.state == "failed"
    inbox.after_failure(cfg, updated, "diarize: no_speech")  # what worker._on_terminal_failure does

    error_dir = Path(cfg.inbox.ruta) / cfg.inbox.carpeta_error
    assert (error_dir / "reunion.wav").is_file()
    assert (error_dir / "reunion.wav.txt").read_text() == "diarize: no_speech"
    assert not audio.exists()

    for _ in range(3):  # the error dir is never scanned, so the failed job never comes back
        inbox.poll_once(cfg, store, tracker)
    assert len(store.list_jobs(limit=10)) == 1


# ------------------------------------------------------------------------------ speaker hint


def test_speaker_hint_extracted_and_stripped_from_source_name(tmp_path: Path, store: Store):
    cfg = _cfg_and_roots(tmp_path, estabilidad_segundos=0)
    audio = Path(cfg.inbox.ruta) / "Reunion [3p] pruebas.wav"
    audio.write_bytes(b"x" * 30)
    tracker: inbox.Tracker = {}
    inbox.poll_once(cfg, store, tracker)
    inbox.poll_once(cfg, store, tracker)

    jobs = store.list_jobs(limit=10)
    assert len(jobs) == 1
    assert jobs[0].speakers_user == 3
    assert "[3p]" not in jobs[0].source_name


# ------------------------------------------------------------------------------- ignore rules


def test_ignores_hidden_temp_and_processed_folders(tmp_path: Path, store: Store):
    cfg = _cfg_and_roots(tmp_path, estabilidad_segundos=0)
    inbox_root = Path(cfg.inbox.ruta)
    (inbox_root / ".DS_Store").write_bytes(b"junk")
    (inbox_root / "uploading.part").write_bytes(b"junk")
    (inbox_root / cfg.inbox.carpeta_procesados).mkdir()
    (inbox_root / cfg.inbox.carpeta_procesados / "viejo.wav").write_bytes(b"junk")
    (inbox_root / cfg.inbox.carpeta_error).mkdir()
    (inbox_root / cfg.inbox.carpeta_error / "malo.wav").write_bytes(b"junk")

    tracker: inbox.Tracker = {}
    for _ in range(3):
        assert inbox.poll_once(cfg, store, tracker) is True
    assert store.list_jobs(limit=10) == []


def test_unsupported_extension_moves_to_error_with_reason(tmp_path: Path, store: Store):
    cfg = _cfg_and_roots(tmp_path, estabilidad_segundos=0)
    inbox_root = Path(cfg.inbox.ruta)
    bad = inbox_root / "notas.txt"
    bad.write_bytes(b"no es audio")
    tracker: inbox.Tracker = {}
    inbox.poll_once(cfg, store, tracker)
    inbox.poll_once(cfg, store, tracker)

    assert store.list_jobs(limit=10) == []
    error_dir = inbox_root / cfg.inbox.carpeta_error
    assert (error_dir / "notas.txt").is_file()
    assert (error_dir / "notas.txt.txt").is_file()
    assert "unsupported_media_type" in (error_dir / "notas.txt.txt").read_text()


def test_oversize_file_moves_to_error(tmp_path: Path, store: Store):
    cfg = _cfg_and_roots(tmp_path, estabilidad_segundos=0)
    inbox_root = Path(cfg.inbox.ruta)
    big = inbox_root / "enorme.wav"
    big.write_bytes(b"x" * (cfg.audio.max_bytes + 1))
    tracker: inbox.Tracker = {}
    inbox.poll_once(cfg, store, tracker)
    inbox.poll_once(cfg, store, tracker)

    assert store.list_jobs(limit=10) == []
    assert (inbox_root / cfg.inbox.carpeta_error / "enorme.wav").is_file()


# ------------------------------------------------------------------------- mount tolerance


def test_poll_once_tolerates_missing_mount(tmp_path: Path, store: Store):
    cfg = _cfg_and_roots(tmp_path)
    import shutil

    shutil.rmtree(cfg.inbox.ruta)
    tracker: inbox.Tracker = {}
    assert inbox.poll_once(cfg, store, tracker) is False  # caller should back off
    assert store.list_jobs(limit=10) == []


def test_poll_once_noop_when_disabled(tmp_path: Path, store: Store):
    import dataclasses

    cfg = _cfg_and_roots(tmp_path)
    cfg = dataclasses.replace(cfg, inbox=dataclasses.replace(cfg.inbox, habilitado=False))
    audio = Path(cfg.inbox.ruta) / "reunion.wav"
    audio.write_bytes(b"x" * 10)
    tracker: inbox.Tracker = {}
    assert inbox.poll_once(cfg, store, tracker) is True
    assert store.list_jobs(limit=10) == []


# -------------------------------------------------------------------- after_success / after_failure


def _fake_job(origin_ref: str) -> Job:
    return Job(
        id="01jfakejobfakejobfakejobfa",
        state="delivered",
        created_at="2026-09-19T00:00:00+02:00",
        updated_at="2026-09-19T00:00:00+02:00",
        source_name="reunion.wav",
        origin="inbox",
        audio_sha256="a" * 64,
        origin_ref=origin_ref,
    )


def test_after_success_delete(tmp_path: Path):
    cfg = _cfg_and_roots(tmp_path, after_success="delete")
    original = Path(cfg.inbox.ruta) / "reunion.wav"
    original.write_bytes(b"x")
    inbox.after_success(cfg, _fake_job(str(original)))
    assert not original.exists()


def test_after_success_archive(tmp_path: Path):
    cfg = _cfg_and_roots(tmp_path, after_success="archive")
    original = Path(cfg.inbox.ruta) / "reunion.wav"
    original.write_bytes(b"x")
    inbox.after_success(cfg, _fake_job(str(original)))
    assert not original.exists()
    archived = Path(cfg.inbox.ruta) / cfg.inbox.carpeta_procesados / "reunion.wav"
    assert archived.is_file()


def test_after_success_idempotent_when_already_gone(tmp_path: Path):
    cfg = _cfg_and_roots(tmp_path)
    missing = Path(cfg.inbox.ruta) / "ya-no-esta.wav"
    inbox.after_success(cfg, _fake_job(str(missing)))  # must not raise


def test_after_failure_moves_and_writes_reason(tmp_path: Path):
    cfg = _cfg_and_roots(tmp_path)
    original = Path(cfg.inbox.ruta) / "reunion.wav"
    original.write_bytes(b"x")
    inbox.after_failure(cfg, _fake_job(str(original)), "no_speech")
    error_dir = Path(cfg.inbox.ruta) / cfg.inbox.carpeta_error
    assert (error_dir / "reunion.wav").is_file()
    assert (error_dir / "reunion.wav.txt").read_text() == "no_speech"
    assert not original.exists()


# -------------------------------------------------------- origin_ref path-guard bypass (finding 1)


def _make_outside_target(cfg) -> Path:
    """A file that sits next to (NOT inside) the configured inbox root: any of `after_success`/
    `after_failure`'s escape vectors should resolve here without ever touching it."""
    outside_dir = Path(cfg.inbox.ruta).parent / "fuera_del_inbox"
    outside_dir.mkdir(parents=True, exist_ok=True)
    target = outside_dir / "secreto.wav"
    target.write_bytes(b"private")
    return target


def test_after_success_refuses_origin_ref_outside_inbox_sibling(tmp_path: Path):
    cfg = _cfg_and_roots(tmp_path, after_success="delete")
    outside = _make_outside_target(cfg)
    inbox.after_success(cfg, _fake_job(str(outside)))
    assert outside.exists()  # never touched: resolves outside the configured inbox root


def test_after_success_refuses_origin_ref_via_dotdot(tmp_path: Path):
    cfg = _cfg_and_roots(tmp_path, after_success="delete")
    outside = _make_outside_target(cfg)
    traversal = Path(cfg.inbox.ruta) / ".." / "fuera_del_inbox" / "secreto.wav"
    inbox.after_success(cfg, _fake_job(str(traversal)))
    assert outside.exists()


def test_after_success_refuses_origin_ref_via_symlink(tmp_path: Path):
    cfg = _cfg_and_roots(tmp_path, after_success="delete")
    outside = _make_outside_target(cfg)
    link = Path(cfg.inbox.ruta) / "enlace.wav"
    link.symlink_to(outside)
    inbox.after_success(cfg, _fake_job(str(link)))
    assert outside.exists()
    assert link.exists()  # the symlink itself is left alone too


def test_after_success_archive_refuses_origin_ref_outside_inbox(tmp_path: Path):
    cfg = _cfg_and_roots(tmp_path, after_success="archive")
    outside = _make_outside_target(cfg)
    inbox.after_success(cfg, _fake_job(str(outside)))
    assert outside.exists()
    procesados = Path(cfg.inbox.ruta) / cfg.inbox.carpeta_procesados
    assert not procesados.exists() or not any(procesados.iterdir())


def test_after_failure_refuses_origin_ref_outside_inbox_sibling(tmp_path: Path):
    cfg = _cfg_and_roots(tmp_path)
    outside = _make_outside_target(cfg)
    inbox.after_failure(cfg, _fake_job(str(outside)), "some reason")
    assert outside.exists()
    error_dir = Path(cfg.inbox.ruta) / cfg.inbox.carpeta_error
    assert not error_dir.exists() or not any(error_dir.iterdir())


def test_after_failure_refuses_origin_ref_via_dotdot(tmp_path: Path):
    cfg = _cfg_and_roots(tmp_path)
    outside = _make_outside_target(cfg)
    traversal = Path(cfg.inbox.ruta) / ".." / "fuera_del_inbox" / "secreto.wav"
    inbox.after_failure(cfg, _fake_job(str(traversal)), "some reason")
    assert outside.exists()


def test_after_failure_refuses_origin_ref_via_symlink(tmp_path: Path):
    cfg = _cfg_and_roots(tmp_path)
    outside = _make_outside_target(cfg)
    link = Path(cfg.inbox.ruta) / "enlace.wav"
    link.symlink_to(outside)
    inbox.after_failure(cfg, _fake_job(str(link)), "some reason")
    assert outside.exists()
    assert link.exists()

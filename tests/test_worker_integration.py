"""Worker state machine: resume after a crash, the summary_pending flow, delivery holds
(ARCHITECTURE.md §1, §4, §8.4, §13; test plan §15 #4).

Every model is mocked — `audio.normalize`, `asr.run`, `diarize.run` and `summarize.summarize` are
replaced by fakes that count their calls — so these tests run in milliseconds and assert the one
thing the fakes cannot fake: which stages the worker decides to run. `merge`, `render` and
`indices` are the real implementations, so a delivered note here is a real note on a real (tmp)
Drive layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcription247 import asr as asr_mod
from transcription247 import audio as audio_mod
from transcription247 import diarize as diarize_mod
from transcription247 import render, summarize as summarize_mod, worker as worker_mod
from transcription247.models import (
    SCHEMA_VERSION,
    AsrResult,
    AsrSegment,
    DiarizationResult,
    NormalizedAudio,
    PuntoClave,
    SpeakerSegment,
    SummaryResult,
    Tarea,
    Word,
)
from transcription247.notify import NullNotifier
from transcription247.store import Store
from transcription247.worker import Worker

RECORDED_AT = "2026-09-19T12:35:11+02:00"


# --------------------------------------------------------------------------------------- fakes


class Calls(dict):
    """Per-stage call counter with an optional injected failure."""

    def __init__(self):
        super().__init__(normalize=0, asr=0, diarize=0, summarize=0)
        self.fail: dict[str, Exception] = {}

    def break_summary(self, message: str = "claude CLI not logged in") -> None:
        """Make the summary stage raise, the way a logged-out/rate-limited CLI does (§13 #3).

        Injected through the fake rather than by re-patching, because `monkeypatch.undo()` would
        also restore the *real* `summarize.summarize` — and this suite must never make a live
        `claude` call.
        """
        self.fail["summarize"] = summarize_mod.SummaryError(message)

    def fix_summary(self) -> None:
        self.fail.pop("summarize", None)

    def hit(self, stage: str):
        self[stage] += 1
        exc = self.fail.get(stage)
        if exc is not None:
            raise exc


@pytest.fixture
def calls(monkeypatch, cfg) -> Calls:
    counter = Calls()

    def fake_normalize(src, dst_wav, **kw):
        counter.hit("normalize")
        Path(dst_wav).parent.mkdir(parents=True, exist_ok=True)
        Path(dst_wav).write_bytes(b"RIFF" + b"\0" * 64)
        return NormalizedAudio(
            schema=SCHEMA_VERSION, job_id=kw.get("job_id", ""), source_name=kw.get("source_name", "x.m4a"),
            source_sha256=kw.get("source_sha256") or "0" * 64, source_bytes=1234, wav_path=str(dst_wav),
            duration_s=240.0, sample_rate=16000, channels=1, container="mov,mp4,m4a", codec="aac",
            creation_time=RECORDED_AT, start_source="metadata",
        )

    def fake_asr_run(wav, cfg_asr, *, job_id="", duration_s=None, **kw):
        counter.hit("asr")
        words = [
            Word(start=1.0, end=1.4, word=" Hola.", probability=0.99),
            Word(start=1.5, end=2.0, word=" Buenas.", probability=0.98),
            Word(start=60.0, end=60.5, word=" Perfecto.", probability=0.97),
            Word(start=60.6, end=61.0, word=" Gracias.", probability=0.96),
        ]
        return AsrResult(
            schema=SCHEMA_VERSION, job_id=job_id, language="es", language_probability=0.99,
            language_forced=False, duration_s=240.0, model="fake",
            segments=[AsrSegment(id=0, start=1.0, end=61.0, text=" Hola…", avg_logprob=-0.2, no_speech_prob=0.01)],
            words=words,
        )

    def fake_diarize_run(wav, cfg_diar, *, duration_s=0.0, job_id="", **kw):
        counter.hit("diarize")
        return DiarizationResult(
            schema=SCHEMA_VERSION, job_id=job_id, backend="fake", num_speakers_detected=2,
            segments=[
                SpeakerSegment(start=0.5, end=30.0, speaker=0),
                SpeakerSegment(start=30.0, end=120.0, speaker=1),
            ],
        )

    def fake_summarize(turns, hints, summary_cfg, runtime=None):
        counter.hit("summarize")  # raises when calls.break_summary() armed a failure
        return SummaryResult(
            schema=SCHEMA_VERSION, job_id=turns.job_id, pending=False, chunks=1, model="sonnet",
            titulo="Reunión de prueba", resumen_corto="Un resumen corto.", resumen="Un párrafo.",
            puntos_clave=[PuntoClave(emoji="📌", tema="Tema", texto="Texto")],
            tareas=[Tarea(responsable="Owner (tú)", texto="Preparar el acta", t_seg=12)],
            temas=["Tema"], palabras_clave=["prueba", "acta"], categoria="tutoria", idioma="es",
            cliente_sugerido=None, confianza=0.9,
        )

    monkeypatch.setattr(audio_mod, "normalize", fake_normalize)
    monkeypatch.setattr(asr_mod, "run", fake_asr_run)
    monkeypatch.setattr(diarize_mod, "run", fake_diarize_run)
    monkeypatch.setattr(summarize_mod, "summarize", fake_summarize)
    return counter


@pytest.fixture
def store(cfg) -> Store:
    s = Store.open(worker_mod.db_path(cfg))
    yield s
    s.close()


@pytest.fixture
def worker(cfg, store) -> Worker:
    return Worker(cfg, store, notifier=NullNotifier())


def _enqueue(cfg, store, *, title="Reunión de prueba", client=None, sha="a" * 64):
    job = store.create_job(
        source_name="reunion.m4a", origin="upload", audio_sha256=sha,
        title_user=title, client_user=client,
    )
    source = worker_mod.job_dir(cfg, job.id) / "source.m4a"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"fake audio")
    return store.advance_stage(job.id, "received", source_path=str(source))


# --------------------------------------------------------------------------------- happy path


def test_full_pipeline_delivers_a_real_note(cfg, store, worker, calls):
    job = _enqueue(cfg, store)
    worker.process_job(job)

    delivered = store.get_job(job.id)
    assert delivered.state == "delivered"
    assert delivered.client_key == "sin_clasificar"
    assert delivered.client_source == "fallback"

    note = Path(delivered.note_path)
    assert note.is_file()
    text = note.read_text(encoding="utf-8")
    assert "# Reunión de prueba" in text
    assert "Resumen pendiente" not in text
    assert "**Speaker 1** · [00:01]" in text

    # every artifact is on disk, and the audio is not (borrar_audio_tras_entrega)
    directory = worker_mod.job_dir(cfg, job.id)
    for name in ("normalized.json", "asr.json", "diarization.json", "turns.json", "summary.json"):
        assert (directory / name).is_file(), name
    assert not (directory / "audio.wav").exists()
    assert not list(directory.glob("source.*"))

    # and the indices were rebuilt from the filesystem
    root = Path(cfg.archivo.raiz)
    assert (root / "INDICE.md").is_file()
    assert (root / "indice_transcripciones.json").is_file()
    assert (root / "LEEME.md").is_file()
    assert (root / "_Sin_clasificar" / "INDICE.md").is_file()


# ----------------------------------------------------------------------- resume after a crash


def test_resume_after_crash_mid_pipeline_reuses_finished_stages(cfg, store, worker, calls):
    """Diarization blows up, then succeeds on the next pass: the normalize and ASR stages must
    not run a second time — their artifacts are reused (§4 "Stage skip / resume", §13 #10)."""
    job = _enqueue(cfg, store)
    calls.fail["diarize"] = diarize_mod.DiarizeError("boom: onnxruntime died")

    worker.process_job(job)

    after_crash = store.get_job(job.id)
    assert after_crash.state == "transcribed"  # last completed stage
    assert after_crash.attempts == 1
    assert after_crash.next_attempt_at is not None  # backed off, not failed
    assert after_crash.lease_until is None  # lease released, so a restart can claim it
    assert calls == {"normalize": 1, "asr": 1, "diarize": 1, "summarize": 0}

    directory = worker_mod.job_dir(cfg, job.id)
    assert (directory / "asr.json").is_file()
    assert not (directory / "diarization.json").exists()

    # a restart clears the backoff (simulated) and re-claims the job
    store.advance_stage(job.id, "transcribed", next_attempt_at=None)
    assert store.requeue_stale() == 0  # the lease was already released cleanly
    calls.fail.clear()
    worker.process_job(store.claim_next(cfg.worker.lease_minutes))

    assert store.get_job(job.id).state == "delivered"
    assert calls == {"normalize": 1, "asr": 1, "diarize": 2, "summarize": 1}


def test_corrupt_artifact_is_discarded_and_its_stage_re_runs(cfg, store, worker, calls):
    """A half-written artifact (the realistic outcome of an OOM kill mid-write) must not be
    trusted: it is deleted and the stage runs again (§4)."""
    job = _enqueue(cfg, store)
    worker.process_job(job)
    assert calls["asr"] == 1

    directory = worker_mod.job_dir(cfg, job.id)
    (directory / "asr.json").write_text('{"schema": 1, "job_id": "trunc', encoding="utf-8")
    (directory / "audio.wav").write_bytes(b"RIFF")  # deleted after delivery; the stage needs it

    store.advance_stage(job.id, "normalized", note_path=None)
    worker.process_job(store.claim_next(cfg.worker.lease_minutes))

    assert calls["asr"] == 2
    assert store.get_job(job.id).state == "delivered"


def test_stale_lease_is_requeued_on_startup(cfg, store, worker, calls):
    """The OOM-killed worker case (§13 #10): the lease outlives the process, startup clears it."""
    job = _enqueue(cfg, store)
    store.advance_stage(job.id, "transcribed", lease_until="2020-01-01T00:00:00+01:00")
    assert store.claim_next(cfg.worker.lease_minutes) is not None  # a past lease is claimable

    store.advance_stage(job.id, "transcribed", lease_until="2020-01-01T00:00:00+01:00")
    assert store.requeue_stale() == 1
    assert store.get_job(job.id).lease_until is None


def test_no_speech_fails_terminally_without_a_note(cfg, store, worker, calls, monkeypatch):
    """§13 #7: zero words is a hard failure, no retries, no note."""
    def silent_asr(wav, cfg_asr, *, job_id="", duration_s=None, **kw):
        calls.hit("asr")
        return AsrResult(
            schema=SCHEMA_VERSION, job_id=job_id, language="es", language_probability=0.5,
            language_forced=False, duration_s=240.0, model="fake", segments=[], words=[],
        )

    monkeypatch.setattr(asr_mod, "run", silent_asr)
    job = _enqueue(cfg, store)
    worker.process_job(job)

    failed = store.get_job(job.id)
    assert failed.state == "failed"
    assert failed.error == "no_speech"
    assert failed.note_path is None
    assert calls["diarize"] == 0


# --------------------------------------------------------------------------- summary_pending


def test_summary_failure_still_delivers_with_the_pending_marker(cfg, store, worker, calls):
    """§8.4: the note ships, the job lands in summary_pending, the marker is in the body."""
    calls.break_summary()
    job = _enqueue(cfg, store)
    worker.process_job(job)

    pending = store.get_job(job.id)
    assert pending.state == "summary_pending"
    assert "not logged in" in (pending.error or "")
    note = Path(pending.note_path)
    assert "⚠️ **Resumen pendiente**" in note.read_text(encoding="utf-8")

    summary_json = json.loads((worker_mod.job_dir(cfg, job.id) / "summary.json").read_text("utf-8"))
    assert summary_json["pending"] is True

    catalog = json.loads((Path(cfg.archivo.raiz) / "indice_transcripciones.json").read_text("utf-8"))
    assert catalog["reuniones"][0]["resumen_pendiente"] is True


def test_pending_summary_retry_pass_rewrites_the_same_note(cfg, store, worker, calls):
    """The periodic pass re-summarizes and overwrites the note we wrote, in place, and the job
    becomes `delivered` (§8.4, D4)."""
    calls.break_summary()
    job = _enqueue(cfg, store)
    worker.process_job(job)
    original_path = Path(store.get_job(job.id).note_path)

    calls.fix_summary()
    assert worker.retry_pending_summaries() == 1

    done = store.get_job(job.id)
    assert done.state == "delivered"
    assert Path(done.note_path) == original_path  # same file, overwritten in place
    text = original_path.read_text(encoding="utf-8")
    assert "Resumen pendiente" not in text
    assert "Un resumen corto." in text
    assert not list(original_path.parent.glob("* (2).md"))  # no sibling was created

    catalog = json.loads((Path(cfg.archivo.raiz) / "indice_transcripciones.json").read_text("utf-8"))
    assert catalog["reuniones"][0]["resumen_pendiente"] is False


def test_pending_retry_never_overwrites_a_hand_edited_note(cfg, store, worker, calls):
    """§9.6: once the body no longer matches its own hash_contenido the file is the user's, so the
    automatic pass leaves it alone (and stops retrying it) instead of destroying the edit."""
    calls.break_summary()
    job = _enqueue(cfg, store)
    worker.process_job(job)
    note = Path(store.get_job(job.id).note_path)

    edited = note.read_text(encoding="utf-8") + "\n\nNota a mano del owner.\n"
    note.write_text(edited, encoding="utf-8")

    calls.fix_summary()
    assert worker.retry_pending_summaries() == 0
    assert note.read_text(encoding="utf-8") == edited
    assert store.get_job(job.id).state == "summary_pending"
    assert job.id in worker._hand_edited

    # the explicit command is still allowed to act, and preserves the edit as a (2) sibling
    assert worker.resummarize_job(job.id, manual=True) is True
    assert job.id not in worker._hand_edited
    assert note.read_text(encoding="utf-8") == edited
    sibling = Path(store.get_job(job.id).note_path)
    assert sibling.name.endswith(" (2).md")
    assert store.get_job(job.id).state == "delivered"


# ------------------------------------------------------------------------------- holds (§13)


def test_drive_down_holds_delivery_without_failing_the_job(cfg, store, worker, calls, monkeypatch):
    """§13 #1: the note cannot be written, so the job waits at `summarized` with a retry time —
    attempts untouched, nothing failed, nothing lost."""
    job = _enqueue(cfg, store)
    monkeypatch.setattr(worker_mod, "_drive_up", lambda _cfg: False)
    worker.process_job(job)

    held = store.get_job(job.id)
    assert held.state == "summarized"
    assert held.attempts == 0
    assert held.next_attempt_at is not None
    assert held.lease_until is None
    assert held.note_path is None

    monkeypatch.undo()
    worker.process_job(store.get_job(job.id))
    assert store.get_job(job.id).state == "delivered"
    assert calls["asr"] == 1  # the whole pipeline was reused from artifacts


def test_low_memory_holds_the_job_instead_of_failing_it(cfg, store, worker, calls, monkeypatch):
    """§13 #5: below `mem_disponible_minima_mb` the job stays queued; it is never a failure."""
    object.__setattr__(cfg.worker, "mem_disponible_minima_mb", 999_999)
    object.__setattr__(cfg.worker, "mem_max_esperas", 1)
    monkeypatch.setattr(worker_mod, "mem_available_mb", lambda: 128)

    job = _enqueue(cfg, store)
    worker.process_job(job)

    held = store.get_job(job.id)
    assert held.state == "received"
    assert held.attempts == 0
    assert held.next_attempt_at is not None
    assert calls["normalize"] == 0


def test_shutdown_checkpoints_between_stages(cfg, store, worker, calls):
    """SIGTERM handling (§11): the stage in flight finishes, its artifact is kept, the lease is
    released and nothing else runs."""
    job = _enqueue(cfg, store)
    worker.shutdown.set()
    worker.process_job(job)

    stopped = store.get_job(job.id)
    assert stopped.state == "normalized"
    assert stopped.lease_until is None
    assert calls == {"normalize": 1, "asr": 0, "diarize": 0, "summarize": 0}
    assert (worker_mod.job_dir(cfg, job.id) / "normalized.json").is_file()


# -------------------------------------------------------------------------------- pruning


def test_prune_removes_old_finished_jobs_only(cfg, store, worker, calls):
    import os
    import time

    job = _enqueue(cfg, store)
    worker.process_job(job)
    pending = _enqueue(cfg, store, sha="b" * 64)
    store.mark_summary_pending(pending.id, "/tmp/x.md", "pending")

    old = time.time() - (cfg.worker.conservar_artefactos_dias + 1) * 86400
    for job_id in (job.id, pending.id):
        os.utime(worker_mod.job_dir(cfg, job_id), (old, old))

    assert worker.prune_artifacts() == 1
    assert not worker_mod.job_dir(cfg, job.id).exists()
    assert worker_mod.job_dir(cfg, pending.id).exists()  # resummarize still needs its turns.json


def test_resummarize_refuses_a_job_that_is_not_summary_pending(cfg, store, worker, calls):
    """A finished job must not be re-summarized by accident: that is a real `claude` call and,
    because a fresh `titulo` means a fresh path, it can silently rename the delivered note."""
    from transcription247.store import StoreError

    job = _enqueue(cfg, store)
    worker.process_job(job)
    assert store.get_job(job.id).state == "delivered"

    before = calls["summarize"]
    with pytest.raises(StoreError, match="not 'summary_pending'"):
        worker.resummarize_job(job.id, manual=True)
    assert calls["summarize"] == before

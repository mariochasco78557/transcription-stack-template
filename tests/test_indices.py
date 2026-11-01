"""Small indices rebuild test (ARCHITECTURE.md §10, §15).

Writes a couple of notes straight through render.write_note (so front-matter is realistic), then
checks that indices.rebuild_all() produces a root INDICE.md, a per-client INDICE.md per client
folder actually present on disk, indice_transcripciones.json sorted newest-first, and LEEME.md —
and that a per-client index disappears once that client's only note is removed (D8: filesystem is
the source of truth, so a hand-deleted note makes the index self-heal).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from transcription247 import indices, paths, render
from transcription247.config import (
    ApiConfig,
    ArchivoConfig,
    AsrConfig,
    AudioConfig,
    ClienteConfig,
    Config,
    DiarizationConfig,
    GeneralConfig,
    InboxConfig,
    LoggingConfig,
    MergeConfig,
    PathsConfig,
    SummaryConfig,
    WorkerConfig,
)
from transcription247.models import SummaryResult, Turn, TurnsResult


def _make_config() -> Config:
    archivo = ArchivoConfig(
        raiz="/tmp/does-not-matter",
        clientes={
            "centro_ejemplo": ClienteConfig(
                key="centro_ejemplo", carpeta="Centro_Ejemplo",
                nombre="Centro Ejemplo", marcadores=(),
            ),
            "cliente_alfa": ClienteConfig(key="cliente_alfa", carpeta="Cliente_Alfa", nombre="Cliente Alfa", marcadores=()),
            "sin_clasificar": ClienteConfig(
                key="sin_clasificar", carpeta="_Sin_clasificar", nombre="Sin clasificar", marcadores=(),
            ),
        },
        categorias={"tutoria": "Tutoría", "otras": "Otras"},
    )
    return Config(
        general=GeneralConfig(),
        paths=PathsConfig(),
        audio=AudioConfig(),
        asr=AsrConfig(),
        diarization=DiarizationConfig(),
        merge=MergeConfig(),
        summary=SummaryConfig(),
        archivo=archivo,
        inbox=InboxConfig(ruta="/tmp/inbox"),
        api=ApiConfig(),
        worker=WorkerConfig(),
        logging=LoggingConfig(),
        project_root=Path("/tmp"),
    )


def _write_note(cfg, *, job_id, titulo, client_field, inicio, duration_s=90.0) -> Path:
    turns = TurnsResult(
        schema=1, job_id=job_id, language="es", duration_s=duration_s,
        participantes=["Speaker 1"],
        turns=[Turn(start=0.5, end=duration_s, speaker=0, speaker_label="Speaker 1", text="Hola a todos.")],
    )
    summary = SummaryResult(
        schema=1, job_id=job_id, pending=False, chunks=1, model="sonnet",
        resumen_corto="Resumen breve.", resumen="Resumen.", categoria="tutoria", idioma="es",
        palabras_clave=["prueba"], confianza=0.9,
    )
    cliente = render.resolve_client(cfg, client_field=client_field, titulo_usuario=titulo, summary=summary)
    titulo_resuelto, content = render.render_markdown(
        cfg, turns, summary, job_id=job_id, titulo_usuario=titulo, inicio=inicio,
        audio_sha256="0" * 64, cliente=cliente,
    )
    target = paths.note_path(cliente.carpeta, inicio, titulo_resuelto)
    return render.write_note(target, content)


@pytest.fixture(autouse=True)
def _tmp_roots(tmp_path: Path):
    output_root = tmp_path / "drive" / "Transcripciones_24_7"
    inbox_root = output_root / "_INBOX_AUDIO"
    data_root = tmp_path / "data"
    output_root.mkdir(parents=True)
    inbox_root.mkdir(parents=True)
    data_root.mkdir(parents=True)
    paths.set_roots(output_root=output_root, inbox_root=inbox_root, data_root=data_root)
    yield output_root, inbox_root, data_root


def test_rebuild_all_produces_indices_sorted_newest_first(_tmp_roots):
    output_root, _, _ = _tmp_roots
    cfg = _make_config()
    tz = ZoneInfo("Europe/Madrid")

    note1 = _write_note(
        cfg, job_id="job1", titulo="Reunión antigua", client_field="centro_ejemplo",
        inicio=datetime(2026, 1, 10, 9, 0, 0, tzinfo=tz),
    )
    note2 = _write_note(
        cfg, job_id="job2", titulo="Reunión reciente", client_field="centro_ejemplo",
        inicio=datetime(2026, 9, 19, 12, 35, 11, tzinfo=tz),
    )
    _write_note(
        cfg, job_id="job3", titulo="ReuniÃ³n de producto", client_field="cliente_alfa",
        inicio=datetime(2026, 5, 1, 10, 0, 0, tzinfo=tz),
    )
    assert note1.exists() and note2.exists()

    indices.rebuild_all(cfg)

    root_index = (output_root / "INDICE.md").read_text(encoding="utf-8")
    assert "reuniones: 3" in root_index
    assert root_index.index("Reunión reciente") < root_index.index("Reunión antigua")

    assert (output_root / "Centro_Ejemplo" / "INDICE.md").exists()
    assert (output_root / "Cliente_Alfa" / "INDICE.md").exists()
    assert not (output_root / "_Sin_clasificar" / "INDICE.md").exists()

    catalog = json.loads((output_root / "indice_transcripciones.json").read_text(encoding="utf-8"))
    assert catalog["version"] == cfg.general.version_modulo
    assert len(catalog["reuniones"]) == 3
    assert [r["titulo"] for r in catalog["reuniones"][:2]] == ["Reunión reciente", "ReuniÃ³n de producto"]
    assert all(r["resumen_pendiente"] is False for r in catalog["reuniones"])

    assert (output_root / "LEEME.md").exists()


def test_rebuild_all_removes_stale_client_index_when_notes_disappear(_tmp_roots):
    output_root, _, _ = _tmp_roots
    cfg = _make_config()
    tz = ZoneInfo("Europe/Madrid")

    note = _write_note(
        cfg, job_id="job1", titulo="Único corte", client_field="cliente_alfa",
        inicio=datetime(2026, 5, 1, 10, 0, 0, tzinfo=tz),
    )
    indices.rebuild_all(cfg)
    assert (output_root / "Cliente_Alfa" / "INDICE.md").exists()

    note.unlink()
    indices.rebuild_all(cfg)
    assert not (output_root / "Cliente_Alfa" / "INDICE.md").exists()

    catalog = json.loads((output_root / "indice_transcripciones.json").read_text(encoding="utf-8"))
    assert catalog["reuniones"] == []


def test_rebuild_all_tolerates_foreign_files(_tmp_roots):
    output_root, _, _ = _tmp_roots
    cfg = _make_config()
    tz = ZoneInfo("Europe/Madrid")

    _write_note(
        cfg, job_id="job1", titulo="Reunión", client_field="centro_ejemplo",
        inicio=datetime(2026, 9, 19, 12, 0, 0, tzinfo=tz),
    )
    foreign_dir = output_root / "Centro_Ejemplo" / "2026-09"
    (foreign_dir / "notas-sueltas.md").write_text("# No es una nota generada\n\nTexto libre.\n", encoding="utf-8")

    indices.rebuild_all(cfg)  # must not raise

    catalog = json.loads((output_root / "indice_transcripciones.json").read_text(encoding="utf-8"))
    assert len(catalog["reuniones"]) == 1

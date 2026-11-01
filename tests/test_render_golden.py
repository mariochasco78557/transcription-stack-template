"""Golden-file test for render.py (ARCHITECTURE.md §9, §15 #2).

Fixed inputs + a fixed `generado_en` -> byte-exact comparison with tests/golden/*.md. The
non-pending fixture mirrors the worked example in ARCHITECTURE.md §9.5 exactly (same job id,
speakers, timestamps, front-matter values); its body hashes to the same SHA-256 as the doc's own
example body text, which is strong independent confirmation that `hash_contenido` covers exactly
what §9.3/§9.6 specify ("everything after the front-matter's closing --- and the blank line
following it").
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from transcription247 import paths, render
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
from transcription247.models import (
    MasDetalleItem,
    PuntoClave,
    SummaryResult,
    Tarea,
    Turn,
    TurnsResult,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
JOB_ID = "01k5r2m8q7v3x9b0c1d2e3f4g5"
AUDIO_SHA256 = "87aa1a29fc25fdcb13b368e26d2f7032877a89b0c2973f77d2521be7d9dfeb41"
INICIO = datetime(2026, 9, 19, 12, 35, 11, tzinfo=ZoneInfo("Europe/Madrid"))
GENERADO_EN = datetime(2026, 9, 19, 12, 41, 2, tzinfo=ZoneInfo("Europe/Madrid"))


def _make_config(**archivo_overrides) -> Config:
    archivo = ArchivoConfig(
        raiz="/tmp/does-not-matter",
        clientes={
            "cliente_alfa": ClienteConfig(
                key="cliente_alfa",
                carpeta="Cliente_Alfa",
                nombre="Cliente Alfa",
                marcadores=("alfa",),
            ),
            "sin_clasificar": ClienteConfig(
                key="sin_clasificar", carpeta="_Sin_clasificar", nombre="Sin clasificar", marcadores=(),
            ),
        },
        categorias={"producto": "Producto", "otras": "Otras"},
        **archivo_overrides,
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


def _make_turns() -> TurnsResult:
    return TurnsResult(
        schema=1,
        job_id=JOB_ID,
        language="es",
        duration_s=241.44,
        participantes=["Speaker 1", "Speaker 2"],
        turns=[
            Turn(
                start=1.02, end=121.44, speaker=0, speaker_label="Speaker 1",
                text="Buenos días. Hoy repasamos el roadmap del tercer trimestre y la fecha de lanzamiento de la aplicación móvil.",
            ),
            Turn(
                start=121.44, end=241.44, speaker=1, speaker_label="Speaker 2",
                text="Perfecto. Propongo cerrar el alcance esta semana y fijar el lanzamiento para mediados de octubre.",
            ),
        ],
    )


def _make_summary() -> SummaryResult:
    return SummaryResult(
        schema=1, job_id=JOB_ID, pending=False, chunks=1, model="sonnet", titulo=None,
        resumen_corto="El equipo de producto acuerda cerrar el alcance del roadmap y fijar el lanzamiento móvil.",
        resumen="El equipo de producto revisa el roadmap del tercer trimestre y acuerda cerrar el alcance y fijar la fecha de lanzamiento de la aplicación móvil.",
        mas_detalle=[
            MasDetalleItem(tema="Alcance", texto="Se congela la lista de funcionalidades del lanzamiento esta semana."),
            MasDetalleItem(tema="Calendario", texto="El lanzamiento se fija para mediados de octubre."),
        ],
        puntos_clave=[
            PuntoClave(emoji="🗺️", tema="Roadmap", texto="Las funcionalidades se ordenan por impacto y esfuerzo."),
            PuntoClave(emoji="🚀", tema="Lanzamiento", texto="La versión móvil sale a mediados de octubre."),
        ],
        tareas=[
            Tarea(responsable="Producto", texto="Publicar el alcance cerrado del roadmap", t_seg=12),
            Tarea(responsable="Ingeniería", texto="Confirmar la fecha de lanzamiento con el equipo", t_seg=121),
        ],
        temas=["Roadmap del tercer trimestre", "Lanzamiento de la aplicación móvil"],
        palabras_clave=["roadmap", "alcance", "lanzamiento", "aplicación móvil", "prioridades"],
        categoria="producto", idioma="es", cliente_sugerido=None, confianza=0.95,
    )


def test_render_markdown_matches_golden():
    cfg = _make_config()
    turns = _make_turns()
    summary = _make_summary()

    cliente = render.resolve_client(
        cfg, client_field="cliente_alfa", titulo_usuario="Revisión de roadmap de producto", summary=summary
    )
    assert cliente.source == "upload"

    titulo, content = render.render_markdown(
        cfg, turns, summary, job_id=JOB_ID, titulo_usuario="Revisión de roadmap de producto",
        inicio=INICIO, audio_sha256=AUDIO_SHA256, cliente=cliente, generado_en=GENERADO_EN,
    )

    assert titulo == "Revisión de roadmap de producto"
    expected = (GOLDEN_DIR / "expected_note.md").read_text(encoding="utf-8")
    assert content == expected


def test_render_markdown_pending_summary_matches_golden():
    cfg = _make_config()
    turns = _make_turns()
    summary = SummaryResult(
        schema=1, job_id=JOB_ID, pending=True, chunks=1, model="sonnet", error="claude_unavailable",
    )

    cliente = render.resolve_client(
        cfg, client_field=None, titulo_usuario="Revisión de roadmap de producto", summary=summary
    )
    assert cliente.key == "sin_clasificar"
    assert cliente.source == "fallback"

    titulo, content = render.render_markdown(
        cfg, turns, summary, job_id=JOB_ID, titulo_usuario="Revisión de roadmap de producto",
        inicio=INICIO, audio_sha256=AUDIO_SHA256, cliente=cliente, generado_en=GENERADO_EN,
    )

    expected = (GOLDEN_DIR / "expected_note_pending.md").read_text(encoding="utf-8")
    assert content == expected
    assert "⚠️ **Resumen pendiente**" in content
    assert "_Sin resumen listado._" in content


# ------------------------------------------------------------------------------- write_note (§9.6)


@pytest.fixture(autouse=True)
def _tmp_roots(tmp_path: Path):
    output_root = tmp_path / "drive" / "Transcripciones_24_7"
    inbox_root = tmp_path / "drive" / "Transcripciones_24_7" / "_INBOX_AUDIO"
    data_root = tmp_path / "data"
    output_root.mkdir(parents=True)
    inbox_root.mkdir(parents=True)
    data_root.mkdir(parents=True)
    paths.set_roots(output_root=output_root, inbox_root=inbox_root, data_root=data_root)
    yield output_root, inbox_root, data_root


def test_write_note_is_idempotent_on_rerun(_tmp_roots):
    output_root, _, _ = _tmp_roots
    target = output_root / "Cliente_Alfa" / "2026-09" / "nota.md"
    content = (GOLDEN_DIR / "expected_note.md").read_text(encoding="utf-8")

    written1 = render.write_note(target, content)
    assert written1 == target
    written2 = render.write_note(target, content)
    assert written2 == target
    assert target.read_text(encoding="utf-8") == content
    # no sibling created
    assert sorted(p.name for p in target.parent.iterdir()) == ["nota.md"]


def test_write_note_protects_hand_edited_file(_tmp_roots):
    output_root, _, _ = _tmp_roots
    target = output_root / "Cliente_Alfa" / "2026-09" / "nota.md"
    content = (GOLDEN_DIR / "expected_note.md").read_text(encoding="utf-8")

    render.write_note(target, content)
    # The owner edits the body by hand, front-matter's hash_contenido now stale.
    edited = target.read_text(encoding="utf-8").replace(
        "Las funcionalidades se ordenan por impacto", "Las funcionalidades se ordenan por prioridad"
    )
    target.write_text(edited, encoding="utf-8")

    written = render.write_note(target, content)
    assert written == output_root / "Cliente_Alfa" / "2026-09" / "nota (2).md"
    assert target.read_text(encoding="utf-8") == edited  # original untouched
    assert written.read_text(encoding="utf-8") == content

    # Re-running again with the same content reuses the (2) sibling, not a (3).
    written_again = render.write_note(target, content)
    assert written_again == written
    assert sorted(p.name for p in target.parent.iterdir()) == ["nota (2).md", "nota.md"]

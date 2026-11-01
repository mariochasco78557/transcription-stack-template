"""`## Tareas abiertas del owner` in each client index (ARCHITECTURE.md §10).

Parity feature with the old Fireflies archive: the per-client `INDICE.md` gathers the still
unchecked `- [ ]` items that each note files under a owner-owned `### <responsable>` heading,
newest meeting first, each linking back to its note. "owner-owned" is whatever
`archivo.responsables_owner` says (default `["Owner", "(tú)"]`), so the rule is configuration, not
a hard-coded name.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from transcription247 import indices, render
from transcription247.config import ClienteConfig
from transcription247.models import SCHEMA_VERSION, SummaryResult, Tarea, Turn, TurnsResult

MADRID = ZoneInfo("Europe/Madrid")


def _turns(job_id: str) -> TurnsResult:
    return TurnsResult(
        schema=SCHEMA_VERSION, job_id=job_id, language="es", duration_s=600.0,
        participantes=["Speaker 1"],
        turns=[Turn(start=1.0, end=9.0, speaker=0, speaker_label="Speaker 1", text="Hola.")],
    )


def _summary(job_id: str, tareas: list[Tarea]) -> SummaryResult:
    return SummaryResult(
        schema=SCHEMA_VERSION, job_id=job_id, pending=False, chunks=1, model="sonnet",
        resumen_corto="Corto.", resumen="Párrafo.", tareas=tareas, palabras_clave=["a"],
        categoria="tutoria", idioma="es", confianza=0.9,
    )


def _write_note(cfg, *, job_id: str, titulo: str, inicio: datetime, tareas: list[Tarea]) -> Path:
    cliente = render.resolve_client(
        cfg, client_field="centro_ejemplo", titulo_usuario=titulo, summary=_summary(job_id, tareas)
    )
    _titulo, markdown = render.render_markdown(
        cfg, _turns(job_id), _summary(job_id, tareas),
        job_id=job_id, titulo_usuario=titulo, inicio=inicio, audio_sha256="f" * 64, cliente=cliente,
    )
    from transcription247 import paths

    return render.write_note(paths.note_path(cliente.carpeta, inicio, titulo), markdown)


@pytest.fixture
def client_index(cfg) -> Path:
    """Two notes for the same client: an older one and a newer one, each with a mix of owners."""
    _write_note(
        cfg,
        job_id="01antigua",
        titulo="Claustro de septiembre",
        inicio=datetime(2026, 9, 10, 9, 0, tzinfo=MADRID),
        tareas=[
            Tarea(responsable="Tutores", texto="Repartir los libros", t_seg=30),
            Tarea(responsable="Owner (tú)", texto="Enviar el acta del claustro", t_seg=61),
        ],
    )
    _write_note(
        cfg,
        job_id="01reciente",
        titulo="Departamento de octubre",
        inicio=datetime(2026, 10, 2, 11, 30, tzinfo=MADRID),
        tareas=[
            Tarea(responsable="Owner (tú)", texto="Preparar la programación", t_seg=12),
            Tarea(responsable="Owner (tú)", texto="Hablar con jefatura", t_seg=200),
            Tarea(responsable="Speaker 2", texto="Revisar el material", t_seg=300),
        ],
    )
    indices.rebuild_all(cfg)
    return Path(cfg.archivo.raiz) / "Centro_Ejemplo" / "INDICE.md"


def test_client_index_lists_owners_open_tasks_newest_first(client_index: Path):
    text = client_index.read_text(encoding="utf-8")
    assert "## Tareas abiertas del owner" in text

    section = text.split("## Tareas abiertas del owner", 1)[1]
    assert "- [ ] Preparar la programación · [00:12]" in section
    assert "- [ ] Hablar con jefatura · [03:20]" in section
    assert "- [ ] Enviar el acta del claustro · [01:01]" in section

    # other people's tasks never leak in
    assert "Repartir los libros" not in section
    assert "Revisar el material" not in section

    # newest meeting first, each heading linking to its note (relative to the client folder)
    assert section.index("Departamento de octubre") < section.index("Claustro de septiembre")
    assert "2026-10/2026-10-02%2011h30%20-%20Departamento%20de%20octubre.md" in section


def test_checking_a_box_in_the_note_removes_it_from_the_index(cfg, client_index: Path):
    """The index is regenerated from the notes (D8), so ticking a box in a note is all it takes."""
    note = next((Path(cfg.archivo.raiz) / "Centro_Ejemplo").rglob("*Departamento*.md"))
    note.write_text(
        note.read_text(encoding="utf-8").replace(
            "- [ ] Hablar con jefatura", "- [x] Hablar con jefatura"
        ),
        encoding="utf-8",
    )
    indices.rebuild_all(cfg)

    section = client_index.read_text(encoding="utf-8").split("## Tareas abiertas del owner", 1)[1]
    assert "Hablar con jefatura" not in section
    assert "Preparar la programación" in section


def test_section_is_present_with_a_placeholder_when_nobody_owes_owner_anything(cfg):
    _write_note(
        cfg,
        job_id="01sinowner",
        titulo="Reunión sin tareas propias",
        inicio=datetime(2026, 9, 12, 10, 0, tzinfo=MADRID),
        tareas=[Tarea(responsable="Tutores", texto="Repartir los libros", t_seg=30)],
    )
    indices.rebuild_all(cfg)

    text = (Path(cfg.archivo.raiz) / "Centro_Ejemplo" / "INDICE.md").read_text(encoding="utf-8")
    assert "## Tareas abiertas del owner" in text
    assert "_Sin tareas abiertas._" in text


def test_owner_markers_come_from_config(cfg):
    """The default markers are `["Owner", "(tú)"]`; a deployment that files its own tasks under a
    different name only changes `archivo.responsables_owner`."""
    body = (
        "## Tareas y acuerdos\n\n"
        "### Dirección\n- [ ] Ajena\n\n"
        "### Coordinación TIC (tú)\n- [ ] Propia con sufijo\n- [x] Ya hecha\n\n"
        "### owner\n- [ ] Propia por nombre\n\n"
        "## Temas tratados\n\n- [ ] no es una tarea, es otra sección\n"
    )
    assert indices.open_tasks_for(body, cfg.archivo.responsables_owner) == [
        "Propia con sufijo",
        "Propia por nombre",
    ]
    assert indices.open_tasks_for(body, ("Dirección",)) == ["Ajena"]
    assert indices.open_tasks_for(body, ()) == []


def test_tasks_are_scoped_per_client(cfg):
    """A note filed under another client must not pull its tasks into this client's index."""
    cfg.archivo.clientes["cliente_alfa"] = ClienteConfig(
        key="cliente_alfa", carpeta="Cliente_Alfa", nombre="Cliente Alfa", marcadores=("alfa",)
    )
    _write_note(
        cfg, job_id="01cen", titulo="Solo Centro",
        inicio=datetime(2026, 9, 14, 10, 0, tzinfo=MADRID),
        tareas=[Tarea(responsable="Owner (tú)", texto="Tarea de la organización", t_seg=10)],
    )
    from transcription247 import paths

    summary = _summary("01tx", [Tarea(responsable="Owner (tú)", texto="Tarea de Alfa", t_seg=10)])
    cliente = render.resolve_client(cfg, client_field="cliente_alfa", titulo_usuario="Solo Alfa", summary=summary)
    inicio = datetime(2026, 9, 15, 10, 0, tzinfo=MADRID)
    _titulo, markdown = render.render_markdown(
        cfg, _turns("01tx"), summary, job_id="01tx", titulo_usuario="Solo Alfa",
        inicio=inicio, audio_sha256="a" * 64, cliente=cliente,
    )
    render.write_note(paths.note_path(cliente.carpeta, inicio, "Solo Alfa"), markdown)
    indices.rebuild_all(cfg)

    centro = (Path(cfg.archivo.raiz) / "Centro_Ejemplo" / "INDICE.md").read_text(encoding="utf-8")
    alfa = (Path(cfg.archivo.raiz) / "Cliente_Alfa" / "INDICE.md").read_text(encoding="utf-8")
    assert "Tarea de la organización" in centro and "Tarea de Alfa" not in centro
    assert "Tarea de Alfa" in alfa and "Tarea de la organización" not in alfa

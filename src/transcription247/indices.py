"""INDICE.md, per-client indices, JSON catalog, LEEME.md (ARCHITECTURE.md §10).

Always regenerated in full from a filesystem scan of `<Cliente>/AAAA-MM/*.md` front-matter (D8:
"self-heal" rather than incremental patching) — so a hand-deleted note disappears, a hand-edited
one is picked up, and a foreign file is silently ignored. Client grouping is derived entirely from
what the notes on disk say (`cliente`/`cliente_nombre` in their own front-matter), never from
`cfg.archivo.clientes`, so a client removed from config.toml after notes were written still gets an
index instead of losing it.
"""

from __future__ import annotations

import logging
import unicodedata
from datetime import datetime, timezone
from urllib.parse import quote

from transcription247 import paths
from transcription247.config import Config
from transcription247.render import (
    GENERADOR,
    TIPO_REUNION,
    atomic_write_text,
    parse_front_matter,
    render_front_matter,
)

log = logging.getLogger(__name__)

_CATALOG_NAME = "indice_transcripciones.json"
_ROOT_INDEX_NAME = "INDICE.md"
_LEEME_NAME = "LEEME.md"

#: Headings the "Tareas abiertas del owner" scan (§10) walks in each note's body.
_TAREAS_HEADING = "## Tareas y acuerdos"
_TAREAS_SECTION_TITLE = "Tareas abiertas del owner"
_TAREAS_PLACEHOLDER = "_Sin tareas abiertas._"

#: Catalog/table columns, in the order §10 documents.
_REUNION_FIELDS = (
    "id", "fecha", "hora", "inicio", "titulo", "cliente", "cliente_nombre", "categoria",
    "palabras_clave", "duracion_min", "idioma", "resumen_corto", "participantes",
    "resumen_pendiente", "ruta",
)


def rebuild_all(cfg: Config) -> None:
    """`t247 reindex`, and called after every delivery (§10). Rebuilds
    `INDICE.md` / `<Cliente>/INDICE.md` / `indice_transcripciones.json` / `LEEME.md` from scratch."""
    root = paths.OUTPUT_ROOT
    if root is None:
        raise RuntimeError("paths.configure()/set_roots() must be called before indices.rebuild_all()")
    root.mkdir(parents=True, exist_ok=True)

    reuniones = _scan_notes(root, cfg)
    reuniones.sort(key=lambda r: r["_inicio_dt"], reverse=True)

    now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    _write_catalog(root, cfg, reuniones, now_iso)
    _write_root_index(root, cfg, reuniones, now_iso)
    _write_client_indices(root, cfg, reuniones, now_iso)
    _write_leeme(root, cfg)


# --------------------------------------------------------------------------------------- scan


def _fold(s: str) -> str:
    """Case-insensitive, accent-folded form, as render.py uses for client markers (§9.1)."""
    normalized = unicodedata.normalize("NFKD", s)
    return "".join(c for c in normalized if not unicodedata.combining(c)).casefold()


def open_tasks_for(body: str, marcadores: tuple[str, ...] | list[str]) -> list[str]:
    """Unchecked `- [ ]` items under a owner-owned `### <responsable>` heading (§10).

    "owner-owned" is defined by `archivo.responsables_owner` (default `["Owner", "(tú)"]`), matched
    accent-folded and case-insensitively as a substring of the heading text — so `### Owner (tú)`,
    `### owner` and `### Dirección (tú)` all qualify while `### Tutores` does not. Only the
    `## Tareas y acuerdos` section is scanned, and only unchecked boxes: `- [x]` is done.
    Returns the task lines with the `- [ ] ` prefix stripped, in document order.
    """
    folded_markers = [_fold(m) for m in marcadores if m and m.strip()]
    if not folded_markers:
        return []

    out: list[str] = []
    in_section = False
    owned = False
    for raw in body.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            in_section = line.strip() == _TAREAS_HEADING
            owned = False
            continue
        if not in_section:
            continue
        if line.startswith("### "):
            heading = _fold(line[4:].strip())
            owned = any(marker in heading for marker in folded_markers)
            continue
        if owned and line.lstrip().startswith("- [ ] "):
            out.append(line.lstrip()[len("- [ ] ") :].strip())
    return out


def _scan_notes(root, cfg: Config) -> list[dict]:
    """`<Cliente>/AAAA-MM/*.md` only — one level for the client folder's own INDICE.md, zero for
    the root's, so neither is ever mistaken for a meeting note."""
    reuniones: list[dict] = []
    for md_path in sorted(root.glob("*/*/*.md")):
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("reindex: could not read %s: %s", md_path, exc)
            continue
        parsed = parse_front_matter(text)
        if parsed is None:
            continue  # foreign file, tolerated
        fields, body = parsed
        if fields.get("tipo") != TIPO_REUNION:
            continue
        try:
            inicio_dt = datetime.fromisoformat(str(fields["inicio"]))
        except (KeyError, ValueError):
            log.warning("reindex: skipping %s: unparseable/missing 'inicio'", md_path)
            continue
        ruta = md_path.relative_to(root).as_posix()
        reuniones.append(
            {
                "id": fields.get("origen_id", ""),
                "fecha": fields.get("fecha", ""),
                "hora": fields.get("hora", ""),
                "inicio": fields.get("inicio", ""),
                "titulo": fields.get("titulo", ""),
                "cliente": fields.get("cliente", ""),
                "cliente_nombre": fields.get("cliente_nombre", ""),
                "categoria": fields.get("categoria", ""),
                "palabras_clave": fields.get("palabras_clave") or [],
                "duracion_min": fields.get("duracion_min", 0),
                "idioma": fields.get("idioma", ""),
                "resumen_corto": fields.get("resumen_corto", ""),
                "participantes": fields.get("participantes") or [],
                "resumen_pendiente": "Resumen pendiente" in body,
                "ruta": ruta,
                "_inicio_dt": inicio_dt,
                "_carpeta": ruta.split("/", 1)[0],
                "_tareas_owner": open_tasks_for(body, cfg.archivo.responsables_owner),
            }
        )
    return reuniones


# --------------------------------------------------------------------------------------- catalog


def _write_catalog(root, cfg: Config, reuniones: list[dict], now_iso: str) -> None:
    import json

    payload = {
        "version": cfg.general.version_modulo,
        "generado_en": now_iso,
        "reuniones": [{k: r[k] for k in _REUNION_FIELDS} for r in reuniones],
    }
    atomic_write_text(root / _CATALOG_NAME, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


# ------------------------------------------------------------------------------------ table rows


def _escape_cell(text: str) -> str:
    return str(text).replace("|", "\\|")


def _link(text: str, href: str) -> str:
    return f"[{_escape_cell(text)}]({quote(href, safe='/')})"


def _table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _categoria_display(cfg: Config, key: str) -> str:
    return cfg.archivo.categorias.get(key, key)


# --------------------------------------------------------------------------------- root INDICE.md


def _write_root_index(root, cfg: Config, reuniones: list[dict], now_iso: str) -> None:
    groups = _group_by_client(reuniones)

    front_matter = render_front_matter(
        {
            "generador": GENERADOR,
            "tipo": "indice_archivo",
            "reuniones": len(reuniones),
            "version_modulo": cfg.general.version_modulo,
        }
    )

    lines = [
        "# Índice de transcripciones",
        "",
        f"_Generado automáticamente el {now_iso}. Ver [LEEME.md](LEEME.md) "
        f"y [{_CATALOG_NAME}]({_CATALOG_NAME})._",
        "",
    ]
    if groups:
        parts = []
        for g in groups:
            href = f"{g['carpeta']}/{_ROOT_INDEX_NAME}"
            parts.append(f"{_link(g['nombre'], href)} ({len(g['reuniones'])})")
        lines.append(f"**Por cliente:** {' · '.join(parts)}")
        lines.append("")

    lines.append(_table_row(["Fecha", "Hora", "Cliente", "Reunión", "Categoría", "Palabras clave", "Duración"]))
    lines.append(_table_row(["---"] * 7))
    for r in reuniones:
        lines.append(
            _table_row(
                [
                    _escape_cell(r["fecha"]),
                    _escape_cell(r["hora"]),
                    _escape_cell(r["cliente_nombre"]),
                    _link(r["titulo"], r["ruta"]),
                    _escape_cell(_categoria_display(cfg, r["categoria"])),
                    _escape_cell(", ".join(r["palabras_clave"])),
                    f"{r['duracion_min']} min",
                ]
            )
        )
    body = "\n".join(lines) + "\n"
    atomic_write_text(root / _ROOT_INDEX_NAME, front_matter + body)


# ------------------------------------------------------------------------------ per-client index


def _group_by_client(reuniones: list[dict]) -> list[dict]:
    """Group by the folder a note actually lives in (`_carpeta`), not by config — a client removed
    from config.toml after notes were written still gets an index (D8, self-heal)."""
    by_carpeta: dict[str, dict] = {}
    order: list[str] = []
    for r in reuniones:
        carpeta = r["_carpeta"]
        if carpeta not in by_carpeta:
            by_carpeta[carpeta] = {
                "carpeta": carpeta,
                "key": r["cliente"],
                "nombre": r["cliente_nombre"] or carpeta,
                "reuniones": [],
            }
            order.append(carpeta)
        by_carpeta[carpeta]["reuniones"].append(r)
    groups = [by_carpeta[c] for c in order]
    groups.sort(key=lambda g: _fold_for_sort(g["nombre"]))
    return groups


def _fold_for_sort(s: str) -> str:
    return s.casefold()


def _owner_tasks_lines(reuniones: list[dict]) -> list[str]:
    """The `## Tareas abiertas del owner` block of a client index (§10).

    `reuniones` is already sorted newest first, so the meetings appear in that order; each group is
    a link to its note followed by the still-unchecked `- [ ]` items, kept as checkboxes so the
    index reads as one live to-do list. The section is always emitted (with a placeholder when
    empty), matching the invariant-section style of the notes themselves (§9.4).
    """
    lines = ["", f"## {_TAREAS_SECTION_TITLE}", ""]
    emitted = False
    for r in reuniones:
        tareas = r.get("_tareas_owner") or []
        if not tareas:
            continue
        emitted = True
        ruta_relativa = r["ruta"].split("/", 1)[1] if "/" in r["ruta"] else r["ruta"]
        lines.append(f"**{_link(r['titulo'], ruta_relativa)}** · {_escape_cell(r['fecha'])}")
        for tarea in tareas:
            lines.append(f"- [ ] {tarea}")
        lines.append("")
    if not emitted:
        lines.append(_TAREAS_PLACEHOLDER)
        lines.append("")
    return lines[:-1] if lines and lines[-1] == "" else lines


def _write_client_indices(root, cfg: Config, reuniones: list[dict], now_iso: str) -> None:
    groups = _group_by_client(reuniones)
    live_carpetas = {g["carpeta"] for g in groups}

    for g in groups:
        carpeta_dir = root / g["carpeta"]
        front_matter = render_front_matter(
            {
                "generador": GENERADOR,
                "tipo": "indice_cliente",
                "cliente": g["key"],
                "cliente_nombre": g["nombre"],
                "reuniones": len(g["reuniones"]),
                "version_modulo": cfg.general.version_modulo,
            }
        )
        lines = [
            f"# {g['nombre']}",
            "",
            f"_Generado automáticamente el {now_iso}. Ver [índice general](../{_ROOT_INDEX_NAME}) "
            f"y [LEEME.md](../{_LEEME_NAME})._",
            "",
            _table_row(["Fecha", "Hora", "Reunión", "Categoría", "Palabras clave", "Duración"]),
            _table_row(["---"] * 6),
        ]
        for r in g["reuniones"]:
            ruta_relativa = r["ruta"].split("/", 1)[1] if "/" in r["ruta"] else r["ruta"]
            lines.append(
                _table_row(
                    [
                        _escape_cell(r["fecha"]),
                        _escape_cell(r["hora"]),
                        _link(r["titulo"], ruta_relativa),
                        _escape_cell(_categoria_display(cfg, r["categoria"])),
                        _escape_cell(", ".join(r["palabras_clave"])),
                        f"{r['duracion_min']} min",
                    ]
                )
            )
        lines.extend(_owner_tasks_lines(g["reuniones"]))
        body = "\n".join(lines) + "\n"
        atomic_write_text(carpeta_dir / _ROOT_INDEX_NAME, front_matter + body)

    # A client INDICE.md exists only while that client has >=1 note (§10).
    for existing in root.glob(f"*/{_ROOT_INDEX_NAME}"):
        carpeta = existing.parent.name
        if carpeta not in live_carpetas:
            try:
                paths.guard_write(existing)
                existing.unlink()
            except OSError as exc:
                log.warning("reindex: could not remove stale %s: %s", existing, exc)


# ----------------------------------------------------------------------------------------- LEEME


def _write_leeme(root, cfg: Config) -> None:
    leeme_path = root / _LEEME_NAME
    if leeme_path.exists():
        try:
            existing = parse_front_matter(leeme_path.read_text(encoding="utf-8"))
        except OSError:
            existing = None
        if existing is not None and existing[0].get("version_modulo") == cfg.general.version_modulo:
            return  # up to date, leave any manual edits alone

    front_matter = render_front_matter(
        {"generador": GENERADOR, "tipo": "leeme", "version_modulo": cfg.general.version_modulo}
    )

    clientes_rows = "\n".join(
        _table_row([_escape_cell(c.key), _escape_cell(c.carpeta), _escape_cell(c.nombre)])
        for c in cfg.archivo.clientes.values()
    )

    body = f"""# LEEME

Este directorio contiene las notas de reunión generadas automáticamente por Transcription 24/7
(`version_modulo {cfg.general.version_modulo}`) a partir de grabaciones de audio. No se guarda
texto de transcripción fuera de cada nota individual.

## Estructura

```
<Cliente>/AAAA-MM/AAAA-MM-DD HHhMM - Título.md   una nota por reunión
<Cliente>/INDICE.md                              índice de ese cliente (si tiene >=1 nota)
{_ROOT_INDEX_NAME}                               índice general (todas las reuniones)
{_CATALOG_NAME}                    catálogo en JSON (sin texto de transcripción)
{_LEEME_NAME}                                    este fichero
```

## Clientes configurados

| Clave | Carpeta | Nombre |
| --- | --- | --- |
{clientes_rows}

## Cabecera de cada nota (front-matter)

generador, tipo, origen_id, titulo, fecha, hora, inicio, duracion_min, idioma, categoria,
palabras_clave, participantes, enlace, resumen_corto, cliente, cliente_nombre, hash_origen,
version_modulo, generado_en, hash_contenido.

## Secciones de cada nota

Resumen, Puntos clave, Tareas y acuerdos, Temas tratados, Transcripción completa.

## Tareas abiertas del owner

Cada `<Cliente>/INDICE.md` termina con una sección **{_TAREAS_SECTION_TITLE}**: las tareas sin
marcar (`- [ ]`) que en cada nota aparecen bajo un responsable propio (por defecto los que
contienen «Owner» o «(tú)», configurable en `archivo.responsables_owner`), de la reunión más
reciente a la más antigua y con enlace a la nota. Marcar una casilla en la nota original la retira
del índice en la siguiente regeneración.

## Catálogo `{_CATALOG_NAME}`

Objeto con `version`, `generado_en` y `reuniones` (lista). Cada reunión: {", ".join(_REUNION_FIELDS)}.

## Hablantes

Los hablantes se numeran por orden de primera intervención (Speaker 1, Speaker 2, …); no se
identifican por nombre.
"""
    atomic_write_text(leeme_path, front_matter + body)

"""Markdown note renderer + atomic, non-destructive write (ARCHITECTURE.md §9).

Mirrors an existing Fireflies archive format byte-for-byte so downstream tools keep working:
YAML-ish front-matter with one JSON-encoded value per line, then a body with fixed Spanish
headings. `render_markdown()` is pure (no I/O); `write_note()` is the only place that touches the
filesystem, and it always goes through `paths.guard_write`.
"""

from __future__ import annotations

import json
import logging
import os
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from transcription247.config import Config
from transcription247.ids import sha256_text
from transcription247.models import SummaryResult, TurnsResult
from transcription247.paths import fmt_long_date_es, fmt_ts, guard_write, sanitize_title

log = logging.getLogger(__name__)

#: Fixed front-matter values (§9.4).
GENERADOR = "transcription-24-7"
TIPO_REUNION = "reunion"

#: Placeholder text for otherwise-empty sections, matching the "_Sin … listados._" style (§9.4).
_PLACEHOLDERS = {
    "resumen": "_Sin resumen listado._",
    "puntos_clave": "_Sin puntos clave listados._",
    "tareas": "_Sin tareas listadas._",
    "temas": "_Sin temas listados._",
}


# --------------------------------------------------------------------------------- client (§9.1)


@dataclass(slots=True)
class ClientResolution:
    key: str
    nombre: str
    carpeta: str
    source: str  # "upload" | "marker" | "claude" | "fallback"


def _fold(s: str) -> str:
    """Case-insensitive, accent-folded form for marker matching (§9.1)."""
    normalized = unicodedata.normalize("NFKD", s)
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


def resolve_client(
    cfg: Config,
    *,
    client_field: str | None,
    titulo_usuario: str | None,
    summary: SummaryResult,
) -> ClientResolution:
    """First hit wins (§9.1), recorded as `source`:
    (1) explicit `client_field` (must be a configured key);
    (2) case-insensitive, accent-folded substring match of any `marcadores` entry against
        `titulo + " " + " ".join(palabras_clave)`;
    (3) `summary.cliente_sugerido` when `summary.confianza >= cfg.summary.confianza_minima_cliente`;
    (4) `sin_clasificar` fallback.
    A client folder is never invented for a key absent from config.toml."""
    clientes = cfg.archivo.clientes

    if client_field and client_field in clientes:
        c = clientes[client_field]
        return ClientResolution(key=c.key, nombre=c.nombre, carpeta=c.carpeta, source="upload")

    titulo_for_match = titulo_usuario or summary.titulo or ""
    haystack = _fold(titulo_for_match + " " + " ".join(summary.palabras_clave))
    for c in clientes.values():
        for marcador in c.marcadores:
            if marcador and _fold(marcador) in haystack:
                return ClientResolution(key=c.key, nombre=c.nombre, carpeta=c.carpeta, source="marker")

    if (
        summary.cliente_sugerido
        and summary.cliente_sugerido in clientes
        and summary.confianza >= cfg.summary.confianza_minima_cliente
    ):
        c = clientes[summary.cliente_sugerido]
        return ClientResolution(key=c.key, nombre=c.nombre, carpeta=c.carpeta, source="claude")

    if "sin_clasificar" in clientes:
        c = clientes["sin_clasificar"]
        return ClientResolution(key=c.key, nombre=c.nombre, carpeta=c.carpeta, source="fallback")
    # config.toml always defines archivo.clientes.sin_clasificar (§5); this is a last-resort net
    # only, still never inventing a folder outside cfg.archivo.carpeta_fallback.
    return ClientResolution(
        key="sin_clasificar", nombre="Sin clasificar",
        carpeta=cfg.archivo.carpeta_fallback, source="fallback",
    )


# --------------------------------------------------------------------------------- front-matter


def render_front_matter(fields: dict) -> str:
    """`---`, one JSON-encoded `key: value` per line in dict order, `---`, one blank line (§9.4)."""
    lines = [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in fields.items()]
    return "---\n" + "\n".join(lines) + "\n---\n\n"


def parse_front_matter(text: str) -> tuple[dict, str] | None:
    """Inverse of `render_front_matter`: returns (fields, body) or None if `text` doesn't start
    with a well-formed front-matter block. Used both for overwrite-protection (§9.6) and by
    indices.py to rebuild the catalog from disk (§10)."""
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n\n", 4)
    if end == -1:
        return None
    fm_text = text[4:end]
    body = text[end + len("\n---\n\n") :]
    fields: dict = {}
    for line in fm_text.split("\n"):
        if not line.strip():
            continue
        key, sep, raw_value = line.partition(":")
        if not sep:
            return None
        try:
            fields[key.strip()] = json.loads(raw_value.strip())
        except json.JSONDecodeError:
            return None
    return fields, body


# --------------------------------------------------------------------------------- body (§9.4)


def _render_body(
    *,
    titulo: str,
    fecha_larga: str,
    hora: str,
    duracion_min: int,
    categoria_display: str,
    palabras_clave: list[str],
    summary: SummaryResult,
    turns: TurnsResult,
) -> str:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)

    emit(f"# {titulo}")
    emit()
    emit(f"> **{fecha_larga}** · {hora} · {duracion_min} min · {categoria_display}  ")
    emit(f"> Palabras clave: {', '.join(palabras_clave)}  ")
    if summary.pending:
        emit(
            "> ⚠️ **Resumen pendiente** — la transcripción está completa; "
            "el resumen se generará más tarde.  "
        )
    emit()

    emit("## Resumen")
    emit()
    emit(summary.resumen if summary.resumen else _PLACEHOLDERS["resumen"])
    if summary.mas_detalle:
        emit()
        emit("**Más detalle:**")
        for item in summary.mas_detalle:
            emit()
            emit(f"- **{item.tema}:** {item.texto}")
    emit()

    emit("## Puntos clave")
    emit()
    if summary.puntos_clave:
        for p in summary.puntos_clave:
            emit(f"- {p.emoji} {p.tema}: {p.texto}")
    else:
        emit(_PLACEHOLDERS["puntos_clave"])
    emit()

    emit("## Tareas y acuerdos")
    emit()
    if summary.tareas:
        order: list[str] = []
        grouped: dict[str, list] = {}
        for t in summary.tareas:
            if t.responsable not in grouped:
                grouped[t.responsable] = []
                order.append(t.responsable)
            grouped[t.responsable].append(t)
        first = True
        for responsable in order:
            if not first:
                emit()
            first = False
            emit(f"### {responsable}")
            for t in grouped[responsable]:
                emit(f"- [ ] {t.texto} · [{fmt_ts(t.t_seg)}]")
    else:
        emit(_PLACEHOLDERS["tareas"])
    emit()

    emit("## Temas tratados")
    emit()
    if summary.temas:
        for tema in summary.temas:
            emit(f"- {tema}")
    else:
        emit(_PLACEHOLDERS["temas"])
    emit()

    emit("## Transcripción completa")
    emit()
    first = True
    for turn in turns.turns:
        if not first:
            emit()
        first = False
        emit(f"**{turn.speaker_label}** · [{fmt_ts(turn.start)}]")
        emit(turn.text)

    return "\n".join(lines).rstrip("\n") + "\n"


# --------------------------------------------------------------------------------- top level


def render_markdown(
    cfg: Config,
    turns: TurnsResult,
    summary: SummaryResult,
    *,
    job_id: str,
    titulo_usuario: str | None,
    inicio: datetime,
    audio_sha256: str,
    cliente: ClientResolution,
    generado_en: datetime | None = None,
) -> tuple[str, str]:
    """Build the full note (front-matter + body). `inicio` must be tz-aware; it is converted to
    `cfg.general.zona_horaria` here (§9.4 `inicio` requires the Europe/Madrid offset via
    zoneinfo). Returns `(titulo_resuelto, markdown)`; the caller uses `titulo_resuelto` (already
    through `sanitize_title`) to build the target path via `paths.note_path` before calling
    `write_note`."""
    tz = ZoneInfo(cfg.general.zona_horaria)
    inicio_local = inicio.astimezone(tz)
    generado_en_local = (generado_en.astimezone(tz) if generado_en else datetime.now(tz))

    titulo = sanitize_title(titulo_usuario or summary.titulo or "")
    duracion_min = max(1, round(turns.duration_s / 60))
    idioma = turns.language if summary.pending else (summary.idioma or turns.language)
    categoria_key = summary.categoria or "otras"
    categoria_display = cfg.archivo.categorias.get(categoria_key, categoria_key)
    palabras_clave = list(summary.palabras_clave)
    participantes = list(turns.participantes)

    body = _render_body(
        titulo=titulo,
        fecha_larga=fmt_long_date_es(inicio_local.date()),
        hora=inicio_local.strftime("%H:%M"),
        duracion_min=duracion_min,
        categoria_display=categoria_display,
        palabras_clave=palabras_clave,
        summary=summary,
        turns=turns,
    )
    hash_contenido = sha256_text(body)

    front_matter = render_front_matter(
        {
            "generador": GENERADOR,
            "tipo": TIPO_REUNION,
            "origen_id": job_id,
            "titulo": titulo,
            "fecha": inicio_local.date().isoformat(),
            "hora": inicio_local.strftime("%H:%M"),
            "inicio": inicio_local.isoformat(timespec="seconds"),
            "duracion_min": duracion_min,
            "idioma": idioma,
            "categoria": categoria_key,
            "palabras_clave": palabras_clave,
            "participantes": participantes,
            "enlace": None,
            "resumen_corto": summary.resumen_corto,
            "cliente": cliente.key,
            "cliente_nombre": cliente.nombre,
            "hash_origen": audio_sha256,
            "version_modulo": cfg.general.version_modulo,
            "generado_en": generado_en_local.isoformat(timespec="seconds"),
            "hash_contenido": hash_contenido,
        }
    )
    return titulo, front_matter + body


# --------------------------------------------------------------------------------- write (§9.6)


def atomic_write_text(path: Path, content: str) -> None:
    """Temp file in the same directory (dot-prefixed, so an rclone FUSE mount never sees a
    half-written target name), flush + best-effort fsync, then `os.replace`. No directory fsync
    and no hardlinks — both are FUSE-hostile tricks the target mount may reject."""
    guard_write(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            log.warning("fsync failed for %s (FUSE mount may not support it); continuing", tmp)
    os.replace(tmp, path)


def _body_matches_stored_hash(path: Path) -> bool:
    try:
        existing = path.read_text(encoding="utf-8")
    except OSError:
        return False
    parsed = parse_front_matter(existing)
    if parsed is None:
        return False
    meta, body = parsed
    return meta.get("hash_contenido") == sha256_text(body)


def _next_available_sibling(path: Path) -> Path:
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not candidate.exists() or _body_matches_stored_hash(candidate):
            return candidate
        n += 1


def write_note(path: Path, content: str) -> Path:
    """Atomic, non-destructive write (§9.6). If `path` already exists and its on-disk body's
    SHA-256 no longer matches its own stored `hash_contenido`, the user edited it by hand: write a
    `<stem> (2).md` sibling (then `(3)`, …) instead of overwriting, and log a warn. Re-running with
    identical content is idempotent: an unchanged body matches its own stored hash (whether at the
    original path or at a previously-written sibling), so the same file is reused. Returns the path
    actually written; the caller stores it in `jobs.note_path`."""
    target = guard_write(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    final_path = target
    if final_path.exists() and not _body_matches_stored_hash(final_path):
        final_path = _next_available_sibling(final_path)
        log.warning("note at %s was edited by hand; writing sibling %s instead", target, final_path)

    atomic_write_text(final_path, content)
    return final_path

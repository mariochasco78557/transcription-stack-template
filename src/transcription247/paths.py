"""Path guard, filename sanitization, Drive layout, time/date formatting (ARCHITECTURE.md §3.1, §12).

`guard_write()` is the single choke point every filesystem write in the project must go through
(§12 Security). The three roots are module-level globals, set once at process startup via
`configure(cfg)`; tests use `set_roots()` to point at tmp dirs so they never touch the real Drive.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from pathlib import Path

from transcription247.config import Config

# Set via configure()/set_roots(); None means "not configured yet".
OUTPUT_ROOT: Path | None = None  # config archivo.raiz
INBOX_ROOT: Path | None = None  # config inbox.ruta
DATA_ROOT: Path | None = None  # <project>/data

#: Path segments that must never appear in a resolved write target, regardless of root (§12).
_DENIED_SEGMENTS = ("_Protected_Archive", "_Protected_Private")

_DIAS_ES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_MESES_ES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)

_CONTROL_CHARS = {chr(c) for c in list(range(0, 32)) + [127]}
_REPLACE_WITH_DASH = str.maketrans({c: "-" for c in '/\\:*?"<>|'})
_SPEAKER_HINT_RE = re.compile(r"[\[(](\d{1,2})p[\])]", re.IGNORECASE)


class PathGuardError(RuntimeError):
    """A write was attempted outside every owned root, or configure() was never called."""


def configure(cfg: Config) -> None:
    """Set OUTPUT_ROOT/INBOX_ROOT/DATA_ROOT from a loaded Config. Call once at process startup."""
    set_roots(
        output_root=Path(cfg.archivo.raiz).expanduser(),
        inbox_root=Path(cfg.inbox.ruta).expanduser(),
        data_root=cfg.project_root / cfg.paths.data_dir,
    )


def set_roots(output_root: Path, inbox_root: Path, data_root: Path) -> None:
    """Directly set the three roots. Intended for tests, so they operate on tmp dirs only."""
    global OUTPUT_ROOT, INBOX_ROOT, DATA_ROOT
    OUTPUT_ROOT = output_root.resolve(strict=False)
    INBOX_ROOT = inbox_root.resolve(strict=False)
    DATA_ROOT = data_root.resolve(strict=False)


def ensure_inside(root: Path, candidate: Path) -> Path:
    """Resolve(strict=False) and assert candidate is root or below it, else PathGuardError."""
    resolved_root = root.resolve(strict=False)
    resolved = candidate.resolve(strict=False)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise PathGuardError(f"{candidate} is not inside {root}")
    return resolved


def guard_write(candidate: Path) -> Path:
    """ensure_inside against OUTPUT_ROOT / INBOX_ROOT / DATA_ROOT; additionally hard-deny any
    resolved path containing the segment '_Protected_Archive' (a protected sibling archive) or
    '_Protected_Private'. EVERY filesystem write in the project goes through this function."""
    if OUTPUT_ROOT is None or INBOX_ROOT is None or DATA_ROOT is None:
        raise PathGuardError("paths.configure()/set_roots() must be called before guard_write()")

    resolved = candidate.resolve(strict=False)
    for segment in _DENIED_SEGMENTS:
        if segment in resolved.parts:
            raise PathGuardError(f"{candidate} touches denied segment {segment!r}")

    for root in (OUTPUT_ROOT, INBOX_ROOT, DATA_ROOT):
        try:
            return ensure_inside(root, resolved)
        except PathGuardError:
            continue
    raise PathGuardError(f"{candidate} is outside OUTPUT_ROOT, INBOX_ROOT and DATA_ROOT")


def sanitize_title(raw: str, max_len: int = 120) -> str:
    """NFC-normalize; drop control chars; map / \\ : * ? " < > | to '-'; collapse whitespace;
    strip leading dots and trailing dots/spaces; truncate on a word boundary;
    '' -> 'Reunión sin título'."""
    text = unicodedata.normalize("NFC", raw or "")
    text = "".join(c for c in text if c not in _CONTROL_CHARS)
    text = text.translate(_REPLACE_WITH_DASH)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lstrip(".")
    text = text.rstrip(". ")
    if len(text) > max_len:
        truncated = text[:max_len]
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        text = truncated.rstrip(". ")
    return text or "Reunión sin título"


def extract_speaker_hint(name: str) -> tuple[str, int | None]:
    """Find a case-insensitive [Np] or (Np) token (N in 1-10) anywhere in `name`; return
    (name with the matched token removed, N) or (name unchanged, None) if absent/out of range.
    Used by api.py and inbox.py (§6) before sanitize_title, so the token never reaches
    source_name."""
    match = _SPEAKER_HINT_RE.search(name)
    if not match:
        return name, None
    n = int(match.group(1))
    if not (1 <= n <= 10):
        return name, None
    stripped = re.sub(r"\s+", " ", name[: match.start()] + " " + name[match.end() :]).strip()
    return stripped, n


def note_path(cliente_folder: str, start: datetime, titulo: str) -> Path:
    """OUTPUT_ROOT/<cliente_folder>/<YYYY-MM>/<YYYY-MM-DD HHhMM> - <titulo>.md, guarded."""
    if OUTPUT_ROOT is None:
        raise PathGuardError("paths.configure()/set_roots() must be called before note_path()")
    year_month = start.strftime("%Y-%m")
    stamp = start.strftime("%Y-%m-%d %Hh%M")
    candidate = OUTPUT_ROOT / cliente_folder / year_month / f"{stamp} - {titulo}.md"
    return guard_write(candidate)


def fmt_ts(seconds: float) -> str:
    """'MM:SS' below 3600 s, else 'H:MM:SS'; zero-padded."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fmt_long_date_es(d: date) -> str:
    """'martes 1 de septiembre de 2026'; module-level Spanish DIAS/MESES tables — never the C
    locale."""
    return f"{_DIAS_ES[d.weekday()]} {d.day} de {_MESES_ES[d.month - 1]} de {d.year}"

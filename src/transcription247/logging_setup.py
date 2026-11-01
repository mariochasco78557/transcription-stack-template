"""Logging setup: journald (via systemd's stdout/journal capture, §11) + RotatingFileHandler.

No `python-systemd` dependency: units run with `StandardOutput=journal`, so a plain
`StreamHandler` on stdout already lands in the journal under `SyslogIdentifier=t247-*`. This
module only adds the rotating file handler config.logging points at.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from transcription247.config import LoggingConfig

_CONFIGURED = False


def setup_logging(cfg: LoggingConfig, project_root: Path) -> None:
    """Idempotent: safe to call more than once (e.g. from tests)."""
    global _CONFIGURED
    root = logging.getLogger("transcription247")
    root.setLevel(getattr(logging, cfg.nivel.upper(), logging.INFO))

    if _CONFIGURED:
        return
    _CONFIGURED = True

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    log_path = Path(cfg.fichero)
    if not log_path.is_absolute():
        log_path = project_root / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=cfg.max_bytes, backupCount=cfg.backups, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

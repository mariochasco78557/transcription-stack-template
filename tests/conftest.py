"""Shared pytest fixtures (ARCHITECTURE.md §15, [INT]).

The one thing every test in this suite shares is `paths`' three module-level roots. They are
process-global by design (§3.1: `guard_write` is a single choke point configured once at startup),
which makes them order-dependent state in a test session: a test that calls `paths.set_roots()` or
`paths.configure()` leaves them pointing at its own `tmp_path`, and the next test — or a later run
of the same test in a different order — sees roots that no longer exist. `_isolate_paths_roots`
below snapshots and restores them around every single test, so tests can be run in any order,
individually, or with `-p no:randomly`, and no test can leak a real Drive path into another.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcription247 import paths
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


@pytest.fixture(autouse=True)
def _isolate_paths_roots():
    """Save `paths.OUTPUT_ROOT`/`INBOX_ROOT`/`DATA_ROOT` before each test and restore them after.

    Autouse and unconditional: a test that never touches the roots is unaffected, and one that
    does cannot influence anything that runs after it.
    """
    saved = (paths.OUTPUT_ROOT, paths.INBOX_ROOT, paths.DATA_ROOT)
    try:
        yield
    finally:
        paths.OUTPUT_ROOT, paths.INBOX_ROOT, paths.DATA_ROOT = saved


# --------------------------------------------------------------------------------- config factory


def make_config(tmp_path: Path, **overrides) -> Config:
    """A complete `Config` rooted entirely under `tmp_path` — never the real Drive.

    Every section is overridable by keyword (`summary=SummaryConfig(...)`, `worker=...`), so
    integration tests can tune one knob without rebuilding the whole object. The output/inbox
    directories are created, since the worker and the API both probe them (`is_dir()`).
    """
    output_root = tmp_path / "drive" / "Transcripciones_24_7"
    inbox_root = output_root / "_INBOX_AUDIO"
    output_root.mkdir(parents=True, exist_ok=True)
    inbox_root.mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)

    archivo = ArchivoConfig(
        raiz=str(output_root),
        clientes={
            "centro_ejemplo": ClienteConfig(
                key="centro_ejemplo",
                carpeta="Centro_Ejemplo",
                nombre="Centro Ejemplo",
                marcadores=("centro", "claustro"),
            ),
            "sin_clasificar": ClienteConfig(
                key="sin_clasificar", carpeta="_Sin_clasificar", nombre="Sin clasificar",
            ),
        },
        categorias={"otras": "Otras", "tutoria": "Tutoría", "claustro": "Claustro"},
    )
    defaults = dict(
        general=GeneralConfig(),
        paths=PathsConfig(data_dir="data", secrets_dir=str(tmp_path / "secrets")),
        audio=AudioConfig(),
        asr=AsrConfig(),
        diarization=DiarizationConfig(),
        merge=MergeConfig(),
        summary=SummaryConfig(),
        archivo=archivo,
        inbox=InboxConfig(ruta=str(inbox_root), estabilidad_segundos=0),
        api=ApiConfig(host="127.0.0.1"),
        worker=WorkerConfig(
            mem_disponible_minima_mb=0,  # never block a test on the box's live memory
            mem_espera_segundos=0,
            idle_poll_segundos=0,
        ),
        logging=LoggingConfig(fichero=str(tmp_path / "data" / "logs" / "t247.log")),
        project_root=tmp_path,
    )
    defaults.update(overrides)
    return Config(**defaults)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """`make_config(tmp_path)` with `paths` already pointed at it."""
    config = make_config(tmp_path)
    paths.configure(config)
    return config

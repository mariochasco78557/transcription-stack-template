"""`t247 doctor`: environment checks (ARCHITECTURE.md §11, §12, §13, §16 D9).

Everything the pipeline needs that is *not* code: the Drive mount, the two model trees, the
`claude` CLI login-free sanity check, the tailnet address, free RAM, secrets-file permissions, the
ffmpeg binaries and the two systemd user units. Read-only by design — the single exception is one
tiny probe file written into (and immediately removed from) the owned Drive root, because "the
mount is listed" and "the mount accepts writes" are different failures and only the second matters.

`run(cfg) -> int` is a process exit code (0 = everything required passed), which is what
`cli.py`'s `doctor` subcommand returns.
"""

from __future__ import annotations

import os
import shutil
import socket
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from transcription247 import paths
from transcription247.config import Config

OK = "ok"
WARN = "warn"
FAIL = "fail"

_SYMBOL = {OK: "✓", WARN: "!", FAIL: "✗"}

#: The two user units §11 installs.
UNITS = ("t247-worker.service", "t247-api.service")

#: `claude --version` must answer within this many seconds. It is a local version print — no model
#: call, no network, no token spend (§8.1 is the only place a model is ever invoked).
CLAUDE_VERSION_TIMEOUT_S = 30


@dataclass(slots=True)
class Check:
    name: str
    status: str
    detail: str

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def _ok(name: str, detail: str = "") -> Check:
    return Check(name, OK, detail)


def _warn(name: str, detail: str) -> Check:
    return Check(name, WARN, detail)


def _fail(name: str, detail: str) -> Check:
    return Check(name, FAIL, detail)


# --------------------------------------------------------------------------------------- checks


def check_output_mount(cfg: Config) -> Check:
    """`archivo.raiz` exists and accepts a write — and the probe goes through `guard_write`, so
    doctor can only ever touch the folder this project owns (§12)."""
    root = Path(cfg.archivo.raiz).expanduser()
    if not root.is_dir():
        return _fail("drive: salida", f"{root} is missing (rclone mount down?)")
    probe = root / f".t247-doctor-{os.getpid()}"
    try:
        target = paths.guard_write(probe)
    except paths.PathGuardError as exc:
        return _fail("drive: salida", f"refusing to write inside {root}: {exc}")
    try:
        target.write_text("t247 doctor\n", encoding="utf-8")
        target.unlink()
    except OSError as exc:
        return _fail("drive: salida", f"{root} is not writable: {exc}")
    return _ok("drive: salida", str(root))


def check_inbox(cfg: Config) -> Check:
    inbox_root = Path(cfg.inbox.ruta).expanduser()
    if not cfg.inbox.habilitado:
        return _warn("drive: _INBOX_AUDIO", "inbox.habilitado = false (Drive ingestion is off)")
    if not inbox_root.is_dir():
        return _fail("drive: _INBOX_AUDIO", f"{inbox_root} is missing")
    if not os.access(inbox_root, os.R_OK | os.W_OK):
        return _fail("drive: _INBOX_AUDIO", f"{inbox_root} is not readable/writable")
    return _ok("drive: _INBOX_AUDIO", str(inbox_root))


def check_asr_model(cfg: Config) -> Check:
    path = Path(cfg.asr.model_path)
    if not path.is_absolute():
        path = cfg.project_root / path
    if not path.is_dir():
        return _fail("modelo ASR", f"{path} is missing (see README: model download)")
    missing = [f for f in ("model.bin", "config.json", "tokenizer.json") if not (path / f).is_file()]
    if missing:
        return _fail("modelo ASR", f"{path} is incomplete: missing {', '.join(missing)}")
    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        return _fail("modelo ASR", f"faster-whisper is not importable: {exc}")
    return _ok("modelo ASR", str(path.name))


def check_diarization_models(cfg: Config) -> Check:
    for label, raw in (
        ("segmentation", cfg.diarization.segmentation_model),
        ("embedding", cfg.diarization.embedding_model),
    ):
        path = Path(raw)
        if not path.is_absolute():
            path = cfg.project_root / path
        if not path.is_file():
            return _fail("modelos diarización", f"{label} model is missing: {path}")
    try:
        import sherpa_onnx  # noqa: F401  (D9 regression guard: the libonnxruntime soname pin)
    except ImportError as exc:
        return _fail("modelos diarización", f"sherpa_onnx is not importable (§16 D9): {exc}")
    return _ok("modelos diarización", cfg.diarization.backend)


def check_ffmpeg(cfg: Config) -> Check:
    missing = [b for b in (cfg.audio.ffmpeg, cfg.audio.ffprobe) if shutil.which(b) is None]
    if missing:
        return _fail("ffmpeg/ffprobe", f"not on PATH: {', '.join(missing)} (apt install ffmpeg)")
    return _ok("ffmpeg/ffprobe", shutil.which(cfg.audio.ffmpeg) or "")


def check_claude(cfg: Config) -> Check:
    """Presence + `--version`. Deliberately not a model call: doctor must be free to run."""
    binary = cfg.summary.claude_bin
    resolved = binary if Path(binary).is_file() else (shutil.which(binary) or "")
    if not resolved:
        return _fail("claude CLI", f"not found: {binary} (config summary.claude_bin)")
    if not os.access(resolved, os.X_OK):
        return _fail("claude CLI", f"{resolved} is not executable")
    try:
        proc = subprocess.run(
            [resolved, "--version"],
            capture_output=True, text=True, timeout=CLAUDE_VERSION_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _fail("claude CLI", f"{resolved} --version failed: {exc}")
    if proc.returncode != 0:
        return _fail("claude CLI", f"{resolved} --version exited {proc.returncode}")
    version = (proc.stdout or proc.stderr).strip().splitlines()[0] if (proc.stdout or proc.stderr) else "?"
    return _ok("claude CLI", version)


def check_tailnet(cfg: Config) -> Check:
    """`api.host` must be a real local address, and never a wildcard (§12: tailnet only)."""
    host = cfg.api.host
    if host in ("0.0.0.0", "::"):
        return _fail("tailnet", f"api.host = {host} is a wildcard bind address and is refused (§12)")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
    except OSError as exc:
        return _fail("tailnet", f"{host} is not on any local interface ({exc}); is Tailscale up?")
    return _ok("tailnet", f"{host}:{cfg.api.puerto}")


def check_memory(cfg: Config) -> Check:
    from transcription247.worker import mem_available_mb

    available = mem_available_mb()
    need = cfg.worker.mem_disponible_minima_mb
    if available == 0:
        return _warn("memoria libre", "/proc/meminfo unreadable")
    if available < need:
        return _warn("memoria libre", f"{available} MB available, worker waits below {need} MB")
    return _ok("memoria libre", f"{available} MB (mínimo {need} MB)")


def check_secrets(cfg: Config) -> Check:
    """0700 directory, 0600 token; telegram.json optional but held to the same bar (§5, §12)."""
    secrets_dir = Path(cfg.paths.secrets_dir).expanduser()
    if not secrets_dir.is_dir():
        return _fail("secretos", f"{secrets_dir} is missing (run ops/install.sh)")
    dir_mode = stat.S_IMODE(secrets_dir.stat().st_mode)
    if dir_mode != 0o700:
        return _fail("secretos", f"{secrets_dir} has mode {oct(dir_mode)}, expected 0700")

    token_path = secrets_dir / "api_token"
    if not token_path.is_file():
        return _fail("secretos", f"{token_path} is missing (run ops/install.sh)")
    token_mode = stat.S_IMODE(token_path.stat().st_mode)
    if token_mode != 0o600:
        return _fail("secretos", f"{token_path} has mode {oct(token_mode)}, expected 0600")
    if not cfg.secret_api_token():
        return _fail("secretos", f"{token_path} is empty")

    telegram = secrets_dir / "telegram.json"
    if telegram.is_file():
        tg_mode = stat.S_IMODE(telegram.stat().st_mode)
        if tg_mode != 0o600:
            return _fail("secretos", f"{telegram} has mode {oct(tg_mode)}, expected 0600")
        if cfg.secret_telegram() is None:
            return _warn("secretos", f"{telegram} is present but not valid JSON with token/chat_id")
        return _ok("secretos", "api_token + telegram.json")
    return _ok("secretos", "api_token (sin telegram.json: notificaciones desactivadas)")


def check_database(cfg: Config) -> Check:
    from transcription247.store import Store

    path = cfg.project_root / cfg.paths.data_dir / "t247.db"
    try:
        store = Store.open(path)
    except Exception as exc:  # noqa: BLE001 - any sqlite/OS failure is the same answer here
        return _fail("cola sqlite", f"{path}: {exc}")
    try:
        counts = store.counts_by_state()
    finally:
        store.close()
    if not counts:
        return _ok("cola sqlite", f"{path.name} (vacía)")
    return _ok("cola sqlite", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


def _systemctl(*args: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["systemctl", "--user", *args],
            capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return proc.returncode, (proc.stdout or proc.stderr).strip()


def check_units(cfg: Config) -> list[Check]:
    if shutil.which("systemctl") is None:
        return [_warn("systemd", "systemctl not found; units cannot be checked")]
    checks: list[Check] = []
    for unit in UNITS:
        unit_file = Path.home() / ".config" / "systemd" / "user" / unit
        if not unit_file.is_file():
            checks.append(_fail(f"unit {unit}", f"{unit_file} not installed (run ops/install.sh)"))
            continue
        _, enabled = _systemctl("is-enabled", unit)
        code, active = _systemctl("is-active", unit)
        detail = f"{enabled}/{active}"
        checks.append(_ok(f"unit {unit}", detail) if code == 0 else _fail(f"unit {unit}", detail))
    return checks


# ----------------------------------------------------------------------------------------- run


def collect(cfg: Config) -> list[Check]:
    """Every check, in report order. Pure data, so tests and `/healthz` could reuse it."""
    checks = [
        check_output_mount(cfg),
        check_inbox(cfg),
        check_database(cfg),
        check_asr_model(cfg),
        check_diarization_models(cfg),
        check_ffmpeg(cfg),
        check_claude(cfg),
        check_tailnet(cfg),
        check_memory(cfg),
        check_secrets(cfg),
    ]
    checks.extend(check_units(cfg))
    return checks


def run(cfg: Config) -> int:
    """Print the report; exit code 0 when nothing failed (warnings do not fail the run)."""
    paths.configure(cfg)
    checks = collect(cfg)
    width = max(len(c.name) for c in checks)
    print(f"t247 doctor — {cfg.general.version_modulo}")
    for check in checks:
        detail = f"  {check.detail}" if check.detail else ""
        print(f"  {_SYMBOL[check.status]} {check.name:<{width}}{detail}")
    failures = [c for c in checks if c.failed]
    warnings = [c for c in checks if c.status == WARN]
    print()
    if failures:
        print(f"{len(failures)} problema(s): " + ", ".join(c.name for c in failures))
        return 1
    print("Todo correcto." + (f" ({len(warnings)} aviso(s))" if warnings else ""))
    return 0

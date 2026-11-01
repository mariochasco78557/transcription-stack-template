"""`t247` argparse entrypoint (ARCHITECTURE.md §14 P1).

cli.py owns every subcommand's argument parsing; the actual work for `worker`, `serve`, `enqueue`,
`status`, `retry`, `resummarize`, `reindex` and `doctor` lives in other packages (`worker.py`,
`api.py`, `indices.py`, `doctor.py`), imported lazily *inside* each handler. That way `t247 --help`
and the P1-only commands work the moment this file lands, and a not-yet-implemented command fails
with one clean line instead of a traceback — no other package ever needs to edit this file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcription247 import paths
from transcription247.config import Config, ConfigError, load_config
from transcription247.logging_setup import setup_logging


def _missing_module(command: str, module: str, exc: Exception) -> int:
    print(
        f"t247 {command}: not available yet — {module}.py is not implemented "
        f"(import failed: {exc}). See ARCHITECTURE.md §14 for the owning package.",
        file=sys.stderr,
    )
    return 1


def _load(config_path: str | None) -> Config:
    cfg = load_config(Path(config_path) if config_path else None)
    paths.configure(cfg)
    setup_logging(cfg.logging, cfg.project_root)
    return cfg


def _cmd_worker(cfg: Config, args: argparse.Namespace) -> int:
    try:
        from transcription247.worker import run_worker
    except ModuleNotFoundError as exc:
        return _missing_module("worker", "worker", exc)
    run_worker(cfg)
    return 0


def _cmd_serve(cfg: Config, args: argparse.Namespace) -> int:
    try:
        from transcription247.api import serve
    except ModuleNotFoundError as exc:
        return _missing_module("serve", "api", exc)
    serve(cfg)
    return 0


def _cmd_enqueue(cfg: Config, args: argparse.Namespace) -> int:
    try:
        from transcription247.worker import enqueue_file
    except ModuleNotFoundError as exc:
        return _missing_module("enqueue", "worker", exc)
    job = enqueue_file(
        Path(args.path),
        cfg,
        title=args.title,
        language=args.language,
        client=args.client,
        recorded_at=args.recorded_at,
        speakers=args.speakers,
    )
    print(f"enqueued {job.id} ({job.source_name})")
    return 0


def _cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    try:
        from transcription247.worker import print_status
    except ModuleNotFoundError as exc:
        return _missing_module("status", "worker", exc)
    print_status(cfg)
    return 0


def _cmd_retry(cfg: Config, args: argparse.Namespace) -> int:
    try:
        from transcription247.worker import retry
    except ModuleNotFoundError as exc:
        return _missing_module("retry", "worker", exc)
    retry(args.job_id, cfg)
    return 0


def _cmd_resummarize(cfg: Config, args: argparse.Namespace) -> int:
    try:
        from transcription247.worker import resummarize
    except ModuleNotFoundError as exc:
        return _missing_module("resummarize", "worker", exc)
    resummarize(args.job_id, cfg)
    return 0


def _cmd_reindex(cfg: Config, args: argparse.Namespace) -> int:
    try:
        from transcription247.indices import rebuild_all
    except ModuleNotFoundError as exc:
        return _missing_module("reindex", "indices", exc)
    rebuild_all(cfg)
    return 0


def _cmd_doctor(cfg: Config, args: argparse.Namespace) -> int:
    try:
        from transcription247.doctor import run as doctor_run
    except ModuleNotFoundError as exc:
        return _missing_module("doctor", "doctor", exc)
    return int(doctor_run(cfg))


def _speakers_type(value: str) -> int:
    n = int(value)
    if not (1 <= n <= 10):
        raise argparse.ArgumentTypeError("--speakers must be between 1 and 10")
    return n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="t247", description="Transcription 24/7 (ARCHITECTURE.md)")
    parser.add_argument("--config", default=None, help="path to config.toml (default: project root)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("worker", help="run the pipeline worker daemon").set_defaults(func=_cmd_worker)
    sub.add_parser("serve", help="run the upload API (uvicorn)").set_defaults(func=_cmd_serve)

    p_enqueue = sub.add_parser("enqueue", help="enqueue a local audio file")
    p_enqueue.add_argument("path", help="path to an audio/video file")
    p_enqueue.add_argument("--title", default=None)
    p_enqueue.add_argument("--language", default=None, choices=["es", "en"])
    p_enqueue.add_argument("--client", default=None, help="a key from config archivo.clientes")
    p_enqueue.add_argument("--recorded-at", dest="recorded_at", default=None, help="ISO 8601")
    p_enqueue.add_argument("--speakers", type=_speakers_type, default=None, help="known speaker count, 1-10")
    p_enqueue.set_defaults(func=_cmd_enqueue)

    sub.add_parser("status", help="print job counts / recent jobs").set_defaults(func=_cmd_status)

    p_retry = sub.add_parser("retry", help="retry a failed job")
    p_retry.add_argument("job_id")
    p_retry.set_defaults(func=_cmd_retry)

    p_resum = sub.add_parser("resummarize", help="re-run the summary for a summary_pending job")
    p_resum.add_argument("job_id")
    p_resum.set_defaults(func=_cmd_resummarize)

    sub.add_parser("reindex", help="rebuild INDICE.md / catalog from a filesystem scan").set_defaults(
        func=_cmd_reindex
    )
    sub.add_parser("doctor", help="run environment checks").set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = _load(args.config)
    except ConfigError as exc:
        print(f"t247: config error: {exc}", file=sys.stderr)
        return 2
    return args.func(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())

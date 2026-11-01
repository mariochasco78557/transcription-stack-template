"""FastAPI upload API: /upload, /jobs, /jobs/{id}, /healthz, / (ARCHITECTURE.md §6, §12, §14 P5).

`serve(cfg)` is the function `cli.py`'s `serve` subcommand lazily imports (`from
transcription247.api import serve`). `create_app(cfg, store, token)` builds the ASGI app on its
own so tests can drive it with `fastapi.testclient.TestClient` against a tmp Store/tmp roots,
without touching uvicorn, a real Tailscale interface, or a real secrets file.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import socket
import stat
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from transcription247 import paths
from transcription247.config import Config
from transcription247.models import Job
from transcription247.store import DuplicateJobError, Store

log = logging.getLogger(__name__)

_UPLOAD_CHUNK = 1024 * 1024  # 1 MiB


class _TooLarge(Exception):
    """Raised mid-stream once the configured max upload size is exceeded."""


# --------------------------------------------------------------------------------------- helpers


def _mem_available_mb() -> int:
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _relative_note_path(cfg: Config, note_path: str | None) -> str | None:
    if not note_path:
        return None
    root = paths.OUTPUT_ROOT or Path(cfg.archivo.raiz).expanduser()
    try:
        return str(Path(note_path).relative_to(root))
    except ValueError:
        return note_path


def _job_summary(cfg: Config, job: Job) -> dict:
    return {
        "id": job.id,
        "state": job.state,
        "created_at": job.created_at,
        "source_name": job.source_name,
        "title": job.title_user,
        "client_key": job.client_key,
        "duration_s": job.duration_s,
        "note_path": _relative_note_path(cfg, job.note_path),
        "error": job.error,
    }


def _artifact_names(cfg: Config, job_id: str) -> list[str]:
    job_dir = cfg.project_root / cfg.paths.data_dir / "jobs" / job_id
    if not job_dir.is_dir():
        return []
    return sorted(p.name for p in job_dir.iterdir() if p.is_file())


def _validate_upload_fields(cfg: Config, fields: dict[str, str]) -> str | None:
    """Returns an error detail string, or None when every present field is valid."""
    language = fields.get("language")
    if language and language not in cfg.general.idiomas_permitidos:
        return f"unknown language: {language}"
    client = fields.get("client")
    if client and client not in cfg.archivo.clientes:
        return f"unknown client: {client}"
    speakers = fields.get("speakers")
    if speakers:
        try:
            n = int(speakers)
        except ValueError:
            return "speakers must be an integer"
        if not (1 <= n <= 10):
            return "speakers must be between 1 and 10"
    recorded_at = fields.get("recorded_at")
    if recorded_at:
        try:
            datetime.fromisoformat(recorded_at)
        except ValueError:
            return "recorded_at must be ISO 8601"
    return None


async def _stream_upload_file(upload_file: UploadFile, dst: Path, max_bytes: int) -> tuple[int, str]:
    hasher = hashlib.sha256()
    total = 0
    with open(dst, "wb") as out:
        while True:
            chunk = await upload_file.read(_UPLOAD_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise _TooLarge()
            hasher.update(chunk)
            out.write(chunk)
    return total, hasher.hexdigest()


async def _stream_request_body(request: Request, dst: Path, max_bytes: int) -> tuple[int, str]:
    hasher = hashlib.sha256()
    total = 0
    with open(dst, "wb") as out:
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_bytes:
                raise _TooLarge()
            hasher.update(chunk)
            out.write(chunk)
    return total, hasher.hexdigest()


# ------------------------------------------------------------------------------------------ auth


def _auth_dependency(request: Request) -> None:
    """`Authorization: Bearer <token>` compared with `hmac.compare_digest` (§12)."""
    import hmac

    token: str = request.app.state.token
    header = request.headers.get("authorization", "")
    scheme, _, provided = header.partition(" ")
    if scheme.lower() != "bearer" or not provided or not hmac.compare_digest(provided, token):
        raise HTTPException(status_code=401, detail={"error": "unauthorized"})


async def _store_dependency(request: Request):
    """One SQLite connection per request, opened and closed on the event-loop thread that runs
    the (async) endpoint. `sqlite3.Connection` objects are not safe to share across threads
    (`check_same_thread` defaults to True in `store.connect()`, §1 P1); a plain `def` dependency
    would run in Starlette's threadpool instead, a *different* thread from the `async def`
    endpoint that uses it (observed under TestClient, but not guaranteed to hold for any ASGI
    server/config), so this must stay `async def` to share the endpoint's thread."""
    store = Store.open(request.app.state.db_path)
    try:
        yield store
    finally:
        store.close()


# -------------------------------------------------------------------------------------- handlers


async def _healthz(request: Request, store: Store = Depends(_store_dependency)) -> JSONResponse:
    cfg: Config = request.app.state.cfg
    try:
        counts = store.counts_by_state()
        db_ok = True
    except Exception:
        counts = {}
        db_ok = False
    inbox_ok = Path(cfg.inbox.ruta).expanduser().is_dir()
    output_ok = Path(cfg.archivo.raiz).expanduser().is_dir()
    pending = sum(
        c for state, c in counts.items() if state not in ("delivered", "failed", "summary_pending")
    )
    ok = db_ok and inbox_ok and output_ok
    body = {
        "ok": ok,
        "version": cfg.general.version_modulo,
        "db": db_ok,
        "inbox_mount": inbox_ok,
        "output_mount": output_ok,
        "queue_pending": pending,
        "mem_available_mb": _mem_available_mb(),
    }
    return JSONResponse(status_code=200 if ok else 503, content=body)


async def _index(request: Request) -> HTMLResponse:
    from transcription247.web import index_html

    cfg: Config = request.app.state.cfg
    return HTMLResponse(index_html(cfg))


async def _upload(
    request: Request,
    _auth: None = Depends(_auth_dependency),
    store: Store = Depends(_store_dependency),
) -> JSONResponse:
    cfg: Config = request.app.state.cfg
    max_bytes = cfg.api.max_upload_bytes

    content_type = request.headers.get("content-type", "")
    is_multipart = content_type.startswith("multipart/form-data")

    declared: int | None = None
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            declared = None

    # `await request.form()` (below) spools the whole multipart body before any per-chunk limit
    # would apply, and python-multipart's own `max_part_size` does not bound a *file* part (only
    # non-file fields), so a request with no valid Content-Length — e.g. chunked transfer-encoding
    # — would otherwise sail straight past every size check. Require it, before parsing.
    if is_multipart and declared is None:
        return JSONResponse(status_code=411, content={"error": "length_required"})

    if declared is not None and declared > max_bytes:
        return JSONResponse(status_code=413, content={"error": "too_large", "limit": max_bytes})

    tmp_dir = cfg.project_root / cfg.paths.data_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    estimate = declared if declared is not None else max_bytes
    try:
        free = shutil.disk_usage(tmp_dir).free
        if free < 2 * estimate:
            return JSONResponse(status_code=507, content={"error": "insufficient_storage"})
    except OSError:
        pass  # can't check free space; proceed and let ENOSPC surface naturally

    tmp_path = paths.guard_write(tmp_dir / f"upload-{uuid.uuid4()}.part")
    fields: dict[str, str] = {}
    raw_name = ""

    try:
        if is_multipart:
            # `declared` is already known valid and <= max_bytes (checked above, before any
            # parsing started) — HTTP/1.1 framing means the body the parser reads cannot exceed
            # it. `max_part_size`/`max_files`/`max_fields` (supported by the installed Starlette
            # version's `Request.form()`) additionally bound the non-file fields and part/file
            # counts; Starlette turns a violation into an HTTPException(400) itself.
            form = await request.form(max_files=1, max_fields=8, max_part_size=65536)
            upload_file = form.get("file")
            if upload_file is None or not hasattr(upload_file, "filename"):
                return JSONResponse(status_code=400, content={"error": "bad_request", "detail": "missing file"})
            raw_name = upload_file.filename or ""
            for field in ("title", "language", "client", "recorded_at", "speakers"):
                value = form.get(field)
                if value is not None:
                    fields[field] = str(value)
            try:
                total, sha256 = await _stream_upload_file(upload_file, tmp_path, max_bytes)
            finally:
                await upload_file.close()
        else:
            raw_name = request.headers.get("x-filename", "")
            if not raw_name:
                return JSONResponse(
                    status_code=400, content={"error": "bad_request", "detail": "missing X-Filename"}
                )
            header_map = (
                ("title", "x-title"), ("language", "x-language"), ("client", "x-client"),
                ("recorded_at", "x-recorded-at"), ("speakers", "x-speakers"),
            )
            for field, header in header_map:
                value = request.headers.get(header)
                if value is not None:
                    fields[field] = value
            total, sha256 = await _stream_request_body(request, tmp_path, max_bytes)
    except _TooLarge:
        tmp_path.unlink(missing_ok=True)
        return JSONResponse(status_code=413, content={"error": "too_large", "limit": max_bytes})

    if not raw_name:
        tmp_path.unlink(missing_ok=True)
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": "missing filename"})

    raw_name = Path(raw_name).name
    ext = Path(raw_name).suffix.lower().lstrip(".")
    if ext not in cfg.audio.extensiones:
        tmp_path.unlink(missing_ok=True)
        return JSONResponse(status_code=415, content={"error": "unsupported_media_type", "ext": f".{ext}"})

    detail = _validate_upload_fields(cfg, fields)
    if detail is not None:
        tmp_path.unlink(missing_ok=True)
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": detail})

    hint_name, speakers_from_name = paths.extract_speaker_hint(raw_name)
    source_name = paths.sanitize_title(hint_name)

    speakers_user: int | None
    if fields.get("speakers"):
        speakers_user = int(fields["speakers"])
    else:
        speakers_user = speakers_from_name

    try:
        job = store.create_job(
            source_name=source_name,
            origin="upload",
            audio_sha256=sha256,
            title_user=fields.get("title") or None,
            client_user=fields.get("client") or None,
            language_user=fields.get("language") or None,
            recorded_at_user=fields.get("recorded_at") or None,
            speakers_user=speakers_user,
        )
    except DuplicateJobError as exc:
        tmp_path.unlink(missing_ok=True)
        existing = exc.existing
        return JSONResponse(
            status_code=200,
            content={
                "duplicate": True,
                "job_id": existing.id,
                "state": existing.state,
                "note_path": _relative_note_path(cfg, existing.note_path),
            },
        )

    dest = paths.guard_write(cfg.project_root / cfg.paths.data_dir / "jobs" / job.id / f"source.{ext}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path, dest)
    store.advance_stage(job.id, "received", source_path=str(dest))

    return JSONResponse(
        status_code=200,
        content={
            "job_id": job.id,
            "state": "received",
            "duplicate": False,
            "source_name": job.source_name,
            "bytes": total,
        },
    )


async def _list_jobs(
    request: Request,
    limit: int = 50,
    state: str | None = None,
    _auth: None = Depends(_auth_dependency),
    store: Store = Depends(_store_dependency),
) -> JSONResponse:
    cfg: Config = request.app.state.cfg
    jobs = store.list_jobs(state=state, limit=limit)
    counts = store.counts_by_state()
    total = counts.get(state, 0) if state else sum(counts.values())
    return JSONResponse(
        status_code=200,
        content={"total": total, "jobs": [_job_summary(cfg, j) for j in jobs]},
    )


async def _get_job(
    job_id: str,
    request: Request,
    _auth: None = Depends(_auth_dependency),
    store: Store = Depends(_store_dependency),
) -> JSONResponse:
    cfg: Config = request.app.state.cfg
    job = store.get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    events = store.events(job_id, limit=50)
    body = _job_summary(cfg, job)
    body["events"] = [{"ts": e.ts, "level": e.level, "stage": e.stage, "message": e.message} for e in events]
    body["artifacts"] = _artifact_names(cfg, job_id)
    return JSONResponse(status_code=200, content=body)


async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    content = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=content)


# ------------------------------------------------------------------------------------- app build


def create_app(cfg: Config, db_path: Path, token: str) -> FastAPI:
    """Build the ASGI app. `db_path` is `data/t247.db` (§4) — each request opens and closes its
    own `Store` (`_store_dependency`), never a connection shared across requests/threads. `token`
    is the already-loaded bearer token (§12); `serve()` reads it from the secrets file, tests pass
    any string directly."""
    app = FastAPI(title=cfg.api.titulo_web, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.cfg = cfg
    app.state.db_path = db_path
    app.state.token = token

    app.add_exception_handler(HTTPException, _http_exception_handler)

    app.get("/healthz")(_healthz)
    app.get("/", response_class=HTMLResponse)(_index)
    app.post("/upload")(_upload)
    app.get("/jobs")(_list_jobs)
    app.get("/jobs/{job_id}")(_get_job)
    return app


# ---------------------------------------------------------------------------------- serve entry


def _host_is_local(host: str) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
    except OSError:
        return False
    return True


def _check_bind_host(cfg: Config) -> None:
    """Refuse 0.0.0.0/:: unconditionally, and refuse any host absent from a local interface (§12).

    Tailnet-only is a requirement, not a default: there is deliberately no override flag, and
    `config.py` rejects a wildcard `api.host` at load time as well.
    """
    host = cfg.api.host
    if host in ("0.0.0.0", "::"):
        raise SystemExit(
            f"t247 serve: refusing to bind to {host} (a public/wildcard bind address). "
            "Set [api] host in config.toml to this machine's Tailscale IP (see `tailscale ip -4`)."
        )
    if not _host_is_local(host):
        raise SystemExit(
            f"t247 serve: {host} is not present on any local network interface. Check that "
            f"Tailscale is connected (`tailscale status`) and that [api] host in config.toml "
            f"matches its IP, then retry."
        )


def _require_api_token(cfg: Config) -> str:
    """Refuse to start if the token file is missing or has loose perms (§12)."""
    token_path = Path(cfg.paths.secrets_dir).expanduser() / "api_token"
    if not token_path.is_file():
        raise SystemExit(
            f"t247 serve: token file missing: {token_path}\n"
            "Create it with:\n"
            f"  mkdir -p {token_path.parent} && chmod 700 {token_path.parent}\n"
            f"  openssl rand -hex 32 > {token_path} && chmod 600 {token_path}"
        )
    mode = stat.S_IMODE(token_path.stat().st_mode)
    if mode != 0o600:
        raise SystemExit(
            f"t247 serve: token file {token_path} has mode {oct(mode)}, expected 0600.\n"
            f"Fix with: chmod 600 {token_path}"
        )
    token = cfg.secret_api_token()
    if not token:
        raise SystemExit(f"t247 serve: token file {token_path} is present but empty/unreadable.")
    return token


def serve(cfg: Config) -> None:
    """`cli.py`'s `serve` subcommand imports this lazily (`from transcription247.api import
    serve`). Binds ONLY to `cfg.api.host` (§12) and refuses to start without a valid token file."""
    _check_bind_host(cfg)
    token = _require_api_token(cfg)
    paths.configure(cfg)

    db_path = cfg.project_root / cfg.paths.data_dir / "t247.db"
    Store.open(db_path).close()  # fail fast if the DB file/dir can't be created, before binding
    app = create_app(cfg, db_path, token=token)

    import uvicorn

    uvicorn.run(
        app, host=cfg.api.host, port=cfg.api.puerto, workers=1, log_config=None,
        timeout_keep_alive=30,
    )

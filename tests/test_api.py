"""P5 api.py: auth, /upload streaming + validation + dedup, /jobs, /healthz (ARCHITECTURE.md §6,
§12). FastAPI TestClient only — no real uvicorn, no network, everything under tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from transcription247 import paths
from transcription247.api import create_app
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
    PipelineConfig,
    SummaryConfig,
    WorkerConfig,
)
from transcription247.store import Store

TOKEN = "test-token-123"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _make_config(tmp_path: Path) -> Config:
    output_root = tmp_path / "drive" / "salida"
    inbox_root = tmp_path / "drive" / "_INBOX_AUDIO"
    output_root.mkdir(parents=True)
    inbox_root.mkdir(parents=True)
    return Config(
        general=GeneralConfig(),
        paths=PathsConfig(data_dir="data"),
        audio=AudioConfig(max_bytes=2000),
        asr=AsrConfig(),
        diarization=DiarizationConfig(),
        merge=MergeConfig(),
        pipeline=PipelineConfig(),
        summary=SummaryConfig(),
        archivo=ArchivoConfig(
            raiz=str(output_root),
            clientes={"acme": ClienteConfig(key="acme", carpeta="ACME", nombre="Acme S.A.")},
        ),
        inbox=InboxConfig(ruta=str(inbox_root)),
        api=ApiConfig(max_upload_bytes=2000),  # small on purpose: cheap 413 test
        worker=WorkerConfig(),
        logging=LoggingConfig(),
        project_root=tmp_path,
    )


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    c = _make_config(tmp_path)
    paths.set_roots(
        output_root=Path(c.archivo.raiz), inbox_root=Path(c.inbox.ruta), data_root=tmp_path / "data"
    )
    return c


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "data" / "t247.db")
    yield s
    s.close()


@pytest.fixture
def client(cfg: Config, store: Store) -> TestClient:
    # `store` fixture and the app's own per-request Store both point at the same db_path file.
    return TestClient(create_app(cfg, cfg.project_root / "data" / "t247.db", token=TOKEN))


# ------------------------------------------------------------------------------------- healthz


def test_healthz_no_auth_ok(client: TestClient):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["db"] is True
    assert body["inbox_mount"] is True
    assert body["output_mount"] is True
    assert body["queue_pending"] == 0


def test_healthz_reports_missing_output_mount(client: TestClient, cfg: Config):
    import shutil

    shutil.rmtree(cfg.archivo.raiz)
    resp = client.get("/healthz")
    assert resp.status_code == 503
    assert resp.json()["ok"] is False


# --------------------------------------------------------------------------------------- auth


def test_upload_requires_auth(client: TestClient):
    resp = client.post("/upload", files={"file": ("reunion.m4a", b"x" * 10, "audio/mp4")})
    assert resp.status_code == 401
    assert resp.json() == {"error": "unauthorized"}


def test_upload_rejects_wrong_token(client: TestClient):
    resp = client.post(
        "/upload",
        files={"file": ("reunion.m4a", b"x" * 10, "audio/mp4")},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_jobs_requires_auth(client: TestClient):
    assert client.get("/jobs").status_code == 401
    assert client.get("/jobs/whatever").status_code == 401


# ------------------------------------------------------------------------------------- upload


def test_upload_happy_path_multipart(client: TestClient, cfg: Config, store: Store):
    resp = client.post(
        "/upload",
        files={"file": ("Reunion [2p] equipo.m4a", b"a" * 50, "audio/mp4")},
        data={"title": "Reunión de equipo", "client": "acme", "language": "es"},
        headers=AUTH,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["duplicate"] is False
    assert body["bytes"] == 50
    assert "[2p]" not in body["source_name"]
    job_id = body["job_id"]

    job = store.get_job(job_id)
    assert job is not None
    assert job.state == "received"
    assert job.speakers_user == 2  # from the [2p] filename hint
    assert job.title_user == "Reunión de equipo"
    assert job.client_user == "acme"
    assert job.language_user == "es"

    dest = Path(job.source_path)
    assert dest.is_file()
    assert dest.read_bytes() == b"a" * 50


def test_upload_explicit_speakers_overrides_filename_hint(client: TestClient, store: Store):
    resp = client.post(
        "/upload",
        files={"file": ("Reunion [2p].m4a", b"b" * 10, "audio/mp4")},
        data={"speakers": "5"},
        headers=AUTH,
    )
    assert resp.status_code == 200
    job = store.get_job(resp.json()["job_id"])
    assert job.speakers_user == 5


def test_upload_raw_body(client: TestClient, store: Store):
    resp = client.post(
        "/upload",
        content=b"raw-bytes-here",
        headers={
            **AUTH,
            "Content-Type": "application/octet-stream",
            "X-Filename": "grabacion.wav",
            "X-Title": "Nota rapida",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_name"] == "grabacion.wav"
    job = store.get_job(body["job_id"])
    assert job.title_user == "Nota rapida"


def test_upload_dedup_returns_existing_job(client: TestClient, store: Store):
    payload = b"same-bytes" * 5
    first = client.post(
        "/upload", files={"file": ("a.m4a", payload, "audio/mp4")}, headers=AUTH
    ).json()
    second = client.post(
        "/upload", files={"file": ("b.m4a", payload, "audio/mp4")}, headers=AUTH
    ).json()
    assert second["duplicate"] is True
    assert second["job_id"] == first["job_id"]
    assert len(store.list_jobs(limit=10)) == 1


def test_upload_oversize_returns_413(client: TestClient, store: Store, cfg: Config):
    payload = b"x" * (cfg.api.max_upload_bytes + 500)
    resp = client.post(
        "/upload", files={"file": ("grande.m4a", payload, "audio/mp4")}, headers=AUTH
    )
    assert resp.status_code == 413
    assert resp.json()["error"] == "too_large"
    assert store.list_jobs(limit=10) == []
    leftover_parts = list((cfg.project_root / "data" / "tmp").glob("*.part"))
    assert leftover_parts == []


def _multipart_body_and_content_type(client: TestClient, **kwargs) -> tuple[bytes, str]:
    """Build a real multipart body + its `Content-Type` (with boundary) the way `files=`/`data=`
    normally would, so it can be replayed with `content=<generator>` to strip the `Content-Length`
    header httpx would otherwise compute (see `test_upload_multipart_missing_content_length...`)."""
    req = client.build_request("POST", "/upload", **kwargs)
    return req.read(), req.headers["content-type"]


def test_upload_multipart_missing_content_length_returns_411(client: TestClient, store: Store, cfg: Config):
    """A multipart request with no (valid) Content-Length — e.g. chunked transfer-encoding — must
    be rejected before `request.form()` is ever called, since neither the pre-parse size check nor
    python-multipart's `max_part_size` bounds a *file* part's size (finding #3)."""
    body, content_type = _multipart_body_and_content_type(
        client, files={"file": ("reunion.m4a", b"a" * 50, "audio/mp4")}
    )

    def chunked_body():
        yield body

    resp = client.post(
        "/upload",
        content=chunked_body(),
        headers={**AUTH, "content-type": content_type},
    )
    assert resp.status_code == 411
    assert resp.json()["error"] == "length_required"
    assert store.list_jobs(limit=10) == []
    leftover_parts = list((cfg.project_root / "data" / "tmp").glob("*.part"))
    assert leftover_parts == []


def test_upload_multipart_oversize_returns_413_without_writing_file(
    client: TestClient, store: Store, cfg: Config
):
    payload = b"x" * (cfg.api.max_upload_bytes + 500)
    resp = client.post(
        "/upload", files={"file": ("grande.m4a", payload, "audio/mp4")}, headers=AUTH
    )
    assert resp.status_code == 413
    assert resp.json()["error"] == "too_large"
    assert store.list_jobs(limit=10) == []
    # rejected on the declared Content-Length, before tmp_path (or anything else) was created
    leftover_parts = list((cfg.project_root / "data" / "tmp").glob("*.part"))
    assert leftover_parts == []


def test_upload_normal_multipart_still_succeeds(client: TestClient, store: Store):
    """Sanity check for finding #3's fix: an ordinary multipart upload with a valid Content-Length
    (what iOS Shortcuts sends) still goes through end to end."""
    resp = client.post(
        "/upload",
        files={"file": ("reunion.m4a", b"a" * 50, "audio/mp4")},
        data={"title": "Reunión"},
        headers=AUTH,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["duplicate"] is False
    job = store.get_job(body["job_id"])
    assert job is not None
    assert job.state == "received"


def test_upload_bad_extension_returns_415(client: TestClient):
    resp = client.post(
        "/upload", files={"file": ("notas.txt", b"hola", "text/plain")}, headers=AUTH
    )
    assert resp.status_code == 415
    assert resp.json()["error"] == "unsupported_media_type"


@pytest.mark.parametrize("speakers", ["0", "11", "abc"])
def test_upload_bad_speakers_returns_400(client: TestClient, speakers: str):
    resp = client.post(
        "/upload",
        files={"file": ("a.m4a", b"x" * 10, "audio/mp4")},
        data={"speakers": speakers},
        headers=AUTH,
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "bad_request"


def test_upload_unknown_client_returns_400(client: TestClient):
    resp = client.post(
        "/upload",
        files={"file": ("a.m4a", b"x" * 10, "audio/mp4")},
        data={"client": "no-existe"},
        headers=AUTH,
    )
    assert resp.status_code == 400


def test_upload_unknown_language_returns_400(client: TestClient):
    resp = client.post(
        "/upload",
        files={"file": ("a.m4a", b"x" * 10, "audio/mp4")},
        data={"language": "fr"},
        headers=AUTH,
    )
    assert resp.status_code == 400


# --------------------------------------------------------------------------------------- jobs


def test_jobs_list_and_get(client: TestClient, store: Store):
    resp = client.post(
        "/upload", files={"file": ("a.m4a", b"x" * 10, "audio/mp4")}, headers=AUTH
    )
    job_id = resp.json()["job_id"]

    listing = client.get("/jobs", headers=AUTH)
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 1
    assert body["jobs"][0]["id"] == job_id
    assert body["jobs"][0]["source_name"] == "a.m4a"

    detail = client.get(f"/jobs/{job_id}", headers=AUTH)
    assert detail.status_code == 200
    detail_body = detail.json()
    assert detail_body["id"] == job_id
    assert "events" in detail_body
    assert "artifacts" in detail_body
    assert "source.m4a" in detail_body["artifacts"]


def test_get_unknown_job_returns_404(client: TestClient):
    resp = client.get("/jobs/does-not-exist", headers=AUTH)
    assert resp.status_code == 404
    assert resp.json() == {"error": "not_found"}

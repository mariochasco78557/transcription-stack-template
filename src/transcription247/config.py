"""Load and validate config.toml + the secrets directory (ARCHITECTURE.md §5).

`load_config()` never touches the network or a model; it is pure parsing + validation so every
other package can build against `Config` without a real environment.
"""

from __future__ import annotations

import json
import logging
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """config.toml is missing, malformed, or fails validation."""


def project_root() -> Path:
    """Directory containing pyproject.toml (three levels up from this file)."""
    return Path(__file__).resolve().parents[2]


DEFAULT_CONFIG_PATH = project_root() / "config.toml"


# --------------------------------------------------------------------------- section dataclasses


@dataclass(frozen=True, slots=True)
class GeneralConfig:
    zona_horaria: str = "Europe/Madrid"
    version_modulo: str = "1.0.0"
    idiomas_permitidos: tuple[str, ...] = ("es", "en")
    idioma_fallback: str = "es"


@dataclass(frozen=True, slots=True)
class PathsConfig:
    data_dir: str = "data"
    models_dir: str = "models"
    secrets_dir: str = "~/.config/transcription247"


@dataclass(frozen=True, slots=True)
class AudioConfig:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    sample_rate: int = 16000
    min_duracion_s: float = 1.0
    max_bytes: int = 1_073_741_824
    extensiones: tuple[str, ...] = (
        "m4a", "mp3", "wav", "aac", "ogg", "opus", "flac", "mp4", "mov", "webm", "caf",
    )


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Long-audio windowing shared by the ASR and diarization stages (§3.4/§3.5).

    `window_max_s` is what keeps peak RSS flat in the recording's duration: both backends only
    ever see one window. Raising it raises the peak roughly linearly (~65 MB of ASR and ~54 MB of
    diarization per extra audio-minute), lowering it costs a little accuracy at the joins.
    """

    window_max_s: float = 900.0  # 15 min; 0 disables windowing (the old behaviour, OOMs > ~35 min)
    window_cut_search_s: float = 90.0  # look for a pause in this much of the window's tail
    window_min_s: float = 60.0  # shortest window the planner will emit
    asr_partial_resume: bool = True  # keep asr.partial.json so a killed ASR resumes per window


@dataclass(frozen=True, slots=True)
class AsrConfig:
    model_path: str = "models/faster-whisper-large-v3-turbo"
    compute_type: str = "int8"
    cpu_threads: int = 4
    beam_size: int = 5
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 200
    language: str = ""


@dataclass(frozen=True, slots=True)
class DiarizationConfig:
    backend: str = "sherpa_onnx"
    segmentation_model: str = "models/sherpa-onnx-diarization/segmentation.onnx"
    embedding_model: str = "models/sherpa-onnx-diarization/embedding.onnx"
    num_threads: int = 4
    num_clusters: int = -1  # -1 = auto (see auto_count); set per-job from speakers_user
    auto_count: str = "overcluster"  # "overcluster" (§3.5) | "threshold" (legacy fallback)
    count_embedding_model: str = ""  # auto_count="overcluster" re-embedding; "" = embedding_model
    overcluster_threshold: float = 0.15  # auto_count="overcluster": sherpa's over-split pass
    merge_threshold: float = 0.475  # cosine distance at which pooled centroids stop merging
    min_cluster_share: float = 0.03  # clusters below this share of speech are absorbed
    max_speakers: int = 8  # hard cap on the auto-detected count
    cluster_threshold: float = 0.90  # auto_count="threshold" only
    min_duration_on: float = 0.30
    min_duration_off: float = 0.50


@dataclass(frozen=True, slots=True)
class MergeConfig:
    max_gap_s: float = 2.0
    min_run_s: float = 0.60
    min_run_words: int = 2
    smooth_passes: int = 3
    turn_gap_s: float = 1.20
    max_turn_s: float = 90.0


@dataclass(frozen=True, slots=True)
class SummaryConfig:
    claude_bin: str = "/home/USER/.local/bin/claude"
    modelo: str = "sonnet"
    timeout_s: int = 300
    chunk_seconds: int = 1500
    max_reintentos: int = 1
    confianza_minima_cliente: float = 0.70


@dataclass(frozen=True, slots=True)
class ClienteConfig:
    key: str
    carpeta: str
    nombre: str
    marcadores: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ArchivoConfig:
    raiz: str = ""
    carpeta_fallback: str = "_Sin_clasificar"
    renombrar_speakers: bool = False
    #: Responsible-name markers that make a `### <responsable>` task group in a note "the owner's own"
    #: (§10 "Tareas abiertas del owner"). Matched case-insensitively and accent-folded as a
    #: substring of the heading, so "(tú)" also matches "Owner (tú)".
    responsables_owner: tuple[str, ...] = ("Owner", "(tú)")
    clientes: dict[str, ClienteConfig] = field(default_factory=dict)
    categorias: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InboxConfig:
    ruta: str = ""
    habilitado: bool = True
    poll_segundos: int = 15
    estabilidad_segundos: int = 30
    after_success: str = "delete"  # "delete" | "archive"
    carpeta_procesados: str = "_procesados"
    carpeta_error: str = "_error"


@dataclass(frozen=True, slots=True)
class ApiConfig:
    habilitado: bool = True
    host: str = "127.0.0.1"
    puerto: int = 8247
    max_upload_bytes: int = 1_073_741_824
    titulo_web: str = "Transcripciones 24/7"


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    mem_disponible_minima_mb: int = 3072
    mem_espera_segundos: int = 60
    mem_max_esperas: int = 60
    lease_minutes: int = 120
    max_attempts: int = 3
    retry_backoff_s: tuple[int, ...] = (60, 300, 1800)
    borrar_audio_tras_entrega: bool = True
    conservar_artefactos_dias: int = 14
    idle_poll_segundos: int = 5


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    nivel: str = "INFO"
    fichero: str = "data/logs/t247.log"
    max_bytes: int = 5_242_880
    backups: int = 3


@dataclass(frozen=True, slots=True)
class Config:
    general: GeneralConfig
    paths: PathsConfig
    audio: AudioConfig
    asr: AsrConfig
    diarization: DiarizationConfig
    merge: MergeConfig
    summary: SummaryConfig
    archivo: ArchivoConfig
    inbox: InboxConfig
    api: ApiConfig
    worker: WorkerConfig
    logging: LoggingConfig
    project_root: Path
    #: Last and defaulted on purpose: every `Config(...)` written before windowing existed keeps
    #: working, and a `[pipeline]`-less config.toml gets the shipped defaults.
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def secret_api_token(self) -> str | None:
        """~/.config/transcription247/api_token, one line. Refuses a file whose mode != 0600."""
        path = Path(self.paths.secrets_dir).expanduser() / "api_token"
        text = _read_secret_file(path)
        return text.strip() if text is not None else None

    def secret_telegram(self) -> dict | None:
        """~/.config/transcription247/telegram.json -> {"token": ..., "chat_id": ...}."""
        path = Path(self.paths.secrets_dir).expanduser() / "telegram.json"
        text = _read_secret_file(path)
        if text is None:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.warning("secret file %s is not valid JSON", path)
            return None
        if not isinstance(data, dict) or "token" not in data or "chat_id" not in data:
            log.warning("secret file %s missing token/chat_id", path)
            return None
        return data


def _read_secret_file(path: Path) -> str | None:
    try:
        st = path.stat()
    except OSError:
        return None
    mode = stat.S_IMODE(st.st_mode)
    if mode != 0o600:
        log.warning("refusing to read %s: mode is %o, expected 0600", path, mode)
        return None
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- loader


def load_config(path: Path | None = None) -> Config:
    """Parse and validate config.toml. Missing optional keys fall back to the dataclass defaults
    above (which already mirror §5); a missing file or an unreadable/malformed one raises
    ConfigError."""
    cfg_path = path or DEFAULT_CONFIG_PATH
    try:
        raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {cfg_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config file is not valid TOML: {cfg_path}: {exc}") from exc

    try:
        general = GeneralConfig(**_pick(raw.get("general", {}), GeneralConfig))
        paths_cfg = PathsConfig(**_pick(raw.get("paths", {}), PathsConfig))
        audio = AudioConfig(**_pick(raw.get("audio", {}), AudioConfig))
        pipeline = PipelineConfig(**_pick(raw.get("pipeline", {}), PipelineConfig))
        asr = AsrConfig(**_pick(raw.get("asr", {}), AsrConfig))
        diarization = DiarizationConfig(**_pick(raw.get("diarization", {}), DiarizationConfig))
        merge = MergeConfig(**_pick(raw.get("merge", {}), MergeConfig))
        summary = SummaryConfig(**_pick(raw.get("summary", {}), SummaryConfig))

        archivo_raw = dict(raw.get("archivo", {}))
        clientes_raw = archivo_raw.pop("clientes", {})
        categorias_raw = archivo_raw.pop("categorias", {})
        clientes = {
            key: ClienteConfig(
                key=key,
                carpeta=val.get("carpeta", key),
                nombre=val.get("nombre", key),
                marcadores=tuple(val.get("marcadores", [])),
            )
            for key, val in clientes_raw.items()
        }
        archivo = ArchivoConfig(
            **_pick(archivo_raw, ArchivoConfig),
            clientes=clientes,
            categorias=dict(categorias_raw),
        )

        inbox = InboxConfig(**_pick(raw.get("inbox", {}), InboxConfig))
        api = ApiConfig(**_pick(raw.get("api", {}), ApiConfig))
        worker = WorkerConfig(**_pick(raw.get("worker", {}), WorkerConfig))
        logging_cfg = LoggingConfig(**_pick(raw.get("logging", {}), LoggingConfig))
    except TypeError as exc:
        raise ConfigError(f"config file has an unexpected shape: {exc}") from exc

    cfg = Config(
        general=general,
        paths=paths_cfg,
        audio=audio,
        pipeline=pipeline,
        asr=asr,
        diarization=diarization,
        merge=merge,
        summary=summary,
        archivo=archivo,
        inbox=inbox,
        api=api,
        worker=worker,
        logging=logging_cfg,
        project_root=project_root(),
    )
    _validate(cfg)
    return cfg


def _pick(section: dict, cls: type) -> dict:
    """Keep only keys that are real fields of `cls`, converting known list-typed keys to tuples
    so the resulting dataclass stays hashable/frozen-safe."""
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    out = {}
    for key, value in section.items():
        if key not in field_names:
            continue
        if isinstance(value, list):
            value = tuple(value)
        out[key] = value
    return out


def _validate(cfg: Config) -> None:
    if not cfg.archivo.raiz:
        raise ConfigError("archivo.raiz must be set")
    if not cfg.inbox.ruta:
        raise ConfigError("inbox.ruta must be set")
    if cfg.inbox.after_success not in ("delete", "archive"):
        raise ConfigError("inbox.after_success must be 'delete' or 'archive'")
    if cfg.api.host in ("0.0.0.0", "::"):
        raise ConfigError("api.host must not be a wildcard bind address (§12)")
    if cfg.pipeline.window_max_s < 0:
        raise ConfigError("pipeline.window_max_s must be >= 0 (0 disables windowing)")
    if cfg.pipeline.window_max_s and cfg.pipeline.window_min_s > cfg.pipeline.window_max_s:
        raise ConfigError("pipeline.window_min_s must not exceed pipeline.window_max_s")
    if cfg.pipeline.window_cut_search_s < 0:
        raise ConfigError("pipeline.window_cut_search_s must be >= 0")

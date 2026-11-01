"""Dataclasses mirroring every stage artifact (ARCHITECTURE.md §3) and the job store rows (§4).

Kept intentionally dumb: no I/O, no validation logic beyond typing. `audio.py`/`asr.py`/
`diarize.py`/`merge.py`/`summarize.py` build these; `store.py` persists `Job`/`JobEvent`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Current artifact schema version. Bump on incompatible change; readers reject unknown values.
SCHEMA_VERSION = 1

#: Job state machine (§4). `state` is the *last completed stage*.
STAGE_ORDER: tuple[str, ...] = (
    "received",
    "normalized",
    "transcribed",
    "diarized",
    "merged",
    "summarized",
    "delivered",
)
#: All valid values of jobs.state, including the two terminal/side states.
JOB_STATES: tuple[str, ...] = STAGE_ORDER + ("failed", "summary_pending")

# --------------------------------------------------------------------------- audio.py [P2]


@dataclass(slots=True)
class NormalizedAudio:
    """normalized.json (§3.3)."""

    schema: int
    job_id: str
    source_name: str
    source_sha256: str
    source_bytes: int
    wav_path: str
    duration_s: float
    sample_rate: int
    channels: int
    container: str
    codec: str
    creation_time: str | None
    start_source: str  # "upload" | "metadata" | "filename" | "mtime"


# --------------------------------------------------------------------------- asr.py [P2]


@dataclass(slots=True)
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass(slots=True)
class AsrSegment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


@dataclass(slots=True)
class AsrResult:
    """asr.json (§3.4). `words` is one flat list over the whole recording, ordered by start.

    `windows` is the `[[start_s, end_s], ...]` plan the stage actually used (§3.4 "Long audio");
    timestamps are already global, so consumers never need it — it is persisted so a resumed or
    inspected job shows exactly where the cuts were. Absent in artifacts written before windowing
    existed, hence the default: `[]` simply means "not windowed".
    """

    schema: int
    job_id: str
    language: str
    language_probability: float
    language_forced: bool
    duration_s: float
    model: str
    segments: list[AsrSegment] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    windows: list[list[float]] = field(default_factory=list)


# --------------------------------------------------------------------------- diarize.py [P2]


@dataclass(slots=True)
class SpeakerSegment:
    """One row of diarization.json's `segments` (§3.5). `speaker` is an opaque cluster id."""

    start: float
    end: float
    speaker: int


@dataclass(slots=True)
class DiarizationResult:
    """diarization.json (§3.5). `windows` as in `AsrResult` (§3.5 "Long audio")."""

    schema: int
    job_id: str
    backend: str
    num_speakers_detected: int
    segments: list[SpeakerSegment] = field(default_factory=list)
    windows: list[list[float]] = field(default_factory=list)


# --------------------------------------------------------------------------- merge.py [P2]


@dataclass(slots=True)
class Turn:
    start: float
    end: float
    speaker: int  # raw cluster id, kept for debugging
    speaker_label: str  # "Speaker N", rendered
    text: str


@dataclass(slots=True)
class TurnsResult:
    """turns.json (§3.6)."""

    schema: int
    job_id: str
    language: str
    duration_s: float
    participantes: list[str] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)


# --------------------------------------------------------------------------- summarize.py [P3]


@dataclass(slots=True)
class SummaryHints:
    """Inputs to summarize() so Claude can only choose keys that exist (§3.7)."""

    titulo_usuario: str | None
    cliente_usuario: str | None
    clientes_catalogo: list[dict]
    categorias_catalogo: list[dict]


@dataclass(slots=True)
class MasDetalleItem:
    tema: str
    texto: str


@dataclass(slots=True)
class PuntoClave:
    emoji: str
    tema: str
    texto: str


@dataclass(slots=True)
class Tarea:
    responsable: str
    texto: str
    t_seg: int


@dataclass(slots=True)
class SummaryResult:
    """summary.json (§3.7, §8.2). `pending=True` means the note ships with `Resumen pendiente`."""

    schema: int
    job_id: str
    pending: bool
    chunks: int
    model: str
    titulo: str | None = None
    resumen_corto: str = ""
    resumen: str = ""
    mas_detalle: list[MasDetalleItem] = field(default_factory=list)
    puntos_clave: list[PuntoClave] = field(default_factory=list)
    tareas: list[Tarea] = field(default_factory=list)
    temas: list[str] = field(default_factory=list)
    palabras_clave: list[str] = field(default_factory=list)
    categoria: str = "otras"
    idioma: str = "es"
    cliente_sugerido: str | None = None
    confianza: float = 0.0
    error: str | None = None


# --------------------------------------------------------------------------- store.py [P1]


@dataclass(slots=True)
class Job:
    """One row of the `jobs` table (§4 DDL)."""

    id: str
    state: str
    created_at: str
    updated_at: str
    source_name: str
    origin: str  # "upload" | "inbox"
    audio_sha256: str
    source_path: str | None = None
    origin_ref: str | None = None
    duration_s: float | None = None
    title_user: str | None = None
    client_user: str | None = None
    language_user: str | None = None
    recorded_at_user: str | None = None
    speakers_user: int | None = None
    recorded_at: str | None = None
    start_source: str | None = None
    language: str | None = None
    client_key: str | None = None
    client_source: str | None = None
    note_path: str | None = None
    attempts: int = 0
    summary_attempts: int = 0
    next_attempt_at: str | None = None
    lease_until: str | None = None
    error: str | None = None


@dataclass(slots=True)
class JobEvent:
    """One row of the `job_events` table (§4 DDL)."""

    id: int
    job_id: str
    ts: str
    level: str  # "info" | "warn" | "error"
    stage: str
    message: str

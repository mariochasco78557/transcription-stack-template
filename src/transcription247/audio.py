"""ffprobe metadata + ffmpeg normalization to 16 kHz mono WAV (ARCHITECTURE.md §3.3, §9.2).

Stage 1 of the pipeline: `received -> normalized`. Produces `normalized.json`
(`dataclasses.asdict(NormalizedAudio)` is exactly the §3.3 payload).

No model is loaded here; the only external dependencies are the `ffmpeg`/`ffprobe` binaries named
in `[audio]` of config.toml. Every subprocess call has a timeout and turns any failure into
`AudioError`, which the worker maps to failure mode §13 #6 (no retries).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from transcription247 import paths
from transcription247.config import AudioConfig
from transcription247.ids import sha256_file
from transcription247.models import SCHEMA_VERSION, NormalizedAudio

log = logging.getLogger(__name__)

SUPPORTED_EXT = {
    ".m4a", ".mp3", ".wav", ".aac", ".ogg", ".opus", ".flac", ".mp4", ".mov", ".webm", ".caf",
}

#: Local timezone for every resolved recording start (§5 general.zona_horaria).
DEFAULT_TZ = "Europe/Madrid"

#: Subprocess timeouts. ffprobe only reads headers; ffmpeg decodes the whole file (a 2 h m4a
#: normalizes in well under a minute on this box, so an hour is a generous upper bound).
FFPROBE_TIMEOUT_S = 120
FFMPEG_TIMEOUT_S = 3600


class AudioError(RuntimeError):
    """Missing/unsupported/corrupt audio, a missing ffmpeg binary, or a non-zero ffmpeg exit."""


# --------------------------------------------------------------------------- subprocess plumbing


def _run(argv: list[str], timeout: float, what: str) -> subprocess.CompletedProcess[bytes]:
    """Run `argv`, never raising anything but AudioError. Output is captured, stdin is closed."""
    try:
        proc = subprocess.run(  # noqa: S603 - argv is built here, never shell-interpolated
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AudioError(f"{what}: binary not found: {argv[0]!r}") from exc
    except OSError as exc:
        raise AudioError(f"{what}: could not run {argv[0]!r}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"{what}: {argv[0]!r} timed out after {timeout:.0f}s") from exc
    if proc.returncode != 0:
        raise AudioError(f"{what}: {argv[0]!r} exited {proc.returncode}: {_tail(proc.stderr)}")
    return proc


def _tail(raw: bytes | None, max_chars: int = 400) -> str:
    text = (raw or b"").decode("utf-8", errors="replace").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return "(no stderr output)"
    return text[-max_chars:]


# --------------------------------------------------------------------------- probing


def probe(src: Path, cfg: AudioConfig | None = None) -> dict:
    """`ffprobe -v quiet -print_format json -show_format -show_streams <src>` as a dict."""
    cfg = cfg or AudioConfig()
    src = Path(src)
    if not src.is_file():
        raise AudioError(f"audio file does not exist: {src}")
    proc = _run(
        [cfg.ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
         str(src)],
        timeout=FFPROBE_TIMEOUT_S,
        what="probe",
    )
    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="replace") or "{}")
    except json.JSONDecodeError as exc:
        raise AudioError(f"probe: ffprobe returned unparseable JSON for {src.name}") from exc
    if not isinstance(data, dict) or not data.get("streams"):
        raise AudioError(f"probe: no decodable stream in {src.name} (corrupt or unsupported)")
    return data


def _audio_stream(info: dict) -> dict:
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            return stream
    raise AudioError("probe: file has no audio stream")


def _duration_from(info: dict) -> float | None:
    candidates = [info.get("format", {}).get("duration")]
    candidates += [s.get("duration") for s in info.get("streams", [])]
    for value in candidates:
        try:
            seconds = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            return seconds
    return None


# --------------------------------------------------------------------------- start-time helpers


def _parse_iso(raw: str, tz: str) -> datetime | None:
    text = raw.strip().replace("Z", "+00:00")
    # ffmpeg writes microseconds, some muxers write nanoseconds; fromisoformat wants <= 6 digits.
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(tz))
    return parsed.astimezone(ZoneInfo(tz))


def creation_time(src: Path, cfg: AudioConfig | None = None, tz: str = DEFAULT_TZ) -> datetime | None:
    """`format.tags.creation_time` (or the first stream tag) converted to `tz`; None if absent."""
    try:
        info = probe(src, cfg)
    except AudioError:
        return None
    return creation_time_from_probe(info, tz)


def creation_time_from_probe(info: dict, tz: str = DEFAULT_TZ) -> datetime | None:
    """Same as creation_time() but on an already-probed dict, so normalize() probes only once."""
    tag_sources = [info.get("format", {}).get("tags", {})]
    tag_sources += [s.get("tags", {}) for s in info.get("streams", [])]
    for tags in tag_sources:
        if not isinstance(tags, dict):
            continue
        for key, value in tags.items():
            if key.lower() == "creation_time" and isinstance(value, str):
                parsed = _parse_iso(value, tz)
                if parsed is not None:
                    return parsed
    return None


#: §9.2 filename patterns, in order: 'YYYYMMDD HHMMSS', 'YYYY-MM-DD HH.MM.SS',
#: 'YYYY-MM-DD_HH-MM-SS', 'YYYYMMDD_HHMMSS'. Matched anywhere in the original basename.
_FILENAME_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(\d{8} \d{6})"), "%Y%m%d %H%M%S"),
    (re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}\.\d{2}\.\d{2})"), "%Y-%m-%d %H.%M.%S"),
    (re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"), "%Y-%m-%d_%H-%M-%S"),
    (re.compile(r"(\d{8}_\d{6})"), "%Y%m%d_%H%M%S"),
)


def start_time_from_filename(name: str, tz: str = DEFAULT_TZ) -> datetime | None:
    """First §9.2 pattern that matches the basename, as a tz-aware datetime in `tz`."""
    base = Path(name).name
    for pattern, fmt in _FILENAME_PATTERNS:
        match = pattern.search(base)
        if not match:
            continue
        try:
            parsed = datetime.strptime(match.group(1), fmt)
        except ValueError:
            continue
        return parsed.replace(tzinfo=ZoneInfo(tz))
    return None


def resolve_start_time(
    src: Path,
    *,
    recorded_at_user: str | datetime | None = None,
    source_name: str | None = None,
    probe_info: dict | None = None,
    cfg: AudioConfig | None = None,
    tz: str = DEFAULT_TZ,
) -> tuple[datetime, str]:
    """§9.2 resolution order -> (start, start_source).

    upload (`recorded_at_user`) -> ffprobe `creation_time` -> filename patterns -> file mtime.
    `source_name` is the *original* filename (`jobs.source_name`), which is what the filename
    patterns must be matched against — `src` is usually the opaque `data/jobs/<id>/source.<ext>`.
    """
    if recorded_at_user:
        parsed = (
            recorded_at_user.astimezone(ZoneInfo(tz))
            if isinstance(recorded_at_user, datetime)
            else _parse_iso(recorded_at_user, tz)
        )
        if parsed is not None:
            return parsed, "upload"
        log.warning("recorded_at %r is not ISO8601; falling through to metadata", recorded_at_user)

    info = probe_info if probe_info is not None else None
    if info is None:
        try:
            info = probe(src, cfg)
        except AudioError:
            info = {}
    from_meta = creation_time_from_probe(info, tz)
    if from_meta is not None:
        return from_meta, "metadata"

    from_name = start_time_from_filename(source_name or Path(src).name, tz)
    if from_name is not None:
        return from_name, "filename"

    mtime = datetime.fromtimestamp(Path(src).stat().st_mtime, ZoneInfo(tz))
    return mtime, "mtime"


# --------------------------------------------------------------------------- normalization


def normalize(
    src: Path,
    dst_wav: Path,
    *,
    cfg: AudioConfig | None = None,
    job_id: str = "",
    source_name: str | None = None,
    source_sha256: str | None = None,
    recorded_at_user: str | datetime | None = None,
    tz: str = DEFAULT_TZ,
) -> NormalizedAudio:
    """ffmpeg -nostdin -hide_banner -loglevel error -i <src> -vn -ac 1 -ar 16000
    -acodec pcm_s16le -f wav <dst_wav>.

    AudioError on non-zero exit or duration < cfg.min_duracion_s.

    The positional signature is §3.3's; everything else is keyword-only with a safe default, so the
    worker can fill in the job fields `NormalizedAudio` carries (§3.3's JSON has job_id/source_*/
    start_source, which `normalize(src, dst)` alone cannot know).
    """
    cfg = cfg or AudioConfig()
    src = Path(src)
    dst_wav = Path(dst_wav)
    name = source_name or src.name

    if not src.is_file():
        raise AudioError(f"audio file does not exist: {src}")
    suffix = src.suffix.lower()
    if suffix not in SUPPORTED_EXT:
        raise AudioError(f"unsupported extension {suffix!r} for {name} (§13 #6)")
    source_bytes = src.stat().st_size
    if source_bytes == 0:
        raise AudioError(f"{name} is empty (0 bytes)")

    info = probe(src, cfg)
    stream = _audio_stream(info)
    container = info.get("format", {}).get("format_name", "") or ""
    codec = stream.get("codec_name", "") or ""
    src_duration = _duration_from(info)

    # Every filesystem write in the project goes through the guard (§12).
    dst_wav = paths.guard_write(dst_wav)
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_wav = dst_wav.with_name(dst_wav.name + ".part")

    log.info(
        "normalize: %s (%s/%s, %.1f s, %.1f MiB) -> %d Hz mono wav",
        name, container, codec, src_duration or -1.0, source_bytes / 1048576, cfg.sample_rate,
    )
    try:
        _run(
            [
                cfg.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(src), "-vn", "-ac", "1", "-ar", str(cfg.sample_rate),
                "-acodec", "pcm_s16le", "-f", "wav", str(tmp_wav),
            ],
            timeout=FFMPEG_TIMEOUT_S,
            what="normalize",
        )
        if not tmp_wav.is_file() or tmp_wav.stat().st_size <= 44:  # 44 = bare WAV header
            raise AudioError(f"normalize: ffmpeg produced no audio for {name} (corrupt input)")
        wav_info = probe(tmp_wav, cfg)
        duration_s = _duration_from(wav_info) or src_duration
        if duration_s is None:
            raise AudioError(f"normalize: could not determine the duration of {name}")
        if duration_s < cfg.min_duracion_s:
            raise AudioError(
                f"normalize: {name} is {duration_s:.2f}s, below min_duracion_s "
                f"({cfg.min_duracion_s}s) (§13 #6)"
            )
    except BaseException:
        tmp_wav.unlink(missing_ok=True)
        raise
    tmp_wav.replace(dst_wav)

    start, start_source = resolve_start_time(
        src,
        recorded_at_user=recorded_at_user,
        source_name=name,
        probe_info=info,
        cfg=cfg,
        tz=tz,
    )
    wav_stream = _audio_stream(wav_info)
    log.info("normalize: done, %.2f s, start=%s (%s)", duration_s, start.isoformat(), start_source)

    return NormalizedAudio(
        schema=SCHEMA_VERSION,
        job_id=job_id,
        source_name=name,
        source_sha256=source_sha256 if source_sha256 is not None else sha256_file(src),
        source_bytes=source_bytes,
        wav_path=str(dst_wav),
        duration_s=round(float(duration_s), 3),
        sample_rate=int(wav_stream.get("sample_rate", cfg.sample_rate)),
        channels=int(wav_stream.get("channels", 1)),
        container=container,
        codec=codec,
        creation_time=start.isoformat(),
        start_source=start_source,
    )

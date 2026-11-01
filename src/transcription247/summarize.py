"""Headless Claude Code CLI summarizer client (ARCHITECTURE.md §3.7, §8).

Calls the *subscription* `claude` CLI (never the API — `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`
are scrubbed from the child environment, §8.1/§12), extracts the strict JSON object the model was
asked for, validates it against the §8.2 schema, and retries once with a corrective preamble on
failure. `summarize()` raises `SummaryError` only when the corrected retry also fails (or, in a
map-reduce run, when the reduce call fails); the caller (`worker.py`, [INT]) is responsible for
writing the `pending` `summary.json` and delivering with the `Resumen pendiente` marker (§8.4).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from transcription247 import paths, prompts
from transcription247.config import Config, SummaryConfig, project_root
from transcription247.models import (
    SCHEMA_VERSION,
    MasDetalleItem,
    PuntoClave,
    SummaryHints,
    SummaryResult,
    Tarea,
    Turn,
    TurnsResult,
)

log = logging.getLogger(__name__)

#: Fallback for `general.idiomas_permitidos` (§5) when a caller uses the bare §3.7 signature.
#: Production callers never rely on it: the worker builds a `SummaryRuntime` from the full
#: `Config` (see `runtime_from_config`), which carries both the real allowed-language set and the
#: real `data/tmp/claude` cwd, so neither is guessed here.
_IDIOMAS_PERMITIDOS = ("es", "en")


@dataclass(frozen=True, slots=True)
class SummaryRuntime:
    """The bits of the *full* `Config` that §8 needs but `SummaryConfig` does not carry.

    §3.7 documents `summarize(turns_result, hints, cfg: SummaryConfig)`; that signature is kept,
    with this object passed as an optional keyword so the worker can hand over
    `general.idiomas_permitidos` (§8.4's `idioma` validation) and the `data/tmp/claude` working
    directory (§8.1) explicitly instead of either being inferred from module state.
    """

    idiomas_permitidos: tuple[str, ...] = _IDIOMAS_PERMITIDOS
    cwd: Path | None = None


def runtime_from_config(config: Config) -> SummaryRuntime:
    """Build the runtime context from a loaded `Config` — the worker's entry point."""
    return SummaryRuntime(
        idiomas_permitidos=tuple(config.general.idiomas_permitidos),
        cwd=config.project_root / config.paths.data_dir / "tmp" / "claude",
    )


class SummaryError(RuntimeError):
    """Raised when the summarizer cannot produce a valid summary after the one retry (§8.4)."""


class _ValidationFailed(SummaryError):
    """Internal: unparseable/empty model response. Caught the same way as a hard call failure so
    a single retry loop (`_call_and_validate_with_retry`) covers both (§8.4)."""


_NOT_LOGGED_IN_RE = re.compile(
    r"not logged in|please (run|use).{0,20}login|/login|authentication required", re.IGNORECASE
)
_RATE_LIMIT_RE = re.compile(r"rate.?limit|usage limit|quota exceeded|too many requests", re.IGNORECASE)

_MAX_FAILURE_DETAIL = 300


# --------------------------------------------------------------------------- chunking (§8.3)


def chunk_turns(turns: list[Turn], chunk_seconds: int) -> list[list[Turn]]:
    """Accumulate whole turns into chunks spanning ~`chunk_seconds` of wall-clock time, on turn
    boundaries only (never splitting a turn). A single turn longer than `chunk_seconds` still gets
    its own chunk. Empty input -> `[]`."""
    if not turns:
        return []
    chunks: list[list[Turn]] = []
    current: list[Turn] = []
    chunk_start = turns[0].start
    for turn in turns:
        if current and (turn.end - chunk_start) > chunk_seconds:
            chunks.append(current)
            current = []
            chunk_start = turn.start
        current.append(turn)
    if current:
        chunks.append(current)
    return chunks


def render_transcript_for_prompt(turns: list[Turn]) -> str:
    """"[MM:SS] Speaker N: texto" lines, one per turn (§8.3)."""
    lines = []
    for turn in turns:
        text = " ".join(turn.text.split())
        lines.append(f"[{paths.fmt_ts(turn.start)}] {turn.speaker_label}: {text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI invocation (§8.1)


def _claude_cwd(runtime: SummaryRuntime | None = None) -> Path:
    """`data/tmp/claude` (§8.1). Taken from `runtime.cwd` when the caller supplied one (the worker
    always does, from the full `Config`); otherwise resolved against the configured `DATA_ROOT`,
    else against the project root — the fallback only exists so a bare §3.7-signature call still
    works in tests. Routed through `paths.guard_write()` before creation (§12: it lives under the
    project's data dir, and every write under Drive/data goes through the guard, no exceptions)."""
    if runtime is not None and runtime.cwd is not None:
        cwd = runtime.cwd
    else:
        base = paths.DATA_ROOT if paths.DATA_ROOT is not None else (project_root() / "data")
        cwd = base / "tmp" / "claude"
    cwd = paths.guard_write(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


def _extract_json_object(text: str) -> dict:
    """Robustly pull the first top-level JSON object out of `text`, tolerating a leading ```` ```json
    ```` fence or leading/trailing prose: find the first `{` and decode from there, ignoring
    whatever follows the matching `}`."""
    if not text or not text.strip():
        raise _ValidationFailed("empty model response")
    start = text.find("{")
    if start == -1:
        raise _ValidationFailed("no JSON object found in model response")
    try:
        obj, _end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise _ValidationFailed(f"unparseable JSON in model response: {exc}") from exc
    if not isinstance(obj, dict):
        raise _ValidationFailed("model response JSON is not an object")
    return obj


def _failure_reason(returncode: int, stdout: str, stderr: str, envelope: dict | None = None) -> str:
    haystack = " ".join(filter(None, [stdout, stderr, json.dumps(envelope) if envelope else ""]))
    if _NOT_LOGGED_IN_RE.search(haystack):
        return "claude CLI not logged in"
    if _RATE_LIMIT_RE.search(haystack):
        return "claude CLI rate-limited"
    detail = (stderr or stdout or "").strip()[:_MAX_FAILURE_DETAIL]
    if detail:
        return f"claude CLI failed (exit {returncode}): {detail}"
    return f"claude CLI failed (exit {returncode})"


def call_claude(prompt: str, cfg: SummaryConfig, runtime: SummaryRuntime | None = None) -> dict:
    """Run the CLI, parse the envelope, strip fences, `json.loads` the payload.
    `SummaryError` on timeout, non-zero exit, `is_error`, or unparseable JSON (§8.1)."""
    argv = [
        cfg.claude_bin,
        "-p",
        "--model",
        cfg.modelo,
        "--output-format",
        "json",
        "--tools",
        "",
        "--safe-mode",
        "--no-session-persistence",
    ]
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)

    try:
        proc = subprocess.run(
            argv,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_s,
            cwd=_claude_cwd(runtime),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise SummaryError(f"claude CLI timed out after {cfg.timeout_s}s") from exc
    except FileNotFoundError as exc:
        raise SummaryError(f"claude binary not found: {cfg.claude_bin}") from exc

    if proc.returncode != 0:
        raise SummaryError(_failure_reason(proc.returncode, proc.stdout, proc.stderr))

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SummaryError(f"claude CLI returned a non-JSON envelope: {exc}") from exc
    if not isinstance(envelope, dict):
        raise SummaryError("claude CLI envelope is not a JSON object")

    if (
        envelope.get("is_error", True) is not False
        or envelope.get("subtype") != "success"
        or envelope.get("api_error_status") is not None
    ):
        raise SummaryError(_failure_reason(proc.returncode, proc.stdout, proc.stderr, envelope))

    result_text = envelope.get("result")
    if not isinstance(result_text, str):
        raise SummaryError("claude CLI envelope is missing 'result'")

    return _extract_json_object(result_text)


# --------------------------------------------------------------------------- validation (§8.2, §8.4)


def _coerce_str(value, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _coerce_opt_str(value) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _coerce_str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]


def _coerce_mas_detalle(value) -> list[MasDetalleItem]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("tema"), str) and isinstance(item.get("texto"), str):
            out.append(MasDetalleItem(tema=item["tema"], texto=item["texto"]))
    return out


def _coerce_puntos_clave(value) -> list[PuntoClave]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, dict) and all(isinstance(item.get(k), str) for k in ("emoji", "tema", "texto")):
            out.append(PuntoClave(emoji=item["emoji"], tema=item["tema"], texto=item["texto"]))
    return out


def _coerce_t_seg(value, duration_s: float) -> int:
    try:
        t = int(float(value))
    except (TypeError, ValueError):
        t = 0
    t = max(0, t)
    if duration_s and duration_s > 0:
        t = min(t, int(duration_s))
    return t


def _coerce_tareas(value, duration_s: float) -> list[Tarea]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        responsable, texto = item.get("responsable"), item.get("texto")
        if not isinstance(responsable, str) or not isinstance(texto, str):
            continue
        out.append(Tarea(responsable=responsable, texto=texto, t_seg=_coerce_t_seg(item.get("t_seg"), duration_s)))
    return out


def _coerce_confianza(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def _catalog_keys(catalog: list[dict]) -> set[str]:
    return {item.get("key") for item in catalog if isinstance(item, dict) and item.get("key")}


def _validate(
    obj: dict,
    *,
    turns_result: TurnsResult,
    hints: SummaryHints,
    model: str,
    chunks: int,
    idiomas_permitidos: tuple[str, ...] = _IDIOMAS_PERMITIDOS,
) -> SummaryResult:
    """Coerce+validate a full-schema object into a `SummaryResult` (§8.2, §8.4). Missing optional
    keys -> empty list/null; wrong-typed elements dropped; unknown enum values fall back to their
    documented default. Raises `_ValidationFailed` only when both `resumen_corto` and
    `puntos_clave` are empty — every other deviation is silently coerced."""
    if not isinstance(obj, dict):
        raise _ValidationFailed("model response is not a JSON object")

    resumen_corto = _coerce_str(obj.get("resumen_corto"))
    puntos_clave = _coerce_puntos_clave(obj.get("puntos_clave"))
    if not resumen_corto.strip() and not puntos_clave:
        raise _ValidationFailed("both resumen_corto and puntos_clave are empty")

    categorias_validas = _catalog_keys(hints.categorias_catalogo)
    clientes_validos = _catalog_keys(hints.clientes_catalogo)

    categoria = obj.get("categoria")
    if categoria not in categorias_validas:
        categoria = "otras"

    cliente_sugerido = obj.get("cliente_sugerido")
    if cliente_sugerido not in clientes_validos:
        cliente_sugerido = None

    idioma = obj.get("idioma")
    if idioma not in idiomas_permitidos:
        idioma = turns_result.language

    return SummaryResult(
        schema=SCHEMA_VERSION,
        job_id=turns_result.job_id,
        pending=False,
        chunks=chunks,
        model=model,
        titulo=_coerce_opt_str(obj.get("titulo")),
        resumen_corto=resumen_corto,
        resumen=_coerce_str(obj.get("resumen")),
        mas_detalle=_coerce_mas_detalle(obj.get("mas_detalle")),
        puntos_clave=puntos_clave,
        tareas=_coerce_tareas(obj.get("tareas"), turns_result.duration_s),
        temas=_coerce_str_list(obj.get("temas")),
        palabras_clave=_coerce_str_list(obj.get("palabras_clave")),
        categoria=categoria,
        idioma=idioma,
        cliente_sugerido=cliente_sugerido,
        confianza=_coerce_confianza(obj.get("confianza")),
    )


def _validate_partial(obj: dict, duration_s: float) -> dict:
    """Coerce+validate one map chunk's partial-schema response (§8.3) into a plain dict, ready to
    be embedded (as JSON) in the reduce prompt. Same empty-content failure rule as `_validate`."""
    if not isinstance(obj, dict):
        raise _ValidationFailed("model response is not a JSON object")

    resumen_corto = _coerce_str(obj.get("resumen_corto"))
    puntos_clave = _coerce_puntos_clave(obj.get("puntos_clave"))
    if not resumen_corto.strip() and not puntos_clave:
        raise _ValidationFailed("both resumen_corto and puntos_clave are empty")

    return {
        "resumen_corto": resumen_corto,
        "puntos_clave": [{"emoji": p.emoji, "tema": p.tema, "texto": p.texto} for p in puntos_clave],
        "tareas": [
            {"responsable": t.responsable, "texto": t.texto, "t_seg": t.t_seg}
            for t in _coerce_tareas(obj.get("tareas"), duration_s)
        ],
        "temas": _coerce_str_list(obj.get("temas")),
        "palabras_clave": _coerce_str_list(obj.get("palabras_clave")),
    }


# --------------------------------------------------------------------------- retry loop (§8.4)


def _call_and_validate_with_retry(
    prompt: str, cfg: SummaryConfig, validate_fn, runtime: SummaryRuntime | None = None
):
    """Call + validate; on any `SummaryError` (call failure or validation failure), retry once
    (per `cfg.max_reintentos`, default 1) with the corrective preamble prepended. Re-raises the
    last error if every attempt fails."""
    last_exc: SummaryError | None = None
    for attempt in range(cfg.max_reintentos + 1):
        attempt_prompt = prompt if attempt == 0 else f"{prompts.RETRY_PREAMBLE}\n\n{prompt}"
        try:
            obj = call_claude(attempt_prompt, cfg, runtime)
            return validate_fn(obj)
        except SummaryError as exc:
            last_exc = exc
            log.info("summarize: attempt %d/%d failed: %s", attempt + 1, cfg.max_reintentos + 1, exc)
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------- orchestration (§8.3)


def summarize(
    turns_result: TurnsResult,
    hints: SummaryHints,
    cfg: SummaryConfig,
    runtime: SummaryRuntime | None = None,
) -> SummaryResult:
    """Map-reduce when needed, validate, one retry per call, raise `SummaryError` when a required
    call (single or reduce) still fails after its retry (§8.3-8.4).

    `runtime` carries what only the full `Config` knows (allowed languages, the claude cwd);
    `worker.py` builds it with `runtime_from_config(cfg)`.
    """
    runtime = runtime or SummaryRuntime()
    chunks = chunk_turns(turns_result.turns, cfg.chunk_seconds)

    if len(chunks) <= 1:
        turns_for_call = chunks[0] if chunks else []
        prompt = prompts.build_single_prompt(
            transcript=render_transcript_for_prompt(turns_for_call),
            hints=hints,
            language=turns_result.language,
        )
        return _call_and_validate_with_retry(
            prompt,
            cfg,
            lambda obj: _validate(
                obj,
                turns_result=turns_result,
                hints=hints,
                model=cfg.modelo,
                chunks=1,
                idiomas_permitidos=runtime.idiomas_permitidos,
            ),
            runtime,
        )

    return _map_reduce(turns_result, chunks, hints, cfg, runtime)


def _map_reduce(
    turns_result: TurnsResult,
    chunks: list[list[Turn]],
    hints: SummaryHints,
    cfg: SummaryConfig,
    runtime: SummaryRuntime,
) -> SummaryResult:
    total = len(chunks)
    partials: list[dict] = []
    for index, chunk in enumerate(chunks, start=1):
        prompt = prompts.build_map_prompt(
            transcript=render_transcript_for_prompt(chunk),
            hints=hints,
            language=turns_result.language,
            index=index,
            total=total,
        )
        try:
            partials.append(
                _call_and_validate_with_retry(
                    prompt,
                    cfg,
                    lambda obj: _validate_partial(obj, turns_result.duration_s),
                    runtime,
                )
            )
        except SummaryError as exc:
            log.warning("summarize: map chunk %d/%d failed and was skipped: %s", index, total, exc)

    if not partials:
        raise SummaryError("all map chunks failed to summarize")

    reduce_prompt = prompts.build_reduce_prompt(map_results=partials, hints=hints, language=turns_result.language)
    return _call_and_validate_with_retry(
        reduce_prompt,
        cfg,
        lambda obj: _validate(
            obj,
            turns_result=turns_result,
            hints=hints,
            model=cfg.modelo,
            chunks=total,
            idiomas_permitidos=runtime.idiomas_permitidos,
        ),
        runtime,
    )

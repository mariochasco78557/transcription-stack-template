"""Summarizer: JSON extraction, validate/retry/fail flow, chunking (ARCHITECTURE.md §3.7, §8).

The `claude` subprocess is always mocked here (`summarize.subprocess.run`) — no network, no real
CLI call, no cost. The one real end-to-end smoke test against the live `claude` CLI is run
separately (max 2 real calls total, per the task brief) and is not part of this suite.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from transcription247 import paths, prompts, summarize
from transcription247.config import SummaryConfig
from transcription247.models import SummaryHints, Turn, TurnsResult


@pytest.fixture(autouse=True)
def _tmp_claude_cwd(tmp_path):
    """`call_claude()` resolves its subprocess cwd from `paths.DATA_ROOT` when set, and
    `_claude_cwd()` now creates that directory through `paths.guard_write()` (finding #2), which
    requires OUTPUT_ROOT/INBOX_ROOT/DATA_ROOT to all be configured. Point all three at tmp dirs —
    mirroring what `paths.configure(cfg)` does at real process startup — instead of the previous
    `DATA_ROOT`-only monkeypatch, so nothing is written under the real project tree and
    `guard_write` does not raise for lack of configuration."""
    paths.set_roots(
        output_root=tmp_path / "salida", inbox_root=tmp_path / "inbox", data_root=tmp_path
    )


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _envelope(result_text: str) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "api_error_status": None,
            "result": result_text,
        }
    )


def _cfg(**overrides) -> SummaryConfig:
    return replace(SummaryConfig(), **overrides)


def _turn(start: float, end: float, speaker: int, text: str) -> Turn:
    label = f"Speaker {speaker + 1}"
    return Turn(start=start, end=end, speaker=speaker, speaker_label=label, text=text)


def _turns_result(turns: list[Turn], job_id: str = "01job", language: str = "es") -> TurnsResult:
    duration = turns[-1].end if turns else 0.0
    participantes = sorted({t.speaker_label for t in turns})
    return TurnsResult(
        schema=1, job_id=job_id, language=language, duration_s=duration,
        participantes=participantes, turns=turns,
    )


def _hints() -> SummaryHints:
    return SummaryHints(
        titulo_usuario=None,
        cliente_usuario=None,
        clientes_catalogo=[{"key": "centro_ejemplo", "nombre": "Centro Ejemplo"}],
        categorias_catalogo=[{"key": "claustro", "nombre": "Claustro"}, {"key": "otras", "nombre": "Otras"}],
    )


def _full_obj(**overrides) -> dict:
    obj = {
        "titulo": None,
        "resumen_corto": "Resumen corto de la reunión.",
        "resumen": "Resumen largo de la reunión.",
        "mas_detalle": [{"tema": "Tema", "texto": "Detalle"}],
        "puntos_clave": [{"emoji": "📌", "tema": "Tema", "texto": "Punto"} for _ in range(4)],
        "tareas": [{"responsable": "Speaker 1", "texto": "Hacer algo", "t_seg": 10}],
        "temas": ["tema1"],
        "palabras_clave": ["a", "b", "c", "d"],
        "categoria": "claustro",
        "idioma": "es",
        "cliente_sugerido": "centro_ejemplo",
        "confianza": 0.9,
    }
    obj.update(overrides)
    return obj


# --------------------------------------------------------------------------- JSON extraction


def test_extract_json_object_tolerates_fence_and_leading_prose():
    text = 'Aquí tienes el resultado:\n```json\n{"a": 1, "b": [1, 2]}\n```\nGracias.'
    assert summarize._extract_json_object(text) == {"a": 1, "b": [1, 2]}


def test_extract_json_object_tolerates_prose_without_fence():
    text = 'Claro, aquí está el JSON solicitado: {"a": 1} -- espero que ayude'
    assert summarize._extract_json_object(text) == {"a": 1}


def test_extract_json_object_raises_on_garbage():
    with pytest.raises(summarize.SummaryError):
        summarize._extract_json_object("no hay ningún json en este texto")


# --------------------------------------------------------------------------- validate -> retry -> success


def test_validation_failure_then_retry_succeeds(monkeypatch):
    calls: list[str] = []

    def fake_run(argv, input=None, **kwargs):
        calls.append(input)
        if len(calls) == 1:
            stdout = _envelope("esto no es JSON {{{")
        else:
            stdout = _envelope(json.dumps(_full_obj()))
        return _FakeProc(returncode=0, stdout=stdout)

    monkeypatch.setattr(summarize.subprocess, "run", fake_run)

    turns = [_turn(0, 30, 0, "Hola a todos."), _turn(30, 60, 1, "Buenos días.")]
    result = summarize.summarize(_turns_result(turns), _hints(), _cfg())

    assert result.resumen_corto == "Resumen corto de la reunión."
    assert result.pending is False
    assert len(calls) == 2
    assert prompts.RETRY_PREAMBLE in calls[1]
    assert prompts.RETRY_PREAMBLE not in calls[0]


def test_hard_failure_raises_summary_error(monkeypatch):
    def fake_run(argv, input=None, **kwargs):
        return _FakeProc(returncode=0, stdout=_envelope("sigue sin ser JSON"))

    monkeypatch.setattr(summarize.subprocess, "run", fake_run)

    turns = [_turn(0, 30, 0, "Hola a todos.")]
    with pytest.raises(summarize.SummaryError):
        summarize.summarize(_turns_result(turns), _hints(), _cfg())


def test_call_claude_nonzero_exit_surfaces_not_logged_in(monkeypatch):
    def fake_run(argv, input=None, **kwargs):
        return _FakeProc(returncode=1, stdout="", stderr="Error: not logged in. Run `claude login`.")

    monkeypatch.setattr(summarize.subprocess, "run", fake_run)

    with pytest.raises(summarize.SummaryError, match="not logged in"):
        summarize.call_claude("prompt", _cfg())


def test_call_claude_detects_rate_limit(monkeypatch):
    def fake_run(argv, input=None, **kwargs):
        return _FakeProc(returncode=1, stdout="", stderr="429 rate limit exceeded, try later")

    monkeypatch.setattr(summarize.subprocess, "run", fake_run)

    with pytest.raises(summarize.SummaryError, match="rate-limited"):
        summarize.call_claude("prompt", _cfg())


def test_call_claude_scrubs_api_key_from_child_env(monkeypatch):
    captured_env = {}

    def fake_run(argv, input=None, env=None, **kwargs):
        captured_env.update(env or {})
        return _FakeProc(returncode=0, stdout=_envelope(json.dumps(_full_obj())))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "also-should-not-be-passed")
    monkeypatch.setattr(summarize.subprocess, "run", fake_run)

    summarize.call_claude("prompt", _cfg())

    assert "ANTHROPIC_API_KEY" not in captured_env
    assert "ANTHROPIC_AUTH_TOKEN" not in captured_env


# --------------------------------------------------------------------------- chunking (§8.3)


def test_chunk_turns_splits_on_time_boundary_without_splitting_a_turn():
    a = _turn(0, 100, 0, "a")
    b = _turn(200, 1400, 1, "b")
    c = _turn(1450, 1600, 0, "c")
    d = _turn(1650, 1700, 1, "d")
    chunks = summarize.chunk_turns([a, b, c, d], chunk_seconds=1500)
    assert chunks == [[a, b], [c, d]]


def test_chunk_turns_never_splits_a_single_long_turn():
    long_turn = _turn(0, 2000, 0, "monólogo largo")
    chunks = summarize.chunk_turns([long_turn], chunk_seconds=1500)
    assert chunks == [[long_turn]]


def test_chunk_turns_empty_input_returns_empty_list():
    assert summarize.chunk_turns([], chunk_seconds=1500) == []


def test_render_transcript_for_prompt_format():
    turns = [_turn(61.4, 70, 0, "Hola   mundo  ")]
    text = summarize.render_transcript_for_prompt(turns)
    assert text == "[01:01] Speaker 1: Hola mundo"


# --------------------------------------------------------------------------- map-reduce (§8.3)


def test_map_reduce_skips_a_failed_chunk_but_still_succeeds(monkeypatch):
    # chunk_seconds=100 forces two chunks out of these turns.
    turns = [
        _turn(0, 50, 0, "Primer tramo, parte uno."),
        _turn(60, 90, 1, "Primer tramo, parte dos."),
        _turn(150, 190, 0, "Segundo tramo, parte uno."),
        _turn(200, 230, 1, "Segundo tramo, parte dos."),
    ]
    calls: list[str] = []

    def fake_run(argv, input=None, **kwargs):
        calls.append(input)
        n = len(calls)
        if n <= 2:  # both attempts for chunk 1 fail -> chunk is skipped
            return _FakeProc(returncode=0, stdout=_envelope("no json"))
        if n == 3:  # chunk 2 succeeds first try (partial schema)
            partial = {
                "resumen_corto": "Resumen del segundo tramo.",
                "puntos_clave": [{"emoji": "📌", "tema": "t", "texto": "x"}],
                "tareas": [],
                "temas": ["t2"],
                "palabras_clave": ["a", "b", "c", "d"],
            }
            return _FakeProc(returncode=0, stdout=_envelope(json.dumps(partial)))
        return _FakeProc(returncode=0, stdout=_envelope(json.dumps(_full_obj())))  # reduce call

    monkeypatch.setattr(summarize.subprocess, "run", fake_run)

    cfg = _cfg(chunk_seconds=100)
    result = summarize.summarize(_turns_result(turns), _hints(), cfg)

    assert len(calls) == 4  # 2 failed attempts (chunk 1) + 1 success (chunk 2) + 1 reduce
    assert result.resumen_corto == "Resumen corto de la reunión."
    assert result.chunks == 2


def test_map_reduce_raises_when_every_chunk_fails(monkeypatch):
    turns = [
        _turn(0, 50, 0, "Primer tramo."),
        _turn(150, 190, 1, "Segundo tramo."),
    ]

    def fake_run(argv, input=None, **kwargs):
        return _FakeProc(returncode=0, stdout=_envelope("no json"))

    monkeypatch.setattr(summarize.subprocess, "run", fake_run)

    cfg = _cfg(chunk_seconds=100)
    with pytest.raises(summarize.SummaryError, match="all map chunks failed"):
        summarize.summarize(_turns_result(turns), _hints(), cfg)


# --------------------------------------------------------------------------- validate() coercions


def test_validate_falls_back_unknown_categoria_and_cliente_and_clamps_confianza():
    turns = [_turn(0, 30, 0, "Hola.")]
    tr = _turns_result(turns)
    obj = _full_obj(categoria="no_existe", cliente_sugerido="no_existe", confianza=5.0)
    result = summarize._validate(obj, turns_result=tr, hints=_hints(), model="sonnet", chunks=1)
    assert result.categoria == "otras"
    assert result.cliente_sugerido is None
    assert result.confianza == 1.0


def test_validate_clamps_t_seg_to_duration():
    turns = [_turn(0, 30, 0, "Hola.")]
    tr = _turns_result(turns)
    obj = _full_obj(tareas=[{"responsable": "Speaker 1", "texto": "x", "t_seg": 9999}])
    result = summarize._validate(obj, turns_result=tr, hints=_hints(), model="sonnet", chunks=1)
    assert result.tareas[0].t_seg == int(tr.duration_s)

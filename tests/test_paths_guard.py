"""guard_write() must be the only way to touch the filesystem safely (ARCHITECTURE.md §3.1, §12).

Every test points OUTPUT_ROOT/INBOX_ROOT/DATA_ROOT at tmp_path via paths.set_roots() — never at a
real Drive path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcription247 import paths


@pytest.fixture(autouse=True)
def _tmp_roots(tmp_path: Path):
    output_root = tmp_path / "drive" / "Transcripciones_24_7"
    inbox_root = tmp_path / "drive" / "Transcripciones_24_7" / "_INBOX_AUDIO"
    data_root = tmp_path / "data"
    output_root.mkdir(parents=True)
    inbox_root.mkdir(parents=True)
    data_root.mkdir(parents=True)
    paths.set_roots(output_root=output_root, inbox_root=inbox_root, data_root=data_root)
    yield output_root, inbox_root, data_root


def test_guard_write_accepts_legitimate_targets(_tmp_roots):
    output_root, inbox_root, data_root = _tmp_roots
    assert paths.guard_write(output_root / "Cliente" / "2026-09" / "nota.md") == (
        output_root / "Cliente" / "2026-09" / "nota.md"
    )
    assert paths.guard_write(inbox_root / "grabacion.m4a").exists() is False
    assert paths.guard_write(data_root / "jobs" / "01x" / "asr.json").is_relative_to(data_root)
    # the root itself is a legitimate "candidate" too
    assert paths.guard_write(output_root) == output_root.resolve()


def test_guard_write_rejects_dot_dot_escape(_tmp_roots):
    output_root, _, _ = _tmp_roots
    with pytest.raises(paths.PathGuardError):
        paths.guard_write(output_root / ".." / "escaped.md")


def test_guard_write_rejects_absolute_escape(_tmp_roots):
    with pytest.raises(paths.PathGuardError):
        paths.guard_write(Path("/etc/passwd"))


def test_guard_write_rejects_symlink_escape(tmp_path: Path, _tmp_roots):
    output_root, _, _ = _tmp_roots
    outside = tmp_path / "outside"
    outside.mkdir()
    link = output_root / "escape_link"
    link.symlink_to(outside)
    with pytest.raises(paths.PathGuardError):
        paths.guard_write(link / "nota.md")


def test_guard_write_denies_sibling_fireflies_archive(tmp_path: Path):
    # Even inside an otherwise-legitimate root, a path segment named exactly like the sibling
    # protected archive or the private folder is a hard deny (§12).
    output_root = tmp_path / "drive" / "_Protected_Archive"  # not the normal output root
    output_root.mkdir(parents=True)
    paths.set_roots(output_root=output_root, inbox_root=tmp_path / "inbox", data_root=tmp_path / "data")
    with pytest.raises(paths.PathGuardError):
        paths.guard_write(output_root / "nota.md")


def test_guard_write_denies_private_segment(tmp_path: Path, _tmp_roots):
    output_root, _, _ = _tmp_roots
    with pytest.raises(paths.PathGuardError):
        paths.guard_write(output_root / "_Protected_Private" / "nota.md")


def test_guard_write_before_configure_raises():
    paths.OUTPUT_ROOT = None
    paths.INBOX_ROOT = None
    paths.DATA_ROOT = None
    with pytest.raises(paths.PathGuardError):
        paths.guard_write(Path("/tmp/whatever.md"))


def test_sanitize_title_basic_rules():
    assert paths.sanitize_title("") == "Reunión sin título"
    assert paths.sanitize_title("a/b\\c:d*e?f\"g<h>i|j") == "a-b-c-d-e-f-g-h-i-j"
    assert paths.sanitize_title("  hola   mundo  ") == "hola mundo"
    assert paths.sanitize_title("...leading dots") == "leading dots"
    assert paths.sanitize_title("trailing dots...   ") == "trailing dots"
    assert paths.sanitize_title("a\x00b\x1fc") == "abc"


def test_sanitize_title_truncates_on_word_boundary():
    raw = "palabra " * 30  # far over max_len
    out = paths.sanitize_title(raw, max_len=20)
    assert len(out) <= 20
    assert not out.endswith(" ")


def test_fmt_ts():
    assert paths.fmt_ts(59.4) == "00:59"
    assert paths.fmt_ts(3600) == "1:00:00"
    assert paths.fmt_ts(61) == "01:01"


def test_extract_speaker_hint():
    assert paths.extract_speaker_hint("Reunion [2p] equipo.m4a") == ("Reunion equipo.m4a", 2)
    assert paths.extract_speaker_hint("Reunion (3P) equipo.m4a") == ("Reunion equipo.m4a", 3)
    assert paths.extract_speaker_hint("Reunion equipo.m4a") == ("Reunion equipo.m4a", None)
    assert paths.extract_speaker_hint("Reunion [15p] equipo.m4a") == ("Reunion [15p] equipo.m4a", None)

"""Transcription 24/7 — self-hosted audio-in / Markdown-meeting-notes-out pipeline.

See docs/ARCHITECTURE.md for the full spec. `__version__` mirrors the default of
`config.GeneralConfig.version_modulo` (§5) — the "design freeze" version of this contract, not the
PyPI-style package version in pyproject.toml.
"""

__version__ = "1.0.0"


def main() -> None:
    from transcription247.cli import main as cli_main

    raise SystemExit(cli_main())

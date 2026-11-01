"""ULID job ids and sha256 helpers (ARCHITECTURE.md §3.2)."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

_CROCKFORD = "0123456789abcdefghjkmnpqrstvwxyz"  # lowercase, excludes i l o u
_CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming reads


def new_job_id() -> str:
    """26-char lowercase Crockford ULID: 48-bit ms timestamp + 80 random bits, time-sortable."""
    ms = time.time_ns() // 1_000_000
    randomness = int.from_bytes(os.urandom(10), "big")
    value = (ms << 80) | randomness
    chars = []
    for _ in range(26):
        value, rem = divmod(value, 32)
        chars.append(_CROCKFORD[rem])
    return "".join(reversed(chars))


def sha256_file(p: Path) -> str:
    """Streaming SHA-256 of a file, 1 MiB chunks, lowercase hex."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(_CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

"""Pluggable notifier: Telegram Bot API via stdlib urllib (ARCHITECTURE.md §3.8).

Best-effort only. A notifier must never raise into the worker: `send()` always returns a bool and
logs on failure instead of propagating.
"""

from __future__ import annotations

import html
import json
import logging
import urllib.error
import urllib.request
from typing import Protocol

from transcription247.config import Config

log = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class Notifier(Protocol):
    def send(self, text: str) -> bool: ...


class NullNotifier:
    """Used when no telegram.json secret is present. Always a no-op success."""

    def send(self, text: str) -> bool:
        log.debug("NullNotifier: %s", text)
        return True


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, timeout: float = 10.0) -> None:
        self._token = token
        self._chat_id = chat_id
        self._timeout = timeout

    def send(self, text: str) -> bool:
        """POST sendMessage via urllib.request (stdlib only). Never raises; returns False and
        logs on any failure (network error, timeout, non-2xx response)."""
        url = _TELEGRAM_API.format(token=self._token)
        payload = json.dumps(
            {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                ok = 200 <= resp.status < 300
                if not ok:
                    log.warning("telegram notify: unexpected status %s", resp.status)
                return ok
        except (urllib.error.URLError, OSError, ValueError) as exc:
            log.warning("telegram notify failed: %s", exc)
            return False


def get_notifier(cfg: Config) -> Notifier:
    """TelegramNotifier when ~/.config/transcription247/telegram.json is present and valid
    (0600, {token, chat_id}); NullNotifier otherwise (no secrets file, or unreadable)."""
    secret = cfg.secret_telegram()
    if not secret:
        return NullNotifier()
    token = secret.get("token")
    chat_id = secret.get("chat_id")
    if not token or not chat_id:
        log.warning("telegram.json missing token/chat_id; falling back to NullNotifier")
        return NullNotifier()
    return TelegramNotifier(token=token, chat_id=chat_id)


# --------------------------------------------------------------------------- message formatting
#
# Optional helpers for the worker (§14 [INT]) to build the text passed to Notifier.send(). Not
# part of the §3.8 signature block, but "success + failure messages per spec" is explicitly listed
# under notify.py's ownership, so they live here rather than being reimplemented per caller.


def format_success_message(
    *, title: str, duration_s: float | None, client_name: str, resumen_corto: str, note_path: str
) -> str:
    """Title, duration, client, resumen_corto, Drive path (§3.8)."""
    minutes = max(1, round((duration_s or 0.0) / 60))
    return (
        f"✅ <b>{html.escape(title)}</b>\n"
        f"Duración: {minutes} min\n"
        f"Cliente: {html.escape(client_name)}\n"
        f"{html.escape(resumen_corto)}\n"
        f"{html.escape(note_path)}"
    )


def format_failure_message(*, job_id: str, source_name: str, stage: str, error: str) -> str:
    """Job id, source name, stage, error (§3.8)."""
    return (
        f"⚠️ Fallo en transcripción\n"
        f"Job: {html.escape(job_id)}\n"
        f"Fichero: {html.escape(source_name)}\n"
        f"Etapa: {html.escape(stage)}\n"
        f"Error: {html.escape(error)}"
    )

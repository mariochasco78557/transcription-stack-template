#!/usr/bin/env bash
# Remove the Transcription 24/7 systemd --user units (ARCHITECTURE.md §11).
#
# Units only. It never removes data/, never removes the SQLite queue, never removes the secrets
# directory and never touches a single file in Google Drive — it prints where they all are instead.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
UNIT_DIR="${HOME}/.config/systemd/user"
UNITS=(t247-worker.service t247-api.service)

printf '\n== Parando y deshabilitando\n'
for unit in "${UNITS[@]}"; do
  systemctl --user disable --now "${unit}" 2>/dev/null || true
  printf '  %s\n' "${unit}"
done

printf '\n== Borrando los ficheros de unit\n'
for unit in "${UNITS[@]}"; do
  rm -f "${UNIT_DIR}/${unit}"
  printf '  %s\n' "${UNIT_DIR}/${unit}"
done

systemctl --user daemon-reload
systemctl --user reset-failed "${UNITS[@]}" 2>/dev/null || true

ARCHIVO_RAIZ="$(grep -m1 '^raiz' "${PROJECT}/config.toml" | cut -d'"' -f2 || true)"

cat <<EOF

== Qué NO se ha borrado (a propósito)
  Cola y artefactos:  ${PROJECT}/data/
  Secretos:           ${HOME}/.config/transcription247/
  Notas en Drive:     ${ARCHIVO_RAIZ:-ver archivo.raiz en config.toml}

  Para volver a instalar: ${SCRIPT_DIR}/install.sh
EOF

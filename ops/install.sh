#!/usr/bin/env bash
# Install / refresh the Transcription 24/7 systemd --user units (ARCHITECTURE.md §11).
#
# Idempotent: safe to re-run after every change. It never prints the API token, never touches the
# Drive archive beyond creating the folders this project owns, and never deletes anything.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_T247="${PROJECT}/.venv/bin/t247"
VENV_PY="${PROJECT}/.venv/bin/python"
UNIT_DIR="${HOME}/.config/systemd/user"
SECRETS_DIR="${HOME}/.config/transcription247"
TOKEN_FILE="${SECRETS_DIR}/api_token"
UNITS=(t247-worker.service t247-api.service)

say() { printf '  %s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }

# --------------------------------------------------------------------------- 1. preconditions
step "Comprobando el proyecto"
[[ -x "${VENV_T247}" ]] || {
  echo "ERROR: ${VENV_T247} no existe o no es ejecutable. Ejecuta 'uv sync' primero." >&2
  exit 1
}
[[ -f "${PROJECT}/config.toml" ]] || { echo "ERROR: falta ${PROJECT}/config.toml (cópialo desde config.example.toml y edítalo)" >&2; exit 1; }
say "proyecto: ${PROJECT}"
say "entrypoint: ${VENV_T247}"

# --------------------------------------------------------------------------- 2. secrets
step "Directorio de secretos"
mkdir -p "${SECRETS_DIR}"
chmod 700 "${SECRETS_DIR}"
if [[ -f "${TOKEN_FILE}" ]]; then
  chmod 600 "${TOKEN_FILE}"
  say "token de API: ya existe (sin tocar)"
else
  ( umask 077
    if command -v openssl >/dev/null 2>&1; then
      openssl rand -hex 32 > "${TOKEN_FILE}"
    else
      "${VENV_PY}" -c 'import secrets; print(secrets.token_hex(32))' > "${TOKEN_FILE}"
    fi )
  chmod 600 "${TOKEN_FILE}"
  say "token de API: creado en ${TOKEN_FILE} (no se imprime aquí)"
  say "para leerlo:  cat ${TOKEN_FILE}"
fi

# --------------------------------------------------------------------------- 3. directories
# The Drive folders are created through paths.guard_write(), so this script can only ever create
# directories inside the tree the project owns — never the sibling Fireflies archive (§12).
step "Directorios de datos y de Drive"
"${VENV_PY}" - "${PROJECT}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from transcription247 import paths
from transcription247.config import load_config

cfg = load_config(Path(sys.argv[1]) / "config.toml")
paths.configure(cfg)

for sub in ("jobs", "tmp", "logs", "tmp/claude"):
    (cfg.project_root / cfg.paths.data_dir / sub).mkdir(parents=True, exist_ok=True)

targets = [
    Path(cfg.archivo.raiz).expanduser(),
    Path(cfg.inbox.ruta).expanduser(),
    Path(cfg.inbox.ruta).expanduser() / cfg.inbox.carpeta_procesados,
    Path(cfg.inbox.ruta).expanduser() / cfg.inbox.carpeta_error,
]
for target in targets:
    guarded = paths.guard_write(target)
    guarded.mkdir(parents=True, exist_ok=True)
    print(f"  {guarded}")
PY

# --------------------------------------------------------------------------- 4. units
step "Instalando units de usuario"
mkdir -p "${UNIT_DIR}"
for unit in "${UNITS[@]}"; do
  install -m 644 "${SCRIPT_DIR}/${unit}" "${UNIT_DIR}/${unit}"
  say "${UNIT_DIR}/${unit}"
done

systemctl --user daemon-reload
if ! systemd-analyze --user verify "${UNIT_DIR}/t247-worker.service" "${UNIT_DIR}/t247-api.service"; then
  echo "AVISO: systemd-analyze ha reportado problemas en los units (ver arriba)." >&2
fi

step "Habilitando y arrancando"
systemctl --user enable --now "${UNITS[@]}"
sleep 2
for unit in "${UNITS[@]}"; do
  say "$(systemctl --user is-enabled "${unit}" 2>&1) / $(systemctl --user is-active "${unit}" 2>&1)  ${unit}"
done

# --------------------------------------------------------------------------- 5. doctor
step "Comprobaciones de entorno (t247 doctor)"
set +e
( cd "${PROJECT}" && "${VENV_T247}" doctor )
DOCTOR_RC=$?
set -e

API_HOST="$(grep -m1 '^host' "${PROJECT}/config.toml" | cut -d'"' -f2)"
API_PORT="$(grep -m1 '^puerto' "${PROJECT}/config.toml" | tr -dc '0-9')"

cat <<EOF

== Siguientes pasos
  Estado:      systemctl --user status t247-worker t247-api
  Logs:        journalctl --user -u t247-worker -f
  Cola:        cd "${PROJECT}" && .venv/bin/t247 status
  Web/API:     http://${API_HOST}:${API_PORT}/
  Token:       cat ${TOKEN_FILE}   (no se ha impreso arriba a propósito)
  Desinstalar: ${SCRIPT_DIR}/uninstall.sh   (no borra datos ni notas de Drive)
EOF

exit "${DOCTOR_RC}"

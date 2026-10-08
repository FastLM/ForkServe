#!/usr/bin/env bash
# Control-plane P/D connector table (paper tab:connector). No GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${FORKSERVE_PYTHON:-}" && -x "${FORKSERVE_PYTHON}" ]]; then
  PYTHON="$FORKSERVE_PYTHON"
elif [[ -x "${FORKSERVE_ENV:-$HOME/envs/forkserve}/bin/python" ]]; then
  PYTHON="${FORKSERVE_ENV:-$HOME/envs/forkserve}/bin/python"
else
  PYTHON="$(command -v python3 || command -v python)"
fi
cd "$ROOT"
OUT="${OUT:-$ROOT/logs/connector/connector.json}"
mkdir -p "$(dirname "$OUT")"
exec "$PYTHON" -u -m experiments.connector_bench \
  --sessions "${SESSIONS:-10}" \
  --k "${K:-4}" \
  --trunk "${TRUNK:-256}" \
  --residual "${RESIDUAL:-32}" \
  --loops-dropped "${LOOPS_DROPPED:-1}" \
  --out "$OUT"

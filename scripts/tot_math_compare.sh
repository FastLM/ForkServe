#!/usr/bin/env bash
# Tree-of-Thoughts math comparison: vLLM APC vs ForkServe vs APP.
#
# Main run: 4-way ToT on GSM8K, SVAMP, MATH-500, AIME, AMC23, Game-of-24.
# Sweep: branching 2/6 on a smaller slice (efficiency vs ToT width).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_BIN="${FORKSERVE_ENV:-$HOME/envs/forkserve}/bin"
PYTHON="${ENV_BIN}/python"
OUT_DIR="${ROOT}/logs/tot_math"
export FORKSERVE_MODEL="${FORKSERVE_MODEL:-$HOME/models/Qwen3-14B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export FORKSERVE_TP="${FORKSERVE_TP:-2}"
export FORKSERVE_ROOT="$ROOT"
mkdir -p "$OUT_DIR"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# Drop our occupy hold (not other users' jobs) so the real bench can start.
release_own_occupy() {
  local pidfile cmd
  for pidfile in \
    "$ROOT/logs/occupy_when_free.pid" \
    "$ROOT/logs/full_quality/occupy.pid"
  do
    [[ -f "$pidfile" ]] || continue
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    [[ -d "/proc/$pid" ]] || continue
    cmd="$(ps -p "$pid" -o args= --no-headers 2>/dev/null || true)"
    if [[ "$cmd" == *occupy_gpus.py* || "$cmd" == *occupy_when_free.sh* ]]; then
      log "releasing own occupy pid=$pid"
      kill "$pid" 2>/dev/null || true
      sleep 2
    fi
  done
}

release_own_occupy

exec "$ROOT/scripts/wait_gpu_then_run.sh" -- "$PYTHON" -u "$ROOT/scripts/tot_math_compare.py"

#!/usr/bin/env bash
# Multiturn Tree-of-Thoughts: 4 agents per problem, depth 3, many math sets.
# vLLM APC vs ForkServe vs APP.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_BIN="${FORKSERVE_ENV:-$HOME/envs/forkserve}/bin"
PYTHON="${ENV_BIN}/python"
OUT_DIR="${ROOT}/logs/tot_mt"
export FORKSERVE_MODEL="${FORKSERVE_MODEL:-$HOME/models/Qwen3-4B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORKSERVE_TP="${FORKSERVE_TP:-2}"
export FORKSERVE_ROOT="$ROOT"
export TOT_MT_TURNS="${TOT_MT_TURNS:-3}"
export TOT_MT_BRANCHING="${TOT_MT_BRANCHING:-4}"
export TOT_MT_CHUNK="${TOT_MT_CHUNK:-1}"
mkdir -p "$OUT_DIR"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

release_own_occupy() {
  local pidfile cmd pid
  for pidfile in \
    "$ROOT/logs/occupy_when_free.pid" \
    "$ROOT/logs/full_quality/occupy.pid"
  do
    [[ -f "$pidfile" ]] || continue
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

exec "$ROOT/scripts/wait_gpu_then_run.sh" -- "$PYTHON" -u "$ROOT/scripts/tot_mt_compare.py"

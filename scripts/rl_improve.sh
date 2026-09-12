#!/usr/bin/env bash
# RL improve loop: occupy GPUs → bench vs vLLM → if we lose, occupy + Cursor edit.
#
# Usage:
#   ./scripts/rl_improve.sh
#   ./scripts/rl_improve.sh --max-rounds 100 --tp 2,4 --limit 4 --decode 256
#   ./scripts/rl_improve.sh --from-json logs/bench_gpu.json   # judge only
#   RL_EDIT=wait ./scripts/rl_improve.sh                     # occupy, wait for CONTINUE
#   RL_EDIT=cursor CURSOR_API_KEY=... ./scripts/rl_improve.sh
#
# During --edit wait (or cursor fallback):
#   1. GPUs stay occupied (~92% VRAM).
#   2. Read / edit from logs/rl_improve/NEXT.md in Cursor.
#   3. touch logs/rl_improve/CONTINUE
#   4. Script releases the hold, re-occupies, and re-runs the bench.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_BIN="${FORKSERVE_ENV:-$HOME/envs/forkserve}/bin"
PYTHON="${ENV_BIN}/python"
SITE="${ENV_BIN%/bin}/lib/python3.12/site-packages"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORKSERVE_MODEL="${FORKSERVE_MODEL:-$HOME/models/Qwen3-8B}"
export FORKSERVE_TP="${FORKSERVE_TP:-2,4}"
export FORKSERVE_ROOT="$ROOT"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "error: $*"; exit 1; }

[[ -x "$PYTHON" ]] || die "python not found: $PYTHON"

setup_runtime() {
  export PATH="${ENV_BIN}:${PATH}"
  export PYTHONUNBUFFERED=1
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  if [[ -d /usr/local/cuda-12.8 ]]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
  elif [[ -d /usr/local/cuda-12.9 ]]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
  fi
  local ld="" d
  for d in \
    "${CUDA_HOME:-}/lib64" \
    "${SITE}/nvidia/cuda_runtime/lib" \
    "${SITE}/nvidia/cublas/lib" \
    "${SITE}/nvidia/cudnn/lib" \
    "${SITE}/nvidia/nccl/lib" \
    "${SITE}/nvidia/nvtx/lib" \
    "${SITE}/nvidia/cuda_nvrtc/lib" \
    "${SITE}/nvidia/cuda_cupti/lib" \
    "${SITE}/nvidia/nvjitlink/lib" \
    "${SITE}/nvidia/cusparse/lib" \
    "${SITE}/nvidia/cusolver/lib" \
    "${SITE}/nvidia/cufft/lib" \
    "${SITE}/nvidia/curand/lib" \
    "${SITE}/torch/lib"
  do
    [[ -d "$d" ]] && ld="${ld}${d}:"
  done
  local cleaned="" part
  IFS=':' read -ra _ld_parts <<< "${LD_LIBRARY_PATH:-}"
  for part in "${_ld_parts[@]}"; do
    [[ "$part" == *"/nvidia/cu13/"* || "$part" == *"/cu13/lib"* ]] && continue
    [[ -n "$part" ]] && cleaned="${cleaned}${part}:"
  done
  export LD_LIBRARY_PATH="${ld}${cleaned}"
}

mkdir -p "$ROOT/logs/rl_improve"
setup_runtime
cd "$ROOT"
log "rl_improve python=$PYTHON devices=${CUDA_VISIBLE_DEVICES} edit=${RL_EDIT:-both}"
exec "$PYTHON" -u "$ROOT/scripts/rl_improve.py" "$@"

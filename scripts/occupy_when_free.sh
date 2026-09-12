#!/usr/bin/env bash
# Wait until every listed GPU is idle, then occupy them immediately.
#
# Same pattern as the box's other long-running watchers
# (poll nvidia-smi → the moment the card is free, launch a 4-GPU job):
#   while gpu_busy; do sleep 1; done
#   python occupy_gpus.py   # allocate ~92% VRAM on all visible devices
#   python -m forkserve.bench
#
# Usage:
#   nohup ./scripts/occupy_when_free.sh > logs/occupy.log 2>&1 &
#   HOLD_ONLY=1 nohup ./scripts/occupy_when_free.sh > logs/occupy.log 2>&1 &
#   POLL_SEC=1 CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/occupy_when_free.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_BIN="${FORKSERVE_ENV:-$HOME/envs/forkserve}/bin"
PYTHON="${ENV_BIN}/python"
SITE="${ENV_BIN%/bin}/lib/python3.12/site-packages"
POLL_SEC="${POLL_SEC:-1}"
MEM_FREE_MIB="${MEM_FREE_MIB:-1024}"
HOLD_FRAC="${HOLD_FRAC:-0.92}"
HOLD_SEC="${HOLD_SEC:-3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORKSERVE_MODEL="${FORKSERVE_MODEL:-$HOME/models/Qwen3-8B}"
export FORKSERVE_TP="${FORKSERVE_TP:-2,4}"
export FORKSERVE_ROOT="$ROOT"
PID_FILE="${PID_FILE:-$ROOT/logs/occupy_when_free.pid}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "error: $*"; exit 1; }

command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
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
    "${SITE}/nvidia/curand/lib"
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

mkdir -p "$ROOT/logs"
echo $$ > "$PID_FILE"
log "pid=$$  devices=${CUDA_VISIBLE_DEVICES}  poll=${POLL_SEC}s  idle<${MEM_FREE_MIB}MiB"
if [[ "${HOLD_ONLY:-0}" == "1" ]]; then
  log "will occupy ${HOLD_FRAC} of free VRAM and HOLD until killed"
else
  log "will occupy ${HOLD_FRAC} of free VRAM, then start ForkServe bench"
fi
setup_runtime
cd "$ROOT"

extra=()
if [[ "${HOLD_ONLY:-0}" == "1" ]]; then
  extra+=(--hold-only)
fi
if [[ $# -gt 0 ]]; then
  extra+=(-- "$@")
fi

exec "$PYTHON" -u "$ROOT/scripts/occupy_gpus.py" \
  --poll-sec "$POLL_SEC" \
  --idle-mib "$MEM_FREE_MIB" \
  --fraction "$HOLD_FRAC" \
  --hold-sec "$HOLD_SEC" \
  "${extra[@]}"

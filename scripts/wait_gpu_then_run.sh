#!/usr/bin/env bash
# Wait until the GPU jobs that are running *right now* finish, then run the
# ForkServe vs vLLM multi-GPU comparison (tensor parallel 2 and/or 4).
#
# Usage:
#   ./scripts/wait_gpu_then_run.sh
#   CUDA_VISIBLE_DEVICES=0,1 ./scripts/wait_gpu_then_run.sh
#   ./scripts/wait_gpu_then_run.sh -- python -m forkserve.bench --tp 2
#
# Env:
#   POLL_SEC / MEM_FREE_MIB   wait loop
#   CUDA_VISIBLE_DEVICES      GPUs to claim (default: 0,1,2,3)
#   FORKSERVE_MODEL / FORKSERVE_TP / FORKSERVE_GPU_UTIL
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_BIN="${FORKSERVE_ENV:-$HOME/envs/forkserve}/bin"
PYTHON="${ENV_BIN}/python"
SITE="${ENV_BIN%/bin}/lib/python3.12/site-packages"
POLL_SEC="${POLL_SEC:-30}"
MEM_FREE_MIB="${MEM_FREE_MIB:-1024}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORKSERVE_MODEL="${FORKSERVE_MODEL:-$HOME/models/Qwen3-8B}"
export FORKSERVE_TP="${FORKSERVE_TP:-2,4}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

die() { log "error: $*"; exit 1; }

command -v nvidia-smi >/dev/null || die "nvidia-smi not found"

gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' \
    | grep -E '^[0-9]+$' \
    || true
}

# Other users' GPU jobs are not signalable; /proc is the reliable liveness check.
pid_alive() { [[ -d "/proc/$1" ]]; }

describe_pids() {
  local pid
  for pid in "$@"; do
    if pid_alive "$pid"; then
      ps -p "$pid" -o user=,pid=,etime=,cmd= --no-headers 2>/dev/null \
        || echo "  pid=$pid (gone during ps)"
    else
      echo "  pid=$pid (exited)"
    fi
  done
}

snapshot_gpu() {
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
}

target_gpus() {
  local IFS=','
  # shellcheck disable=SC2086
  echo ${CUDA_VISIBLE_DEVICES}
}

gpus_idle() {
  local idx used
  for idx in $(target_gpus); do
    used="$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    [[ "$used" =~ ^[0-9]+$ ]] || return 1
    if (( used > MEM_FREE_MIB )); then
      return 1
    fi
  done
  return 0
}

setup_runtime() {
  export PATH="${ENV_BIN}:${PATH}"
  export PYTHONUNBUFFERED=1
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  local ld="" d
  for d in "${SITE}"/nvidia/*/lib "${SITE}"/nvidia/cu13/lib; do
    [[ -d "$d" ]] && ld="${d}:${ld}"
  done
  export LD_LIBRARY_PATH="${ld}${LD_LIBRARY_PATH:-}"
}

cd "$ROOT"

mapfile -t WAIT_PIDS < <(gpu_pids)
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  idle threshold=${MEM_FREE_MIB} MiB  poll=${POLL_SEC}s"
log "current GPU state:"
snapshot_gpu | while IFS= read -r line; do log "  $line"; done

if ((${#WAIT_PIDS[@]} == 0)); then
  log "no compute apps on GPU right now"
else
  log "waiting for ${#WAIT_PIDS[@]} GPU process(es): ${WAIT_PIDS[*]}"
  describe_pids "${WAIT_PIDS[@]}" | while IFS= read -r line; do log "  $line"; done
fi

while true; do
  still=()
  for pid in "${WAIT_PIDS[@]+"${WAIT_PIDS[@]}"}"; do
    [[ -n "$pid" ]] || continue
    if pid_alive "$pid"; then
      still+=("$pid")
    fi
  done

  if ((${#still[@]} > 0)); then
    log "still running (${#still[@]}): ${still[*]}"
    snapshot_gpu | while IFS= read -r line; do log "  $line"; done
    sleep "$POLL_SEC"
    continue
  fi

  if ! gpus_idle; then
    log "snapshot jobs are gone, but GPU ${CUDA_VISIBLE_DEVICES} is not idle yet (need < ${MEM_FREE_MIB} MiB)"
    snapshot_gpu | while IFS= read -r line; do log "  $line"; done
    leftover="$(gpu_pids | tr '\n' ' ')"
    [[ -n "$leftover" ]] && log "  leftover compute pids: $leftover"
    sleep "$POLL_SEC"
    continue
  fi

  break
done

log "GPU ${CUDA_VISIBLE_DEVICES} is free — starting ForkServe vs vLLM bench"
snapshot_gpu | while IFS= read -r line; do log "  $line"; done

[[ -x "$PYTHON" ]] || die "python not found: $PYTHON"
setup_runtime

if [[ $# -gt 0 ]]; then
  log "exec: $*"
  exec "$@"
fi

log "exec: $PYTHON -m forkserve.bench --tp ${FORKSERVE_TP} --model ${FORKSERVE_MODEL}"
exec "$PYTHON" -m forkserve.bench \
  --tp "${FORKSERVE_TP}" \
  --model "${FORKSERVE_MODEL}" \
  --out "${ROOT}/logs/bench_gpu.json"

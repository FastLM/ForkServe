#!/usr/bin/env bash
# APC vs ForkServe on multiple math ToT workloads using in-tree vllm_fs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_BIN="${FORKSERVE_ENV:-$HOME/envs/forkserve}/bin"
PYTHON="${ENV_BIN}/python"
SITE="${ENV_BIN%/bin}/lib/python3.12/site-packages"
# env may be 3.11
if [[ ! -d "$SITE" ]]; then
  SITE="${ENV_BIN%/bin}/lib/python3.11/site-packages"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORKSERVE_MODEL="${FORKSERVE_MODEL:-$HOME/models/Qwen3-8B}"
export FORKSERVE_BENCHMARKS="${FORKSERVE_BENCHMARKS:-$HOME/benchmarks}"
export FORKSERVE_ROOT="$ROOT"
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
if [[ -d /usr/local/cuda-12.8 ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
elif [[ -d /usr/local/cuda-12.9 ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
fi
ld=""
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
cleaned=""
IFS=':' read -ra _ld_parts <<< "${LD_LIBRARY_PATH:-}"
for part in "${_ld_parts[@]}"; do
  [[ "$part" == *"/nvidia/cu13/"* || "$part" == *"/cu13/lib"* ]] && continue
  [[ -n "$part" ]] && cleaned="${cleaned}${part}:"
done
export LD_LIBRARY_PATH="${ld}${cleaned}"
export PATH="${ENV_BIN}:${PATH}"

OUT="${OUT:-$ROOT/logs/math_apc_vs_fs/eval.json}"
mkdir -p "$(dirname "$OUT")" "$ROOT/logs/math_apc_vs_fs"
PROGRESS="${PROGRESS:-$ROOT/logs/math_apc_vs_fs/progress.log}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

wait_model() {
  local dest="$FORKSERVE_MODEL"
  log "waiting for weights in $dest"
  while true; do
    if [[ -f "$dest/config.json" ]] \
      && ls "$dest"/model-*.safetensors >/dev/null 2>&1 \
      && ! ls "$dest"/*.incomplete >/dev/null 2>&1; then
      local n
      n="$(ls -1 "$dest"/model-*.safetensors | wc -l)"
      log "model ready: $n shards"
      return 0
    fi
    ls -lh "$dest"/*.safetensors* 2>/dev/null | awk '{print $5, $9}' | while read -r line; do log "  $line"; done
    sleep 20
  done
}

log "holding GPUs until the 8B weights finish downloading"
HOLD_ONLY=1 "$PYTHON" -u "$ROOT/scripts/occupy_gpus.py" \
  --poll-sec 1 --idle-mib 1024 --fraction 0.90 --hold-only \
  >"$ROOT/logs/math_apc_vs_fs/occupy.log" 2>&1 &
HOLD_PID=$!
cleanup_hold() {
  if kill -0 "$HOLD_PID" 2>/dev/null; then
    log "releasing occupy pid=$HOLD_PID"
    kill "$HOLD_PID" 2>/dev/null || true
    wait "$HOLD_PID" 2>/dev/null || true
  fi
}
trap cleanup_hold EXIT

wait_model
cleanup_hold
trap - EXIT
sleep 1

cd "$ROOT"
log "exec math APC vs ForkServe"
exec "$PYTHON" -u -m forkserve.bench \
  --model "$FORKSERVE_MODEL" \
  --systems vllm_apc,forkserve \
  --tp "${FORKSERVE_TP:-2}" \
  --workloads gsm8k,svamp,gsmhard,math500,aime,amc23,game24 \
  --limit "${LIMIT:-80}" \
  --chunk "${CHUNK:-8}" \
  --decode "${DECODE:-512}" \
  --gsm8k-decode "${GSM8K_DECODE:-512}" \
  --branching "${BRANCHING:-4}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-batched-tokens "${MAX_BATCHED:-2048}" \
  --gpu-util "${FORKSERVE_GPU_UTIL:-0.90}" \
  --progress-log "$PROGRESS" \
  --out "$OUT"

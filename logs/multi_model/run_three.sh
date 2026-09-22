#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/net/liudong_workspace/ForkServe
ENV=/home/net/liudong_workspace/envs/forkserve
PYSITE="$ENV/lib/python3.11/site-packages"
LOGDIR="$ROOT/logs/multi_model"

export PATH="$ENV/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

NVIDIA_LIBS="$(find "$PYSITE/nvidia" -type d \( -name lib -o -name lib64 \) 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

run_one() {
  local name="$1" model="$2" style="$3" outdir="$4"
  mkdir -p "$outdir"
  if [[ -f "$outdir/eval.json" ]]; then
    echo "$(date -Is) skip $name — eval.json exists"
    return 0
  fi
  echo "$(date -Is) ==> bench $name model=$model style=$style"
  cd "$ROOT"
  PYTHONPATH=. FORKSERVE_CHAT_STYLE="$style" "$ENV/bin/python" -u -m forkserve.bench \
    --model "$model" \
    --systems vllm_recompute,vllm_apc,forkserve,forkserve_plus \
    --tp 2,4 \
    --workloads gsm8k,game24,humaneval \
    --limit 40 \
    --chunk 8 \
    --decode 512 \
    --gsm8k-decode 512 \
    --branching 4 \
    --sessions 2 \
    --trunk-tokens 512 \
    --max-model-len 8192 \
    --max-batched-tokens 2048 \
    --gpu-util 0.88 \
    --progress-log "$outdir/progress.log" \
    --out "$outdir/eval.json"
  echo "$(date -Is) <== finished $name"
}

ready() {
  local log="$1"
  grep -q "^done " "$log" 2>/dev/null
}

# Prefer the first model that finishes downloading.
# mistral / llama usually land before Qwen3-14B.
names=(mistral7b llama8b qwen3_14b)
declare -A MODEL STYLE DL OUT
MODEL[mistral7b]=/data/liudong_workspace/models/Mistral-7B-Instruct-v0.3
STYLE[mistral7b]=mistral
DL[mistral7b]=$LOGDIR/dl_mistral7b.log
OUT[mistral7b]=$LOGDIR/mistral7b

MODEL[llama8b]=/data/liudong_workspace/models/Llama-3-8B-Instruct
STYLE[llama8b]=llama
DL[llama8b]=$LOGDIR/dl_llama8b.log
OUT[llama8b]=$LOGDIR/llama8b

MODEL[qwen3_14b]=/home/net/liudong_workspace/models/Qwen3-14B
STYLE[qwen3_14b]=qwen
DL[qwen3_14b]=$LOGDIR/dl_qwen3_14b.log
OUT[qwen3_14b]=$LOGDIR/qwen3_14b

echo "$(date -Is) orchestrator waiting for downloads"
pending=3
while (( pending > 0 )); do
  pending=0
  for name in "${names[@]}"; do
    if [[ -f "${OUT[$name]}/eval.json" ]]; then
      continue
    fi
    if ready "${DL[$name]}"; then
      run_one "$name" "${MODEL[$name]}" "${STYLE[$name]}" "${OUT[$name]}"
    else
      pending=$((pending + 1))
    fi
  done
  if (( pending > 0 )); then
    echo "$(date -Is) still waiting for $pending download(s)"
    sleep 30
  fi
done
echo "$(date -Is) all three benches finished"

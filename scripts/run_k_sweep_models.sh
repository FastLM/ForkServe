#!/usr/bin/env bash
# k=4,8,16,32 sweeps across models and experiment suites.
#
# Suites (EXP):
#   plugin   - ESC/SR/DPTS base vs +tau_pre=0.15   (paper e2e / host plug-in)
#   prefill  - ESC prefill-threshold sweep
#   thresh   - score-bar sweep at 0.15/0.45
#   grow     - MATH-500 probe admission (ignores KS; uses built-in k grid)
#   wide     - full method x policy at each k
#   connector- control-plane P/D table (no GPU)
#
# Examples:
#   MODELS="Qwen3-8B,Qwen3-4B" EXP=plugin KS=4,8,16,32 \
#     MODEL_ROOT=/home/dliu/models bash scripts/run_k_sweep_models.sh
#
#   EXP=connector bash scripts/run_k_sweep_models.sh
#   EXP=plugin,prefill,thresh MODELS=Qwen3-8B KS=4,8,16,32 bash scripts/run_k_sweep_models.sh
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

MODEL_ROOT="${MODEL_ROOT:-${FORKSERVE_MODEL_ROOT:-/home/dliu/models}}"
MODELS="${MODELS:-Qwen3-8B}"
EXP="${EXP:-plugin}"
KS="${KS:-4,8,16,32}"
TP="${TP:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/logs/k_sweep}"
PREFILL_THRESHOLDS="${PREFILL_THRESHOLDS:-0.0,0.15,0.45}"
THRESHOLDS="${THRESHOLDS:-0.15,0.45}"
GPU_UTIL="${GPU_UTIL:-0.90}"

mkdir -p "$OUT_ROOT"

run_gpu() {
  local model_name="$1"
  local exp="$2"
  local model_path="$MODEL_ROOT/$model_name"
  if [[ ! -e "$model_path" && -n "${FORKSERVE_MODEL:-}" && "$MODELS" == "$model_name" ]]; then
    model_path="$FORKSERVE_MODEL"
  fi
  if [[ ! -e "$model_path" ]]; then
    echo "WARN: missing model $model_path — skip" >&2
    return 0
  fi
  local out_dir="$OUT_ROOT/${model_name//\//_}/$exp"
  mkdir -p "$out_dir"
  local out="$out_dir/all.json"
  local args=(
    -m experiments.prune_ablation_gpu
    --model "$model_path"
    --tp "$TP"
    --ks "$KS"
    --out "$out"
    --gpu-util "$GPU_UTIL"
  )
  case "$exp" in
    plugin)  args+=(--plugin) ;;
    prefill) args+=(--prefill --prefill-thresholds "$PREFILL_THRESHOLDS") ;;
    thresh)  args+=(--thresh --thresholds "$THRESHOLDS") ;;
    wide)    args+=(--wide) ;;
    grow)
      # grow has its own k grid inside the builder; --ks still stamped when supported
      args=(
        -m experiments.prune_ablation_gpu
        --model "$model_path"
        --tp "$TP"
        --grow
        --out "$out"
        --gpu-util "$GPU_UTIL"
      )
      ;;
    *)
      echo "unknown EXP=$exp" >&2
      return 2
      ;;
  esac
  if [[ -f "$out" && "${FORCE:-0}" != "1" ]]; then
    echo "skip $model_name/$exp (exists $out)"
    return 0
  fi
  echo "=== $model_name  EXP=$exp  KS=$KS ==="
  "$PYTHON" -u "${args[@]}"
}

IFS=',' read -r -a EXP_ARR <<<"$EXP"
IFS=',' read -r -a MODEL_ARR <<<"$MODELS"

for exp in "${EXP_ARR[@]}"; do
  exp="$(echo "$exp" | xargs)"
  if [[ "$exp" == "connector" ]]; then
    OUT="$OUT_ROOT/connector/connector.json" \
      bash "$ROOT/scripts/run_connector_bench.sh"
    # also sweep k for connector accounting
    for k in ${KS//,/ }; do
      "$PYTHON" -u -m experiments.connector_bench \
        --k "$k" \
        --out "$OUT_ROOT/connector/connector_k${k}.json"
    done
    continue
  fi
  for model in "${MODEL_ARR[@]}"; do
    model="$(echo "$model" | xargs)"
    run_gpu "$model" "$exp"
  done
done

echo "all done -> $OUT_ROOT"

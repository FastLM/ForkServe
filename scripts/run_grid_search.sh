#!/usr/bin/env bash
# Full parameter grid for PD-Prune (tau_pre x tau_dec x t0 x alpha x k).
# Default grid is large (~4*3*3*3*4 = 432 cells with method=esc).
# Start with --list, then run a subset or shard by CELL_START/CELL_END.
#
# Examples:
#   MODEL=/path/Qwen3-8B bash scripts/run_grid_search.sh --list
#   MODEL=/path/Qwen3-8B KS=4,8,16,32 bash scripts/run_grid_search.sh
#   MODEL=/path/Qwen3-8B CELL_START=0 CELL_END=20 bash scripts/run_grid_search.sh
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

MODEL="${MODEL:-${FORKSERVE_MODEL:-}}"
TP="${TP:-1}"
OUT_DIR="${OUT_DIR:-$ROOT/logs/grid_search}"
TAU_PRE="${TAU_PRE:-0.05,0.15,0.30,0.45}"
TAU_DEC="${TAU_DEC:-0.30,0.45,0.60}"
T0="${T0:-64,128,256}"
ALPHA="${ALPHA:-0.10,0.20,0.30}"
KS="${KS:-4,8,16,32}"
METHODS="${METHODS:-esc}"
N="${N:-16}"
BUDGET="${BUDGET:-512}"
WORKLOAD="${WORKLOAD:-gsm8k}"
CELL_START="${CELL_START:-0}"
CELL_END="${CELL_END:--1}"   # -1 = all

mkdir -p "$OUT_DIR"
PLAN="$OUT_DIR/plan.json"

COMMON=(
  -m experiments.grid_search_params
  --tau-pre "$TAU_PRE"
  --tau-dec "$TAU_DEC"
  --t0 "$T0"
  --alpha "$ALPHA"
  --ks "$KS"
  --methods "$METHODS"
  --n "$N"
  --budget "$BUDGET"
  --workload "$WORKLOAD"
)

if [[ "${1:-}" == "--list" ]]; then
  exec "$PYTHON" -u "${COMMON[@]}" --list
fi

"$PYTHON" -u "${COMMON[@]}" --plan "$PLAN"
N_CELLS="$("$PYTHON" -c "import json; print(json.load(open('$PLAN'))['n_cells'])")"
END="$CELL_END"
if [[ "$END" -lt 0 ]]; then END=$((N_CELLS - 1)); fi
echo "grid: $N_CELLS cells; running [$CELL_START .. $END]"

if [[ -z "$MODEL" ]]; then
  echo "ERROR: set MODEL=/path/to/weights" >&2
  exit 2
fi

for ((i=CELL_START; i<=END; i++)); do
  OUT="$OUT_DIR/cell_${i}.json"
  if [[ -f "$OUT" && "${FORCE:-0}" != "1" ]]; then
    echo "skip cell $i (exists)"
    continue
  fi
  "$PYTHON" -u "${COMMON[@]}" \
    --run-cell "$i" \
    --model "$MODEL" \
    --tp "$TP" \
    --python "$PYTHON" \
    --out "$OUT"
done
echo "done -> $OUT_DIR"

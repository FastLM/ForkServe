#!/usr/bin/env bash
# Unified driver for the new paper experiments (GPU + control-plane).
#
# Stages (comma list, STAGES=...):
#   connector  - P/D connector table (CPU, no GPU)           → logs/paper_new/connector/
#   plugin     - ESC/SR/DPTS ± τ_pre=0.15 at k∈KS            → logs/paper_new/k_sweep/
#   prefill    - ESC prefill-bar sweep                       → logs/paper_new/k_sweep/
#   thresh     - score-bar 0.15/0.45 × hosts                 → logs/paper_new/k_sweep/
#   grow       - MATH-500 probe admission                    → logs/paper_new/k_sweep/
#   wide       - base/draft/app × hosts                      → logs/paper_new/k_sweep/
#   grid       - (τ_pre,τ_dec,t0,α,k) grid                   → logs/paper_new/grid_search/
#   all        - connector,plugin,prefill,thresh,grow,grid
#
# Quick start (one model, paper defaults):
#   MODEL_ROOT=/home/dliu/models MODELS=Qwen3-8B \
#     bash scripts/run_paper_new_exps.sh
#
# Multi-model + full k set:
#   MODEL_ROOT=/home/dliu/models \
#   MODELS="Qwen3-8B,Qwen3-4B,Llama-3-8B-Instruct" \
#   KS=4,8,16,32 TP=1 \
#   STAGES=connector,plugin,prefill,thresh,grow,grid \
#   bash scripts/run_paper_new_exps.sh
#
# Reduced grid (recommended first pass; FULL_GRID=1 for 432 cells):
#   FULL_GRID=0  → τ_pre∈{0.15,0.45}, τ_dec=0.45, t0=128, α=0.20, k∈KS
#   FULL_GRID=1  → default cartesian product in run_grid_search.sh
#
# Resume / skip:
#   FORCE=0 (default) skips outputs that already exist
#   FORCE=1 re-runs everything
#
# List only (no GPU):
#   bash scripts/run_paper_new_exps.sh --list
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${FORKSERVE_PYTHON:-}" && -x "${FORKSERVE_PYTHON}" ]]; then
  PYTHON="$FORKSERVE_PYTHON"
elif [[ -x "${FORKSERVE_ENV:-$HOME/envs/forkserve}/bin/python" ]]; then
  PYTHON="${FORKSERVE_ENV:-$HOME/envs/forkserve}/bin/python"
else
  PYTHON="$(command -v python3 || command -v python)"
fi
export FORKSERVE_PYTHON="$PYTHON"

MODEL_ROOT="${MODEL_ROOT:-${FORKSERVE_MODEL_ROOT:-/home/dliu/models}}"
MODELS="${MODELS:-Qwen3-8B}"
KS="${KS:-4,8,16,32}"
TP="${TP:-1}"
GPU_UTIL="${GPU_UTIL:-0.90}"
STAGES="${STAGES:-all}"
FULL_GRID="${FULL_GRID:-0}"
FORCE="${FORCE:-0}"
OUT_ROOT="${OUT_ROOT:-$ROOT/logs/paper_new}"
PREFILL_THRESHOLDS="${PREFILL_THRESHOLDS:-0.0,0.15,0.45}"
THRESHOLDS="${THRESHOLDS:-0.15,0.45}"
METHODS="${METHODS:-esc}"          # grid host(s)
N="${N:-16}"
BUDGET="${BUDGET:-512}"
WORKLOAD="${WORKLOAD:-gsm8k}"
CELL_START="${CELL_START:-0}"
CELL_END="${CELL_END:--1}"

# First model path → MODEL for grid stage
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf '%s' "$s"; }
IFS=',' read -r -a MODEL_ARR <<<"$MODELS"
PRIMARY_MODEL_NAME="$(trim "${MODEL_ARR[0]}")"
PRIMARY_MODEL_PATH="${MODEL:-$MODEL_ROOT/$PRIMARY_MODEL_NAME}"
if [[ ! -e "$PRIMARY_MODEL_PATH" && -n "${FORKSERVE_MODEL:-}" ]]; then
  PRIMARY_MODEL_PATH="$FORKSERVE_MODEL"
fi

mkdir -p "$OUT_ROOT"
MASTER_LOG="$OUT_ROOT/run_$(date '+%Y%m%d-%H%M%S').log"
MANIFEST="$OUT_ROOT/manifest.json"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"; }
die() { log "ERROR: $*"; exit 1; }

resolve_stages() {
  local s="$1"
  if [[ "$s" == "all" ]]; then
    echo "connector,plugin,prefill,thresh,grow,grid"
  else
    echo "$s"
  fi
}

STAGES="$(resolve_stages "$STAGES")"

if [[ "${1:-}" == "--list" ]]; then
  echo "STAGES=$STAGES"
  echo "MODELS=$MODELS"
  echo "MODEL_ROOT=$MODEL_ROOT"
  echo "PRIMARY_MODEL_PATH=$PRIMARY_MODEL_PATH"
  echo "KS=$KS  TP=$TP  FULL_GRID=$FULL_GRID  FORCE=$FORCE"
  echo "OUT_ROOT=$OUT_ROOT"
  echo
  echo "--- grid cells ---"
  if [[ "$FULL_GRID" == "1" ]]; then
    MODEL="$PRIMARY_MODEL_PATH" KS="$KS" METHODS="$METHODS" \
      bash "$ROOT/scripts/run_grid_search.sh" --list
  else
    MODEL="$PRIMARY_MODEL_PATH" \
    TAU_PRE="${TAU_PRE:-0.15,0.45}" \
    TAU_DEC="${TAU_DEC:-0.45}" \
    T0="${T0:-128}" \
    ALPHA="${ALPHA:-0.20}" \
    KS="$KS" METHODS="$METHODS" \
      bash "$ROOT/scripts/run_grid_search.sh" --list
  fi
  echo
  echo "--- k_sweep job counts (per model × stage) ---"
  for stage in plugin prefill thresh grow wide; do
    echo "  $stage: KS=$KS via scripts/run_k_sweep_models.sh EXP=$stage"
  done
  exit 0
fi

log "==== paper new experiments ===="
log "STAGES=$STAGES"
log "MODELS=$MODELS  KS=$KS  TP=$TP  FULL_GRID=$FULL_GRID"
log "PRIMARY_MODEL_PATH=$PRIMARY_MODEL_PATH"
log "OUT_ROOT=$OUT_ROOT"
log "master log: $MASTER_LOG"

# Write a small JSON manifest up front
"$PYTHON" - <<PY | tee -a "$MASTER_LOG"
import json, os, time
from pathlib import Path
m = {
  "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
  "stages": os.environ.get("STAGES", ""),
  "models": os.environ.get("MODELS", ""),
  "model_root": os.environ.get("MODEL_ROOT", ""),
  "primary_model": os.environ.get("PRIMARY_MODEL_PATH", ""),
  "ks": os.environ.get("KS", ""),
  "tp": os.environ.get("TP", ""),
  "full_grid": os.environ.get("FULL_GRID", ""),
  "out_root": os.environ.get("OUT_ROOT", ""),
}
# env not automatically visible; rebuild
m.update({
  "stages": "$STAGES",
  "models": "$MODELS",
  "model_root": "$MODEL_ROOT",
  "primary_model": "$PRIMARY_MODEL_PATH",
  "ks": "$KS",
  "tp": "$TP",
  "full_grid": "$FULL_GRID",
  "out_root": "$OUT_ROOT",
  "force": "$FORCE",
})
Path("$MANIFEST").write_text(json.dumps(m, indent=2))
print("manifest -> $MANIFEST")
PY

run_connector() {
  log "[stage] connector (CPU)"
  OUT_DIR="$OUT_ROOT/connector"
  mkdir -p "$OUT_DIR"
  OUT="$OUT_DIR/connector.json" \
    bash "$ROOT/scripts/run_connector_bench.sh" 2>&1 | tee -a "$MASTER_LOG"
  # also per-k tables for the paper k sweep
  IFS=',' read -r -a KARR <<<"$KS"
  for k in "${KARR[@]}"; do
    k="$(trim "$k")"
    "$PYTHON" -u -m experiments.connector_bench \
      --k "$k" \
      --out "$OUT_DIR/connector_k${k}.json" 2>&1 | tee -a "$MASTER_LOG"
  done
}

run_ksweep_stage() {
  local stage="$1"
  log "[stage] k_sweep EXP=$stage  MODELS=$MODELS  KS=$KS"
  MODEL_ROOT="$MODEL_ROOT" \
  MODELS="$MODELS" \
  EXP="$stage" \
  KS="$KS" \
  TP="$TP" \
  GPU_UTIL="$GPU_UTIL" \
  PREFILL_THRESHOLDS="$PREFILL_THRESHOLDS" \
  THRESHOLDS="$THRESHOLDS" \
  FORCE="$FORCE" \
  OUT_ROOT="$OUT_ROOT/k_sweep" \
    bash "$ROOT/scripts/run_k_sweep_models.sh" 2>&1 | tee -a "$MASTER_LOG"
}

run_grid() {
  log "[stage] grid  FULL_GRID=$FULL_GRID  model=$PRIMARY_MODEL_PATH"
  [[ -e "$PRIMARY_MODEL_PATH" ]] || die "grid needs a model at PRIMARY_MODEL_PATH=$PRIMARY_MODEL_PATH"

  local tau_pre tau_dec t0 alpha
  if [[ "$FULL_GRID" == "1" ]]; then
    tau_pre="${TAU_PRE:-0.05,0.15,0.30,0.45}"
    tau_dec="${TAU_DEC:-0.30,0.45,0.60}"
    t0="${T0:-64,128,256}"
    alpha="${ALPHA:-0.10,0.20,0.30}"
  else
    # paper-adjacent reduced grid (fast enough for one overnight)
    tau_pre="${TAU_PRE:-0.15,0.45}"
    tau_dec="${TAU_DEC:-0.45}"
    t0="${T0:-128}"
    alpha="${ALPHA:-0.20}"
  fi

  MODEL="$PRIMARY_MODEL_PATH" \
  TP="$TP" \
  OUT_DIR="$OUT_ROOT/grid_search" \
  TAU_PRE="$tau_pre" \
  TAU_DEC="$tau_dec" \
  T0="$t0" \
  ALPHA="$alpha" \
  KS="$KS" \
  METHODS="$METHODS" \
  N="$N" \
  BUDGET="$BUDGET" \
  WORKLOAD="$WORKLOAD" \
  CELL_START="$CELL_START" \
  CELL_END="$CELL_END" \
  FORCE="$FORCE" \
    bash "$ROOT/scripts/run_grid_search.sh" 2>&1 | tee -a "$MASTER_LOG"
}

IFS=',' read -r -a STAGE_ARR <<<"$STAGES"
for stage in "${STAGE_ARR[@]}"; do
  stage="$(trim "$stage")"
  [[ -n "$stage" ]] || continue
  case "$stage" in
    connector) run_connector ;;
    plugin|prefill|thresh|grow|wide) run_ksweep_stage "$stage" ;;
    grid) run_grid ;;
    *) die "unknown stage: $stage (use connector|plugin|prefill|thresh|grow|wide|grid|all)" ;;
  esac
done

# Summarize what landed
log "==== finished; summarizing outputs ===="
OUT_ROOT="$OUT_ROOT" "$PYTHON" - <<'PY' 2>&1 | tee -a "$MASTER_LOG"
import json, os, time
from pathlib import Path
root = Path(os.environ["OUT_ROOT"])
files = sorted(p for p in root.rglob("*.json") if p.name != "manifest.json")
print(f"json artifacts: {len(files)}")
for p in files:
    rel = p.relative_to(root)
    try:
        sz = p.stat().st_size
    except OSError:
        sz = -1
    print(f"  {rel}  ({sz} bytes)")
man = root / "manifest.json"
if man.exists():
    m = json.loads(man.read_text())
    m["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    m["n_json"] = len(files)
    man.write_text(json.dumps(m, indent=2))
print(f"manifest -> {man}")
PY

log "done. logs: $OUT_ROOT"
log "master: $MASTER_LOG"

#!/usr/bin/env bash
# Mock ForkServe+ evaluation: fan-out split, prune, concurrency bound.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${FORKSERVE_ENV:-$HOME/envs/forkserve}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON:-python3}"
fi
export PYTHONUNBUFFERED=1
cd "$ROOT"
OUT="${OUT:-$ROOT/logs/eval_plus/eval_plus.json}"
mkdir -p "$(dirname "$OUT")"
exec "$PYTHON" -u -c "
from pathlib import Path
from forkserve.eval_plus import run_suite
r = run_suite(Path('$(dirname "$OUT")'))
print('wrote', r.get('wrote'))
print('micro_plus pruned', r['micro_plus']['pruned_branches'],
      'prefill_ms', round(r['micro_plus']['prefill_ms'], 3),
      'abort_mark_ms', round(r['micro_plus']['abort_mark_ms'], 3))
print('peak tok/s APC', round(r['peak_tok_s']['apc'], 1),
      'FS+', round(r['peak_tok_s']['forkserve_plus'], 1),
      'gain', round(100*r['tok_s_gain'], 1), '%')
print('max QPS @ P99 TTFT<=1s APC', r['max_qps_p99_1s']['apc'],
      'FS+', r['max_qps_p99_1s']['forkserve_plus'])
"

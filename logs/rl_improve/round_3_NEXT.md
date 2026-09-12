# ForkServe RL improve — round 3

Verdict: **FAIL** — efficiency did not beat vLLM
Bench JSON: `/home/dliu/ForkServe/logs/rl_improve_round_3.json`

You are editing the ForkServe repo on this machine. GPUs are occupied
by `occupy_gpus.py` so another user cannot steal them. Do **not** start
another GPU bench yourself.

## Goal
Make ForkServe more efficient than vLLM (lower peak KV than
`vllm_recompute`, not worse than `vllm_apc`) **and** keep latency
within the configured slack of the vLLM baseline (APC if present).

Focus on serving path: CoW / two-class scheduler / speculative prefill
(`forkserve/engine/`, `forkserve/bench.py` worker). Keep unit tests green.
Do not rewrite occupy scripts unless required.

## Pair results
- tp=2 game24: ok — lat 200.1 vs 224.9 (-11.0%); peak_kv 818 vs recompute 2228 / apc 818 (efficiency_beats=True, perf_drop=False)
- tp=2 gsm8k: ok — lat 224.6 vs 235.6 (-4.7%); peak_kv 815 vs recompute 2468 / apc 815 (efficiency_beats=True, perf_drop=False)
- tp=2 humaneval: NEED FIX — lat 195.9 vs 200.0 (-2.0%); peak_kv 1807 vs recompute 1807 / apc 1807 (efficiency_beats=False, perf_drop=False)
- tp=4 game24: ok — lat 142.1 vs 159.2 (-10.7%); peak_kv 818 vs recompute 2228 / apc 818 (efficiency_beats=True, perf_drop=False)
- tp=4 gsm8k: ok — lat 163.6 vs 174.5 (-6.2%); peak_kv 815 vs recompute 2468 / apc 815 (efficiency_beats=True, perf_drop=False)
- tp=4 humaneval: NEED FIX — lat 129.7 vs 140.1 (-7.4%); peak_kv 1807 vs recompute 1807 / apc 1807 (efficiency_beats=False, perf_drop=False)

After you finish editing, if this file was written for `--edit wait`,
create `/home/dliu/ForkServe/logs/rl_improve/CONTINUE` so the loop can re-occupy and re-bench.

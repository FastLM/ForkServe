# ForkServe RL improve — round 0

Verdict: **FAIL** — efficiency did not beat vLLM; clear latency drop vs vLLM
Bench JSON: `/home/dliu/ForkServe/logs/bench_gpu.json`

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
- tp=2 game24: NEED FIX — lat 899.5 vs 682.7 (+31.8%); peak_kv 206 vs recompute 563 / apc 206 (efficiency_beats=True, perf_drop=True)
- tp=2 gsm8k: NEED FIX — lat 914.3 vs 709.6 (+28.8%); peak_kv 223 vs recompute 694 / apc 223 (efficiency_beats=True, perf_drop=True)
- tp=2 humaneval: NEED FIX — lat 207.1 vs 157.8 (+31.2%); peak_kv 626 vs recompute 588 / apc 588 (efficiency_beats=False, perf_drop=True)
- tp=4 game24: NEED FIX — lat 652.9 vs 461.5 (+41.5%); peak_kv 206 vs recompute 563 / apc 206 (efficiency_beats=True, perf_drop=True)
- tp=4 gsm8k: NEED FIX — lat 683.8 vs 481.7 (+42.0%); peak_kv 223 vs recompute 694 / apc 223 (efficiency_beats=True, perf_drop=True)
- tp=4 humaneval: NEED FIX — lat 152.7 vs 114.2 (+33.7%); peak_kv 626 vs recompute 588 / apc 588 (efficiency_beats=False, perf_drop=True)

After you finish editing, if this file was written for `--edit wait`,
create `/data/dliu/ForkServe/logs/rl_improve/CONTINUE` so the loop can re-occupy and re-bench.

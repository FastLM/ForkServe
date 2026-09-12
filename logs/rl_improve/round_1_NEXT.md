# ForkServe RL improve — round 1

Verdict: **FAIL** — efficiency did not beat vLLM; clear latency drop vs vLLM
Bench JSON: `/home/dliu/ForkServe/logs/rl_improve_round_1.json`

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
- tp=2 game24: ok — lat 239.8 vs 223.0 (+7.6%); peak_kv 818 vs recompute 2228 / apc 818 (efficiency_beats=True, perf_drop=False)
- tp=2 gsm8k: NEED FIX — lat 270.2 vs 243.0 (+11.2%); peak_kv 815 vs recompute 2468 / apc 815 (efficiency_beats=True, perf_drop=True)
- tp=2 humaneval: NEED FIX — lat 219.7 vs 210.8 (+4.2%); peak_kv 1895 vs recompute 1807 / apc 1807 (efficiency_beats=False, perf_drop=False)
- tp=4 game24: NEED FIX — lat 175.1 vs 155.8 (+12.4%); peak_kv 818 vs recompute 2228 / apc 818 (efficiency_beats=True, perf_drop=True)
- tp=4 gsm8k: NEED FIX — lat 205.9 vs 182.7 (+12.7%); peak_kv 815 vs recompute 2468 / apc 815 (efficiency_beats=True, perf_drop=True)
- tp=4 humaneval: NEED FIX — lat 144.2 vs 136.1 (+6.0%); peak_kv 1895 vs recompute 1807 / apc 1807 (efficiency_beats=False, perf_drop=False)

After you finish editing, if this file was written for `--edit wait`,
create `/home/dliu/ForkServe/logs/rl_improve/CONTINUE` so the loop can re-occupy and re-bench.

# ForkServe RL improve — round 2

Verdict: **FAIL** — generation quality dropped vs vLLM
Bench JSON: `/home/dliu/ForkServe/logs/rl_improve_round_2.json`

You are editing the ForkServe repo on this machine. GPUs are occupied
by `occupy_gpus.py` so another user cannot steal them. Do **not** start
another GPU bench yourself.

## Goal
Make ForkServe more efficient than vLLM (lower peak KV than
`vllm_recompute`, not worse than `vllm_apc`) **and** keep latency
within the configured slack of the vLLM baseline (APC if present),
and keep **task quality** almost unchanged vs APC: GSM8K accuracy,
Game24 success rate, HumanEval pass@1 (not token overlap).

Focus on serving path: CoW / two-class scheduler / speculative prefill
(`forkserve/engine/`, `forkserve/bench.py` worker). Keep unit tests green.
Do not rewrite occupy scripts unless required.

## Pair results
- tp=2 game24: ok — lat 2141.7 vs 2173.6 (-1.5%); peak_kv 818 vs recompute 2228 / apc 818; success_rate 0.000 vs 0.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=2 gsm8k: ok — lat 2191.8 vs 2190.5 (+0.1%); peak_kv 815 vs recompute 2468 / apc 815; accuracy 0.250 vs 0.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=2 humaneval: NEED FIX — lat 2188.3 vs 2184.2 (+0.2%); peak_kv 1059 vs recompute 1918 / apc 1059; pass_at_1 0.250 vs 0.500 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=True)
- tp=4 game24: ok — lat 1416.8 vs 1424.6 (-0.5%); peak_kv 818 vs recompute 2228 / apc 818; success_rate 0.000 vs 0.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=4 gsm8k: ok — lat 1438.9 vs 1460.0 (-1.4%); peak_kv 815 vs recompute 2468 / apc 815; accuracy 0.250 vs 0.250 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=4 humaneval: NEED FIX — lat 1450.0 vs 1430.0 (+1.4%); peak_kv 1059 vs recompute 1918 / apc 1059; pass_at_1 0.250 vs 0.500 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=True)

After you finish editing, if this file was written for `--edit wait`,
create `/home/dliu/ForkServe/logs/rl_improve/CONTINUE` so the loop can re-occupy and re-bench.

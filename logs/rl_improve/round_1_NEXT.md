# ForkServe RL improve — round 1

Verdict: **FAIL** — clear latency drop vs vLLM
Bench JSON: `/home/dliu/ForkServe/logs/rl_improve_round_1.json`

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
- tp=2 game24: ok — lat 202.5 vs 222.5 (-9.0%); peak_kv 818 vs recompute 2228 / apc 818; success_rate 0.250 vs 0.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=2 gsm8k: ok — lat 234.1 vs 244.1 (-4.1%); peak_kv 815 vs recompute 2468 / apc 815; accuracy 0.000 vs 0.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=2 humaneval: ok — lat 206.2 vs 209.3 (-1.5%); peak_kv 1059 vs recompute 1918 / apc 1059; pass_at_1 0.000 vs 0.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=4 game24: ok — lat 143.8 vs 156.5 (-8.1%); peak_kv 818 vs recompute 2228 / apc 818; success_rate 0.250 vs 0.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=4 gsm8k: ok — lat 161.6 vs 169.6 (-4.7%); peak_kv 815 vs recompute 2468 / apc 815; accuracy 0.000 vs 0.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=4 humaneval: NEED FIX — lat 159.3 vs 139.4 (+14.2%); peak_kv 1059 vs recompute 1918 / apc 1059; pass_at_1 0.000 vs 0.000 (vllm_apc) (efficiency_beats=True, perf_drop=True, quality_drop=False)

After you finish editing, if this file was written for `--edit wait`,
create `/home/dliu/ForkServe/logs/rl_improve/CONTINUE` so the loop can re-occupy and re-bench.

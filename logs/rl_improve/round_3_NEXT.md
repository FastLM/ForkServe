# ForkServe RL improve — round 3

Verdict: **FAIL** — generation quality dropped vs vLLM
Bench JSON: `/home/dliu/ForkServe/logs/rl_improve_round_3.json`

You are editing the ForkServe repo on this machine. GPUs are occupied
by `occupy_gpus.py` so another user cannot steal them. Do **not** start
another GPU bench yourself.

## Goal
Make ForkServe more efficient than vLLM (lower peak KV than
`vllm_recompute`, not worse than `vllm_apc`) **and** keep latency
within the configured slack of the vLLM baseline (APC if present),
and keep **task quality** almost unchanged vs APC: GSM8K accuracy,
Game24 success rate, HumanEval pass@1 (not token overlap).

Quality is only informative if each winner can finish. Pass `--decode`
through to bench (default 256 **per item**, not 16). JSON `decode_tokens`
is n_items × decode — do not treat 64 as 64 tokens per problem.
HumanEval pass@1 must score official function-body completions
(`prompt + body + check(entry)`), not the ReAct/chat tail.

Focus on serving path: CoW / two-class scheduler / speculative prefill
(`forkserve/engine/`, `forkserve/bench.py` worker). Keep unit tests green.
Do not rewrite occupy scripts unless required.

## Pair results
- tp=2 game24: ok — lat 3557.8 vs 3583.1 (-0.7%); peak_kv 390 vs recompute 1508 / apc 638; success_rate 1.000 vs 1.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=2 gsm8k: ok — lat 6992.0 vs 7010.8 (-0.3%); peak_kv 459 vs recompute 1844 / apc 659; accuracy 0.750 vs 0.750 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=2 humaneval: ok — lat 3568.1 vs 3565.7 (+0.1%); peak_kv 695 vs recompute 1406 / apc 739; pass_at_1 0.750 vs 0.750 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=4 game24: ok — lat 2188.0 vs 2233.5 (-2.0%); peak_kv 390 vs recompute 1508 / apc 638; success_rate 1.000 vs 1.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=4 gsm8k: NEED FIX — lat 4292.4 vs 4308.1 (-0.4%); peak_kv 459 vs recompute 1844 / apc 659; accuracy 0.750 vs 1.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=True)
- tp=4 humaneval: ok — lat 2236.3 vs 2226.7 (+0.4%); peak_kv 695 vs recompute 1406 / apc 739; pass_at_1 0.750 vs 0.750 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)

After you finish editing, if this file was written for `--edit wait`,
create `/home/dliu/ForkServe/logs/rl_improve/CONTINUE` so the loop can re-occupy and re-bench.

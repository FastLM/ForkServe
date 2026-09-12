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

Quality is only informative if each winner can finish. Pass `--decode`
through to bench (default 256 **per item**, not 16). JSON `decode_tokens`
is n_items × decode — do not treat 64 as 64 tokens per problem.
HumanEval pass@1 must score official function-body completions
(`prompt + body + check(entry)`), not the ReAct/chat tail.

Focus on serving path: CoW / two-class scheduler / speculative prefill
(`forkserve/engine/`, `forkserve/bench.py` worker). Keep unit tests green.
Do not rewrite occupy scripts unless required.

## Pair results
- tp=2 game24: ok — lat 3559.9 vs 3593.2 (-0.9%); peak_kv 638 vs recompute 1508 / apc 638; success_rate 1.000 vs 1.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=2 gsm8k: ok — lat 6983.7 vs 7012.8 (-0.4%); peak_kv 659 vs recompute 1844 / apc 659; accuracy 0.750 vs 0.750 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=2 humaneval: ok — lat 3595.0 vs 3590.0 (+0.1%); peak_kv 739 vs recompute 1406 / apc 739; pass_at_1 0.500 vs 0.250 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=4 game24: ok — lat 2191.2 vs 2235.2 (-2.0%); peak_kv 638 vs recompute 1508 / apc 638; success_rate 1.000 vs 1.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=4 gsm8k: ok — lat 4282.7 vs 4311.4 (-0.7%); peak_kv 659 vs recompute 1844 / apc 659; accuracy 0.750 vs 0.750 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=4 humaneval: NEED FIX — lat 2229.0 vs 2230.0 (-0.0%); peak_kv 739 vs recompute 1406 / apc 739; pass_at_1 0.250 vs 0.500 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=True)

After you finish editing, if this file was written for `--edit wait`,
create `/home/dliu/ForkServe/logs/rl_improve/CONTINUE` so the loop can re-occupy and re-bench.

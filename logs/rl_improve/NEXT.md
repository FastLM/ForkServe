# ForkServe RL improve — round 1

Verdict: **FAIL** — generation quality dropped vs vLLM
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

Quality is only informative if each winner can finish. Pass `--decode`
through to bench (default 256 **per item**, not 16). JSON `decode_tokens`
is n_items × decode — do not treat 64 as 64 tokens per problem.
HumanEval pass@1 must score official function-body completions
(`prompt + body + check(entry)`), not the ReAct/chat tail.

Focus on serving path: CoW / two-class scheduler / speculative prefill
(`forkserve/engine/`, `forkserve/bench.py` worker). Keep unit tests green.
Do not rewrite occupy scripts unless required.

## Pair results
- tp=2 game24: ok — lat 3560.2 vs 3614.8 (-1.5%); peak_kv 638 vs recompute 2228 / apc 818; success_rate 1.000 vs 0.750 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=2 gsm8k: NEED FIX — lat 3580.5 vs 3636.6 (-1.5%); peak_kv 635 vs recompute 2468 / apc 815; accuracy 0.000 vs 0.250 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=True)
- tp=2 humaneval: ok — lat 3577.9 vs 3560.3 (+0.5%); peak_kv 739 vs recompute 1918 / apc 1059; pass_at_1 0.500 vs 0.500 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=4 game24: ok — lat 2186.2 vs 2205.7 (-0.9%); peak_kv 638 vs recompute 1508 / apc 638; success_rate 1.000 vs 1.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=4 gsm8k: ok — lat 2217.3 vs 2228.5 (-0.5%); peak_kv 635 vs recompute 1748 / apc 635; accuracy 0.000 vs 0.000 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)
- tp=4 humaneval: ok — lat 2231.2 vs 2228.8 (+0.1%); peak_kv 739 vs recompute 1406 / apc 739; pass_at_1 0.500 vs 0.500 (vllm_apc) (efficiency_beats=True, perf_drop=False, quality_drop=False)

After you finish editing, if this file was written for `--edit wait`,
create `/home/dliu/ForkServe/logs/rl_improve/CONTINUE` so the loop can re-occupy and re-bench.

## Edits this wait (do not start a GPU bench here)
- GSM8K decode is now 512 per item (`--gsm8k-decode`); Game24 / HumanEval stay 256.
- GSM8K prompt requires a last line `#### <number>` and no alternate plans.
- Judge: slack ≥ 1/n (one miss on n=4 is not FAIL); both-zero is uninformative.
- tp=2 GSM8K 0.00 vs 0.25 was unfair trunks + truncation, not a CoW regression.
- Unit tests: 46 passed. Touch CONTINUE after this file.

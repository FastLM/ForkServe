# ForkServe RL improve — round 4

Verdict: **FAIL** — efficiency beats vLLM and latency is within slack
Bench JSON: `/home/dliu/ForkServe/logs/rl_improve_round_4.json`

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
- tp=2 game24: ok — lat 198.1 vs 222.7 (-11.0%); peak_kv 818 vs recompute 2228 / apc 818 (efficiency_beats=True, perf_drop=False)
- tp=2 gsm8k: ok — lat 228.1 vs 248.1 (-8.1%); peak_kv 815 vs recompute 2468 / apc 815 (efficiency_beats=True, perf_drop=False)
- tp=2 humaneval: ok — lat 223.4 vs 204.2 (+9.4%); peak_kv 1059 vs recompute 1918 / apc 1059 (efficiency_beats=True, perf_drop=False)
- tp=4 game24: ok — lat 154.8 vs 157.3 (-1.6%); peak_kv 818 vs recompute 2228 / apc 818 (efficiency_beats=True, perf_drop=False)
- tp=4 gsm8k: ok — lat 161.5 vs 173.4 (-6.8%); peak_kv 815 vs recompute 2468 / apc 815 (efficiency_beats=True, perf_drop=False)
- tp=4 humaneval: ok — lat 154.0 vs 144.9 (+6.2%); peak_kv 1059 vs recompute 1918 / apc 1059 (efficiency_beats=True, perf_drop=False)

After you finish editing, if this file was written for `--edit wait`,
create `/home/dliu/ForkServe/logs/rl_improve/CONTINUE` so the loop can re-occupy and re-bench.

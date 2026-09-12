# ForkServe RL improve — round 1 (edit done)

0.00 on GSM8K / HumanEval was **not** 8B being too weak. `--decode`
defaulted to 16 and `rl_improve.py` never forwarded it, so each winner
only continued ~16 tokens. JSON `decode_tokens: 64` is 4×16, not 64
per problem.

## What changed
- `rl_improve` now passes `--decode` (default **256 per item**).
- `forkserve.bench` default decode is 256 (same meaning).
- HumanEval quality uses official completions (`prompt + body +
  check(entry)`), not the ReAct/chat tail (`<|eot_id|>`, `</think>`).

Do **not** start another GPU bench here. Touch CONTINUE so the loop
re-occupies and re-benches with these knobs.

After you finish editing, if this file was written for `--edit wait`,
create `/home/dliu/ForkServe/logs/rl_improve/CONTINUE` so the loop can re-occupy and re-bench.

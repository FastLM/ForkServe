# ForkServe RL improve — round 2 (edit done)

Root cause of HumanEval 0.25 vs 0.50: CoW `node_snap` keyed only by
per-tree `node_id` (always 1 at the root). Four sessions aliased to
the last prompt. GSM8K/Game24 showed the same last-item collapse.

## Fixes
- Session-scoped CoW keys (`session:node`) on snapshot / alias / extra_args
- Game24 scores the first valid expression, not the longest noisy line
- `quality_collapsed` fails the judge on identical decodes
- HumanEval e2e no longer includes the idle sleep pad
- JSON `decode_per_item`; Qwen chat template + `/no_think` on math trunks

Do **not** start another GPU bench here. Touch CONTINUE so the loop
re-occupies and re-benches.

After you finish editing, if this file was written for `--edit wait`,
create `/home/dliu/ForkServe/logs/rl_improve/CONTINUE` so the loop can re-occupy and re-bench.

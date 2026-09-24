# Setting P′ — contest forest e2e sweep

Why Table 9 e2e did not move: decode is ~99% of e2e, and each chunk’s
wall clock is the longest request. ForkServe+ already skips the three
non-winner thoughts (`skip_known_losers`). Raising the draft threshold
does not cut that decode. On the n=32 pilot, decode tokens fell
65536 → 42456 while every chunk of 8 still contained a 2048-token
trace, so e2e stayed 52.9s vs 52.1s.

## Fixed knobs (same as the Table 9 driver)

| knob | value |
|---|---|
| systems | `vllm_apc`, `forkserve`, `forkserve_plus` |
| workloads | `math500` (n=200), `aime` (n=73; `--limit 200` stops at the file) |
| tp | 2 |
| decode `D` | 2048 |
| max_model_len | 8192 |
| branching `k` | 4 |
| chunk | 8 |
| gpu_util | 0.85 |
| max_batched_tokens | 2048 (bench default) |
| dtype | bfloat16 |
| prefix cache | on |
| chunked prefill | on |
| GPUs | physical 0,1 (`CUDA_VISIBLE_DEVICES=0,1`). The original Table 9 driver used 2,3. |
| env | `VLLM_USE_FLASHINFER_SAMPLER=0`, `PYTHONPATH=$HOME/vllm_fs` |

## Method changes in this sweep

| knob | Table 9 | this sweep |
|---|---|---|
| `prune_threshold` | 0.15 | **0.45** (`FORKSERVE_PRUNE_THRESHOLD`) |
| answer stop | closed `\boxed{}` only if a newline follows | also if any later token follows the closing brace |
| stop check | full-string decode every step | last 192 tokens, every 8th step |

Models: DeepSeek-R1-Distill-Llama-8B, Qwen3-4B.
Pilot before the full run: Qwen3-4B, math500, n=32, APC vs ForkServe+ only
(`qwen3-4b_n32_thr045.json`).

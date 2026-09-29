"""4-GPU check: next-turn prefill with and without slack fill.

Qwen3-4B, tp=4, 8 GSM8K questions. No answer stop: every generate
asks for exactly one token, so decode length does not differ.

Each item has a 512-token spine, four 64-token residuals (three
dropped), a 192-token known suffix, and a 64-token observation.
The pin is the spine plus that suffix. The timed request is the
next turn, spine + suffix + observation. Prefix caching is on, so
a pin hit leaves only the observation to prefill.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from forkserve.bench_tasks import load_gsm8k

N = 8
SPINE = 512
RESIDUAL = 64
SUFFIX = 192
OBS = 64
MODEL = os.environ.get("FORKSERVE_MODEL", "/home/dliu/models/Qwen3-4B")
OUT = Path(os.environ.get("SLACK_FILL_OUT", "logs/slack_fill_gpu/qwen3-4b_n8_tp4.json"))


def _fit(tok, text: str, n: int, salt: int) -> list[int]:
    ids = tok.encode(text, add_special_tokens=False) or [1]
    ids = [salt % 10007] + ids
    out: list[int] = []
    while len(out) < n:
        out.extend(ids)
    return out[:n]


def _gen(llm, params, seqs: list[list[int]]) -> float:
    t0 = time.perf_counter()
    llm.generate(
        [TokensPrompt(prompt_token_ids=list(s)) for s in seqs],
        params,
        use_tqdm=False,
    )
    return (time.perf_counter() - t0) * 1000.0


def main() -> None:
    problems = load_gsm8k(N)
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=4,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=4096,
        disable_log_stats=True,
    )
    tok = llm.get_tokenizer()
    params = SamplingParams(max_tokens=1, temperature=0.0)
    items = []
    for i, p in enumerate(problems):
        spine = _fit(tok, p.question, SPINE, 1000 + i)
        residuals = [_fit(tok, f"thought {j} {p.question}", RESIDUAL, 2000 + 10 * i + j) for j in range(4)]
        suffix = _fit(tok, "tool result wrapper " + p.question, SUFFIX, 3000 + i)
        obs = _fit(tok, "observation " + p.answer, OBS, 4000 + i)
        items.append((spine, residuals, suffix, obs))

    # Warm the engine on a prompt that is not reused.
    _gen(llm, params, [[1, 2, 3, 4] * 32])

    summary = []
    for mode, salt in (("bare", 41), ("fill", 43)):
        batch = []
        for i, (spine, residuals, suffix, obs) in enumerate(items):
            sp = [salt, i] + spine[2:]
            batch.append((sp, residuals, suffix, obs))
        if mode == "bare":
            fanout_ms = _gen(llm, params, [sp + res for sp, residuals, _, _ in batch for res in residuals])
            pin_ms = 0.0
        else:
            fanout_ms = _gen(llm, params, [sp + residuals[0] for sp, residuals, _, _ in batch])
            pin_ms = _gen(llm, params, [sp + suffix for sp, _, suffix, _ in batch])
        next_ms = _gen(llm, params, [sp + suffix + obs for sp, _, suffix, obs in batch])
        summary.append(
            {
                "mode": mode,
                "n": N,
                "tp": 4,
                "answer_stop": "",
                "decode_tokens_per_call": 1,
                "fanout_ms": round(fanout_ms, 1),
                "pin_ms": round(pin_ms, 1),
                "next_ms": round(next_ms, 1),
                "next_prompt_tokens": SPINE + SUFFIX + OBS,
                "pinned_suffix_tokens": SUFFIX if mode == "fill" else 0,
                "freed_residual_tokens": 3 * RESIDUAL,
            }
        )
        print(
            f"{mode}: fanout={fanout_ms:.1f} ms pin={pin_ms:.1f} ms next={next_ms:.1f} ms",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": MODEL, "rows": summary}, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

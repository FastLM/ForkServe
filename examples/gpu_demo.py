"""Run ForkServe on a real GPU via VllmBackend.

Default model is local Qwen3-8B. Override with env:

  FORKSERVE_MODEL   path to weights (default: /home/dliu/models/Qwen3-8B)
  FORKSERVE_TP      tensor parallel size (default: 1)
  FORKSERVE_GPU_UTIL  vLLM gpu_memory_utilization (default: 0.90)
  FORKSERVE_MAX_LEN   max_model_len (default: 4096)
  FORKSERVE_PROMPT    open() prompt
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    from forkserve import Engine, ForkServeConfig
    from forkserve.engine.vllm_backend import VllmBackend

    model = os.environ.get("FORKSERVE_MODEL", "/home/dliu/models/Qwen3-8B")
    tp = int(os.environ.get("FORKSERVE_TP", "1"))
    util = float(os.environ.get("FORKSERVE_GPU_UTIL", "0.90"))
    max_len = int(os.environ.get("FORKSERVE_MAX_LEN", "4096"))
    prompt = os.environ.get(
        "FORKSERVE_PROMPT",
        "You are a coding agent. Fix the failing unit test in test_parser.py.",
    )
    print(
        f"loading VllmBackend model={model} tp={tp} util={util} max_len={max_len}",
        flush=True,
    )
    cfg = ForkServeConfig(
        max_batched_tokens=2048,
        hbm_capacity_bytes=40.0 * (1 << 30),
        bytes_per_token=1024.0,
    )
    backend = VllmBackend(
        config=cfg,
        model=model,
        tensor_parallel=tp,
        max_model_len=max_len,
        gpu_memory_utilization=util,
        enforce_eager=os.environ.get("FORKSERVE_ENFORCE_EAGER", "0") == "1",
    )
    eng = Engine(backend, cfg)

    print("open ...", flush=True)
    h = eng.open(prompt)

    wrap = ' {"name": "bash", "arguments": {"cmd": "pytest"}}'
    recov = " I will retry with a smaller command."
    print("fork + speculate ...", flush=True)
    happy = eng.fork(h.id, h.tip, "bash", wrap)
    fail = eng.fork(h.id, h.tip, "err", recov)
    eng.speculate(h.id, happy, t_idle_ms=2900)
    eng.speculate(h.id, fail, priority=0.2, t_idle_ms=2900)
    n_spec = eng.drain_slack()
    print(f"drained spec tokens={n_spec}", flush=True)

    print("commit ...", flush=True)
    cr = eng.commit(h.id, h.tip, wrap + "\nstdout of pytest\npassed\n")
    print(
        "commit",
        {
            "winner": cr.winner,
            "lcp": cr.lcp,
            "known_suffix_hit": cr.known_suffix_hit,
            "skipped_prefill": cr.skipped_prefill,
            "aborted": len(cr.aborted),
        },
        flush=True,
    )

    print("generate ...", flush=True)
    out = eng.generate(h.id, 32)
    text = backend.detokenize(tuple(out))
    print("generated:", text, flush=True)
    print("metrics:", eng.metrics[h.id].summary(), flush=True)
    print("forkserve gpu demo ok", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)

# ForkServe

Branch-aware speculative prefilling and copy-on-write KV state for agentic LLM serving.

The next LLM call in a production agent is almost never unique. A ReAct step chooses among tools; a planner fans out to specialists; Tree-of-Thoughts expands siblings. Those branches share a long trunk and differ in a short residual. ForkServe makes the **branch** the first-class serving object.

```
harness  --open/fork/speculate/commit/join/abort-->  Engine
                                                       ├─ SpeculatePlanner   Algorithm 1
                                                       ├─ TwoClassScheduler  committed ≻ spec
                                                       ├─ ContextTree + CoW pages
                                                       └─ LCP commit (Theorem 2)
```

## What is implemented

| Paper | Code |
|---|---|
| Definition 1 context tree | `forkserve/tree.py` |
| CoW pages, Lemma 3 abort cost, Eq. (3) | `forkserve/pages.py` |
| Algorithm 1 + Eq. (4)(5) + Prop. 1 | `forkserve/planner.py` |
| LCP commit, Theorem 2 | `forkserve/commit.py` |
| Two-class token budget Eq. (6)(7) | `forkserve/scheduler.py` |
| Node TTL Eq. (1), leaf idleness Eq. (2) | `forkserve/retention.py` |
| Tree-sticky routing + residual steal | `forkserve/router.py` |
| Join scaffolds | `forkserve/join.py` |
| ReAct / LangGraph / OpenHands / ToT | `forkserve/adapters/` |
| vLLM seam (optional extra) | `forkserve/engine/vllm_backend.py` |

Correctness is by construction: speculative KV is an input-side cache. Decode of committed tokens uses only KV that matches the committed token sequence. No speculative output token is ever shown or fed back.

## Install

```bash
pip install -e ".[dev]"
pytest -q
```

vLLM is optional (`pip install -e ".[vllm]"`). The core algorithms and adapters run on the mock backend without a GPU.

## API

```python
from forkserve import Engine, ForkServeConfig
from forkserve.engine import MockBackend

eng = Engine(MockBackend(ForkServeConfig()))
h = eng.open(system_and_user_tokens)

happy = eng.fork(h.id, h.tip, "bash", wrap)   # p = 1 known suffix
fail  = eng.fork(h.id, h.tip, "err", recov)
eng.speculate(h.id, happy, t_idle_ms=2900)
eng.speculate(h.id, fail, priority=0.2, t_idle_ms=2900)
eng.drain_slack()                            # leftover SM / tool pause

cr = eng.commit(h.id, h.tip, wrap + observation)  # LCP binds
eng.generate(h.id, 128)                      # committed decode only
```

`commit` is the only verb on the critical path of user-visible tokens.

## Design principles (paper §4.3)

1. **Correctness by construction** — speculation never enters the sampler.
2. **Known before guessed** — p=1 suffixes starve p<1 residuals.
3. **Misses cost residual, not trunk** — CoW is what makes multi-child speculation rational.
4. **Speculation is slack, not load** — under saturation `B^s_t = 0` and the system degrades to retention + CoW fan-out.
5. **No workflow oracle** — branch ids and known suffixes come from the harness at the moment they exist.

## Defaults

Paper §9 knobs: page size `P=16`, speculative chunk `C_spec=512`, `λ` such that 1 ms TBT ≡ 4 ms TTFT, grammar top-`m` ≤ 3, branch cap 6, `q_min=0.35`.

## Status

This repository is a complete, testable implementation of the ForkServe control plane (state, planner, scheduler, commit, adapters). The vLLM backend is an integration seam: production kernels still live in vLLM's PagedAttention / chunked-prefill path; ForkServe owns page identity, admission, and the branch tree.

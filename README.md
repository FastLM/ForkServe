# ForkServe

**Branch-aware speculative prefilling and copy-on-write KV state for agentic LLM serving.**

Dong Liu, Chuan Wu, and contributors

Production agents do not issue a unique next prompt. A ReAct step chooses among tools, a planner fans out to specialists, Tree-of-Thoughts expands siblings, a failed call opens recovery. Those branches share a long trunk and differ in a short residual, yet engines still recompute the trunk, serialize the fan-out, or wait until the chosen child is fully known before prefilling.

ForkServe makes the **branch** the first-class serving object: a forkable context tree, speculative prefill of known suffixes (and, when profitable, observation residuals), and a two-class scheduler that spends only idle GPU cycles.

```
harness  -- open / fork / speculate / commit / join / abort -->  Engine
                                                                  ├─ SpeculatePlanner     Algorithm 1
                                                                  ├─ TwoClassScheduler    committed ≻ spec
                                                                  ├─ ContextTree + CoW    Definition 1
                                                                  └─ LCP commit           Theorem 2
```

`commit` is the only verb on the critical path of user-visible tokens. Speculative KV is an input-side cache: decode attends only to the committed sequence. No speculative output token is ever shown or fed back.

## Mechanisms

1. **Forkable CoW tree.** `fork` aliases parent pages in O(1). Writes allocate residual pages. Abort cost is residual-only, independent of trunk length. Join reuses the shared trunk plus a scaffold; unused children die at residual cost.

2. **Speculative prefill.** During tool idle and decode slack, admit work that maximizes expected TTFT reduction subject to idle horizon, residual HBM, and committed TBT margin. Known suffixes (`p=1`) starve guessed residuals (`p<1`). Candidates come from the harness, constrained-decoding mass (top-`m`), and a session-local prior that never invents a branch id.

3. **Two-class schedule.** Committed decode / commit-tail / root prefill take token budget first. Speculative chunks fill the remainder, are preemptible at `C_spec`, and are cancelled on generation mismatch. Under saturation `B^s_t = 0`: retention + CoW fan-out only.

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
eng.drain_slack()

cr = eng.commit(h.id, h.tip, wrap + observation)  # LCP binds
eng.generate(h.id, 128)                            # committed decode only
```

Harness adapters lower ReAct, LangGraph `Send`, OpenHands / SWE, and Tree-of-Thoughts onto these verbs. They announce structure; they do not rewrite prompts.

## Paper → code

| Paper | Code |
|---|---|
| Definition 1 context tree, prefix / spec-isolation / cascade abort | `forkserve/tree.py` |
| CoW pages, abort cost, \(M_\mathrm{CoW} = b(L+\sum \ell_i)\) | `forkserve/pages.py` |
| Algorithm 1, \(G(w)\) / \(C(w)\), Proposition 1, grammar + prior + n-gram | `forkserve/planner.py` |
| LCP commit, Theorem 2 (output identity) | `forkserve/commit.py` |
| \(B^c_t\), \(B^s_t\), PLAS, VTC \(\kappa=0.25\) | `forkserve/scheduler.py` |
| Node TTL on residual, leaf-first relative offload | `forkserve/retention.py` |
| Tree-sticky routing, residual-only steal | `forkserve/router.py` |
| Join scaffolds (all / first / k-of-n / winner) | `forkserve/join.py` |
| ReAct / LangGraph / OpenHands / ToT | `forkserve/adapters/` |
| vLLM engine loop: CoW bit + two-class ``schedule()`` | `forkserve/engine/vllm_loop.py` |
| vLLM backend (tags requests, installs the loop) | `forkserve/engine/vllm_backend.py` |

Defaults: page size \(P=16\), \(C_\mathrm{spec}=512\), \(\lambda\) such that 1 ms TBT ≡ 4 ms TTFT, grammar top-\(m\) ≤ 3, branch cap 6, \(q_\min=0.35\).

## Status

This repository is the ForkServe control plane plus a **vLLM V1 engine-loop comparison substrate**:

- **CoW bit** — `install_vllm_cow()` wraps `BlockPool.touch` so any block with `ref_cnt > 1` is pinned read-only. A child that still finds its parent in `req_to_blocks` aliases those blocks in O(1) (`get_computed_blocks`) instead of hashing the trunk. Abort decrefs; the trunk stays while a sibling holds a ref.
- **Two-class scheduler** — `TwoClassVllmScheduler` subclasses vLLM's iteration scheduler and reorders `running` / `waiting` so committed work takes the token budget first; speculative requests (`extra_args['forkserve_class']='speculative'`) fill the leftover. Saturation ⇒ spec not scheduled.

Still out of tree: prefill/decode disaggregation, and real HBM↔DRAM tensor movement.

## Install

```bash
pip install -e ".[dev]"
pytest -q
```

vLLM is optional (`pip install -e ".[vllm]"`). Core algorithms run on `MockBackend` without a GPU.

```python
from forkserve.engine.vllm_backend import VllmBackend

backend = VllmBackend(
    ForkServeConfig(),
    model="/path/to/model",
    gpu_memory_utilization=0.45,
    two_class=True,
    cow_blocks=True,
)
eng = Engine(backend)
```

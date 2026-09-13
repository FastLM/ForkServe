# ForkServe

**Branch-aware speculative prefilling and copy-on-write KV state for agentic LLM serving.**

Dong Liu, Chuan Wu, and contributors

Production agents do not issue a unique next prompt. A ReAct step chooses among tools, a planner fans out to specialists, Tree-of-Thoughts expands siblings, a failed call opens recovery. Those branches share a long trunk and differ in a short residual, yet engines still recompute the trunk, serialize the fan-out, or wait until the chosen child is fully known before prefilling.

ForkServe makes the **branch** the first-class serving object: a forkable context tree, speculative prefill of known suffixes, and a two-class scheduler that spends only idle GPU cycles.

`commit` is the only verb on the critical path of user-visible tokens. Speculative KV is an input-side cache: decode attends only to the committed sequence. No speculative output token is ever shown or fed back.

## Architecture

```
 L3  harness     structure only — no prompt rewrite
 +------------------------------------------------------------------+
 |   ReAct          ToT         LangGraph Send         tool I/O     |
 +-------------------------------+----------------------------------+
                                 |
          open  fork  speculate  commit  join  abort
                                 |
 L2  control plane
 +-------------------------------v----------------------------------+
 |                                                                  |
 |    Planner              Scheduler              Commit            |
 |    admit by G/C         two-class batch        LCP bind          |
 |    G(w) / C(w)          B_c  before  B_s       abort losers      |
 |         |                    |                      |            |
 |         +--------------------+----------------------+            |
 |                              |                                   |
 |                              v                                   |
 |                   Context tree  (per session)                    |
 |              shared trunk pages (refcount, read-only)            |
 |              private residual pages · cascade abort              |
 |                              |                                   |
 |                   Router (sticky)   Retention (TTL)              |
 +-------------------------------+----------------------------------+
                                 |
 L1  execution
 +-------------------------------v----------------------------------+
 |   MockBackend                         vLLM V1                    |
 |   algorithms only                     CoW pin / page alias       |
 +------------------------------------------------------------------+
```

Three layers, one contract: the harness announces branch structure; the control plane decides what to prefill and what to keep; the backend only moves pages and tokens.

```
session
 └── trunk          shared, refcounted, never rewritten in place
      ├── thought-0     residual  ──►  winner  ──►  decode
      ├── thought-1     residual  ──►  abort (residual cost only)
      └── thought-2     residual  ──►  abort
```

## Methods

**CoW tree.** `fork` aliases parent pages in O(1). A write allocates residual pages only. Live KV is \(M_{\mathrm{CoW}}=b\bigl(L+\sum_i \ell_i\bigr)\) versus clone \(M_{\mathrm{clone}}=b\bigl(kL+\sum_i \ell_i\bigr)\). Abort and cascade-free cost the residual, not the trunk. Join reuses the shared trunk plus a scaffold.

**Speculative admit.** Rank candidates by expected TTFT gain over cost. Gain \(G(w)=p(w)\cdot\min(T_{\mathrm{pre}},T_{\mathrm{idle}})\cdot\eta(w)\). Cost \(C(w)=b|w|+\lambda\max(0,T_{\mathrm{pre}}-\gamma)\). Known suffixes (\(p=1\)) starve guessed residuals (\(p<1\)). Sources: harness-declared wraps, constrained-decoding mass (top-\(m\)), session-local prior (never invents a branch id).

**LCP commit.** Bind the observed token prefix; CoW-split a mid-page; abort siblings that lost the LCP. Decode uses only KV that matches the committed sequence (output identity).

**Two-class batch.** Each tick: committed jobs take \(B^c_t\) first (decode, commit-tail, root prefill); speculative chunks fill \(B^s_t\), preemptible at \(C_{\mathrm{spec}}\), dropped on generation mismatch. Under saturation \(B^s_t=0\).

**Retention and placement.** TTL on the residual (not the session); leaf-first relative idleness for offload. Tree-sticky routing; steal residual pages only.

Defaults: page size \(P=16\), \(C_{\mathrm{spec}}=512\), \(\lambda\) such that 1 ms TBT ≡ 4 ms TTFT, grammar top-\(m\) ≤ 3, branch cap 6, \(q_{\min}=0.35\), VTC spec billed at \(\kappa=0.25\).

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

Harness adapters lower ReAct, LangGraph `Send`, OpenHands / SWE, and Tree-of-Thoughts onto these verbs.

## Status

Control plane plus a vLLM V1 comparison substrate:

- **One generate per phase** — open, CoW fan-out, and winner decode each map to one `LLM.generate` for the whole slice (same shape as APC). `generate_committed_many` fuses commit-tails and drops speculative siblings so HumanEval TTFT is not a recovery prefill.
- **CoW bit** — `install_vllm_cow()` snapshots + extra-pins parent prompt blocks. Children alias complete pages only. Abort decrefs residuals; the trunk stays while a sibling holds a ref.
- **Stock async scheduler by default** — two-class reorder is opt-in (`AsyncScheduler` subclass). A factory `scheduler_cls` made vLLM fall back to sync `Scheduler`.

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
    two_class=False,
    cow_blocks=True,
)
eng = Engine(backend)
```

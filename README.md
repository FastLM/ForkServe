# ForkServe

**Copy-on-write context trees and speculative prefilling for agentic LLM serving.**

Dong Liu, Chuan Wu, and contributors

An agentic step is a tree of token sequences: trunk length \(L\), \(k\) residuals \(\ell_i\), committed path \(L+\ell_\star\). Engines that treat each child as an independent request recompute the trunk, serialize the fan-out, or wait until the residual exists before sharing KV. ForkServe aliases the trunk at `fork`, prefills known suffixes in leftover tokens, and binds the observed prefix by longest common prefix.

Speculative KV is an input-side cache. Decode attends only to the committed sequence. `commit` is the only verb on the critical path of user-visible tokens.

Specification: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). vLLM mapping: [`docs/FORKSERVE_VLLM_DESIGN.md`](docs/FORKSERVE_VLLM_DESIGN.md).

## Architecture

```
L3  harness        structure only — adapters do not rewrite prompts
    ReAct · ToT · LangGraph Send · OpenHands
                    │
    open  fork  speculate  commit  join  abort
                    │
L2  control plane
    Planner            Scheduler           Commit / Join
    G(w)/C(w) knapsack two-class batch     LCP bind; trunk+scaffold
    known suffixes ≻ residuals             abort losers
                    │
    Context tree T_s   σ_v = (parent, alias_len, residual pages)
    Router (tree-sticky) · Retention (TTL on residual, not session)
                    │
L1  execution
    MockBackend · vLLM V1 (CoW pin / page alias)
```

| | Recompute | APC / radix | ForkServe |
|---|---|---|---|
| Share | never | after tokens; full blocks | at `fork`; pages, including unaligned tail in the design |
| Fan-out KV | \(kL+\sum\ell_i\) | \(L+\sum\ell_i\) on hit; clone on first fill | \(L+\sum\ell_i\) |
| After abort | — | hashed residuals until LRU | \(L+\ell_\star\) |
| Spec pages | — | — | residual only; not sampled |

Full invariants, admission, and the backend contract: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Methods

**CoW tree.** `fork` aliases \(\sigma_u\) in \(O(1)\). A write allocates residual pages only. Live KV is \(M_{\mathrm{CoW}}=b(L+\sum_i\ell_i)\) versus clone \(M_{\mathrm{clone}}=b(kL+\sum_i\ell_i)\). Abort costs the residual, not the trunk.

**Speculative admit.** \(G(w)=p(w)\cdot\min(T_{\mathrm{pre}},T_{\mathrm{idle}})\cdot\eta(w)\), \(C(w)=b|w|+\lambda\max(0,T_{\mathrm{pre}}-\gamma)\). Known suffixes (\(p=1\)) starve guessed residuals (\(p<1\)). Sources: harness wraps, constrained-decoding mass (top-\(m\)), session-local prior (never invents a branch id).

**LCP commit.** Bind the observed token prefix; CoW-split a mid-page; abort siblings. Decode uses only KV that matches the committed sequence.

**Two-class batch.** Committed jobs take \(B^c_t\) (decode, commit-tail, root prefill); speculative chunks fill \(B^s_t\), preemptible at \(C_{\mathrm{spec}}\), dropped on generation mismatch. Under saturation \(B^s_t=0\).

**Retention and placement.** TTL on the residual. Leaf-first relative idleness for offload. Tree-sticky routing; steal residual pages only.

Defaults: \(P=16\), \(C_{\mathrm{spec}}=512\), \(1\,\mathrm{ms}\) TBT \(\equiv 4\,\mathrm{ms}\) TTFT, grammar top-\(m\le 3\), branch cap \(6\), \(q_{\min}=0.35\), VTC spec billed at \(\kappa=0.25\).

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

cr = eng.commit(h.id, h.tip, wrap + observation)
eng.generate(h.id, 128)
```

Adapters lower ReAct, LangGraph `Send`, OpenHands / SWE, and Tree-of-Thoughts onto these verbs.

## Status

Control plane plus a vLLM V1 comparison substrate:

- **One generate per phase** — open, CoW fan-out, and winner decode each map to one `LLM.generate`. `generate_committed_many` fuses commit-tails and drops speculative siblings.
- **CoW bit** — `install_vllm_cow()` snapshots + extra-pins parent prompt blocks. Children alias complete pages. Abort decrefs residuals.
- **Stock async scheduler by default** — two-class reorder is opt-in (`AsyncScheduler` subclass).

Out of tree: prefill/decode disaggregation; HBM\(\leftrightarrow\)DRAM movement; freeze-tail slot map.

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

# ForkServe

**The serving object is the branch. Its cost is paid for the residual that is committed, and only at the cheapest stage where that rejection is still correct.**

Dong Liu, Chuan Wu, and contributors

An agent step is a tree: one trunk of length \(L\), \(k\) residuals of length \(\ell_i\), and one committed path \(L+\ell_\star\). A prefix cache can share the trunk only after those tokens exist and a hash hits, so the first fan-out still copies the trunk once per child. Prefill/decode disaggregation then transfers every page the prefiller computed, including residuals the harness later aborts.

ForkServe sits between the harness and the engine. The harness announces structure. The control plane admits, rejects, binds, and retains. The backend moves pages and tokens. `commit` is the only verb on the critical path of user-visible tokens. Speculative KV is a cache of a future prompt. Decode attends the committed spine only.

Specification: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Page contract on vLLM: [`docs/FORKSERVE_VLLM_DESIGN.md`](docs/FORKSERVE_VLLM_DESIGN.md). Hash index and measurements: [`docs/HASH_FORKSERVE.md`](docs/HASH_FORKSERVE.md).

## Three mechanisms

They are one decision. Each exists so the next one can reject less work.

**1. Copy-on-write context tree.** `fork` copies two words — the parent page table and the alias length — and increfs the trunk. A write allocates residual pages only. Live KV in flight is \(M_{\mathrm{CoW}}=b(L+\sum_i\ell_i)\). Abort decrefs the residual, so occupancy collapses to the spine \(M_{\mathrm{spine}}=b(L+\ell_\star)\). A deeper fan-out is a sequence of these aliases: live paths share every page on the common prefix, and a subtree with no live child cascades until the committed spine.

**2. One action per residual.** Each residual exits at the first stage that can reject it. Stage cost increases. Index 0, the winner, is always kept.

| Action | When | GPU work |
| --- | --- | --- |
| Hash skip | Published full pages cover the residual | 0 |
| Hash partial | A proper prefix is already published | Miss tail |
| Draft skip | A repeated loop (score \(0.02\), below \(\tau_{\mathrm{pre}}=0.15\)) | 0 |
| Early abort | The first fraction of the residual already fails | That prefix, then stop |
| Prefill | Otherwise, on the trunk aliased at `fork` | The residual |

Illegal text scores \(0.22\) and is prefilled. A low score that is not a loop is not a skip: that residual is prefilled, decoded for a short probe, and extended to the budget only if the generated prefix clears the decode threshold \(\tau=0.45\). The connector inserts only the miss tail of a residual that is kept. A hash-local hit stays on the decode instance.

One admission rule is on at a time (`admit_mode`): `score` (the threshold above), `winner`, `top_m`, or `alpha`.

**3. The spine on the critical path.** Decode slots are \(C=\lfloor H/M_{\mathrm{spine}}\rfloor\). Each tick fills committed work first — decode, the commit tail, root prefill. Speculative chunks take only what remains, stop at a chunk boundary, and drop when the parent’s generation changes. Under saturation that remainder is zero.

A known suffix (\(p=1\): a harness wrapper) is admitted before any guessed observation. If it fits in the tool pause, it is pinned while the tool runs. `commit` then prefills only the unmatched tail. After a residual is dropped, the freed tokens are a budget: a page-aligned prefix of the next known suffix is published into it, and nothing longer.

`commit` binds by longest common prefix, splits a mid-page, aborts the other speculative children, and releases decode. Committed tokens match a baseline that prefills only committed tokens.

```
L3  harness          structure only — adapters do not rewrite prompts
    ReAct · ToT · LangGraph Send · OpenHands
                     │
    open  fork  speculate  commit  join  abort
                     │
L2  control plane    one decision per residual
    tree alias  →  hash / loop / early-abort / miss tail  →  LCP bind
    committed budget first; known suffix in the tool pause
                     │
L1  execution        MockBackend · vLLM V1 (page alias)
```

APC remains the index of committed full pages, for a later session that never forked from this one. `fork` does not walk that index. Speculative pages are not published.

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

Adapters lower ReAct, LangGraph `Send`, OpenHands / SWE, and Tree-of-Thoughts onto these verbs. The cascade is on when the config comes from `plus_config()` / `app_config()`, or when the bench is run with `--system forkserve_plus`.

## Control-plane snapshot

In-repo cost model, not a GPU trace. Ten sessions, 4-way fan-out, trunk 256, residual 32, page size 16. Of the four thoughts, one is a repeated loop and one is illegal text. The loop is the only draft skip.

| Method | Prefill tok | Fan-out ms | Transfer tok | Peak KV | Pruned |
| --- | ---: | ---: | ---: | ---: | ---: |
| APC (first expand) | 11520 | 139.0 | 0 | 11520 | 0 |
| ForkServe (alias, every residual) | 3840 | 47.1 | 0 | 3840 | 0 |
| Disagg, no prune | 11520 | 162.1 | 11520 | 11520 | 0 |
| Cascade | 3232 | 40.3 | 672 | 2880 | 10 |

Against APC on this fan-out: prefill \(-71.9\%\), fan-out \(-71.0\%\), peak KV \(-75\%\). Peak KV is the spine, \(10\times(256+32)\). Replay of the published winner prefills 0 tokens.

```bash
pip install -e ".[dev]"
PYTHONPATH=. python experiments/prefill_prune_bench.py
pytest -q
```

Protocol and the GPU forest: [`docs/HASH_FORKSERVE_EXPERIMENT.md`](docs/HASH_FORKSERVE_EXPERIMENT.md).

## Install

vLLM is optional (`pip install -e ".[vllm]"`). The control plane runs on `MockBackend` without a GPU.

```python
from forkserve import Engine, ForkServeConfig
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

Defaults: page size \(P=16\), speculative chunk \(C_{\mathrm{spec}}=512\), \(1\,\mathrm{ms}\) TBT weighed as \(4\,\mathrm{ms}\) TTFT, grammar top-\(m\le 3\), branch cap \(6\), \(q_{\min}=0.35\), prefill threshold \(0.15\), decode threshold \(0.45\).

## What is in tree

- Context tree, CoW page pool, \(G/C\) admission, two-class batch, LCP commit, node TTL, tree-sticky routing.
- The prefill cascade: hash probe, loop-only draft skip, early abort, disagg gate on the miss tail. `admit_mode` selects one rule.
- Slack fill of the next known suffix into the budget a dropped residual just freed.
- vLLM V1 backend: one `LLM.generate` per phase, parent blocks extra-pinned so a child aliases them. Two-class reorder is opt-in; the stock async scheduler stays the default.

Specified and not wired into attention: a freeze-tail slot map for an unaligned trunk, HBM\(\leftrightarrow\)DRAM movement, and a production prefill/decode connector. The in-tree connector is the gate that decides what would be inserted.

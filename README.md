# ForkServe

**Copy-on-write context trees and speculative prefilling for agentic LLM serving.**

Dong Liu, Chuan Wu, and contributors

An agentic step is a tree of token sequences: trunk length $L$, $k$ residuals $\ell_i$, committed path $L+\ell_\star$. Engines that treat each child as an independent request recompute the trunk, serialize the fan-out, or wait until the residual exists before sharing KV. ForkServe aliases the trunk at `fork`, prefills known suffixes in leftover tokens, and binds the observed prefix by longest common prefix.

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

| &nbsp; | Recompute | APC / radix | ForkServe |
| --- | --- | --- | --- |
| Share | never | after tokens; full blocks | at `fork`; pages, including unaligned tail in the design |
| Fan-out KV | $kL+\sum\ell_i$ | $L+\sum\ell_i$ on hit; clone on first fill | $L+\sum\ell_i$ |
| After abort | — | hashed residuals until LRU | $L+\ell_\star$ |
| Spec pages | — | — | residual only; not sampled |

Full invariants, admission, and the backend contract: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Design vs APC

APC is the right primitive for *unrelated* HTTP requests that happen to share a system prompt. It is the wrong primitive for a ReAct / ToT / LangGraph `Send` step, where the engine already knows the parent page table.

vLLM V1 Automatic Prefix Caching is **content-addressed**: a hash of a *full* block, discovered *after* those tokens exist, by walking the new prompt. A same-batch fan-out looks up before any child has published a hash, so APC misses and clones $k$ trunks. Partial hits CoW-copy a whole scheduler page. Unaligned tails ($r = L \bmod P$) stay private per child. Speculative pages cannot wait for tokens that have not been issued.

ForkServe is **location-addressed copy-on-write**. `fork` aliases the parent’s frames *before* the residual exists. Live identity is $M_{\mathrm{CoW}}=b(L+\sum_i\ell_i)$ versus clone $M_{\mathrm{clone}}=b(kL+\sum_i\ell_i)$. Abort decrefs residual pages only. Speculative pages are never hashed and never sampled.

APC is kept as a **secondary** index. Committed full pages are published so a *different* session can hash-hit them. Intra-session fan-out never consults the hash on the critical path. Existing APC for ordinary requests is unchanged.

The unaligned trunk tail is **adaptive**:

```
pages_pack   = k * ceil((r + ℓ) / P)     # clone r into each child's first page
pages_freeze = 1 + k * ceil(ℓ / P)       # one RO tail + private residuals
```

Pack when the wrapper fits in one page; freeze when it does not. Production attention still assumes full blocks except the last, so the in-tree path is pack plus `cow_copy_kv_rows` ($n_{\mathrm{valid}}$ rows, not $P$). Freeze-tail needs a fork-aware slot map (Tier B); it is specified, not wired into FlashAttention.

The storage win is first fan-out / same-batch miss, unaligned tails, and abort — not an already-hashed aligned single continuation, where APC is already $M_{\mathrm{CoW}}$.

## Methods

**CoW tree.** `fork` aliases $\sigma_u$ in $O(1)$. A write allocates residual pages only. Live KV is $M_{\mathrm{CoW}}=b(L+\sum_i\ell_i)$ versus clone $M_{\mathrm{clone}}=b(kL+\sum_i\ell_i)$. Abort costs the residual, not the trunk.

**Speculative admit.** $G(w)=p(w)\cdot\min(T_{\mathrm{pre}},T_{\mathrm{idle}})\cdot\eta(w)$, $C(w)=b|w|+\lambda\max(0,T_{\mathrm{pre}}-\gamma)$. Known suffixes ($p=1$) starve guessed residuals ($p < 1$). Sources: harness wraps, constrained-decoding mass (top-$m$), session-local prior (never invents a branch id).

**LCP commit.** Bind the observed token prefix; CoW-split a mid-page; abort siblings. Decode uses only KV that matches the committed sequence.

**Two-class batch.** Committed jobs take $B^c_t$ (decode, commit-tail, root prefill); speculative chunks fill $B^s_t$, preemptible at $C_{\mathrm{spec}}$, dropped on generation mismatch. Under saturation $B^s_t=0$.

**Retention and placement.** TTL on the residual. Leaf-first relative idleness for offload. Tree-sticky routing; steal residual pages only.

Defaults: $P=16$, $C_{\mathrm{spec}}=512$, $1\,\mathrm{ms}$ TBT $\equiv 4\,\mathrm{ms}$ TTFT, grammar top-$m\le 3$, branch cap $6$, $q_{\min}=0.35$, VTC spec billed at $\kappa=0.25$.

## Results

Measured on the in-tree vLLM V1 `KVCacheManager` (CPU, no GPU). APC children all call `get_computed_blocks` *before* any allocate, so hashes are unpublished — the same-batch miss / first planner fan-out. ForkServe snapshots the parent, then each child aliases through `forkserve_parent` (hash bypass). $P=16$.

Live **blk** is distinct frames in the unit-test pool (one dummy layer). **GiB** is Llama-3-8B GQA bf16 closed-form allocated slots ($b = 128\,\mathrm{KiB/tok}$), not the dummy pool.

| Scenario | $L$ | $k$ | $\ell$ | APC blk | FS blk | Save | APC GiB | FS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned_fanout_4 | 2048 | 4 | 64 | 528 | 144 | 72.7% | 1.03 | 0.28 |
| aligned_fanout_8 | 4096 | 8 | 128 | 2112 | 320 | 84.8% | 4.12 | 0.62 |
| unaligned_pack | 2050 | 4 | 48 | 528 | 145 | 72.5% | 1.03 | 0.28 |
| tot_wide | 8192 | 8 | 32 | 4112 | 528 | 87.2% | 8.03 | 1.03 |
| short_residual | 1024 | 6 | 4 | 390 | 70 | 82.1% | 0.76 | 0.14 |
| planner_specialists | 16384 | 4 | 256 | 4160 | 1088 | 73.8% | 8.12 | 2.12 |

Reproduce (venv at `$HOME/envs/forkserve`; `vllm_fs` on `PYTHONPATH`, not `pip install -e`):

```bash
source "$HOME/envs/forkserve/bin/activate"
python -m pytest "$HOME/vllm_fs/tests/v1/core/test_forkserve.py" -q --noconftest
python "$HOME/vllm_fs/benchmarks/forkserve/compare_apc.py"
```

`--noconftest` skips vLLM’s repo-wide HF fixtures. These tests cover fork alias, extra-pin, adaptive tail, and $n_{\mathrm{valid}}$ row copy — not a compiled GPU `LLM.generate()`.

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

Control plane plus a vLLM V1 comparison substrate (`vllm_fs`):

- **One generate per phase** — open, CoW fan-out, and winner decode each map to one `LLM.generate`. `generate_committed_many` fuses commit-tails and drops speculative siblings.
- **In-tree CoW pool** — `KVCacheBlock` carries `ro` / `n_valid` / `is_speculative`. `ForkServeTracker` aliases the parent at `get_computed_blocks`, extra-pins owned frames so a parent `free` does not drop the trunk, and prefers speculative pages in the free queue. Pack-path CoW copies `n_valid` rows (`cow_copy_kv_rows`), not a full page.
- **Out-of-tree hook** — `install_vllm_cow()` still extra-pins full parent blocks on a stock engine.
- **Stock async scheduler by default** — two-class reorder is opt-in (`AsyncScheduler` subclass).

Out of tree: prefill/decode disaggregation; HBM $\leftrightarrow$ DRAM movement; freeze-tail attention slot map.

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

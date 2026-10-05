# Hash index

Control-plane architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md) §8 and §10. vLLM page contract: [`FORKSERVE_VLLM_DESIGN.md`](FORKSERVE_VLLM_DESIGN.md).

The hash index is the first stage of the per-residual decision. It is not a second system bolted onto the context tree. `fork` aliases by node identity and does not walk hashes. LCP commit publishes full pages of the winner. A later session that never forked from this one can then hash-hit those pages.

## What each index is for

| &nbsp; | vLLM Automatic Prefix Caching | ForkServe |
| --- | --- | --- |
| When sharing appears | After tokens exist | At harness `fork` (tokens may not exist yet) |
| What is shared | Full blocks with identical content | Trunk pages by refcount (CoW) |
| Best at | Cross-session / opportunistic reuse | Same-session fan-out + speculative prefill |
| Index key | `hash(parent, block_tokens, extra)` | Node id in a context tree |
| Eviction | LRU free-queue of cached blocks | Leaf-first (spec → idle → spine) |
| Partial block | Never cached | Residual pages OK |

A prefix cache used alone serializes a fan-out as independent requests, so the first expand clones the trunk until a later request publishes a hash. The context tree used alone cannot reuse a trunk published by a different session. Commit is the join: the session forks by reference, then publishes the committed full pages.

```
                    ┌─────────────────────────────┐
  harness fork ───► │ CoW page-table (node → pages)│ ◄── speculative leaves
                    │         ↕ incref / CoW       │
                    │  Physical page pool          │
                    │         ↕ publish on commit  │
  new HTTP req ───► │ Hash index (digest → page)   │ ◄── APC cross-session
                    └─────────────────────────────┘
```

## APC mechanics (from the vLLM design doc)

1. **Block key** = `hash(parent_hash, tokens_in_block, extra)` where `extra` can be LoRA id, multimodal image hash, or `cache_salt`.
2. **Only full blocks** enter the cache. A 3/4-full tail never hits.
3. **New request path**: `get_computed_blocks()` walks hashes → `touch` (incref, pull out of free queue) → allocate miss tail from free-queue head (evicting a cached block if needed).
4. **Free**: release blocks in reverse order (last block least reusable) onto the free-queue *tail*.
5. **Evict**: when allocating from a free-queue head that is still hashed, drop it from `cached_blocks`.
6. **Salt**: inject into the *first* block hash so tenants cannot time-attack each other's prefixes.
7. **Duplicates (v1)**: block tables are immutable; a second full block with the same hash may temporarily coexist until free.

## HashForkServe rules

1. Speculative pages are **never hashed** until LCP / `commit` promotes them (preserves Theorem 2 isolation).
2. `fork` is still O(1) aliasing — no hash walk on the critical path of fan-out.
3. After commit, full pages are published; a later `open` with the same token prefix APC-hits.
4. Eviction: prefer dropping speculative / unhashed free pages; among hashed free pages use APC LRU order.
5. `cache_salt` isolates tenants exactly as in APC.

## The rest of the decision

After the hash probe, one residual still has one action (`forkserve/prefill_prune.py`, [`ARCHITECTURE.md`](ARCHITECTURE.md) §10):

* A full hit is a retrieval. No kernel, and no transfer: the same blocks are already visible on the decode instance.
* A proper prefix prefills the miss tail only.
* A repeated loop (score $0.02 < \tau_{\mathrm{pre}}=0.15$) never starts prefill and is never inserted as a hash key.
* Illegal text scores $0.22$ and is prefilled. A low score that is not a loop is not a skip.
* The connector inserts only the miss tail of a residual that is kept. Stock disaggregation transfers the computed span, including the $k$-way trunk.

```
  residual ──► hash probe ──► loop? ──► early prefix ──► prefill miss tail
                  │              │            │
                  full hit       skip         stop
                  ▼              ▼            ▼
             stay local     not inserted   not inserted ──► kept tail may insert
```

Compare APC / ForkServe / hash_prefill / disagg_prefill / the cascade: `experiments/prefill_prune_bench.py`. Protocol: [`HASH_FORKSERVE_EXPERIMENT.md`](HASH_FORKSERVE_EXPERIMENT.md).

## Code

* `forkserve/hash_forkserve.py` — `hash_block`, `HashPageIndex`, `HashForkPool`, `HashForkServe`
* `forkserve/prefill_prune.py` — APP planner + `PrefillHashIndex`
* `forkserve/disagg.py` — P/D transfer gate
* `tests/test_hash_forkserve.py` — 8 unit tests
* `tests/test_prefill_prune.py` — APP + method compare
* `experiments/hash_forkserve_bench.py` — APC / CoW / hybrid microbench
* `experiments/prefill_prune_bench.py` — five-way prefill compare

## Microbench snapshot (page_size=16, control-plane)

| Mode | live pages | hash hits | fork aliases |
|---|---|---|---|
| APC only (40 sessions, shared trunk) | 96 | 624 | 0 |
| CoW fork only (10×4 fan-out) | 56 | — | 640 |
| HashForkServe (fan-out + 20 replays) | 56 | 484 | 640 |

Hybrid keeps CoW’s low live-page count on fan-out **and** APC’s cross-session hits on replay.

## APP vs APC / ForkServe (control-plane)

`PYTHONPATH=. python experiments/prefill_prune_bench.py` — 10 sessions × 4-way ToT, trunk 256, residual 32. Prefill ms from the token cost model.

| Method | phase | prefill tok | prefill ms | xfer tok | fan-out ms | peak KV | pruned |
|---|---|---:|---:|---:|---:|---:|---:|
| APC | fan-out | 11520 | 138.2 | 0 | 139.0 | 11520 | 0 |
| ForkServe | fan-out | 3840 | 46.1 | 0 | 47.1 | 3840 | 0 |
| hash_prefill | fan-out | 11520 | 138.2 | 0 | 139.0 | 11520 | 0 |
| disagg_prefill | fan-out | 11520 | 138.2 | 11520 | 162.1 | 11520 | 0 |
| Cascade | fan-out | **3232** | **38.8** | **672** | **40.3** | **2880** | 10 |

Against APC on this fan-out: prefill −71.9%, fan-out −71.0%, peak KV −75.0%. The ten prunes are the repeated loop, one per session. Illegal text is kept. Replay hash-skips the published winner (0 prefill tokens). The synthetic decode curve still reaches 84.5% at 256 tokens (APC) and 153 (cascade). Concurrency from the spine formula: P99 ≤ 1s QPS 192 → 256, tokens/s +35.1%. These rows are the token-cost model, not a GPU trace.

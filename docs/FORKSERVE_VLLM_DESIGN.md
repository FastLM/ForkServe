# ForkServe on vLLM: storage-efficient branch KV vs APC

Control-plane architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md). This is the vLLM mapping of ForkServe. The goal is not “APC plus a
`fork()` API”. APC is a **content-addressed** cache of *full* blocks
discovered *after* tokens exist. Agent fan-out needs a **location-addressed**
copy-on-write tree of pages that can be aliased *before* the residual
exists, including the unaligned trunk tail. That is the storage gap.

The current `install_vllm_cow()` hook extra-pins the parent’s *full*
blocks and falls back to APC for everything else. It is already better
than clone-on-fan-out, but it still pays APC’s partial-tail tax and
APC v1’s duplicate-block tax. This design closes those.

---

## 1. What APC actually stores

vLLM V1 prefix caching (`docs/design/prefix_caching.md`,
`vllm/v1/core/block_pool.py`, `kv_cache_manager.py`):

1. **Key** = `hash(parent_hash, tokens_in_block, extra)` (`extra` =
   LoRA / MM / `cache_salt`).
2. **Only full blocks** enter `cached_block_hash_to_block`.
   `get_computed_blocks` returns a prefix whose length is a multiple of
   `block_size` (default 16), capped at `num_tokens - 1`.
3. A miss tail is allocated from the free-queue head. If that head is
   still hashed, it is evicted (LRU).
4. **v1 does not de-duplicate.** `BlockHashToBlockMap` may map one hash
   to several physical blocks because block tables are append-only
   (NOTE #1 in the design doc). The duplicate lives until the request
   that owns it is freed.
5. A **partial prefix hit** (or a write into a shared last block)
   allocates a private destination and runs
   `copy_kv_cache_blocks_inplace`: `blocks[dst] = blocks[src]` — a
   **full scheduler block**, not the valid prefix of that block.
6. Sharing is discovered by walking hashes of the new prompt. There is
   no parent pointer. A child that has not been issued cannot alias
   anything.

APC is the right primitive for *unrelated* HTTP requests that happen to
share a system prompt. It is the wrong primitive for a ReAct / ToT /
LangGraph `Send` step, where the engine *already knows* the parent page
table.

### 1.1 Storage identity

Let $b$ be bytes of KV per token (one sequence, all layers), $P$
the scheduler block size, $L$ the trunk length, $k$ live children,
$\ell_i$ the residual of child $i$, $r = L \bmod P$ the unaligned
tail.

| Scheme | Live KV tokens (distinct physical rows) |
|---|---|
| Clone / APC miss on a new trunk | $kL + \sum \ell_i$ |
| APC hash-hit, aligned ($r=0$) | $L + \sum \ell_i$ |
| APC hash-hit, unaligned | $L + \sum \ell_i + (k-1)r$  (tail is private per child) |
| APC v1 with a duplicate full block | previous $+\;P$ per duplicate |
| APC partial-hit CoW copy | previous $+\;kP$  (full-block memcpy) |
| Extra-pin CoW (today’s hook) | $L - r + \sum (\ell_i + r)$  (full pages shared; tail cloned) |
| **ForkServe freeze-tail** | $L + \sum \ell_i$  (tail is one RO page with $n_\mathrm{valid}=r$) |

The aligned hash-hit case is already $M_\mathrm{CoW}$. ForkServe
beats APC when any of these hold: the trunk is not yet hashed (first
fan-out / same-batch siblings), $r \neq 0$, a duplicate was published,
a partial-hit copied a full block, or children are speculative and
must not wait for tokens.

Worked 8B bf16 (Llama-3-8B GQA: $b = 2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 128\,\mathrm{KiB/tok}$),
$L=8192$, $k=4$, $\ell_i=128$, $P=16$, $r=0$:

* clone = $4\cdot 8192 + 512 = 33280$ tok $\approx 4.06\,\mathrm{GiB}$
* APC hit / ForkServe = $8192+512=8704$ tok $\approx 1.06\,\mathrm{GiB}$
* ratio $\approx 3.82\times$

Same with $L=8191$ ($r=15$):

* APC hit = $8191 + 512 + 3\cdot 15 = 8748$ tok
* freeze-tail = $8191 + 512 = 8703$ tok
* small on one turn; $\Theta(k)$ in $r$ on a ToT tree of depth $d$
  (each level pays the tail tax again).

Same with APC *miss* (four specialists issued before any block is
hashed, typical first planner fan-out): APC pays the clone number;
ForkServe still pays $8704$.

---

## 2. Why the kernel is involved

PagedAttention / FlashAttention / FlashInfer address token $t$ as

```
block = block_table[t / P]
offset = t % P
```

Every block except the last in the sequence is assumed **full**. That
assumption is why APC refuses to cache a partial block, and why today’s
hook only aliases `parent_tokens // block_size` pages
(`select_full_blocks` in `forkserve/engine/vllm_loop.py`).

Freeze-tail puts a page with $n_{\mathrm{valid}}=r \lt P$ **in the
middle** of the child’s sequence (shared tail, then residual pages).
A stock kernel would read $P-r$ garbage rows as extra tokens and
shift the residual. That is a correctness bug, not a performance
footnote.

So the storage-efficient design has three layers that have to move
together:

1. **Physical pool** — refcount + RO bit + `n_valid` (OS page frame).
2. **Logical table** — parent alias + residual list, not a cloned
   `block_ids` vector of length $L/P$.
3. **Kernel slot map** — token $t$ is *not* always $t/P$ once a
   frozen page sits on the fork boundary.

Attention itself does **not** copy KV when two sequences share a
`block_id`. Sharing is free at the FA kernel as soon as the slot map
is right. The new kernels are the slot map and the partial CoW copy,
not a new attention algorithm.

---

## 3. Architecture (vLLM V1)

```
 harness  open / fork / speculate / commit / join / abort
                         │
          ┌──────────────┴──────────────┐
          │  BranchState  (per session) │  node → LogicalTable, mode, gen
          │  PrefillPlanner             │  p=1 suffixes before p<1 residuals
          │  TwoClassScheduler          │  B^c before B^s
          └──────────────┬──────────────┘
                         │
          ┌──────────────┴──────────────┐
          │  ForkBlockPool  extends     │
          │  vllm.v1.core.block_pool    │  ref, ro, n_valid, kind, hash
          │  KVCacheManager.fork_alias  │  no hash walk on fan-out
          │  APC index  (commit only)   │  cross-session reuse
          └──────────────┬──────────────┘
                         │
          worker: slot_mapping from ForkBlockTable
                  reshape_and_cache  (never writes RO pages)
                  paged attn         (fork-aware t → (block, off))
                  cow_copy_kv_rows   (n_valid rows, not P)
```

ForkServe **keeps** APC as a secondary index. Committed full pages are
published so a *different* session can hash-hit them. Speculative pages
are never hashed (output identity: Theorem 2 in the paper). Fan-out
inside a session never consults the hash on the critical path.

### 3.1 Physical page (`KVCacheBlock` extensions)

Add to `KVCacheBlock` / a side table on `BlockPool` (do not break the
free-queue intrusive pointers):

```
ro: bool              # CoW bit. Writes allocate a new page.
n_valid: int          # occupied token rows in [0, P]
kind: COMMIT | SPEC   # SPEC is invisible to the sampler and to APC
owner: NodeId | None
```

Invariants:

* `ref_cnt` = number of non-dead logical tables that map the page
  (same as today, but tables are nodes, not only requests).
* A write (`reshape_and_cache` slot, or decode append) that lands on
  `ro or ref_cnt > 1` **must not** store in place. The manager CoW-splits
  first.
* `n_valid < P` is legal for RO pages. Those pages are the freeze-tail.
* SPEC pages are allocated from the same pool but are the first
  eviction class (see §5).

`cache_full_blocks` gains a **dedupe** path that APC v1 explicitly
refused: if `cached_block_hash_to_block.get_one_block(h)` already
holds a COMMIT page with `n_valid == P`, the new page is freed and
the request’s last slot is rewritten to the existing `block_id`.
ForkServe owns the page table; it is not append-only across a CoW
boundary. This removes NOTE #1 duplicates.

### 3.2 Logical table (O(1) fork, not O(L/P) incref of a copied vector)

`LogicalTable` already exists in `forkserve/pages.py`:

```
σ_v = (parent, alias_len, residual[], residual_tokens)
```

`fork(u)` copies **two words** (parent pointer, `alias_len = |x_u|`)
and increfs the parent’s page *frames* in a tree-walk that can be
deferred: the parent’s `ref_cnt` is enough if we incref the table
object rather than every page. Two implementations, pick one:

| &nbsp; | Incref every page id | Incref the parent table |
| --- | --- | --- |
| fork CPU | $O(L/P)$ | $O(1)$ |
| abort | decref residual pages | decref table; cascade when table ref hits 0 |
| kernel flatten | walk chain | walk chain |

Use **table-level refcount**. Today’s hook copies the parent’s
`req_to_blocks` list and `touch`es every block — $O(L/P)$ and it
extra-pins even after the parent request is gone. Table-level ref
keeps the frames alive without a second pin list.

On a write into the shared tail:

```
new = alloc()
cow_copy_kv_rows(src, new, n_valid=src.n_valid)   # kernel
σ.residual = [new] + …   # child no longer aliases that frame
src.ref_cnt -= 1
```

The parent is unchanged. This is the POSIX CoW fault.

### 3.3 Freeze-tail (the APC gap)

On `fork(u)`:

1. Let $r = |x_u| \bmod P$. Full pages $[0, |x_u|//P)$ stay
   shared RO.
2. If $r > 0$, the last page is marked RO with `n_valid = r`.
   It is **not** filled by any child. Each child allocates a fresh
   residual page for tokens after $|x_u|$.
3. Internal fragmentation: the frozen page wastes $P-r$ slots,
   **once**. APC wastes $r$ valid rows **per child**.

Do **not** pad the prompt with dummy tokens to force alignment.
That changes positions and logits.

Allocated pages are not the same as valid rows. If $r+\ell_i \le P$
for every child, APC packs the unaligned tail and the residual into
**one** private page per child ($kP$ slots). Freeze-tail keeps a
frozen page **plus** $k$ residual pages ($(k+1)P$ slots) and can
lose. Typical agent wrappers are tens to thousands of tokens, so
$\ell_i > P-r$ and freeze wins. Use an **adaptive tail**:

```
pages_pack  = k * ceil((r + ℓ) / P)      # clone r into each child's first page
pages_freeze = 1 + k * ceil(ℓ / P)       # one RO tail + private residuals
if pages_freeze < pages_pack: freeze
else: cow_copy_kv_rows(src, child, n_valid=r)  # pack, copy r rows not P
```

Valid-row identity is still $L+\sum\ell_i$ only under freeze. Pack
pays $(k-1)r$ extra valid rows in exchange for fewer frames. The
kernel in §4.1 makes pack cheap (copy $r$ rows, not $P$).

### 3.4 Commit by LCP, then publish

`commit(u, x)` is a token procedure (the harness may have rewritten
the wrapper):

1. $v^\star = \arg\max_{v \in \mathrm{children}(u)} \mathrm{LCP}(x, x_v)$.
2. Pages of $v^\star$ covering the LCP become COMMIT. If LCP lands
   mid-page, `split_at` (copy `keep_valid` rows, drop the guessed
   tail).
3. Prefill the unmatched tail as committed, high-priority, chunked.
4. `abort` the other children of $u$ — decref residual frames only.
5. **Then** `publish_full` every full COMMIT page into the APC index
   (`hash_block` as in `forkserve/hash_forkserve.py`). Cross-session
   `open` uses `get_computed_blocks` unchanged.

Speculative KV never enters decode. Decode at a COMMIT node attends
only to `KV(x_u)` for the committed token sequence (paper Theorem 2).

---

## 4. Kernel plan

Three kernels. Attention math is unchanged.

### 4.1 `cow_copy_kv_rows` (replace full-block CoW)

Today (`vllm/v1/worker/utils.py`):

```python
blocks = cache.view(num_blocks, -1)
blocks[dst] = blocks[src]          # entire scheduler block, all layers' view
```

Needed:

```
cow_copy_kv_rows(kv_caches, src, dst, n_valid, layout, kernel_block_size)
```

* Copy only the first `n_valid` token rows of K and of V.
* Honour `KVCacheLayout` (NHD / HND), FlashAttention vs FlashInfer
  packing, MLA compressed KV, and `kernel_blocks_per_block` when the
  scheduler block is a multiple of the kernel block.
* Cross-layer shared storage: copy each `data_ptr` once (keep the
  existing `seen` / `copied_storages` logic).
* Launch as a short elementwise / `memcpy_async` grid; `n_valid=P`
  degenerates to the current path.

This is the win on every APC-style partial hit even without freeze-tail:
a 1-token extra write no longer clones 16 tokens × all layers.

File: `vllm_fs/csrc/cache_kernels.cu` (or a sibling `cow_copy.cu`) plus
a Python wrapper next to `copy_kv_cache_blocks_inplace`. Wire
`single_type_kv_cache_manager._apply_cow` to pass `n_valid`.

### 4.2 Fork-aware slot mapping (required for freeze-tail)

Per sequence, a compact descriptor (fits in the existing
`BlockTable` CPU buffer as extra columns, or a small side tensor):

```
n_shared      # number of full shared blocks
shared_ptr    # into a trunk block-id array (may be reused across siblings)
frozen_id     # -1 if r = 0
frozen_valid  # r
n_residual
residual_ptr
```

Token $t$ (0-based in the sequence):

```
if t < n_shared * P:           return shared[t / P], t % P
t -= n_shared * P
if frozen_id >= 0:
    if t < frozen_valid:       return frozen_id, t
    t -= frozen_valid
return residual[t / P], t % P
```

Two implementation tiers:

**Tier A — flatten on the host (no FA fork).** Before each attention
launch, write a *virtual* flat `block_table` for the child. A frozen
page cannot sit in the middle of a flat table, so Tier A **cannot**
implement freeze-tail. It can still share full pages (today’s hook)
and use `cow_copy_kv_rows` for the private tail. Ship this first.

**Tier B — kernel walks the descriptor.** FlashAttention-2/3 and
FlashInfer paged decode currently take `block_table[seq][t/P]`. Add
an optional path:

* `block_table` is the residual table only;
* `trunk_table` is a batch-shared tensor of shape `[n_trunks, max_trunk_blocks]`;
* `trunk_index[seq]` selects which trunk;
* `frozen_id[seq]`, `frozen_valid[seq]`.

The kernel’s block-id load becomes the piecewise map above. Decode
is memory-bandwidth bound; one extra integer compare per block is
noise relative to KV traffic. Prefill (chunked) uses the same map
for the KV side.

Fallback if a backend cannot take the descriptor (xpu, some MLA
paths): flatten by **copying the frozen rows into the first residual
page** with `cow_copy_kv_rows` (Tier A tax on that backend only).

`reshape_and_cache` / `append_paged_kv` already take a per-token
`slot_mapping`. Build that mapping from the descriptor so **new**
tokens never land on a RO page. This is mandatory even on Tier A:
it is the CoW fault.

### 4.3 `reshape_and_cache` RO guard

If a slot’s block has `ro=1`, the kernel should assert in debug and
skip in prod (the manager is supposed to remap first). Cheaper than
silently corrupting a sibling’s trunk.

### 4.4 What we do **not** write

* No new attention score kernel. PagedAttention stays.
* No token-level page size in the FA inner loop. Residuals stay
  size $P$; fragmentation of short residuals is $\lt P$ per child,
  which is the same as APC.
* No mixed page sizes in v1. A later optimisation (sub-page residual
  $P_r=4$) needs every backend to advertise a second kernel block
  size; not required to beat APC.

---

## 5. Eviction, offload, speculation budget

APC’s free queue is LRU over blocks. A tool pause with `ref_cnt=0`
puts the trunk at the LRU head; the next allocate can evict it and
the next turn recomputes $L$ tokens. Continuum/MORI fight this at
session granularity.

ForkServe ranks **nodes**, not requests:

1. SPEC leaves (TTL on residual size, not $L+\ell$).
2. COMMIT idle leaves.
3. COMMIT spine / shared trunk. A trunk page’s TTL does not run while
   any child maps it (`ref_cnt > 0` or table-level ref).

Offload (MORI-style) moves residual frames CPU-side first. Trunk
pages stay in HBM while any live child needs them.

Speculative admission (paper Algorithm 1) is what keeps ForkServe
from *losing* to APC on storage: unbounded multi-child observation
guesses allocate $\sum |x^o|$ that APC never would. Hard caps:

* `M_free` residual HBM;
* idle horizon $T_\mathrm{idle}+\gamma$;
* `B^s_t = 0` under saturation.

A miss aborts at residual cost. That is the only reason multi-child
speculation is rational.

---

## 6. Comparison

### 6.1 Serving systems (paper Table 1, storage column added)

| System | How KV is shared | Fan-out storage | Partial tail | Spec pages | Idle use |
|---|---|---|---|---|---|
| vLLM APC | hash of full blocks, after tokens | $L+\sum\ell_i$ on hit; $kL+\sum\ell_i$ on miss; duplicates until free | cloned per child | n/a | none |
| SGLang Radix | token-identity radix, after tokens | same as APC hit once inserted | radix node = full page | n/a | none |
| Continuum / MORI | retain/offload current session | 1 trunk, no siblings | n/a | n/a | retain / offload |
| Sutradhara | linear tool-independent prefix | 1 continuation | suffix on critical path | no tree | overlap one prefix |
| ForkKV | CoW LoRA residual vs base | adapters, not control flow | n/a | n/a | n/a |
| Extra-pin hook (today) | parent snapshot + `touch` | full pages shared | cloned (`select_full_blocks`) | not hashed | two-class opt-in |
| **ForkServe** | node-identity CoW tree, before tokens | $L+\sum\ell_i$ | one RO freeze | residual only, evicted first | spec prefill of $x^k$ |

### 6.2 APC vs ForkServe, mechanism by mechanism

| APC behaviour (vLLM V1) | ForkServe |
|---|---|
| Share after `hash(parent, toks, extra)` | Share at `fork` by parent pointer |
| Full blocks only | Full pages + freeze-tail with `n_valid` |
| Append-only block table → duplicate full blocks | Dedupe on publish; rewrite last slot |
| Partial hit: full-block `blocks[dst]=blocks[src]` | `cow_copy_kv_rows(..., n_valid)` |
| Evict LRU free-queue head | Evict SPEC leaves, then idle, then spine |
| Child is an independent `Request` | Child is a node; `Request` is a view |
| No unissued child | Speculative residual pages, never sampled |
| Cross-session reuse | **Kept**: publish COMMIT pages into the same hash index |
| `cache_salt` tenant isolation | Same salt on first published block |

They compose. HashForkServe (`forkserve/hash_forkserve.py`) is the
control-plane prototype of that composition. This document is the
worker/kernel contract that makes the HBM numbers match the prototype.

### 6.3 Measured control-plane / bench snapshots (already in-tree)

Hash microbench (`docs/HASH_FORKSERVE.md`, page_size=16):

| Mode | live pages | hash hits | fork aliases |
|---|---|---|---|
| APC only (40 sessions, shared trunk) | 96 | 624 | 0 |
| CoW fork only (10×4 fan-out) | 56 | — | 640 |
| HashForkServe (fan-out + 20 replays) | 56 | 484 | 640 |

GPU peak KV vs APC (`logs/rl_improve/NEXT.md`, tp=2, extra-pin hook,
not yet freeze-tail):

| Workload | ForkServe peak KV | vLLM APC | vLLM recompute |
|---|---|---|---|
| Game24 | 390 | 638 | 1508 |
| GSM8K | 459 | 659 | 1844 |
| HumanEval | 695 | 739 | 1406 |

The remaining APC gap on HumanEval is consistent with a short trunk
and a large residual (CoW helps less) plus the unaligned-tail clone
the hook still pays. Freeze-tail and dedupe are the next HBM cuts;
they should not change quality (same tokens, same KV after LCP).

---

## 7. Implementation map (`vllm_fs` + `ForkServe`)

Ship in this order. Each step is independently testable and each
step is a strict storage improvement over the previous.

| Step | Change | Beats APC by | Kernel? |
|---|---|---|---|
| 0 | Keep extra-pin hook + APC publish (status quo) | aligned fan-out vs miss/clone | no |
| 1 | `cow_copy_kv_rows`; `_apply_cow` passes `n_valid` | partial-hit copies | yes |
| 2 | Dedupe in `cache_full_blocks` / `publish_full` | NOTE #1 duplicates | no |
| 3 | `LogicalTable` in `KVCacheManager`; `fork_alias` skips hash; table-level ref (drop extra-pin list) | $O(1)$ fork metadata; no double pin | no |
| 4 | Freeze-tail + fork-aware slot map (Tier B) or copy-rows fallback (Tier A) | $(k-1)r$ tokens | yes for Tier B |
| 5 | SPEC kind + leaf-first free-queue class | speculative HBM ≤ `M_free` | no |
| 6 | Two-class `AsyncScheduler` default-on when mixed batches exist | not storage; TBT isolation | no |

Files:

* `vllm/v1/core/block_pool.py` — RO, `n_valid`, kind, dedupe, SPEC LRU class.
* `vllm/v1/core/kv_cache_manager.py` — `fork_alias`, LCP `split_at`, publish on commit.
* `vllm/v1/core/single_type_kv_cache_manager.py` — `_apply_cow` with `n_valid`; do not copy shared full pages.
* `vllm/v1/worker/utils.py` — `cow_copy_kv_rows`.
* `vllm/v1/worker/block_table.py` — descriptor columns; flatten or pass through.
* `vllm/v1/attention/backends/*` — optional descriptor path (FlashInfer first: it already has `kv_last_page_len` for the *last* page; extend to a mid-sequence frozen page).
* `csrc/cache*.cu` — row-level copy; RO assert in `reshape_and_cache`.
* `forkserve/engine/vllm_loop.py` — stop extra-pinning full tables once step 3 lands; keep `extra_args` (`forkserve_node`, `forkserve_parent_node`, session).
* Tests: `ForkServe/tests/test_pages.py`, `test_hash_forkserve.py`, plus new
  `vllm_fs/tests/v1/core/prefix_cache/test_fork_cow.py` (fan-out live pages,
  freeze-tail, dedupe, LCP split).

Degrade path, same as the paper: if CoW is off, `fork` is APC
`get_computed_blocks`; if the harness never forks, the tree is one
node and we are Continuum+APC.

---

## 8. Correctness (do not regress APC)

1. **Output identity.** Committed decode tokens match a baseline that
   prefills each committed prompt from scratch (or from a prefix cache
   of committed tokens only), equal seeds and weights. Speculative
   nodes never enter the sampler.
2. **No write into a shared frame.** RO + `ref_cnt>1` is a hard fault
   in the cache kernel.
3. **LCP mid-page.** `split_at` copies `keep_valid` rows; guessed
   tokens after the split are not attended.
4. **Salt.** Published hashes inject `cache_salt` into block 0 exactly
   as APC, so tenants cannot time-attack across salts.
5. **Last-token recompute.** When a child prompt is an exact prefix
   hit, still recompute the last token for logits (`max_cache_hit_length
   = num_tokens - 1`), same as APC.

---

## 9. What this is not

* Not speculative *decoding* (Medusa/EAGLE/SPORK). Those draft output
  tokens of the current step. ForkServe caches **input** prefixes of
  the next branch.
* Not “always cheaper than APC”. A single linear continuation with a
  block-aligned, already-hashed trunk is the same HBM as APC. The
  wins are fan-out, unaligned tails, first-fill, duplicates, and
  abort. Speculation is a win on TTFT and a *loss* on HBM unless
  Algorithm 1 caps it.
* Not a replacement for APC cross-session hits. Publish on commit.

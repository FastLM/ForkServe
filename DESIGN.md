# ForkServe advanced design

This note maps the paper onto the code so a systems reader can audit invariants without rereading every section.

## 1. Object model

A session is a rooted tree `T = (V, E)` (`ContextTree`). Each node `u` stores:

| Field | Paper | Notes |
|---|---|---|
| `tokens` | `x_u` | full sequence; prefix of parent is an invariant |
| `residual` | `δ_u` | `x_u[|x_π(u)|:]` |
| `mode` | `ρ_u ∈ {Spec, Commit, Idle, Dead}` | Spec is invisible to the sampler |
| `generation` | `g_u` | commit increments; scheduler drops stale `R^s` in O(1) |
| `table` | `σ_u = (off_u, ρ_u)` | parent alias + private residual page list |

`fork(u, bid, x^k)` copies O(1) words and increfs aliased pages. Materializing `x^k` is optional; the planner may do it later.

## 2. Copy-on-write pages

A physical page is `(id, ref, ro, owner)`.

- **Read.** Attention walks the node's table. Shared trunk pages are concurrent reads. Trunk pages are pinned `ro` so a buggy write cannot pollute siblings.
- **Write.** Decode / residual prefill: if `ref > 1` or `ro`, allocate a fresh page and copy only the valid tail rows (`PagePool.cow_if_needed`).
- **LCP mid-page.** `split_at` keeps `keep_valid` rows for the winner and leaves the speculative tail on the old page if any sibling still needs it (usually they abort).

Memory identity, Equation (3):

```
M_CoW   = b (L + Σ ℓ_i)
M_clone = b (k L + Σ ℓ_i)
```

Abort cost is Lemma 3: `b · P · ⌈ℓ/P⌉`, independent of trunk length `L`. `ContextTree.abort` is post-order so children drop their alias increfs before the parent frees residual pages.

## 3. Speculative prefill planner

Candidates `B(u)` come from three sources (`planner.py`):

1. **Harness-declared** forks (zero-ML, the common path).
2. **Grammar mass** from constrained decoding, top-`m` ≤ 3; leftover mass collapses to an `"other"` generic wrapper.
3. **Count-min prior** of `(parent_role, bid)`. Session-local, decaying. Never invents a branch the harness did not declare.

Each candidate is `(x^k, x̂^o)`. Observation residuals are admitted only when `q_b ≥ q_min` **and** the tool is schema-stable (JSON / XML / unified diff / typed return). We never draft `x̂^o` from an LLM.

Value / cost (Equations 4–5):

```
G(w) = p(w) · min(T_pre(|w|), T_idle) · η(w)
C(w) = b|w| + λ · max(0, T_pre(|w|) − γ)
```

`λ` default: 1 ms TBT regression costs as much as a 4 ms TTFT win.

Algorithm 1 is a greedy knapsack with three hard filters: idle horizon, residual HBM, committed TBT margin. Known suffixes are sorted first (Proposition 1). Admitted work is chunked at `C_spec=512` so a commit arriving mid-prefill loses at most one chunk.

## 4. Commit by longest common prefix

When the harness presents realized prompt `x` of parent `u`:

1. `v* = arg max_v LCP(x, x_v)` among live children. Short-circuit on harness `bid` when it matches (Appendix B).
2. Pages of `v*` covering `x[0:ℓ]` become committed; `x_v[ℓ:]` is dropped (CoW residual).
3. Prefill `x[ℓ:]` as a committed, high-priority chunked prefill. Skip if `ℓ = |x|`.
4. Abort every other speculative child of `u`.
5. Advance `v*` to Commit and release decode.

Theorem 2 (output identity): decode at a committed node attends only to `KV(x_u)`. That KV is produced by a committed prefill or by a speculative prefix that LCP verified token-for-token. Speculative nodes never enter the sampler.

## 5. Two-class scheduler

Each tick with budget `B_t`:

```
B^c_t = min(B_t, Σ_r max(s_r Δ, 1))
B^s_t = B_t − B^c_t
```

If `B^s_t = 0`, no speculative chunk is admitted (principle 4). Speculative jobs are ordered by `G/C`, preemptible at chunk boundaries, and cancelled when `g(w) ≠ g_u`. Committed jobs use program-FCFS with a PLAS aging term (`attained_s`). Speculative tokens are VTC-billed at `κ=0.25`.

## 6. Retention and placement

- **TTL** (Equation 1) is Continuum's estimator on the **node residual**, not the session. Trunk pages do not expire while any child is live.
- **Relative offload** (Equation 2) ranks **leaves**: spec > committed-idle > spine. Offload drops speculation first, then dead-end recoveries, then long tools, never the trunk of a live fan-out.
- **Router** pins the tree to the worker that prefills the root. Residual-only steal is allowed for speculative children; commit always returns to the trunk owner.

## 7. Harness adapters

Adapters announce structure; they do not rewrite prompts.

| Harness | Lowering |
|---|---|
| ReAct | `fork(wrap)` + `fork(recovery)` at tool-parse; `commit(wrap+obs)` |
| LangGraph `Send` | `fork` per role system prompt; `join` with a speculative scaffold |
| OpenHands / SWE | same as ReAct; residual n-gram only for schema-stable tools |
| ToT | `fork` of `b` thought prefixes; `abort` losers; `join(WINNER)` |

Streaming JSON/XML scan (`engine/scanner.py`) emits `fork` as soon as a complete tool-call object is parsed, overlapping remaining parent decode (charged as committed) with known-suffix prefill.

Security: adapters fork wrappers with placeholders and splice tool arguments as a second residual, so a cancelled speculation does not retain secrets longer than a committed pause would.

## 8. Degradation

| Failure | Fallback |
|---|---|
| Planner crash | `R^c_t` on live CoW trees |
| CoW disabled | radix sharing + Sutradhara-style linear split |
| Harness never forks | Continuum + MORI on a one-node tree |
| Saturation | `B^s_t = 0`, retention + CoW fan-out only |

## 9. vLLM seam

`VllmBackend` talks to public `LLM` / `TokensPrompt`. Production deployment still needs the in-engine patch described in §8 of the paper (block-manager CoW bit, two-class scheduler inside the engine loop, tokenizer-side LCP). This repository implements that control plane in-process so the algorithms are testable without vendoring vLLM.

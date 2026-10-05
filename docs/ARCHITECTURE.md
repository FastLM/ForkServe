# ForkServe Architecture

Control-plane specification. The vLLM page contract is [`FORKSERVE_VLLM_DESIGN.md`](FORKSERVE_VLLM_DESIGN.md). The hash index and the control-plane measurements are [`HASH_FORKSERVE.md`](HASH_FORKSERVE.md).

The serving object is the branch. Its cost is paid for the residual that is committed, and only at the cheapest stage where that rejection is still correct. Three mechanisms implement that decision: a copy-on-write context tree (§2–§4), one action per residual (§10), and the committed spine on the critical path (§5).

## 1. Problem

An agentic step is a tree of token sequences, not a linear prompt. A trunk $x$ of length $L$ is shared by $k$ children with residuals $\ell_i$. After a winner $\star$ is bound, only $L+\ell_\star$ remains live.

| Scheme | When sharing is discovered | Live KV (distinct rows) |
|---|---|---|
| Recompute | never | $kL+\sum_i\ell_i$ |
| APC / radix | after tokens exist; full blocks only | $L+\sum_i\ell_i$ on hit; $kL+\sum_i\ell_i$ on first fan-out |
| ForkServe | at `fork`, before the residual exists | $L+\sum_i\ell_i$ in flight; $L+\ell_\star$ after abort |

Prefill work matches APC on a warmed, aligned trunk. Storage after abort does not: APC retains hashed residuals until LRU eviction; ForkServe decrefs them.

Speculative KV is an **input-side** cache of the *next* branch. It is never sampled and never fed back. `commit` is the only verb on the critical path of user-visible tokens.

## 2. Model

Per session $s$, a rooted tree $T_s=(V,E)$. Node $v\in V$:

| Symbol | Meaning |
|---|---|
| $x_v$ | token sequence at $v$ |
| $\pi(v)$ | parent; $\pi(r)=\bot$ for root $r$ |
| $\rho_v$ | residual: $x_v[\,\lvert x_{\pi(v)}\rvert\,:]$ |
| $m_v$ | mode $\in\{\mathrm{Spec},\mathrm{Commit},\mathrm{Idle},\mathrm{Dead}\}$ |
| $g_v$ | generation; incremented on commit at $v$ |
| $\sigma_v$ | logical page table (parent alias + private pages) |
| $b$ | bytes of KV per token (all layers, one sequence) |
| $P$ | page size (tokens) |
| $B_t$ | batched-token budget on tick $t$ |

**Invariants.**

1. **Prefix.** $\forall v\neq r:\; x_{\pi(v)}\preceq x_v$.
2. **Spine.** Exactly one committed child per node, except during an in-flight `join`.
3. **Isolation.** Spec nodes are invisible to the sampler. Decode at a Commit node attends only to $\mathrm{KV}(x_v)$ for the committed sequence.
4. **Liveness.** Physical-page `ref` equals the number of non-Dead nodes that map the page.
5. **Generation.** Commit at $u$ increments $g_u$. Speculative jobs tagged $g\neq g_u$ drop in $O(1)$.
6. **Cascade abort.** A Spec node with no live child is aborted. The walk stops at the committed spine or a parent that still has a live child.

Let $b$ be as above. Then

$$
M_{\mathrm{CoW}}=b\Bigl(L+\sum_i\ell_i\Bigr),\qquad
M_{\mathrm{clone}}=b\Bigl(kL+\sum_i\ell_i\Bigr),\qquad
M_{\mathrm{spine}}=b\bigl(L+\ell_\star\bigr).
$$

Abort and cascade-free cost $\Theta(\ell_i/P)$ pages, independent of $L$.

## 3. Layers

Three layers, one contract: the harness announces structure; the control plane admits, binds, and retains; the backend moves pages and tokens. Adapters do not rewrite prompts.

```mermaid
flowchart TB
  subgraph L3["L3 harness"]
    direction LR
    R[ReAct]
    T[ToT]
    G[LangGraph Send]
    H[OpenHands / SWE]
  end

  V["open · fork · speculate · commit · join · abort"]

  subgraph L2["L2 control plane"]
    direction TB
    subgraph CP["per-session objects"]
      PL["Planner  admit by G/C"]
      SC["Scheduler  Bᶜ before Bˢ"]
      CM["Commit  LCP bind"]
      JN["Join  trunk + scaffold"]
    end
    CT["Context tree T_s"]
    AUX["Router (tree-sticky) · Retention (node TTL)"]
    PL --> CT
    SC --> CT
    CM --> CT
    JN --> CT
    CT --> AUX
  end

  subgraph L1["L1 execution"]
    direction LR
    M["MockBackend"]
    VL["vLLM V1  CoW pin / page alias"]
  end

  L3 --> V --> L2 --> L1
```

| Layer | Owns | Does not own |
|---|---|---|
| L3 | branch ids, wrappers, join policy | sampling, page tables |
| L2 | $T_s$, admission, LCP, TTL, placement | kernels, weights |
| L1 | `prefill` / `decode` / `run_batch`, physical KV | branch identity |

`EngineBackend.prefill` is required to be bit-identical for a given $(\text{tokens},\;\text{parent KV})$.

## 4. State

### 4.1 Logical table

A physical page is $(\mathrm{id},\;\mathrm{ref},\;\mathrm{ro},\;n_{\mathrm{valid}})$. Writes never mutate $\mathrm{ref} > 1$ or $\mathrm{ro}=\mathrm{true}$; they CoW the tail.

$$
\sigma_v=(\mathrm{parent},\;\mathrm{alias\_len},\;\rho^{\mathrm{pages}}_v,\;\rho^{\mathrm{tok}}_v).
$$

`fork(u)` copies two words $(\mathrm{parent}\leftarrow\sigma_u,\;\mathrm{alias\_len}\leftarrow|x_u|)$ and increfs the aliased frames. Residual pages are allocated only when $x^k$ or a write is supplied. Mid-page LCP uses `split_at(keep_valid)`: siblings retain the speculative tail; the winner keeps the prefix rows.

```
session
 └── r          Commit   x_r = trunk          shared, refcounted, RO
      ├── v0    Spec     ρ = wrap_0           winner → Commit → decode
      ├── v1    Spec     ρ = wrap_1           abort (decref ρ only)
      └── v2    Spec     ρ = wrap_2           abort
```

### 4.2 Modes

| Mode | Sampler | Prefill | Eviction rank |
|---|---|---|---|
| Spec | no | slack only | 3 (first) |
| Idle | no | none | 2 |
| Commit | yes | $B^c$ | 0 if it has live children, else 1 |
| Dead | — | — | released |

## 5. Control

### 5.1 Admission

Work items are known suffixes ($p=1$, $\eta=1$) or observation residuals ($p=p_b q_b$, admitted only if $q_b\ge q_{\min}$ and the schema is stable). Known suffixes starve residuals. Sources: harness-declared wraps; constrained-decoding mass (top-$m$); session-local decaying count-min over $(\mathrm{parent\_role},\;b_{\mathrm{id}})$. The prior never invents a branch id. Observation tokens $\hat x^o$ are not drafted by the LLM.

$$
G(w)=p(w)\cdot\min(T_{\mathrm{pre}},T_{\mathrm{idle}})\cdot\eta(w),
\qquad
C(w)=b|w|+\lambda\max(0,T_{\mathrm{pre}}-\gamma).
$$

Greedy fractional knapsack on $G/C$, subject to idle horizon $T_{\mathrm{idle}}+\gamma$, residual HBM $M_{\mathrm{free}}$, branch cap $m$, and TBT margin when the parent is Commit. Chunks of size $\le C_{\mathrm{spec}}$. Under saturation the scheduler sets $B^s_t=0$; the planner must not fill that gap by admitting more.

### 5.2 Two-class batch

Each tick $t$:

$$
B_t = B^c_t + B^s_t,\qquad B^c_t \text{ first (decode, commit-tail, root prefill)},\qquad B^s_t=\max(0,B_t-B^c_t).
$$

Committed jobs: program-FCFS + PLAS aging. Speculative chunks: preemptible at $C_{\mathrm{spec}}$; dropped on generation mismatch. VTC bills spec at $\kappa=0.25$; TBT-miss victims at $1$.

```mermaid
flowchart LR
  Q_c["Qᶜ  decode / commit-tail / root"]
  Q_s["Qˢ  speculative chunks"]
  B["B_t"]
  Q_c -->|"fill Bᶜ"| B
  Q_s -->|"remainder Bˢ, else 0"| B
  B --> GPU["run_batch"]
  CM["commit at u"] -->|"g_u ← g_u+1"| Q_s
```

### 5.3 LCP commit

`commit(u, x)` is a token procedure, not a branch-id lookup. The harness may have rewritten the wrapper. If $x$ does not extend $x_u$, it is treated as a residual and concatenated.

```
1  v* ← child with preferred bid if it covers x_u, else arg max_v LCP(x, x_v)
   if none: fork a Commit continuation; ℓ ← |x_u|
2  truncate_residual(v*, ℓ); CoW-split the mid-page
3  append x[ℓ:] as committed (or skip prefill if empty)
4  abort every other Spec child of u
5  g_u ← g_u + 1; tip ← v*; m_u ← Commit
```

Decode uses only KV that matches the committed sequence (output identity). Hit class: known-suffix if $\ell\le|x_u|+|x^k|$; else observation residual.

### 5.4 Join

Attention is not a homomorphism of concatenation: $\mathrm{KV}(w)$ is not algebraic in the children's KV. `join` reuses the shared trunk plus a harness scaffold and prefills the unshared tail. Policies: `all`, `first`, `kofn`, `concat`, `winner`, `summary`. Winner/first abort non-chosen children; `all`/`concat` leave them until the harness aborts.

## 6. Retention and placement

TTL is on the residual, not the session:

$$
\tau(v)=\min\bigl(\tau_{\max},\;\alpha\,t_b+\beta(c_r+q_q)/p_g\bigr).
$$

Trunk pages do not expire while any child maps them. Offload rank (higher = more idle): Spec leaves $\succ$ Commit idle leaves $\succ$ committed spine. Relative idleness $\iota(v)=t_{\mathrm{since}}(v)/t_{\mathrm{since}}(s)$.

The worker that prefills the root owns the trunk (tree-sticky). Forks schedule there first. Speculative residuals may steal to another worker without shipping the trunk. Commit returns to the trunk owner.

## 7. Backend contract

```
prefill(tokens, parent KV) → wall-ms     bit-identical
decode(node, n) → tokens                 Commit nodes only
run_batch(Bᶜ prefills, Bᶜ decodes, Bˢ)
cancel_prefill(node)
```

vLLM mapping (present substrate, not the freeze-tail design):

- One `LLM.generate` per phase (open, CoW fan-out, winner decode). Covered trunk rows are dropped so fan-out is one generate.
- `install_vllm_cow()` extra-pins parent full blocks; children alias complete pages. Abort decrefs residuals.
- Two-class reorder is opt-in. A custom `scheduler_cls` forces vLLM onto the sync scheduler.

Out of tree: prefill/decode disaggregation; HBM $\leftrightarrow$ DRAM tensor movement; freeze-tail slot map ([`FORKSERVE_VLLM_DESIGN.md`](FORKSERVE_VLLM_DESIGN.md) §4).

## 8. Hash composition

APC indexes full blocks by $\mathrm{hash}(\mathrm{parent},\;\mathrm{block\_tokens},\;\mathrm{extra})$ after tokens exist. The context tree aliases by node identity before they exist. The two meet at commit: `fork` does not hash; LCP commit publishes full pages of the winner; a later `open` hash-hits. That index is the first stage of §10, not a second system. Spec pages are never hashed (invariant 3). `cache_salt` on block 0 is unchanged.

## 9. Interface

```
open(x) → (s, r)
fork(s, u, bid, xᵏ) → v          O(1) alias; mode Spec
speculate(s, v | {candidates})   planner → Qˢ
commit(s, u, x) → v*             LCP; abort losers
generate(s, n) → tokens          tip, Commit only
join(s, children, policy)
abort(s, v, lazy=)               residual pages; cascade
drain_aborts()                   reclaim lazy KV after decode
close(s)
```

Adapters (`ReAct`, `ToT`, `LangGraph Send`, `OpenHands`) lower harness structure onto these verbs. `Orchestrator.react_turn` forks wrap and recovery at the first parsed tool call, speculates over $T_{\mathrm{idle}}$, then commits $\mathrm{wrap}\Vert\mathrm{obs}$.

## 10. One action per residual

Copy-on-write places the trunk. It does not decide which residual enters a kernel, or which KV would cross a prefill–decode connector. `PrefillPruner` makes both decisions. The stages are ordered by cost. The next stage runs only when this one has not rejected the residual. Index 0 is always kept. Speculative pages stay unpublished until LCP commit, so invariant 3 is unchanged.

| Action | Condition | GPU work | Transfer |
|---|---|---|---|
| `HASH_SKIP` | Published full pages cover the residual | 0 | 0 |
| `HASH_PARTIAL` | A proper published prefix | Miss tail | That tail, if kept |
| `DRAFT_SKIP` | Repeated loop (score $0.02 < \tau_{\mathrm{pre}}=0.15$) | 0 | 0 |
| `EARLY_ABORT` | The first `early_prune_frac` already fails | That prefix, then stop | 0 |
| `PREFILL` | Otherwise, on the trunk aliased at `fork` | The residual | Miss tail, if kept and disagg is on |

Illegal text scores $0.22$ and is prefilled. A low score that is not a loop is not a draft skip. That residual is prefilled, decoded for a short probe, and extended to the budget only when the generated prefix clears the decode threshold ($\tau=0.45$). The prefill scorer and the decode scorer do not share weights.

`admit_mode` selects one rule. The others stay off.

| Mode | What is kept |
|---|---|
| `score` (default) | Winner, plus every residual at or above $\tau_{\mathrm{pre}}$ |
| `winner` | The known winner only |
| `top_m` | Winner plus the next `prefill_keep_m - 1` by $G/C$ |
| `alpha` | Winner plus the next $\lfloor\alpha k\rfloor - 1$ by score |

`fork` does not walk the hash index. LCP commit publishes full pages of the winner, which is what makes a later session a hash hit. A hash-local hit is not a transfer: those blocks are already visible on the decode instance.

Abort of a rejected residual is lazy: the node is marked dead off the TTFT path, and `generate` reclaims the pages after the winner batch. `PagePool.pointer_swap` increfs page ids. Fan-out does not memcpy the trunk. Slots follow the spine, $C=\lfloor H/M_{\mathrm{spine}}\rfloor$, so the same pool admits more sessions once occupancy is $L+\ell_\star$ rather than the aborted residuals.

Slack fill is the same rule on the next turn. Tokens freed by a dropped residual are a budget. A page-aligned prefix of the next known suffix ($p=1$) is published into that budget, and nothing longer.

Enable with `plus_config()` / `app_config()` or `--system forkserve_plus`. Measurements: `experiments/prefill_prune_bench.py`. Protocol: [`HASH_FORKSERVE_EXPERIMENT.md`](HASH_FORKSERVE_EXPERIMENT.md).

## 11. Defaults and modules

$P=16$, $C_{\mathrm{spec}}=512$, $\lambda$ such that $1\,\mathrm{ms}$ TBT $\equiv 4\,\mathrm{ms}$ TTFT, grammar top-$m\le 3$, branch cap $6$, $q_{\min}=0.35$, $\kappa=0.25$.

| Module | Role |
|---|---|
| `tree.ContextTree` / `Forest` | $T_s$, spine, cascade abort |
| `pages.PagePool` / `LogicalTable` | CoW frames, `split_at` |
| `planner.SpeculatePlanner` | $G/C$ knapsack |
| `scheduler.TwoClassScheduler` | $B^c\prec B^s$, generation drop |
| `commit.CommitProtocol` | LCP bind |
| `join.JoinExecutor` | trunk + scaffold |
| `retention.RetentionManager` | node TTL, leaf-first |
| `router.TreeStickyRouter` | pin root, residual steal |
| `hash_forkserve` | Publish committed full pages; `fork` does not hash |
| `prefill_prune.PrefillPruner` | One action per residual (§10) |
| `disagg.DisaggPrefillConnector` | Insert only a kept miss tail |
| `prune.BranchPruner` | Prefill score; loops fall under $\tau_{\mathrm{pre}}$, illegal text does not |
| `spec_pool` | extra slots from KV saving |
| `eval_plus` | fan-out microbench, token–acc, QPS |
| `engine.protocol.EngineBackend` | L1 |
| `api.Engine` | verbs |

"""Branch-aware speculative prefill planner (§6, Algorithm 1).

Known suffixes (p=1) starve observation residuals (p<1). We never draft
future thoughts (IdleSpec's job) and never fabricate x̂^o from an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Iterable, Sequence

from forkserve.config import ForkServeConfig
from forkserve.types import (
    BranchId,
    JobClass,
    NodeId,
    NodeMode,
    SchemaKind,
    SessionId,
    TokenSeq,
    WorkKind,
    as_tokens,
)


@dataclass(slots=True)
class Candidate:
    """One child of parent u: exact known suffix plus optional residual guess."""

    branch_id: BranchId
    node_id: NodeId | None
    known: TokenSeq
    residual_hat: TokenSeq = ()
    p_b: float = 1.0
    q_b: float = 0.0
    schema: SchemaKind = SchemaKind.FREEFORM
    declared: bool = True

    @property
    def schema_stable(self) -> bool:
        return self.schema is not SchemaKind.FREEFORM


@dataclass(slots=True)
class WorkItem:
    kind: WorkKind
    branch_id: BranchId
    node_id: NodeId | None
    tokens: TokenSeq
    p: float
    q: float
    eta: float
    finish_after_pause: bool = False

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass(slots=True)
class PrefillChunk:
    session: SessionId
    parent: NodeId
    node_id: NodeId | None
    branch_id: BranchId
    tokens: TokenSeq
    kind: WorkKind
    job_class: JobClass = JobClass.SPECULATIVE
    generation: int = 0
    preemptible: bool = True
    gain: float = 0.0
    cost: float = 0.0
    offset: int = 0  # already-prefilled prefix of this item


@dataclass(slots=True)
class BudgetResult:
    items: list[WorkItem] = field(default_factory=list)
    chunks: list[PrefillChunk] = field(default_factory=list)
    rejected: list[tuple[WorkItem, str]] = field(default_factory=list)
    t_used_ms: float = 0.0
    hbm_used: float = 0.0


class PrefillCostModel:
    """T_pre(ℓ) given current batching. Backend may replace this."""

    def __init__(self, config: ForkServeConfig) -> None:
        self.cfg = config

    def __call__(self, n_tokens: int) -> float:
        return self.cfg.prefill_ms(n_tokens)


def expected_gain(
    item: WorkItem,
    t_pre: float,
    t_idle: float,
) -> float:
    """Equation (4): G(w) = p(w) · min(T_pre, T_idle) · η(w)."""
    useful = min(t_pre, t_idle) if t_idle < inf else t_pre
    if t_pre > t_idle and t_idle > 0:
        # Partial chunks still help under chunked prefill.
        item.finish_after_pause = True
        frac = t_idle / t_pre
        return item.p * useful * item.eta * max(frac, 0.0)
    return item.p * useful * item.eta


def expected_cost(
    item: WorkItem,
    t_pre: float,
    gamma_ms: float,
    bytes_per_token: float,
    lambda_tbt: float,
) -> float:
    """Equation (5): C(w) = b|w| + λ max(0, T_pre − γ)."""
    hbm = bytes_per_token * len(item)
    interference = lambda_tbt * max(0.0, t_pre - gamma_ms)
    return hbm + interference


class SpeculatePlanner:
    """Greedy fractional knapsack of Algorithm 1. Known suffixes first."""

    def __init__(
        self,
        config: ForkServeConfig,
        cost_model: PrefillCostModel | None = None,
    ) -> None:
        self.cfg = config
        self.cost_model = cost_model or PrefillCostModel(config)

    def build_items(self, candidates: Sequence[Candidate]) -> list[WorkItem]:
        items: list[WorkItem] = []
        for cand in candidates:
            p_known = 1.0 if cand.declared else cand.p_b
            if cand.known:
                items.append(
                    WorkItem(
                        kind=WorkKind.KNOWN_SUFFIX,
                        branch_id=cand.branch_id,
                        node_id=cand.node_id,
                        tokens=as_tokens(cand.known),
                        p=p_known,
                        q=1.0,
                        eta=1.0,
                    )
                )
            if (
                cand.residual_hat
                and cand.q_b >= self.cfg.q_min
                and cand.schema_stable
            ):
                items.append(
                    WorkItem(
                        kind=WorkKind.OBS_RESIDUAL,
                        branch_id=cand.branch_id,
                        node_id=cand.node_id,
                        tokens=as_tokens(cand.residual_hat),
                        p=cand.p_b * cand.q_b,
                        q=cand.q_b,
                        eta=max(cand.q_b, self.cfg.eta_partial),
                    )
                )
        return items

    def allocate(
        self,
        session: SessionId,
        parent: NodeId,
        candidates: Sequence[Candidate],
        *,
        t_idle_ms: float,
        gamma_ms: float,
        m_free: float,
        parent_mode: NodeMode = NodeMode.IDLE,
        generation: int = 0,
    ) -> BudgetResult:
        """Algorithm 1. Filters: idle horizon, residual HBM, committed TBT."""
        raw = self.build_items(candidates)
        scored: list[tuple[float, WorkItem, float, float]] = []
        for w in raw:
            t_pre = self.cost_model(len(w))
            g = expected_gain(w, t_pre, t_idle_ms)
            c = expected_cost(
                w, t_pre, gamma_ms, self.cfg.bytes_per_token, self.cfg.lambda_tbt
            )
            ratio = g / c if c > 0 else 0.0
            scored.append((ratio, w, g, c))

        def _key(row: tuple[float, WorkItem, float, float]) -> tuple[int, float]:
            _, w, _, _ = row
            known = 1 if w.kind is WorkKind.KNOWN_SUFFIX else 0
            return (known, row[0])

        scored.sort(key=_key, reverse=True)

        result = BudgetResult()
        t = 0.0
        m = 0.0
        admitted = 0
        for ratio, w, g, c in scored:
            if admitted >= self.cfg.branch_cap_m and w.kind is WorkKind.OBS_RESIDUAL:
                result.rejected.append((w, "branch_cap"))
                continue
            t_pre = self.cost_model(len(w))
            if t + t_pre > t_idle_ms + gamma_ms:
                result.rejected.append((w, "idle_horizon"))
                continue
            hbm = self.cfg.bytes_per_token * len(w)
            if m + hbm > m_free:
                result.rejected.append((w, "hbm"))
                continue
            if t_pre > gamma_ms and parent_mode is NodeMode.COMMIT:
                result.rejected.append((w, "tbt_margin"))
                continue
            result.items.append(w)
            result.chunks.extend(
                self._chunk(session, parent, w, generation, g, c)
            )
            t += t_pre
            m += hbm
            admitted += 1
        result.t_used_ms = t
        result.hbm_used = m
        return result

    def _chunk(
        self,
        session: SessionId,
        parent: NodeId,
        item: WorkItem,
        generation: int,
        gain: float,
        cost: float,
    ) -> list[PrefillChunk]:
        cap = self.cfg.c_spec
        tokens = item.tokens
        chunks: list[PrefillChunk] = []
        off = 0
        while off < len(tokens):
            piece = tokens[off : off + cap]
            chunks.append(
                PrefillChunk(
                    session=session,
                    parent=parent,
                    node_id=item.node_id,
                    branch_id=item.branch_id,
                    tokens=piece,
                    kind=item.kind,
                    generation=generation,
                    gain=gain,
                    cost=cost,
                    offset=off,
                )
            )
            off += cap
        return chunks


# ---------------------------------------------------------------------------
# Probability sources (§6.1)
# ---------------------------------------------------------------------------


class CountMinPrior:
    """Session-local decaying count-min of (parent_role, bid). Never invents branches."""

    def __init__(self, width: int = 2048, depth: int = 4, decay: float = 0.97) -> None:
        self.width = width
        self.depth = depth
        self.decay = decay
        self.tables = [[0.0] * width for _ in range(depth)]
        self._salts = [1_000_003 + 97 * i for i in range(depth)]

    def _idx(self, key: str, d: int) -> int:
        return (hash((key, self._salts[d])) & 0x7FFFFFFF) % self.width

    def observe(self, parent_role: str, bid: str, weight: float = 1.0) -> None:
        key = f"{parent_role}\0{bid}"
        for d in range(self.depth):
            i = self._idx(key, d)
            self.tables[d][i] = self.tables[d][i] * self.decay + weight

    def mass(self, parent_role: str, bids: Sequence[str]) -> dict[str, float]:
        raw: dict[str, float] = {}
        for bid in bids:
            key = f"{parent_role}\0{bid}"
            raw[bid] = min(self.tables[d][self._idx(key, d)] for d in range(self.depth))
        total = sum(raw.values())
        if total <= 0:
            n = max(len(bids), 1)
            return {b: 1.0 / n for b in bids}
        return {b: v / total for b, v in raw.items()}


class NGramResidual:
    """Deterministic template + first-n tokens of a per-tool n-gram predictor."""

    def __init__(self, order: int = 3, prefix: int = 32) -> None:
        self.order = order
        self.prefix = prefix
        self.counts: dict[str, dict[TokenSeq, dict[int, int]]] = {}

    def observe(self, tool: str, tokens: TokenSeq) -> None:
        table = self.counts.setdefault(tool, {})
        toks = as_tokens(tokens)
        for i in range(len(toks)):
            ctx = toks[max(0, i - self.order) : i]
            nxt = table.setdefault(ctx, {})
            nxt[toks[i]] = nxt.get(toks[i], 0) + 1

    def predict(self, tool: str, seed: TokenSeq = ()) -> TokenSeq:
        table = self.counts.get(tool)
        if not table:
            return ()
        out = list(as_tokens(seed))
        for _ in range(self.prefix):
            ctx = tuple(out[-self.order :]) if out else ()
            dist = table.get(ctx) or table.get(())
            if not dist:
                break
            tok = max(dist.items(), key=lambda kv: kv[1])[0]
            out.append(tok)
        return tuple(out[len(seed) :])


def grammar_branches(
    masses: dict[str, float],
    wrappers: dict[str, TokenSeq],
    generic_wrapper: TokenSeq,
    top_m: int = 3,
) -> list[Candidate]:
    """Top-m tool wrappers from constrained-decoding mass; rest collapse to 'other'."""
    ranked = sorted(masses.items(), key=lambda kv: kv[1], reverse=True)
    out: list[Candidate] = []
    rest = 0.0
    for i, (name, p) in enumerate(ranked):
        if i < top_m and name in wrappers:
            out.append(
                Candidate(
                    branch_id=BranchId(name),
                    node_id=None,
                    known=as_tokens(wrappers[name]),
                    p_b=p,
                    q_b=1.0,
                    schema=SchemaKind.JSON,
                    declared=False,
                )
            )
        else:
            rest += p
    if rest > 0 and generic_wrapper:
        out.append(
            Candidate(
                branch_id=BranchId("other"),
                node_id=None,
                known=as_tokens(generic_wrapper),
                p_b=rest,
                q_b=1.0,
                schema=SchemaKind.JSON,
                declared=False,
            )
        )
    return out


def schema_kind_of(tool: str, mime: str | None = None) -> SchemaKind:
    t = (tool or "").lower()
    m = (mime or "").lower()
    if "json" in m or t in {"call", "function", "bfcl"}:
        return SchemaKind.JSON
    if "xml" in m or "tool_response" in t:
        return SchemaKind.XML
    if t in {"diff", "apply_patch", "edit"} or "diff" in m:
        return SchemaKind.UNIFIED_DIFF
    if t in {"bfcl", "typed"}:
        return SchemaKind.TYPED_RETURN
    return SchemaKind.FREEFORM

"""Branch-aware TTL and leaf-first relative offload (§5.4).

Continuum's TTL and MORI's relative idleness are lifted from sessions to
nodes. Trunk pages do not expire while any child is live. Speculative
leaves sort as more idle than committed idle leaves, which sort as more
idle than the committed spine.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from forkserve.config import ForkServeConfig
from forkserve.tree import ContextTree, Forest, Node
from forkserve.types import NodeId, NodeMode


@dataclass(slots=True)
class OffloadDecision:
    node_id: NodeId
    session: str
    score: float
    dest: str  # "dram" | "abort" | "keep"
    residual_tokens: int


def node_ttl_s(
    residual_tokens: int,
    t_b: float,
    c_r: float,
    q_q: float,
    p_g: float,
    cfg: ForkServeConfig,
) -> float:
    """Equation (1): Continuum estimator on the node's residual, not the session."""
    if p_g <= 0:
        p_g = 1.0
    est = cfg.ttl_alpha * t_b + cfg.ttl_beta * (c_r + q_q) / p_g
    # residual-size hint: larger residuals keep slightly longer (reload cost).
    est += 0.05 * (residual_tokens / max(cfg.page_size, 1))
    return min(cfg.tau_max_s, max(0.05, est))


def relative_idleness(node: Node, session_last_decode: float, now: float | None = None) -> float:
    """Equation (2): ι(u) = t_since_decode(u) / t_since_decode(session)."""
    now = now if now is not None else monotonic()
    num = max(now - node.last_decode_at, 1e-6)
    den = max(now - session_last_decode, 1e-6)
    return num / den


def leaf_rank_key(node: Node, session_last_decode: float, now: float | None = None) -> tuple:
    """Spec leaves > committed idle leaves > committed spine. Higher = more idle."""
    idle = relative_idleness(node, session_last_decode, now)
    if node.mode is NodeMode.SPEC:
        tier = 3
    elif node.mode is NodeMode.IDLE:
        tier = 2
    elif node.mode is NodeMode.COMMIT and node.children:
        tier = 0  # spine with live children: never first
    else:
        tier = 1
    return (tier, idle)


class RetentionManager:
    def __init__(self, forest: Forest, config: ForkServeConfig) -> None:
        self.forest = forest
        self.cfg = config

    def arm_idle(self, tree: ContextTree, nid: NodeId, *, tool_s: float, queue_s: float = 0.0) -> float:
        node = tree.get(nid)
        tau = node_ttl_s(
            residual_tokens=len(node.residual),
            t_b=tool_s,
            c_r=self.cfg.prefill_ms(len(node.residual)) / 1000.0,
            q_q=queue_s,
            p_g=1.0,
            cfg=self.cfg,
        )
        tree.mark_idle(nid, monotonic() + tau)
        return tau

    def expire(self, now: float | None = None) -> list[NodeId]:
        now = now if now is not None else monotonic()
        dead: list[NodeId] = []
        for tree in self.forest.live_trees():
            for node in list(tree.live_nodes()):
                if node.ttl_deadline is None:
                    continue
                if now < node.ttl_deadline:
                    continue
                if node.mode is NodeMode.SPEC:
                    tree.abort(node.id)
                    dead.append(node.id)
                elif node.mode is NodeMode.IDLE and not tree.children_of(node.id):
                    # Residual of an idle leaf may expire; trunk stays via children.
                    tree.abort(node.id)
                    dead.append(node.id)
        return dead

    def plan_offload(self, hbm_target_bytes: float | None = None) -> list[OffloadDecision]:
        """Leaf-first drop until live KV fits the HBM target."""
        target = hbm_target_bytes if hbm_target_bytes is not None else self.cfg.hbm_capacity_bytes
        now = monotonic()
        ranked: list[tuple[tuple, Node, ContextTree]] = []
        for tree in self.forest.live_trees():
            for node in tree.live_nodes():
                if node.parent is None:
                    continue  # never offload the root trunk as a first victim
                ranked.append((leaf_rank_key(node, tree.last_decode_at, now), node, tree))
        ranked.sort(key=lambda r: r[0], reverse=True)

        live = self.forest.footprint_bytes()
        decisions: list[OffloadDecision] = []
        for _key, node, tree in ranked:
            if live <= target:
                break
            dest = "abort" if node.mode is NodeMode.SPEC else "dram"
            # Never drop a trunk that still has live children.
            if node.mode is NodeMode.COMMIT and tree.children_of(node.id):
                decisions.append(
                    OffloadDecision(node.id, str(tree.session), _key[1], "keep", len(node.residual))
                )
                continue
            bytes_free = len(node.residual) * self.cfg.bytes_per_token
            if dest == "abort":
                tree.abort(node.id)
            else:
                tree.pool.offload(node.table.residual)
            live -= bytes_free
            decisions.append(
                OffloadDecision(node.id, str(tree.session), _key[1], dest, len(node.residual))
            )
        return decisions

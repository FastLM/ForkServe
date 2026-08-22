"""Join / context-engineering children (§5.3).

KV of the join node is not algebraic in the children's KV — attention is
not a homomorphism of concatenation. We reuse the shared trunk plus a
speculative join scaffold and prefill only the unshared tail.
"""

from __future__ import annotations

from dataclasses import dataclass

from forkserve.tree import ContextTree
from forkserve.types import (
    BranchId,
    JoinPolicy,
    NodeId,
    NodeMode,
    TokenSeq,
    as_tokens,
    lcp_len,
)


@dataclass(slots=True)
class JoinResult:
    node_id: NodeId
    reused_prefix: int
    tail: TokenSeq
    policy: JoinPolicy
    winners: list[NodeId]


class JoinExecutor:
    def apply(
        self,
        tree: ContextTree,
        children: list[NodeId],
        policy: JoinPolicy,
        *,
        scaffold: TokenSeq = (),
        blend: TokenSeq = (),
        k: int = 1,
        parent: NodeId | None = None,
    ) -> JoinResult:
        if not children:
            raise ValueError("join requires at least one child")
        nodes = [tree.get(c) for c in children]
        for n in nodes:
            n.join_open = True
            if n.mode is NodeMode.SPEC:
                n.mode = NodeMode.COMMIT

        parent_id = parent if parent is not None else nodes[0].parent
        if parent_id is None:
            raise ValueError("join parent missing")
        parent_node = tree.get(parent_id)

        if policy is JoinPolicy.FIRST_SUCCESS or policy is JoinPolicy.WINNER:
            chosen = nodes[:1]
        elif policy is JoinPolicy.K_OF_N:
            chosen = nodes[: max(1, k)]
        else:
            chosen = nodes

        # Token sequence of w is harness-defined. Default: trunk + scaffold + blend
        # (or concatenation of child residuals for CONCAT).
        if policy is JoinPolicy.CONCAT:
            body = tuple(t for n in chosen for t in n.residual)
        elif policy in (JoinPolicy.WINNER, JoinPolicy.FIRST_SUCCESS):
            body = chosen[0].residual
        else:
            body = as_tokens(blend) or tuple(t for n in chosen for t in n.residual)

        scaffold_t = as_tokens(scaffold)
        desired = parent_node.tokens + scaffold_t + body

        # Reuse whatever prefix w shares with some child (usually trunk + scaffold).
        best_n, best_l = chosen[0], len(parent_node.tokens)
        for n in chosen:
            L = lcp_len(desired, n.tokens)
            if L > best_l:
                best_n, best_l = n, L

        w = tree.fork(parent_id, BranchId("join"), (), mode=NodeMode.COMMIT)
        # Install reused child's pages by aliasing through a synthetic residual
        # only for the unshared tail.
        if best_l > len(parent_node.tokens):
            reused = desired[len(parent_node.tokens) : best_l]
            if reused:
                tree.append_tokens(w.id, reused, committed=True)
        tail = desired[best_l:]
        if tail:
            tree.append_tokens(w.id, tail, committed=True)

        # Close the fan-out: abort children that are not part of the join spine
        # only under winner/first; ALL/CONCAT keep them until the harness aborts.
        if policy in (JoinPolicy.FIRST_SUCCESS, JoinPolicy.WINNER):
            keep = {c.id for c in chosen} | {w.id}
            for n in nodes:
                if n.id not in keep:
                    tree.abort(n.id)
        for n in nodes:
            n.join_open = False
        tree.tip = w.id
        return JoinResult(
            node_id=w.id,
            reused_prefix=best_l,
            tail=tail,
            policy=policy,
            winners=[n.id for n in chosen],
        )

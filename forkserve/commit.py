"""Longest-common-prefix commit protocol (§6.4) and Theorem 2.

Commit is a token procedure, not a branch-id lookup. The harness may have
rewritten the wrapper. Speculative KV is an input-side cache: decode of
committed tokens uses only KV that matches the committed token sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

from forkserve.tree import ContextTree
from forkserve.types import (
    NodeId,
    NodeMode,
    TokenSeq,
    WorkKind,
    as_tokens,
    lcp_len,
)


@dataclass(slots=True)
class CommitResult:
    winner: NodeId
    lcp: int
    tail: TokenSeq
    aborted: list[NodeId]
    known_suffix_hit: bool
    residual_full_hit: bool
    residual_partial_hit: bool
    skipped_prefill: bool
    generation: int


class CommitProtocol:
    """Five steps of §6.4. Never changes model outputs (Theorem 2)."""

    def apply(
        self,
        tree: ContextTree,
        parent_id: NodeId,
        prompt: TokenSeq,
        *,
        preferred_bid: str | None = None,
    ) -> CommitResult:
        prompt = as_tokens(prompt)
        parent = tree.get(parent_id)
        # Appendix D passes wrap+obs (residual). §6.4 also allows the full x_u.
        if prompt[: len(parent.tokens)] != parent.tokens:
            prompt = parent.tokens + prompt

        winner, lcp = self._select(tree, parent_id, prompt, preferred_bid)
        if winner is None:
            # No speculative child: fork a committed continuation and prefill all.
            child = tree.fork(parent_id, preferred_bid or "commit", (), mode=NodeMode.COMMIT)
            winner_node = child
            lcp = len(parent.tokens)
        else:
            winner_node = winner

        # Step 2: keep pages covering x[0:ℓ]; drop the speculative tail.
        tree.truncate_residual(winner_node.id, lcp)
        tail = prompt[lcp:]
        if tail:
            tree.append_tokens(winner_node.id, tail, committed=True)
        else:
            tree.set_mode(winner_node.id, NodeMode.COMMIT)

        # Step 4: abort every other speculative child of u.
        aborted: list[NodeId] = []
        for sib in tree.children_of(parent_id):
            if sib.id != winner_node.id and sib.mode is NodeMode.SPEC:
                tree.abort(sib.id)
                aborted.append(sib.id)

        gen = tree.bump_generation(parent_id)
        tree.tip = winner_node.id
        tree.set_mode(parent_id, NodeMode.COMMIT)
        parent.join_open = False

        parent_len = len(parent.tokens)
        residual_lcp = max(0, lcp - parent_len)
        winner_known = 0
        # Reconstruct: residual that was present before truncate is gone;
        # counters on the winner record admitted known-suffix length if set.
        known_hit = lcp >= parent_len and residual_lcp > 0
        residual_full = lcp == len(prompt) and residual_lcp > winner_known
        residual_partial = known_hit and 0 < residual_lcp < (len(prompt) - parent_len)

        winner_node.counters.lcp_length = lcp
        winner_node.counters.known_suffix_hit = known_hit
        winner_node.counters.residual_full_hit = residual_full
        winner_node.counters.residual_partial_hit = residual_partial
        winner_node.counters.residual_tokens = len(winner_node.residual)

        return CommitResult(
            winner=winner_node.id,
            lcp=lcp,
            tail=tail,
            aborted=aborted,
            known_suffix_hit=known_hit,
            residual_full_hit=lcp == len(prompt),
            residual_partial_hit=residual_partial,
            skipped_prefill=len(tail) == 0,
            generation=gen,
        )

    def _select(
        self,
        tree: ContextTree,
        parent_id: NodeId,
        prompt: TokenSeq,
        preferred_bid: str | None,
    ) -> tuple:
        # Short-circuit on harness branch id when it matches (Appendix B).
        if preferred_bid is not None:
            for child in tree.children_of(parent_id):
                if child.branch_id == preferred_bid:
                    n = lcp_len(prompt, child.tokens)
                    if n >= len(tree.get(parent_id).tokens):
                        return child, n
        return tree.best_lcp_child(parent_id, prompt)


def classify_hit(result: CommitResult, known_len: int, residual_len: int) -> WorkKind | None:
    if result.lcp <= 0:
        return None
    if result.lcp <= known_len:
        return WorkKind.KNOWN_SUFFIX
    if residual_len > 0:
        return WorkKind.OBS_RESIDUAL
    return WorkKind.KNOWN_SUFFIX

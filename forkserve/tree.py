"""Forkable context tree (Definition 1, §5).

Invariants
----------
* Prefix: ∀u ≠ r : x_π(u) ⪯ x_u
* Single committed spine, except during an in-flight join
* Spec isolation: Spec nodes are invisible to sampling
* Liveness: page refcount = # of non-Dead nodes mapping the page
* Generation: commit increments g_u; stale speculative jobs drop in O(1)
* Cascade abort: a Spec node with no live children is aborted; walk stops at
  the committed spine or a parent that still has a live child (Figure 3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Iterable, Iterator

from forkserve.config import ForkServeConfig
from forkserve.pages import LogicalTable, PagePool, pages_for_tokens
from forkserve.types import (
    BranchId,
    Counters,
    NodeId,
    NodeMode,
    SessionId,
    TokenSeq,
    as_tokens,
    lcp_len,
)


@dataclass(slots=True)
class Node:
    id: NodeId
    session: SessionId
    parent: NodeId | None
    branch_id: BranchId
    tokens: TokenSeq
    residual: TokenSeq
    mode: NodeMode
    generation: int
    table: LogicalTable
    created_at: float
    last_decode_at: float
    ttl_deadline: float | None = None
    worker: int = 0
    tenant: str = "default"
    children: list[NodeId] = field(default_factory=list)
    counters: Counters = field(default_factory=Counters)
    join_open: bool = False

    @property
    def trunk_len(self) -> int:
        return len(self.tokens) - len(self.residual)

    def is_live(self) -> bool:
        return self.mode is not NodeMode.DEAD


class ContextTree:
    """Per-session rooted tree T = (V, E)."""

    def __init__(
        self,
        session: SessionId,
        pool: PagePool,
        config: ForkServeConfig,
        *,
        tenant: str = "default",
        worker: int = 0,
    ) -> None:
        self.session = session
        self.pool = pool
        self.cfg = config
        self.tenant = tenant
        self.worker = worker
        self._nodes: dict[NodeId, Node] = {}
        self._next_id = 1
        self.root: NodeId | None = None
        self.tip: NodeId | None = None
        self.opened_at = monotonic()
        self.last_decode_at = self.opened_at

    def __contains__(self, nid: NodeId) -> bool:
        return nid in self._nodes

    def __len__(self) -> int:
        return sum(1 for n in self._nodes.values() if n.is_live())

    def get(self, nid: NodeId) -> Node:
        return self._nodes[nid]

    def live_nodes(self) -> Iterator[Node]:
        for n in self._nodes.values():
            if n.is_live():
                yield n

    def children_of(self, nid: NodeId, *, live_only: bool = True) -> list[Node]:
        node = self._nodes[nid]
        out = [self._nodes[c] for c in node.children]
        if live_only:
            out = [c for c in out if c.is_live()]
        return out

    def ancestors(self, nid: NodeId) -> list[Node]:
        walk: list[Node] = []
        cur: Node | None = self._nodes[nid]
        while cur is not None:
            walk.append(cur)
            cur = self._nodes[cur.parent] if cur.parent is not None else None
        walk.reverse()
        return walk

    def committed_spine(self) -> list[Node]:
        if self.root is None:
            return []
        spine = [self._nodes[self.root]]
        while True:
            kids = [c for c in self.children_of(spine[-1].id) if c.mode is NodeMode.COMMIT]
            if not kids:
                break
            if len(kids) > 1 and not any(k.join_open for k in kids):
                raise InvariantError(f"multiple committed children at {spine[-1].id}")
            spine.append(kids[0] if not any(k.join_open for k in kids) else kids[-1])
        return spine

    def open_root(self, tokens: TokenSeq, *, branch_id: str = "root") -> Node:
        tokens = as_tokens(tokens)
        nid = NodeId(self._next_id)
        self._next_id += 1
        table = LogicalTable(parent=None, alias_len=0)
        node = Node(
            id=nid,
            session=self.session,
            parent=None,
            branch_id=BranchId(branch_id),
            tokens=tokens,
            residual=tokens,
            mode=NodeMode.COMMIT,
            generation=0,
            table=table,
            created_at=monotonic(),
            last_decode_at=monotonic(),
            worker=self.worker,
            tenant=self.tenant,
        )
        self._materialize_residual(node, tokens)
        self.pool.pin_readonly(node.table.residual)
        self._nodes[nid] = node
        self.root = nid
        self.tip = nid
        return node

    def fork(
        self,
        parent_id: NodeId,
        branch_id: BranchId | str,
        known_suffix: TokenSeq | None = None,
        *,
        mode: NodeMode = NodeMode.SPEC,
    ) -> Node:
        """O(1) in the trunk. Materializes x^k only if the caller supplies it."""
        parent = self._nodes[parent_id]
        if not parent.is_live():
            raise InvariantError(f"fork from dead node {parent_id}")
        known = as_tokens(known_suffix or ())
        nid = NodeId(self._next_id)
        self._next_id += 1
        table = LogicalTable(
            parent=parent.table,
            alias_len=len(parent.tokens),
        )
        # Alias: incref every page the parent currently maps.
        for pid in self.pool.alias_pages(parent.table):
            self.pool.incref(pid)
        node = Node(
            id=nid,
            session=self.session,
            parent=parent_id,
            branch_id=BranchId(str(branch_id)),
            tokens=parent.tokens + known,
            residual=known,
            mode=mode,
            generation=parent.generation,
            table=table,
            created_at=monotonic(),
            last_decode_at=parent.last_decode_at,
            worker=parent.worker,
            tenant=parent.tenant,
        )
        if known:
            self._materialize_residual(node, known)
        parent.children.append(nid)
        self._nodes[nid] = node
        self._check_prefix(node)
        return node

    def append_tokens(self, nid: NodeId, tokens: TokenSeq, *, committed: bool) -> None:
        """Decode / residual prefill write path with CoW of the tail page."""
        node = self._nodes[nid]
        if node.mode is NodeMode.DEAD:
            raise InvariantError("append on dead node")
        tokens = as_tokens(tokens)
        if not tokens:
            return
        if committed and node.mode is NodeMode.SPEC:
            node.mode = NodeMode.COMMIT
        self._materialize_residual(node, tokens)
        node.residual = node.residual + tokens
        node.tokens = node.tokens + tokens
        node.last_decode_at = monotonic()
        self.last_decode_at = node.last_decode_at

    def abort(self, nid: NodeId, *, cascade: bool = True) -> int:
        """Mark v and descendants Dead; free residual pages (Lemma 3).

        Post-order: children release their fork-time alias increfs before the
        parent drops its own residual, so a live grandchild cannot see a freed trunk.

        If ``cascade`` and aborting ``nid`` leaves a Spec ancestor with no live
        children, abort that ancestor and continue upward. The walk stops at the
        root, a committed/idle spine node, or a parent that still has a live child
        (Figure 3: C's four leaves die ⇒ C1, C2, then C; Root lives via A, B).
        """
        if nid not in self._nodes:
            return 0
        freed = self._abort_down(nid)
        if cascade:
            freed += self._cascade_orphans(nid)
        return freed

    def _abort_down(self, nid: NodeId) -> int:
        order: list[NodeId] = []
        stack = [nid]
        seen: set[NodeId] = set()
        while stack:
            cur_id = stack.pop()
            if cur_id in seen:
                continue
            seen.add(cur_id)
            order.append(cur_id)
            stack.extend(self._nodes[cur_id].children)
        freed = 0
        for cur_id in reversed(order):
            node = self._nodes[cur_id]
            if node.mode is NodeMode.DEAD:
                continue
            freed += self._release_node(node)
            node.mode = NodeMode.DEAD
            node.counters.cancelled += 1
        return freed

    def _cascade_orphans(self, nid: NodeId) -> int:
        """Abort Spec ancestors that no longer have any live child."""
        if nid not in self._nodes:
            return 0
        freed = 0
        parent_id = self._nodes[nid].parent
        while parent_id is not None and parent_id != self.root:
            parent = self._nodes[parent_id]
            if parent.mode is NodeMode.DEAD:
                parent_id = parent.parent
                continue
            if parent.mode is not NodeMode.SPEC:
                break
            if self.children_of(parent.id, live_only=True):
                break
            next_id = parent.parent
            freed += self._abort_down(parent.id)
            parent_id = next_id
        return freed

    def bump_generation(self, nid: NodeId) -> int:
        node = self._nodes[nid]
        node.generation += 1
        return node.generation

    def set_mode(self, nid: NodeId, mode: NodeMode) -> None:
        node = self._nodes[nid]
        if node.mode is NodeMode.DEAD and mode is not NodeMode.DEAD:
            raise InvariantError("cannot revive a dead node")
        node.mode = mode

    def mark_idle(self, nid: NodeId, deadline: float | None) -> None:
        node = self._nodes[nid]
        if node.mode is NodeMode.COMMIT:
            node.mode = NodeMode.IDLE
        node.ttl_deadline = deadline

    def best_lcp_child(self, parent_id: NodeId, prompt: TokenSeq) -> tuple[Node | None, int]:
        """Step 1 of §6.4: v* = arg max LCP(x, x_v) among live children."""
        prompt = as_tokens(prompt)
        best: Node | None = None
        best_l = -1
        for child in self.children_of(parent_id):
            # Compare against the child's full sequence (trunk + residual).
            n = lcp_len(prompt, child.tokens)
            if n > best_l:
                best_l = n
                best = child
        return best, max(best_l, 0)

    def truncate_residual(self, nid: NodeId, keep_tokens: int) -> None:
        """Drop pages covering x_v[ℓ:] after LCP commit. CoW-split mid-page."""
        node = self._nodes[nid]
        keep = max(0, keep_tokens - node.table.alias_len)
        if keep >= node.table.residual_tokens:
            node.tokens = node.tokens[:keep_tokens]
            node.residual = node.residual[:keep]
            return
        ps = self.pool.page_size
        keep_pages = keep // ps
        rem = keep % ps
        drop = node.table.residual[keep_pages + (1 if rem else 0) :]
        for pid in drop:
            self.pool.decref(pid)
        node.table.residual = node.table.residual[: keep_pages + (1 if rem else 0)]
        if rem and node.table.residual:
            last = node.table.residual[-1]
            node.table.residual[-1] = self.pool.split_at(last, rem, node.id)
        node.table.residual_tokens = keep
        node.residual = node.residual[:keep]
        node.tokens = node.tokens[: node.table.alias_len + keep]

    def live_kv_tokens(self) -> int:
        """Distinct live tokens: trunk counted once (Equation 3)."""
        if self.root is None:
            return 0
        root = self._nodes[self.root]
        extra = 0
        for n in self.live_nodes():
            if n.id != self.root:
                extra += len(n.residual)
        return len(root.tokens) + extra

    def _materialize_residual(self, node: Node, tokens: TokenSeq) -> None:
        ps = self.pool.page_size
        remaining = list(tokens)
        # Fill the current tail page if it is private.
        if node.table.residual:
            last_id = node.table.residual[-1]
            last = self.pool.get(last_id)
            space = ps - last.n_valid
            if space > 0:
                last_id = self.pool.cow_if_needed(last_id, node.id)
                node.table.residual[-1] = last_id
                chunk = tuple(remaining[:space])
                self.pool.write_tokens(last_id, chunk, self.pool.get(last_id).n_valid)
                remaining = remaining[len(chunk) :]
        while remaining:
            pid = self.pool.alloc_page(node.id, tenant=node.tenant)
            take = tuple(remaining[:ps])
            self.pool.write_tokens(pid, take, 0)
            node.table.residual.append(pid)
            remaining = remaining[len(take) :]
        node.table.residual_tokens += len(tokens)

    def _release_node(self, node: Node) -> int:
        """Decref residual pages and aliased ancestor pages once."""
        n_res = len(node.table.residual)
        for pid in node.table.residual:
            self.pool.decref(pid)
        node.table.residual.clear()
        node.table.residual_tokens = 0
        # Drop the fork-time incref on aliased pages.
        if node.table.parent is not None:
            for pid in self.pool.alias_pages(node.table.parent):
                self.pool.decref(pid)
        return n_res

    def _check_prefix(self, node: Node) -> None:
        if node.parent is None:
            return
        parent = self._nodes[node.parent]
        if node.tokens[: len(parent.tokens)] != parent.tokens:
            raise InvariantError(f"prefix invariant broken at {node.id}")


class Forest:
    """All sessions on one worker. Eviction and offload walk trees, not LRU."""

    def __init__(self, pool: PagePool, config: ForkServeConfig) -> None:
        self.pool = pool
        self.cfg = config
        self.sessions: dict[SessionId, ContextTree] = {}

    def create(self, session: SessionId, *, tenant: str = "default", worker: int = 0) -> ContextTree:
        if session in self.sessions:
            raise InvariantError(f"session {session} already open")
        tree = ContextTree(session, self.pool, self.cfg, tenant=tenant, worker=worker)
        self.sessions[session] = tree
        return tree

    def get(self, session: SessionId) -> ContextTree:
        return self.sessions[session]

    def close(self, session: SessionId) -> None:
        tree = self.sessions.pop(session, None)
        if tree is None:
            return
        if tree.root is not None:
            tree.abort(tree.root)

    def live_trees(self) -> Iterable[ContextTree]:
        return self.sessions.values()

    def footprint_bytes(self) -> float:
        return self.pool.footprint_bytes()


class InvariantError(RuntimeError):
    pass


def residual_pages_of(node: Node, page_size: int) -> int:
    return pages_for_tokens(len(node.residual), page_size)

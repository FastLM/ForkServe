"""Tree-sticky placement and residual-only steal (§7.2).

The worker that prefills the root owns the trunk. Forks schedule there first.
If residual HBM is exhausted we may steal a speculative residual without
shipping the trunk. Commit always returns to the trunk owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forkserve.config import ForkServeConfig
from forkserve.types import SessionId, WorkerId


@dataclass(slots=True)
class WorkerState:
    id: WorkerId
    free_residual_pages: int
    live_trees: int = 0
    load_tokens: int = 0


@dataclass(slots=True)
class Placement:
    worker: WorkerId
    stolen: bool
    reason: str


class TreeStickyRouter:
    def __init__(self, config: ForkServeConfig) -> None:
        self.cfg = config
        self.owner: dict[SessionId, WorkerId] = {}
        self.workers: dict[WorkerId, WorkerState] = {
            WorkerId(i): WorkerState(id=WorkerId(i), free_residual_pages=1_000_000)
            for i in range(config.num_workers)
        }

    def pin_root(self, session: SessionId, worker: WorkerId | None = None) -> WorkerId:
        if session in self.owner:
            return self.owner[session]
        if worker is None:
            worker = min(self.workers.values(), key=lambda w: (w.live_trees, w.load_tokens)).id
        self.owner[session] = worker
        self.workers[worker].live_trees += 1
        return worker

    def place_fork(
        self,
        session: SessionId,
        residual_tokens: int,
        *,
        speculative: bool,
    ) -> Placement:
        owner = self.owner.get(session)
        if owner is None:
            owner = self.pin_root(session)
        need_pages = max(1, (residual_tokens + self.cfg.page_size - 1) // self.cfg.page_size)
        w = self.workers[owner]
        if w.free_residual_pages >= need_pages:
            w.free_residual_pages -= need_pages
            w.load_tokens += residual_tokens
            return Placement(owner, stolen=False, reason="sticky")
        if speculative and self.cfg.residual_steal_enabled and len(self.workers) > 1:
            victim = min(
                (x for x in self.workers.values() if x.id != owner),
                key=lambda x: x.load_tokens,
            )
            if victim.free_residual_pages >= need_pages:
                victim.free_residual_pages -= need_pages
                victim.load_tokens += residual_tokens
                return Placement(victim.id, stolen=True, reason="residual_steal")
        return Placement(owner, stolen=False, reason="sticky_oversub")

    def release(self, session: SessionId, residual_tokens: int, worker: WorkerId) -> None:
        pages = max(1, (residual_tokens + self.cfg.page_size - 1) // self.cfg.page_size)
        w = self.workers[worker]
        w.free_residual_pages += pages
        w.load_tokens = max(0, w.load_tokens - residual_tokens)

    def close(self, session: SessionId) -> None:
        owner = self.owner.pop(session, None)
        if owner is not None:
            self.workers[owner].live_trees = max(0, self.workers[owner].live_trees - 1)

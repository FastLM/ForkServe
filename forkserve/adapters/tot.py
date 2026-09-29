"""Tree-of-Thoughts expander: fork b thought prefixes; abort failed thoughts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from forkserve.adapters.templates import ToolWrappers
from forkserve.api import Engine
from forkserve.planner import Candidate
from forkserve.types import BranchId, JoinPolicy, NodeId, SchemaKind, SessionId


@dataclass
class ToTAdapter:
    engine: Engine
    wrappers: ToolWrappers
    branching: int = 3

    def expand(
        self,
        session: SessionId,
        parent: NodeId,
        *,
        b: int | None = None,
        t_idle_ms: float = 1e9,
        thoughts: Sequence[str] | None = None,
    ) -> list[NodeId]:
        k = b if b is not None else self.branching
        if thoughts is None:
            texts = [self.wrappers.thought_prefix(i) for i in range(k)]
        else:
            texts = list(thoughts)[:k]
            k = len(texts)
        children: list[NodeId] = []
        cands: list[Candidate] = []
        for i, prefix in enumerate(texts):
            nid = self.engine.fork(session, parent, f"thought-{i}", prefix)
            children.append(nid)
            cands.append(
                Candidate(
                    branch_id=BranchId(f"thought-{i}"),
                    node_id=nid,
                    known=self.engine._tok(prefix),
                    p_b=1.0 / k,
                    schema=SchemaKind.FREEFORM,
                    declared=True,
                )
            )
        self.engine.speculate_set(session, parent, cands, t_idle_ms=t_idle_ms)
        self.engine.drain_slack()
        return children

    def run_turns(
        self,
        session: SessionId,
        parent: NodeId,
        thought_turns: Sequence[Sequence[str]],
        *,
        winner: int = 0,
        t_idle_ms: float = 1e9,
    ) -> NodeId:
        """Depth-d ToT: expand k agents, keep the winner, repeat on the spine."""
        tip = parent
        for thoughts in thought_turns:
            kids = self.expand(session, tip, thoughts=thoughts, t_idle_ms=t_idle_ms)
            if not kids:
                continue
            pick = winner if winner < len(kids) else 0
            tip = self.select(session, kids, pick, parent=tip)
        return tip

    def select(
        self,
        session: SessionId,
        children: list[NodeId],
        winner: int,
        *,
        parent: NodeId | None = None,
    ) -> NodeId:
        keep = children[winner]
        for i, cid in enumerate(children):
            if i != winner:
                self.engine.abort(session, cid)
        if hasattr(self.engine, "promote"):
            return self.engine.promote(session, keep)
        result = self.engine.join(
            session,
            [keep],
            JoinPolicy.WINNER,
            parent=parent,
        )
        return result.node_id

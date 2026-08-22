"""LangGraph ``Send`` lowers to fork+join. Adapters do not rewrite prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from forkserve.adapters.templates import ToolWrappers
from forkserve.api import Engine
from forkserve.planner import Candidate
from forkserve.types import BranchId, JoinPolicy, NodeId, SchemaKind, SessionId


@dataclass
class Send:
    role: str
    prompt: str


@dataclass
class LangGraphAdapter:
    engine: Engine
    wrappers: ToolWrappers

    def send(
        self,
        session: SessionId,
        parent: NodeId,
        sends: Iterable[Send],
        *,
        t_idle_ms: float = 1e9,
    ) -> list[NodeId]:
        children: list[NodeId] = []
        cands: list[Candidate] = []
        for s in sends:
            suffix = self.wrappers.sibling_system(s.role, s.prompt)
            nid = self.engine.fork(session, parent, s.role, suffix)
            children.append(nid)
            cands.append(
                Candidate(
                    branch_id=BranchId(s.role),
                    node_id=nid,
                    known=self.engine._tok(suffix),
                    p_b=1.0,
                    schema=SchemaKind.FREEFORM,
                    declared=True,
                )
            )
        self.engine.speculate_set(session, parent, cands, t_idle_ms=t_idle_ms)
        self.engine.drain_slack()
        return children

    def join(
        self,
        session: SessionId,
        children: list[NodeId],
        policy: JoinPolicy = JoinPolicy.ALL,
        *,
        blend: str = "",
        parent: NodeId | None = None,
    ) -> NodeId:
        scaffold = self.wrappers.join_scaffold()
        result = self.engine.join(
            session,
            children,
            policy,
            scaffold=scaffold,
            blend=blend,
            parent=parent,
        )
        return result.node_id

"""ReAct / tool-choice adapter. Appendix D, simplified and complete."""

from __future__ import annotations

from dataclasses import dataclass

from forkserve.adapters.templates import ToolWrappers
from forkserve.api import Engine
from forkserve.engine.scanner import StreamingToolScanner
from forkserve.planner import Candidate, schema_kind_of
from forkserve.types import BranchId, NodeId, SchemaKind, SessionId


@dataclass
class ReActAdapter:
    engine: Engine
    wrappers: ToolWrappers

    def on_tool_parsed(
        self,
        session: SessionId,
        parent: NodeId,
        tool: str,
        *,
        t_idle_ms: float,
        include_recovery: bool = True,
    ) -> tuple[NodeId, NodeId | None]:
        wrap = self.wrappers.observation(tool)
        recov = self.wrappers.recovery(tool)
        happy = self.engine.fork(session, parent, tool, wrap)
        fail: NodeId | None = None
        cands = [
            Candidate(
                branch_id=BranchId(tool),
                node_id=happy,
                known=self.engine._tok(wrap),
                p_b=1.0,
                schema=schema_kind_of(tool),
                declared=True,
            )
        ]
        if include_recovery:
            fail = self.engine.fork(session, parent, "err", recov)
            cands.append(
                Candidate(
                    branch_id=BranchId("err"),
                    node_id=fail,
                    known=self.engine._tok(recov),
                    p_b=0.2,
                    schema=SchemaKind.FREEFORM,
                    declared=True,
                )
            )
        self.engine.speculate_set(session, parent, cands, t_idle_ms=t_idle_ms)
        return happy, fail

    def bind_observation(
        self,
        session: SessionId,
        parent: NodeId,
        tool: str,
        observation: str,
        *,
        ok: bool,
    ) -> NodeId:
        wrap = self.wrappers.observation(tool) if ok else self.wrappers.recovery(tool)
        close = self.wrappers.close_observation() if ok else self.wrappers.close_observation()
        prompt = wrap + observation + close
        cr = self.engine.commit(
            session, parent, prompt, preferred_bid=tool if ok else "err"
        )
        return cr.winner

    def scanner(self) -> StreamingToolScanner:
        return StreamingToolScanner()

"""Userspace orchestrator wrapping OpenAI-compatible streaming (~1.1K paper LoC).

Adapters announce structure the harness already has; they do not rewrite
prompts. A closed-source gateway can still infer wrappers via the scanner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Awaitable

from forkserve.adapters.templates import ToolWrappers
from forkserve.api import Engine, SessionHandle
from forkserve.config import ForkServeConfig
from forkserve.engine.scanner import ParsedToolCall, StreamingToolScanner
from forkserve.planner import Candidate, schema_kind_of
from forkserve.types import (
    BranchId,
    NodeId,
    SchemaKind,
    SessionId,
    TokenSeq,
)


ToolFn = Callable[[str, str], Awaitable[str]]


@dataclass
class TurnResult:
    session: SessionId
    parent: NodeId
    winner: NodeId
    output_tokens: TokenSeq
    output_text: str
    tool: ParsedToolCall | None
    ttft_ms: float


@dataclass
class Orchestrator:
    engine: Engine
    wrappers: ToolWrappers = field(default_factory=ToolWrappers)
    config: ForkServeConfig = field(default_factory=ForkServeConfig)

    def open_session(self, system: str, user: str, *, tenant: str = "default") -> SessionHandle:
        text = self.wrappers.chat(system, user)
        return self.engine.open(text, tenant=tenant)

    async def react_turn(
        self,
        handle: SessionHandle,
        *,
        run_tool: ToolFn,
        max_decode: int = 256,
        recovery: bool = True,
        t_idle_ms: float = 2000.0,
    ) -> TurnResult:
        """ReAct: fork happy-path + recovery the moment a tool call is parsed."""
        tree = self.engine.tree(handle.id)
        parent = tree.tip
        assert parent is not None
        scanner = StreamingToolScanner()
        decoded: list[int] = []
        parsed: ParsedToolCall | None = None

        # Incremental decode so we can fork mid-generation (Sutradhara / SPORK).
        for _ in range(max_decode):
            tok = self.engine.generate(handle.id, 1)
            if not tok:
                break
            decoded.extend(tok)
            piece = self.engine.backend.detokenize(tok)
            hits = scanner.feed(piece)
            if hits:
                parsed = hits[-1]
                break
            if tok[-1] == 2:  # mock eos
                break

        if parsed is None:
            text = self.engine.backend.detokenize(tuple(decoded))
            return TurnResult(handle.id, parent, parent, tuple(decoded), text, None, 0.0)

        wrap = self.wrappers.observation(parsed.name)
        recov = self.wrappers.recovery(parsed.name)
        # Placeholders only — splice arguments as a second residual (§10).
        happy = self.engine.fork(handle.id, parent, parsed.name, wrap, speculate=False)
        fail: NodeId | None = None
        cands = [
            Candidate(
                branch_id=BranchId(parsed.name),
                node_id=happy,
                known=self.engine._tok(wrap),
                p_b=1.0,
                q_b=0.0,
                schema=schema_kind_of(parsed.name),
                declared=True,
            )
        ]
        if recovery:
            fail = self.engine.fork(handle.id, parent, "err", recov, speculate=False)
            cands.append(
                Candidate(
                    branch_id=BranchId("err"),
                    node_id=fail,
                    known=self.engine._tok(recov),
                    p_b=0.2,
                    q_b=0.0,
                    schema=SchemaKind.FREEFORM,
                    declared=True,
                )
            )
        self.engine.mark_tool_idle(handle.id, parent, tool_s=t_idle_ms / 1000.0)
        self.engine.speculate_set(handle.id, parent, cands, t_idle_ms=t_idle_ms)
        self.engine.drain_slack()

        try:
            obs = await run_tool(parsed.name, parsed.arguments)
            prompt = wrap + obs
            bid = parsed.name
        except Exception as exc:  # harness recovery path
            prompt = recov + f"{type(exc).__name__}: {exc}"
            bid = "err"

        cr = self.engine.commit(handle.id, parent, prompt, preferred_bid=bid)
        out = self.engine.generate(handle.id, max_decode)
        text = self.engine.backend.detokenize(out)
        m = self.engine.metrics[handle.id]
        ttft = m.ttft_ms[-1] if m.ttft_ms else 0.0
        handle.tip = cr.winner
        return TurnResult(handle.id, parent, cr.winner, out, text, parsed, ttft)

    async def stream_react(
        self,
        handle: SessionHandle,
        run_tool: ToolFn,
        turns: int = 8,
    ) -> AsyncIterator[TurnResult]:
        for _ in range(turns):
            result = await self.react_turn(handle, run_tool=run_tool)
            yield result
            if result.tool is None:
                return

"""OpenHands / mini-SWE-agent: observation wrapper + recovery wrapper.

OpenHands occasionally inlines extra metadata in the wrapper; LCP commit
still recovers the shared prefix (known-suffix hit rate 93% on SWE).
"""

from __future__ import annotations

from dataclasses import dataclass

from forkserve.adapters.react import ReActAdapter
from forkserve.adapters.templates import ToolWrappers
from forkserve.api import Engine
from forkserve.planner import NGramResidual, schema_kind_of
from forkserve.types import NodeId, SchemaKind, SessionId


SWE_TOOLS = frozenset({"bash", "cmd_run", "str_replace_editor", "edit", "search", "ipython"})


@dataclass
class OpenHandsAdapter:
    engine: Engine
    wrappers: ToolWrappers
    react: ReActAdapter

    def __init__(self, engine: Engine, wrappers: ToolWrappers | None = None) -> None:
        self.engine = engine
        self.wrappers = wrappers or ToolWrappers(style="openai_xml")
        self.react = ReActAdapter(engine, self.wrappers)

    def on_tool(
        self,
        session: SessionId,
        parent: NodeId,
        tool: str,
        *,
        t_idle_ms: float,
        residual_hat: str = "",
    ) -> tuple[NodeId, NodeId | None]:
        happy, fail = self.react.on_tool_parsed(
            session, parent, tool, t_idle_ms=t_idle_ms
        )
        kind = schema_kind_of(tool)
        if residual_hat and kind is not SchemaKind.FREEFORM:
            self.engine.speculate(
                session,
                happy,
                residual_hat=residual_hat,
                q_b=0.5,
                schema=kind,
                t_idle_ms=t_idle_ms,
            )
        return happy, fail

    def predict_pytest_header(self, ngrams: NGramResidual) -> str:
        """Schema-stable leader of a focused pytest log — residual speculation."""
        guessed = ngrams.predict("bash")
        if guessed:
            return self.engine.backend.detokenize(guessed)
        return "============================= test session starts =============================\n"

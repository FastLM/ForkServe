"""ReAct adapter: fork happy-path + recovery at tool-parse time, commit by LCP."""

from __future__ import annotations

import asyncio

from forkserve.api import Engine
from forkserve.config import ForkServeConfig
from forkserve.engine.mock import MockBackend
from forkserve.orchestrator import Orchestrator


async def fake_tool(name: str, args: str) -> str:
    return f"stdout of {name}({args})\npassed\n"


async def main() -> None:
    cfg = ForkServeConfig()
    backend = MockBackend(cfg)
    # Script a JSON tool call then a short final answer.
    engine = Engine(backend, cfg)
    orch = Orchestrator(engine)
    handle = orch.open_session("You are a coding agent.", "Fix the failing test.")

    tip = engine.tree(handle.id).tip
    assert tip is not None
    call = '{"name": "bash", "arguments": {"cmd": "pytest"}}'
    backend.decode_script[tip] = list(backend.tokenize(call))

    result = await orch.react_turn(handle, run_tool=fake_tool, t_idle_ms=800)
    print("winner", result.winner, "ttft_ms", round(result.ttft_ms, 2))
    print(engine.metrics[handle.id].summary())


if __name__ == "__main__":
    asyncio.run(main())

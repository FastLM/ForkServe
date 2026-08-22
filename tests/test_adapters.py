import pytest

from forkserve.adapters.langgraph import LangGraphAdapter, Send
from forkserve.adapters.react import ReActAdapter
from forkserve.adapters.templates import ToolWrappers
from forkserve.adapters.tot import ToTAdapter
from forkserve.api import Engine
from forkserve.config import ForkServeConfig
from forkserve.engine.mock import MockBackend
from forkserve.engine.scanner import StreamingToolScanner
from forkserve.types import JoinPolicy, NodeMode


@pytest.fixture
def eng() -> Engine:
    cfg = ForkServeConfig(page_size=8, bytes_per_token=1.0)
    return Engine(MockBackend(cfg), cfg)


def test_scanner_json_and_xml() -> None:
    s = StreamingToolScanner()
    hits = s.feed('think {"name": "bash", "arguments": {"cmd": "ls"}}')
    assert hits and hits[0].name == "bash"
    s.reset()
    hits = s.feed("<tool_call><name>edit</name><arguments>{}</arguments></tool_call>")
    assert hits and hits[0].name == "edit"


def test_react_bind(eng: Engine) -> None:
    wrap = ToolWrappers()
    ad = ReActAdapter(eng, wrap)
    h = eng.open("sys")
    happy, fail = ad.on_tool_parsed(h.id, h.tip, "bash", t_idle_ms=500)
    assert fail is not None
    winner = ad.bind_observation(h.id, h.tip, "bash", "ok\n", ok=True)
    assert eng.tree(h.id).get(winner).mode is NodeMode.COMMIT
    assert eng.tree(h.id).get(fail).mode is NodeMode.DEAD


def test_langgraph_send_join(eng: Engine) -> None:
    ad = LangGraphAdapter(eng, ToolWrappers())
    h = eng.open("planner trunk")
    kids = ad.send(
        h.id,
        h.tip,
        [Send("engineer", "implement"), Send("tester", "write tests")],
    )
    assert len(kids) == 2
    joined = ad.join(h.id, kids, JoinPolicy.ALL, blend="both done")
    assert eng.tree(h.id).get(joined).mode is NodeMode.COMMIT


def test_tot_expand_select(eng: Engine) -> None:
    ad = ToTAdapter(eng, ToolWrappers(), branching=3)
    h = eng.open("problem")
    kids = ad.expand(h.id, h.tip)
    assert len(kids) == 3
    winner = ad.select(h.id, kids, winner=1)
    live = [n for n in eng.tree(h.id).live_nodes() if n.branch_id.startswith("thought")]
    assert eng.tree(h.id).get(winner).mode is NodeMode.COMMIT
    assert all(n.mode is NodeMode.DEAD or n.id == kids[1] for n in live)

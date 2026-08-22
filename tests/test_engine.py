import pytest

from forkserve.api import Engine
from forkserve.config import ForkServeConfig
from forkserve.engine.mock import MockBackend
from forkserve.types import NodeMode


@pytest.fixture
def eng() -> Engine:
    cfg = ForkServeConfig(page_size=8, bytes_per_token=1.0, max_batched_tokens=2048)
    return Engine(MockBackend(cfg), cfg)


def test_open_fork_speculate_commit_roundtrip(eng: Engine) -> None:
    h = eng.open("system history trunk tokens here")
    tree = eng.tree(h.id)
    wrap = "<tool_response>\n"
    recov = "ERROR bash: "
    happy = eng.fork(h.id, h.tip, "bash", wrap)
    fail = eng.fork(h.id, h.tip, "err", recov)
    eng.speculate(h.id, happy, t_idle_ms=1000)
    eng.speculate(h.id, fail, priority=0.2, t_idle_ms=1000)
    n = eng.drain_slack()
    assert n >= 0
    prompt = wrap + '{"ok": true}'
    cr = eng.commit(h.id, h.tip, prompt, preferred_bid="bash")
    assert cr.known_suffix_hit
    assert tree.get(cr.winner).mode is NodeMode.COMMIT
    assert tree.get(fail).mode is NodeMode.DEAD
    m = eng.metrics[h.id]
    assert m.commits == 1
    assert m.known_suffix_hit_rate == 1.0


def test_generate_refuses_spec_tip(eng: Engine) -> None:
    h = eng.open("trunk")
    spec = eng.fork(h.id, h.tip, "x", "suffix")
    eng.tree(h.id).tip = spec
    with pytest.raises(Exception):
        eng.generate(h.id, 4)


def test_fanout_cow_memory(eng: Engine) -> None:
    h = eng.open(" ".join(f"tok{i}" for i in range(200)))
    parent = h.tip
    kids = [eng.fork(h.id, parent, f"r{i}", f"role-{i} prompt") for i in range(4)]
    tree = eng.tree(h.id)
    trunk = len(tree.get(parent).tokens)
    extra = sum(len(tree.get(k).residual) for k in kids)
    live = tree.live_kv_tokens()
    assert live == trunk + extra
    # clone would be 4*trunk + extra
    assert live < 2 * trunk

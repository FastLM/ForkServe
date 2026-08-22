from forkserve.commit import CommitProtocol
from forkserve.config import ForkServeConfig
from forkserve.pages import PagePool, TokenKvStore
from forkserve.tree import ContextTree
from forkserve.types import NodeMode, SessionId


def _ready() -> ContextTree:
    cfg = ForkServeConfig(page_size=8)
    pool = PagePool(cfg, TokenKvStore())
    t = ContextTree(SessionId("s"), pool, cfg)
    t.open_root((0, 1, 2, 3, 4, 5, 6, 7))
    return t


def test_lcp_commit_keeps_matching_prefix() -> None:
    t = _ready()
    root = t.root
    assert root is not None
    t.fork(root, "ok", (10, 11, 12, 13))
    t.fork(root, "err", (90, 91))
    prompt = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 99, 100)
    result = CommitProtocol().apply(t, root, prompt, preferred_bid="ok")
    assert result.lcp == 12
    assert result.tail == (99, 100)
    assert result.known_suffix_hit
    winner = t.get(result.winner)
    assert winner.mode is NodeMode.COMMIT
    assert winner.tokens == prompt
    # sibling aborted
    err = [c for c in t.children_of(root, live_only=False) if c.branch_id == "err"][0]
    assert err.mode is NodeMode.DEAD


def test_full_hit_skips_prefill() -> None:
    t = _ready()
    root = t.root
    assert root is not None
    t.fork(root, "ok", (10, 11))
    prompt = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)
    result = CommitProtocol().apply(t, root, prompt)
    assert result.skipped_prefill
    assert result.tail == ()


def test_no_child_forks_committed_continuation() -> None:
    t = _ready()
    root = t.root
    assert root is not None
    prompt = t.get(root).tokens + (42, 43)
    result = CommitProtocol().apply(t, root, prompt)
    assert result.tail == (42, 43)
    assert t.get(result.winner).mode is NodeMode.COMMIT


def test_generation_bumps() -> None:
    t = _ready()
    root = t.root
    assert root is not None
    t.fork(root, "a", (1,))
    g0 = t.get(root).generation
    CommitProtocol().apply(t, root, t.get(root).tokens + (1, 2))
    assert t.get(root).generation == g0 + 1

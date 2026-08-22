from forkserve.config import ForkServeConfig
from forkserve.pages import PagePool, TokenKvStore
from forkserve.tree import ContextTree, InvariantError
from forkserve.types import NodeMode, SessionId


def _tree() -> ContextTree:
    cfg = ForkServeConfig(page_size=8, bytes_per_token=1.0)
    pool = PagePool(cfg, TokenKvStore())
    t = ContextTree(SessionId("s"), pool, cfg)
    t.open_root(tuple(range(32)))
    return t


def test_fork_shares_trunk_pages() -> None:
    t = _tree()
    root = t.get(t.root)  # type: ignore[arg-type]
    trunk_pages = list(root.table.residual)
    child = t.fork(root.id, "happy", (100, 101, 102))
    for pid in trunk_pages:
        assert t.pool.get(pid).ref >= 2
    assert child.tokens[:32] == root.tokens
    assert child.residual == (100, 101, 102)
    assert child.mode is NodeMode.SPEC


def test_prefix_invariant() -> None:
    t = _tree()
    c = t.fork(t.root, "a", (7, 8))  # type: ignore[arg-type]
    assert c.tokens[:32] == t.get(t.root).tokens  # type: ignore[arg-type]


def test_abort_frees_only_residual() -> None:
    t = _tree()
    root = t.get(t.root)  # type: ignore[arg-type]
    live_before = len(t.pool)
    c = t.fork(root.id, "tmp", tuple(range(200, 216)))
    live_mid = len(t.pool)
    assert live_mid > live_before
    t.abort(c.id)
    assert c.mode is NodeMode.DEAD
    assert len(t.pool) == live_before


def test_cannot_fork_dead() -> None:
    t = _tree()
    c = t.fork(t.root, "x", (1,))  # type: ignore[arg-type]
    t.abort(c.id)
    try:
        t.fork(c.id, "y", (2,))
        raise AssertionError("should have failed")
    except InvariantError:
        pass


def test_lcp_selects_matching_wrapper() -> None:
    t = _tree()
    root = t.root
    assert root is not None
    t.fork(root, "ok", (1, 2, 3))
    t.fork(root, "err", (9, 9, 9))
    prompt = t.get(root).tokens + (1, 2, 3, 4, 5)
    winner, n = t.best_lcp_child(root, prompt)
    assert winner is not None
    assert winner.branch_id == "ok"
    assert n == 32 + 3

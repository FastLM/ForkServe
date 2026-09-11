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


def test_cascade_abort_when_all_children_die() -> None:
    """Figure 3: C's four leaves die ⇒ C1, C2, then C; Root lives via A, B."""
    t = _tree()
    root = t.root
    assert root is not None
    a = t.fork(root, "A", (10,))
    b = t.fork(root, "B", (20,))
    c = t.fork(root, "C", (30,))
    c1 = t.fork(c.id, "C1", (31,))
    c2 = t.fork(c.id, "C2", (32,))
    c1a = t.fork(c1.id, "C1a", (311,))
    c1b = t.fork(c1.id, "C1b", (312,))
    c2a = t.fork(c2.id, "C2a", (321,))
    c2b = t.fork(c2.id, "C2b", (322,))

    t.abort(c1a.id)
    assert t.get(c1.id).mode is NodeMode.SPEC  # C1b still live
    t.abort(c1b.id)
    assert t.get(c1.id).mode is NodeMode.DEAD
    assert t.get(c.id).mode is NodeMode.SPEC  # C2 still live

    t.abort(c2a.id)
    t.abort(c2b.id)
    assert t.get(c2.id).mode is NodeMode.DEAD
    assert t.get(c.id).mode is NodeMode.DEAD
    assert t.get(root).mode is NodeMode.COMMIT
    assert t.get(a.id).mode is NodeMode.SPEC
    assert t.get(b.id).mode is NodeMode.SPEC

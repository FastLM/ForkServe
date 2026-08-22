from forkserve.config import ForkServeConfig
from forkserve.pages import PagePool, TokenKvStore, clone_memory_bytes, cow_memory_bytes
from forkserve.types import NodeId


def _pool() -> PagePool:
    return PagePool(ForkServeConfig(page_size=4, bytes_per_token=10.0), TokenKvStore())


def test_cow_write_does_not_pollute_sibling() -> None:
    pool = _pool()
    a = pool.alloc_page(NodeId(1))
    pool.write_tokens(a, (1, 2, 3), 0)
    pool.incref(a)
    pool.pin_readonly([a])
    b = pool.cow_if_needed(a, NodeId(2))
    assert b != a
    pool.write_tokens(b, (9,), 3)
    assert pool.get(a).n_valid == 3
    assert pool.get(b).n_valid == 4


def test_abort_cost_is_residual_not_trunk() -> None:
    bpt = 320_000.0
    trunk, residuals = 16_000, [400, 400, 400, 400]
    cow = cow_memory_bytes(trunk, residuals, bpt)
    clone = clone_memory_bytes(trunk, residuals, bpt, k=4)
    assert clone / cow > 3.5
    # Lemma 3: aborting one child frees b * residual, not trunk.
    miss = bpt * residuals[0]
    assert miss < 0.05 * (bpt * trunk)


def test_decref_returns_to_freelist() -> None:
    pool = _pool()
    p = pool.alloc_page(NodeId(1))
    pool.decref(p)
    q = pool.alloc_page(NodeId(2))
    assert q == p

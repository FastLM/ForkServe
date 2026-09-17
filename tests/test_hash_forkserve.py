"""Compare APC-only, CoW-only, and HashForkServe hybrid."""

from __future__ import annotations

from forkserve.config import ForkServeConfig
from forkserve.hash_forkserve import HashForkPool, HashForkServe, hash_block
from forkserve.pages import pages_for_tokens
from forkserve.types import NodeId


def test_hash_chain_matches_parent_tokens() -> None:
    t1 = tuple(range(16))
    t2 = tuple(range(16, 32))
    h0 = hash_block(None, t1)
    h1 = hash_block(h0, t2)
    h1b = hash_block(h0, t2)
    assert h1 == h1b
    assert h0 != h1
    # Different prefix → different child hash even with same block tokens.
    h_bad = hash_block(hash_block(None, tuple(range(100, 116))), t2)
    assert h_bad != h1


def test_apc_cross_session_hit() -> None:
    cfg = ForkServeConfig(page_size=8, bytes_per_token=1.0)
    hfs = HashForkServe(cfg)
    trunk = tuple(range(24))  # 3 full pages
    s1 = hfs.open("a", trunk)
    assert len(s1.pages) == 3
    hfs.close("a")
    # Second session should reuse all 3 full blocks via hash.
    before = hfs.stats.hash_hits
    s2 = hfs.open("b", trunk + (99, 100))
    assert s2.pages[0] == s1.pages[0] or hfs.stats.hash_hits > before
    assert hfs.stats.hash_hits >= before + 3
    assert len(s2.pages) >= 3


def test_fork_aliases_without_hash_walk() -> None:
    cfg = ForkServeConfig(page_size=8, bytes_per_token=1.0)
    hfs = HashForkServe(cfg)
    trunk = tuple(range(32))
    hfs.open("root", trunk)
    aliases_before = hfs.stats.fork_aliases
    child = hfs.fork("root", "child", known_suffix=(200, 201, 202))
    assert hfs.stats.fork_aliases == aliases_before + 4  # 32/8 pages
    # Child shares physical trunk pages with parent.
    parent = hfs.sessions["root"]
    assert child.pages[:4] == parent.pages
    assert child.tokens[:32] == parent.tokens
    assert child.speculative


def test_hybrid_fork_then_other_session_hits_published() -> None:
    """ForkServe path creates trunk; commit publishes; APC reuses across sessions."""
    cfg = ForkServeConfig(page_size=8, bytes_per_token=1.0)
    hfs = HashForkServe(cfg)
    trunk = tuple(range(16))
    hfs.open("planner", trunk)
    hfs.fork("planner", "eng", known_suffix=tuple(range(100, 108)))
    hfs.commit("eng", trunk + tuple(range(100, 108)))
    # New session with same eng prompt should APC-hit trunk + suffix.
    hits_before = hfs.stats.hash_hits
    s = hfs.open("replay", trunk + tuple(range(100, 108)))
    assert hfs.stats.hash_hits >= hits_before + 2
    assert len(s.pages) == 3  # 24 tokens / 8


def test_cache_salt_isolates_tenants() -> None:
    cfg = ForkServeConfig(page_size=8, bytes_per_token=1.0)
    pool = HashForkPool(cfg)
    toks = tuple(range(16))
    a = pool.materialize_with_hash(NodeId(1), toks, cache_salt=b"tenant-a")
    b = pool.materialize_with_hash(NodeId(2), toks, cache_salt=b"tenant-b")
    # Different salt → first blocks differ; no cross-tenant reuse.
    assert a.page_ids[0] != b.page_ids[0]
    assert a.reused_hash_pages == 0
    assert b.reused_hash_pages == 0


def test_only_full_blocks_hashed() -> None:
    cfg = ForkServeConfig(page_size=8, bytes_per_token=1.0)
    pool = HashForkPool(cfg)
    mat = pool.materialize_with_hash(NodeId(1), tuple(range(10)))  # 8 + 2
    assert pool.index.stats.pages_hashed == 1
    assert len(mat.page_ids) == 2


def test_speculative_not_in_hash_until_commit() -> None:
    cfg = ForkServeConfig(page_size=8, bytes_per_token=1.0)
    hfs = HashForkServe(cfg)
    hfs.open("r", tuple(range(8)))
    hashed = hfs.stats.pages_hashed
    hfs.fork("r", "spec", known_suffix=tuple(range(50, 58)))
    assert hfs.stats.pages_hashed == hashed  # no new hash for spec
    hfs.commit("spec", tuple(range(8)) + tuple(range(50, 58)))
    assert hfs.stats.pages_hashed >= hashed + 1


def test_memory_fork_beats_clone_on_fanout() -> None:
    cfg = ForkServeConfig(page_size=16, bytes_per_token=100.0)
    hfs = HashForkServe(cfg)
    trunk = tuple(range(64))
    hfs.open("root", trunk)
    kids = [hfs.fork("root", f"c{i}", known_suffix=(1000 + i,)) for i in range(4)]
    live = len([p for p in hfs.hf.pool._pages.values() if p.ref > 0])
    # CoW: ~4 trunk pages + 4 tiny residuals, not 4× trunk.
    assert live < pages_for_tokens(64, 16) * 4
    assert all(k.pages[:4] == hfs.sessions["root"].pages for k in kids)

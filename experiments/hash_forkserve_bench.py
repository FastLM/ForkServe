"""Microbench: APC-only vs CoW-fork vs HashForkServe hybrid.

Measures (control-plane, no GPU):
  - pages allocated / shared
  - hash hit rate
  - fork alias count
  - wall time for a fan-out + cross-session replay workload
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from forkserve.config import ForkServeConfig
from forkserve.hash_forkserve import HashForkPool, HashForkServe
from forkserve.types import NodeId


@dataclass
class ModeResult:
    mode: str
    wall_ms: float
    live_pages: int
    hash_hits: int
    hash_misses: int
    fork_aliases: int
    pages_hashed: int
    sessions: int


def _cfg() -> ForkServeConfig:
    return ForkServeConfig(page_size=16, bytes_per_token=1.0)


def run_apc_only(n_sessions: int = 40, trunk_len: int = 256, residual: int = 32) -> ModeResult:
    """Every session is independent; sharing only via hash after first publish."""
    cfg = _cfg()
    pool = HashForkPool(cfg)
    trunk = tuple(range(trunk_len))
    t0 = time.perf_counter()
    for i in range(n_sessions):
        toks = trunk + tuple(range(10_000 + i * residual, 10_000 + (i + 1) * residual))
        owner = NodeId(i + 1)
        pool.materialize_with_hash(owner, toks, publish=True)
    wall = (time.perf_counter() - t0) * 1000
    live = sum(1 for p in pool.pool._pages.values() if p.ref > 0)
    return ModeResult(
        "apc_only",
        wall,
        live,
        pool.stats.hash_hits,
        pool.stats.hash_misses,
        pool.stats.fork_aliases,
        pool.stats.pages_hashed,
        n_sessions,
    )


def run_cow_only(n_parents: int = 10, fanout: int = 4, trunk_len: int = 256) -> ModeResult:
    """Same-session fan-out via fork alias; no cross-session hash."""
    cfg = _cfg()
    # Use HashForkServe but never open a second session with the same trunk
    # after close — measure fork aliases only.
    hfs = HashForkServe(cfg)
    t0 = time.perf_counter()
    for p in range(n_parents):
        trunk = tuple(range(trunk_len))
        sid = f"p{p}"
        hfs.open(sid, trunk)
        for j in range(fanout):
            hfs.fork(sid, f"p{p}-c{j}", known_suffix=(1000 + j, 1001 + j))
    wall = (time.perf_counter() - t0) * 1000
    live = sum(1 for pg in hfs.hf.pool._pages.values() if pg.ref > 0)
    return ModeResult(
        "cow_fork_only",
        wall,
        live,
        hfs.stats.hash_hits,
        hfs.stats.hash_misses,
        hfs.stats.fork_aliases,
        hfs.stats.pages_hashed,
        n_parents * (1 + fanout),
    )


def run_hybrid(
    n_planners: int = 10,
    fanout: int = 4,
    trunk_len: int = 256,
    replays: int = 20,
) -> ModeResult:
    """Fork for fan-out, commit+publish, then APC-hit on replay sessions."""
    cfg = _cfg()
    hfs = HashForkServe(cfg)
    trunk = tuple(range(trunk_len))
    t0 = time.perf_counter()
    for p in range(n_planners):
        sid = f"plan{p}"
        hfs.open(sid, trunk)
        for j in range(fanout):
            cid = f"plan{p}-c{j}"
            suffix = tuple(range(2000 + j * 16, 2000 + (j + 1) * 16))
            hfs.fork(sid, cid, known_suffix=suffix)
            hfs.commit(cid, trunk + suffix)
    for r in range(replays):
        # Replay a committed child prompt — should APC-hit trunk (+ suffix if full).
        j = r % fanout
        suffix = tuple(range(2000 + j * 16, 2000 + (j + 1) * 16))
        hfs.open(f"replay{r}", trunk + suffix)
    wall = (time.perf_counter() - t0) * 1000
    live = sum(1 for pg in hfs.hf.pool._pages.values() if pg.ref > 0)
    return ModeResult(
        "hash_forkserve",
        wall,
        live,
        hfs.stats.hash_hits,
        hfs.stats.hash_misses,
        hfs.stats.fork_aliases,
        hfs.stats.pages_hashed,
        n_planners * (1 + fanout) + replays,
    )


def main() -> None:
    results = [run_apc_only(), run_cow_only(), run_hybrid()]
    out = Path(__file__).parent / "hash_forkserve_bench.json"
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"{'mode':<16} {'wall_ms':>8} {'live':>6} {'hits':>6} {'miss':>6} {'fork':>6} {'hash':>6}")
    for r in results:
        print(
            f"{r.mode:<16} {r.wall_ms:8.2f} {r.live_pages:6d} {r.hash_hits:6d} "
            f"{r.hash_misses:6d} {r.fork_aliases:6d} {r.pages_hashed:6d}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

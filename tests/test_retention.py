from time import monotonic

from forkserve.config import ForkServeConfig
from forkserve.pages import PagePool, TokenKvStore
from forkserve.retention import RetentionManager, leaf_rank_key, node_ttl_s
from forkserve.tree import Forest
from forkserve.types import NodeMode, SessionId


def test_ttl_uses_residual_not_session() -> None:
    cfg = ForkServeConfig(tau_max_s=8.0, ttl_alpha=1.2, ttl_beta=1.0)
    short = node_ttl_s(8, t_b=0.4, c_r=0.01, q_q=0.0, p_g=1.0, cfg=cfg)
    long = node_ttl_s(4000, t_b=0.4, c_r=0.01, q_q=0.0, p_g=1.0, cfg=cfg)
    assert short <= long
    assert short <= cfg.tau_max_s


def test_spec_leaves_evicted_first() -> None:
    now = monotonic()
    from forkserve.tree import Node
    from forkserve.pages import LogicalTable
    from forkserve.types import BranchId, NodeId

    spec = Node(
        id=NodeId(2),
        session=SessionId("s"),
        parent=NodeId(1),
        branch_id=BranchId("s"),
        tokens=(1,),
        residual=(1,),
        mode=NodeMode.SPEC,
        generation=0,
        table=LogicalTable(None, 0),
        created_at=now,
        last_decode_at=now - 10,
    )
    spine = Node(
        id=NodeId(1),
        session=SessionId("s"),
        parent=None,
        branch_id=BranchId("r"),
        tokens=(0,),
        residual=(0,),
        mode=NodeMode.COMMIT,
        generation=0,
        table=LogicalTable(None, 0),
        created_at=now,
        last_decode_at=now,
        children=[NodeId(2)],
    )
    assert leaf_rank_key(spec, now, now) > leaf_rank_key(spine, now, now)


def test_expire_drops_spec_not_trunk() -> None:
    cfg = ForkServeConfig(page_size=8, bytes_per_token=1.0, tau_max_s=0.01)
    forest = Forest(PagePool(cfg, TokenKvStore()), cfg)
    tree = forest.create(SessionId("s"))
    root = tree.open_root(tuple(range(16)))
    child = tree.fork(root.id, "tmp", (99, 100))
    mgr = RetentionManager(forest, cfg)
    tree.mark_idle(child.id, monotonic() - 1.0)
    dead = mgr.expire()
    assert child.id in dead
    assert tree.get(root.id).mode is not NodeMode.DEAD


def test_dram_offload_parks_idle_residual() -> None:
    cfg = ForkServeConfig(page_size=8, bytes_per_token=1.0, hbm_capacity_bytes=8.0)
    pool = PagePool(cfg, TokenKvStore())
    forest = Forest(pool, cfg)
    tree = forest.create(SessionId("s"))
    root = tree.open_root(tuple(range(16)))
    idle = tree.fork(root.id, "idle", tuple(range(20, 36)), mode=NodeMode.COMMIT)
    tree.set_mode(idle.id, NodeMode.IDLE)
    mgr = RetentionManager(forest, cfg)
    decisions = mgr.plan_offload(hbm_target_bytes=1.0)
    assert any(d.dest == "dram" for d in decisions)
    assert tree.get(idle.id).mode is NodeMode.IDLE

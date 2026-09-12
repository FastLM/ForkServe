from forkserve.config import ForkServeConfig
from forkserve.planner import Candidate, SpeculatePlanner, expected_cost, expected_gain
from forkserve.types import BranchId, NodeId, NodeMode, SchemaKind, SessionId, WorkKind


def test_known_suffix_starves_residual() -> None:
    cfg = ForkServeConfig(bytes_per_token=1.0, lambda_tbt=4.0, q_min=0.3)
    planner = SpeculatePlanner(cfg)
    cands = [
        Candidate(
            branch_id=BranchId("tool"),
            node_id=NodeId(1),
            known=(1, 2, 3, 4),
            residual_hat=tuple(range(10, 40)),
            p_b=0.9,
            q_b=0.5,
            schema=SchemaKind.JSON,
        ),
        Candidate(
            branch_id=BranchId("err"),
            node_id=NodeId(2),
            known=(8, 8),
            p_b=0.2,
            schema=SchemaKind.FREEFORM,
        ),
    ]
    plan = planner.allocate(
        SessionId("s"),
        NodeId(0),
        cands,
        t_idle_ms=50.0,
        gamma_ms=20.0,
        m_free=1e12,
        parent_mode=NodeMode.IDLE,
    )
    kinds = [c.kind for c in plan.chunks]
    # Every known suffix is admitted before any residual.
    first_residual = next((i for i, k in enumerate(kinds) if k is WorkKind.OBS_RESIDUAL), len(kinds))
    assert all(k is WorkKind.KNOWN_SUFFIX for k in kinds[:first_residual])
    assert any(c.branch_id == "err" for c in plan.chunks)


def test_proposition1_ratio() -> None:
    cfg = ForkServeConfig(bytes_per_token=10.0, lambda_tbt=4.0)
    from forkserve.planner import WorkItem

    known = WorkItem(WorkKind.KNOWN_SUFFIX, BranchId("a"), None, (1,) * 16, 1.0, 1.0, 1.0)
    resid = WorkItem(WorkKind.OBS_RESIDUAL, BranchId("a"), None, (1,) * 16, 0.5, 0.5, 0.5)
    t_pre = cfg.prefill_ms(16)
    gk = expected_gain(known, t_pre, 100.0)
    ck = expected_cost(known, t_pre, 20.0, cfg.bytes_per_token, cfg.lambda_tbt)
    gr = expected_gain(resid, t_pre, 100.0)
    cr = expected_cost(resid, t_pre, 20.0, cfg.bytes_per_token, cfg.lambda_tbt)
    assert gk / ck >= gr / cr


def test_hbm_filter_drops_tail() -> None:
    cfg = ForkServeConfig(bytes_per_token=100.0, c_spec=8, q_min=0.1)
    planner = SpeculatePlanner(cfg)
    cands = [
        Candidate(
            branch_id=BranchId("big"),
            node_id=NodeId(1),
            known=tuple(range(64)),
            residual_hat=tuple(range(64, 200)),
            p_b=1.0,
            q_b=0.9,
            schema=SchemaKind.JSON,
        )
    ]
    plan = planner.allocate(
        SessionId("s"),
        NodeId(0),
        cands,
        t_idle_ms=1e9,
        gamma_ms=1e9,
        m_free=50.0,  # only 0.5 tokens at 100 B/tok — nothing fits... wait 100*64 >> 50
        parent_mode=NodeMode.IDLE,
    )
    assert plan.items == []
    assert plan.rejected


def test_tbt_filter_when_parent_committed() -> None:
    cfg = ForkServeConfig(prefill_us_per_token=1000.0)  # 1 ms / token
    planner = SpeculatePlanner(cfg)
    cands = [Candidate(BranchId("a"), NodeId(1), known=tuple(range(32)))]
    plan = planner.allocate(
        SessionId("s"),
        NodeId(0),
        cands,
        t_idle_ms=1e9,
        gamma_ms=0.1,
        m_free=1e12,
        parent_mode=NodeMode.COMMIT,
    )
    assert plan.items == []
    assert plan.rejected[0][1] == "tbt_margin"


def test_chunking_bounds_cancel_loss() -> None:
    cfg = ForkServeConfig(c_spec=8)
    planner = SpeculatePlanner(cfg)
    cands = [Candidate(BranchId("a"), NodeId(1), known=tuple(range(32)))]
    plan = planner.allocate(
        SessionId("s"),
        NodeId(0),
        cands,
        t_idle_ms=1e9,
        gamma_ms=1e9,
        m_free=1e12,
        parent_mode=NodeMode.IDLE,
    )
    assert len(plan.chunks) == 4
    assert all(len(c.tokens) <= 8 for c in plan.chunks)


def test_gpu_prefill_false_skips_recovery() -> None:
    cfg = ForkServeConfig(bytes_per_token=1.0)
    planner = SpeculatePlanner(cfg)
    cands = [
        Candidate(BranchId("bash"), NodeId(1), known=(1, 2, 3), gpu_prefill=True),
        Candidate(BranchId("err"), NodeId(2), known=(8, 8), p_b=0.2, gpu_prefill=False),
    ]
    plan = planner.allocate(
        SessionId("s"),
        NodeId(0),
        cands,
        t_idle_ms=1e9,
        gamma_ms=1e9,
        m_free=1e12,
        parent_mode=NodeMode.IDLE,
    )
    assert [c.branch_id for c in plan.chunks] == ["bash"]

from forkserve.config import ForkServeConfig
from forkserve.planner import PrefillChunk
from forkserve.scheduler import CommittedJob, TwoClassScheduler
from forkserve.types import BranchId, JobClass, NodeId, SessionId, WorkKind


def test_spec_gets_only_leftover() -> None:
    cfg = ForkServeConfig(max_batched_tokens=32, tick_ms=8.0, c_spec=16)
    sch = TwoClassScheduler(cfg)
    sch.submit_committed(
        CommittedJob(SessionId("s"), NodeId(1), tokens=24, kind="decode", slo_tokens_per_s=1000.0)
    )
    sch.submit_speculative(
        [
            PrefillChunk(
                SessionId("s"),
                NodeId(0),
                NodeId(2),
                BranchId("w"),
                tokens=tuple(range(8)),
                kind=WorkKind.KNOWN_SUFFIX,
                gain=10.0,
                cost=1.0,
            )
        ]
    )
    plan = sch.schedule()
    assert plan.B_c == 24
    assert plan.B_s == 8
    assert len(plan.speculative) == 1


def test_saturation_admits_no_spec() -> None:
    cfg = ForkServeConfig(max_batched_tokens=16)
    sch = TwoClassScheduler(cfg)
    sch.submit_committed(
        CommittedJob(SessionId("s"), NodeId(1), tokens=64, kind="decode", slo_tokens_per_s=1e9)
    )
    sch.submit_speculative(
        [
            PrefillChunk(
                SessionId("s"),
                NodeId(0),
                NodeId(2),
                BranchId("w"),
                tokens=tuple(range(8)),
                kind=WorkKind.KNOWN_SUFFIX,
                gain=1.0,
                cost=1.0,
            )
        ]
    )
    plan = sch.schedule()
    assert plan.saturated
    assert plan.B_s == 0
    assert plan.speculative == []
    assert sch.pending_spec_tokens() == 8  # still queued


def test_generation_invalidation() -> None:
    cfg = ForkServeConfig()
    sch = TwoClassScheduler(cfg)
    chunk = PrefillChunk(
        SessionId("s"),
        NodeId(7),
        NodeId(2),
        BranchId("w"),
        tokens=(1, 2, 3),
        kind=WorkKind.KNOWN_SUFFIX,
        generation=0,
        gain=1.0,
        cost=1.0,
    )
    sch.submit_speculative([chunk])
    n = sch.invalidate(NodeId(7), generation=1)
    assert n == 1
    assert sch.pending_spec_tokens() == 0


def test_spec_billed_at_kappa() -> None:
    cfg = ForkServeConfig(max_batched_tokens=16, spec_bill_kappa=0.25)
    sch = TwoClassScheduler(cfg)
    sch.submit_speculative(
        [
            PrefillChunk(
                SessionId("s"),
                NodeId(0),
                NodeId(2),
                BranchId("w"),
                tokens=tuple(range(8)),
                kind=WorkKind.KNOWN_SUFFIX,
                gain=1.0,
                cost=1.0,
            )
        ]
    )
    sch.schedule()
    assert sch.meters["default"].speculative == 8 * 0.25
    assert JobClass.SPECULATIVE.value == "speculative"

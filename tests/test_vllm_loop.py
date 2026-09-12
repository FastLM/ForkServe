from types import SimpleNamespace

from forkserve.engine.vllm_backend import (
    drop_covered_prefills,
    partition_fused,
    prompt_ids_of,
    split_pending_for_decode,
)
from forkserve.engine.protocol import PrefillRequest
from forkserve.engine.vllm_loop import (
    CowBlockTable,
    committed_first,
    cow_node_key,
    extra_of,
    forkserve_extra,
    is_speculative,
    select_full_blocks,
)
from forkserve.types import NodeId, SessionId


def _req(spec: bool, rid: str = "r") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=rid,
        sampling_params=SimpleNamespace(extra_args=forkserve_extra(speculative=spec, node_id=1)),
    )


def test_forkserve_extra_tags_class() -> None:
    s = forkserve_extra(speculative=True, node_id=7, parent_node=3, generation=2)
    assert s["forkserve_class"] == "speculative"
    assert s["forkserve_node"] == 7
    assert s["forkserve_parent_node"] == 3
    c = forkserve_extra(speculative=False, node_id=1)
    assert c["forkserve_class"] == "committed"


def test_committed_first_engine_loop_order() -> None:
    spec_a = _req(True, "s1")
    spec_b = _req(True, "s2")
    cmt = _req(False, "c1")
    ordered = committed_first([spec_a, cmt, spec_b])
    assert [r.request_id for r in ordered] == ["c1", "s1", "s2"]
    assert is_speculative(spec_a)
    assert not is_speculative(cmt)


def test_cow_bit_pins_shared_blocks() -> None:
    table = CowBlockTable()
    table.pin_ro(9)
    table.pin_ro([10, 11])
    assert not table.writable(9, ref_cnt=1)
    assert not table.writable(10, ref_cnt=1)
    assert table.writable(12, ref_cnt=1)
    assert not table.writable(12, ref_cnt=2)
    table.note_fork(2)
    table.note_cow_copy()
    assert table.forks == 1
    assert table.alias_blocks == 2
    assert table.cow_copies == 1


def test_cow_bind_parent_lookup() -> None:
    table = CowBlockTable()
    table.bind_node(4, "req-parent")
    assert table.req_for_node(4) == "req-parent"
    assert extra_of(_req(False))["forkserve_class"] == "committed"


def test_cow_keys_are_session_scoped() -> None:
    """Per-tree node ids restart at 1; snaps must not collide across sessions."""
    assert cow_node_key("s-a", 1) != cow_node_key("s-b", 1)
    table = CowBlockTable()
    table.bind_node(1, "req-a", session="s-a")
    table.bind_node(1, "req-b", session="s-b")
    assert table.req_for_node(1, session="s-a") == "req-a"
    assert table.req_for_node(1, session="s-b") == "req-b"
    extra = forkserve_extra(speculative=False, node_id=1, parent_node=1, session="s-a")
    assert extra["forkserve_session"] == "s-a"
    assert table.release_node(1, session="s-a")
    assert table.req_for_node(1, session="s-a") is None
    assert table.req_for_node(1, session="s-b") == "req-b"
    assert table.releases == 1


def test_select_full_blocks_keeps_complete_last_page() -> None:
    blocks = [SimpleNamespace(block_id=i) for i in range(4)]
    # 64 tokens / page 16 → 4 full pages; do not drop the last.
    use, n = select_full_blocks(blocks, 16, 64)
    assert [b.block_id for b in use] == [0, 1, 2, 3]
    assert n == 64


def test_select_full_blocks_drops_partial_tail() -> None:
    blocks = [SimpleNamespace(block_id=i) for i in range(3)]
    use, n = select_full_blocks(blocks, 16, 40)  # 2 full + 8 remainder
    assert [b.block_id for b in use] == [0, 1]
    assert n == 32


def test_partition_fused_commit_tail_rides_decode() -> None:
    trunk = list(range(10))
    full = trunk + [99, 100]
    req = PrefillRequest(
        session=SessionId("s"),
        node_id=NodeId(1),
        tokens=(99, 100),
        speculative=False,
        page_ids=(),
        full_prompt=tuple(full),
    )
    sibling = PrefillRequest(
        session=SessionId("s"),
        node_id=NodeId(2),
        tokens=(7,),
        speculative=True,
        page_ids=(),
        full_prompt=tuple(trunk + [7]),
    )
    fused, rest = partition_fused([req, sibling], full)
    assert fused == [req]
    assert rest == [sibling]
    assert prompt_ids_of(req) == full


def test_split_pending_drops_spec_siblings() -> None:
    trunk = list(range(10))
    full = trunk + [99, 100]
    tail = PrefillRequest(
        session=SessionId("s"),
        node_id=NodeId(1),
        tokens=(99, 100),
        speculative=False,
        page_ids=(),
        full_prompt=tuple(full),
    )
    recov = PrefillRequest(
        session=SessionId("s"),
        node_id=NodeId(2),
        tokens=(7, 8),
        speculative=True,
        page_ids=(),
        full_prompt=tuple(trunk + [7, 8]),
    )
    fused, committed, dropped = split_pending_for_decode([tail, recov], [full])
    assert fused == [tail]
    assert committed == []
    assert dropped == [recov]


def test_drop_covered_prefills_kills_trunk_round() -> None:
    sid = SessionId("s")
    trunk = PrefillRequest(
        session=sid,
        node_id=NodeId(0),
        tokens=tuple(range(8)),
        speculative=False,
        page_ids=(),
        full_prompt=tuple(range(8)),
    )
    child_a = PrefillRequest(
        session=sid,
        node_id=NodeId(1),
        tokens=(8, 9),
        speculative=True,
        page_ids=(),
        full_prompt=tuple(range(10)),
    )
    child_b = PrefillRequest(
        session=sid,
        node_id=NodeId(2),
        tokens=(8, 11),
        speculative=True,
        page_ids=(),
        full_prompt=tuple(range(8)) + (11, 12),
    )
    kept = drop_covered_prefills([trunk, child_a, child_b])
    assert trunk not in kept
    assert child_a in kept and child_b in kept


def test_split_pending_fuses_covered_trunk() -> None:
    sid = SessionId("s")
    trunk = PrefillRequest(
        session=sid,
        node_id=NodeId(0),
        tokens=tuple(range(4)),
        speculative=False,
        page_ids=(),
        full_prompt=tuple(range(4)),
    )
    decode = tuple(range(8))
    fused, committed, dropped = split_pending_for_decode([trunk], [decode])
    assert fused == [trunk]
    assert committed == []
    assert dropped == []

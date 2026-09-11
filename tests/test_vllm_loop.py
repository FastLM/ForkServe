from types import SimpleNamespace

from forkserve.engine.vllm_loop import (
    CowBlockTable,
    committed_first,
    extra_of,
    forkserve_extra,
    is_speculative,
)


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
    table.pin_ro([10, 11])
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

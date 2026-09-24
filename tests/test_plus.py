from forkserve.api import Engine
from forkserve.config import ForkServeConfig
from forkserve.engine.mock import MockBackend
from forkserve.engine.vllm_loop import CowBlockTable
from forkserve.eval_plus import concurrency_sweep, micro_fanout, time_accuracy_curve
from forkserve.pages import PagePool, TokenKvStore
from forkserve.prune import BranchPruner, plus_config
from forkserve.spec_pool import plan_spec_pool
from forkserve.tree import ContextTree
from forkserve.types import NodeMode, SessionId


def test_lazy_abort_marks_dead_then_drain_frees() -> None:
    cfg = plus_config(ForkServeConfig(page_size=8, bytes_per_token=1.0))
    pool = PagePool(cfg, TokenKvStore())
    t = ContextTree(SessionId("s"), pool, cfg)
    t.open_root(tuple(range(32)))
    live0 = len(pool)
    child = t.fork(t.root, "tmp", tuple(range(200, 216)))  # type: ignore[arg-type]
    assert len(pool) > live0
    t.abort(child.id, lazy=True)
    assert child.mode is NodeMode.DEAD
    assert len(pool) > live0
    t.drain_lazy()
    assert len(pool) == live0


def test_engine_lazy_abort_does_not_block_generate() -> None:
    cfg = plus_config(ForkServeConfig(page_size=8, bytes_per_token=1.0, max_batched_tokens=2048))
    eng = Engine(MockBackend(cfg), cfg)
    h = eng.open("trunk history tokens")
    a = eng.fork(h.id, h.tip, "thought-0", "plan a")
    b = eng.fork(h.id, h.tip, "thought-1", "plan b")
    eng.abort(h.id, b)
    assert eng.tree(h.id).get(b).mode is NodeMode.DEAD
    eng.promote(h.id, a)
    out = eng.generate(h.id, 4)
    assert len(out) >= 1
    eng.close(h.id)


def test_pointer_swap_on_fork_no_copy() -> None:
    cfg = ForkServeConfig(page_size=8, bytes_per_token=1.0)
    pool = PagePool(cfg, TokenKvStore())
    t = ContextTree(SessionId("s"), pool, cfg)
    t.open_root(tuple(range(32)))
    copies = pool.cow_copies
    t.fork(t.root, "c", (1, 2, 3))  # type: ignore[arg-type]
    assert pool.pointer_swaps > 0
    assert pool.cow_copies == copies


def test_answer_stop_waits_for_the_number() -> None:
    from forkserve.answer_stop import answer_ready, stop_mode_for

    assert not answer_ready("reasoning ####", "gsm")
    assert not answer_ready("</think>", "gsm")
    assert not answer_ready("step #### 5\nthen keep going to the real total", "gsm")
    assert answer_ready("work\n#### 72\n", "gsm")
    assert not answer_ready("</think>\n#### 72", "gsm")
    assert answer_ready("</think>\n#### 72\nnext", "gsm")
    assert not answer_ready("therefore \\boxed{42}", "math")
    assert answer_ready("therefore \\boxed{42}\n", "math")
    assert not answer_ready("\\boxed{42", "math")
    assert answer_ready("def f():\n    return 1\ndef ", "code")
    assert stop_mode_for("gsm8k", "auto") == "gsm"
    assert stop_mode_for("game24", "") == ""


def test_known_winner_skips_sibling_prefill() -> None:
    from forkserve.prefill_prune import PrefillAction, PrefillPruner

    cfg = plus_config()
    plan = PrefillPruner(cfg, winner=0).plan(
        [
            "Thought 1: add the numbers and boxed the answer.",
            "Use substitution then combine.",
            "Count the groups first.",
            "Estimate then adjust.",
        ],
        token_counts=[32, 32, 32, 32],
    )
    assert plan.decisions[0].action is PrefillAction.PREFILL
    assert all(d.reason == "winner_selected" for d in plan.decisions[1:])
    assert plan.prefill_tokens == 32


def test_pruner_keeps_winner_drops_loop() -> None:
    cfg = plus_config()
    pruner = BranchPruner(cfg, winner=0)
    ranked = pruner.rank(
        [
            "Thought 1: add the numbers and boxed the answer.",
            "****loop****loop****loop****loop****loop",
            "undefined nan junk",
            "Use substitution then combine.",
        ]
    )
    assert ranked[0].keep
    assert not ranked[1].keep
    assert not ranked[2].keep


def test_spec_pool_gives_more_slots_than_apc() -> None:
    plan = plan_spec_pool(
        hbm_bytes=40 * (1 << 30),
        bytes_per_token=147_456.0,
        trunk=1024,
        residuals=[128, 128, 128, 128],
        k=4,
        base_batched_tokens=2048,
        spec_pool_frac=0.25,
    )
    assert plan.kv_saving > 0.5
    assert plan.slots_fs > plan.slots_apc
    assert plan.extra_batched_tokens > 0


def test_cow_table_lazy_release() -> None:
    table = CowBlockTable()
    table.store_snapshot(3, [[object()]], 16, 16, session="s")
    assert table.release_node(3, session="s", lazy=True)
    assert "s:3" in table.node_snap
    assert table.drain_releases() == 1
    assert "s:3" not in table.node_snap


def test_micro_plus_prunes_and_splits_fanout() -> None:
    base = micro_fanout(plus=False, sessions=4)
    plus = micro_fanout(plus=True, sessions=4)
    assert plus["system"] == "forkserve_plus"
    assert plus["pruned_branches"] >= 1
    assert plus["prefilled_branches"] >= 1
    assert plus["abort_mark_ms"] >= 0
    assert base["prefilled_branches"] >= plus["prefilled_branches"]


def test_time_accuracy_curve_and_concurrency() -> None:
    curve = time_accuracy_curve(
        [{"task_correct": [True, False, True, True], "decode_per_item": 100}]
    )
    assert curve[-1]["accuracy"] == 0.75
    assert curve[-1]["decode_tokens"] == 400
    short = time_accuracy_curve(
        [{"task_correct": [True, False, True, True], "decode_per_item": 100}],
        stop_frac=0.6,
    )
    assert short[-1]["decode_tokens"] == 240
    conc = concurrency_sweep(qps_grid=(8, 64, 256))
    assert conc[0]["slots_fs"] > conc[0]["slots_apc"]
    assert conc[-1]["fs_tok_s"] >= conc[-1]["apc_tok_s"]

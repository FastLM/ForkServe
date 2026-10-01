"""Advanced Prefill Pruning: hash skip, draft/early prune, disagg gate."""

from __future__ import annotations

from typing import Any

from forkserve.config import ForkServeConfig
from forkserve.disagg import DisaggPrefillConnector
from forkserve.prefill_prune import (
    PrefillAction,
    PrefillHashIndex,
    PrefillPruner,
    app_config,
)
from forkserve.prune import apply_admit_mode, plus_config

from experiments.prefill_prune_bench import (
    run_apc,
    run_app,
    run_decoding_prune,
    run_forkserve,
    run_prefill_on_decoding,
    run_suite,
)


def _cfg() -> ForkServeConfig:
    return plus_config(ForkServeConfig(page_size=8, bytes_per_token=1.0, prefill_us_per_token=12.0))


def test_hash_index_skips_published_prefix() -> None:
    idx = PrefillHashIndex(page_size=8)
    trunk = tuple(range(16))
    idx.publish(trunk)
    assert idx.lookup(trunk) == 16
    assert idx.lookup(trunk + (99, 100)) == 16
    assert idx.lookup(tuple(range(100, 116))) == 0


def test_app_draft_skips_illegal_keeps_winner() -> None:
    cfg = _cfg()
    plan = PrefillPruner(cfg, winner=0).plan(
        [
            "Thought 1: add the numbers and boxed the answer.",
            "****loop****loop****loop****loop****loop",
            "undefined nan junk",
            "Use substitution then combine.",
        ],
        token_counts=[32, 32, 32, 32],
    )
    assert plan.decisions[0].keep
    assert plan.decisions[0].action is PrefillAction.PREFILL
    assert plan.draft_skips + plan.early_aborts >= 2
    assert plan.prefill_tokens < 4 * 32
    assert plan.decisions[1].keep is False
    # Live sibling is above the score bar; APP does not drop it just
    # because it is not the designated winner.
    assert plan.decisions[3].keep


def test_app_hash_skip_on_replay() -> None:
    cfg = _cfg()
    idx = PrefillHashIndex(page_size=8)
    prompt = tuple(range(24))
    idx.publish(prompt)
    plan = PrefillPruner(cfg, winner=0, hash_index=idx).plan(
        [prompt],
        full_prompts=[prompt],
        token_counts=[8],
    )
    assert plan.decisions[0].action is PrefillAction.HASH_SKIP
    assert plan.decisions[0].work_tokens == 0
    assert plan.hash_skips == 1


def test_app_hash_partial_after_trunk_publish() -> None:
    cfg = _cfg()
    idx = PrefillHashIndex(page_size=8)
    trunk = tuple(range(16))
    residual = tuple(range(200, 208))
    idx.publish(trunk)
    plan = PrefillPruner(cfg, winner=0, hash_index=idx).plan(
        ["a careful algebraic substitution that finishes the proof"],
        full_prompts=[trunk + residual],
        token_counts=[8],
    )
    assert plan.decisions[0].action is PrefillAction.HASH_PARTIAL
    assert plan.decisions[0].work_tokens == 8
    assert plan.decisions[0].matched_tokens == 16


def test_disagg_gate_drops_pruned_and_hash_local() -> None:
    cfg = _cfg()
    idx = PrefillHashIndex(page_size=8)
    good = tuple(range(16))
    idx.publish(good)
    plan = PrefillPruner(cfg, winner=0, hash_index=idx).plan(
        [
            "Thought 1: compute carefully and box the answer.",
            "****loop****loop****loop****loop****loop",
        ],
        full_prompts=[good, tuple(range(80, 96))],
        token_counts=[16, 16],
    )
    xfer = DisaggPrefillConnector(cfg).gate(plan)
    assert xfer.inserted == 0  # hash-local winner + pruned loser
    assert xfer.dropped == 2
    assert xfer.shipped_tokens == 0


def test_disagg_ships_only_survivors() -> None:
    cfg = _cfg()
    plan = PrefillPruner(cfg, winner=0).plan(
        [
            "Thought 1: compute carefully and box the answer.",
            "****loop****loop****loop****loop****loop",
        ],
        token_counts=[32, 32],
    )
    xfer = DisaggPrefillConnector(cfg).gate(plan)
    assert xfer.inserted == 1
    assert xfer.shipped_tokens == 32
    assert xfer.dropped >= 1


def test_bench_app_beats_apc_on_prefill_and_kv() -> None:
    apc = run_apc(n_sessions=4, fanout=4, trunk_len=64, residual=16, replays=4)[0]
    fs = run_forkserve(n_parents=4, fanout=4, trunk_len=64, residual=16)[0]
    app = run_app(n_parents=4, fanout=4, trunk_len=64, residual=16)[0]
    assert app.prefill_tokens < apc.prefill_tokens
    assert app.prefill_ms < apc.prefill_ms
    assert app.peak_kv_tokens < apc.peak_kv_tokens
    assert fs.peak_kv_tokens < apc.peak_kv_tokens
    assert app.pruned + app.early_aborts >= 1
    assert app.transfer_tokens < apc.prefill_tokens


def test_decoding_prune_pays_for_losers_before_the_cut() -> None:
    """ESC, Speculative Rejection, and DPTS decode a prefix before they can drop."""
    app = run_app()[0]
    by = {row.method: row for row in run_decoding_prune()}
    assert set(by) == {"esc", "specrej", "dpts"}
    # Published decision prefix: DPTS mini-step, SR partial-reward horizon, ESC full budget.
    expect = {"dpts": (100, 20), "specrej": (256, 20), "esc": (512, 0)}
    for method, (step, pruned) in expect.items():
        row = by[method]
        assert row.extra["decision_tokens"] == step
        assert row.pruned == pruned
        assert row.prefill_tokens == 10 * 256
        assert row.extra["decode_until_cut"] == 10 * 4 * step
        assert row.extra["loser_tokens"] == 10 * 2 * step
        assert row.peak_kv_tokens == 10 * (256 + 4 * step)
        assert row.extra["loser_tokens"] > 0
        assert row.fanout_ms > app.fanout_ms
        assert row.peak_kv_tokens > app.peak_kv_tokens
    # Earliest decoding cut (DPTS, 100 tokens) is still later than withholding prefill.
    assert by["dpts"].fanout_ms > app.fanout_ms
    assert by["dpts"].extra["loser_tokens"] < by["specrej"].extra["loser_tokens"]
    assert by["specrej"].extra["loser_tokens"] < by["esc"].extra["loser_tokens"]


def test_prefill_on_decoding_still_cuts_the_wasted_prefix() -> None:
    """Draft admission removes the prefix a decoder would pay before its cut."""
    by = {row.method: row for row in run_prefill_on_decoding()}
    assert set(by) == {
        "esc",
        "esc+draft",
        "esc+app",
        "specrej",
        "specrej+draft",
        "specrej+app",
        "dpts",
        "dpts+draft",
        "dpts+app",
    }
    for method in ("esc", "specrej", "dpts"):
        base = by[method]
        draft = by[f"{method}+draft"]
        app = by[f"{method}+app"]
        # Text heuristic and the APP score bar both keep the two live thoughts
        # on this mix (illegal / loop sit at 0.02; live strategies sit near 1).
        assert draft.extra["admitted"] == 2
        assert app.extra["admitted"] == 2
        assert draft.extra["avoided_decode"] > 0
        assert app.decode_tokens == draft.decode_tokens < base.decode_tokens
        assert app.extra["e2e_tokens"] == draft.extra["e2e_tokens"] < base.extra["e2e_tokens"]
    # SR and DPTS already planned to drop those two. Draft only moves the cut
    # earlier, so the survivor count stays 2 and the shorter checkpoint saves less.
    assert by["dpts"].extra["kept"] == by["dpts+draft"].extra["kept"] == 2
    assert by["specrej+draft"].extra["kept"] == 2
    assert by["dpts+draft"].extra["avoided_decode"] < by["specrej+draft"].extra["avoided_decode"]
    assert by["specrej+draft"].extra["avoided_decode"] < by["esc+draft"].extra["avoided_decode"]
    # DPTS peak is the two survivors at the budget, so an earlier cut does not
    # shrink it. Default APP is the score bar, same two live thoughts as draft.
    assert by["dpts+draft"].peak_kv_tokens == by["dpts"].peak_kv_tokens
    assert by["dpts+app"].peak_kv_tokens == by["dpts"].peak_kv_tokens
    assert by["specrej+draft"].peak_kv_tokens < by["specrej"].peak_kv_tokens
    assert by["esc+draft"].peak_kv_tokens < by["esc"].peak_kv_tokens


def test_suite_writes_summary() -> None:
    report: dict[str, Any] = run_suite()
    summary = report["summary"]
    curve = report["decode_curve"]
    conc = report["concurrency"]
    assert set(summary["fanout_ms"]) >= {"apc", "forkserve", "app"}
    assert set(summary["decoding_prune"]) == {"esc", "specrej", "dpts"}
    assert summary["decoding_prune"]["dpts"]["loser_tokens"] > 0
    stacked = summary["prefill_on_decoding"]
    assert stacked["dpts+draft"]["e2e_cut_vs_base"] > 0
    assert stacked["dpts+app"]["e2e_tokens"] == stacked["dpts+draft"]["e2e_tokens"]
    assert summary["app_vs_apc_prefill"] > 0.3
    assert summary["app_vs_apc_peak_kv"] > 0.5
    assert conc["tok_s_gain"] > 0.0
    assert curve["app_tokens_to_target"] < curve["apc_tokens_to_target"]


def test_shared_prefix_prefills_the_tail_once() -> None:
    cfg = ForkServeConfig(
        page_size=8,
        prune_enabled=True,
        hash_prune=True,
        disagg_prefill=True,
        share_prefixes=True,
        gc_admit=False,
        skip_known_losers=False,
        prune_threshold=0.15,
    )
    trunk = tuple(range(16))
    shared = tuple(range(100, 116))
    a = trunk + shared + (1, 2, 3, 4)
    b = trunk + shared + (9, 8, 7, 6)
    plan = PrefillPruner(cfg, winner=0).plan(
        [a, b],
        full_prompts=[a, b],
        token_counts=[len(a) - len(trunk), len(b) - len(trunk)],
    )
    assert plan.decisions[0].work_tokens == len(a) - len(trunk)
    assert plan.decisions[1].action is PrefillAction.HASH_PARTIAL
    assert plan.decisions[1].work_tokens == 4
    assert plan.prefill_tokens == (len(a) - len(trunk)) + 4
    xfer = DisaggPrefillConnector(cfg).gate(plan)
    assert xfer.shipped_tokens == plan.prefill_tokens
    assert xfer.shipped_tokens < 2 * (len(a) - len(trunk))


def test_gc_admit_drops_low_value_siblings() -> None:
    cfg = plus_config(ForkServeConfig(page_size=8, bytes_per_token=1.0))
    apply_admit_mode(cfg, "top_m", keep_m=2)
    plan = PrefillPruner(cfg, winner=0).plan(
        [
            "Thought 1: add the numbers and boxed the answer.",
            "Use substitution then combine.",
            "Count the groups first.",
            "Estimate then adjust.",
        ],
        token_counts=[32, 32, 32, 32],
    )
    assert plan.decisions[0].keep
    assert sum(1 for d in plan.decisions if d.keep) == 2
    assert any(d.reason == "marginal_gc" for d in plan.decisions)
    assert plan.prefill_tokens == 64


def test_admit_mode_parameter_selects_rule() -> None:
    thoughts = [
        "Thought 1: add the numbers and boxed the answer.",
        "****loop****loop****loop****loop****loop",
        "undefined nan junk",
        "Use substitution then combine.",
    ]
    score = apply_admit_mode(plus_config(ForkServeConfig(page_size=8)), "score", threshold=0.45)
    winner = apply_admit_mode(plus_config(ForkServeConfig(page_size=8)), "winner")
    topm = apply_admit_mode(plus_config(ForkServeConfig(page_size=8)), "top_m", keep_m=2)
    frac = apply_admit_mode(plus_config(ForkServeConfig(page_size=8)), "alpha", alpha=0.5)
    kept = lambda cfg: [d.index for d in PrefillPruner(cfg, winner=0).plan(thoughts, token_counts=[32] * 4).decisions if d.keep]
    assert kept(score) == [0, 3]
    assert kept(winner) == [0]
    assert kept(topm) == [0, 3]
    assert 0 in kept(frac) and len(kept(frac)) == 2


def test_app_config_enables_layers() -> None:
    cfg = app_config()
    assert cfg.lazy_abort
    assert cfg.prune_enabled
    assert cfg.hash_prune
    assert cfg.disagg_prefill


def test_tot_multiturn_app_beats_apc_as_depth_grows() -> None:
    from experiments.tot_multiturn_bench import run_method, summarize

    rows = []
    for method in ("apc", "forkserve", "app"):
        rows.extend(
            run_method(method, sessions=4, branching=4, turns=3, trunk_len=64, residual=16, step=8)
        )
    summary = summarize(rows)
    assert summary["app"]["prefill_tokens"] < summary["apc"]["prefill_tokens"]
    assert summary["forkserve"]["prefill_tokens"] < summary["apc"]["prefill_tokens"]
    assert summary["app"]["peak_kv_last"] < summary["apc"]["peak_kv_last"]
    assert summary["forkserve"]["peak_kv_last"] < summary["apc"]["peak_kv_last"]
    assert summary["app"]["pinned_tokens"] > 0
    assert summary["app"]["next_miss_tokens"] < summary["forkserve"]["next_miss_tokens"]
    assert summary["app"]["prefill_cut_vs_apc"] > 0.3

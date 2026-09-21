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
from forkserve.prune import plus_config

from experiments.prefill_prune_bench import run_apc, run_app, run_forkserve, run_suite


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


def test_suite_writes_summary() -> None:
    report: dict[str, Any] = run_suite()
    summary = report["summary"]
    curve = report["decode_curve"]
    conc = report["concurrency"]
    assert set(summary["fanout_ms"]) >= {"apc", "forkserve", "app"}
    assert summary["app_vs_apc_prefill"] > 0.3
    assert summary["app_vs_apc_peak_kv"] > 0.5
    assert conc["tok_s_gain"] > 0.0
    assert curve["app_tokens_to_target"] < curve["apc_tokens_to_target"]


def test_app_config_enables_layers() -> None:
    cfg = app_config()
    assert cfg.lazy_abort
    assert cfg.prune_enabled
    assert cfg.hash_prune
    assert cfg.disagg_prefill

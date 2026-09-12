import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rl_improve", ROOT / "scripts" / "rl_improve.py")
assert spec and spec.loader
rl = importlib.util.module_from_spec(spec)
sys.modules["rl_improve"] = rl
spec.loader.exec_module(rl)


def _row(system: str, tp: int, workload: str, e2e: float, peak: int, ttft: float = 0.0) -> dict:
    return {
        "system": system,
        "tp": tp,
        "workload": workload,
        "e2e_ms": e2e,
        "ttft_from_obs_ms": ttft,
        "peak_kv_tokens": peak,
    }


def test_judge_success_when_kv_and_latency_win() -> None:
    rows = [
        _row("vllm_recompute", 2, "gsm8k", 700, 694),
        _row("vllm_apc", 2, "gsm8k", 680, 223),
        _row("forkserve", 2, "gsm8k", 650, 180),
    ]
    v = rl.judge_rows(rows)
    assert v.ok
    assert v.pairs[0].efficiency_beats
    assert not v.pairs[0].perf_drop


def test_judge_fails_on_latency_drop() -> None:
    rows = [
        _row("vllm_recompute", 2, "gsm8k", 700, 694),
        _row("vllm_apc", 2, "gsm8k", 680, 223),
        _row("forkserve", 2, "gsm8k", 900, 180),
    ]
    v = rl.judge_rows(rows, perf_drop=0.10)
    assert not v.ok
    assert v.pairs[0].efficiency_beats
    assert v.pairs[0].perf_drop


def test_judge_fails_when_peak_ties_apc() -> None:
    rows = [
        _row("vllm_recompute", 2, "gsm8k", 700, 694),
        _row("vllm_apc", 2, "gsm8k", 680, 223),
        _row("forkserve", 2, "gsm8k", 650, 223),
    ]
    v = rl.judge_rows(rows)
    assert not v.ok
    assert not v.pairs[0].efficiency_beats


def test_judge_fails_when_kv_not_better_than_recompute() -> None:
    rows = [
        _row("vllm_recompute", 2, "gsm8k", 700, 694),
        _row("vllm_apc", 2, "gsm8k", 680, 223),
        _row("forkserve", 2, "gsm8k", 680, 600),
    ]
    v = rl.judge_rows(rows, kv_gain=0.20)
    assert not v.ok
    assert not v.pairs[0].efficiency_beats


def test_judge_humaneval_cow_fanout_beats_recompute() -> None:
    """Honest ReAct: recompute clones both wrappers; CoW shares the trunk."""
    rows = [
        _row("vllm_recompute", 2, "humaneval", 8000, 2024, ttft=260),
        _row("vllm_apc", 2, "humaneval", 8000, 1112, ttft=210),
        _row("forkserve", 2, "humaneval", 8000, 900, ttft=200),
    ]
    v = rl.judge_rows(rows, kv_gain=0.20, perf_drop=0.10)
    assert v.ok
    assert v.pairs[0].efficiency_beats
    assert not v.pairs[0].perf_drop


def test_judge_fails_on_collapsed_decode() -> None:
    rows = [
        _row("vllm_recompute", 2, "humaneval", 8000, 1918, ttft=200),
        _row("vllm_apc", 2, "humaneval", 8000, 1059, ttft=200),
        _row("forkserve", 2, "humaneval", 8000, 1059, ttft=200),
    ]
    for r in rows:
        r["task_score"] = 0.25
        r["task_metric"] = "pass_at_1"
    rows[1]["task_score"] = 0.25
    rows[2]["quality_collapsed"] = True
    v = rl.judge_rows(rows, quality_min=0.05)
    assert not v.ok
    assert v.pairs[0].quality_drop


def test_judge_fails_on_quality_drop() -> None:
    rows = [
        _row("vllm_recompute", 2, "gsm8k", 700, 694),
        _row("vllm_apc", 2, "gsm8k", 680, 223),
        _row("forkserve", 2, "gsm8k", 650, 180),
    ]
    rows[1]["task_score"] = 0.75
    rows[1]["task_n"] = 4
    rows[1]["task_metric"] = "accuracy"
    rows[2]["task_score"] = 0.25
    rows[2]["task_n"] = 4
    rows[2]["task_metric"] = "accuracy"
    rows[2]["quality_vs"] = "vllm_apc"
    v = rl.judge_rows(rows, quality_min=0.05)
    assert not v.ok
    assert v.pairs[0].quality_drop
    assert "quality" in v.reason


def test_judge_ignores_missing_quality() -> None:
    rows = [
        _row("vllm_recompute", 2, "gsm8k", 700, 694),
        _row("vllm_apc", 2, "gsm8k", 680, 223),
        _row("forkserve", 2, "gsm8k", 650, 180),
    ]
    v = rl.judge_rows(rows, quality_min=0.05)
    assert v.ok
    assert not v.pairs[0].quality_drop


def test_decode_default_is_long_enough_for_answers() -> None:
    args = rl.parse_args([])
    assert args.decode >= 256
    assert args.gsm8k_decode >= 512


def test_judge_ignores_all_zero_and_one_item_noise() -> None:
    rows = [
        _row("vllm_recompute", 2, "gsm8k", 700, 694),
        _row("vllm_apc", 2, "gsm8k", 680, 223),
        _row("forkserve", 2, "gsm8k", 650, 180),
    ]
    for r in rows:
        r["task_score"] = 0.0
        r["task_n"] = 4
        r["task_metric"] = "accuracy"
    assert rl.judge_rows(rows, quality_min=0.05).ok
    rows[1]["task_score"] = 0.25
    # 0 vs 0.25 on n=4 is one item — slack is 1/n
    v = rl.judge_rows(rows, quality_min=0.05)
    assert v.ok
    assert not v.pairs[0].quality_drop


def test_humaneval_uses_ttft() -> None:
    rows = [
        _row("vllm_recompute", 2, "humaneval", 8000, 588, ttft=100),
        _row("vllm_apc", 2, "humaneval", 8000, 588, ttft=100),
        _row("forkserve", 2, "humaneval", 8000, 588, ttft=200),
    ]
    v = rl.judge_rows(rows, kv_gain=0.0, perf_drop=0.10)
    assert v.pairs[0].perf_drop
    assert v.pairs[0].fork_latency_ms == 200
    assert v.pairs[0].vllm_latency_ms == 100

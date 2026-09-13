"""Scan the full local GSM8K / Game24 / HumanEval files (no generation)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from forkserve.bench_tasks import (
    benchmarks_dir,
    load_game24,
    load_gsm8k,
    load_humaneval,
)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    mid = len(ys) // 2
    if len(ys) % 2:
        return float(ys[mid])
    return 0.5 * (ys[mid - 1] + ys[mid])


def _bucket_steps(n: int) -> str:
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    return "4+"


def _bucket_game24(rate: float) -> str:
    if rate < 0:
        return "unknown"
    if rate >= 90:
        return "easy>=90%"
    if rate >= 50:
        return "medium50-90%"
    return "hard<50%"


def inventory(limit: int = 0) -> dict[str, Any]:
    """Read every line of the local jsonl/csv and summarize the sets."""
    root = benchmarks_dir()
    gsm = load_gsm8k(limit)
    game = load_game24(limit)
    he = load_humaneval(limit)

    gsm_words = [float(len(p.question.split())) for p in gsm]
    gsm_steps = [int(p.n_steps) for p in gsm]
    gsm_ans: list[float] = []
    for p in gsm:
        try:
            gsm_ans.append(abs(float(p.answer)))
        except ValueError:
            pass
    step_hist = Counter(_bucket_steps(n) for n in gsm_steps)

    game_rates = [p.solved_rate for p in game if p.solved_rate >= 0]
    game_hist = Counter(_bucket_game24(p.solved_rate) for p in game)

    he_prompt = [float(len(p.prompt)) for p in he]
    he_asserts = [float(p.tests.count("assert")) for p in he]

    return {
        "benchmarks_dir": str(root),
        "files": {
            "gsm8k": str(root / "gsm8k" / "test.jsonl"),
            "game24": str(root / "game24" / "24.csv"),
            "humaneval": str(root / "humaneval" / "HumanEval.jsonl"),
        },
        "counts": {
            "gsm8k_test": len(gsm),
            "game24": len(game),
            "humaneval": len(he),
            "total": len(gsm) + len(game) + len(he),
        },
        "gsm8k": {
            "split": "test",
            "n": len(gsm),
            "question_words_mean": _mean(gsm_words),
            "question_words_median": _median(gsm_words),
            "calc_steps_mean": _mean([float(x) for x in gsm_steps]),
            "calc_steps_median": _median([float(x) for x in gsm_steps]),
            "calc_steps_hist": dict(step_hist),
            "answer_abs_median": _median(gsm_ans),
            "first_ids": [p.item_id for p in gsm[:3]],
            "first_golds": [p.answer for p in gsm[:3]],
        },
        "game24": {
            "n": len(game),
            "solved_rate_mean": _mean(game_rates),
            "solved_rate_median": _median(game_rates),
            "difficulty_hist": dict(game_hist),
            "first_ids": [p.item_id for p in game[:3]],
        },
        "humaneval": {
            "n": len(he),
            "prompt_chars_mean": _mean(he_prompt),
            "prompt_chars_median": _median(he_prompt),
            "assert_mean": _mean(he_asserts),
            "first_ids": [p.item_id for p in he[:3]],
            "first_entry_points": [p.entry_point for p in he[:3]],
        },
    }


def dump_inventory(path: Path, limit: int = 0) -> dict[str, Any]:
    data = inventory(limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data


def bucket_results(
    workload: str,
    item_ids: list[str],
    correct: list[bool],
    *,
    gsm_steps: list[int] | None = None,
    game_rates: list[float] | None = None,
) -> dict[str, Any]:
    """Accuracy by dataset difficulty slice (full-set analysis)."""
    n = min(len(item_ids), len(correct))
    groups: dict[str, list[bool]] = {}
    for i in range(n):
        if workload == "gsm8k" and gsm_steps is not None and i < len(gsm_steps):
            key = f"steps_{_bucket_steps(gsm_steps[i])}"
        elif workload == "game24" and game_rates is not None and i < len(game_rates):
            key = _bucket_game24(game_rates[i])
        else:
            key = "all"
        groups.setdefault(key, []).append(bool(correct[i]))
    out = {}
    for key, flags in sorted(groups.items()):
        out[key] = {"n": len(flags), "score": sum(flags) / len(flags) if flags else 0.0}
    return out

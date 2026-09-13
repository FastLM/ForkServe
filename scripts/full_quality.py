#!/usr/bin/env python3
"""Full-file forest comparison: ToT / ReAct fan-out, then official metrics.

Default is the same multi-branch recipe as ``rl_improve`` (shared trunk,
branching thoughts or wrap+recovery, abort losers, winner decode). That is
what peak_kv and fanout_ms measure. ``--quality-only`` is single-path and
must not be used for serving efficiency.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("FORKSERVE_ROOT", Path(__file__).resolve().parent.parent))
LOG = ROOT / "logs" / "full_quality"


def log(msg: str) -> None:
    from time import strftime

    print(strftime("[%F %T]"), msg, flush=True)


def attach_buckets(report: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from forkserve.bench_tasks import load_game24, load_gsm8k
    from forkserve.dataset_stats import bucket_results

    gsm = load_gsm8k(0)
    game = load_game24(0)
    gsm_steps = {p.item_id: p.n_steps for p in gsm}
    game_rates = {p.item_id: p.solved_rate for p in game}
    for row in report.get("rows") or []:
        ids = list(row.get("item_ids") or [])
        flags = list(row.get("task_correct") or [])
        wl = str(row.get("workload") or "")
        if not ids or not flags:
            continue
        if wl == "gsm8k":
            row["score_by_slice"] = bucket_results(
                wl, ids, flags, gsm_steps=[gsm_steps.get(i, 0) for i in ids]
            )
        elif wl == "game24":
            row["score_by_slice"] = bucket_results(
                wl, ids, flags, game_rates=[game_rates.get(i, -1.0) for i in ids]
            )
        else:
            row["score_by_slice"] = bucket_results(wl, ids, flags)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Full-set forest bench + official metrics")
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--model", default=os.environ.get("FORKSERVE_MODEL", str(Path.home() / "models/Qwen3-14B")))
    p.add_argument("--systems", default="vllm_recompute,vllm_apc,forkserve")
    p.add_argument("--tp", default="2")
    p.add_argument("--workloads", default="gsm8k,game24,humaneval")
    p.add_argument("--limit", type=int, default=0, help="0 = entire jsonl/csv")
    p.add_argument("--chunk", type=int, default=8, help="sessions per forest generate")
    p.add_argument("--decode", type=int, default=256)
    p.add_argument("--gsm8k-decode", type=int, default=512)
    p.add_argument(
        "--quality-only",
        action="store_true",
        help="single-path generate (no fan-out); not a serving comparison",
    )
    p.add_argument("--out", default=str(LOG / "eval.json"))
    args = p.parse_args(argv)

    LOG.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from forkserve.dataset_stats import dump_inventory

    inv_path = LOG / "inventory.json"
    inv = dump_inventory(inv_path, args.limit)
    log(f"inventory {inv['counts']} -> {inv_path}")
    print(json.dumps(inv, indent=2), flush=True)
    if args.inventory_only:
        return 0

    cmd = [
        sys.executable,
        "-m",
        "forkserve.bench",
        "--model",
        args.model,
        "--systems",
        args.systems,
        "--tp",
        args.tp,
        "--workloads",
        args.workloads,
        "--limit",
        str(args.limit),
        "--chunk",
        str(args.chunk),
        "--decode",
        str(args.decode),
        "--gsm8k-decode",
        str(args.gsm8k_decode),
        "--out",
        args.out,
    ]
    if args.quality_only:
        cmd.append("--quality-only")
        log("WARNING: --quality-only zeros peak_kv/fanout (single-path)")
    progress_log = LOG / "progress.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["FORKSERVE_PROGRESS_LOG"] = str(progress_log)
    cmd.extend(["--progress-log", str(progress_log)])
    log("exec: " + " ".join(cmd))
    log(f"progress -> {progress_log}  (tail -f this file)")
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        return proc.returncode
    dest = Path(args.out)
    report = json.loads(dest.read_text())
    report["inventory"] = inv
    attach_buckets(report)
    dest.write_text(json.dumps(report, indent=2))
    log(f"wrote {dest}")
    print(report.get("table") or "", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Parameter grid for PD-Prune thresholds (paper sensitivity).

No full (tau_pre x tau_dec x t0 x alpha x k) grid exists in the repo yet.
This module enumerates the grid, can --list it, and can launch
``prune_ablation_gpu.py`` jobs one cell at a time (or emit a JSON plan).

Grid axes (defaults match the paper operating point and nearby probes)
--------------------------------------------------------------------
* tau_pre  : prefill admission bar          (default 0.05,0.15,0.30,0.45)
* tau_dec  : decode / probe bar             (default 0.30,0.45,0.60)
* t0       : probe length                   (default 64,128,256)
* alpha    : early-abort fraction           (default 0.10,0.20,0.30)
* k        : fan-out                        (default 4,8,16,32)

Usage
-----
  # Print the cartesian product (no GPU)
  python -m experiments.grid_search_params --list

  # Write a JSON plan for a driver script
  python -m experiments.grid_search_params --plan logs/grid_search/plan.json

  # Run one cell through prune_ablation_gpu (GPU)
  python -m experiments.grid_search_params --run-cell 0 \\
      --model /path/to/Qwen3-8B --out logs/grid_search/cell0.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]


def parse_floats(text: str) -> tuple[float, ...]:
    vals = tuple(float(p.strip()) for p in text.split(",") if p.strip())
    if not vals:
        raise ValueError(f"empty float list: {text!r}")
    return vals


def parse_ints(text: str) -> tuple[int, ...]:
    vals = tuple(int(p.strip()) for p in text.split(",") if p.strip())
    if not vals or any(v < 1 for v in vals):
        raise ValueError(f"need positive ints: {text!r}")
    return vals


def build_grid(
    *,
    tau_pre: Sequence[float],
    tau_dec: Sequence[float],
    t0: Sequence[int],
    alpha: Sequence[float],
    ks: Sequence[int],
    methods: Sequence[str] = ("esc",),
    n: int = 16,
    budget: int = 512,
    workload: str = "gsm8k",
) -> list[dict[str, Any]]:
    """Cartesian product. Each cell is one GPU ablation job descriptor."""
    cells: list[dict[str, Any]] = []
    for i, (tp, td, probe, a, k, method) in enumerate(
        itertools.product(tau_pre, tau_dec, t0, alpha, ks, methods)
    ):
        cells.append(
            {
                "cell": i,
                "tag": "grid",
                "method": method,
                "policy": "app",
                "workload": workload,
                "n": int(n),
                "k": int(k),
                "budget": int(budget),
                "tau": 256,  # SR horizon (unused when method=esc)
                "dpts_step": 100,
                "alpha": float(a),
                "prefill_threshold": float(tp),
                "decode_threshold": float(td),
                "probe": int(probe),
                "threshold": float(tp),
                "admit_mode": "score",
                "mix": "hopeless",
            }
        )
    return cells


def write_plan(path: Path, cells: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_cells": len(cells), "cells": cells}, indent=2))
    print(f"wrote {path} ({len(cells)} cells)")


def run_cell(
    cell: dict[str, Any],
    *,
    model: str,
    tp: int,
    out: Path,
    python: str,
    extra: Sequence[str] = (),
) -> int:
    """Launch prune_ablation_gpu for one grid cell via --prefill + env overrides.

    Prefill sweep alone does not vary tau_dec / t0 / alpha; we pass those
    through environment variables consumed by the ablation config path when
    present, and also stamp them into the output JSON sidecar.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out.with_suffix(".cell.json")
    sidecar.write_text(json.dumps(cell, indent=2))

    env = os.environ.copy()
    env["FORKSERVE_PREFILL_THRESHOLD"] = str(cell["prefill_threshold"])
    env["FORKSERVE_DECODE_THRESHOLD"] = str(cell["decode_threshold"])
    env["FORKSERVE_PROBE"] = str(cell["probe"])
    env["FORKSERVE_EARLY_ABORT_ALPHA"] = str(cell["alpha"])
    env["FORKSERVE_ADMIT_MODE"] = "score"

    cmd = [
        python,
        "-u",
        "-m",
        "experiments.prune_ablation_gpu",
        "--model",
        model,
        "--tp",
        str(tp),
        "--prefill",
        "--ks",
        str(cell["k"]),
        "--prefill-thresholds",
        str(cell["prefill_threshold"]),
        "--out",
        str(out),
        *extra,
    ]
    print("exec:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PD-Prune parameter grid search")
    p.add_argument("--tau-pre", default="0.05,0.15,0.30,0.45")
    p.add_argument("--tau-dec", default="0.30,0.45,0.60")
    p.add_argument("--t0", default="64,128,256")
    p.add_argument("--alpha", default="0.10,0.20,0.30")
    p.add_argument("--ks", default="4,8,16,32")
    p.add_argument("--methods", default="esc", help="comma list: esc,specrej,dpts")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--workload", default="gsm8k", choices=("gsm8k", "math"))
    p.add_argument("--list", action="store_true")
    p.add_argument("--plan", default="", help="write JSON plan path")
    p.add_argument("--run-cell", type=int, default=-1, help="run this cell index")
    p.add_argument("--model", default=os.environ.get("FORKSERVE_MODEL", ""))
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--out", default="logs/grid_search/cell.json")
    p.add_argument(
        "--python",
        default=os.environ.get("FORKSERVE_PYTHON", sys.executable),
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    cells = build_grid(
        tau_pre=parse_floats(args.tau_pre),
        tau_dec=parse_floats(args.tau_dec),
        t0=parse_ints(args.t0),
        alpha=parse_floats(args.alpha),
        ks=parse_ints(args.ks),
        methods=methods,
        n=args.n,
        budget=args.budget,
        workload=args.workload,
    )
    if args.list:
        print(f"{len(cells)} cells")
        for c in cells:
            print(
                f"[{c['cell']:4d}] k={c['k']:<2} method={c['method']:<8} "
                f"tau_pre={c['prefill_threshold']:<4g} "
                f"tau_dec={c['decode_threshold']:<4g} "
                f"t0={c['probe']:<3} alpha={c['alpha']:<4g}"
            )
        return 0
    if args.plan:
        write_plan(Path(args.plan), cells)
        return 0
    if args.run_cell >= 0:
        if not args.model:
            print("ERROR: --model required for --run-cell", file=sys.stderr)
            return 2
        if args.run_cell >= len(cells):
            print(f"ERROR: cell {args.run_cell} out of range ({len(cells)})", file=sys.stderr)
            return 2
        return run_cell(
            cells[args.run_cell],
            model=args.model,
            tp=args.tp,
            out=Path(args.out),
            python=args.python,
        )
    print("Pass --list, --plan PATH, or --run-cell N", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

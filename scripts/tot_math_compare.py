#!/usr/bin/env python3
"""Run ToT math serving comparison: vLLM APC vs ForkServe vs APP, then summarize."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("FORKSERVE_ROOT", Path(__file__).resolve().parent.parent))
OUT = ROOT / "logs" / "tot_math"
PYTHON = sys.executable
MODEL = os.environ.get("FORKSERVE_MODEL", str(Path.home() / "models/Qwen3-14B"))
TP = os.environ.get("FORKSERVE_TP", "2")
SYSTEMS = os.environ.get("FORKSERVE_SYSTEMS", "vllm_apc,forkserve,forkserve_plus")
LIMIT = int(os.environ.get("TOT_MATH_LIMIT", "48"))
CHUNK = int(os.environ.get("TOT_MATH_CHUNK", "8"))
SWEEP_LIMIT = int(os.environ.get("TOT_MATH_SWEEP_LIMIT", "16"))
WORKLOADS = "gsm8k,svamp,math500,aime,amc23,game24"
SWEEP_WL = "gsm8k,math500,game24"


def log(msg: str) -> None:
    from time import strftime

    print(strftime("[%F %T]"), msg, flush=True)


def run_bench(tag: str, extra: list[str], out: Path) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    progress = OUT / f"{tag}.progress.log"
    cmd = [
        PYTHON,
        "-m",
        "forkserve.bench",
        "--model",
        MODEL,
        "--systems",
        SYSTEMS,
        "--tp",
        TP,
        "--chunk",
        str(CHUNK),
        "--out",
        str(out),
        "--progress-log",
        str(progress),
        *extra,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["FORKSERVE_PROGRESS_LOG"] = str(progress)
    log("exec: " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    log(f"{tag} rc={proc.returncode} -> {out}")
    return proc.returncode


def _load_summarize():
    import importlib.util

    path = ROOT / "scripts" / "summarize_tot_math.py"
    spec = importlib.util.spec_from_file_location("summarize_tot_math", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.write_summary


def main() -> int:
    write_summary = _load_summarize()

    OUT.mkdir(parents=True, exist_ok=True)
    main_out = OUT / "eval_b4.json"
    rc = run_bench(
        "b4",
        [
            "--workloads",
            WORKLOADS,
            "--limit",
            str(LIMIT),
            "--branching",
            "4",
            "--decode",
            "256",
        ],
        main_out,
    )
    if rc != 0:
        return rc

    for b in (2, 6):
        sweep_out = OUT / f"eval_b{b}_sweep.json"
        rc = run_bench(
            f"b{b}",
            [
                "--workloads",
                SWEEP_WL,
                "--limit",
                str(SWEEP_LIMIT),
                "--branching",
                str(b),
                "--decode",
                "256",
            ],
            sweep_out,
        )
        if rc != 0:
            return rc

    summary = write_summary(OUT)
    log(f"summary -> {OUT / 'summary.json'}")
    print(json.dumps(summary.get("headline") or summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""4-agent multiturn ToT on the math slices: APC vs ForkServe vs APP.

Each batch is one problem with k=4 agents (Tree-of-Thoughts depth ``turns``).
Intermediate turns every agent emits ``step`` tokens; losers abort; the
winner spine is the parent of the next fan-out. Final turn is the published
winner decode. Same gold metrics as the single-turn forest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("FORKSERVE_ROOT", Path(__file__).resolve().parent.parent))
OUT = ROOT / "logs" / "tot_mt"
PYTHON = sys.executable
MODEL = os.environ.get("FORKSERVE_MODEL", str(Path.home() / "models/Qwen3-4B"))
TP = os.environ.get("FORKSERVE_TP", "2")
SYSTEMS = os.environ.get("FORKSERVE_SYSTEMS", "vllm_apc,forkserve,forkserve_plus")
LIMIT = int(os.environ.get("TOT_MT_LIMIT", "16"))
# One problem per generate: the batch is exactly the 4 agents.
CHUNK = int(os.environ.get("TOT_MT_CHUNK", "1"))
TURNS = int(os.environ.get("TOT_MT_TURNS", "3"))
STEP = int(os.environ.get("TOT_MT_STEP", "32"))
BRANCHING = int(os.environ.get("TOT_MT_BRANCHING", "4"))
WORKLOADS = os.environ.get(
    "TOT_MT_WORKLOADS",
    "gsm8k,svamp,math500,aime,amc23,game24",
)
DECODE = os.environ.get("TOT_MT_DECODE", "256")


def log(msg: str) -> None:
    from time import strftime

    print(strftime("[%F %T]"), msg, flush=True)


def run_bench(tag: str, extra: list[str], out: Path, *, turns: int | None = None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    depth = TURNS if turns is None else turns
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
        "--turns",
        str(depth),
        "--step-decode",
        str(STEP),
        "--branching",
        str(BRANCHING),
        "--out",
        str(out),
        "--progress-log",
        str(progress),
        *extra,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["FORKSERVE_PROGRESS_LOG"] = str(progress)
    env["FORKSERVE_TURNS"] = str(depth)
    log("exec: " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    log(f"{tag} rc={proc.returncode} -> {out}")
    return proc.returncode


def _load_summarize():
    import importlib.util

    path = ROOT / "scripts" / "summarize_tot_mt.py"
    spec = importlib.util.spec_from_file_location("summarize_tot_mt", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_summary


def main() -> int:
    write_summary = _load_summarize()
    OUT.mkdir(parents=True, exist_ok=True)
    main_out = OUT / "eval.json"
    rc = run_bench(
        "mt",
        [
            "--workloads",
            WORKLOADS,
            "--limit",
            str(LIMIT),
            "--decode",
            str(DECODE),
        ],
        main_out,
    )
    if rc != 0:
        return rc
    if os.environ.get("TOT_MT_SWEEP", "").strip() in ("1", "true", "yes"):
        for depth in (2, 4):
            sweep_out = OUT / f"eval_d{depth}.json"
            cmd_rc = run_bench(
                f"d{depth}",
                [
                    "--workloads",
                    "gsm8k,math500,game24",
                    "--limit",
                    str(min(LIMIT, 8)),
                    "--decode",
                    str(DECODE),
                ],
                sweep_out,
                turns=depth,
            )
            if cmd_rc != 0:
                return cmd_rc
    summary = write_summary(OUT)
    log(f"summary -> {OUT / 'summary.json'}")
    print(json.dumps(summary.get("headline") or summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

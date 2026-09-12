#!/usr/bin/env python3
"""Claim visible GPUs the instant they are free, then hold or run the bench.

Mirrors the shared-box pattern: poll nvidia-smi, then immediately allocate
VRAM on every visible device so a slower watcher cannot slip in.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


def log(msg: str) -> None:
    print(time.strftime("[%F %T]"), msg, flush=True)


def query_used_mib() -> list[int]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used: list[int] = []
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    want = (
        {int(x) for x in vis.split(",") if x.strip().isdigit()}
        if vis
        else None
    )
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        idx, mem = int(parts[0]), int(parts[1])
        if want is None or idx in want:
            used.append(mem)
    return used


def others_on_gpu() -> list[tuple[int, str]]:
    """Compute apps that are not this process (or its children)."""
    me = os.getpid()
    kids = set()
    try:
        out = subprocess.check_output(["pgrep", "-P", str(me)], text=True)
        kids = {int(x) for x in out.split() if x.isdigit()}
    except subprocess.CalledProcessError:
        pass
    mine = {me, *kids}
    rows: list[tuple[int, str]] = []
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except subprocess.CalledProcessError:
        return rows
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid in mine:
            continue
        rows.append((pid, parts[1] if len(parts) > 1 else ""))
    return rows


def bench_lock_path() -> str:
    root = os.environ.get("FORKSERVE_ROOT", os.path.expanduser("~/ForkServe"))
    return os.environ.get(
        "FORKSERVE_BENCH_LOCK",
        os.path.join(root, "logs", ".forkserve_bench.lock"),
    )


def bench_running() -> bool:
    return os.path.isfile(bench_lock_path())


def gpus_idle(max_used_mib: int) -> bool:
    if bench_running():
        return False
    if others_on_gpu():
        return False
    used = query_used_mib()
    return bool(used) and all(u <= max_used_mib for u in used)


def wait_idle(poll_sec: float, max_used_mib: int) -> None:
    last = ""
    while True:
        others = others_on_gpu()
        used = query_used_mib()
        msg = f"used_mib={used} others={others[:4]} bench_lock={bench_running()}"
        if msg != last:
            log(f"waiting {msg}")
            last = msg
        if gpus_idle(max_used_mib):
            log("GPUs idle — occupying now")
            return
        time.sleep(poll_sec)


def occupy(fraction: float) -> list[object]:
    import torch

    n = torch.cuda.device_count()
    if n <= 0:
        raise SystemExit("no CUDA devices visible")
    held: list[object] = []
    for i in range(n):
        free, total = torch.cuda.mem_get_info(i)
        nbytes = max(int(free * fraction), 1 << 20)
        nelem = nbytes // 4
        x = torch.empty(nelem, dtype=torch.float32, device=f"cuda:{i}")
        x.fill_(1.0)
        held.append(x)
        log(
            f"held gpu{i} {nbytes / (1024 * 1024):.0f} MiB "
            f"({fraction:.0%} of {free / (1024 * 1024):.0f} MiB free / "
            f"{total / (1024 * 1024):.0f} MiB total)"
        )
    return held


def main() -> int:
    p = argparse.ArgumentParser(description="Wait for idle GPUs, then occupy them")
    p.add_argument("--poll-sec", type=float, default=1.0)
    p.add_argument("--idle-mib", type=int, default=1024)
    p.add_argument("--fraction", type=float, default=0.92)
    p.add_argument(
        "--hold-only",
        action="store_true",
        help="keep VRAM reserved until SIGTERM (do not start the bench)",
    )
    p.add_argument("--hold-sec", type=float, default=3.0, help="hold before handing off to bench")
    args, rest = p.parse_known_args()

    wait_idle(args.poll_sec, args.idle_mib)
    bufs = occupy(args.fraction)

    stop = {"flag": False}

    def _stop(signum, _frame):  # type: ignore[no-untyped-def]
        log(f"got signal {signum}, releasing")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    if args.hold_only:
        log("HOLDING all GPUs (kill this process to release)")
        while not stop["flag"]:
            time.sleep(1.0)
        del bufs
        return 0

    log(f"holding {args.hold_sec:.1f}s so the claim sticks, then starting bench")
    t0 = time.time()
    while time.time() - t0 < args.hold_sec and not stop["flag"]:
        time.sleep(0.2)
    del bufs
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    cmd = rest[1:] if rest and rest[0] == "--" else rest
    if not cmd:
        root = os.environ.get("FORKSERVE_ROOT", os.path.expanduser("~/ForkServe"))
        model = os.environ.get("FORKSERVE_MODEL", os.path.expanduser("~/models/Qwen3-8B"))
        tp = os.environ.get("FORKSERVE_TP", "2,4")
        cmd = [
            sys.executable,
            "-m",
            "forkserve.bench",
            "--tp",
            tp,
            "--model",
            model,
            "--out",
            os.path.join(root, "logs/bench_gpu.json"),
        ]
    log("exec: " + " ".join(cmd))
    os.chdir(os.environ.get("FORKSERVE_ROOT", os.path.expanduser("~/ForkServe")))
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    raise SystemExit(main())

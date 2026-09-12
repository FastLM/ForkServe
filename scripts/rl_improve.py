#!/usr/bin/env python3
"""Occupy GPUs → bench ForkServe vs vLLM → if we lose, occupy again and edit.

Loop
----
1. Release any hold, occupy the cards, then immediately run ``forkserve.bench``.
2. Judge ForkServe against vLLM:

   * **efficiency** = lower ``peak_kv`` than ``vllm_recompute`` by ``--kv-gain``,
     and not worse than ``vllm_apc``.
   * **perf drop** = latency worse than the vLLM baseline by ``--perf-drop``.
     HumanEval / ReAct use ``ttft_from_obs_ms``; others use ``e2e_ms``.

3. Goal: efficiency beats vLLM **and** no clear latency drop. Otherwise hold
   the GPUs and let Cursor change code, then go back to step 1.

Cursor edit (while occupying)
-----------------------------
* ``--edit cursor`` — ``cursor agent -p`` (needs ``CURSOR_API_KEY``).
* ``--edit wait`` — write ``logs/rl_improve/NEXT.md`` and wait for
  ``logs/rl_improve/CONTINUE`` (edit in this IDE, then ``touch`` the file).
* ``--edit both`` (default) — try Cursor agent, fall back to wait.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("FORKSERVE_ROOT", Path(__file__).resolve().parent.parent))
LOG_DIR = ROOT / "logs" / "rl_improve"
CONTINUE_NAME = "CONTINUE"
LATENCY_WORKLOADS = {"humaneval", "react"}


def log(msg: str) -> None:
    print(time.strftime("[%F %T]"), msg, flush=True)


def latency_of(row: dict[str, Any]) -> float:
    if row.get("workload") in LATENCY_WORKLOADS:
        return float(row.get("ttft_from_obs_ms") or row.get("e2e_ms") or 0.0)
    return float(row.get("e2e_ms") or 0.0)


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    for r in rows:
        out[(str(r["system"]), int(r["tp"]), str(r["workload"]))] = r
    return out


@dataclass
class PairJudge:
    tp: int
    workload: str
    fork_latency_ms: float
    vllm_latency_ms: float
    fork_peak_kv: int
    vllm_recompute_peak_kv: int
    vllm_apc_peak_kv: int
    efficiency_beats: bool
    perf_drop: bool
    notes: str = ""

    @property
    def needs_improve(self) -> bool:
        return (not self.efficiency_beats) or self.perf_drop


@dataclass
class RoundVerdict:
    ok: bool
    pairs: list[PairJudge] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "pairs": [asdict(p) for p in self.pairs],
        }


def judge_rows(
    rows: list[dict[str, Any]],
    *,
    kv_gain: float = 0.20,
    perf_drop: float = 0.10,
    peak_slack: float = 0.05,
) -> RoundVerdict:
    """Return whether ForkServe already beats vLLM without a clear latency drop."""
    idx = index_rows(rows)
    keys = {(int(r["tp"]), str(r["workload"])) for r in rows if r.get("system") == "forkserve"}
    if not keys:
        return RoundVerdict(ok=False, reason="no forkserve rows")

    pairs: list[PairJudge] = []
    for tp, wl in sorted(keys):
        fs = idx.get(("forkserve", tp, wl))
        rec = idx.get(("vllm_recompute", tp, wl))
        apc = idx.get(("vllm_apc", tp, wl))
        if fs is None or rec is None:
            pairs.append(
                PairJudge(
                    tp=tp,
                    workload=wl,
                    fork_latency_ms=0.0,
                    vllm_latency_ms=0.0,
                    fork_peak_kv=0,
                    vllm_recompute_peak_kv=0,
                    vllm_apc_peak_kv=0,
                    efficiency_beats=False,
                    perf_drop=True,
                    notes="missing forkserve or vllm_recompute row",
                )
            )
            continue

        base = apc or rec
        fork_lat = latency_of(fs)
        vllm_lat = latency_of(base)
        fork_pk = int(fs.get("peak_kv_tokens") or 0)
        rec_pk = int(rec.get("peak_kv_tokens") or 0)
        apc_pk = int((apc or rec).get("peak_kv_tokens") or 0)

        beats_recompute = rec_pk > 0 and fork_pk <= rec_pk * (1.0 - kv_gain)
        not_worse_than_apc = apc_pk <= 0 or fork_pk <= apc_pk * (1.0 + peak_slack)
        efficiency_beats = beats_recompute and not_worse_than_apc
        dropped = vllm_lat > 0 and fork_lat > vllm_lat * (1.0 + perf_drop)

        notes = (
            f"lat {fork_lat:.1f} vs {vllm_lat:.1f} "
            f"({((fork_lat / vllm_lat) - 1.0) * 100 if vllm_lat else 0:+.1f}%); "
            f"peak_kv {fork_pk} vs recompute {rec_pk} / apc {apc_pk}"
        )
        pairs.append(
            PairJudge(
                tp=tp,
                workload=wl,
                fork_latency_ms=fork_lat,
                vllm_latency_ms=vllm_lat,
                fork_peak_kv=fork_pk,
                vllm_recompute_peak_kv=rec_pk,
                vllm_apc_peak_kv=apc_pk,
                efficiency_beats=efficiency_beats,
                perf_drop=dropped,
                notes=notes,
            )
        )

    bad = [p for p in pairs if p.needs_improve]
    if not bad:
        return RoundVerdict(ok=True, pairs=pairs, reason="efficiency beats vLLM and latency is within slack")
    bits = []
    if any(not p.efficiency_beats for p in bad):
        bits.append("efficiency did not beat vLLM")
    if any(p.perf_drop for p in bad):
        bits.append("clear latency drop vs vLLM")
    return RoundVerdict(ok=False, pairs=pairs, reason="; ".join(bits))


def write_cursor_prompt(round_id: int, verdict: RoundVerdict, bench_path: Path) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# ForkServe RL improve — round {round_id}",
        "",
        f"Verdict: **FAIL** — {verdict.reason}",
        f"Bench JSON: `{bench_path}`",
        "",
        "You are editing the ForkServe repo on this machine. GPUs are occupied",
        "by `occupy_gpus.py` so another user cannot steal them. Do **not** start",
        "another GPU bench yourself.",
        "",
        "## Goal",
        "Make ForkServe more efficient than vLLM (lower peak KV than",
        "`vllm_recompute`, not worse than `vllm_apc`) **and** keep latency",
        "within the configured slack of the vLLM baseline (APC if present).",
        "",
        "Focus on serving path: CoW / two-class scheduler / speculative prefill",
        "(`forkserve/engine/`, `forkserve/bench.py` worker). Keep unit tests green.",
        "Do not rewrite occupy scripts unless required.",
        "",
        "## Pair results",
    ]
    for p in verdict.pairs:
        flag = "NEED FIX" if p.needs_improve else "ok"
        lines.append(
            f"- tp={p.tp} {p.workload}: {flag} — {p.notes} "
            f"(efficiency_beats={p.efficiency_beats}, perf_drop={p.perf_drop})"
        )
    lines += [
        "",
        "After you finish editing, if this file was written for `--edit wait`,",
        f"create `{LOG_DIR / CONTINUE_NAME}` so the loop can re-occupy and re-bench.",
        "",
    ]
    dest = LOG_DIR / f"round_{round_id}_NEXT.md"
    dest.write_text("\n".join(lines))
    latest = LOG_DIR / "NEXT.md"
    latest.write_text(dest.read_text())
    (LOG_DIR / f"round_{round_id}_verdict.json").write_text(json.dumps(verdict.to_dict(), indent=2))
    return dest


class GpuHold:
    """Background HOLD_ONLY occupy via occupy_when_free.sh."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.proc: subprocess.Popen[str] | None = None
        self.pid_file = root / "logs" / "rl_improve_occupy.pid"

    def start(self) -> None:
        self.stop()
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["HOLD_ONLY"] = "1"
        env["PID_FILE"] = str(self.pid_file)
        env["FORKSERVE_ROOT"] = str(self.root)
        script = self.root / "scripts" / "occupy_when_free.sh"
        log(f"occupy START {script}")
        self.proc = subprocess.Popen(
            ["bash", str(script)],
            cwd=str(self.root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            text=True,
        )
        t0 = time.time()
        while time.time() - t0 < 6 * 3600:
            if self.proc.poll() is not None:
                raise RuntimeError(f"occupy exited early rc={self.proc.returncode}")
            used = _gpu_used_mib()
            if used and all(u > 2048 for u in used.values()) and _pid_on_all_gpus(self.proc.pid):
                log(f"occupy held used_mib={used}")
                return
            time.sleep(1.0)
        raise RuntimeError("occupy did not claim all GPUs within 6h")

    def stop(self) -> None:
        pids: set[int] = set()
        if self.proc and self.proc.poll() is None:
            pids.add(self.proc.pid)
        if self.pid_file.is_file():
            try:
                pids.add(int(self.pid_file.read_text().strip()))
            except ValueError:
                pass
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                log(f"occupy SIGTERM pid={pid}")
            except OSError:
                pass
        t0 = time.time()
        while time.time() - t0 < 20:
            used = _gpu_used_mib()
            if used and all(u <= 2048 for u in used.values()):
                break
            time.sleep(0.25)
        self.proc = None


def _gpu_used_mib() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    used: dict[int, int] = {}
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    want = {int(x) for x in vis.split(",") if x.strip().isdigit()} if vis else None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        idx = int(parts[0])
        if want is None or idx in want:
            used[idx] = int(float(parts[1]))
    return used


def _pid_on_all_gpus(pid: int) -> bool:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    seen = [int(x.strip()) for x in raw.splitlines() if x.strip().isdigit()]
    return seen.count(pid) >= max(len(_gpu_used_mib()), 1)


def stop_foreign_occupy() -> None:
    """Stop leftover occupy_gpus.py from a previous HOLD_ONLY so bench can start."""
    try:
        raw = subprocess.check_output(["pgrep", "-af", "occupy_gpus.py"], text=True)
    except subprocess.CalledProcessError:
        return
    me = os.getpid()
    for line in raw.splitlines():
        parts = line.split(None, 1)
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid == me:
            continue
        if "occupy_gpus.py" not in line:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"released leftover occupy pid={pid}")
        except OSError:
            pass
    t0 = time.time()
    while time.time() - t0 < 20:
        used = _gpu_used_mib()
        if not used or all(u <= 2048 for u in used.values()):
            return
        time.sleep(0.25)


def run_bench(args: argparse.Namespace, round_id: int) -> Path:
    out = ROOT / "logs" / f"rl_improve_round_{round_id}.json"
    cmd = [
        sys.executable,
        "-m",
        "forkserve.bench",
        "--tp",
        args.tp,
        "--model",
        args.model,
        "--workloads",
        args.workloads,
        "--limit",
        str(args.limit),
        "--out",
        str(out),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    log("bench: " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"bench failed rc={proc.returncode}")
    return out


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return list(data.get("rows") or [])


def run_cursor_agent(prompt_path: Path) -> bool:
    cursor = shutil.which("cursor")
    if not cursor:
        log("cursor CLI not found")
        return False
    if not os.environ.get("CURSOR_API_KEY"):
        log("CURSOR_API_KEY unset — skip auto agent, use wait mode")
        return False
    prompt = prompt_path.read_text()
    cmd = [
        cursor,
        "agent",
        "-p",
        "--force",
        "--output-format",
        "text",
        prompt,
    ]
    log("cursor agent: editing while GPUs are held")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    log(f"cursor agent rc={proc.returncode}")
    return proc.returncode == 0


def wait_for_continue(timeout_s: float) -> bool:
    marker = LOG_DIR / CONTINUE_NAME
    if marker.exists():
        marker.unlink()
    log(f"occupy HOLD — edit in Cursor, then: touch {marker}")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if marker.is_file():
            marker.unlink(missing_ok=True)
            log("CONTINUE seen")
            return True
        time.sleep(1.0)
    log("CONTINUE wait timed out")
    return False


def edit_round(args: argparse.Namespace, hold: GpuHold, prompt_path: Path) -> None:
    hold.start()
    mode = args.edit
    tried = False
    if mode in ("cursor", "both"):
        tried = run_cursor_agent(prompt_path)
        if tried:
            return
        if mode == "cursor":
            raise RuntimeError("cursor agent did not run (need CURSOR_API_KEY)")
    if mode in ("wait", "both"):
        if not wait_for_continue(args.wait_sec):
            raise RuntimeError(f"no {CONTINUE_NAME} after {args.wait_sec:.0f}s")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL loop: occupy → bench → Cursor edit → occupy → bench")
    p.add_argument("--max-rounds", type=int, default=int(os.environ.get("RL_MAX_ROUNDS", "100")))
    p.add_argument("--tp", default=os.environ.get("FORKSERVE_TP", "2,4"))
    p.add_argument("--model", default=os.environ.get("FORKSERVE_MODEL", str(Path.home() / "models/Qwen3-8B")))
    p.add_argument("--workloads", default="gsm8k,game24,humaneval")
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--kv-gain", type=float, default=0.20, help="ForkServe peak_kv must be this fraction below recompute")
    p.add_argument("--perf-drop", type=float, default=0.10, help="latency slack vs vLLM APC (or recompute)")
    p.add_argument("--peak-slack", type=float, default=0.05, help="allowed peak_kv above APC")
    p.add_argument("--edit", choices=("cursor", "wait", "both"), default=os.environ.get("RL_EDIT", "both"))
    p.add_argument("--wait-sec", type=float, default=float(os.environ.get("RL_WAIT_SEC", "86400")))
    p.add_argument("--from-json", default="", help="judge this JSON only (no GPU)")
    p.add_argument("--enforce-eager", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.from_json:
        rows = load_rows(Path(args.from_json))
        verdict = judge_rows(
            rows, kv_gain=args.kv_gain, perf_drop=args.perf_drop, peak_slack=args.peak_slack
        )
        print(json.dumps(verdict.to_dict(), indent=2))
        write_cursor_prompt(0, verdict, Path(args.from_json))
        return 0 if verdict.ok else 2

    hold = GpuHold()
    try:
        for rnd in range(1, args.max_rounds + 1):
            log(f"===== round {rnd}/{args.max_rounds} =====")
            hold.stop()
            stop_foreign_occupy()
            hold.start()
            time.sleep(2.0)
            hold.stop()
            try:
                bench_path = run_bench(args, rnd)
                rows = load_rows(bench_path)
                verdict = judge_rows(
                    rows, kv_gain=args.kv_gain, perf_drop=args.perf_drop, peak_slack=args.peak_slack
                )
            except Exception as exc:
                log(f"bench failed: {exc}")
                verdict = RoundVerdict(ok=False, reason=f"bench crashed: {exc}")
                bench_path = ROOT / "logs" / f"rl_improve_round_{rnd}.json"
            print(json.dumps(verdict.to_dict(), indent=2), flush=True)
            prompt = write_cursor_prompt(rnd, verdict, bench_path)
            if verdict.ok:
                log(f"SUCCESS round {rnd}: {verdict.reason}")
                hold.start()
                return 0
            log(f"FAIL round {rnd}: {verdict.reason} — occupy + Cursor edit")
            if rnd >= args.max_rounds:
                hold.start()
                log(f"max rounds reached; GPUs held. prompt={prompt}")
                return 2
            edit_round(args, hold, prompt)
        return 2
    except KeyboardInterrupt:
        log("interrupted — holding GPUs")
        hold.start()
        return 130
    except Exception as exc:
        log(f"error: {exc}")
        hold.start()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluation for ForkServe+: fan-out microbench, accuracy curves, concurrency.

These instruments match the three serving claims:

1. Fan-out split into CoW / prefill / abort-mark (reclaim is off TTFT).
2. Decode-token vs accuracy curves from graded traces (and early-stop plus).
3. High-concurrency capacity from peak-KV (paper eq. 9) plus a mock sweep.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from forkserve.api import Engine
from forkserve.bench import run_tot_forest_forkserve
from forkserve.config import ForkServeConfig
from forkserve.engine.mock import MockBackend
from forkserve.prune import plus_config
from forkserve.spec_pool import extra_batched_tokens, throughput_tokens_per_s


def _engine(plus: bool) -> Engine:
    cfg = ForkServeConfig(
        page_size=8,
        bytes_per_token=1.0,
        max_batched_tokens=2048,
        hbm_capacity_bytes=64.0 * (1 << 20),
    )
    if plus:
        cfg = plus_config(cfg)
        cfg.extra["spec_pool_tokens"] = 256.0
    return Engine(MockBackend(cfg), cfg)


def micro_fanout(
    *,
    sessions: int = 8,
    branching: int = 4,
    trunk: str = "shared trunk tokens for a math problem " * 8,
    plus: bool = False,
) -> dict[str, Any]:
    """One ToT forest; return the CoW / prefill / abort split."""
    eng = _engine(plus)
    thoughts = [
        "Thought 1: compute carefully and box the answer.",
        "Thought 2: ****loop****loop****loop****loop",
        "Thought 3: undefined nan junk residual",
        "Thought 4: try a substitution then combine terms.",
    ][:branching]
    trunks = [f"{trunk} item {i}" for i in range(sessions)]
    row = run_tot_forest_forkserve(
        eng, eng.config, trunks, thoughts, decode_n=16, idle_ms=0.0, workload="micro"
    )
    _close(eng)
    return {
        "system": row.system,
        "plus": plus,
        "sessions": sessions,
        "branching": branching,
        "cow_ms": row.cow_ms,
        "prefill_ms": row.prefill_ms,
        "abort_mark_ms": row.abort_mark_ms,
        "fanout_ms": row.fanout_ms,
        "decode_ms": row.decode_ms,
        "e2e_ms": row.e2e_ms,
        "pointer_swaps": row.pointer_swaps,
        "pruned_branches": row.pruned_branches,
        "prefilled_branches": row.prefilled_branches,
        "hash_skips": row.hash_skips,
        "early_aborts": row.early_aborts,
        "transfer_tokens": row.transfer_tokens,
        "peak_kv_tokens": row.peak_kv_tokens,
        "decode_tokens": row.decode_tokens,
        "kv_saving": row.kv_saving,
    }


def time_accuracy_curve(
    rows: Sequence[dict[str, Any]],
    *,
    stop_frac: float = 1.0,
) -> list[dict[str, float]]:
    """Running accuracy vs cumulative decode tokens.

    ``stop_frac`` < 1 models plus-mode early EOS (answer marker) on the same
    traces — it never invents extra correct items.
    """
    correct = list(rows[0].get("task_correct") or []) if rows else []
    n = len(correct)
    if n == 0:
        return []
    per = int(rows[0].get("decode_per_item") or 1)
    use = max(1, int(per * stop_frac))
    out: list[dict[str, float]] = []
    got = 0
    toks = 0
    for i, ok in enumerate(correct, start=1):
        got += int(bool(ok))
        toks += use
        out.append(
            {
                "items": float(i),
                "decode_tokens": float(toks),
                "accuracy": got / i,
                "solved": float(got),
            }
        )
    return out


def tokens_to_hit_accuracy(curve: Sequence[dict[str, float]], target: float) -> float | None:
    for pt in curve:
        if pt["accuracy"] + 1e-12 >= target:
            return pt["decode_tokens"]
    return None


def concurrency_sweep(
    *,
    m_apc_tokens: int = 1563,
    m_fs_tokens: int = 1163,
    decode_tokens: int = 256,
    e2e_s: float = 0.4,
    hbm_bytes: float | None = None,
    spec_pool_frac: float = 0.25,
    qps_grid: Sequence[int] = (8, 16, 32, 48, 64, 96, 128, 192, 256),
    base_batched_tokens: int = 2048,
) -> list[dict[str, Any]]:
    """QPS sweep using measured ToT peak KV (GSM8K APC vs ForkServe), not clone.

    ``hbm_bytes`` is the *KV pool* after weights (default 8 GiB on a 40 GB A100).
    """
    bpt = 147_456.0
    hbm = hbm_bytes if hbm_bytes is not None else 8.0 * (1 << 30)
    m_apc = m_apc_tokens * bpt
    m_fs = m_fs_tokens * bpt
    saving = 0.0 if m_apc <= 0 else max(0.0, 1.0 - m_fs / m_apc)
    cap_apc = max(1, int(hbm // m_apc))
    cap_fs = max(1, int(hbm // m_fs))
    extra_tok = extra_batched_tokens(base_batched_tokens, saving, spec_pool_frac)
    out: list[dict[str, Any]] = []
    for qps in qps_grid:
        # Concurrent in-flight ≈ QPS * e2e. Saturate at slot cap.
        inflight = max(1, int(round(qps * e2e_s)))
        run_apc = min(inflight, cap_apc)
        run_fs = min(inflight, cap_fs)
        # P99 TTFT rises once we exceed capacity (simple M/M/1-style queue).
        p99_apc = _p99_ttft(e2e_s, inflight, cap_apc)
        p99_fs = _p99_ttft(e2e_s, inflight, cap_fs)
        out.append(
            {
                "qps": qps,
                "inflight": inflight,
                "apc_running": run_apc,
                "fs_running": run_fs,
                "apc_tok_s": throughput_tokens_per_s(run_apc, decode_tokens, e2e_s),
                "fs_tok_s": throughput_tokens_per_s(run_fs, decode_tokens, e2e_s),
                "apc_p99_ttft_s": p99_apc,
                "fs_p99_ttft_s": p99_fs,
                "apc_slo_ok": p99_apc <= 1.0,
                "fs_slo_ok": p99_fs <= 1.0,
                "kv_saving": saving,
                "slots_apc": cap_apc,
                "slots_fs": cap_fs,
                "extra_batched_tokens": extra_tok,
            }
        )
    return out


def _p99_ttft(e2e_s: float, inflight: int, slots: int) -> float:
    """Queueing P99: base decode TTFT plus twice the excess-batch wait."""
    if slots <= 0:
        return 1e9
    base = 0.05 * e2e_s
    extra = max(0, inflight - slots) * (e2e_s / slots)
    return base + 2.0 * extra


def load_eval_rows(path: Path, system: str, tp: int, workload: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return [
        r
        for r in data.get("rows", [])
        if r.get("system") == system and r.get("tp") == tp and r.get("workload") == workload
    ]


def run_suite(out_dir: Path | None = None) -> dict[str, Any]:
    """Mock microbench + analytical concurrency. Writes JSON if ``out_dir`` set."""
    base = micro_fanout(plus=False)
    plus = micro_fanout(plus=True)
    conc = concurrency_sweep()
    slo_apc = max((r["qps"] for r in conc if r["apc_slo_ok"]), default=0)
    slo_fs = max((r["qps"] for r in conc if r["fs_slo_ok"]), default=0)
    peak_apc = max(r["apc_tok_s"] for r in conc)
    peak_fs = max(r["fs_tok_s"] for r in conc)
    report = {
        "micro_baseline": base,
        "micro_plus": plus,
        "fanout_plus_vs_base": {
            "abort_mark_ms": plus["abort_mark_ms"],
            "pruned_branches": plus["pruned_branches"],
            "prefilled_branches": plus["prefilled_branches"],
            "pointer_swaps": plus["pointer_swaps"],
        },
        "concurrency": conc,
        "max_qps_p99_1s": {"apc": slo_apc, "forkserve_plus": slo_fs},
        "peak_tok_s": {"apc": peak_apc, "forkserve_plus": peak_fs},
        "tok_s_gain": (peak_fs / peak_apc - 1.0) if peak_apc else 0.0,
        "prefill_methods": "experiments.prefill_prune_bench",
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "eval_plus.json"
        dest.write_text(json.dumps(report, indent=2))
        report["wrote"] = str(dest)
    return report


def _close(eng: Engine) -> None:
    for sid in list(eng.forest.sessions):
        eng.close(sid)


def curve_from_eval_json(path: Path, system: str, tp: int, workload: str) -> list[dict[str, float]]:
    return time_accuracy_curve(load_eval_rows(path, system, tp, workload))

#!/usr/bin/env python3
"""Compact ToT math serving report: accuracy + efficiency breakdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def _i(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def per_item(row: dict[str, Any]) -> dict[str, Any]:
    n = max(_i(row, "sessions"), 1)
    e2e = _f(row, "e2e_ms")
    fan = _f(row, "fanout_ms")
    dec = _f(row, "decode_ms")
    toks = _f(row, "decode_tokens")
    return {
        "n": n,
        "e2e_s": e2e / 1000.0,
        "e2e_ms_per_item": e2e / n,
        "fanout_ms_per_item": fan / n,
        "decode_ms_per_item": dec / n,
        "fanout_share": (fan / e2e) if e2e > 0 else 0.0,
        "decode_share": (dec / e2e) if e2e > 0 else 0.0,
        "decode_tok_s": (toks / (dec / 1000.0)) if dec > 0 else 0.0,
        "items_per_min": (n / (e2e / 60000.0)) if e2e > 0 else 0.0,
        "peak_kv_tokens": _i(row, "peak_kv_tokens"),
        "peak_kv_per_item": _i(row, "peak_kv_tokens") / n,
        "m_cow_mib": _f(row, "m_cow_mib"),
        "m_clone_mib": _f(row, "m_clone_mib"),
        "kv_saving": _f(row, "kv_saving"),
        "gpu_mem_mib": row.get("gpu_mem_mib") or [],
        "branching": _i(row, "branching"),
        "decode_per_item": _i(row, "decode_per_item"),
        "task_metric": row.get("task_metric") or "",
        "task_n": _i(row, "task_n"),
        "task_score": _f(row, "task_score", -1.0),
        "quality_delta": _f(row, "quality_delta"),
        "quality_vs": row.get("quality_vs") or "",
        "quality_collapsed": bool(row.get("quality_collapsed")),
        "score_by_slice": row.get("score_by_slice") or {},
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    return list(data.get("rows") or [])


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for r in rows:
        key = (str(r.get("system")), str(r.get("workload")), _i(r, "branching") or _i(r, "tp"))
        # Prefer branching in the value; key uses branching if present else 0.
        b = _i(r, "branching")
        out[(str(r.get("system")), str(r.get("workload")), b)] = r
    return out


def compare_pair(apc: dict[str, Any], other: dict[str, Any], label: str = "forkserve") -> dict[str, Any]:
    a = per_item(apc)
    f = per_item(other)
    e2e_a = a["e2e_ms_per_item"]
    e2e_f = f["e2e_ms_per_item"]
    fan_a = a["fanout_ms_per_item"]
    fan_f = f["fanout_ms_per_item"]
    kv_a = a["peak_kv_tokens"]
    kv_f = f["peak_kv_tokens"]
    return {
        "workload": apc.get("workload"),
        "branching": a["branching"] or f["branching"],
        "n": min(a["n"], f["n"]),
        "system": label,
        "apc": a,
        "forkserve": f,
        "acc_apc": a["task_score"],
        "acc_fs": f["task_score"],
        "acc_delta": f["task_score"] - a["task_score"] if a["task_score"] >= 0 and f["task_score"] >= 0 else None,
        "e2e_speedup": (e2e_a / e2e_f) if e2e_f > 0 else 0.0,
        "fanout_speedup": (fan_a / fan_f) if fan_f > 0 else 0.0,
        "kv_ratio_fs_over_apc": (kv_f / kv_a) if kv_a > 0 else 0.0,
        "kv_save_vs_apc": (1.0 - kv_f / kv_a) if kv_a > 0 else 0.0,
    }


def write_summary(out_dir: Path) -> dict[str, Any]:
    main = load_rows(out_dir / "eval_b4.json")
    sweep: list[dict[str, Any]] = []
    for b in (2, 4, 6):
        sweep.extend(load_rows(out_dir / f"eval_b{b}_sweep.json"))
    # b=4 sweep may be absent; main already has b=4 at LIMIT.

    pairs = []
    app_pairs = []
    by = {}
    for r in main:
        by.setdefault((r.get("workload"), _i(r, "branching")), {})[r.get("system")] = r
    for (_wl, _b), sysmap in sorted(by.items(), key=lambda x: str(x[0][0])):
        if "vllm_apc" not in sysmap:
            continue
        if "forkserve" in sysmap:
            pairs.append(compare_pair(sysmap["vllm_apc"], sysmap["forkserve"]))
        if "forkserve_plus" in sysmap:
            app_pairs.append(compare_pair(sysmap["vllm_apc"], sysmap["forkserve_plus"], "forkserve_plus"))

    sweep_pairs = []
    app_sweep = []
    sby: dict[tuple[Any, int], dict[str, Any]] = {}
    for r in sweep + [x for x in main if x.get("workload") in ("gsm8k", "math500", "game24")]:
        sby.setdefault((r.get("workload"), _i(r, "branching")), {})[r.get("system")] = r
    for key, sysmap in sorted(sby.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        if "vllm_apc" not in sysmap:
            continue
        if "forkserve" in sysmap:
            sweep_pairs.append(compare_pair(sysmap["vllm_apc"], sysmap["forkserve"]))
        if "forkserve_plus" in sysmap:
            app_sweep.append(compare_pair(sysmap["vllm_apc"], sysmap["forkserve_plus"], "forkserve_plus"))

    acc_ok = [p for p in pairs if p["acc_delta"] is not None]
    app_ok = [p for p in app_pairs if p["acc_delta"] is not None]
    headline = {
        "n_workloads": len(pairs),
        "mean_acc_apc": sum(p["acc_apc"] for p in acc_ok) / len(acc_ok) if acc_ok else None,
        "mean_acc_fs": sum(p["acc_fs"] for p in acc_ok) / len(acc_ok) if acc_ok else None,
        "mean_e2e_speedup": sum(p["e2e_speedup"] for p in pairs) / len(pairs) if pairs else None,
        "mean_fanout_speedup": sum(p["fanout_speedup"] for p in pairs) / len(pairs) if pairs else None,
        "mean_kv_save_vs_apc": sum(p["kv_save_vs_apc"] for p in pairs) / len(pairs) if pairs else None,
        "mean_acc_app": sum(p["acc_fs"] for p in app_ok) / len(app_ok) if app_ok else None,
        "mean_kv_save_app_vs_apc": sum(p["kv_save_vs_apc"] for p in app_pairs) / len(app_pairs) if app_pairs else None,
    }
    report = {
        "model": os_model(),
        "headline": headline,
        "pairs": pairs,
        "app_pairs": app_pairs,
        "sweep": sweep_pairs,
        "app_sweep": app_sweep,
        "rows_main": [
            {
                "system": r.get("system"),
                "workload": r.get("workload"),
                "tp": r.get("tp"),
                **per_item(r),
            }
            for r in main
        ],
    }
    dest = out_dir / "summary.json"
    dest.write_text(json.dumps(report, indent=2))
    table = _text_table(pairs, "ForkServe")
    if app_pairs:
        table += "\n" + _text_table(app_pairs, "APP")
    (out_dir / "summary.txt").write_text(table)
    print(table, flush=True)
    return report


def os_model() -> str:
    import os

    return os.environ.get("FORKSERVE_MODEL", "")


def _text_table(pairs: list[dict[str, Any]], other: str = "ForkServe") -> str:
    lines = [
        f"APC vs {other}",
        "workload  b   n  acc_apc  acc_fs  d_acc  e2e_ms/item  APC  FS  speedup  fanout APC  FS  peak_kv APC   FS  kv_save",
        "-" * 118,
    ]
    for p in pairs:
        a, f = p["apc"], p["forkserve"]
        dacc = p["acc_delta"]
        dacc_s = f"{dacc:+.3f}" if dacc is not None else "  n/a"
        lines.append(
            f"{str(p['workload']):<8} {p['branching']:>2} {p['n']:>3}  "
            f"{p['acc_apc']:7.3f} {p['acc_fs']:7.3f} {dacc_s:>6}  "
            f"{a['e2e_ms_per_item']:8.1f} {f['e2e_ms_per_item']:8.1f} {p['e2e_speedup']:6.2f}x  "
            f"{a['fanout_ms_per_item']:8.1f} {f['fanout_ms_per_item']:8.1f}  "
            f"{a['peak_kv_tokens']:8d} {f['peak_kv_tokens']:6d}  {p['kv_save_vs_apc']:6.2f}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/tot_math")
    write_summary(dest)

#!/usr/bin/env python3
"""Multiturn ToT serving report: accuracy + KV / e2e vs APC."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from summarize_tot_math import compare_pair, per_item  # noqa: E402


def _i(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    return list(data.get("rows") or [])


def write_summary(out_dir: Path) -> dict[str, Any]:
    main = load_rows(out_dir / "eval.json")
    if not main:
        for p in sorted(out_dir.glob("bench_*.json")):
            main.extend(load_rows(p))
    by: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for r in main:
        key = (str(r.get("workload")), _i(r, "turns") or 1)
        by.setdefault(key, {})[str(r.get("system"))] = r
    pairs = []
    app_pairs = []
    for (wl, turns), sysmap in sorted(by.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        if "vllm_apc" not in sysmap:
            continue
        if "forkserve" in sysmap:
            p = compare_pair(sysmap["vllm_apc"], sysmap["forkserve"])
            p["turns"] = turns
            p["workload"] = wl
            pairs.append(p)
        if "forkserve_plus" in sysmap:
            p = compare_pair(sysmap["vllm_apc"], sysmap["forkserve_plus"], "forkserve_plus")
            p["turns"] = turns
            p["workload"] = wl
            app_pairs.append(p)
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
        "mean_kv_save_app_vs_apc": (
            sum(p["kv_save_vs_apc"] for p in app_pairs) / len(app_pairs) if app_pairs else None
        ),
        "turns": max((_i(r, "turns") for r in main), default=1),
        "branching": max((_i(r, "branching") for r in main), default=4),
    }
    report = {
        "headline": headline,
        "pairs": pairs,
        "app_pairs": app_pairs,
        "rows": [
            {
                "system": r.get("system"),
                "workload": r.get("workload"),
                "tp": r.get("tp"),
                "turns": r.get("turns"),
                "branching": r.get("branching"),
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


def _text_table(pairs: list[dict[str, Any]], other: str = "ForkServe") -> str:
    lines = [
        f"APC vs {other}  (multiturn ToT, 4 agents / problem)",
        "workload   b  d   n  acc_apc  acc_fs  d_acc   e2e/item APC      FS  speedup  peak_kv APC    FS  kv_save",
        "-" * 114,
    ]
    for p in pairs:
        a, f = p["apc"], p["forkserve"]
        dacc = p["acc_delta"]
        dacc_s = f"{dacc:+.3f}" if dacc is not None else "  n/a"
        lines.append(
            f"{str(p['workload']):<9} {p['branching']:>2} {_i(p, 'turns'):>2} {p['n']:>3}  "
            f"{p['acc_apc']:7.3f} {p['acc_fs']:7.3f} {dacc_s:>6}  "
            f"{a['e2e_ms_per_item']:8.1f} {f['e2e_ms_per_item']:8.1f} {p['e2e_speedup']:6.2f}x  "
            f"{a['peak_kv_tokens']:8d} {f['peak_kv_tokens']:6d}  {p['kv_save_vs_apc']:6.2f}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/tot_mt")
    write_summary(dest)

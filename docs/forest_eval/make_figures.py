#!/usr/bin/env python3
"""Publication figures for the Qwen3-14B forest full-set report."""
from __future__ import annotations

import csv
import re
import statistics as st
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/dliu/ForkServe")
LOG = ROOT / "logs/full_quality/progress.log"
OUT = Path(__file__).resolve().parent
FIG = OUT / "fig"
DATA = OUT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# Colorblind-safe (Tol muted)
C_REC = "#BBBBBB"
C_APC = "#0077BB"
C_FS = "#EE7733"
C_INK = "#1A1A1A"
C_MUTE = "#5C5C5C"
C_GRID = "#E6E6E6"

PAT = re.compile(
    r"\[(.*?)\] (\S+) tp=\d+ forest (\S+) (\d+)/(\d+) \+\d+ "
    r"peak=(\d+) fanout=([\d.]+)ms (\w+)=([\d.]+) \((\d+)/(\d+)\)"
)


def setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.6,
            "axes.edgecolor": C_INK,
            "axes.labelcolor": C_INK,
            "xtick.color": C_INK,
            "ytick.color": C_INK,
            "text.color": C_INK,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse() -> dict[tuple[str, str], list[dict]]:
    by: dict[tuple[str, str], list[dict]] = {}
    for line in LOG.read_text(errors="replace").splitlines():
        m = PAT.search(line)
        if not m:
            continue
        ts, sys, wl, done, tot, peak, fan, metric, score, corr, n = m.groups()
        by.setdefault((sys, wl), []).append(
            {
                "ts": ts,
                "done": int(done),
                "tot": int(tot),
                "peak": int(peak),
                "fan": float(fan),
                "score": float(score),
                "corr": int(corr),
                "n": int(n),
                "metric": metric,
            }
        )
    return by


def dump_csv(by: dict) -> None:
    path = DATA / "forest_chunks.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["system", "workload", "done", "tot", "peak", "fan_ms", "score", "corr", "n", "ts"],
        )
        w.writeheader()
        for (sys, wl), xs in sorted(by.items()):
            for x in xs:
                w.writerow(
                    {
                        "system": sys,
                        "workload": wl,
                        "done": x["done"],
                        "tot": x["tot"],
                        "peak": x["peak"],
                        "fan_ms": x["fan"],
                        "score": x["score"],
                        "corr": x["corr"],
                        "n": x["n"],
                        "ts": x["ts"],
                    }
                )


def series(by, sys, wl):
    return by.get((sys, wl), [])


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def fig_peak_trace(by) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    for sys, color, label, lw, z in [
        ("vllm_recompute", C_REC, "vLLM recompute", 1.3, 1),
        ("vllm_apc", C_APC, "vLLM APC", 1.5, 2),
        ("forkserve", C_FS, "ForkServe", 1.7, 3),
    ]:
        xs = series(by, sys, "gsm8k")
        ax.plot([x["done"] for x in xs], [x["peak"] for x in xs], color=color, lw=lw, label=label, zorder=z)
    ax.set_xlabel("Items completed")
    ax.set_ylabel("Peak KV tokens (chunk of 8)")
    ax.set_title("GSM8K forest: resident KV after each chunk")
    ax.legend(loc="upper right", ncol=3)
    ax.set_xlim(0, 1320)
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "peak_trace_gsm8k")


def fig_peak_scatter(by) -> None:
    """Per-batch peaks from the GSM8K run log."""
    ap = series(by, "vllm_apc", "gsm8k")
    fs = series(by, "forkserve", "gsm8k")
    fig, ax = plt.subplots(figsize=(4.6, 3.35))
    ax.plot(
        [x["done"] for x in ap],
        [x["peak"] for x in ap],
        color=C_APC,
        lw=0.9,
        marker="o",
        ms=2.4,
        mew=0,
        alpha=0.88,
        label="APC",
        zorder=2,
    )
    ax.plot(
        [x["done"] for x in fs],
        [x["peak"] for x in fs],
        color=C_FS,
        lw=0.9,
        marker="o",
        ms=2.4,
        mew=0,
        alpha=0.88,
        label="ForkServe",
        zorder=3,
    )
    ax.set_xlabel("Items completed")
    ax.set_ylabel("Peak KV tokens")
    ax.set_xlim(0, 1320)
    ymax = max(x["peak"] for x in ap + fs)
    ymin = min(x["peak"] for x in ap + fs)
    ax.set_ylim(ymin - 40, ymax + 40)
    ax.legend(loc="upper right", ncol=2)
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "peak_scatter_fs_vs_apc")


def fig_peak_violin(by) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    data, colors, labels = [], [], []
    for sys, color, label in [
        ("vllm_recompute", C_REC, "recompute"),
        ("vllm_apc", C_APC, "APC"),
        ("forkserve", C_FS, "ForkServe"),
    ]:
        data.append([x["peak"] for x in series(by, sys, "gsm8k")])
        colors.append(color)
        labels.append(label)
    parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.72)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(colors[i])
        b.set_edgecolor(C_INK)
        b.set_alpha(0.85)
        b.set_linewidth(0.4)
    for k in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[k].set_color(C_INK)
        parts[k].set_linewidth(0.7)
    ax.set_xticks([1, 2, 3], labels)
    ax.set_ylabel("Peak KV tokens")
    ax.set_title("GSM8K peak-KV distribution (165 chunks)")
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "peak_violin_gsm8k")


def fig_acc_running(by) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    for sys, color, label, lw in [
        ("vllm_recompute", C_REC, "vLLM recompute", 1.3),
        ("vllm_apc", C_APC, "vLLM APC", 1.5),
        ("forkserve", C_FS, "ForkServe", 1.8),
    ]:
        xs = series(by, sys, "gsm8k")
        ax.plot([x["done"] for x in xs], [x["score"] for x in xs], color=color, lw=lw, label=label)
    ax.set_ylim(0.72, 0.92)
    ax.set_xlim(0, 1320)
    ax.set_xlabel("Items completed")
    ax.set_ylabel("Running accuracy")
    ax.set_title("GSM8K official extractor, running score")
    ax.legend(loc="lower right", ncol=3)
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "accuracy_running_gsm8k")


def fig_fanout(by) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    for sys, color, label, lw in [
        ("vllm_recompute", C_REC, "vLLM recompute", 1.2),
        ("vllm_apc", C_APC, "vLLM APC", 1.4),
        ("forkserve", C_FS, "ForkServe", 1.6),
    ]:
        xs = series(by, sys, "gsm8k")
        ax.plot([x["done"] for x in xs], [x["fan"] for x in xs], color=color, lw=lw, alpha=0.95, label=label)
    ax.set_xlabel("Items completed")
    ax.set_ylabel("Fan-out time (ms)")
    ax.set_title("GSM8K four-way ToT fan-out latency per chunk")
    ax.legend(loc="upper right", ncol=3)
    ax.set_xlim(0, 1320)
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fanout_trace_gsm8k")


def fig_quality_bars() -> None:
    # forest vs quality-only (recompute/APC/FS quality-only from eval)
    cats = ["GSM8K acc.", "Game24 success", "HumanEval pass@1"]
    qo = {
        "recompute": [0.941, 0.018, 0.616],
        "APC": [0.940, 0.020, 0.616],
        "ForkServe": [0.941, 0.018, 0.622],
    }
    forest = {
        "recompute": [0.847, 0.283, 0.634],
        "APC": [0.844, 0.269, 0.628],
        "ForkServe": [0.845, np.nan, np.nan],
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    x = np.arange(3)
    w = 0.26
    for ax, bag, title in [
        (axes[0], qo, "Single-path (quality-only)"),
        (axes[1], forest, "Forest (ToT / ReAct)"),
    ]:
        ax.bar(x - w, bag["recompute"], w, color=C_REC, label="recompute", edgecolor=C_INK, lw=0.3)
        ax.bar(x, bag["APC"], w, color=C_APC, label="APC", edgecolor=C_INK, lw=0.3)
        ax.bar(x + w, bag["ForkServe"], w, color=C_FS, label="ForkServe", edgecolor=C_INK, lw=0.3)
        ax.set_xticks(x, cats)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Official score")
    axes[1].legend(loc="upper right")
    fig.suptitle("Same model: recipe change vs serving system", y=1.03, fontsize=11)
    save(fig, "quality_forest_vs_single")


def fig_saving_bars(by) -> None:
    def med_upto(sys, wl, done_max=None):
        xs = series(by, sys, wl)
        if done_max is not None:
            xs = [x for x in xs if x["done"] <= done_max]
        return st.median([x["peak"] for x in xs]) if xs else np.nan

    workloads = ["GSM8K\n(full 1319)", "Game24\n(first 112)"]
    rec = [med_upto("vllm_recompute", "gsm8k"), med_upto("vllm_recompute", "game24", 112)]
    apc = [med_upto("vllm_apc", "gsm8k"), med_upto("vllm_apc", "game24", 112)]
    fs = [med_upto("forkserve", "gsm8k"), med_upto("forkserve", "game24", 112)]
    fig, ax = plt.subplots(figsize=(5.4, 3.35))
    x = np.arange(2)
    w = 0.25
    ax.bar(x - w, rec, w, color=C_REC, label="recompute", edgecolor=C_INK, lw=0.3)
    ax.bar(x, apc, w, color=C_APC, label="APC", edgecolor=C_INK, lw=0.3)
    ax.bar(x + w, fs, w, color=C_FS, label="ForkServe", edgecolor=C_INK, lw=0.3)
    ax.set_xticks(x, workloads)
    ax.set_ylabel("Median peak KV tokens")
    ax.set_title("Committed-spine KV vs prefix cache vs recompute")
    ax.legend()
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "peak_median_bars")


def fig_ratio_hist(by) -> None:
    fs = {x["done"]: x["peak"] for x in series(by, "forkserve", "gsm8k")}
    ap = {x["done"]: x["peak"] for x in series(by, "vllm_apc", "gsm8k")}
    keys = sorted(set(fs) & set(ap))
    ratio = np.array([fs[k] / ap[k] for k in keys])
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.hist(ratio, bins=18, color=C_FS, edgecolor=C_INK, lw=0.4, alpha=0.9)
    ax.axvline(1.0, color=C_APC, ls="--", lw=1.1, label="APC (ratio $=1$)")
    ax.axvline(np.median(ratio), color=C_INK, lw=1.0, label=f"median {np.median(ratio):.2f}")
    ax.set_xlabel("ForkServe / APC peak KV")
    ax.set_ylabel("Chunks")
    ax.set_title("GSM8K: relative peak vs APC")
    ax.legend()
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "peak_ratio_hist")


def fig_game24_partial(by) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    for sys, color, label, lw in [
        ("vllm_apc", C_APC, "APC (same 112 items)", 1.5),
        ("forkserve", C_FS, "ForkServe", 1.7),
    ]:
        xs = [x for x in series(by, sys, "game24") if x["done"] <= 112]
        ax.plot([x["done"] for x in xs], [x["peak"] for x in xs], color=color, lw=lw, label=label)
    ax.set_xlabel("Items completed")
    ax.set_ylabel("Peak KV tokens")
    ax.set_title("Game24 before crash (112 / 1362)")
    ax.legend()
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "peak_trace_game24_partial")


def fig_e2e_gsm8k() -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    labs = ["recompute", "APC", "ForkServe"]
    vals = [19.4, 18.6, 18.9]
    colors = [C_REC, C_APC, C_FS]
    bars = ax.bar(labs, vals, color=colors, edgecolor=C_INK, lw=0.3, width=0.62)
    ax.set_ylabel("Wall-clock (min)")
    ax.set_title("GSM8K generate time (first to last chunk)")
    ax.set_ylim(0, 24)
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.1f}", ha="center", fontsize=8.5)
    save(fig, "generate_wallclock_gsm8k")


def fig_rl_round3() -> None:
    """Successful forest pairs from rl_improve round 3 (n=4). Omit tp=4 GSM8K."""
    workloads = ["GSM8K", "Game24", "HumanEval"]
    rec = [1844, 1508, 1406]
    apc = [659, 638, 739]
    fs = [459, 390, 695]
    fig, ax = plt.subplots(figsize=(6.4, 3.35))
    x = np.arange(3)
    w = 0.25
    ax.bar(x - w, rec, w, color=C_REC, label="recompute", edgecolor=C_INK, lw=0.3)
    ax.bar(x, apc, w, color=C_APC, label="APC", edgecolor=C_INK, lw=0.3)
    ax.bar(x + w, fs, w, color=C_FS, label="ForkServe", edgecolor=C_INK, lw=0.3)
    ax.set_xticks(x, workloads)
    ax.set_ylabel("peak_kv tokens")
    ax.set_title("RL improve round 3, $n=4$, tp=2: peak KV")
    ax.legend()
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "rl3_peak_tp2")

    # latency relative to APC (%)
    # gsm8k -0.3, game24 -0.7, humaneval +0.1  (e2e vs APC; HE uses ttft in judge)
    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    delta = [-0.27, -0.71, 0.07]  # (FS-APC)/APC * 100 from e2e_ms
    colors = [C_FS if d <= 0 else "#CC3311" for d in delta]
    ax.axhline(0, color=C_MUTE, lw=0.8)
    ax.bar(workloads, delta, color=colors, edgecolor=C_INK, lw=0.3, width=0.55)
    ax.set_ylabel("ForkServe e2e vs APC (%)")
    ax.set_title("RL improve round 3, $n=4$, tp=2: latency vs APC")
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "rl3_lat_tp2")


def fig_gsm8k_median_only(by) -> None:
    def med(sys):
        xs = series(by, sys, "gsm8k")
        return st.median([x["peak"] for x in xs])

    fig, ax = plt.subplots(figsize=(4.6, 3.25))
    labs = ["recompute", "APC", "ForkServe"]
    vals = [med("vllm_recompute"), med("vllm_apc"), med("forkserve")]
    colors = [C_REC, C_APC, C_FS]
    bars = ax.bar(labs, vals, color=colors, edgecolor=C_INK, lw=0.3, width=0.62)
    ax.set_ylabel("Median peak KV tokens")
    ax.set_title("GSM8K full set: median peak KV")
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:.0f}", ha="center", fontsize=8.5)
    save(fig, "peak_median_gsm8k")


def fig_e2e_stack() -> None:
    labs = ["GSM8K generate", "Game24 generate", "HumanEval generate"]
    rec = [19.4, 11.9, 3.3]
    apc = [18.6, 11.5, 3.3]
    fs = [18.9, np.nan, np.nan]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(3)
    w = 0.25
    ax.bar(x - w, rec, w, color=C_REC, label="recompute", edgecolor=C_INK, lw=0.3)
    ax.bar(x, apc, w, color=C_APC, label="APC", edgecolor=C_INK, lw=0.3)
    ax.bar(x + w, fs, w, color=C_FS, label="ForkServe", edgecolor=C_INK, lw=0.3)
    ax.set_xticks(x, labs)
    ax.set_ylabel("Wall-clock (min), first-to-last chunk")
    ax.set_title("Generate time (model already loaded)")
    ax.legend()
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "generate_wallclock")


def fig_rl_round3_tp4_ok() -> None:
    workloads = ["Game24", "HumanEval"]
    rec = [1508, 1406]
    apc = [638, 739]
    fs = [390, 695]
    fig, ax = plt.subplots(figsize=(5.2, 3.25))
    x = np.arange(2)
    w = 0.25
    ax.bar(x - w, rec, w, color=C_REC, label="recompute", edgecolor=C_INK, lw=0.3)
    ax.bar(x, apc, w, color=C_APC, label="APC", edgecolor=C_INK, lw=0.3)
    ax.bar(x + w, fs, w, color=C_FS, label="ForkServe", edgecolor=C_INK, lw=0.3)
    ax.set_xticks(x, workloads)
    ax.set_ylabel("peak_kv tokens")
    ax.set_title("RL improve round 3, $n=4$, tp=4: peak KV (ok pairs)")
    ax.legend()
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "rl3_peak_tp4_ok")


def write_stats(by) -> None:
    lines = []
    for key, xs in sorted(by.items()):
        peaks = [x["peak"] for x in xs]
        fans = [x["fan"] for x in xs]
        last = xs[-1]
        lines.append(
            f"{key[0]:16} {key[1]:10} n={len(xs):3} last={last['done']}/{last['tot']} "
            f"score={last['score']:.3f} peak_med={st.median(peaks):.0f} peak_max={max(peaks)} "
            f"fan_med={st.median(fans):.0f}"
        )
    (DATA / "summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    setup_mpl()
    by = parse()
    dump_csv(by)
    write_stats(by)
    fig_peak_trace(by)
    fig_peak_scatter(by)
    fig_peak_violin(by)
    fig_acc_running(by)
    fig_fanout(by)
    fig_quality_bars()
    fig_saving_bars(by)
    fig_ratio_hist(by)
    fig_game24_partial(by)
    fig_e2e_stack()
    fig_e2e_gsm8k()
    fig_gsm8k_median_only(by)
    fig_rl_round3()
    fig_rl_round3_tp4_ok()
    print("wrote", len(list(FIG.glob("*.pdf"))), "pdfs to", FIG)


if __name__ == "__main__":
    main()

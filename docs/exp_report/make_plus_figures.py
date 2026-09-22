#!/usr/bin/env python3
"""Figures for setting L (APP control plane) and setting M (Qwen3-8B GPU)."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
FIG = OUT / "fig"
FIG.mkdir(exist_ok=True)
PAPER_FIG = OUT.parent / "paper" / "fig"
PAPER_FIG.mkdir(exist_ok=True)

C_REC = "#BBBBBB"
C_APC = "#0077BB"
C_FS = "#EE7733"
C_PLUS = "#009988"
C_INK = "#1A1A1A"
C_MUTE = "#5C5C5C"
C_GRID = "#E6E6E6"


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
            "legend.fontsize": 8.2,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig, name: str) -> None:
    for dest in (FIG, PAPER_FIG):
        fig.savefig(dest / f"{name}.pdf")
        fig.savefig(dest / f"{name}.png")
    plt.close(fig)


def fig_app_methods() -> None:
    labels = ["APC", "hash", "disagg", "ForkServe", "APP"]
    fan = [139.0, 139.0, 162.1, 47.1, 35.9]
    peak = [11520, 11520, 11520, 3840, 2880]
    colors = [C_APC, "#88CCEE", "#AA3377", C_FS, C_PLUS]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.05))
    x = np.arange(len(labels))
    axes[0].bar(x, fan, color=colors, edgecolor=C_INK, lw=0.35)
    axes[0].set_ylabel("Fan-out (ms)")
    axes[0].set_title("Control-plane fan-out")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].bar(x, peak, color=colors, edgecolor=C_INK, lw=0.35)
    axes[1].set_ylabel("Peak KV tokens")
    axes[1].set_title("Resident KV after fan-out")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    for ax in axes:
        ax.yaxis.grid(True, color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Setting L: five prefill methods, $10{\\times}4$ ToT", fontsize=11, y=1.02)
    save(fig, "app_prefill_methods")


def fig_plus_peak() -> None:
    wls = ["GSM8K", "Game24", "HumanEval"]
    rec = [4744, 3068, 3796]
    apc = [1582, 1289, 1970]
    fs = [1182, 793, 1882]
    fig, ax = plt.subplots(figsize=(6.6, 3.15))
    x = np.arange(3)
    w = 0.22
    ax.bar(x - 1.5 * w, rec, w, color=C_REC, label="recompute", edgecolor=C_INK, lw=0.3)
    ax.bar(x - 0.5 * w, apc, w, color=C_APC, label="APC", edgecolor=C_INK, lw=0.3)
    ax.bar(x + 0.5 * w, fs, w, color=C_FS, label="ForkServe / +", edgecolor=C_INK, lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(wls)
    ax.set_ylabel("Peak KV tokens")
    ax.set_title("Setting M peak KV (same at tp=2 and tp=4)")
    ax.legend(ncol=3, loc="upper right")
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "plus_peak_kv")


def fig_plus_gsm8k() -> None:
    systems = ["recompute", "APC", "ForkServe", "ForkServe+"]
    colors = [C_REC, C_APC, C_FS, C_PLUS]
    series = {
        2: {"dec": [40.960, 40.960, 21.235, 13.356], "e2e": [37.34, 36.00, 36.04, 28.22]},
        4: {"dec": [40.960, 40.960, 20.997, 12.826], "e2e": [23.36, 22.72, 23.30, 16.94]},
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    x = np.arange(4)
    w = 0.36
    for i, (tp, hatch, shift) in enumerate([(2, None, -w / 2), (4, "//", w / 2)]):
        axes[0].bar(
            x + shift,
            series[tp]["dec"],
            w,
            color=colors,
            edgecolor=C_INK,
            lw=0.35,
            hatch=hatch,
            label=f"tp={tp}",
        )
        axes[1].bar(
            x + shift,
            series[tp]["e2e"],
            w,
            color=colors,
            edgecolor=C_INK,
            lw=0.35,
            hatch=hatch,
            label=f"tp={tp}",
        )
    axes[0].set_ylabel(r"Decode tokens ($\times 10^3$)")
    axes[0].set_title("GSM8K emitted decode tokens")
    axes[1].set_ylabel("End-to-end (s)")
    axes[1].set_title(r"GSM8K wall-clock, $n{=}80$")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=18, ha="right")
        ax.yaxis.grid(True, color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper right")
    fig.suptitle("Setting M decoding acceleration, Qwen3-8B", fontsize=11, y=1.02)
    save(fig, "plus_gsm8k_decode_e2e")


def fig_plus_e2e_workloads() -> None:
    wls = ["GSM8K", "Game24", "HumanEval"]
    # tp=4 wall-clock seconds
    rec = [23.36, 28.97, 30.42]
    apc = [22.72, 28.51, 29.85]
    fs = [23.30, 28.36, 29.69]
    plus = [16.94, 28.40, 29.77]
    fig, ax = plt.subplots(figsize=(6.8, 3.15))
    x = np.arange(3)
    w = 0.18
    ax.bar(x - 1.5 * w, rec, w, color=C_REC, label="recompute", edgecolor=C_INK, lw=0.3)
    ax.bar(x - 0.5 * w, apc, w, color=C_APC, label="APC", edgecolor=C_INK, lw=0.3)
    ax.bar(x + 0.5 * w, fs, w, color=C_FS, label="ForkServe", edgecolor=C_INK, lw=0.3)
    ax.bar(x + 1.5 * w, plus, w, color=C_PLUS, label="ForkServe+", edgecolor=C_INK, lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(wls)
    ax.set_ylabel("End-to-end (s), tp=4")
    ax.set_title("Setting M e2e: stop list only moves GSM8K")
    ax.legend(ncol=2, loc="upper left")
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 38)
    save(fig, "plus_e2e_tp4")


if __name__ == "__main__":
    setup_mpl()
    fig_app_methods()
    fig_plus_peak()
    fig_plus_gsm8k()
    fig_plus_e2e_workloads()
    print("wrote", FIG)
    print("wrote", PAPER_FIG)

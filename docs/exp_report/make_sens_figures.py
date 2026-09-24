#!/usr/bin/env python3
"""Setting O/P figures from the completed gap-fill and contest forests."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
FIG = OUT / "fig"
PAPER = OUT.parent / "paper" / "fig"
FIG.mkdir(exist_ok=True)
PAPER.mkdir(exist_ok=True)

C_APC = "#0077BB"
C_FS = "#EE7733"
C_PLUS = "#009988"
C_INK = "#1A1A1A"
C_GRID = "#E6E6E6"

# GSM8K peak KV, n=16, tp=2, D=512. ForkServe+ matches ForkServe.
MODELS = ["Llama-3-8B", "Mistral-7B", "R1-Llama-8B"]
CONFIGS = [r"$k{=}2$", r"$k{=}4$", r"$k{=}8$", "depth 2"]
APC = {
    "Llama-3-8B": [610, 750, 998, 814],
    "Mistral-7B": [629, 769, 1033, 841],
    "R1-Llama-8B": [629, 761, 1053, 841],
}
FS = {
    "Llama-3-8B": [550, 550, 550, 614],
    "Mistral-7B": [569, 569, 569, 641],
    "R1-Llama-8B": [561, 561, 561, 641],
}

# Contest peak and fan-out, D=2048, k=4, tp=2.
CONTEST = {
    "R1 MATH": {"APC": (1863, 1604), "FS": (1391, 2375), "FS+": (1391, 1553)},
    "R1 AIME": {"APC": (2118, 613), "FS": (1646, 988), "FS+": (1646, 670)},
    "4B MATH": {"APC": (1937, 1178), "FS": (1521, 1750), "FS+": (1521, 1223)},
    "4B AIME": {"APC": (2150, 462), "FS": (1734, 748), "FS+": (1734, 501)},
}


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.2,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.6,
            "axes.edgecolor": C_INK,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig, name: str) -> None:
    for dest in (FIG, PAPER):
        fig.savefig(dest / f"{name}.pdf")
        fig.savefig(dest / f"{name}.png")
    plt.close(fig)


def fig_branch() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7), sharey=True)
    x = np.arange(len(CONFIGS))
    w = 0.36
    for ax, name in zip(axes, MODELS):
        ax.bar(x - w / 2, APC[name], w, color=C_APC, edgecolor=C_INK, lw=0.3, label="APC")
        ax.bar(x + w / 2, FS[name], w, color=C_FS, edgecolor=C_INK, lw=0.3, label="ForkServe")
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels(CONFIGS, rotation=20, ha="right")
        ax.yaxis.grid(True, color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("GSM8K peak KV (tokens)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "branch_peak")


def fig_longcot() -> None:
    labels = list(CONTEST)
    x = np.arange(len(labels))
    w = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.85))
    for ax, idx, ylab, title in (
        (axes[0], 0, "Peak KV (tokens)", "Contest peak KV"),
        (axes[1], 1, "Fan-out (ms)", "Contest fan-out"),
    ):
        apc = [CONTEST[k]["APC"][idx] for k in labels]
        fs = [CONTEST[k]["FS"][idx] for k in labels]
        plus = [CONTEST[k]["FS+"][idx] for k in labels]
        ax.bar(x - w, apc, w, color=C_APC, edgecolor=C_INK, lw=0.3, label="APC")
        ax.bar(x, fs, w, color=C_FS, edgecolor=C_INK, lw=0.3, label="ForkServe")
        ax.bar(x + w, plus, w, color=C_PLUS, edgecolor=C_INK, lw=0.3, label="ForkServe+")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.yaxis.grid(True, color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "longcot_peak")


if __name__ == "__main__":
    setup()
    fig_branch()
    fig_longcot()
    print("wrote", FIG, "and", PAPER)

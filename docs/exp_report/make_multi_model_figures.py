#!/usr/bin/env python3
"""Setting N: six-model GPU forest (n=40, D=512)."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
FIG = OUT / "fig"
FIG.mkdir(exist_ok=True)

C_REC = "#BBBBBB"
C_APC = "#0077BB"
C_FS = "#EE7733"
C_PLUS = "#009988"
C_INK = "#1A1A1A"
C_GRID = "#E6E6E6"

MODELS = [
    "Qwen3-4B",
    "Qwen2.5-M 1.5B",
    "Qwen3-14B",
    "Llama-3-8B",
    "Mistral-7B",
    "R1-Llama-8B",
]

# GSM8K peak KV (identical at tp=2 and tp=4)
PEAK_APC = [1512, 1512, 1512, 1509, 1569, 1405]
PEAK_FS = [1112, 1112, 1112, 1109, 1169, 1005]
PEAK_REC = [4464, 4464, 4464, 4452, 4668, 4036]

# GSM8K decode tokens tp=2
DEC_APC = [20480] * 6
DEC_FS = [9860, 17070, 12087, 5560, 10013, 14974]
DEC_PLUS = [7206, 11704, 11162, 5262, 9660, 6422]

# GSM8K accuracy tp=2
ACC_APC = [0.925, 0.225, 0.850, 0.625, 0.500, 0.650]
ACC_FS = [0.875, 0.275, 0.825, 0.675, 0.475, 0.650]
ACC_PLUS = [0.550, 0.275, 0.725, 0.675, 0.350, 0.000]

# Game24 peak
G24_APC = [1287, 1287, 1287, 1248, 1351, 1144]
G24_FS = [791, 791, 791, 760, 783, 656]


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
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def fig_peak_ratio() -> None:
    x = np.arange(len(MODELS))
    w = 0.36
    gsm = [a / b for a, b in zip(PEAK_FS, PEAK_APC)]
    g24 = [a / b for a, b in zip(G24_FS, G24_APC)]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.bar(x - w / 2, gsm, w, color=C_FS, edgecolor=C_INK, lw=0.3, label="GSM8K")
    ax.bar(x + w / 2, g24, w, color=C_PLUS, edgecolor=C_INK, lw=0.3, label="Game24")
    ax.axhline(1.0, color=C_APC, lw=0.8, ls="--", label="APC")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(r"peak$^{\mathrm{FS}}$ / peak$^{\mathrm{APC}}$")
    ax.set_title("Setting N peak KV ratio, $n{=}40$ (same at tp=2 and tp=4)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=18, ha="right")
    ax.legend(ncol=3, loc="upper right")
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbehind = True
    ax.set_axisbelow(True)
    save(fig, "multi_peak_ratio")


def fig_decode() -> None:
    x = np.arange(len(MODELS))
    w = 0.24
    fig, ax = plt.subplots(figsize=(7.6, 3.25))
    ax.bar(x - w, [d / 1000 for d in DEC_APC], w, color=C_APC, edgecolor=C_INK, lw=0.3, label="APC")
    ax.bar(x, [d / 1000 for d in DEC_FS], w, color=C_FS, edgecolor=C_INK, lw=0.3, label="ForkServe")
    ax.bar(x + w, [d / 1000 for d in DEC_PLUS], w, color=C_PLUS, edgecolor=C_INK, lw=0.3, label="ForkServe+")
    ax.set_ylabel(r"GSM8K decode tokens ($\times 10^3$), tp=2")
    ax.set_title("Setting N decoding: $nD{=}20.5$k for APC")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=18, ha="right")
    ax.legend(ncol=3)
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "multi_gsm8k_decode")


def fig_acc() -> None:
    x = np.arange(len(MODELS))
    w = 0.24
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.bar(x - w, ACC_APC, w, color=C_APC, edgecolor=C_INK, lw=0.3, label="APC")
    ax.bar(x, ACC_FS, w, color=C_FS, edgecolor=C_INK, lw=0.3, label="ForkServe")
    ax.bar(x + w, ACC_PLUS, w, color=C_PLUS, edgecolor=C_INK, lw=0.3, label="ForkServe+")
    ax.set_ylabel("GSM8K accuracy, tp=2")
    ax.set_title("Setting N quality: stop list is model-dependent")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, loc="upper right")
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "multi_gsm8k_acc")


if __name__ == "__main__":
    setup()
    fig_peak_ratio()
    fig_decode()
    fig_acc()
    print("wrote", FIG)

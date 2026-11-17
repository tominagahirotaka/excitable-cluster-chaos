#!/usr/bin/env python3
"""Figure for Sec. 7 of paper 2: the slow oscillations of runs 90, 46 and 26 on the fast section (existing data only;
2026-10-28).

Input: the reduced-system records of slow_return.py (analysis/switching_check/slow_return_<run>_<tag>.npz: the full
reduced state X at every upward crossing of X_1 = 0, no noise). Run 90 (N = 400, partition 133:132:109:22:4) with
rtol 1e-10, run 46 (N = 1000, 330:298:296:46:30) with rtol 1e-12, run 26 (N = 1000, 327:321:314:29:9) with
max_step 0.025. The states are projected on their first two principal components (as in RESULTS_NOTES 2026-10-25,
figs/torus_section_46_90.png, redrawn here in the style of the other paper-2 figures).
Output: analysis/paper2_figs/torus_sections.{pdf,png}

  python3 scripts/plot_paper2_ch7.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "analysis/switching_check"
OUT = ROOT / "analysis/paper2_figs"
INK, INK2, SER, GRID = "#0b0b0b", "#52514e", "#2a78d6", "#e4e3df"
CASES = [("T1", "slow_return_0090_r1e-10.npz"), ("T2", "slow_return_0046_r1e-12.npz"),
         ("T3", "slow_return_0026_h0.025.npz")]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(color=GRID, lw=.5)
    ax.set_axisbelow(True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for j, (name, f) in enumerate(CASES):
        z = np.load(S / f)
        X = z["X"]
        Xc = X - X.mean(axis=0)
        _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Xc @ Vt[:2].T
        ax = axs[j]
        style(ax)
        ax.scatter(P[:, 0], P[:, 1], s=.3, color=SER, lw=0, rasterized=True)
        ax.set_xlabel("PC1", fontsize=8, color=INK)
        ax.set_ylabel("PC2", fontsize=8, color=INK)
        ax.set_title(f"({'abc'[j]}) {name}", fontsize=8, color=INK, loc="left")
        print(name, "section points", len(X), "variance fraction of PC1+PC2", float((sv[:2] ** 2).sum() / (sv ** 2).sum()))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"torus_sections.{ext}", dpi=600 if ext == "pdf" else 220, bbox_inches="tight")  # PDF rasters at 600 dpi (AIP, combination art; 2026-11-03); PNG unchanged


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Figure for Sec. 4 of paper 2: the three chaotic cluster states on a Poincare section (existing data only; 2026-10-28).

Input: the phase-aligned full-system runs of chaos_edge_check.py full --phase_sec (_bubsec2; N = 100, noise 1e-8,
200000 time units, seed 1; analysis/chaos_check/edgefull_<run>_q<q>_bubsec2.json). sec_mean holds the cluster means
(X_1..X_m, Y_1..Y_m) at every upward crossing of X_1 = 0; the first 2000 time units are dropped. The states are at
eps K = -1.0 (run 95), -1.21 (run 3444) and -0.804 (run 143).
  top    projection on the two slow coordinates Y with the largest spread;
  bottom return map of the Y with the largest spread.
Output: analysis/paper2_figs/chaos_sections.{pdf,png}

  python3 scripts/plot_paper2_ch4.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "analysis/chaos_check"
OUT = ROOT / "analysis/paper2_figs"
INK, INK2, SER, GRID = "#0b0b0b", "#52514e", "#2a78d6", "#e4e3df"
CASES = [("C7", "edgefull_95_q-1.0000_bubsec2.json", r"$\varepsilon K=-1.0$"),
         ("C5a", "edgefull_3444_q-1.2100_bubsec2.json", r"$\varepsilon K=-1.21$"),
         ("C5b", "edgefull_143_q-0.8040_bubsec2.json", r"$\varepsilon K=-0.804$")]


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
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.4))
    for j, (name, f, lab) in enumerate(CASES):
        d = json.loads((C / f).read_text())
        t = np.asarray(d["sec_t"])
        Z = np.asarray(d["sec_mean"])[t > 2000]
        m = Z.shape[1] // 2
        sp = np.ptp(Z[:, m:], axis=0)
        k1, k2 = np.argsort(-sp)[:2]
        y1, y2 = Z[:, m + k1], Z[:, m + k2]
        for ax in axs[:, j]:
            style(ax)
        axs[0, j].scatter(y1, y2, s=.3, color=SER, lw=0, rasterized=True)
        axs[0, j].set_xlabel(f"$Y_{{{k1 + 1}}}$", fontsize=8, color=INK)
        axs[0, j].set_ylabel(f"$Y_{{{k2 + 1}}}$", fontsize=8, color=INK)
        axs[0, j].set_title(f"({'abc'[j]}) {name}, {lab}", fontsize=8, color=INK, loc="left")
        axs[1, j].scatter(y1[:-1], y1[1:], s=.3, color=SER, lw=0, rasterized=True)
        axs[1, j].set_xlabel(f"$Y_{{{k1 + 1}}}(n)$", fontsize=8, color=INK)
        axs[1, j].set_ylabel(f"$Y_{{{k1 + 1}}}(n+1)$", fontsize=8, color=INK)
        axs[1, j].set_title(f"({'def'[j]})", fontsize=8, color=INK, loc="left")
        print(name, "section points", len(Z), "coordinates", k1 + 1, k2 + 1, "sizes", d["sizes"])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"chaos_sections.{ext}", dpi=600 if ext == "pdf" else 220, bbox_inches="tight")  # PDF rasters at 600 dpi (AIP, combination art; 2026-11-03); PNG unchanged


if __name__ == "__main__":
    main()

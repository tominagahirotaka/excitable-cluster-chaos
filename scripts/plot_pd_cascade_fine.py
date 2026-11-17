#!/usr/bin/env python3
"""Bifurcation diagrams of the period-doubling routes to chaos with eps K step 0.0005 (exploratory, 2026-10-24).

Input: analysis/chaos_check/scan_<tag><index>_q<q>.json written by scripts/chaos_check.py scan with four decimals in
q (these contain the last 300 section states); crossings of -1 from pd_[low133_]L*.json.
For each state (run 95: seven clusters 28:20:17:12:11:7:5, eps = 0.25; run 3444 = low_cluster index 133: five
clusters, eps = 0.35):
  (a) bifurcation diagram: the slow section coordinate y_k (upward x_1 = 0) with the largest spread over the route (eps K
      down to the end of the chaotic range), last 300 returns at each eps K;
  (b) return times minus their mean (last 300 returns);
  (c) largest Lyapunov exponent (two trajectories, 15000 time units); dotted lines: period-doubling points from
      shooting.
Output: analysis/chaos_check/pd_cascade_fine_<name><--suffix>.png and .pdf
(2026-10-28: --width/--height/--suffix for a column-width version for paper 2, and --notitle, --labels for one
 figure of two columns, e.g. --width 3.4 --height 4.2 --suffix _col --notitle --labels abc def (the command of the paper-2 figure, checked to reproduce it pixel for pixel); the defaults leave
 size, names, title and panel letters unchanged.)
"""
from pathlib import Path
import argparse
import glob
import json
import re

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "analysis/chaos_check"
INK, INK2, SER, GRID = "#0b0b0b", "#52514e", "#2a78d6", "#e4e3df"
# (name, scan pattern, crossing pattern, title, eps K range used to choose the plotted coordinate: the route only,
#  excluding the jump to another orbit beyond the end of the chaotic range)
CASES = [("95", "scan_95_q*.json", "pd_L*.json", r"seven clusters 28:20:17:12:11:7:5, $\varepsilon=0.25$", -1.004),
         ("3444", "scan_low133_q*.json", "pd_low133_L*.json", r"five clusters 34:32:20:13:1, $\varepsilon=0.35$ (run 3444)",
          -1.2120)]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=.5)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=7.2)
    ap.add_argument("--height", type=float, default=7.4)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--notitle", action="store_true", help="no figure title (the caption names the state)")
    ap.add_argument("--labels", nargs=2, default=["abc", "abc"], metavar=("RUN95", "RUN3444"),
                    help="panel letters for the two states, e.g. abc def when both columns form one figure")
    a = ap.parse_args()
    for (name, pat, pdpat, title, qsel), letters in zip(CASES, a.labels):
        files = [f for f in glob.glob(str(D / pat)) if re.search(r"_q[+-]\d\.\d{4}\.json$", f)]
        scans = sorted((json.loads(Path(f).read_text()) for f in files), key=lambda s: s["q"])
        if not scans:
            print(name, "no fine scans yet")
            continue
        S = [np.asarray(s["section_states"]) for s in scans]
        allz = np.vstack([z for sc, z in zip(scans, S) if sc["q"] >= qsel])
        m = allz.shape[1] // 2
        spread = np.ptp(allz, axis=0)
        spread[:m] = 0  # slow coordinates only: the x's on the section vary through the fast jumps
        k = int(np.argmax(spread))
        lab = f"$X_{k + 1}$" if k < m else f"$Y_{k - m + 1}$"
        fig, ax = plt.subplots(3, 1, figsize=(a.width, a.height), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1, .7]))
        for x in ax:
            style(x)
        for s, z in zip(scans, S):
            ax[0].scatter(np.full(len(z), s["q"]), z[:, k], s=.4, color=SER, lw=0, rasterized=True)
            rt = np.asarray(s["return_times"])
            ax[1].scatter(np.full(rt.size, s["q"]), rt - rt.mean(), s=.4, color=SER, lw=0, rasterized=True)
        ax[2].plot([s["q"] for s in scans], [s["lyap_max"] for s in scans], color=SER, lw=1, marker="o", ms=1.6)
        ax[2].axhline(0, color=INK2, lw=.6)
        ax[0].set_ylabel(lab + " on the section")
        ax[1].set_ylabel(r"$T_n-\langle T\rangle$")
        ax[2].set_ylabel(r"$\lambda_{\max}$")
        ax[2].set_xlabel(r"$\varepsilon K$")
        lo, hi = scans[0]["q"], scans[-1]["q"]
        ax[2].set_xlim(hi + .001, lo - .001)  # stronger repulsion to the right
        for f in sorted(D.glob(pdpat)):
            if "win" in f.name:
                continue
            for c in json.loads(f.read_text())["crossings"]:
                for x in ax:
                    x.axvline(c["q"], color=INK2, lw=.6, ls=":")
        for x, t in zip(ax, letters):
            x.set_title(f"({t})", loc="left", fontsize=9, color=INK)
        if not a.notitle:
            fig.suptitle(title, fontsize=9, color=INK)
        if a.width < 5:  # column-width version (2026-10-28): room for the panel letters, shorter axis labels
            ax[0].set_ylabel(lab, fontsize=9)
            for x in ax:
                x.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            fig.tight_layout(h_pad=1.4)
        fig.savefig(D / f"pd_cascade_fine_{name}{a.suffix}.png", dpi=220, bbox_inches="tight")
        fig.savefig(D / f"pd_cascade_fine_{name}{a.suffix}.pdf", dpi=600, bbox_inches="tight")  # PDF rasters at 600 dpi (AIP, combination art; 2026-11-03); PNG unchanged
        print(name, len(scans), "points, coordinate", lab)


if __name__ == "__main__":
    main()

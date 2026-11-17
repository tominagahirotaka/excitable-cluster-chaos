#!/usr/bin/env python3
"""Two figures for paper 2 from existing data only (no new integration; 2026-10-28, docs/PAPER2_ORGANIZATION.md).

1. transverse_range: for the three chaotic examples (run 95, run 3444, run 143), against eps K,
   top    the largest Lyapunov exponent of the reduced system,
   bottom the largest transverse (internal) exponent over the clusters of size >= 2 (lambda_perp; < 0: every cluster
          is transversely stable on average along the attractor).
   Sources: independent scans from the saved state (run 95: scan_95_q*.json, run 3444: scan_low133_q*.json, both eps K
   step 0.0005, 15000 time units; run 143: reduced_chaos_census/candidate_run143_q*.json, step 0.002, 15000) and the
   walks of chaos_edge_check.py (each point started from the state of the previous one, 10000 time units; 5000 for
   the fine walks of run 3444) towards stronger repulsion ("forward") and back. Filled marker: chaotic (no period
   found up to lag 60 and lambda_max > 1e-3); open marker: otherwise (periodic, or no period found with lambda_max
   <= 1e-3, e.g. slow convergence to a periodic orbit near the end of a window). Dotted lines: the ends of the chaotic
   range named in RESULTS_NOTES 2026-10-27 (run 95: fold of period 3 at -1.0038; run 3444: entry into the period-5
   window near -1.21175 and the saddle-node of the stable period-5 branch at -1.2128302; run 143: jump to the
   coexisting period-1 orbit A between -0.8084 and -0.8086).
2. p5_branch_3444: the period-5 orbit of run 3444 continued by shooting with pseudo-arclength (p5_arc_3444.py:
   p5arc_3444_exit.json, from -1.21278 towards stronger repulsion through both folds, and p5arc_3444_pdcheck.json,
   from -1.21278 back to the period doubling).
   (a) one slow coordinate of the orbit point on the section x_1 = 0 (the Y with the largest spread along the branch)
       against eps K; solid: stable (all collective multipliers but the trivial one and all internal multipliers
       inside the unit circle), dashed: unstable;
   (b) the largest modulus of the collective (longitudinal) multipliers (trivial one removed) and of the internal
       (transverse) multipliers along the branch. Events from the continuation: saddle-node (fold, a real multiplier
       through +1) and period doubling (a real multiplier through -1; signed values checked 2026-10-28: on branch a
       -0.903 -> -1.017, a period doubling; on branch b the leading pair is complex, -0.43 +- 0.71i, at the point
       before and real, -1.17 and -0.62, at the point after, so the way it leaves the unit circle is not resolved:
       marked as an unresolved loss of stability, not as a period doubling).
   Run 143 in figure 1: below its lambda_perp panel, the chaotic points only (their lambda_perp merges with the
   zero line at the full scale).
   The internal multipliers stay inside the unit circle along the whole computed branch, so the instability on the
   dashed parts is collective (longitudinal).
Output: analysis/paper2_figs/transverse_range.{pdf,png}, p5_branch_3444.{pdf,png}

  python3 scripts/plot_paper2_figs.py
"""
from pathlib import Path
import glob
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "analysis/chaos_check"
RC = ROOT / "analysis/reduced_chaos_census"
OUT = ROOT / "analysis/paper2_figs"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SER = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference categorical slots 1-3 (validated all-pairs)

CASES = [
    dict(name="C7", sub=r"7 clusters, $\varepsilon=0.25$",
         scans=[f for f in glob.glob(str(C / "scan_95_q*.json")) if re.search(r"_q[+-]\d\.\d{4}\.json$", f)],
         walks=["edge_95_onset", "edge_95_end_fwd", "edge_95_end_bwd"], xlim=(-0.94, -1.012),
         ends=[(-1.0038, "period 3, fold\nnot confirmed", "center")]),
    dict(name="C5a", sub=r"5 clusters, $\varepsilon=0.35$",
         scans=[f for f in glob.glob(str(C / "scan_low133_q*.json")) if re.search(r"_q[+-]\d\.\d{4}\.json$", f)],
         walks=["edge_3444_onset", "edge_3444_end_fwd", "edge_3444_end_bwd", "edge_3444_fine_fwd",
                "edge_3444_win_bwd"], xlim=(-1.18, -1.2145),
         ends=[(-1.21175, "period-5\nwindow ", "right"), (-1.2128302, " SN of\n period 5", "left")]),
    dict(name="C5b", sub=r"5 clusters, $\varepsilon=0.375$",
         scans=glob.glob(str(RC / "candidate_run143_q*_r1e-10.json")),
         walks=["edge_143_onset", "edge_143_end_fwd", "edge_143_end_bwd", "edge_143_A_bwd"], xlim=(-0.77, -0.8305),
         ends=[(-0.8085, "jump to\norbit A", "center")]),
]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.2)
    ax.grid(color=GRID, lw=.5)
    ax.set_axisbelow(True)


def scan_points(files):
    pts = []
    for f in files:
        d = json.loads(Path(f).read_text())
        lam = d.get("lyap_max", d.get("lyap"))
        perp = d.get("internal_max", d.get("internal_max_big"))
        pts.append((d["q"], lam, perp, d["lag"]))
    return sorted(pts)


def walk_points(name):
    d = json.loads((C / f"{name}.json").read_text())
    P = d["points"]
    q = [p["q"] for p in P]
    fwd = q[-1] < q[0]  # towards stronger repulsion (more negative eps K)
    return fwd, [(p["q"], p["lam"], p["perp_max"], p["lag"]) for p in P]


def draw(ax, pts, k, color, marker, size):
    for chaotic in (True, False):
        sel = [p for p in pts if (p[3] is None and p[1] > 1e-3) == chaotic]
        if not sel:
            continue
        ax.scatter([p[0] for p in sel], [p[k] for p in sel], s=size, marker=marker,
                   facecolors=color if chaotic else "none", edgecolors=color, linewidths=.7, zorder=3)


def transverse_range():
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 3)
    axs = np.empty((2, 3), dtype=object)
    for j in range(3):
        axs[0, j] = fig.add_subplot(gs[0, j])
        if j < 2:
            axs[1, j] = fig.add_subplot(gs[1, j], sharex=axs[0, j])
    # run 143: the lambda_perp panel is split; below it the chaotic points only, whose lambda_perp (-0.001..-0.005)
    # merges with the zero line at the full scale
    sub = gs[1, 2].subgridspec(2, 1, height_ratios=[1.5, 1], hspace=.55)
    axs[1, 2] = fig.add_subplot(sub[0], sharex=axs[0, 2])
    zoom = fig.add_subplot(sub[1])
    for j, case in enumerate(CASES):
        sp = scan_points(case["scans"])
        for row, k in ((0, 1), (1, 2)):
            ax = axs[row, j]
            style(ax)
            draw(ax, sp, k, SER[0], "o", 9)
            for w in case["walks"]:
                fwd, wp = walk_points(w)
                draw(ax, wp, k, SER[1] if fwd else SER[2], "^" if fwd else "v", 9)
            ax.axhline(0, color=INK2, lw=.7)
            for qe, lab, ha in case["ends"]:
                ax.axvline(qe, color=INK2, lw=.6, ls=":")
                if row == 0:
                    ax.text(qe, 1.0, lab, transform=ax.get_xaxis_transform(), fontsize=8.2, color=INK2,
                            ha=ha, va="bottom")
            ax.set_xlim(*case["xlim"])  # stronger repulsion to the right, as pd_cascade_fine
            if row == 0:
                plt.setp(ax.get_xticklabels(), visible=False)
        axs[0, j].set_title(f"({'abc'[j]}) {case['name']}, {case['sub']}", fontsize=9, color=INK, loc="left",
                            pad=34)
        if j < 2:
            axs[1, j].set_xlabel(r"$\varepsilon K$ (stronger repulsion $\rightarrow$)", fontsize=9, color=INK)
        else:
            style(zoom)
            ch = [p for p in sp if p[3] is None and p[1] > 1e-3]
            draw(zoom, ch, 2, SER[0], "o", 8)
            for w in case["walks"]:
                fwd, wp = walk_points(w)
                draw(zoom, [p for p in wp if p[3] is None and p[1] > 1e-3], 2, SER[1] if fwd else SER[2],
                     "^" if fwd else "v", 8)
            zoom.axhline(0, color=INK2, lw=.7)
            zoom.axvline(case["ends"][0][0], color=INK2, lw=.6, ls=":")
            zoom.set_xlim(-.7985, -.8095)
            zoom.set_ylim(-.0065, .001)
            zoom.set_yticks([-.005, 0])
            zoom.set_xticks([-.800, -.805])
            zoom.set_title("chaotic points only", fontsize=8.2, color=INK2, loc="left", pad=2)
            zoom.set_xlabel(r"$\varepsilon K$ (stronger repulsion $\rightarrow$)", fontsize=9, color=INK)
    axs[0, 0].set_ylabel(r"$\lambda_{\max}$", fontsize=9, color=INK)
    axs[1, 0].set_ylabel(r"$\lambda_\perp$ (largest internal exponent)", fontsize=9, color=INK)
    handles = [Line2D([], [], ls="", marker="o", color=SER[0], ms=4, label="scan from the saved state"),
               Line2D([], [], ls="", marker="^", color=SER[1], ms=4, label=r"continuation, $\varepsilon K$ decreasing"),
               Line2D([], [], ls="", marker="v", color=SER[2], ms=4, label=r"continuation, $\varepsilon K$ increasing"),
               Line2D([], [], ls="", marker="o", mfc="none", mec=INK2, ms=4, label="open: not chaotic")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(.5, -.06))  # two rows (2026-11-03: longer labels)
    fig.subplots_adjust(left=.09, right=.99, top=.88, bottom=.17, wspace=.38, hspace=.32)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"transverse_range.{ext}", dpi=600 if ext == "pdf" else 220, bbox_inches="tight")  # PDF rasters at 600 dpi (AIP, combination art; 2026-11-03); PNG unchanged
    plt.close(fig)
    for case in CASES:
        sp = scan_points(case["scans"])
        ch = [p for p in sp if p[3] is None and p[1] > 1e-3]
        print(case["name"], "scan points", len(sp), "chaotic", len(ch),
              "lambda_perp on chaotic scan points: max", max(p[2] for p in ch) if ch else None)


def p5_branch():
    A = json.loads((C / "p5arc_3444_exit.json").read_text())
    Bp = json.loads((C / "p5arc_3444_pdcheck.json").read_text())
    pts = list(reversed(Bp["points"]))[:-1] + A["points"]  # one path: PD side -> -1.21278 -> folds -> beyond
    q = np.array([p["q"] for p in pts])
    Z = np.array([p["z"] for p in pts])
    m = Z.shape[1] // 2
    k = m + int(np.argmax(np.ptp(Z[:, m:], axis=0)))
    y = Z[:, k]
    coll = np.array([p["collective_max"] for p in pts])
    intl = np.array([p["internal_max"] for p in pts])
    stable = (coll < 1) & (intl < 1)
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1]))
    for a in ax:
        style(a)
    # the branch, solid where stable and dashed where unstable, split into runs of equal stability
    i = 0
    while i < len(q):
        j = i
        while j + 1 < len(q) and stable[j + 1] == stable[i]:
            j += 1
        s = slice(i, min(j + 2, len(q)))
        ax[0].plot(q[s], y[s], color=SER[0] if stable[i] else SER[1], lw=1.6 if stable[i] else 1.2,
                   ls="-" if stable[i] else "--")
        i = j + 1
    ax[1].plot(q, coll, color=SER[0], lw=1.2, label="collective (longitudinal)")
    ax[1].plot(q, intl, color=SER[2], lw=1.2, label="internal (transverse)")
    ax[1].axhline(1, color=INK2, lw=.7)
    ax[1].set_yscale("log")
    ev = [(e, A) for e in A["events"]] + [(e, Bp) for e in Bp["events"]]
    seen = set()
    for e, src in ev:
        qe = float(np.mean(e["between"]))
        kind = "SN" if "fold" in e["kind"] else ("PD" if "-1" in e["kind"] else None)
        # on branch b (file A) the leading collective pair is complex (-0.43 +- 0.71i, modulus 0.83) at the point
        # before and real (-1.17, -0.62) at the point after: whether a real multiplier passed through -1 (period
        # doubling) or the complex pair left the unit circle first is not resolved by the continuation step
        if kind == "PD" and src is A:
            kind = "U"  # unresolved loss of stability (stable before, unstable after; type not resolved)
        if kind is None or (kind, round(qe, 6)) in seen:
            continue
        seen.add((kind, round(qe, 6)))
        # the path point where the event was found (its first neighbour in the continuation file); nearest in q
        # would put an event on the wrong branch of the S
        jj = int(np.argmin(abs(q - e["between"][0])))
        cand = np.nonzero(abs(q - e["between"][0]) < 1e-12)[0]
        if len(cand):
            jj = int(cand[0])
        ax[0].plot(qe, y[jj], marker={"SN": "o", "PD": "s", "U": "D"}[kind], ms=5, mfc="white", mec=INK, mew=.9,
                   zorder=4, ls="none")
        txt = {"SN": "SN", "PD": "PD"}.get(kind, "") + f" {qe:.7f}" if kind != "U" else \
            "type unresolved\n" + f"{e['between'][0]:.7f}\nto {e['between'][1]:.7f}"
        # labels of the two events at the left turn of the S are placed in the empty lower right, with lines
        far = {"U": (-1.21291, 3.13850), "SN": (-1.21291, 3.13728)} if (kind == "U" or (kind == "SN" and qe > -1.2128)) \
            else None
        if far:
            ax[0].annotate(txt, (qe, y[jj]), xytext=far[kind], textcoords="data", fontsize=8, color=INK2, ha="left",
                           va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=.6, shrinkA=2, shrinkB=4))
        else:
            ax[0].annotate(txt, (qe, y[jj]), textcoords="offset points", xytext=(8, -3), fontsize=8, color=INK2,
                           ha="left", va="center")
        ax[1].axvline(qe, color=INK2, lw=.6, ls=":")
        print(kind, qe)
    lab = f"$Y_{k - m + 1}$"
    ax[0].set_ylabel(lab + r" at the section $X_1=0$", fontsize=9, color=INK)
    ax[1].set_ylabel("Floquet multiplier modulus", fontsize=9, color=INK)
    ax[1].set_xlabel(r"$\varepsilon K$ (stronger repulsion $\rightarrow$)", fontsize=9, color=INK)
    ax[0].set_xlim(q.max() + 2e-5, q.min() - 2e-5)  # stronger repulsion to the right
    ax[0].legend(handles=[Line2D([], [], color=SER[0], lw=1.6, label="stable"),
                          Line2D([], [], color=SER[1], lw=1.2, ls="--",
                                 label="unstable (longitudinal)"),
                          Line2D([], [], ls="", marker="o", mfc="white", mec=INK, ms=5,
                                 label="saddle-node (real multiplier +1)"),
                          Line2D([], [], ls="", marker="s", mfc="white", mec=INK, ms=5,
                                 label="period doubling (real multiplier −1)"),
                          Line2D([], [], ls="", marker="D", mfc="white", mec=INK, ms=5,
                                 label="loss of stability, type unresolved")],
                 handlelength=1.6,
                 fontsize=7.5, frameon=False, loc="upper left")
    ax[0].set_ylim(y.min() - 2e-4, y.max() + 1.6e-3)  # room for the legend above the branch
    ax[1].legend(fontsize=7.5, frameon=False, loc="upper left")
    ax[1].set_xticks([-1.2127, -1.2128, -1.2129, -1.2130, -1.2131, -1.2132])
    ax[1].set_xticklabels(["$-1.2127$", "", "$-1.2129$", "", "$-1.2131$", ""])
    ax[0].set_title("(a) period-5 branch of C5a (shooting)", fontsize=9, color=INK, loc="left")
    ax[1].set_title("(b) multipliers along the branch", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"p5_branch_3444.{ext}", dpi=600 if ext == "pdf" else 220, bbox_inches="tight")  # PDF rasters at 600 dpi (AIP, combination art; 2026-11-03); PNG unchanged
    plt.close(fig)
    print("p5 branch points", len(q), "coordinate", lab, "internal max over branch", intl.max(),
          "stable eps K ranges", [(round(q[a], 7), round(q[b], 7)) for a, b in runs(stable)])


def runs(flag):
    out, i = [], 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j + 1 < len(flag) and flag[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    transverse_range()
    p5_branch()


if __name__ == "__main__":
    main()

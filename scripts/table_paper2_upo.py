#!/usr/bin/env python3
"""Table F2 for chapter 6 of paper 2: periodic orbits inside the chaotic ranges and their transverse stability
(existing data only; 2026-10-28, paper2/ch6_skeleton.md).

Input: the continuations of the periodic orbits of period L through the chaotic range made by pd_longitudinal_check.py
(analysis/chaos_check/pdlong_<run>_L<L>_chaos.json: shooting, collective and internal Floquet multipliers at every
eps K), and the points classified as chaotic in figure 1 (plot_paper2_figs.CASES: scans and walks; no period found
and lambda_max > 1e-3), which give the computed chaotic range of each example.
For every orbit, restricted to the eps K of the continuation inside the chaotic range:
  - period (min..max), the cluster with the largest internal multiplier modulus;
  - the largest internal (transverse) multiplier modulus, and the sub-range where it is > 1 (transversely unstable);
  - the largest collective multiplier modulus (trivial one removed): > 1 means the orbit is unstable inside the
    reduced system, as an orbit inside a chaotic range is expected to be.
The internal multiplier of an orbit is a property of that orbit; it is not the exponent lambda_perp of the typical
chaotic trajectory in figure 1. The range of eps K where an orbit was computed is not the range where it exists; the
continuation covers the chaotic range and was not followed beyond it.
chaotic_range in the output is the span of all points classified as chaotic in the scans and the continuation
walks; for C5a it reaches -1.21272, inside the interval with the period-five window and long transients, whereas the
header of Table II gives the interval of the independent scan (-1.1935 to -1.2115; caption of Table II).
Output: analysis/paper2_figs/upo_table.json and upo_table.md

  python3 scripts/table_paper2_upo.py
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_paper2_figs import CASES, scan_points, walk_points  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "analysis/chaos_check"
OUT = ROOT / "analysis/paper2_figs"
RUN = {"C7": "95", "C5a": "3444", "C5b": "143"}  # labels of plot_paper2_figs.CASES -> file tags


def chaotic_range(case):
    pts = scan_points(case["scans"])
    for w in case["walks"]:
        pts += walk_points(w)[1]
    q = [p[0] for p in pts if p[3] is None and p[1] > 1e-3]
    return min(q), max(q)


def main():
    rows = []
    for case in CASES:
        lo, hi = chaotic_range(case)
        tag = RUN[case["name"]]
        for f in sorted(C.glob(f"pdlong_{tag}_L*_chaos.json")):
            d = json.loads(f.read_text())
            P = [p for p in d["points"] if lo - 1e-12 <= p["q"] <= hi + 1e-12]
            if not P:
                continue
            q = np.array([p["q"] for p in P])
            im = np.array([p["internal_max"] for p in P])
            cm = np.array([p["collective_max"] for p in P])
            T = np.array([p["period"] for p in P])
            un = q[im > 1]
            rows.append(dict(example=case["name"], chaotic_range=[lo, hi], file=f.name, period_L=d["lag"],
                             computed_q=[float(q.min()), float(q.max())], points=len(P),
                             period=[float(T.min()), float(T.max())],
                             dominant_cluster_size=sorted({int(p["internal_dominant"][0]) for p in P}),
                             internal_max=[float(im.min()), float(im.max())],
                             transversely_unstable_q=[float(un.min()), float(un.max())] if len(un) else None,
                             transversely_unstable_points=int(len(un)),
                             collective_max=[float(cm.min()), float(cm.max())],
                             collectively_stable_points=int((cm < 1).sum())))
    (OUT / "upo_table.json").write_text(json.dumps(rows, indent=1))
    lines = ["| example (computed chaotic range of εK) | L | period | εK computed | largest internal modulus "
             "(cluster size) | transversely unstable for εK in | largest collective modulus |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        un = r["transversely_unstable_q"]
        lines.append(
            f"| {r['example']} ({r['chaotic_range'][1]:.4f} to {r['chaotic_range'][0]:.4f}) | {r['period_L']} | "
            f"{r['period'][0]:.2f}–{r['period'][1]:.2f} | {r['computed_q'][1]:.4f} to {r['computed_q'][0]:.4f} "
            f"({r['points']} points) | {r['internal_max'][0]:.3f}–{r['internal_max'][1]:.3f} "
            f"({', '.join(map(str, r['dominant_cluster_size']))}) | "
            + (f"{un[1]:.4f} to {un[0]:.4f} ({r['transversely_unstable_points']} points)" if un else "none")
            + f" | {r['collective_max'][0]:.2f}–{r['collective_max'][1]:.2f} "
              f"(stable at {r['collectively_stable_points']} points) |")
    (OUT / "upo_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

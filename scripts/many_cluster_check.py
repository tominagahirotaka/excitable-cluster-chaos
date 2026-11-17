#!/usr/bin/env python3
"""Are the final states with many (6-10) clusters of the N-dependence ensembles stable periodic states or transients?
(exploratory, 2026-10-22)

Question (author): Supplementary S10.2 lists final states with up to 10 clusters for N = 50-400. Each was judged at
t = 40000 by a constant count over the final checkpoints. Could they be transients?

Two tests per run, on the saved final state:
  1. reduced audit (cluster_state_floquet.audit_run): the m-cluster periodic orbit from the centroids, shooting, and
     the full-system Floquet audit (collective multipliers and the internal 2x2 block of every cluster). Internal
     instability cannot grow from an exactly synchronized state in a direct integration, so a state that is
     internally unstable can look converged.
  2. perturbed continuation: independent Gaussian noise (sd --sigma) on all 2N coordinates, then full-system RK4
     (h = 0.005, NumPy) for --tmax time units; cluster counts (max-norm components, threshold 1e-4) at
     --checks.

  extract : write the runs with --counts (unflagged and flagged) and saved final states to
            analysis/many_cluster_check/selected.json
  run     : tests for one selected run (--index)  -> analysis/many_cluster_check/run_<index>.json
  python3 scripts/many_cluster_check.py extract
  python3 scripts/many_cluster_check.py run --index 0        (remote host: scipy)
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import ROOT  # noqa: E402

OUT = ROOT / "analysis/many_cluster_check"
TAGS = {50: "N50", 100: "N100final", 200: "N200", 400: "N400", 1000: "N1000"}
EPS = {.2, .25, .3, .35, .4}
QS = {-.4, -.5, -.6, -.7, -.8, -.9, -1., -1.2}
B, G = .7, .5


def extract(counts):
    sel = []
    for n, tag in TAGS.items():
        for r in json.loads((ROOT / f"analysis/eps_cluster_ensemble/runs_{tag}.json").read_text()):
            if (round(r["eps"], 3) in EPS and round(r["eps"] * r["K"], 3) in QS and 5001 <= r["seed"] < 5051
                    and r["clusters"] in counts and "final_state" in r):
                sel.append(dict(N=n, **{k: r[k] for k in ("eps", "K", "seed", "clusters", "sizes", "final_state",
                                                          "thresholds_agree", "tail_constant")}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selected.json").write_text(json.dumps(sel))
    print(len(sel), "runs")


def count(z, n):
    d = np.maximum(abs(z[:n, None] - z[None, :n]), abs(z[n:, None] - z[None, n:]))
    m, lab = connected_components(d < 1e-4, directed=False)
    return int(m), sorted(np.bincount(lab).tolist(), reverse=True)


def continue_full(z, e, K, tmax, checks, h=.005):
    n = z.size // 2

    def f(v):
        x, y = v[:n], v[n:]
        return np.r_[(x - x ** 3 / 3 - y) / e + K * (x.mean() - x), e * (x + B - G * y)]
    out, t, k = [], 0., 0
    steps = int(round(tmax / h))
    marks = {int(round(c / h)): c for c in checks}
    for s in range(1, steps + 1):
        k1 = f(z)
        k2 = f(z + h / 2 * k1)
        k3 = f(z + h / 2 * k2)
        k4 = f(z + h * k3)
        z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if s in marks:
            m, sizes = count(z, n)
            out.append(dict(t=marks[s], clusters=m, sizes=sizes))
    return out


def run(index, sigma, tmax, checks):
    from cluster_state_floquet import audit_run
    r = json.loads((OUT / "selected.json").read_text())[index]
    row = {k: r[k] for k in ("N", "eps", "K", "seed", "clusters", "sizes", "thresholds_agree", "tail_constant")}
    try:
        au = audit_run(r)
        row["audit"] = {k: au.get(k) for k in ("status", "period", "lag", "collective_max", "internal_max",
                                                "max_all", "dominant_kind", "valid", "stable")}
    except Exception as ex:  # keep going: the perturbation test does not need the audit
        row["audit"] = dict(status=f"error: {ex}")
    z = np.asarray(r["final_state"], float)
    z = z + np.random.default_rng(index).normal(0, sigma, z.size)
    row["perturbed"] = continue_full(z, r["eps"], r["K"], tmax, checks)
    row["sigma"] = sigma
    (OUT / f"run_{index:04d}.json").write_text(json.dumps(row))
    a = row["audit"]
    print(index, row["N"], row["eps"], round(row["eps"] * row["K"], 2), row["seed"], row["clusters"],
          a.get("status"), a.get("stable"), a.get("max_all"),
          [c["clusters"] for c in row["perturbed"]], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "run"])
    ap.add_argument("--counts", default="6,7,8,9,10")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=1e-6)
    ap.add_argument("--tmax", type=float, default=20000.)
    ap.add_argument("--checks", default="1000,5000,10000,20000")
    a = ap.parse_args()
    if a.mode == "extract":
        extract({int(c) for c in a.counts.split(",")})
    else:
        run(a.index, a.sigma, a.tmax, [float(c) for c in a.checks.split(",")])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Floquet audit of the final cluster states of an ensemble with saved final states (exploratory).

Question (docs/RESULTS_NOTES.md, 2026-10-21): near eps = 0.3, groups of five comparable clusters formed early stay
five clusters; elsewhere about half merge into three. How strongly attracting are the 5- and 3-cluster states,
as a function of eps?

Per run of analysis/eps_cluster_ensemble/runs_<tag>.json with the requested cluster counts:
  1. clusters of the final state: components at max-norm threshold 1e-4 (as in fhn_replay.c); centroids;
  2. the reduced m-cluster system (eps_three_cluster.Partition, general m) is integrated from the centroids for up to
     t = 400 with the section x_1 = 0 (upward); the smallest lag (1-4 section returns) whose return error is
     < 1e-6 gives the period guess;
  3. shooting (tolerance 2e-11) and the full audit (Partition.audit: nontrivial collective multipliers and the
     2x2 internal blocks of every cluster).
Runs whose reduction or shooting fails are kept with a status.

  python3 scripts/cluster_state_floquet.py --tag mech --counts 3,5 --jobs 32      (remote host: needs scipy)
Output: analysis/cluster_state_floquet/audit_<tag>.json
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import Partition, ROOT  # noqa: E402

OUT = ROOT / "analysis/cluster_state_floquet"
TOL = 2e-11


def audit_run(r):
    out = {k: r[k] for k in ("eps", "K", "seed", "clusters", "sizes")}
    z = np.asarray(r["final_state"])
    N = z.size // 2
    d = np.maximum(abs(z[:N, None] - z[None, :N]), abs(z[N:, None] - z[None, N:]))
    m, lab = connected_components(d < 1e-4, directed=False)
    if m != r["clusters"]:
        return dict(out, status="component_mismatch", m=int(m))
    sizes = np.bincount(lab)
    part = Partition(sizes)
    z0 = np.r_[[z[:N][lab == a].mean() for a in range(m)], [z[N:][lab == a].mean() for a in range(m)]]
    e, K = r["eps"], r["K"]

    def ev(t, v):
        return v[0]
    ev.direction = 1
    s = solve_ivp(lambda t, v: part.rhs(v, K, e), (0, 400), z0, method="DOP853", rtol=1e-10, atol=1e-12,
                  max_step=.05, events=ev)
    ts, ys = s.t_events[0], s.y_events[0]
    if len(ts) < 6:
        return dict(out, status="no_section_returns")
    for lag in range(1, 5):
        if len(ts) > lag + 1 and np.max(abs(ys[-1] - ys[-1 - lag])) < 1e-6:
            break
    else:
        return dict(out, status="no_recurrence")
    w = part.shoot(np.r_[ys[-1], ts[-1] - ts[-1 - lag]], K, e, TOL)
    if w is None:
        return dict(out, status="shooting_failed", lag=lag)
    a = part.audit(w, K, e, TOL)
    return dict(out, status="ok", lag=lag, cluster_sizes=sizes.tolist(),
                **{k: a[k] for k in ("period", "collective_max", "internal_max", "max_all", "dominant_kind",
                                     "dominant_population", "dominant_eigenvalue", "min_pair_cycle_distance",
                                     "valid", "stable")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mech")
    ap.add_argument("--counts", default="3,5")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first runs only (test)")
    a = ap.parse_args()
    counts = {int(c) for c in a.counts.split(",")}
    runs = [r for r in json.loads((ROOT / f"analysis/eps_cluster_ensemble/runs_{a.tag}.json").read_text())
            if "final_state" in r and r["clusters"] in counts]
    if a.limit:
        runs = runs[:a.limit]
    with ProcessPoolExecutor(a.jobs) as pool:
        rows = list(pool.map(audit_run, runs, chunksize=4))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"audit_{a.tag}{'_test' if a.limit else ''}.json").write_text(json.dumps(rows))
    st = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    print(len(rows), st, flush=True)


if __name__ == "__main__":
    main()

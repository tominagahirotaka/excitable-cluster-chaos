#!/usr/bin/env python3
"""Check of the five-cluster chaos candidate of the second reduced-system census (exploratory, 2026-10-24).

Candidate: analysis/reduced_chaos_census_runs.json entry --run (default 143; 11:24:29:31:5, eps = 0.375,
eps K = -0.8). From its saved centroids, the reduced system at eps K = --q: transient 5000, then --length time units
with the two-trajectory largest exponent, transverse exponents of all clusters, and section returns (smallest lag
<= 60 with return distance < 1e-6 over the last 50 returns), with rtol --rtol.
-> analysis/reduced_chaos_census/candidate_run<run>_q<q>_r<rtol>.json

  python3 scripts/chaos_candidate_check.py --q -0.8 --rtol 1e-11 --length 60000
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import Partition, ROOT  # noqa: E402
import chaos_check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=143)
    ap.add_argument("--q", type=float, default=-.8)
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--length", type=float, default=15000.)
    a = ap.parse_args()
    r = json.loads((ROOT / "analysis/reduced_chaos_census_runs.json").read_text())[a.run]
    c = np.asarray(r["centroids"], float)
    part = Partition(c[:, 2].astype(int).tolist())
    e = r["eps"]
    K = a.q / e
    opt = dict(method="DOP853", rtol=a.rtol, atol=a.rtol / 100, max_step=.05)
    v = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 5000), np.r_[c[:, 0], c[:, 1]], **opt).y[:, -1]
    _, lam, intl, tr, X = chaos_check.two_traj(part, v, K, e, a.length, opt, internal=True, returns=True)
    lag = None
    for L in range(1, 61):
        if len(X) > L + 50 and max(np.max(abs(X[k] - X[k - L])) for k in range(len(X) - 50, len(X))) < 1e-6:
            lag = L
            break
    big = part.sizes > 1
    res = dict(run=a.run, eps=e, q=a.q, sizes=part.sizes.tolist(), rtol=a.rtol, length=a.length, lyap=float(lam),
               internal=intl.tolist(), internal_max_big=float(intl[big].max()), lag=lag,
               return_mean=float(np.diff(tr).mean()), return_sd=float(np.diff(tr).std()),
               section_states=np.asarray(X[-300:]).tolist())
    out = ROOT / f"analysis/reduced_chaos_census/candidate_run{a.run}_q{a.q:+.4f}_r{a.rtol:g}.json"
    out.write_text(json.dumps(res))
    print(a.run, a.q, a.rtol, "lyap", round(lam, 5), "internal_max_big", round(res["internal_max_big"], 5), "lag", lag,
          flush=True)


if __name__ == "__main__":
    main()

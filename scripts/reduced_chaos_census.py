#!/usr/bin/env python3
"""Chaos in the reduced m-cluster system as a function of the cluster count m (exploratory, 2026-10-23).

Question (docs/LITERATURE_PAPER2.md): does chaos between clusters already occur with two clusters (the setting of
Yanagita et al. 2005 and Franovic et al. 2022), or first with five or seven?

One job = one sample: cluster count --m, eps, eps K = --q, sample --s. The partition of N = 100 is drawn from a flat
Dirichlet distribution (sizes >= 1, summing to 100), the centroids uniformly (x in [-2, 2], y in [-0.6, 1.4]), both
from PCG64 seed hash(m, eps, q, s). Reduced system (eps_three_cluster.Partition), DOP853 rtol 1e-10, atol 1e-12,
max step 0.05: transient 3000, then 10000 time units with the two-trajectory largest exponent, the transverse
exponent of every cluster (internal 2x2 block) and upward x_1 = 0 returns (smallest lag <= 60 with return distance
< 1e-6 over the last 50 returns).
-> analysis/reduced_chaos_census/m<m>_e<eps>_q<q>_s<s>.json

  python3 scripts/reduced_chaos_census.py --m 2 --eps 0.25 --q -1.0 --s 0
  python3 scripts/reduced_chaos_census.py jobs     (writes analysis/reduced_chaos_census_jobs.txt)

Second census (2026-10-24; random centroids mostly collapse to fewer clusters for m >= 3): start from the final
states reached by the full system. 'select' draws (PCG64 seed 0) up to 150 runs each with 5, 6 and 7 clusters from
analysis/eps_cluster_ensemble/runs_band.json (N = 100, eps in 0.2-0.4, eps K in [-1.2, -0.4], both flags true) and
writes analysis/reduced_chaos_census_runs.json and analysis/reduced_chaos_census_jobs2.txt; '--run i --dq d' starts
the reduced system of run i from its saved centroids at eps K = (own eps K) + d, d in {0, +-0.02, +-0.05, +-0.1}.
-> analysis/reduced_chaos_census/run<i>_dq<d>.json
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

OUT = ROOT / "analysis/reduced_chaos_census"
EPS = (.2, .25, .3, .35, .4)
QS = (-.4, -.5, -.6, -.7, -.8, -.9, -1., -1.2)
MS = (2, 3, 4, 5, 6, 7)
SAMPLES = 12


def partition(rng, m, n=100):
    while True:
        s = np.floor(rng.dirichlet(np.ones(m)) * n).astype(int)
        s[np.argmax(s)] += n - s.sum()
        if s.min() >= 1:
            return sorted(s.tolist(), reverse=True)


def run(a):
    if a.run is not None:
        r = json.loads((ROOT / "analysis/reduced_chaos_census_runs.json").read_text())[a.run]
        c = np.asarray(r["centroids"], float)  # rows (x, y, size)
        sizes = c[:, 2].astype(int).tolist()
        a.m, a.eps, a.q, a.s = len(sizes), r["eps"], round(r["eps"] * r["K"] + a.dq, 4), a.run
        v = np.r_[c[:, 0], c[:, 1]]
    else:
        rng = np.random.default_rng([a.m, int(round(a.eps * 1000)), int(round(-a.q * 1000)), a.s])
        sizes = partition(rng, a.m)
        v = np.r_[rng.uniform(-2, 2, a.m), rng.uniform(-.6, 1.4, a.m)]
    e, K = a.eps, a.q / a.eps
    part = Partition(sizes)
    opt = dict(method="DOP853", rtol=1e-10, atol=1e-12, max_step=.05)
    s = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 3000), v, **opt)
    v = s.y[:, -1]
    if np.max(abs(v[:a.m, None] - v[None, :a.m])) < 1e-6 and np.max(abs(v[a.m:, None] - v[None, a.m:])) < 1e-6:
        res = dict(status="synchronized")
    else:
        _, lam, intl, tr, X = chaos_check.two_traj(part, v, K, e, 10000., opt, internal=True, returns=True)
        lag = None
        for L in range(1, 61):
            if len(X) > L + 50 and max(np.max(abs(X[k] - X[k - L])) for k in range(len(X) - 50, len(X))) < 1e-6:
                lag = L
                break
        z = v
        d = np.min([np.max(abs(z[[i, i + a.m]] - z[[j, j + a.m]])) for i in range(a.m) for j in range(i)])
        big = np.asarray(sizes) > 1
        res = dict(status="ok", lyap=float(lam), internal=intl.tolist(),
                   internal_max_big=float(intl[big].max()) if big.any() else None, lag=lag, returns=len(tr),
                   min_cluster_distance=float(d))
    res.update(m=a.m, eps=e, q=a.q, s=a.s, sizes=sizes, run=a.run, dq=a.dq)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"run{a.run}_dq{a.dq:g}.json" if a.run is not None else f"m{a.m}_e{e:g}_q{a.q:g}_s{a.s}.json"
    (OUT / name).write_text(json.dumps(res))
    print(a.m, e, a.q, a.s, sizes, res.get("lyap"), res.get("lag"), flush=True)


def jobs():
    lines = [f"scripts/reduced_chaos_census.py --m {m} --eps {e:g} --q {q:g} --s {s}"
             for s in range(SAMPLES) for m in MS for e in EPS for q in QS]
    (ROOT / "analysis/reduced_chaos_census_jobs.txt").write_text("\n".join(lines) + "\n")
    print(len(lines))


def select():
    r = json.loads((ROOT / "analysis/eps_cluster_ensemble/runs_band.json").read_text())
    r = r if isinstance(r, list) else r["runs"]
    rng = np.random.default_rng(0)
    out = []
    for m in (5, 6, 7):
        c = [x for x in r if x["clusters"] == m and x.get("thresholds_agree", True) and x.get("tail_constant", True)
             and .2 - 1e-9 <= x["eps"] <= .4 + 1e-9 and -1.2 - 1e-9 <= x["eps"] * x["K"] <= -.4 + 1e-9]
        idx = rng.choice(len(c), min(150, len(c)), replace=False)
        out += [dict(eps=c[i]["eps"], K=c[i]["K"], seed=c[i]["seed"], sizes=c[i]["sizes"],
                     centroids=c[i]["centroids"]) for i in sorted(idx)]
    (ROOT / "analysis/reduced_chaos_census_runs.json").write_text(json.dumps(out))
    lines = [f"scripts/reduced_chaos_census.py --run {i} --dq {d:g}" for d in (0, .02, -.02, .05, -.05, .1, -.1)
             for i in range(len(out))]
    (ROOT / "analysis/reduced_chaos_census_jobs2.txt").write_text("\n".join(lines) + "\n")
    print(len(out), len(lines))


def main():
    if sys.argv[1:2] == ["jobs"]:
        return jobs()
    if sys.argv[1:2] == ["select"]:
        return select()
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--eps", type=float, default=.25)
    ap.add_argument("--q", type=float, default=-1.)
    ap.add_argument("--s", type=int, default=0)
    ap.add_argument("--run", type=int, default=None)
    ap.add_argument("--dq", type=float, default=0.)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

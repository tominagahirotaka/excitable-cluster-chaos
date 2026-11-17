#!/usr/bin/env python3
"""Is the slow oscillation of the reduced split system a periodic orbit or a torus? (exploratory, 2026-10-24)

Reduced system as in large_n_reduced_floquet.py (run --index, record --rec, clusters cut at --cut, pair --pair).
No noise, transient 5000, then --length time units; full states at the fast section (cluster 0, x = 0 upward).
  - recurrence: for every lag L up to --maxlag fast returns, the smallest max-norm distance between X_k and X_{k+L}
    over all k; the record minima as L grows (a locked periodic orbit reaches the integration error at its period,
    a two-torus only slowly, roughly like 1/L);
    for each record lag also the largest distance over all k, D_max(L) (a periodic orbit after its transient has
    small D_max at its period; a finite-time computation cannot prove a torus);
  - slow section: downward crossings of the log10 pair distance through its median, with a hysteresis of one decade
    (a crossing counts only after the series has been above median + 1 since the last one); the state at the slow
    section is interpolated linearly in the log10 pair distance between the two fast returns around the crossing;
    for consecutive slow cycles the number of fast returns q_n and the distance between consecutive slow-section
    states;
  - convergence: rerun with another --max_step (DOP853 with max_step 0.05 is limited by the step cap, not by --rtol:
    runs at rtol 1e-10 and 1e-12 differ by only ~1e-5).
  -> analysis/switching_check/slow_return_<index><--tag>.json and .npz (fast-section states and times)

  python3 scripts/slow_return.py --index 90 --rec 1576 --pair 109 22 --length 100000
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import Partition  # noqa: E402
import chaos_check  # noqa: E402
from large_n_reduced_floquet import OPT, reduce  # noqa: E402
from renoise_summary import D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--rec", type=int, required=True)
    ap.add_argument("--cut", type=float, default=1e-3)
    ap.add_argument("--pair", type=int, nargs=2, required=True)
    ap.add_argument("--length", type=float, default=100000.)
    ap.add_argument("--maxlag", type=int, default=4000)
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--max_step", type=float, default=.05)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    s = json.loads((D / f"rn_{a.index:04d}_states.json").read_text())
    e, K = s["eps"], s["K"]
    zs = np.load(D / f"rn_{a.index:04d}_states_z.npy")
    sizes, v = reduce(zs[a.rec], zs.shape[1] // 2, a.cut)
    part = Partition(sizes.tolist())
    m = part.m
    i, j = (int(np.nonzero(sizes == p)[0][0]) for p in a.pair)
    opt = dict(OPT, rtol=a.rtol, atol=a.rtol / 100, max_step=a.max_step)
    v = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 5000), v, **opt).y[:, -1]
    # in chunks of 5000 time units, keeping only the section events (all steps of a long run exhaust the memory)
    Xs, ts, t0 = [], [], 0.
    while t0 < a.length - 1e-9:
        t1 = min(t0 + 5000., a.length)
        sol = solve_ivp(lambda _t, u: part.rhs(u, K, e), (t0, t1), v, events=chaos_check.sec, **opt)
        Xs.append(sol.y_events[0])
        ts.append(sol.t_events[0])
        v, t0 = sol.y[:, -1], t1
    X, t = np.concatenate(Xs), np.concatenate(ts)
    keep = np.r_[True, np.diff(t) > 1.]  # an event exactly at a chunk boundary would appear twice
    X, t = X[keep], t[keep]
    R = len(X)
    rec, best = [], np.inf
    for L in range(1, min(a.maxlag, R - 1) + 1):
        dd = np.max(abs(X[L:] - X[:-L]), axis=1)
        if dd.min() < best:
            best = dd.min()
            rec.append([L, float(dd.min()), float(dd.max()), float(np.median(dd))])
    p = np.maximum(abs(X[:, i] - X[:, j]), abs(X[:, m + i] - X[:, m + j]))
    lg = np.log10(p)
    lev = np.median(lg)
    cross, armed = [], False
    for k in range(1, R):
        if lg[k] > lev + 1:
            armed = True
        if armed and lg[k - 1] >= lev > lg[k]:
            cross.append(k)
            armed = False
    Y, Tc = [], []
    for k in cross:
        w = (lg[k - 1] - lev) / (lg[k - 1] - lg[k])
        Y.append((1 - w) * X[k - 1] + w * X[k])
        Tc.append((1 - w) * t[k - 1] + w * t[k])
    cyc = [dict(start_return=int(k0), q=int(k1 - k0), T=float(Tc[n + 1] - Tc[n]), dist_next=float(np.max(abs(Y[n + 1] - Y[n]))))
           for n, (k0, k1) in enumerate(zip(cross[:-1], cross[1:]))]
    np.savez_compressed(D / f"slow_return_{a.index:04d}{a.tag}.npz", X=X, t=t)
    out = dict(index=a.index, rec=a.rec, sizes=sizes.tolist(), pair=a.pair, rtol=a.rtol, length=a.length, returns=R,
               fast_period=float(np.diff(t).mean()), recurrence_records=rec, slow_cycles=cyc)
    (D / f"slow_return_{a.index:04d}{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(a.index, a.pair, "rtol", a.rtol, "returns", R, "recurrence records (lag, D_min, D_max):",
          [(r[0], f"{r[1]:.1e}", f"{r[2]:.1e}") for r in rec[-8:]])
    print("  slow cycles: q", [c["q"] for c in cyc], "\n  dist to next slow-section state",
          [f"{c['dist_next']:.1e}" for c in cyc])


if __name__ == "__main__":
    main()

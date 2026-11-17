#!/usr/bin/env python3
"""Is each period doubling of the paper-2 cascades longitudinal (inside the reduced m-cluster system) and are the
clusters transversely stable there? (2026-10-26)

For a period-L branch of the reduced system (eps_three_cluster.Partition), continued in eps K by shooting from the
attractor at --q0 (as pd_cascade_check.branch), every point records, with their signs:
  - the collective Floquet multipliers (monodromy of the reduced system, neutral one removed), and the real one
    closest to -1 (mu_neg);
  - the internal (transverse) multipliers: eigenvalues of the 2x2 block of every cluster of size >= 2 (the variation
    of one element against its own cluster, Partition.audit), and the one of largest modulus.
A period doubling is longitudinal when a collective multiplier crosses -1 while no internal multiplier is on or
outside the unit circle; it is a cluster-splitting (transverse) one when an internal multiplier crosses -1.
Crossings of -1 by mu_neg and changes of transverse stability (largest internal modulus through 1) between
neighbouring points are located by linear interpolation in q.
  -> analysis/chaos_check/pdlong_<case>_L<lag><tag>.json

Cases: 95 (selected.json index 95, 28:20:17:12:11:7:5, eps 0.25), 3444 (low_cluster_check/perturb_selected.json
index 133, 34:32:20:13:1, eps 0.35), 143 (reduced_chaos_census_runs.json run 143, 11:24:29:31:5, eps 0.375).

  python3 scripts/pd_longitudinal_check.py --case 95 --lag 2 --q0 -0.958 --qs -0.958:-0.949:0.0005 --tag _up
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import Partition, ROOT, GAMMA  # noqa: E402
import chaos_check  # noqa: E402
from chaos_check import sec  # noqa: E402

OUT = ROOT / "analysis/chaos_check"
TOL = 2e-11


def start(case):
    if case == "143":
        r = json.loads((ROOT / "analysis/reduced_chaos_census_runs.json").read_text())[143]
        c = np.asarray(r["centroids"], float)
        return r["eps"], Partition(c[:, 2].astype(int).tolist()), np.r_[c[:, 0], c[:, 1]]
    chaos_check.SRC = ROOT / ("analysis/many_cluster_check/selected.json" if case == "95"
                              else "analysis/low_cluster_check/perturb_selected.json")
    r, part, v = chaos_check.load(95 if case == "95" else 133)
    return r["eps"], part, v


def multipliers(part, w, K, e):
    m, n = part.m, part.n
    T, z = w[-1], w[:n]

    def f(t, v):
        zz = v[:n]
        A = np.zeros((m, 2, 2))
        A[:, 0, 0] = (1 - zz[:m] ** 2) / e - K
        A[:, 0, 1] = -1 / e
        A[:, 1, 0] = e
        A[:, 1, 1] = -e * GAMMA
        return np.r_[part.rhs(zz, K, e), (part.jac(zz, K, e) @ v[n:n + n * n].reshape(n, n)).ravel(),
                     (A @ v[n + n * n:].reshape(m, 2, 2)).ravel()]

    s = solve_ivp(f, (0, T), np.r_[z, np.eye(n).ravel(), np.tile(np.eye(2), (m, 1, 1)).ravel()],
                  method="DOP853", rtol=max(TOL * .1, 2.3e-14), atol=TOL * .001, max_step=.04)
    assert s.success
    v = s.y[:, -1]
    M, B = v[n:n + n * n].reshape(n, n), v[n + n * n:].reshape(m, 2, 2)
    eig = np.linalg.eigvals(M)
    ce = np.delete(eig, int(np.argmin(abs(eig - 1))))
    real = ce[abs(ce.imag) < 1e-9].real
    neg = real[real < 0]
    be = np.linalg.eigvals(B)
    big = [i for i in range(m) if part.sizes[i] > 1]
    ib = [(int(part.sizes[i]), complex(x)) for i in big for x in be[i]]
    dom = max(ib, key=lambda t: abs(t[1]))
    return dict(K=K, q=K * e, period=float(T), return_error=float(max(abs(v[:n] - z))),
                collective=[[float(c.real), float(c.imag)] for c in ce],
                collective_max=float(max(abs(ce))), mu_neg=float(neg.min()) if neg.size else None,
                internal=[[s_, float(c.real), float(c.imag)] for s_, c in ib],
                internal_max=float(abs(dom[1])), internal_dominant=[dom[0], float(dom[1].real), float(dom[1].imag)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["95", "3444", "143"], required=True)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--q0", type=float, required=True)
    ap.add_argument("--qs", required=True, help="q_start:q_end:dq (dq signed)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    e, part, v = start(a.case)
    K0 = a.q0 / e
    s = solve_ivp(lambda _t, u: part.rhs(u, K0, e), (0, 5000), v, method="DOP853", rtol=1e-11, atol=1e-13,
                  max_step=.05, events=sec)
    ts, ys = s.t_events[0], s.y_events[0]
    w = part.shoot(np.r_[ys[-1 - a.lag], ts[-1] - ts[-1 - a.lag]], K0, e, TOL)
    assert w is not None, "initial shooting failed"
    q0, q1, dq = (float(x) for x in a.qs.split(":"))
    pts = []
    for q in np.arange(q0, q1 + dq / 2, dq):
        nxt = part.shoot(w, q / e, e, TOL)
        if nxt is None:
            print("continuation stopped at", q, flush=True)
            break
        w = nxt
        p = multipliers(part, w, q / e, e)
        p["w"] = w.tolist()
        pts.append(p)
        print(f"{a.case} L={a.lag} q={q:+.5f} T={w[-1]:.6f} mu_neg={p['mu_neg']} coll_max={p['collective_max']:.4f} "
              f"int_dom={p['internal_dominant']} int_max={p['internal_max']:.4f}", flush=True)
    events = []
    for p0, p1 in zip(pts, pts[1:]):
        if p0["mu_neg"] is not None and p1["mu_neg"] is not None and (p0["mu_neg"] + 1) * (p1["mu_neg"] + 1) <= 0:
            t = (p0["mu_neg"] + 1) / (p0["mu_neg"] - p1["mu_neg"])
            events.append(dict(kind="collective -1", q=p0["q"] + t * (p1["q"] - p0["q"]),
                               internal_max=p0["internal_max"] + t * (p1["internal_max"] - p0["internal_max"])))
        if (p0["internal_max"] - 1) * (p1["internal_max"] - 1) <= 0:
            t = (p0["internal_max"] - 1) / (p0["internal_max"] - p1["internal_max"])
            events.append(dict(kind="internal |mu| = 1", q=p0["q"] + t * (p1["q"] - p0["q"]),
                               dominant_before=p0["internal_dominant"], dominant_after=p1["internal_dominant"]))
    for ev in events:
        print(ev, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pdlong_{a.case}_L{a.lag}{a.tag}.json").write_text(json.dumps(
        dict(case=a.case, lag=a.lag, q0=a.q0, qs=a.qs, eps=e, sizes=part.sizes.tolist(), points=pts, events=events),
        indent=1))


if __name__ == "__main__":
    main()

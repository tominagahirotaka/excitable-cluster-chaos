#!/usr/bin/env python3
"""Period-doubling cascade of the seven-cluster partition 28:20:17:12:11:7:5 (eps = 0.25) and the chaotic state in
the full system (exploratory, 2026-10-23).

Partition and starting point: cluster centroids of the saved final state of analysis/many_cluster_check/selected.json
index 95 (N = 100, eps = 0.25, eps K = -1.0, seed 5042). See docs/RESULTS_NOTES.md (2026-10-22, chaos).

  branch : the period-L orbit (L = --lag returns of the upward section x_1 = 0) is taken from the attractor of the
           reduced system at eps K = --q0 (transient 5000), shot (eps_three_cluster.Partition.shoot, tolerance 2e-11)
           and continued in eps K over --qs (natural continuation, previous orbit as guess). At each point:
           collective Floquet multipliers (monodromy from Partition.flow, neutral one removed), the real multiplier
           closest to -1, and the audit (validity, internal multipliers). Crossings of -1 are refined by the secant
           method on mu + 1 (|dq| < 1e-6).
           The largest positive real multiplier (mu_pos) is also recorded (fold: mu_pos -> +1).
           -> analysis/chaos_check/pd_L<lag><tag>.json
  full   : the full N-element system (NumPy RK4, h = 0.005) from the saved final state of index 95 for --tmax:
           run A unperturbed, run B with independent Gaussian noise of sd 1e-8 on all 2N coordinates. Every time
           unit: the largest within-cluster spread of A and of B (max-norm), and the largest difference between
           the cluster centroids of A and B. Upward x = 0 crossings of every element of A (firing times).
           -> analysis/chaos_check/full_95.json

  --src and --index select another saved state (as scripts/chaos_check.py); output names then carry the prefix "low"
  (e.g. pd_low133_L1.json, full_low133.json).

  python3 scripts/pd_cascade_check.py branch --lag 1 --q0 -0.93 --qs -0.93:-0.96:0.0025     (remote host: scipy)
  python3 scripts/pd_cascade_check.py full --tmax 3000
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
from chaos_check import load, sec  # noqa: E402

OUT = ROOT / "analysis/chaos_check"
TOL = 2e-11
B_, G_ = .7, .5


def point(part, w, K, e):
    r = part.flow(w, K, e, TOL)
    M = r[1][:part.n, :part.n] + np.eye(part.n)
    eig = np.linalg.eigvals(M)
    ce = np.delete(eig, int(np.argmin(abs(eig - 1))))
    real = ce[abs(ce.imag) < 1e-9].real
    neg = real[real < 0]
    pos = real[real > 0]
    au = part.audit(w, K, e, TOL)
    return dict(K=K, q=K * e, period=float(w[-1]), mu_neg=float(neg.min()) if neg.size else None,
                mu_pos=float(pos.max()) if pos.size else None,
                collective=[[float(c.real), float(c.imag)] for c in ce], collective_max=au["collective_max"],
                internal_max=float(max(m for m, s in zip(au["internal_max"], part.sizes) if s > 1)),
                valid=au["valid"])


def branch(a):
    r, part, v = load(a.index)
    e = r["eps"]
    K0 = a.q0 / e
    s = solve_ivp(lambda _t, u: part.rhs(u, K0, e), (0, 5000), v, method="DOP853", rtol=1e-11, atol=1e-13,
                  max_step=.05, events=sec)
    ts, ys = s.t_events[0], s.y_events[0]
    w = part.shoot(np.r_[ys[-1 - a.lag], ts[-1] - ts[-1 - a.lag]], K0, e, TOL)
    assert w is not None, "initial shooting failed"
    q0, q1, dq = (float(x) for x in a.qs.split(":"))
    qs = np.arange(q0, q1 + dq / 2, dq)
    pts, ws = [], []
    for q in qs:
        nxt = part.shoot(w, q / e, e, TOL)
        if nxt is None:
            print("continuation stopped at", q, flush=True)
            break
        w = nxt
        pts.append(point(part, w, q / e, e))
        ws.append(w)
        print(f"L={a.lag} q={q:+.5f} T={w[-1]:.6f} mu_neg={pts[-1]['mu_neg']} mu_pos={pts[-1]['mu_pos']} "
              f"coll_max={pts[-1]['collective_max']:.4f} int_max={pts[-1]['internal_max']:.4f} "
              f"valid={pts[-1]['valid']}", flush=True)
    cross = []
    for i in range(1, len(pts)):
        f0, f1 = (pts[i - 1]["mu_neg"], pts[i]["mu_neg"])
        if f0 is None or f1 is None or (f0 + 1) * (f1 + 1) > 0:
            continue
        qa, qb, fa, fb, wa = pts[i - 1]["q"], pts[i]["q"], f0 + 1, f1 + 1, ws[i - 1]
        for _ in range(30):
            qc = qb - fb * (qb - qa) / (fb - fa)
            wc = part.shoot(wa, qc / e, e, TOL)
            if wc is None:
                break
            pc = point(part, wc, qc / e, e)
            fc = pc["mu_neg"] + 1
            qa, fa, qb, fb, wa = qb, fb, qc, fc, wc
            if abs(qb - qa) < 1e-6:
                break
        cross.append(dict(q=qb, mu_neg=fb - 1, period=float(wa[-1]), internal_max=pc["internal_max"]))
        print(f"L={a.lag} crossing of -1 at q={qb:.6f} (T={wa[-1]:.6f}, internal_max={pc['internal_max']:.4f})",
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pd_{name(a)}L{a.lag}{a.tag}.json").write_text(json.dumps(dict(lag=a.lag, q0=a.q0, eps=e, sizes=part.sizes.tolist(),
                                                            points=pts, crossings=cross), indent=1))


def full(a):
    r = json.loads(chaos_check.SRC.read_text())[a.index]
    e, K = r["eps"], r["K"]
    zA = np.asarray(r["final_state"], float)
    n = zA.size // 2
    _, part, _ = load(a.index)
    from scipy.sparse.csgraph import connected_components
    d = np.maximum(abs(zA[:n, None] - zA[None, :n]), abs(zA[n:, None] - zA[None, n:]))
    _, lab = connected_components(d < 1e-4, directed=False)
    zB = zA + np.random.default_rng(1).normal(0, 1e-8, zA.size)

    def f(v):
        x, y = v[:n], v[n:]
        return np.r_[(x - x ** 3 / 3 - y) / e + K * (x.mean() - x), e * (x + B_ - G_ * y)]

    def rk(z, h):
        k1 = f(z)
        k2 = f(z + h / 2 * k1)
        k3 = f(z + h / 2 * k2)
        k4 = f(z + h * k3)
        return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def spread(z):
        return max(max(np.ptp(z[:n][lab == c]), np.ptp(z[n:][lab == c])) for c in range(lab.max() + 1))

    def cent(z):
        return np.array([[z[:n][lab == c].mean(), z[n:][lab == c].mean()] for c in range(lab.max() + 1)])
    h, steps_per = .005, 200
    rows, fire = [], [[] for _ in range(n)]
    xprev = zA[:n].copy()
    t = 0.
    for k in range(int(a.tmax)):
        for _ in range(steps_per):
            zA, zB = rk(zA, h), rk(zB, h)
            t += h
            up = np.flatnonzero((xprev < 0) & (zA[:n] >= 0))
            for i in up:
                fire[i].append(t - h * zA[i] / (zA[i] - xprev[i]))
            xprev = zA[:n].copy()
        rows.append(dict(t=round(t, 6), spread_A=float(spread(zA)), spread_B=float(spread(zB)),
                         centroid_diff=float(np.max(abs(cent(zA) - cent(zB))))))
    res = dict(index=a.index, eps=e, K=K, labels=lab.tolist(), sizes=np.bincount(lab).tolist(), rows=rows, fire=fire)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"full_{name(a).rstrip('_') or a.index}.json").write_text(json.dumps(res))
    print("full done", rows[-1], flush=True)


def name(a):
    return f"low{a.index}_" if a.src and "low_cluster" in a.src else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["branch", "full"])
    ap.add_argument("--src", default=None, help="JSON list with final_state entries (default: many_cluster_check)")
    ap.add_argument("--index", type=int, default=95)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--q0", type=float, default=-.93)
    ap.add_argument("--qs", default="-0.93:-0.96:-0.0025")
    ap.add_argument("--tmax", type=float, default=3000.)
    ap.add_argument("--tag", default="", help="suffix of the branch output name")
    a = ap.parse_args()
    if a.src:
        chaos_check.SRC = ROOT / a.src
    {"branch": branch, "full": full}[a.mode](a)


if __name__ == "__main__":
    main()

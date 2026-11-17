#!/usr/bin/env python3
"""Where and how does the period-5 orbit of run 3444 end near eps K = -1.2128? (2026-10-27)

The fine walk (chaos_edge_check walk, edge_3444_fine_fwd) is period 5 at -1.21278..-1.21282 and period 1 from
-1.21284. This script continues the period-5 orbit of the reduced system (eps_three_cluster.Partition) by shooting:
  - start: the final state of the walk point --q0 (edge_3444_fine_fwd.json), a short transient, then the time of 5
    section returns (x_0 = 0 upward) as the period guess, refined by Partition.shoot at fixed K;
  - natural steps in q of --dq0 to a second point, then pseudo-arclength steps of size ds in (z, T, K) as in
    eps_three_cluster.arc, so that a turning point (fold) is passed; ds is halved on a Newton failure and grows
    slowly after easy steps (between --ds_min and --ds_max);
  - at every point: collective Floquet multipliers of the reduced system with the trivial one removed by its
    eigenvector (the one most parallel to f(z), not the one closest to +1, since near a fold two are close to +1),
    internal (transverse) multipliers of every cluster of size >= 2 (2x2 blocks, as pd_longitudinal_check), the
    period, and the minimal-period check: distances |X_j - X_0| of the section points j = 1..4 along the orbit
    (all must be far from 0 for a genuine period 5);
  - events between neighbouring points: turning point of K (fold), a real collective multiplier through +1 or -1,
    a complex pair through modulus 1, the largest internal modulus through 1.
--dir -1 continues towards more negative eps K (the exit), +1 back towards the period-doubling at about -1.21277
(a check of the method: a collective multiplier should cross -1 there).
  -> analysis/chaos_check/p5arc_3444<tag>.json (rewritten after every point)

  python3 scripts/p5_arc_3444.py --q0=-1.21278 --dir -1 --tag _exit
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import GAMMA  # noqa: E402
from chaos_check import sec  # noqa: E402
from pd_longitudinal_check import start, OUT  # noqa: E402

TOL = 2e-12


def evaluate(part, e, Y):
    """Residual and Jacobian in (z, T, K) of the periodic-orbit problem with the phase condition z_0 = 0."""
    n = part.n
    z, T, K = Y[:n], Y[n], Y[-1]

    def f(t, v):
        zz = v[:n]
        J = part.jac(zz, K, e)
        g = np.r_[part.p @ zz[:part.m] - zz[:part.m], np.zeros(part.m)]  # d rhs / d K
        return np.r_[part.rhs(zz, K, e), (J @ v[n:n + n * n].reshape(n, n)).ravel(), J @ v[-n:] + g]

    s = solve_ivp(f, (0, T), np.r_[z, np.eye(n).ravel(), np.zeros(n)], method="DOP853", rtol=TOL, atol=TOL * .01,
                  max_step=.05)
    if not s.success:
        return None
    v = s.y[:, -1]
    A = np.zeros((n + 1, n + 2))
    A[:n, :n] = v[n:n + n * n].reshape(n, n) - np.eye(n)
    A[:n, n] = part.rhs(v[:n], K, e)
    A[:n, -1] = v[-n:]
    A[n, 0] = 1
    return np.r_[v[:n] - z, Y[0]], A


def correct(part, e, Y, pred=None, tangent=None):
    """Newton on the periodic-orbit problem; at fixed K when tangent is None, else with the arclength condition."""
    n = part.n
    for it in range(12):
        r = evaluate(part, e, Y)
        if r is None:
            return None, it
        res, A = r
        if tangent is None:
            res, A = res, A[:, :n + 1]
        else:
            res, A = np.r_[res, (Y - pred) @ tangent], np.vstack([A, tangent])
        if max(abs(res)) < 5e-10:
            return Y, it
        d = np.linalg.solve(A, -res)
        Y = Y + (np.r_[d, 0] if tangent is None else d)
    return None, it


def audit(part, e, Y):
    m, n = part.m, part.n
    z, T, K = Y[:n], Y[n], Y[-1]

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
                  method="DOP853", rtol=TOL * .1, atol=TOL * .001, max_step=.04, events=sec)
    assert s.success
    v = s.y[:, -1]
    M, B = v[n:n + n * n].reshape(n, n), v[n + n * n:].reshape(m, 2, 2)
    eig, V = np.linalg.eig(M)
    fz = part.rhs(z, K, e)
    par = abs(V.conj().T @ fz) / (np.linalg.norm(V, axis=0) * np.linalg.norm(fz))
    triv = int(np.argmax(par))
    ce = np.delete(eig, triv)
    ce = ce[np.argsort(-abs(ce))]
    be = np.linalg.eigvals(B)
    ib = [(int(part.sizes[i]), complex(x)) for i in range(m) if part.sizes[i] > 1 for x in be[i]]
    dom = max(ib, key=lambda t: abs(t[1]))
    X = s.y_events[0][:, :n]
    secd = [float(np.max(abs(x - z))) for x in X if np.max(abs(x - z)) > 1e-6]  # section points other than z itself
    return dict(q=float(K * e), K=float(K), period=float(T), trivial=[float(eig[triv].real), float(eig[triv].imag)],
                trivial_parallel=float(par[triv]), collective=[[float(c.real), float(c.imag)] for c in ce],
                collective_max=float(abs(ce[0])), internal=[[s_, float(c.real), float(c.imag)] for s_, c in ib],
                internal_max=float(abs(dom[1])), internal_dominant=[dom[0], float(dom[1].real), float(dom[1].imag)],
                section_returns=len(X), section_dist=secd, z=z.tolist())


def events(p0, p1):
    out = []
    q0, q1 = p0["q"], p1["q"]
    r0 = [c[0] for c in p0["collective"] if abs(c[1]) < 1e-9]
    r1 = [c[0] for c in p1["collective"] if abs(c[1]) < 1e-9]
    for sgn in (1, -1):
        a = [x for x in r0 if x * sgn > 0]
        b = [x for x in r1 if x * sgn > 0]
        if a and b and (max(abs(x) for x in a) - 1) * (max(abs(x) for x in b) - 1) <= 0:
            out.append(dict(kind=f"real multiplier through {sgn:+d}", between=[q0, q1]))
    c0 = [np.hypot(*c) for c in p0["collective"] if abs(c[1]) >= 1e-9]
    c1 = [np.hypot(*c) for c in p1["collective"] if abs(c[1]) >= 1e-9]
    if c0 and c1 and (max(c0) - 1) * (max(c1) - 1) <= 0:
        out.append(dict(kind="complex pair through modulus 1", between=[q0, q1]))
    if (p0["internal_max"] - 1) * (p1["internal_max"] - 1) <= 0:
        out.append(dict(kind="internal |mu| = 1", between=[q0, q1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q0", type=float, default=-1.21278)
    ap.add_argument("--dir", type=int, choices=[-1, 1], default=-1)
    ap.add_argument("--dq0", type=float, default=2e-6)
    ap.add_argument("--ds", type=float, default=2e-5)
    ap.add_argument("--ds_min", type=float, default=1e-8)
    ap.add_argument("--ds_max", type=float, default=2e-4)
    ap.add_argument("--q_lo", type=float, default=-1.2132)
    ap.add_argument("--q_hi", type=float, default=-1.2126)
    ap.add_argument("--max_points", type=int, default=400)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    e, part, _ = start("3444")
    n = part.n
    walk = json.loads((OUT / "edge_3444_fine_fwd.json").read_text())["points"]
    p = min(walk, key=lambda r: abs(r["q"] - a.q0))
    K0 = p["q"] / e
    s = solve_ivp(lambda _t, u: part.rhs(u, K0, e), (0, 1000), np.asarray(p["state"]), method="DOP853",
                  rtol=1e-11, atol=1e-13, max_step=.05, events=sec)
    ts, ys = s.t_events[0], s.y_events[0]
    w = part.shoot(np.r_[ys[-6], ts[-1] - ts[-6]], K0, e, TOL)
    assert w is not None, "initial shooting failed"
    Y0, it = correct(part, e, np.r_[w, K0])
    Y1, it = correct(part, e, np.r_[w, (p["q"] + a.dir * a.dq0) / e])
    assert Y0 is not None and Y1 is not None
    rows = [audit(part, e, Y0), audit(part, e, Y1)]
    for r in rows:
        print(f"start q={r['q']:+.7f} T={r['period']:.6f} coll={r['collective'][:3]} int_max={r['internal_max']:.4f} "
              f"returns={r['section_returns']} secd_min={min(r['section_dist']):.3f}", flush=True)
    ds, evs = a.ds, []
    fname = OUT / f"p5arc_3444{a.tag}.json"
    while len(rows) < a.max_points:
        tangent = (Y1 - Y0) / np.linalg.norm(Y1 - Y0)
        pred = Y1 + ds * tangent
        Y, it = correct(part, e, pred.copy(), pred, tangent)
        if Y is None:
            ds /= 2
            print(f"  newton failure, ds -> {ds:.2e}", flush=True)
            if ds < a.ds_min:
                rows.append(dict(status="arc_newton_failure", after_q=float(Y1[-1] * e)))
                break
            continue
        r = audit(part, e, Y)
        r["ds"] = ds
        r["dK_sign"] = int(np.sign(Y[-1] - Y1[-1]))
        new = events(rows[-1], r)
        if rows[-1].get("dK_sign") is not None and r["dK_sign"] != rows[-1]["dK_sign"]:
            new.append(dict(kind="turning point of K (fold)", between=[rows[-1]["q"], r["q"]]))
        evs += new
        rows.append(r)
        print(f"q={r['q']:+.8f} T={r['period']:.6f} dK={r['dK_sign']:+d} ds={ds:.1e} "
              f"coll={[(round(c[0], 5), round(c[1], 5)) for c in r['collective'][:3]]} "
              f"int_max={r['internal_max']:.4f} triv={r['trivial'][0]:.6f} returns={r['section_returns']} "
              f"secd_min={min(r['section_dist']):.3f}", flush=True)
        for ev in new:
            print("   EVENT", ev, flush=True)
        fname.write_text(json.dumps(dict(case="3444", q0=a.q0, dir=a.dir, eps=e, sizes=part.sizes.tolist(),
                                         points=rows, events=evs), indent=1))
        if not a.q_lo <= r["q"] <= a.q_hi:
            break
        Y0, Y1 = Y1, Y
        if it <= 3:
            ds = min(ds * 1.5, a.ds_max)
    for ev in evs:
        print(ev, flush=True)
    fname.write_text(json.dumps(dict(case="3444", q0=a.q0, dir=a.dir, eps=e, sizes=part.sizes.tolist(),
                                     points=rows, events=evs), indent=1))


if __name__ == "__main__":
    main()

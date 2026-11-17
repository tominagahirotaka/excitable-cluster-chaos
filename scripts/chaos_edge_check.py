#!/usr/bin/env python3
"""How does the chaotic range of the paper-2 examples end: by a change inside the reduced m-cluster system
(crisis, window), or by transverse instability of the clusters (blowout)? (2026-10-27)

walk : adiabatic continuation of the attractor of the reduced system (eps_three_cluster.Partition) in eps K over
       --qs (q_start:q_end:dq, dq signed); the final state at one q is the start at the next. The first point starts
       from the saved state of --case (pd_longitudinal_check.start) after a transient of 5000. At every q: transient
       --transient, then --length time units with
         - the largest longitudinal exponent lam (one tangent vector of the reduced system, renormalized every DT),
         - the transverse exponent of every cluster (its internal 2x2 block, QR every DT), lam_perp = the largest
           over clusters of size >= 2, its estimates on --nblocks (default 4) equal blocks of the measurement (spread),
         - finite-time transverse exponents of the largest over windows of --window time units: their maximum and
           the fraction of windows above 0 (bubbling indicator; a negative lam_perp with positive windows means
           transversely unstable pieces of the attractor),
         - upward x_1 = 0 returns: return times, and the smallest lag <= 60 with return distance < 1e-6 over the last
           50 returns (None = not periodic).
       -> analysis/chaos_check/edge_<case><tag>.json (rewritten after every q; carries the final state of each q)
       With --from and --qi the first point starts instead from the final state of that point of an earlier walk
       (e.g. a periodic orbit reached by another walk, to follow it back). With --save_all every section point of the
       measurement is kept (section_all), to check a periodicity over separate parts of the measurement.
lifetime : chaotic transients past the end of the range. Initial conditions: the last 100 section points saved by an
       earlier walk at --qi (consecutive returns of one chaotic orbit, so not independent samples). Each is integrated
       at --q (reduced system, no tangent vectors) for at most --tmax, in chunks of 500, until 20 consecutive returns
       differ from the previous one by < 1e-7 (max norm). Per initial condition: all return times and the lag-1 return
       distances (the settling time is defined afterwards from these), whether it settled, the mean of the last 10
       return intervals (identifies the final orbit, e.g. A or B of run 143) and the final section point.
       -> analysis/chaos_check/life_<case>_q<q><tag>.json
laminar : one long orbit at --q from the final state of the walk point at --q in --from: transient --transient, then
       --tmax time units; every return time and section point (laminar phases near a period-3 orbit are counted
       afterwards from |X_k - X_(k-3)|).
       -> analysis/chaos_check/lam_<case>_q<q><tag>.npz
full : the full N-element system (NumPy RK4, h = 0.005; --nmul k: every population times k, the same fractions at k N,
       added 2026-11-03) from a reduced state saved by walk (--from, --qi), every
       cluster spread by independent Gaussian offsets of sd --d0 per element, optionally kicked by Gaussian noise of
       sd --noise on all 2N coordinates every time unit; every time unit the largest within-cluster spread (max - min of
       x or y; spread_y: of y only, which depends little on the firing phase since y is slow). With --phase_sec also, at
       every upward crossing of the mean x of cluster 0 through 0 (the section of the reduced system; state linearly
       interpolated within the RK step), the rms distance of the elements of every cluster of size >= 2 from its mean,
       in x and in y, and the cluster means (x_1..x_m, y_1..y_m, the state of the reduced system, to measure the
       distance to its periodic orbits) (2026-10-27: the per-time-unit spread mixes transverse growth with the firing
       phase).
       -> analysis/chaos_check/edgefull_<case>_q<q><tag>.json

  python3 scripts/chaos_edge_check.py walk --case 95 --qs=-0.999:-1.006:-0.0002 --tag _end_fwd
  python3 scripts/chaos_edge_check.py full --case 95 --from edge_95_end_fwd.json --q=-1.0030 --tmax 20000
  python3 scripts/chaos_edge_check.py walk --case 143 --from edge_143_end_fwd.json --qi=-0.812 --qs=-0.812:-0.800:0.0002
  python3 scripts/chaos_edge_check.py lifetime --case 143 --from edge_143_end_fwd.json --qi=-0.8084 --q=-0.8087
  python3 scripts/chaos_edge_check.py laminar --case 95 --from edge_95_end_fwd.json --q=-1.0036 --tmax 400000
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import ROOT, GAMMA, BETA  # noqa: E402
from pd_longitudinal_check import start  # noqa: E402
from chaos_check import sec  # noqa: E402

OUT = ROOT / "analysis/chaos_check"
DT = 5.


def lag_of(X):
    for L in range(1, 61):
        if len(X) > L + 50 and max(np.max(abs(X[k] - X[k - L])) for k in range(len(X) - 50, len(X))) < 1e-6:
            return L
    return None


def measure(part, v, K, e, a, opt):
    n, m = part.n, part.m
    big = part.sizes > 1

    def f(_t, u):
        z = u[:n]
        A = np.zeros((m, 2, 2))
        A[:, 0, 0] = (1 - z[:m] ** 2) / e - K
        A[:, 0, 1] = -1 / e
        A[:, 1, 0] = e
        A[:, 1, 1] = -e * GAMMA
        return np.r_[part.rhs(z, K, e), part.jac(z, K, e) @ u[n:2 * n],
                     np.einsum("aij,ajk->aik", A, u[2 * n:].reshape(m, 2, 2)).ravel()]
    v = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, a.transient), v, **opt).y[:, -1]
    g = np.random.default_rng(0).standard_normal(n)
    d = g / np.linalg.norm(g)
    W = np.tile(np.eye(2), (m, 1, 1))
    steps = int(round(a.length / DT))
    per_win = int(round(a.window / DT))
    lam = np.zeros(steps)
    perp = np.zeros((steps, m))
    tr, X = [], []
    t = 0.
    for k in range(steps):
        s = solve_ivp(f, (t, t + DT), np.r_[v, d, W.ravel()], events=sec, **opt)
        u = s.y[:, -1]
        v = u[:n]
        nd = np.linalg.norm(u[n:2 * n])
        lam[k] = np.log(nd)
        d = u[n:2 * n] / nd
        W = u[2 * n:].reshape(m, 2, 2).copy()
        for c in range(m):
            q, r = np.linalg.qr(W[c])
            perp[k, c] = np.log(abs(r[0, 0]))
            W[c] = q
        tr += s.t_events[0].tolist()
        X += [y[:n] for y in s.y_events[0]]
        t += DT
    pc = perp.sum(0) / a.length
    cb = int(np.flatnonzero(big)[np.argmax(pc[big])])
    nb = a.nblocks
    blk = steps // nb
    nw = steps // per_win
    win = perp[:nw * per_win, cb].reshape(nw, per_win).sum(1) / a.window
    X = np.asarray(X)
    return v, dict(lam=float(lam.sum() / a.length),
                   lam_blocks=[float(lam[i * blk:(i + 1) * blk].sum() / (blk * DT)) for i in range(nb)],
                   perp=pc.tolist(), perp_max=float(pc[cb]), perp_cluster=cb, perp_size=int(part.sizes[cb]),
                   perp_blocks=[float(perp[i * blk:(i + 1) * blk, cb].sum() / (blk * DT)) for i in range(nb)],
                   win_max=float(win.max()), win_pos_frac=float((win > 0).mean()),
                   lag=lag_of(X), returns=len(tr), return_times=np.diff(tr).tolist(),
                   section_last=X[-100:].tolist(), section_all=X.tolist() if a.save_all else None)


def point(a, q):
    src = json.loads((OUT / a.src).read_text())
    return min(src["points"], key=lambda p: abs(p["q"] - q))


def walk(a):
    e, part, v = start(a.case)
    if a.src:
        v = np.asarray(point(a, a.qi)["state"])
    opt = dict(method="DOP853", rtol=a.rtol, atol=a.rtol / 100, max_step=.05)
    q0, q1, dq = (float(x) for x in a.qs.split(":"))
    v = solve_ivp(lambda _t, u: part.rhs(u, q0 / e, e), (0, 5000), v, **opt).y[:, -1]
    out = OUT / f"edge_{a.case}{a.tag}.json"
    pts = []
    for q in np.arange(q0, q1 + dq / 2, dq):
        v, p = measure(part, v, q / e, e, a, opt)
        p.update(q=float(q), state=v.tolist())
        pts.append(p)
        print(f"{a.case} q={q:+.5f} lam={p['lam']:+.5f} perp_max={p['perp_max']:+.5f} "
              f"blocks={np.round(p['perp_blocks'], 5).tolist()} win_max={p['win_max']:+.4f} "
              f"pos={p['win_pos_frac']:.3f} lag={p['lag']} T={np.mean(p['return_times']):.4f}", flush=True)
        out.write_text(json.dumps(dict(case=a.case, eps=e, sizes=part.sizes.tolist(), qs=a.qs, rtol=a.rtol,
                                       transient=a.transient, length=a.length, window=a.window, points=pts)))


def lifetime(a):
    e, part, _ = start(a.case)
    K = a.q / e
    opt = dict(method="DOP853", rtol=a.rtol, atol=a.rtol / 100, max_step=.05)
    f = lambda _t, u: part.rhs(u, K, e)
    p = point(a, a.qi)
    out = OUT / f"life_{a.case}_q{a.q:+.5f}{a.tag}.json"
    res = []
    for i, x0 in enumerate(p["section_last"]):
        v, t, tr, X = np.asarray(x0), 0., [], []
        settled = False
        while t < a.tmax and not settled:
            s = solve_ivp(f, (t, t + 500), v, events=sec, **opt)
            v, t = s.y[:, -1], t + 500
            tr += s.t_events[0].tolist()
            X += [y for y in s.y_events[0]]
            d = [float(np.max(abs(X[k] - X[k - 1]))) for k in range(1, len(X))]
            settled = len(d) >= 20 and max(d[-20:]) < 1e-7
        res.append(dict(settled=settled, t_end=t, return_times=tr, dist1=d,
                        T_final=float(np.mean(np.diff(tr[-11:]))), x_final=X[-1].tolist()))
        print(f"{a.case} q={a.q:+.5f} ic={i} settled={settled} t_end={t:.0f} returns={len(tr)} "
              f"T_final={res[-1]['T_final']:.4f}", flush=True)
        out.write_text(json.dumps(dict(case=a.case, eps=e, q=a.q, qi=p["q"], src=a.src, tmax=a.tmax, rtol=a.rtol,
                                       runs=res)))


def laminar(a):
    e, part, _ = start(a.case)
    K = a.q / e
    p = point(a, a.q)
    opt = dict(method="DOP853", rtol=a.rtol, atol=a.rtol / 100, max_step=.05)
    f = lambda _t, u: part.rhs(u, K, e)
    v = solve_ivp(f, (0, a.transient), np.asarray(p["state"]), **opt).y[:, -1]
    tr, X, t = [], [], 0.
    while t < a.tmax:
        s = solve_ivp(f, (t, t + 1000), v, events=sec, **opt)
        v, t = s.y[:, -1], t + 1000
        tr += s.t_events[0].tolist()
        X += [y for y in s.y_events[0]]
    np.savez_compressed(OUT / f"lam_{a.case}_q{p['q']:+.5f}{a.tag}.npz", t=np.asarray(tr), X=np.asarray(X), q=p["q"],
                        eps=e, transient=a.transient, tmax=a.tmax, rtol=a.rtol)
    print(f"{a.case} q={p['q']:+.5f} returns={len(tr)}", flush=True)


def full(a):
    src = json.loads((OUT / a.src).read_text())
    e, sizes = src["eps"], np.asarray(src["sizes"]) * a.nmul  # --nmul: same fractions, N times nmul (2026-11-03)
    p = point(a, a.q)
    q, K, w = p["q"], p["q"] / e, np.asarray(p["state"])
    m = sizes.size
    lab = np.repeat(np.arange(m), sizes)
    n = lab.size
    rng = np.random.default_rng(a.seed)
    z = np.r_[w[:m][lab], w[m:][lab]] + rng.normal(0, a.d0, 2 * n) * np.r_[sizes[lab] > 1, sizes[lab] > 1]

    def f(v):
        x, y = v[:n], v[n:]
        return np.r_[(x - x ** 3 / 3 - y) / e + K * (x.mean() - x), e * (x + BETA - GAMMA * y)]

    def rk(z, h):
        k1 = f(z)
        k2 = f(z + h / 2 * k1)
        k3 = f(z + h / 2 * k2)
        k4 = f(z + h * k3)
        return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    h, per = .005, 200
    big = [c for c in range(m) if sizes[c] > 1]
    c0 = lab == 0

    def rms(v):
        """per cluster of size >= 2: rms distance of its elements from the cluster mean, in x and in y."""
        return [[float(np.sqrt(np.mean((v[:n][lab == c] - v[:n][lab == c].mean()) ** 2))),
                 float(np.sqrt(np.mean((v[n:][lab == c] - v[n:][lab == c].mean()) ** 2)))] for c in big]
    rows, rows_y, sec_t, sec_rms, sec_mean = [], [], [], [], []
    for k in range(int(a.tmax)):
        for i in range(per):
            x0 = z[:n][c0].mean()
            zn = rk(z, h)
            x1 = zn[:n][c0].mean()
            if a.phase_sec and x0 < 0 <= x1:  # upward crossing of the mean x of cluster 0 (the reduced section)
                s_ = x0 / (x0 - x1)
                sec_t.append(k + (i + s_) * h)
                zs = z + s_ * (zn - z)
                sec_rms.append(rms(zs))
                sec_mean.append(np.r_[[zs[:n][lab == c].mean() for c in range(m)],
                                      [zs[n:][lab == c].mean() for c in range(m)]].tolist())
            z = zn
        if a.noise > 0:
            z = z + rng.normal(0, a.noise, 2 * n)
        rows.append(max(max(np.ptp(z[:n][lab == c]), np.ptp(z[n:][lab == c])) for c in big))
        rows_y.append(max(np.ptp(z[n:][lab == c]) for c in big))
    rows = np.asarray(rows)
    res = dict(case=a.case, q=q, eps=e, sizes=sizes.tolist(), nmul=a.nmul, N=int(n), d0=a.d0, noise=a.noise, seed=a.seed, tmax=a.tmax,
               spread=rows.tolist(), spread_y=rows_y, final_state=z.tolist())
    if a.phase_sec:
        res.update(sec_clusters=big, sec_t=sec_t, sec_rms=sec_rms, sec_mean=sec_mean)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"edgefull_{a.case}_q{q:+.4f}{a.tag}.json").write_text(json.dumps(res))
    print(a.case, f"q={q:+.4f}", "spread start/max/end", f"{rows[0]:.2e} {rows.max():.2e} {rows[-1]:.2e}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["walk", "full", "lifetime", "laminar"])
    ap.add_argument("--case", choices=["95", "3444", "143"], required=True)
    ap.add_argument("--qs", help="walk: q_start:q_end:dq (dq signed)")
    ap.add_argument("--transient", type=float, default=2000.)
    ap.add_argument("--length", type=float, default=20000.)
    ap.add_argument("--window", type=float, default=100.)
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--from", dest="src", help="walk output file name in analysis/chaos_check")
    ap.add_argument("--q", type=float, help="full, laminar: q of the walk point to start from; lifetime: target q")
    ap.add_argument("--qi", type=float, help="walk: q of the --from point to start at; lifetime: q of the source chaos")
    ap.add_argument("--d0", type=float, default=1e-6)
    ap.add_argument("--noise", type=float, default=0.)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tmax", type=float, default=20000.)
    ap.add_argument("--tag", default="")
    ap.add_argument("--nmul", type=int, default=1, help="full: multiply every population by this (same fractions, larger N)")
    ap.add_argument("--phase_sec", action="store_true", help="full: also record cluster rms deviations at the section")
    ap.add_argument("--nblocks", type=int, default=4, help="walk: number of equal blocks for lam_blocks, perp_blocks")
    ap.add_argument("--save_all", action="store_true", help="walk: keep every section point of the measurement")
    a = ap.parse_args()
    dict(walk=walk, full=full, lifetime=lifetime, laminar=laminar)[a.mode](a)


if __name__ == "__main__":
    main()

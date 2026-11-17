#!/usr/bin/env python3
"""Robustness of representative reduced-system results against the integrator: Radau (implicit, order 5, analytic
Jacobian) vs the saved DOP853 results (2026-10-23).

Periodic orbits: Newton shooting on the Radau flow map (rtol 1e-12, atol 1e-14); the monodromy matrix is obtained by
central differences of the Radau flow (step 1e-6), not from variational equations; the internal 2x2 blocks are
integrated with Radau along the converged orbit. Initial guesses come from a DOP853 trajectory (rtol 1e-8), refined
by DOP853 shooting (continued in eps K from the guess parameter) for the pd and window cases; the converged orbit
depends only on the Radau flow. Newton steps are backtracked (as in Partition.shoot).
  orbit --case k : entry k of CASES
      pd     : saved period-doubling crossing (analysis/chaos_check/pd_*.json); the multiplier closest to -1 at the
               saved crossing, and the implied shift of the crossing from the slope of mu_neg on the saved branch
      low    : saved stable or unstable periodic orbits of analysis/low_cluster_check (period, collective_max,
               internal_max_big)
      window : eps K = -1.206 of the run-3444 partition (period-3 orbit)
Chaotic states: chaos_check.scan with Radau (rtol --rtol, atol rtol/100): transient 5000 from the centroids, two
trajectories over 15000 time units with the internal exponents and section returns.
  chaos --case k
  -> analysis/radau_check/<name>.json

  python3 scripts/radau_check.py list
  python3 scripts/radau_check.py orbit --case 0          (remote host)
"""
from pathlib import Path
import argparse
import glob
import json
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import Partition, ROOT, GAMMA  # noqa: E402
import chaos_check  # noqa: E402

OUT = ROOT / "analysis/radau_check"
SRC95 = "analysis/many_cluster_check/selected.json"
SRC133 = "analysis/low_cluster_check/perturb_selected.json"
RT, AT, DELTA = 1e-12, 1e-14, 1e-6
ERRS = []  # Newton residuals of the last shoot call


def cases():
    c = []
    for tag, src, idx in (("", SRC95, 95), ("low133_", SRC133, 133)):
        for f in sorted(glob.glob(str(ROOT / f"analysis/chaos_check/pd_{tag}L*.json"))):
            d = json.loads(Path(f).read_text())
            for x in d["crossings"]:
                c.append(dict(kind="pd", name=f"pd_{tag}L{d['lag']}", src=src, index=idx, lag=d["lag"], q=x["q"],
                              saved=x, points=[(p["q"], p["mu_neg"]) for p in d["points"]]))
    c.append(dict(kind="window", name="window_low133_q-1.206", src=SRC133, index=133, lag=3, q=-1.206))
    low = {x["index"]: x for f in glob.glob(str(ROOT / "analysis/low_cluster_check/chunk_*.json"))
           for x in json.loads(Path(f).read_text())}
    sel = json.loads((ROOT / "analysis/low_cluster_check/selected.json").read_text())
    pick, seen = [], set()
    for i in sorted(low):  # one stable and one unstable orbit for each cluster number 3, 4, 5 (lag 1)
        x = low[i]
        k = (x["m"], x.get("stable"))
        if x["status"] == "ok" and x["lag"] == 1 and x["valid"] and x["m"] in (3, 4, 5) and k not in seen:
            seen.add(k)
            pick.append(i)
    for i in pick:
        c.append(dict(kind="low", name=f"low_{i}", index=i, lag=1, saved=low[i], sel=sel[i]))
    for tag, src, idx, q in (("95", SRC95, 95, -1.0), ("low133", SRC133, 133, -1.2)):
        c.append(dict(kind="chaos", name=f"chaos_{tag}_q{q:+.3f}", src=src, index=idx, q=q))
    return c


def jac_ext(part, u, K, e, ncopy, blk):
    """Jacobian of ncopy copies of the reduced system plus (optionally) m internal 2x2 blocks along copy 1."""
    n, m = part.n, part.m
    d = u.size
    J = np.zeros((d, d))
    for c in range(ncopy):
        J[c * n:(c + 1) * n, c * n:(c + 1) * n] = part.jac(u[c * n:(c + 1) * n], K, e)
    if blk:
        o = ncopy * n
        for c in range(m):
            q = o + 4 * c
            a = (1 - u[c] ** 2) / e - K
            w = u[q:q + 4]
            J[q, q], J[q, q + 2] = a, -1 / e
            J[q + 1, q + 1], J[q + 1, q + 3] = a, -1 / e
            J[q + 2, q], J[q + 2, q + 2] = e, -e * GAMMA
            J[q + 3, q + 1], J[q + 3, q + 3] = e, -e * GAMMA
            J[q, c] = -2 * u[c] / e * w[0]
            J[q + 1, c] = -2 * u[c] / e * w[1]
    return J


def rhs_ext(part, u, K, e, ncopy, blk):
    n, m = part.n, part.m
    out = [part.rhs(u[c * n:(c + 1) * n], K, e) for c in range(ncopy)]
    if blk:
        W = u[ncopy * n:].reshape(m, 2, 2)
        A = np.zeros((m, 2, 2))
        A[:, 0, 0] = (1 - u[:m] ** 2) / e - K
        A[:, 0, 1] = -1 / e
        A[:, 1, 0] = e
        A[:, 1, 1] = -e * GAMMA
        out.append((A @ W).ravel())
    return np.concatenate(out)


def radau(part, K, e, y0, t0, t1, rtol=RT, atol=AT, ncopy=1, blk=False, events=None, **kw):
    s = solve_ivp(lambda _t, u: rhs_ext(part, u, K, e, ncopy, blk), (t0, t1), y0, method="Radau", rtol=rtol,
                  atol=atol, jac=lambda _t, u: jac_ext(part, u, K, e, ncopy, blk), events=events, **kw)
    assert s.success, s.message
    return s


def monodromy(part, K, e, z, T):
    n = part.n
    M = np.zeros((n, n))
    for j in range(n):
        dz = np.zeros(n)
        dz[j] = DELTA
        M[:, j] = (radau(part, K, e, z + dz, 0, T).y[:, -1] - radau(part, K, e, z - dz, 0, T).y[:, -1]) / (2 * DELTA)
    return M


def shoot(part, K, e, w):
    """Newton on (phi_T(z) - z, z_1) with the central-difference monodromy of the Radau flow."""
    n = part.n
    ERRS.clear()
    for it in range(12):
        z, T = w[:n], w[-1]
        zT = radau(part, K, e, z, 0, T).y[:, -1]
        res = np.r_[zT - z, z[0]]
        err = np.max(abs(res))
        ERRS.append(float(err))
        if err < 1e-9:
            return w, it
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = monodromy(part, K, e, z, T) - np.eye(n)
        A[:n, -1] = part.rhs(zT, K, e)
        A[-1, 0] = 1
        d = np.linalg.lstsq(A, -res, rcond=1e-12)[0]
        for a in (1, .5, .25, .125, .0625):  # backtracking, as in Partition.shoot
            zz = w[:n] + a * d[:n]
            if np.max(abs(np.r_[radau(part, K, e, zz, 0, w[-1] + a * d[-1]).y[:, -1] - zz, zz[0]])) < err:
                break
        w = w + a * d
    return None, it


def guess(part, K, e, v, lag, transient):
    opt = dict(method="DOP853", rtol=1e-8, atol=1e-10, max_step=.05)
    v = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, transient), v, **opt).y[:, -1]
    s = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 40 * (lag + 2)), v, events=chaos_check.sec, **opt)
    ts, ys = s.t_events[0], s.y_events[0]
    return np.r_[ys[-1], ts[-1] - ts[-1 - lag]]


def analyse(part, K, e, w):
    n, m = part.n, part.m
    z, T = w[:n], w[-1]
    M = monodromy(part, K, e, z, T)
    mu = np.linalg.eigvals(M)
    mu = np.delete(mu, int(np.argmin(abs(mu - 1))))
    mu = mu[np.argsort(-abs(mu))]
    s = radau(part, K, e, np.r_[z, np.tile(np.eye(2), (m, 1, 1)).ravel()], 0, T, blk=True)
    B = s.y[n:, -1].reshape(m, 2, 2)
    im = np.max(abs(np.linalg.eigvals(B)), axis=1)
    big = part.sizes > 1
    return dict(period=float(T), mu=[[float(x.real), float(x.imag)] for x in mu],
                collective_max=float(abs(mu[0])), internal_max=im.tolist(),
                internal_max_big=float(im[big].max()) if big.any() else None)


def orbit(c):
    if c["kind"] == "low":
        r = c["sel"]
        e, K = r["eps"], r["K"]
        part, v = Partition(r["sizes"]), np.asarray(r["centroid"])
        w0 = guess(part, K, e, v, 1, 400)
    else:
        chaos_check.SRC = ROOT / c["src"]
        r, part, v = chaos_check.load(c["index"])
        e = r["eps"]
        K = c["q"] / e
        qg = c["q"] + 5e-4 if c["kind"] == "pd" else c["q"]  # pd: the orbit is stable just above the crossing
        w0 = guess(part, qg / e, e, v, c["lag"], 20000)
        # long orbits: converge on the DOP853 flow at qg, then continue to q in five steps (a direct shot from the
        # qg guess fails for L8, residual 0.27)
        for qq in np.linspace(qg, c["q"], 6)[int(qg == c["q"]) * 5:]:
            wd = part.shoot(w0, qq / e, e, 2e-11)
            if wd is None:
                break
            w0 = wd
    t0 = time.perf_counter()
    w, it = shoot(part, K, e, w0)
    res = dict(case=c["name"], kind=c["kind"], eps=e, K=K, q=e * K, lag=c["lag"], newton=it, residuals=list(ERRS))
    if w is None:
        res["status"] = "shooting_failed"
    else:
        res.update(status="ok", **analyse(part, K, e, w))
        if c["kind"] == "pd":
            neg = [x for x in res["mu"] if abs(x[1]) < 1e-9 and x[0] < 0]
            mneg = min(neg, key=lambda x: abs(x[0] + 1))[0] if neg else None
            P = sorted(c["points"])
            k = int(np.argmin([abs(p[0] - c["q"]) for p in P]))
            k = max(1, min(k, len(P) - 1))
            slope = (P[k][1] - P[k - 1][1]) / (P[k][0] - P[k - 1][0])
            res.update(mu_neg=mneg, saved=c["saved"], slope=slope,
                       q_shift=None if mneg is None else float(-(mneg + 1) / slope))
        if c["kind"] == "low":
            res["saved"] = {k: c["saved"][k] for k in ("period", "collective_max", "internal_max_big", "stable")}
    res["seconds"] = time.perf_counter() - t0
    return res


def chaos(c, rtol):
    chaos_check.SRC = ROOT / c["src"]
    r, part, v = chaos_check.load(c["index"])
    e = r["eps"]
    K = c["q"] / e
    n, m = part.n, part.m
    t0 = time.perf_counter()
    v = radau(part, K, e, v, 0, 5000, rtol, rtol / 100).y[:, -1]
    g = np.random.default_rng(0).standard_normal(n)
    b = v + chaos_check.D0 * g / np.linalg.norm(g)
    W = np.tile(np.eye(2), (m, 1, 1))
    lsum, isum, t, tr = 0., np.zeros(m), 0., []
    while t < 15000 - 1e-9:
        s = radau(part, K, e, np.r_[v, b, W.ravel()], t, t + chaos_check.DT, rtol, rtol / 100, ncopy=2, blk=True,
                  events=chaos_check.sec)
        u = s.y[:, -1]
        v, dd = u[:n], u[n:2 * n] - u[:n]
        nd = np.linalg.norm(dd)
        lsum += np.log(nd / chaos_check.D0)
        b = v + chaos_check.D0 * dd / nd
        W = u[2 * n:].reshape(m, 2, 2)
        for k in range(m):
            q, rr = np.linalg.qr(W[k])
            isum[k] += np.log(abs(rr[0, 0]))
            W[k] = q
        tr += s.t_events[0].tolist()
        t += chaos_check.DT
    ret = np.diff(tr)
    big = part.sizes > 1
    intl = isum / t
    return dict(case=c["name"], kind="chaos", rtol=rtol, eps=e, q=c["q"], lyap_max=lsum / t, internal=intl.tolist(),
                internal_max=float(intl[big].max()), returns=len(tr), return_mean=float(ret.mean()),
                return_std=float(ret.std()), seconds=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "orbit", "chaos"])
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-10)
    a = ap.parse_args()
    cs = cases()
    if a.mode == "list":
        for i, c in enumerate(cs):
            print(i, c["kind"], c["name"])
        return
    c = cs[a.case]
    res = orbit(c) if a.mode == "orbit" else chaos(c, a.rtol)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if a.mode == "orbit" else f"_r{a.rtol:g}"
    (OUT / f"{c['name']}{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("saved",)})[:600], flush=True)


if __name__ == "__main__":
    main()

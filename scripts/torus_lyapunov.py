#!/usr/bin/env python3
"""Leading Lyapunov exponents of the slow oscillation of a reduced split system over a long time
(exploratory, 2026-10-25).

Reduced system as in slow_return.py (run --index, record --rec, clusters cut at --cut). No noise; transient 5000.
Then the state with --k tangent vectors (variational equation with Partition.jac), re-orthonormalized (QR) every
10 time units, integrated in chunks of 1000 to --length. The first --skip time units of the tangent dynamics are
discarded (alignment of the initial vectors; cf. RESULTS_NOTES 2026-10-24, the QR transient). The cumulative
exponents lambda_i(T) are recorded every 1000 time units, together with the internal (transverse) exponent of every
cluster (one vector of its 2x2 block per cluster, renormalized every 10 time units, same --skip); fixed windows follow
from differences of lambda_i(T) T. For a two-frequency torus lambda_1 and lambda_2 tend to 0
(flow direction and the torus angle, roughly like 1/T) and lambda_3 < 0; for weak chaos lambda_1 levels off > 0.
  -> analysis/switching_check/torus_lyapunov_<index><--tag>.json

  python3 scripts/torus_lyapunov.py --index 90 --rec 1576 --length 200000
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import GAMMA, Partition  # noqa: E402
from large_n_reduced_floquet import OPT, reduce  # noqa: E402
from renoise_summary import D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--rec", type=int, required=True)
    ap.add_argument("--cut", type=float, default=1e-3)
    ap.add_argument("--length", type=float, default=200000.)
    ap.add_argument("--skip", type=float, default=5000.)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    s = json.loads((D / f"rn_{a.index:04d}_states.json").read_text())
    e, K = s["eps"], s["K"]
    zs = np.load(D / f"rn_{a.index:04d}_states_z.npy")
    sizes, v = reduce(zs[a.rec], zs.shape[1] // 2, a.cut)
    part = Partition(sizes.tolist())
    n, k = part.n, a.k
    v = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 5000), v, **OPT).y[:, -1]

    m = part.m

    def f(_t, u):
        z = u[:n]
        A = np.zeros((m, 2, 2))
        A[:, 0, 0] = (1 - z[:m] ** 2) / e - K
        A[:, 0, 1] = -1 / e
        A[:, 1, 0] = e
        A[:, 1, 1] = -e * GAMMA
        return np.r_[part.rhs(z, K, e), (part.jac(z, K, e) @ u[n:n + n * k].reshape(n, k)).ravel(),
                     np.einsum("aij,aj->ai", A, u[n + n * k:].reshape(m, 2)).ravel()]

    Q = np.linalg.qr(np.random.default_rng(0).standard_normal((n, k)))[0]
    w = np.tile([np.sqrt(.5), np.sqrt(.5)], (m, 1))  # one internal (transverse) vector per cluster
    lsum, isum, t, rows = np.zeros(k), np.zeros(m), 0., []
    while t < a.length + a.skip - 1e-9:
        u = solve_ivp(f, (t, t + 10), np.r_[v, Q.ravel(), w.ravel()], **OPT).y[:, -1]
        v, W, w = u[:n], u[n:n + n * k].reshape(n, k), u[n + n * k:].reshape(m, 2)
        Q, R = np.linalg.qr(W)
        nw = np.linalg.norm(w, axis=1)
        w = w / nw[:, None]
        t += 10
        if t > a.skip:
            lsum += np.log(abs(np.diag(R)))
            isum += np.log(nw)
            if round(t - a.skip) % 1000 == 0:
                T = t - a.skip
                rows.append([T] + (lsum / T).tolist() + (isum / T).tolist())
                if round(T) % 20000 == 0:
                    print(f"T {T:.0f}", np.round(lsum / T, 6), "internal", np.round(isum / T, 6), flush=True)
    out = dict(index=a.index, rec=a.rec, sizes=sizes.tolist(), skip=a.skip, k=k,
               columns=["T"] + [f"lambda{i + 1}" for i in range(k)] + [f"internal_{s_}" for s_ in sizes.tolist()],
               rows=rows)
    (D / f"torus_lyapunov_{a.index:04d}{a.tag}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()

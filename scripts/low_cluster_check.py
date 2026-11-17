#!/usr/bin/env python3
"""Are there non-periodic (e.g. chaotic) or unstable final states among the runs with fewer than six clusters?
(exploratory, 2026-10-23)

Question (author): S11 reports a chaotic seven-cluster state; runs with fewer than six clusters were not tested.
Grid of Supplementary Material S10.2 (N = 50, 100, 200, 400, 1000; eps in {0.2, ..., 0.4}; eps K in {-0.4, ...,
-1.0, -1.2}; seeds 5001-5050), all runs with 2-5 clusters (flagged runs included and marked).

  extract : cluster centroids and sizes of each saved final state (max-norm components, threshold 1e-4)
            -> analysis/low_cluster_check/selected.json
  run     : runs --chunk of --nchunks (index % nchunks == chunk):
            1. as cluster_state_floquet.audit_run: reduced m-cluster system from the centroids to t = 400, upward
               section x_1 = 0, smallest lag <= 4 with return error < 1e-6, shooting (2e-11), full audit
               (collective multipliers; internal multipliers of clusters of size >= 2);
            2. if no recurrence: continue to t = 5000 and search lags <= 20 (error < 1e-6 over the last 5 returns);
            3. if still none: two-trajectory largest Lyapunov exponent and transverse exponents over 10000 time units
               (chaos_check.two_traj, rtol 1e-10).
            -> analysis/low_cluster_check/chunk_<chunk>.json

  pextract: unflagged runs judged unstable (status ok, not stable) or without a periodic orbit (no_recurrence),
            with their saved final states -> analysis/low_cluster_check/perturb_selected.json
  perturb : as many_cluster_check (test 2): Gaussian noise of sd 1e-6 on all 2N coordinates of the final state of
            perturb_selected.json entry --index, full-system RK4 (h = 0.005) to t = 20000, cluster counts at
            t = 1000, 5000, 10000, 20000 -> analysis/low_cluster_check/perturb_<index>.json

  python3 scripts/low_cluster_check.py extract
  python3 scripts/low_cluster_check.py run --chunk 0 --nchunks 96        (remote host: scipy)
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_three_cluster import Partition, ROOT  # noqa: E402
from chaos_check import sec, two_traj  # noqa: E402

OUT = ROOT / "analysis/low_cluster_check"
TAGS = {50: "N50", 100: "N100final", 200: "N200", 400: "N400", 1000: "N1000"}
EPS = {.2, .25, .3, .35, .4}
QS = {-.4, -.5, -.6, -.7, -.8, -.9, -1., -1.2}
TOL = 2e-11
OPT = dict(method="DOP853", rtol=1e-10, atol=1e-12, max_step=.05)


def extract():
    sel = []
    for n, tag in TAGS.items():
        for r in json.loads((ROOT / f"analysis/eps_cluster_ensemble/runs_{tag}.json").read_text()):
            if not (round(r["eps"], 3) in EPS and round(r["eps"] * r["K"], 3) in QS and 5001 <= r["seed"] < 5051
                    and r["clusters"] < 6):
                continue
            z = np.asarray(r["final_state"], float)
            N = z.size // 2
            d = np.maximum(abs(z[:N, None] - z[None, :N]), abs(z[N:, None] - z[None, N:]))
            m, lab = connected_components(d < 1e-4, directed=False)
            sel.append(dict(N=n, eps=r["eps"], K=r["K"], seed=r["seed"], clusters=r["clusters"], m=int(m),
                            unflagged=bool(r["thresholds_agree"] and r["tail_constant"]),
                            sizes=np.bincount(lab).tolist(),
                            centroid=[z[:N][lab == c].mean() for c in range(m)] + [z[N:][lab == c].mean()
                                                                                   for c in range(m)]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selected.json").write_text(json.dumps(sel))
    print(len(sel), "runs")


def check(r):
    e, K = r["eps"], r["K"]
    part = Partition(r["sizes"])
    row = {k: r[k] for k in ("N", "eps", "K", "seed", "clusters", "m", "unflagged", "sizes")}
    big = np.asarray(r["sizes"]) > 1
    s = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 400), np.asarray(r["centroid"]), events=sec, **OPT)
    ts, ys = s.t_events[0], s.y_events[0]
    lag = next((L for L in range(1, 5) if len(ts) > L + 1 and np.max(abs(ys[-1] - ys[-1 - L])) < 1e-6), None)
    stage = 1
    if lag is None and len(ts) >= 6:
        s = solve_ivp(lambda _t, u: part.rhs(u, K, e), (0, 4600), s.y[:, -1], events=sec, **OPT)
        ts, ys = s.t_events[0], s.y_events[0]
        lag = next((L for L in range(1, 21) if len(ts) > L + 5 and
                    max(np.max(abs(ys[-1 - k] - ys[-1 - k - L])) for k in range(5)) < 1e-6), None)
        stage = 2
    if len(ts) < 6:
        return dict(row, status="no_section_returns")
    if lag is None:
        _, lam, intl, tr, _ = two_traj(part, s.y[:, -1], K, e, 10000., OPT, internal=True, returns=True)
        ret = np.diff(tr)
        return dict(row, status="no_recurrence", lyap_max=float(lam),
                    internal_exp_max=float(intl[big].max()) if big.any() else None,
                    return_time_std=float(ret.std()))
    w = part.shoot(np.r_[ys[-1], ts[-1] - ts[-1 - lag]], K, e, TOL)
    if w is None:
        return dict(row, status="shooting_failed", lag=lag, stage=stage)
    a = part.audit(w, K, e, TOL)
    im = np.asarray(a["internal_max"])
    mx = max(a["collective_max"], float(im[big].max()) if big.any() else 0.)
    return dict(row, status="ok", lag=lag, stage=stage, period=a["period"], collective_max=a["collective_max"],
                internal_max_big=float(im[big].max()) if big.any() else None, max_all=mx, valid=a["valid"],
                stable=bool(a["valid"] and mx < 1))


def run(chunk, nchunks):
    sel = json.loads((OUT / "selected.json").read_text())
    out = []
    for i in range(chunk, len(sel), nchunks):
        try:
            res = check(sel[i])
        except Exception as ex:  # keep going; the run is reported with its error
            res = dict({k: sel[i][k] for k in ("N", "eps", "K", "seed", "clusters")}, status=f"error: {ex}")
        res["index"] = i
        out.append(res)
        if res["status"] != "ok" or not res.get("stable"):
            print(i, res["N"], res["eps"], round(res["eps"] * res["K"], 2), res["seed"], res["clusters"],
                  res["status"], res.get("stable"), res.get("max_all"), res.get("lyap_max"), flush=True)
    (OUT / f"chunk_{chunk:03d}.json").write_text(json.dumps(out))
    print("chunk", chunk, "done", len(out), flush=True)


def pextract():
    res = [x for f in sorted(OUT.glob("chunk_*.json")) for x in json.loads(f.read_text())]
    want = {(x["N"], x["eps"], x["K"], x["seed"]): x for x in res if x["unflagged"] and
            (x["status"] == "no_recurrence" or (x["status"] == "ok" and not x["stable"]))}
    sel = []
    for n, tag in TAGS.items():
        for r in json.loads((ROOT / f"analysis/eps_cluster_ensemble/runs_{tag}.json").read_text()):
            x = want.get((n, r["eps"], r["K"], r["seed"]))
            if x and round(r["eps"], 3) in EPS and round(r["eps"] * r["K"], 3) in QS and r["clusters"] < 6:
                sel.append(dict(low_index=x["index"], status=x["status"], final_state=r["final_state"],
                                **{k: r[k] for k in ("eps", "K", "seed", "clusters", "sizes")}, N=n))
    (OUT / "perturb_selected.json").write_text(json.dumps(sel))
    print(len(sel), "runs of", len(want))


def perturb(index):
    from many_cluster_check import continue_full
    r = json.loads((OUT / "perturb_selected.json").read_text())[index]
    z = np.asarray(r["final_state"], float)
    z = z + np.random.default_rng(index).normal(0, 1e-6, z.size)
    row = {k: r[k] for k in r if k != "final_state"}
    row["perturbed"] = continue_full(z, r["eps"], r["K"], 20000., [1000., 5000., 10000., 20000.])
    (OUT / f"perturb_{index:04d}.json").write_text(json.dumps(row))
    print(index, r["N"], r["eps"], round(r["eps"] * r["K"], 2), r["seed"], r["clusters"], r["status"],
          [c["clusters"] for c in row["perturbed"]], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "run", "pextract", "perturb"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=96)
    a = ap.parse_args()
    if a.mode == "extract":
        extract()
    elif a.mode == "pextract":
        pextract()
    elif a.mode == "perturb":
        perturb(a.index)
    else:
        run(a.chunk, a.nchunks)


if __name__ == "__main__":
    main()

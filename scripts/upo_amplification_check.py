#!/usr/bin/env python3
"""Do approaches of the noisy full-system orbit to a transversely unstable UPO precede the transverse amplification
of the cluster that is unstable on that UPO? (2026-10-27; run 143 first)

Input: analysis/chaos_check/edgefull_<case>_q<q><tag>.json written by chaos_edge_check.py full --phase_sec (with the
cluster means at every section crossing, sec_mean) and the UPO of pd_longitudinal_check (pdlong_<case>_L<lag>_chaos
.json, the point nearest to q; if its q differs by more than 1e-9, the UPO is shot again at q from it and
its internal multipliers recomputed).
  - UPO section points: the orbit of the UPO integrated over one period in the reduced system, its upward x_0 = 0
    crossings (lag points);
  - distance d_k at every crossing k: max over the 2m - 1 coordinates (x_2..x_m, y_1..y_m; x_1 = 0 on the section)
    of |z_k - u| / s, s = the
    standard deviation of that coordinate over all crossings of the orbit, minimized over the UPO section points u
    (the section fixes the phase of the UPO up to its lag);
  - transverse RMS r_k of the chosen clusters (sqrt(rms_x^2 + rms_y^2)) and the observed transverse RMS growth rate
    g_k = ln(r_(k+lag) / r_k) / (t_(k+lag) - t_k), over one UPO period so that both crossings are at the same UPO
    phase (with lag 2 the rms alternates between the two section points; with the noise g is not a finite-time
    Lyapunov exponent of the variational equation);
  - corr(d_k, g_(k+j)) for j = -5..10 (g_(k+j) covers crossings k+j .. k+j+lag, so j >= 0 is growth after the
    distance was measured, j = -1 straddles it with lag 2, j <= -lag is before);
  - conditional statistics: crossings binned by d_k (quintiles); per bin the mean of ln(r_(k+j) / r_k) for
    j = 1..--ahead (default 6, a multiple of the lag), and the fraction with max r_(k+j) / r_k >= 3 over j = lag,
    2 lag, .. (same UPO phase only: with lag 2 the rms alternates by a factor 2-3 between the two section points, so
    odd j would compare different phases; corrected 2026-10-27, also for the amplification of approach episodes);
  - events: r crosses --thr times its median upwards; the median of d around the event start (k - 15 .. k + 10),
    compared with the median of d over all crossings; the longest event in detail.
Crossings before t = --skip are dropped.
  --norm l2 uses instead the rms of the scaled coordinates (less dominated by the coordinate of smallest std), l2raw
  the rms of the unscaled coordinates (no weighting by the std).
  -> analysis/chaos_check/upoamp_<case><tag>_<norm>.json

  python3 scripts/upo_amplification_check.py --case 143 --q=-0.8040 --tag _bubsec2 --lag 2 --clusters 4 0
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chaos_check import sec  # noqa: E402
from pd_longitudinal_check import start, OUT, multipliers  # noqa: E402


def runs(ab):
    d = np.diff(np.r_[0, ab.astype(int), 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return st, en - st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="143")
    ap.add_argument("--q", type=float, required=True)
    ap.add_argument("--tag", default="_bubsec2")
    ap.add_argument("--lag", type=int, default=2)
    ap.add_argument("--clusters", type=int, nargs="+", required=True, help="cluster indices (as in sizes)")
    ap.add_argument("--skip", type=float, default=2000.)
    ap.add_argument("--ahead", type=int, default=6, help="a multiple of --lag (same UPO phase)")
    ap.add_argument("--thr", type=float, default=10.)
    ap.add_argument("--merge", type=int, default=2, help="approach episodes closer than this many crossings are merged")
    ap.add_argument("--norm", choices=["max", "l2", "l2raw"], default="max", help="max norm or rms (l2 / sqrt(n)) of the scaled "
                    "coordinates")
    a = ap.parse_args()
    e, part, _ = start(a.case)
    d = json.loads((OUT / f"edgefull_{a.case}_q{a.q:+.4f}{a.tag}.json").read_text())
    up = min(json.loads((OUT / f"pdlong_{a.case}_L{a.lag}_chaos.json").read_text())["points"],
             key=lambda p: abs(p["q"] - a.q))
    w = np.asarray(up["w"])
    reshot = None
    if abs(up["q"] - a.q) > 1e-9:  # the UPO grid misses the orbit's q: shoot again at q from the nearest UPO
        reshot = up["q"]
        w = part.shoot(w, a.q / e, e, 2e-11)
        assert w is not None, f"shooting from the UPO at q={up['q']} to q={a.q} failed"
        up = multipliers(part, w, a.q / e, e)
        print(f"UPO shot again at q={a.q} from the grid point q={reshot}; period {w[-1]:.6f}", flush=True)
    s = solve_ivp(lambda _t, v: part.rhs(v, up["K"], e), (0, 1.02 * w[-1]), w[:-1], method="DOP853", rtol=1e-12,
                  atol=1e-14, max_step=.05, events=sec)
    U = s.y_events[0][s.t_events[0] > 1.][:a.lag]
    t = np.asarray(d["sec_t"])
    keep = t > a.skip
    t, Z, R = t[keep], np.asarray(d["sec_mean"])[keep], np.asarray(d["sec_rms"])[keep]
    Z, U = Z[:, 1:], U[:, 1:]  # x_0 = 0 on the section: not a coordinate there
    sc = Z.std(0)
    D = np.array([abs(Z - u) / sc for u in U])  # (lag points, crossings, coordinates)
    if a.norm == "l2raw":  # no scaling by the std
        D = np.array([abs(Z - u) for u in U])
    nrm = (lambda X: X.max(-1)) if a.norm == "max" else (lambda X: np.sqrt((X ** 2).mean(-1)))
    iu = np.argmin(nrm(D), axis=0)
    Dk = D[iu, np.arange(len(iu))]
    dist = nrm(Dk)
    names = ([f"x{i}" for i in range(2, (len(sc) + 1) // 2 + 1)] + [f"y{i}" for i in range(1, (len(sc) + 1) // 2 + 1)])
    arg = np.bincount(Dk.argmax(1), minlength=len(sc)) / len(dist)
    print("coordinate std:", {nm: f"{v:.3g}" for nm, v in zip(names, sc)}, flush=True)
    print("fraction of crossings where the coordinate sets the max norm:",
          {nm: round(float(v), 3) for nm, v in zip(names, arg)}, flush=True)
    big = d["sec_clusters"]
    out = dict(case=a.case, q=up["q"], upo_reshot_from=reshot, upo_lag=a.lag, upo_period=float(w[-1]), upo_internal=up["internal"],
               distance="max over the cluster-mean coordinates except x_0 (= 0 on the section) of |z - u| / std(coordinate), "
                        "min over UPO section points u", scale=sc.tolist(), d_quantiles=np.quantile(dist, [.01, .1, .5, .9]).tolist(),
               coordinates=names, argmax_fraction=arg.tolist(), clusters={})
    print(f"UPO lag {a.lag} q={up['q']:.4f} T={w[-1]:.3f}; d quantiles 1/10/50/90%: "
          f"{np.round(out['d_quantiles'], 3).tolist()}", flush=True)
    qb = np.quantile(dist, [0, .2, .4, .6, .8, 1.])
    for c in a.clusters:
        j = big.index(c)
        r = np.hypot(R[:, j, 0], R[:, j, 1])
        g = np.log(r[a.lag:] / r[:-a.lag]) / (t[a.lag:] - t[:-a.lag])  # same UPO phase
        n = len(r) - a.ahead
        fut = np.array([np.log(r[k + 1:k + 1 + a.ahead] / r[k]) for k in range(n)])
        rows = []
        for b0, b1 in zip(qb[:-1], qb[1:]):
            m = (dist[:n] >= b0) & (dist[:n] <= b1)
            rows.append(dict(d_range=[float(b0), float(b1)], n=int(m.sum()),
                             mean_log_growth=fut[m].mean(0).tolist(),
                             frac_x3=float((fut[m][:, a.lag - 1::a.lag].max(1) >= np.log(3)).mean()),
                             mean_g_next=float(g[:n][m].mean())))
        st, ln = runs(r > a.thr * np.median(r))
        trig = [dist[k - 15:k + 11] for k in st if 15 <= k < len(dist) - 11]
        trig_med = np.median(trig, axis=0).tolist() if trig else []
        lo, hi = runs(r > 3 * np.median(r))
        if len(lo):
            kL = int(lo[np.argmax(hi)])
            detail = dict(start=kL, length=int(hi.max()), t=t[max(kL - 10, 0):kL + hi.max() + 10].tolist(),
                          d=dist[max(kL - 10, 0):kL + hi.max() + 10].tolist(),
                          r=r[max(kL - 10, 0):kL + hi.max() + 10].tolist())
        else:  # no crossing above 3x the median
            detail = None
        out["clusters"][str(c)] = dict(size=int(d["sizes"][c]), r_median=float(np.median(r)), by_distance=rows,
                                       events=len(st), event_trigger_median_d=trig_med, longest_x3_event=detail,
                                       corr_d_next_g=float(np.corrcoef(dist[:-a.lag], g)[0, 1]))
        print(f"cluster {c} (size {d['sizes'][c]}): median r {np.median(r):.2e}; corr(d_k, g_k) "
              f"{out['clusters'][str(c)]['corr_d_next_g']:+.3f}; events (> {a.thr:g}x median) {len(st)}", flush=True)
        for row in rows:
            print(f"   d in [{row['d_range'][0]:.3f}, {row['d_range'][1]:.3f}] n={row['n']}: mean ln(r_k+j/r_k) "
                  f"j=1..{a.ahead} {np.round(row['mean_log_growth'], 3).tolist()}  P(max >= 3x)={row['frac_x3']:.3f}",
                  flush=True)
        if trig_med:
            print(f"   median d around event starts (k-15..k+10): {np.round(trig_med, 3).tolist()} "
                  f"(all crossings: {np.median(dist):.3f})", flush=True)
        if detail:
            print(f"   longest >3x event: start k={detail['start']}, length {detail['length']}; d: "
                  f"{np.round(detail['d'], 3).tolist()}\n   r/median: "
                  f"{np.round(np.asarray(detail['r']) / np.median(r), 1).tolist()}", flush=True)
        else:
            print("   no crossing above 3x the median", flush=True)
    # robustness: closeness threshold (quantile of d) x number of time blocks of the block bootstrap
    rng = np.random.default_rng(0)
    out["robustness"] = []
    for c in a.clusters:
        j = big.index(c)
        r = np.hypot(R[:, j, 0], R[:, j, 1])
        lr = np.log(r) - np.log(r).mean()
        ac = [float(np.dot(lr[:-L], lr[L:]) / np.dot(lr, lr)) for L in (1, 2, 5, 10, 17, 30, 50)]
        print(f"cluster {c}: autocorrelation of ln r at lags 1,2,5,10,17,30,50 crossings: {np.round(ac, 3).tolist()}",
              flush=True)
        n = len(dist) - a.ahead
        x3 = np.array([np.max(r[k + a.lag:k + 1 + a.ahead:a.lag]) / r[k] >= 3 for k in range(n)])  # same phase
        lg = np.array([np.log(r[k + a.ahead] / r[k]) for k in range(n)])
        for qt in (.05, .1, .2):
            dthr = float(np.quantile(dist, qt))
            cl = dist[:n] <= dthr
            point = [float(x3[cl].mean() - x3[~cl].mean()), float(lg[cl].mean() - lg[~cl].mean())]
            for nb in (10, 20, 50, 100):  # number of equal blocks (block length = n // nb crossings)
                blocks = np.array_split(np.arange(n), nb)
                eff = []
                for _ in range(2000):
                    idx = np.concatenate([blocks[i] for i in rng.integers(0, nb, nb)])
                    eff.append([x3[idx][cl[idx]].mean() - x3[idx][~cl[idx]].mean(),
                                lg[idx][cl[idx]].mean() - lg[idx][~cl[idx]].mean()])
                ci = np.nanquantile(np.array(eff), [.025, .975], axis=0)
                out["robustness"].append(dict(cluster=c, quantile=qt, d_threshold=dthr, blocks=nb,
                                              block_len=n // nb, effect_P3x=point[0], ci_P3x=ci[:, 0].tolist(),
                                              effect_log=point[1], ci_log=ci[:, 1].tolist()))
                print(f"cluster {c}: d <= {dthr:.3f} ({qt:.0%}) vs rest, {nb} blocks of {n // nb}: P(max >= 3x) "
                      f"{point[0]:+.3f} [{ci[0, 0]:+.3f}, {ci[1, 0]:+.3f}], ln(r_k+{a.ahead}/r_k) {point[1]:+.3f} "
                      f"[{ci[0, 1]:+.3f}, {ci[1, 1]:+.3f}]", flush=True)
    out["lagged_corr"] = {}
    for c in a.clusters:  # corr(d_k, g_(k+j)): does closeness precede (j > 0) or follow (j < 0) the growth?
        j = big.index(c)
        r = np.hypot(R[:, j, 0], R[:, j, 1])
        g = np.log(r[a.lag:] / r[:-a.lag]) / (t[a.lag:] - t[:-a.lag])  # same UPO phase
        n = len(g)
        cc = {}
        for L in range(-5, 11):
            k0, k1 = max(0, -L), min(n, n - L)
            cc[L] = float(np.corrcoef(dist[k0:k1], g[k0 + L:k1 + L])[0, 1])
        out["lagged_corr"][str(c)] = cc
        print(f"cluster {c}: corr(d_k, g_(k+j)) for j=-5..10: " +
              " ".join(f"{L}:{v:+.3f}" for L, v in cc.items()), flush=True)
    ld = dist - dist.mean()
    out["autocorr_d"] = [float(np.dot(ld[:-L], ld[L:]) / np.dot(ld, ld)) for L in (1, 2, 5, 10, 17, 30, 50)]
    print(f"autocorrelation of d at lags 1,2,5,10,17,30,50: {np.round(out['autocorr_d'], 3).tolist()}", flush=True)
    # approach episodes: consecutive crossings with d below its 10% quantile; dwell length and what both clusters do
    st, ln = runs(dist <= np.quantile(dist, .1))
    out["episodes_before_merge"] = int(len(st))
    if len(st):  # merge episodes separated by at most --merge crossings above the threshold
        mst, mln = [st[0]], [ln[0]]
        for k, L in zip(st[1:], ln[1:]):
            if k - (mst[-1] + mln[-1]) <= a.merge:
                mln[-1] = k + L - mst[-1]
            else:
                mst.append(k)
                mln.append(L)
        st, ln = np.array(mst), np.array(mln)
    gaps = np.diff(st)
    print(f"approach episodes: {len(st)}; gaps between starts (crossings) median {np.median(gaps):.0f}, "
          f"episodes separated by > 50 crossings from the previous one: {int((gaps > 50).sum()) + 1}", flush=True)
    ep = []
    rs = {c: np.hypot(R[:, big.index(c), 0], R[:, big.index(c), 1]) for c in a.clusters}
    for k, L in zip(st, ln):
        if k + L + 10 >= len(dist):
            continue
        ep.append(dict(start=int(k), dwell=int(L), **{f"amp_{c}": float(np.max(rs[c][k:k + L + 10:a.lag]) / rs[c][k])
                                                       for c in a.clusters}))
    out["approach_episodes"] = ep
    dw = np.array([x["dwell"] for x in ep])
    out["episodes_after_merge"] = len(ep)
    print(f"approach episodes before / after merging gaps <= {a.merge}: {out['episodes_before_merge']} / {len(ep)}; "
          "with amplification >= 3x within dwell + 10 crossings: " +
          ", ".join(f"cluster {c}: {sum(x[f'amp_{c}'] >= 3 for x in ep)}" for c in a.clusters), flush=True)
    print(f"approach episodes (d <= 10% quantile): {len(ep)}; dwell lengths (crossings): "
          f"{np.bincount(dw).tolist()}", flush=True)
    for lo_, hi_ in ((1, 1), (2, 2), (3, 4), (5, 1000)):
        m = (dw >= lo_) & (dw <= hi_)
        if m.sum():
            print(f"   dwell {lo_}-{hi_}: n={m.sum()} " + " ".join(
                f"cluster {c}: median max r/r_start over dwell+10 = {np.median([x[f'amp_{c}'] for x, mm in zip(ep, m) if mm]):.2f}"
                for c in a.clusters), flush=True)
    out["norm"] = a.norm
    out["dist"] = dist.tolist()
    (OUT / f"upoamp_{a.case}{a.tag}_{a.norm}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

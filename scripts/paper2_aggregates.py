#!/usr/bin/env python3
"""Numbers of paper 2 that are counts or ranges over stored outputs (no integration; 2026-10-28).

Each quantity below was first tallied interactively; this script reproduces it from the stored files only.
  1. Continuation walks (analysis/chaos_check/edge_<case>_*.json; points with lag None and lam > 1e-3 are the
     points classified as chaotic): number of points, number with lam > 1e-3 in all four parts (lam_blocks),
     range of lambda_perp (perp_max), largest lambda_perp over the four parts, fraction of 100-unit windows with
     positive lambda_perp (win_pos_frac) and the largest window value (win_max).                       [Sec. III C, VI B, S5]
  2. Independent scans (chaos_check/scan_95_q*.json, scan_low133_q*.json, reduced_chaos_census/
     candidate_run143_q*_r1e-10.json): largest lambda_perp over the clusters with n >= 2 at the chaotic points. [Sec. VI B]
  3. Table I, lambda_perp and return times at one eps K: DOP853 1e-10 (scan) and Radau 1e-10, 1e-11
     (analysis/radau_check/chaos_*), and the three runs of C5b (candidate_run143_q-0.8000_r1e-1{0,1,2}).   [Table I]
  4. T1-T3: partitions of the stored state (switching_check/rn_<i>_states_z.npy, record rec, extracted to
     switching_check/torus_states_paper2.npz) at thresholds 1e-2 to 1e-6, single linkage in the max norm.                                                        [Sec. VII, S4]
  5. Transients beyond the ends (chaos_check/life_<case>_q<q>.json): the settling time of each run is the time of the
     first return after which the distance between adjacent returns (dist1) stays below 1e-7 to the end, or, for a
     period-two destination, the difference of alternate return intervals; mean, median, standard deviation.  [Table S2]
  6. QR spectra (chaos_check/verify_{95,low133}_r1e-1{1,2}.json, last record at 2e5): lambda_1..3 and the
     Kaplan-Yorke dimension 2 + (lambda_1 + lambda_2)/|lambda_3|.                                       [Table I]
  7. Torus-like states (switching_check/torus_lyapunov_<i>.json, cumulative exponents every 1000 time units): means
     and standard deviations of lambda_1, lambda_2 over the ten windows of 2e4 (differences of lambda(T) T),
     lambda_3 and the largest internal exponent at 2e5.                                                  [Table III]
  8. Long full-system runs with noise (chaos_check/edgefull_<case>_q<q>_bub.json): median and maximum of the largest
     within-cluster spread after the first 2000 time units, and the maximum over the whole record.        [Sec. IV B]
  9. Phase-aligned deviations (chaos_check/edgefull_<case>_q<q>_bubsec2.json, crossings after t = 2000), for every
     cluster of size >= 2: ratio of the maximum to the median of the root-mean-square deviation in x and in y, and
     the longest run of consecutive crossings above three times the median (the longer of x and y); the average
     internal exponent of the cluster at the same eps K on the continuation towards stronger repulsion
     (edge_<case>_end_fwd.json) and its internal multiplier on the transversely unstable periodic orbit
     (pdlong_<case>_L<L>_chaos.json, L = 1 for C7, 2 for C5a and C5b).                                    [Table S1]
  -> analysis/paper2_figs/aggregates.json (and a printed summary)

  python3 scripts/paper2_aggregates.py
"""
from pathlib import Path
import json
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "analysis"
C = A / "chaos_check"
CASES = {"C7": "95", "C5a": "3444", "C5b": "143"}
TORUS = {"T1": (90, 1576), "T2": (46, 1961), "T3": (26, 1300)}


def walks(tag):
    pts = []
    for f in sorted(C.glob(f"edge_{tag}_*.json")):
        try:
            P = json.loads(f.read_text())["points"]
        except (KeyError, json.JSONDecodeError):
            continue
        for p in P:
            if p.get("lag") is None and p["lam"] > 1e-3:
                pts.append(dict(file=f.name, q=p["q"], lam=p["lam"], blocks=p["lam_blocks"], perp=p["perp_max"],
                                perp_blocks=p["perp_blocks"], win_pos=p["win_pos_frac"], win_max=p["win_max"]))
    bad = [dict(file=p["file"], q=p["q"], blocks=p["blocks"]) for p in pts if min(p["blocks"]) <= 1e-3]
    return dict(n=len(pts), n_all_parts=len(pts) - len(bad), exceptions=bad,
                perp_range=[min(p["perp"] for p in pts), max(p["perp"] for p in pts)],
                perp_parts_max=max(max(p["perp_blocks"]) for p in pts),
                win_pos_range=[min(p["win_pos"] for p in pts), max(p["win_pos"] for p in pts)],
                win_max_range=[min(p["win_max"] for p in pts), max(p["win_max"] for p in pts)],
                q_range=[max(p["q"] for p in pts), min(p["q"] for p in pts)])


def perp_big(d):
    return max(v for v, s in zip(d["internal"], d["sizes"]) if s >= 2)


def scans(case):
    if case == "C5b":
        files = (A / "reduced_chaos_census").glob("candidate_run143_q*_r1e-10.json")
    else:
        tag = {"C7": "95", "C5a": "low133"}[case]
        files = [f for f in C.glob(f"scan_{tag}_q*.json") if re.search(r"_q[+-]\d\.\d{3,4}\.json$", f.name)]
    pts = {}
    for f in files:
        d = json.loads(f.read_text())
        lam = d.get("lyap_max", d.get("lyap"))
        if d.get("lag") is None and lam > 1e-3:
            pts[round(d["q"], 4)] = perp_big(d)  # the files with three and four decimals duplicate some q
    return dict(n=len(pts), perp_max=max(pts.values()), q_range=[max(pts), min(pts)])


def table1():
    out = {}
    for case, scan, rad in (("C7", "scan_95_q-1.000.json", "chaos_95_q-1.000_r1e-1{}.json"),
                            ("C5a", "scan_low133_q-1.200.json", "chaos_low133_q-1.200_r1e-1{}.json")):
        d = json.loads((C / scan).read_text())
        rt = np.asarray(d["return_times"])
        rows = [dict(run="DOP853 1e-10", lam=d["lyap_max"], perp=perp_big(d), rt_mean=rt.mean(), rt_sd=rt.std())]
        for k in (0, 1):
            r = json.loads((A / "radau_check" / rad.format(k)).read_text())
            rows.append(dict(run=f"Radau 1e-1{k}", lam=r["lyap_max"], perp=r["internal_max"],
                             rt_mean=r["return_mean"], rt_sd=r["return_std"]))
        out[case] = rows
    rows = []
    for k in (0, 1, 2):
        d = json.loads((A / "reduced_chaos_census" / f"candidate_run143_q-0.8000_r1e-1{k}.json").read_text())
        rows.append(dict(run=f"DOP853 1e-1{k}", length=d["length"], lam=d["lyap"], perp=d["internal_max_big"],
                         rt_mean=d["return_mean"], rt_sd=d["return_sd"]))
    out["C5b"] = rows
    return out


def reduce(z, n, th):
    """Clusters of a full state: single linkage in the max norm, cut at th; sizes in decreasing order (the same
    partition as large_n_reduced_floquet.reduce, written out here so that this script needs only NumPy and SciPy)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    Z = linkage(np.c_[z[:n], z[n:]], method="single", metric="chebyshev")
    lab = fcluster(Z, th, criterion="distance") if th > 0 else np.arange(n)
    return np.sort(np.bincount(lab)[1:])[::-1]


def torus():
    # the three stored states (switching_check/rn_<i>_states_z.npy, record rec; time 10 (rec + 1), before the noise
    # of that step) are extracted to switching_check/torus_states_paper2.npz
    S = np.load(A / "switching_check" / "torus_states_paper2.npz")
    out = {}
    for name, (idx, rec) in TORUS.items():
        z = S[name]
        n = z.size // 2
        out[name] = dict(index=idx, rec=rec, time=10 * (rec + 1),
                         partitions={f"{th:.0e}": [int(x) for x in reduce(z, n, th) if x > 0][:8]
                                     for th in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)})
    return out


def settle_time(r, th=1e-7):
    tr, d = np.asarray(r["return_times"]), np.asarray(r["dist1"])

    def tail(x):
        if not x[-1] < th:
            return None
        bad = np.where(~(x < th))[0]
        return 0 if bad.size == 0 else bad[-1] + 1

    k = tail(d)
    if k is not None:
        return tr[min(k + len(tr) - len(d), len(tr) - 1)], 1
    T = np.diff(tr)
    k = tail(np.abs(T[2:] - T[:-2]))
    return (tr[min(k + 2, len(tr) - 1)], 2) if k is not None else (np.nan, 0)


def lifetimes():
    out = {}
    for f in sorted(C.glob("life_*_q*.json")):
        d = json.loads(f.read_text())
        v = [settle_time(r) for r in d["runs"]]
        t = np.array([x[0] for x in v])
        out[f.stem] = dict(q=d["q"], n=len(v), mean=np.nanmean(t), median=np.nanmedian(t), sd=np.nanstd(t),
                           period1=sum(x[1] == 1 for x in v), period2=sum(x[1] == 2 for x in v))
    return out


def qr_spectra():
    out = {}
    for case, tag in (("C7", "95"), ("C5a", "low133")):
        rows = []
        for r in ("1e-11", "1e-12"):
            lam = json.loads((C / f"verify_{tag}_r{r}.json").read_text())["qr"][-1]["lyap"]
            rows.append(dict(rtol=r, l1=lam[0], l2=lam[1], l3=lam[2], ky=2 + (lam[0] + lam[1]) / abs(lam[2])))
        out[case] = rows
    return out


def torus_exponents():
    out = {}
    for name, (idx, _) in TORUS.items():
        d = json.loads((A / "switching_check" / f"torus_lyapunov_{idx:04d}.json").read_text())
        R = np.asarray(d["rows"], float)
        T = R[:, 0]
        k = [i for i, t in enumerate(T) if round(t) % 20000 == 0]
        cum = R[k, 1:3] * T[k, None]
        win = np.diff(np.vstack([np.zeros(2), cum]), axis=0) / 20000.
        out[name] = dict(l1=[win[:, 0].mean(), win[:, 0].std()], l2=[win[:, 1].mean(), win[:, 1].std()],
                         l3=R[-1, 3], internal_max=R[-1, 4:].max(),
                         internal_cluster=d["columns"][4 + int(np.argmax(R[-1, 4:]))], windows=len(k))
    return out


def noise_runs():
    out = {}
    for case, f in (("C7", "edgefull_95_q-1.0000_bub.json"), ("C5a", "edgefull_3444_q-1.2100_bub.json"),
                    ("C5b", "edgefull_143_q-0.8040_bub.json")):
        sp = np.asarray(json.loads((C / f).read_text())["spread"], float)
        out[case] = dict(median=np.median(sp[2000:]), max=sp[2000:].max(), max_all=sp.max(), over_1e_2=int((sp > 1e-2).sum()))
    return out


def table_s1():
    def longest(b):
        best = cur = 0
        for v in b:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    out = {}
    for case, tag, q, L in (("C7", "95", "-1.0000", 1), ("C5a", "3444", "-1.2100", 2), ("C5b", "143", "-0.8040", 2)):
        d = json.loads((C / f"edgefull_{tag}_q{q}_bubsec2.json").read_text())
        t, R = np.asarray(d["sec_t"]), np.asarray(d["sec_rms"], float)
        R = R[t >= 2000]
        w = json.loads((C / f"edge_{tag}_end_fwd.json").read_text())
        pt = min(w["points"], key=lambda p: abs(p["q"] - float(q)))
        o = json.loads((C / f"pdlong_{tag}_L{L}_chaos.json").read_text())["points"]
        po = min(o, key=lambda p: abs(p["q"] - float(q)))
        rows = []
        for j, a in enumerate(d["sec_clusters"]):
            size = d["sizes"][a]
            rx, ry = R[:, j, 0], R[:, j, 1]
            mult = [re for s_, re, im in po["internal"] if s_ == size and abs(re) > 1e-6]
            rows.append(dict(size=size, ratio_x=rx.max() / np.median(rx), ratio_y=ry.max() / np.median(ry),
                             duration=max(longest(rx > 3 * np.median(rx)), longest(ry > 3 * np.median(ry))),
                             max_rms=float(max(rx.max(), ry.max())), avg_internal=pt["perp"][a],
                             orbit_multiplier=mult[0] if mult else None, orbit_q=po["q"], walk_q=pt["q"]))
        out[case] = rows
    return out


def main():
    res = dict(walks={c: walks(t) for c, t in CASES.items()}, scans={c: scans(c) for c in CASES},
               table1=table1(), torus=torus(), lifetimes=lifetimes(), qr=qr_spectra(), torus_exponents=torus_exponents(),
               noise=noise_runs(), table_s1=table_s1())
    (A / "paper2_figs" / "aggregates.json").write_text(json.dumps(res, indent=1, default=float))
    for c in CASES:
        w, s = res["walks"][c], res["scans"][c]
        print(f"{c}: walk {w['n_all_parts']}/{w['n']} all parts, perp {w['perp_range'][0]:.4f}..{w['perp_range'][1]:.4f}"
              f" (parts max {w['perp_parts_max']:.4f}), positive windows {w['win_pos_range'][0]:.2f}..{w['win_pos_range'][1]:.2f},"
              f" window max {w['win_max_range'][0]:.4f}..{w['win_max_range'][1]:.4f};"
              f" scans {s['n']} chaotic, perp max {s['perp_max']:.4f}, eps K {s['q_range']}")
    for c, rows in res["table1"].items():
        print(c, "; ".join(f"{r['run']}: lam {r['lam']:.4f} perp {r['perp']:.5f} T {r['rt_mean']:.4f}/{r['rt_sd']:.4f}"
                           for r in rows))
    for k, v in res["lifetimes"].items():
        print(k, f"mean {v['mean']:.0f} median {v['median']:.0f} sd {v['sd']:.0f} period1 {v['period1']} period2 {v['period2']}")
    for c, rows in res["qr"].items():
        print(c, "QR", "; ".join(f"{r['rtol']}: l1 {r['l1']:.5f} l2 {r['l2']:.1e} l3 {r['l3']:.4f} D_KY {r['ky']:.3f}" for r in rows))
    for name, t in res["torus_exponents"].items():
        print(name, f"1e5 l1 {1e5 * t['l1'][0]:.2f}+-{1e5 * t['l1'][1]:.2f}, 1e5 l2 {1e5 * t['l2'][0]:.2f}+-{1e5 * t['l2'][1]:.2f},"
              f" l3 {t['l3']:.2e}, internal {t['internal_max']:.2e} ({t['internal_cluster']}), windows {t['windows']}")
    for c, v in res["noise"].items():
        print(c, f"noise run: median {v['median']:.2e} max {v['max']:.2e} (whole record {v['max_all']:.2e}), above 1e-2: {v['over_1e_2']}")
    for c, rows in res["table_s1"].items():
        for r in rows:
            print(c, f"S1 size {r['size']}: internal {r['avg_internal']:.4f}, orbit {r['orbit_multiplier']}, ratio "
                  f"{r['ratio_x']:.1f}, {r['ratio_y']:.1f}, duration {r['duration']}, max {r['max_rms']:.1e}")
    for name, t in res["torus"].items():
        print(name, t["time"], {k: v[:5] for k, v in t["partitions"].items()})


if __name__ == "__main__":
    main()

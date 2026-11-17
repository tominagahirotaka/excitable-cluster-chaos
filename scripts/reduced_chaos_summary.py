#!/usr/bin/env python3
"""Summary of scripts/reduced_chaos_census.py (exploratory, 2026-10-24).

Per sample: 'merged' if two clusters came closer than 1e-4 by the end of the transient (the sample then effectively
has fewer clusters), 'synchronized', 'chaotic' (two-trajectory exponent > 2e-3 and no periodic lag), 'periodic'
(lag found), or 'other'. Chaotic samples are split by transverse stability (largest exponent of clusters of size >= 2
below 0). -> analysis/reduced_chaos_census_summary.json
"""
from pathlib import Path
import collections
import json

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "analysis/reduced_chaos_census"


def cls(r):
    if r["status"] != "ok":
        return r["status"]
    if r["min_cluster_distance"] < 1e-4:
        return "merged"
    if r["lyap"] > 2e-3 and r["lag"] is None:
        return "chaotic_tstable" if (r["internal_max_big"] is not None and r["internal_max_big"] < 0) else "chaotic_tunstable"
    return "periodic" if r["lag"] else "other"


def main():
    rows = [json.loads(f.read_text()) for f in sorted(D.glob("m*_e*_q*_s*.json"))]
    by_m = collections.defaultdict(collections.Counter)
    chaos = []
    for r in rows:
        c = cls(r)
        by_m[r["m"]][c] += 1
        if c.startswith("chaotic"):
            chaos.append(dict(m=r["m"], eps=r["eps"], q=r["q"], s=r["s"], sizes=r["sizes"], lyap=r["lyap"],
                              internal_max_big=r["internal_max_big"], cls=c))
    out = dict(by_m={m: dict(c) for m, c in sorted(by_m.items())}, chaotic=chaos)
    (ROOT / "analysis/reduced_chaos_census_summary.json").write_text(json.dumps(out, indent=1))
    for m, c in sorted(by_m.items()):
        print(m, dict(c))
    for x in sorted(chaos, key=lambda x: (x["m"], x["eps"], x["q"])):
        print(x)


if __name__ == "__main__":
    main()

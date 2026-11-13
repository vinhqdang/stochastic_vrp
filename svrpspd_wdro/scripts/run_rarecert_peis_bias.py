"""RARECERT (PEIS version): check of the silent-underestimation proposition.

For the 30 routes x 2 levels of the rival study (results/rarecert/baselines_routes.csv)
compute the two end-event probabilities P(D_0 > c) and P(P_L > c) by the quadrature
program on single-block routes (all pickups, resp. deliveries, set to zero), and
compare the share s = P(weaker end event)/p of the end mode a single shift misses with
the measured relative error of the single CE shift and of Glasserman--Li (shifted
inner step). Output: results/rarecert/peis/bias_check.csv and bias_check.md.
"""
from __future__ import annotations

import csv
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from quantum.rare import RouteSim  # noqa: E402
from rarecert.baselines import exact_route_prob  # noqa: E402
import run_rarecert_smis as rs  # noqa: E402

OUT = ROOT / "results" / "rarecert" / "peis"
SRC = ROOT / "results" / "rarecert" / "baselines_routes.csv"
CV, RHO = 0.3, 0.6


def task(t):
    d = np.asarray(t["dbar"]); p = np.asarray(t["pbar"]); z = np.zeros_like(d)
    pd_ = exact_route_prob(RouteSim(d, z, CV, RHO), t["c"], V=256, n=161)["p"]
    pp_ = exact_route_prob(RouteSim(z, p, CV, RHO), t["c"], V=256, n=161)["p"]
    return {**{k: t[k] for k in ("route_id", "gamma", "c", "p_ref", "ratio", "single_relerr", "gl_shift_relerr")},
            "p_d0": pd_, "p_pL": pp_}


def main():
    means = rs.route_means()
    tasks = []
    with open(SRC, newline="") as fh:
        for r in csv.DictReader(fh):
            m = means[r["route_id"]]
            tasks.append({"route_id": r["route_id"], "gamma": float(r["gamma"]), "c": float(r["c"]),
                          "p_ref": float(r["p_ref"]), "ratio": float(r["ratio"]),
                          "single_relerr": float(r["single_relerr"]),
                          "gl_shift_relerr": float(r["gl_shift_relerr"]),
                          "dbar": m["dbar"], "pbar": m["pbar"]})
    with mp.Pool(4) as pool:
        rows = pool.map(task, tasks)
    for r in rows:
        lo, hi = sorted((r["p_d0"], r["p_pL"]))
        r["s_missed"] = lo / r["p_ref"]
        r["share_ends"] = (lo + hi) / r["p_ref"]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "bias_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ["# Missed end-mode share s vs single-shift error (30 routes, quadrature)", "",
             "| gamma | group | n | median s | median single err | median GL err | median (err/-s) single | median (err/-s) GL |",
             "|---|---|---|---|---|---|---|---|"]
    for g in sorted({r["gamma"] for r in rows}, reverse=True):
        for name, f in (("ratio>0.75", lambda r: r["ratio"] > 0.75), ("ratio<=0.75", lambda r: r["ratio"] <= 0.75)):
            s = [r for r in rows if r["gamma"] == g and f(r)]
            if not s:
                continue
            med = lambda v: float(np.median(v))
            lines.append(f"| {g:g} | {name} | {len(s)} | {med([r['s_missed'] for r in s]):.3g} | "
                         f"{med([r['single_relerr'] for r in s]):+.3%} | {med([r['gl_shift_relerr'] for r in s]):+.3%} | "
                         f"{med([-r['single_relerr']/r['s_missed'] for r in s if r['s_missed']>1e-3] or [math.nan]):.2f} | "
                         f"{med([-r['gl_shift_relerr']/r['s_missed'] for r in s if r['s_missed']>1e-3] or [math.nan]):.2f} |")
    (OUT / "bias_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

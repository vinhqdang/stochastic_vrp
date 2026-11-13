"""RARECERT (PEIS version): check of the silent-underestimation proposition.

For the 30 routes x 2 levels of the rival study (results/rarecert/baselines_routes.csv)
compute the two end-event probabilities P(D_0 > c) and P(P_L > c) on single-block
routes (all pickups, resp. deliveries, set to zero). A single block has one end mode,
so SMIS is efficient there (Theorem 2(c) when the block has a spectral gap); we use
10^6 draws (relative standard error about 0.2%). A block that cannot reach c (normal
scores are clipped) has probability 0. Then
compare the share s = P(weaker end event)/p of the end mode a single shift misses with
the measured signed relative error (estimate / quadrature reference - 1) of the single
CE shift and of Glasserman--Li (shifted inner step). Output: results/rarecert/peis/bias_check.csv and bias_check.md.
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
from rarecert.smis import smis_estimate, Unreachable  # noqa: E402
import run_rarecert_smis as rs  # noqa: E402

OUT = ROOT / "results" / "rarecert" / "peis"
SRC = ROOT / "results" / "rarecert" / "baselines_routes.csv"
CV, RHO = 0.3, 0.6


N_DRAWS = 1_000_000


def _block_prob(dbar, pbar, c, seed):
    sim = RouteSim(dbar, pbar, CV, RHO)
    try:
        r = smis_estimate(sim, c, N_DRAWS, np.random.default_rng(seed))
    except (Unreachable, ValueError, RuntimeError, IndexError):  # no reachable mode: P = 0
        return 0.0, 0.0
    return r["p_hat"], r["rse"]


def task(t):
    d = np.asarray(t["dbar"]); p = np.asarray(t["pbar"]); z = np.zeros_like(d)
    seed = int(round(-math.log10(t["gamma"]))) * 1000 + sum(map(ord, t["route_id"]))
    pd_, rd = _block_prob(d, z, t["c"], seed)
    pp_, rp = _block_prob(z, p, t["c"], seed + 1)
    return {**{k: t[k] for k in ("route_id", "gamma", "c", "p_ref", "ratio", "single_relerr", "gl_shift_relerr")},
            "p_d0": pd_, "rse_d0": rd, "p_pL": pp_, "rse_pL": rp}


def main():
    means = rs.route_means()
    tasks = []
    with open(SRC, newline="") as fh:
        for r in csv.DictReader(fh):
            m = means[r["route_id"]]
            tasks.append({"route_id": r["route_id"], "gamma": float(r["gamma"]), "c": float(r["c"]),
                          "p_ref": float(r["p_ref"]), "ratio": float(r["ratio"]),
                          "single_relerr": float(r["single_p"]) / float(r["p_ref"]) - 1.0,
                          "gl_shift_relerr": float(r["gl_shift_p"]) / float(r["p_ref"]) - 1.0,
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
    lines = ["# Missed end-mode share s vs single-shift error (30 routes; signed errors vs quadrature reference)", "",
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

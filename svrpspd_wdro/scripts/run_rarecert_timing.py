"""RARECERT round 2: wall-clock timing on an idle machine, one core, one task at a time.

For six routes of the baseline set (3 Northern Vietnam, 3 Guangdong) at gamma = 1e-4, at the
threshold c of results/rarecert/baselines_routes.csv, measure (wall clock, time.perf_counter):

  smis        build + sampling N draws; per-draw time from the sample, RV from the sample
  ce_mix      build (default CE mixture: pilot included) + sampling
  ce_single   build + sampling
  gl_shift    Glasserman-Li shifted inner step, N evaluations
  quad_fine   quadrature reference, 41 nodes, V = (256, 512)
  quad_mid    quadrature reference, 41 nodes, V = (128, 256)

Time to relative precision eps = 0.01 at 95%: build + 1.96^2 RV / eps^2 x per-draw time
(quadrature: its own wall time). The machine load (1-minute load average) is recorded before
each task; run it with nothing else running.

Usage (from svrpspd_wdro/): python3 scripts/run_rarecert_timing.py [--n 200000] [--out DIR]
Output: <out>/timing.csv, timing_summary.md
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_rarecert_r3 as r3  # noqa: E402
from quantum.rare import ce_mixture, ce_shift, is_mixture, is_shift  # noqa: E402
from rarecert.baselines import exact_reference, gl_two_step  # noqa: E402
from rarecert.smis import smis_mixture  # noqa: E402

RES = ROOT / "results" / "rarecert"
GAMMA = 1e-4
EPS = 0.01
Z2 = 1.96 ** 2


def pick_routes():
    rows = [r for r in csv.DictReader(open(RES / "baselines_routes.csv"))
            if not r["error"] and math.isclose(float(r["gamma"]), GAMMA, rel_tol=1e-6)]
    vn = [r for r in rows if r["route_id"].startswith("VNN")][:3]
    gd = [r for r in rows if r["route_id"].startswith("GD")][:3]
    return vn + gd


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--out", default=str(RES / "timing"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    means = r3.rs.route_means()
    rows = []
    for rr in pick_routes():
        rid, c = rr["route_id"], float(rr["c"])
        sim = r3._sim_for(rid, means)
        rng = np.random.default_rng([97, int(float(rr["c"])) % 10000])
        load = os.getloadavg()[0]
        row = {"route_id": rid, "L": sim.L, "c": c, "load1": load}

        (mix, diag), t_b = timed(lambda: smis_mixture(sim, c, eta=r3.ETA, n_arc=0))
        (p, RV, rse, _), t_s = timed(lambda: is_mixture(sim, c, mix, args.n, rng))
        row.update({"smis_build": t_b, "smis_per_draw": t_s / args.n, "smis_RV": RV, "smis_p": p})
        row["smis_t"] = t_b + Z2 * RV / EPS ** 2 * t_s / args.n

        cem, t_b = timed(lambda: ce_mixture(sim, c, rng))
        (p, RV, rse, _), t_s = timed(lambda: is_mixture(sim, c, cem, args.n, rng))
        row.update({"ce_build": t_b, "ce_per_draw": t_s / args.n, "ce_RV": RV, "ce_p": p, "ce_pilot": cem.draws})
        row["ce_t"] = t_b + Z2 * RV / EPS ** 2 * t_s / args.n

        theta, t_b = timed(lambda: ce_shift(sim, c, rng))
        (p, RV, rse, _), t_s = timed(lambda: is_shift(sim, c, theta, args.n, rng))
        row.update({"single_build": t_b, "single_RV": RV, "single_p": p})
        row["single_t"] = t_b + Z2 * RV / EPS ** 2 * t_s / args.n

        (p, RV, rse), t_s = timed(lambda: gl_two_step(sim, c, args.n, rng, variant="shift"))
        row.update({"gl_per_draw": t_s / args.n, "gl_RV": RV, "gl_p": p})
        row["gl_t"] = Z2 * RV / EPS ** 2 * t_s / args.n

        for tag, n_grid, vp in (("quad_fine", 41, (256, 512)), ("quad_mid", 41, (128, 256))):
            ref, t_q = timed(lambda: exact_reference(sim, c, n=n_grid, V_pair=vp))
            row.update({f"{tag}_p": ref["p_ext"], f"{tag}_t": t_q})
        rows.append(row)
        print(rid, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k.endswith("_t")},
              flush=True)

    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(out / "timing.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    L = ["# Timing on an idle machine (one core, one task at a time)\n",
         f"gamma = {GAMMA:g}, eps = {EPS:g}, 95% confidence, N = {args.n:,} draws for the per-draw time; "
         "t = build + 1.96^2 RV / eps^2 x per-draw time (quadrature: its wall time).\n",
         "| route | L | load1 | SMIS | CE mixture | single CE shift | GL shifted | quad (41, V 128/256) | quad (41, V 256/512) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['route_id']} | {r['L']} | {r['load1']:.2f} | {r['smis_t']:.2f} | {r['ce_t']:.2f} | "
                 f"{r['single_t']:.2f} | {r['gl_t']:.2f} | {r['quad_mid_t']:.1f} | {r['quad_fine_t']:.1f} |")
    med = lambda k: float(np.median([r[k] for r in rows]))  # noqa: E731
    L.append(f"| median | | | {med('smis_t'):.2f} | {med('ce_t'):.2f} | {med('single_t'):.2f} | {med('gl_t'):.2f} | "
             f"{med('quad_mid_t'):.1f} | {med('quad_fine_t'):.1f} |")
    (out / "timing_summary.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "timing_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

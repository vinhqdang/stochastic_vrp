#!/usr/bin/env python3
"""
validate_otr21.py — Does the enriched statistic (OTR-2.1) beat scalar-W
OTR-2.0? Paired comparison over real routes before any adoption.

For every route (with material overflow risk) of several instances across
the three benchmark families, fit both policies on the same training
scenarios and score on the same large test set under realistic per-stop
prices. Report per-route paired differences, Wilcoxon, and the DP-50k
anchor for context.
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

_SCRIPTS = Path(__file__).resolve().parent
_WDRO    = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_SCRIPTS))

from dethloff_runner import (parse_dethloff, sample_demands, solve_fast,
                             DetGate, InflationGate, CV, DIST, SEED)
from core.otr2 import calibrate_B_empirical_peak
from core.costs import (LastMileCosts, route_cost_schedules,
                        fit_lsm_general, simulate_v2_general)
from core.dp_exec import fit_dp
from core.otr21 import fit_lsm_rich, simulate_rich

INSTANCES = [
    ("data/Dethloff/CON3-0.vrpspd", "det"),
    ("data/Dethloff/SCA3-0.vrpspd", "det"),
    ("data/SalhiNagy/CMT3X.vrpspd", "det"),
    ("data/SalhiNagy/CMT5Y.vrpspd", "det"),
    ("data/City/HANOI-100-1.vrpspd", "gnrs"),
    ("data/City/HCMC-200-1.vrpspd", "gnrs"),
]
N_TRAIN, N_TEST = 2000, 10000


def eval_instance(path, gate_kind):
    D, dem, Q, n, scale = parse_dethloff(str(_WDRO / path))
    dbar = dem[:, 0].astype(float)
    pbar = dem[:, 1].astype(float)
    gate = InflationGate(Q, dbar, pbar, alpha=0.10) if gate_kind == "gnrs" \
        else DetGate(Q, dbar, pbar)
    plan = solve_fast(D, gate, n)

    seed = SEED + abs(hash(Path(path).stem)) % 10_000
    rng = np.random.default_rng(seed)
    dsc_tr = sample_demands(dbar, n, N_TRAIN, CV, DIST, rng)
    psc_tr = sample_demands(pbar, n, N_TRAIN, CV, DIST, rng)
    rng2 = np.random.default_rng(seed + 99_991)
    dsc_te = sample_demands(dbar, n, N_TEST, CV, DIST, rng2)
    psc_te = sample_demands(pbar, n, N_TEST, CV, DIST, rng2)

    costs = LastMileCosts()
    rows = []
    for route in plan:
        if len(route) < 5:
            continue
        r = np.array(route)
        B = float(Q - dbar[r].sum())
        if B <= 0:
            B = calibrate_B_empirical_peak(psc_tr[:, r] - dsc_tr[:, r], 0.10)
        g_tr = psc_tr[:, r] - dsc_tr[:, r]
        g_te = psc_te[:, r] - dsc_te[:, r]
        if float((np.cumsum(g_te, 1).max(1) > B).mean()) < 0.01:
            continue
        H, E = route_cost_schedules(route, D, scale, costs)
        # per-customer net-increment moments (model-free plug-ins)
        mu = g_tr.mean(axis=0)
        sig = g_tr.std(axis=0)

        t0 = time.time()
        cm = fit_lsm_general(g_tr, B, H, E)
        c_v2 = simulate_v2_general(g_te, B, H, E, cm)["mean_cost"]
        t_v2 = time.time() - t0

        t0 = time.time()
        rich = fit_lsm_rich(g_tr, mu, sig, B, H, E)
        c_21 = simulate_rich(g_te, mu, sig, B, H, E, rich)["mean_cost"]
        t_21 = time.time() - t0

        rows.append((Path(path).stem, len(route), c_v2, c_21, t_v2, t_21))
    return rows


def main():
    all_rows = []
    for path, gk in INSTANCES:
        try:
            rows = eval_instance(path, gk)
            all_rows.extend(rows)
            for nm, m, c2, c21, t2, t21 in rows:
                print(f"  {nm:<12} m={m:<3} v2={c2:8.3f}  v2.1={c21:8.3f}  "
                      f"diff={(c2 - c21):+7.3f}  fit_s v2={t2:.1f} v2.1={t21:.1f}",
                      flush=True)
        except Exception as e:
            print(f"  {path}: ERROR {e}", flush=True)
    if not all_rows:
        print("no qualifying routes")
        return
    d = np.array([r[2] - r[3] for r in all_rows])   # v2 - v2.1 (positive = 2.1 wins)
    rel = d / np.maximum(np.array([r[2] for r in all_rows]), 1e-9)
    p = sps.wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else 1.0
    print(f"\nroutes={len(d)}  mean diff={d.mean():+.3f}$ "
          f"({100 * rel.mean():+.2f}%)  v2.1 wins {int((d > 0).sum())}/{len(d)}  "
          f"Wilcoxon(v2.1 better) p={p:.4f}")


if __name__ == "__main__":
    main()

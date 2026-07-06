#!/usr/bin/env python3
"""
run_mip_cert.py — Certify ALNS planning quality with an exact MIP (HiGHS).

Solves the deterministic VRPSPD planning problem with the classical
two-commodity flow formulation of Montane and Galvao (2006):

    min  sum_ij c_ij x_ij  +  omega_V * K
    s.t. every customer has in/out degree 1;  K = arcs leaving the depot;
         delivery flow y (from depot) consumed at customers,
         pickup   flow z (to depot)   generated at customers,
         y_ij + z_ij <= Q x_ij   on every arc (the on-board load).

y + z on an arc equals deliveries-still-onboard + pickups-collected, i.e.
exactly the Model-A load the DetGate's nominal-peak check enforces, so the
MIP and the ALNS planner optimise the same objective over the same
feasible set. Flow conservation eliminates subtours (all demands > 0).

For each instance we report the ALNS plan objective, the MIP incumbent
and dual bound at the time limit, and the resulting certified gaps:

    gap_alns = (ALNS - LB) / ALNS      certified suboptimality of our plan
    gap_mip  = (UB   - LB) / UB        the solver's own residual gap

Modes:
    dethloff   all 40 Dethloff instances (n=50; bounds, rarely optimal)
    small      sub-instances of HCMC-100 (n=12..24; provable optimality)

Usage:
    python scripts/run_mip_cert.py dethloff [tlim=300] [max=N] [workers=3]
    python scripts/run_mip_cert.py small    [tlim=120]
"""

import os
import sys
import glob
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

_SCRIPTS = Path(__file__).resolve().parent
_WDRO    = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_SCRIPTS))

from dethloff_runner import (parse_dethloff, solve_instance, solve_fast,
                             DetGate, route_cost, NO_IMPROVE)

RESULTS_DIR = _WDRO / "results"
PLANS_DIR   = RESULTS_DIR / "plans"


# ═══════════════════════════════════════════════════════════════════════════════
# MIP
# ═══════════════════════════════════════════════════════════════════════════════

def solve_vrpspd_mip(D: np.ndarray, deliv: np.ndarray, pick: np.ndarray,
                     Q: float, omega_V: float, tlim: float,
                     warm_obj: float | None = None,
                     solver: str = "highs") -> dict:
    """Two-commodity flow VRPSPD MIP. deliv/pick are full node vectors
    (index 0 = depot, zeros). Returns incumbent objective, dual bound,
    solver status and wall time."""
    n = D.shape[0]
    A = [(i, j) for i in range(n) for j in range(n) if i != j]

    m = pulp.LpProblem("vrpspd", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", A, cat="Binary")
    y = pulp.LpVariable.dicts("y", A, lowBound=0)
    z = pulp.LpVariable.dicts("z", A, lowBound=0)

    m += (pulp.lpSum(float(D[i, j]) * x[i, j] for (i, j) in A)
          + omega_V * pulp.lpSum(x[0, j] for j in range(1, n)))

    for i in range(1, n):
        m += pulp.lpSum(x[i, j] for j in range(n) if j != i) == 1
        m += pulp.lpSum(x[j, i] for j in range(n) if j != i) == 1
    m += (pulp.lpSum(x[0, j] for j in range(1, n))
          == pulp.lpSum(x[j, 0] for j in range(1, n)))

    for i in range(1, n):
        # deliveries flow out of the depot and are consumed at i
        m += (pulp.lpSum(y[j, i] for j in range(n) if j != i)
              - pulp.lpSum(y[i, j] for j in range(n) if j != i)
              == float(deliv[i]))
        # pickups are generated at i and flow to the depot
        m += (pulp.lpSum(z[i, j] for j in range(n) if j != i)
              - pulp.lpSum(z[j, i] for j in range(n) if j != i)
              == float(pick[i]))
    # nothing delivered flows back into the depot; nothing picked up leaves it
    for j in range(1, n):
        m += y[j, 0] == 0
        m += z[0, j] == 0

    for (i, j) in A:
        m += y[i, j] + z[i, j] <= Q * x[i, j]

    t0 = time.time()
    if solver == "gurobi":
        s = pulp.GUROBI(msg=False, timeLimit=tlim)
    else:
        s = pulp.HiGHS(msg=False, timeLimit=tlim)
    m.solve(s)
    wall = time.time() - t0

    status = pulp.LpStatus[m.status]
    ub = pulp.value(m.objective) if m.status in (pulp.LpStatusOptimal,) or \
        pulp.value(m.objective) is not None else float("nan")
    # dual bound via the underlying solver handle (PuLP doesn't surface it)
    lb = float("nan")
    try:
        if solver == "gurobi":
            lb = m.solverModel.ObjBound                  # gurobipy handle
        else:
            lb = m.solverModel.getInfo().mip_dual_bound  # highspy handle
    except Exception:
        if status == "Optimal":
            lb = ub
    return {"ub": float(ub) if ub is not None else float("nan"),
            "lb": float(lb), "status": status, "time": wall}


# ═══════════════════════════════════════════════════════════════════════════════
# ALNS objective for comparison
# ═══════════════════════════════════════════════════════════════════════════════

def _plan_objective(plan: list, D: np.ndarray, omega_V: float) -> float:
    return sum(route_cost(r, D) for r in plan if r) + \
        omega_V * sum(1 for r in plan if r)


def _load_plan(name: str) -> list | None:
    p = PLANS_DIR / f"{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d["res"].get("Det", {}).get("plan")


# ═══════════════════════════════════════════════════════════════════════════════
# Modes
# ═══════════════════════════════════════════════════════════════════════════════

def _cert_one(path: str, tlim: float, solver: str = "highs") -> dict:
    D, dem, Q, n, scale = parse_dethloff(path)
    name = Path(path).stem
    deliv = np.concatenate([[0.0], dem[1:, 0]]) if dem.shape[0] == n else dem[:, 0]
    # dem is (n,2) with depot row zero in our files
    deliv = dem[:, 0].astype(float)
    pick  = dem[:, 1].astype(float)
    omega_V = float(np.mean(D[D > 0]))

    plan = _load_plan(name)
    if plan is None:
        sol = solve_instance(path, 60.0, NO_IMPROVE, use_prune=True, which=["Det"])
        plan = sol["res"]["Det"]["plan"]
    alns_obj = _plan_objective(plan, D, omega_V)

    mip = solve_vrpspd_mip(D, deliv, pick, Q, omega_V, tlim,
                           warm_obj=alns_obj, solver=solver)
    lb = mip["lb"]
    row = {
        "Instance": name, "n_cust": n - 1, "Q": Q,
        "ALNS_obj": round(alns_obj, 1),
        "MIP_UB": round(mip["ub"], 1) if np.isfinite(mip["ub"]) else np.nan,
        "MIP_LB": round(lb, 1) if np.isfinite(lb) else np.nan,
        "MIP_status": mip["status"],
        "gap_alns_pct": round(100 * (alns_obj - lb) / alns_obj, 2)
                        if np.isfinite(lb) else np.nan,
        "gap_mip_pct": round(100 * (mip["ub"] - lb) / mip["ub"], 2)
                       if np.isfinite(lb) and np.isfinite(mip["ub"]) else np.nan,
        "mip_s": round(mip["time"], 1),
    }
    return row


def run_files(files: list, tlim: float, workers: int, out_stem: str,
              solver: str = "highs"):
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_cert_one, f, tlim, solver): Path(f).stem
                for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
                rows.append(r)
                print(f"[{i}/{len(files)}] {r['Instance']:<12} "
                      f"ALNS={r['ALNS_obj']:>9.1f}  UB={r['MIP_UB']}  "
                      f"LB={r['MIP_LB']}  gap_alns={r['gap_alns_pct']}%  "
                      f"({r['MIP_status']}, {r['mip_s']}s)", flush=True)
            except Exception as e:
                print(f"[{i}/{len(files)}] {futs[fut]} ERROR: {e}", flush=True)
    if rows:
        df = pd.DataFrame(rows).sort_values("Instance")
        RESULTS_DIR.mkdir(exist_ok=True)
        df.to_csv(RESULTS_DIR / f"{out_stem}.csv", index=False)
        ok = df["gap_alns_pct"].dropna()
        print(f"\nmean certified ALNS gap: {ok.mean():.2f}%   "
              f"max: {ok.max():.2f}%   instances with LB: {len(ok)}/{len(df)}")
        print(f"wrote {RESULTS_DIR / (out_stem + '.csv')} "
              f"({(time.time()-t0)/60:.1f} min total)")


def run_small(tlim: float):
    """Provable-optimality check: sub-instances of HCMC-100 with the SAME
    demand/capacity structure, sized so HiGHS closes the gap."""
    src = _WDRO / "data" / "City" / "HCMC-100-1.vrpspd"
    D, dem, Q, n, scale = parse_dethloff(str(src))
    rows = []
    for n_sub in (12, 16, 20, 24):
        idx = np.concatenate([[0], np.arange(1, n_sub + 1)])
        Ds = D[np.ix_(idx, idx)]
        dems = dem[idx]
        omega_V = float(np.mean(Ds[Ds > 0]))
        gate = DetGate(Q, dems[:, 0].astype(float), dems[:, 1].astype(float))
        plan = solve_fast(Ds, gate, n_sub + 1)
        alns_obj = _plan_objective(plan, Ds, omega_V)
        mip = solve_vrpspd_mip(Ds, dems[:, 0].astype(float),
                               dems[:, 1].astype(float), Q, omega_V, tlim)
        gap = 100 * (alns_obj - mip["lb"]) / alns_obj if np.isfinite(mip["lb"]) else np.nan
        rows.append({"n_cust": n_sub, "CW2opt_obj": round(alns_obj, 1),
                     "MIP_UB": round(mip["ub"], 1), "MIP_LB": round(mip["lb"], 1),
                     "status": mip["status"], "gap_pct": round(gap, 2),
                     "mip_s": round(mip["time"], 1)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "results_mip_small.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'results_mip_small.csv'}")


def main():
    args = sys.argv[1:]
    mode = args[0] if args else "dethloff"
    tlim, max_n, workers = 300.0, None, min(3, os.cpu_count() or 1)
    solver, out_stem = "highs", None
    for a in args[1:]:
        if a.startswith("tlim="):    tlim    = float(a[5:])
        elif a.startswith("max="):     max_n   = int(a[4:])
        elif a.startswith("workers="): workers = int(a[8:])
        elif a.startswith("solver="):  solver  = a[7:]
        elif a.startswith("out="):     out_stem = a[4:]

    if mode == "small":
        run_small(tlim)
        return
    files = sorted(glob.glob(str(_WDRO / "data" / "Dethloff" / "*.vrpspd")))
    if max_n:
        files = files[:max_n]
    out_stem = out_stem or ("results_mip_cert_gurobi" if solver == "gurobi"
                            else "results_mip_cert")
    print(f"MIP certification ({solver}): {len(files)} instances, "
          f"tlim={tlim}s, workers={workers}")
    run_files(files, tlim, workers, out_stem, solver)


if __name__ == "__main__":
    main()

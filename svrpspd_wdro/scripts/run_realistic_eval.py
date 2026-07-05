#!/usr/bin/env python3
"""
run_realistic_eval.py — Dethloff evaluation under the realistic last-mile
cost model (core/costs.py): contracted fleet day-rates, variable km costs,
standby-pool planned handoffs priced on the remaining route, and emergency
surge vehicles with SLA fallout for downstream customers.

For each instance and each ALNS planning policy (Det/SAA/WDRO):
  1. Solve with ALNS (or load a persisted plan from results/plans/).
  2. For every route, build per-stop handoff/emergency price schedules from
     the real route geometry (distance matrix).
  3. Fit and score execution policies on independent train/test scenarios:
       none     reactive: never hand off, pay the emergency at the breach
       v1_end   endpoint-label isotonic curves + tau tuned on realized cost
       v1_myo   peak-label curves + myopic tau = mean(H)/mean(E)
       fb_tau   peak-label curves + tau tuned on realized cost
       v2_lsm   OTR-2.0 generalized LSM (per-stop cost comparison, no tau)
       oracle   clairvoyant lower bound
  4. Report realistic TBC = F_plan*K + c_km*Travel + E[recourse], savings
     vs reactive, and each policy's share of the oracle's achievable saving.

Usage (from svrpspd_wdro/ or repo root):
    python scripts/run_realistic_eval.py [key=value ...]

Options:
    dir=<path>    tlim=<s>     n_train=<N>  n_test=<N>
    workers=<N>   max=<N>      out=<stem>   policies=Det,SAA,WDRO
    reuse=0|1     Load persisted plans if available (default 1)
"""

import os
import sys
import glob
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_WDRO    = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_SCRIPTS))

from core.otr import fit_otr
from core.otr2 import calibrate_B_empirical_peak, fit_otr_peak
from core.dp_exec import fit_dp
from core.costs import (
    LastMileCosts,
    route_cost_schedules,
    plan_fixed_cost,
    fit_lsm_general,
    simulate_v2_general,
    simulate_tau_general,
    tune_tau_general,
    oracle_costs_general,
)
from dethloff_runner import (
    parse_dethloff, sample_demands, solve_instance,
    ALPHA, CV, DIST, SEED, NO_IMPROVE,
)

RESULTS_DIR = _WDRO / "results"
PLANS_DIR   = RESULTS_DIR / "plans"

POLICY_LABELS = ["none", "v1_end", "v1_myo", "fb_tau", "v2_lsm",
                 "dp_n", "dp_xl", "oracle"]

N_XL = 50_000       # training scenarios for the near-exact DP anchor


# ═══════════════════════════════════════════════════════════════════════════════
# Plan persistence
# ═══════════════════════════════════════════════════════════════════════════════

def _plan_path(name: str) -> Path:
    return PLANS_DIR / f"{name}.json"


def _save_plans(sol: dict) -> None:
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": sol["name"], "n": sol["n"], "Q": sol["Q"],
        "omega_V": sol["omega_V"],
        "dbar": sol["dbar"].tolist(), "pbar": sol["pbar"].tolist(),
        "res": {p: {"plan": sol["res"][p]["plan"],
                    "K": sol["res"][p]["K"],
                    "dist": sol["res"][p]["dist"]}
                for p in sol["res"]},
    }
    _plan_path(sol["name"]).write_text(json.dumps(payload))


def _load_plans(name: str) -> dict | None:
    p = _plan_path(name)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["dbar"] = np.array(d["dbar"], dtype=float)
    d["pbar"] = np.array(d["pbar"], dtype=float)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# Per-instance runner (worker process)
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_scenarios(dbar, pbar, N, seed):
    n = len(dbar)
    rng = np.random.default_rng(seed)
    dsc = sample_demands(dbar, n, N, CV, DIST, rng)
    psc = sample_demands(pbar, n, N, CV, DIST, rng)
    return dsc, psc


def _eval_route_realistic(route, dbar, Q, D, scale, costs,
                          dsc_tr, psc_tr, dsc_te, psc_te,
                          dsc_xl, psc_xl):
    r = np.array(route)
    g_train = psc_tr[:, r] - dsc_tr[:, r]
    g_test  = psc_te[:, r] - dsc_te[:, r]
    g_xl    = psc_xl[:, r] - dsc_xl[:, r]

    L0 = float(dbar[r].sum())
    B  = float(Q - L0)
    if B <= 0.0:
        B = calibrate_B_empirical_peak(g_train, alpha=1.0 - ALPHA)

    H, E = route_cost_schedules(route, D, scale, costs)

    v1_end_models = fit_otr(g_train, B)          # endpoint label (the v1 bug)
    fb_models     = fit_otr_peak(g_train, B)     # peak label
    tau_myo = float(H.mean() / max(E.mean(), 1e-9))
    tau_end = tune_tau_general(g_train, B, H, E, v1_end_models) if v1_end_models else 1.0
    tau_fb  = tune_tau_general(g_train, B, H, E, fb_models) if fb_models else 1.0
    cm      = fit_lsm_general(g_train, B, H, E)

    # plug-in DP benchmarks: equal data budget, and near-exact 50k anchor
    dp_same = fit_dp(g_train, B, H, E)
    dp_xl   = fit_dp(g_xl,    B, H, E)

    orc = oracle_costs_general(g_test, B, H, E)
    return {
        "none":   simulate_tau_general(g_test, B, H, E, fb_models, tau=1.0),
        "v1_end": simulate_tau_general(g_test, B, H, E, v1_end_models, tau=tau_end),
        "v1_myo": simulate_tau_general(g_test, B, H, E, fb_models, tau=tau_myo),
        "fb_tau": simulate_tau_general(g_test, B, H, E, fb_models, tau=tau_fb),
        "v2_lsm": simulate_v2_general(g_test, B, H, E, cm),
        "dp_n":   simulate_v2_general(g_test, B, H, E, dp_same),
        "dp_xl":  simulate_v2_general(g_test, B, H, E, dp_xl),
        "oracle": {"mean_cost": float(orc.mean()),
                   "handoff_rate": float((orc > 0).mean()),
                   "fail_rate": 0.0, "complete_rate": float((orc == 0).mean())},
    }


def _run_instance(path, tlim, n_train, n_test, active_policies, reuse):
    log = []
    name = Path(path).stem
    costs = LastMileCosts()

    D, _dem, Q, n, scale = parse_dethloff(path)

    sol = _load_plans(name) if reuse else None
    if sol is not None and not all(p in sol["res"] for p in active_policies):
        sol = None                      # persisted plans lack a requested gate
    solve_time = 0.0
    if sol is None:
        t0 = time.time()
        full = solve_instance(path, tlim, NO_IMPROVE, use_prune=True,
                              which=active_policies)
        solve_time = time.time() - t0
        _save_plans(full)
        sol = _load_plans(name)

    dbar, pbar = sol["dbar"], sol["pbar"]
    inst_seed = SEED + abs(hash(name)) % 10_000

    dsc_tr, psc_tr = _gen_scenarios(dbar, pbar, n_train, inst_seed)
    dsc_te, psc_te = _gen_scenarios(dbar, pbar, n_test,  inst_seed + 99_991)
    dsc_xl, psc_xl = _gen_scenarios(dbar, pbar, N_XL,    inst_seed + 424_243)

    rows = []
    for policy in active_policies:
        pdata = sol["res"][policy]
        plan, K, dist = pdata["plan"], pdata["K"], pdata["dist"]

        t0 = time.time()
        agg = {lbl: [] for lbl in POLICY_LABELS}
        ho  = {lbl: [] for lbl in POLICY_LABELS}
        fl  = {lbl: [] for lbl in POLICY_LABELS}
        for route in plan:
            if not route:
                continue
            res = _eval_route_realistic(route, dbar, Q, D, scale, costs,
                                        dsc_tr, psc_tr, dsc_te, psc_te,
                                        dsc_xl, psc_xl)
            for lbl in POLICY_LABELS:
                agg[lbl].append(res[lbl]["mean_cost"])
                ho[lbl].append(res[lbl]["handoff_rate"])
                fl[lbl].append(res[lbl]["fail_rate"])
        eval_time = time.time() - t0

        fixed = plan_fixed_cost(K, dist, costs)
        none_rec = float(sum(agg["none"]))
        orc_rec  = float(sum(agg["oracle"]))

        row = {
            "Instance": name, "Plan": policy, "N_cust": n - 1, "K_routes": K,
            "Travel_km": round(dist, 2),
            "Fixed_cost": round(fixed, 2),
        }
        for lbl in POLICY_LABELS:
            rec = float(sum(agg[lbl]))
            row[f"{lbl}_rec"]  = round(rec, 2)
            row[f"{lbl}_TBC"]  = round(fixed + rec, 2)
            row[f"{lbl}_HO"]   = round(float(np.mean(ho[lbl])), 4)
            row[f"{lbl}_fail"] = round(float(np.mean(fl[lbl])), 4)
            if lbl != "none":
                row[f"{lbl}_saving"] = round(
                    100.0 * (none_rec - rec) / max(none_rec, 1e-9), 2)
            if lbl not in ("none", "oracle"):
                row[f"{lbl}_gap"] = round(
                    100.0 * (rec - orc_rec) / max(none_rec - orc_rec, 1e-9), 2)
        row["solve_s"] = round(solve_time, 1)
        row["eval_s"]  = round(eval_time, 1)
        rows.append(row)

        log.append(
            f"  {policy:<5} K={K}  rec$: none={none_rec:.0f} "
            f"v1_end={row['v1_end_rec']:.0f} fb={row['fb_tau_rec']:.0f} "
            f"v2={row['v2_lsm_rec']:.0f} dp_n={row['dp_n_rec']:.0f} "
            f"dp_xl={row['dp_xl_rec']:.0f} orc={orc_rec:.0f}  "
            f"save%(v1e/fb/v2/dpn/dpxl)={row['v1_end_saving']:.1f}/"
            f"{row['fb_tau_saving']:.1f}/{row['v2_lsm_saving']:.1f}/"
            f"{row['dp_n_saving']:.1f}/{row['dp_xl_saving']:.1f}"
        )
    return rows, "\n".join(log)


# ═══════════════════════════════════════════════════════════════════════════════

def _print_summary(df, policies):
    W = 120
    print("\n" + "=" * W)
    print("  SUMMARY — realistic last-mile costs, means across instances")
    print("=" * W)
    print(f"  {'Plan':<6}{'Fixed$':>9}{'none TBC':>10}"
          f"{'sv% v1e':>9}{'sv% fb':>8}{'sv% v2':>8}"
          f"{'sv% dp_n':>9}{'sv% dp_xl':>10}{'sv% orc':>8}"
          f"{'gap% v2':>9}{'gap% dp_n':>10}{'gap% dp_xl':>11}")
    print("  " + "-" * (W - 2))
    for policy in policies:
        sub = df[df["Plan"] == policy]
        if sub.empty:
            continue
        print(f"  {policy:<6}"
              f"{sub['Fixed_cost'].mean():>9.0f}"
              f"{sub['none_TBC'].mean():>10.0f}"
              f"{sub['v1_end_saving'].mean():>9.1f}"
              f"{sub['fb_tau_saving'].mean():>8.1f}"
              f"{sub['v2_lsm_saving'].mean():>8.1f}"
              f"{sub['dp_n_saving'].mean():>9.1f}"
              f"{sub['dp_xl_saving'].mean():>10.1f}"
              f"{sub['oracle_saving'].mean():>8.1f}"
              f"{sub['v2_lsm_gap'].mean():>9.1f}"
              f"{sub['dp_n_gap'].mean():>10.1f}"
              f"{sub['dp_xl_gap'].mean():>11.1f}")
    print("\n  rec$/TBC in cost units of core/costs.py defaults; "
          "saving% on the recourse term vs reactive")
    print("  gap% = share of the oracle-achievable saving that the policy misses")


def main():
    data_dir  = str(_WDRO / "data" / "Dethloff")
    tlim      = 60.0
    n_train   = 1000
    n_test    = 2000
    max_n     = None
    out_stem  = "results_realistic_eval"
    policies  = ["Det", "SAA", "WDRO"]
    n_workers = os.cpu_count() or 1
    reuse     = True

    for arg in sys.argv[1:]:
        if   arg.startswith("dir="):      data_dir  = arg[4:]
        elif arg.startswith("tlim="):     tlim      = float(arg[5:])
        elif arg.startswith("n_train="):  n_train   = int(arg[8:])
        elif arg.startswith("n_test="):   n_test    = int(arg[7:])
        elif arg.startswith("max="):      max_n     = int(arg[4:])
        elif arg.startswith("out="):      out_stem  = arg[4:]
        elif arg.startswith("workers="):  n_workers = int(arg[8:])
        elif arg.startswith("policies="): policies  = arg[9:].split(",")
        elif arg.startswith("reuse="):    reuse     = bool(int(arg[6:]))
        else:
            print(f"  Unknown argument '{arg}' — ignored")

    files = sorted(glob.glob(str(Path(data_dir) / "*.vrpspd"))) or \
            sorted(glob.glob(str(Path(data_dir) / "*.txt")))
    if not files:
        print(f"ERROR: no instances in '{data_dir}'")
        return
    if max_n:
        files = files[:max_n]
    n_workers = min(n_workers, len(files))

    c = LastMileCosts()
    W = 100
    print("=" * W)
    print("  REALISTIC LAST-MILE EVALUATION — state-dependent recourse prices on Dethloff SVRPSPD")
    print("=" * W)
    print(f"  instances={len(files)}  policies={policies}  workers={n_workers}  reuse_plans={reuse}")
    print(f"  fleet: F_plan=${c.F_plan}/veh-day  c_km=${c.c_km}/km")
    print(f"  handoff: F_ho=${c.F_ho} + {c.s_ho}x km + ${c.c_transfer} transfer   "
          f"emergency: F_emg=${c.F_emg} + {c.s_emg}x km + ${c.p_late}/late cust + ${c.p_breach} churn")
    print("-" * W)

    all_rows = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_run_instance, f, tlim, n_train, n_test,
                        policies, reuse): Path(f).stem
            for f in files
        }
        done = 0
        for fut in as_completed(futures):
            stem = futures[fut]
            done += 1
            try:
                rows, log = fut.result()
                all_rows.extend(rows)
                print(f"\n[{done}/{len(files)}] {stem}")
                print(log, flush=True)
            except Exception as exc:
                print(f"\n[{done}/{len(files)}] {stem} — ERROR: {exc}")
                traceback.print_exc()

    print(f"\n{'-' * W}\n  Total wall time: {(time.time() - t_start) / 60:.1f} min")
    if not all_rows:
        return

    df = pd.DataFrame(all_rows)
    _print_summary(df, policies)

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / (out_stem + ".csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Wrote {csv_path}")


if __name__ == "__main__":
    main()

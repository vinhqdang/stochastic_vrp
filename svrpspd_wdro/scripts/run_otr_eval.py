#!/usr/bin/env python3
"""
run_otr_eval.py — Full evaluation of OTR on all Dethloff VRPSPD instances.

For each of the 40 Dethloff instances the script:
  1. Solves with ALNS under three planning policies: Det / SAA / WDRO.
  2. Generates independent train and test demand scenarios.
  3. For every route in each plan, fits OTR models and simulates execution
     under three threshold policies: tuned tau, myopic tau, no-handoff.
  4. Aggregates per-route stats to plan level and writes a CSV + Excel report.

Parallelism:
  - Instances run concurrently via ProcessPoolExecutor (one process per CPU).
  - Per-route OTR evaluation uses vectorized numpy simulation (~70x speedup
    over the scalar loop), eliminating the tune_tau bottleneck.

Usage (run from svrpspd_wdro/ or from the repo root):
    python scripts/run_otr_eval.py [key=value ...]

Options:
    dir=<path>       Dethloff folder          (default: data/Dethloff)
    tlim=<s>         ALNS time limit/policy   (default: 60)
    n_train=<N>      Train scenarios/route    (default: 1000)
    n_test=<N>       Test  scenarios/route    (default: 2000)
    cfail=<ratio>    Cfail / omegaF           (default: 5.0)
    policies=p,p,p   ALNS policies to run     (default: Det,SAA,WDRO)
    workers=<N>      Parallel processes       (default: all CPUs)
    max=<N>          Cap number of instances  (default: all)
    out=<stem>       Output filename stem     (default: results_otr_eval)
"""

import os
import sys
import glob
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── import paths ─────────────────────────────────────────────────────────────
# Works whether invoked from svrpspd_wdro/ or from the repo root.
_SCRIPTS = Path(__file__).resolve().parent
_WDRO    = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))      # exposes core.*
sys.path.insert(0, str(_SCRIPTS))   # exposes dethloff_runner

from core.otr import (
    fit_otr,
    calibrate_B,
    calibrate_B_empirical,
    tau_myopic,
    tune_tau_fast,
    simulate_fast,
)
from dethloff_runner import (
    sample_demands,
    solve_instance,
    eval_evextra,
    ALPHA, CV, DIST, SEED, OMEGA_RATIO, NO_IMPROVE,
)

# ── run-level defaults ────────────────────────────────────────────────────────
_DEFAULT_DATA_DIR    = str(_WDRO / "data" / "Dethloff")
_DEFAULT_TLIM        = 60.0
_DEFAULT_N_TRAIN     = 1000
_DEFAULT_N_TEST      = 2000
_DEFAULT_CFAIL_RATIO = 5.0
_DEFAULT_POLICIES    = ["Det", "SAA", "WDRO"]
_DEFAULT_OUT_STEM    = "results_otr_eval"


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_scenarios(dbar: np.ndarray, pbar: np.ndarray, N: int, seed: int):
    """Return (dsc, psc) each shape (N, n_nodes)."""
    n = len(dbar)
    rng = np.random.default_rng(seed)
    dsc = sample_demands(dbar, n, N, CV, DIST, rng)
    psc = sample_demands(pbar, n, N, CV, DIST, rng)
    return dsc, psc


def _route_g(dsc: np.ndarray, psc: np.ndarray, route: list) -> np.ndarray:
    """Net-increment matrix for one route. Shape (N, m)."""
    r = np.array(route)
    return psc[:, r] - dsc[:, r]


# ═══════════════════════════════════════════════════════════════════════════════
# Per-route OTR evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _eval_route(
    route:   list,
    dbar:    np.ndarray,
    Q:       float,
    dsc_tr:  np.ndarray,
    psc_tr:  np.ndarray,
    dsc_te:  np.ndarray,
    psc_te:  np.ndarray,
    omegaF:  float,
    Cfail:   float,
) -> dict:
    """Fit OTR on training scenarios and simulate on test scenarios for one route.

    Uses tune_tau_fast and simulate_fast (vectorized numpy) for speed.
    """
    g_train = _route_g(dsc_tr, psc_tr, route)
    g_test  = _route_g(dsc_te, psc_te, route)

    # Physical slack B = Q - departure_load
    L0 = float(dbar[np.array(route)].sum())
    B  = float(Q - L0)
    if B <= 0.0:
        B = calibrate_B_empirical(g_train, alpha=1.0 - ALPHA)

    models  = fit_otr(g_train, B)
    tau_myo = tau_myopic(omegaF, Cfail)
    tau_tun = tune_tau_fast(g_train, B, models, omegaF, Cfail) if models else tau_myo

    result = {"B": B, "tau_tuned": tau_tun}
    for label, tau_val in [("tuned", tau_tun), ("myopic", tau_myo), ("none", 1.0)]:
        result[label] = simulate_fast(g_test, B, tau_val, omegaF, Cfail, models)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Per-plan aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def _eval_plan(
    plan:    list,
    dbar:    np.ndarray,
    pbar:    np.ndarray,
    Q:       float,
    omegaF:  float,
    Cfail:   float,
    n_train: int,
    n_test:  int,
    seed:    int,
) -> dict:
    """Aggregate OTR evaluation over all routes in a plan."""
    dsc_tr, psc_tr = _gen_scenarios(dbar, pbar, n_train, seed)
    dsc_te, psc_te = _gen_scenarios(dbar, pbar, n_test,  seed + 99_991)

    routes = [r for r in plan if r]
    agg = {
        lbl: {"mean_cost": [], "handoff_rate": [], "fail_rate": [], "complete_rate": []}
        for lbl in ("tuned", "myopic", "none")
    }

    for route in routes:
        res = _eval_route(route, dbar, Q, dsc_tr, psc_tr, dsc_te, psc_te, omegaF, Cfail)
        for lbl in ("tuned", "myopic", "none"):
            for key in ("mean_cost", "handoff_rate", "fail_rate", "complete_rate"):
                agg[lbl][key].append(res[lbl][key])

    out = {}
    for lbl, data in agg.items():
        if data["mean_cost"]:
            out[lbl] = {
                "total_exec_cost":          float(sum(data["mean_cost"])),
                "mean_exec_cost_per_route": float(np.mean(data["mean_cost"])),
                "handoff_rate":             float(np.mean(data["handoff_rate"])),
                "fail_rate":                float(np.mean(data["fail_rate"])),
                "complete_rate":            float(np.mean(data["complete_rate"])),
            }
        else:
            out[lbl] = {
                k: 0.0 for k in
                ("total_exec_cost", "mean_exec_cost_per_route",
                 "handoff_rate", "fail_rate", "complete_rate")
            }
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Per-instance runner  (executed inside worker processes)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_instance(
    path:            str,
    tlim:            float,
    n_train:         int,
    n_test:          int,
    cfail_ratio:     float,
    active_policies: list,
) -> tuple[list[dict], str]:
    """Solve one instance + evaluate OTR.  Returns (rows, log_string).

    This function runs inside a worker process — all output is collected into
    log_string and printed by the main process to avoid interleaved output.
    """
    log: list[str] = []

    t_solve = time.time()
    sol = solve_instance(path, tlim, NO_IMPROVE, use_prune=True)
    solve_time = time.time() - t_solve

    dbar    = sol["dbar"]
    pbar    = sol["pbar"]
    Q       = sol["Q"]
    n       = sol["n"]
    omega_V = sol["omega_V"]
    omega_F = OMEGA_RATIO * omega_V
    Cfail   = cfail_ratio * omega_F
    inst_seed = SEED + abs(hash(sol["name"])) % 10_000

    rows: list[dict] = []
    for policy in active_policies:
        pdata = sol["res"][policy]
        plan  = pdata["plan"]
        K     = pdata["K"]
        dist  = pdata["dist"]

        # ALNS execution baseline
        _, evx    = eval_evextra(plan, dbar, pbar, n, Q)
        alns_exec = omega_F * evx
        alns_tbc  = dist + omega_V * K + alns_exec

        # OTR evaluation
        t_otr = time.time()
        otr   = _eval_plan(plan, dbar, pbar, Q, omega_F, Cfail,
                           n_train, n_test, inst_seed)
        otr_time = time.time() - t_otr

        noh_exec   = otr["none"]["total_exec_cost"]
        tune_exec  = otr["tuned"]["total_exec_cost"]
        saving_pct = 100.0 * (noh_exec - tune_exec) / max(noh_exec, 1e-9)

        row: dict = {
            "Instance":  sol["name"],
            "Plan":      policy,
            "N_cust":    n - 1,
            "K_routes":  K,
            "Travel":    round(dist, 2),
            "omega_V":   round(omega_V, 3),
            "omega_F":   round(omega_F, 3),
            "Cfail":     round(Cfail, 3),
            "ALNS_EVx":  round(evx, 4),
            "ALNS_exec": round(alns_exec, 2),
            "ALNS_TBC":  round(alns_tbc, 2),
        }
        _LABELS = [
            ("OTR_tuned",  "tuned"),
            ("OTR_myopic", "myopic"),
            ("NoHandoff",  "none"),
        ]
        for col_pfx, lbl in _LABELS:
            s = otr[lbl]
            row[f"{col_pfx}_exec"]    = round(s["total_exec_cost"], 2)
            row[f"{col_pfx}_TBC"]     = round(dist + omega_V * K + s["total_exec_cost"], 2)
            row[f"{col_pfx}_HO_rate"] = round(s["handoff_rate"], 4)
            row[f"{col_pfx}_fail"]    = round(s["fail_rate"], 4)
            row[f"{col_pfx}_ok"]      = round(s["complete_rate"], 4)

        row["OTR_saving_pct"] = round(saving_pct, 2)
        row["solve_s"]        = round(solve_time, 1)
        row["otr_s"]          = round(otr_time, 1)
        rows.append(row)

        log.append(
            f"  {policy:<5}  K={K}  dist={dist:.0f}  "
            f"ALNS_TBC={alns_tbc:.0f}  "
            f"OTR(tun)={row['OTR_tuned_TBC']:.0f}  "
            f"OTR(myo)={row['OTR_myopic_TBC']:.0f}  "
            f"NoHO={row['NoHandoff_TBC']:.0f}  "
            f"saving={saving_pct:.1f}%  "
            f"fail%={otr['tuned']['fail_rate'] * 100:.1f}  "
            f"t_otr={otr_time:.1f}s"
        )

    return rows, "\n".join(log)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary printer
# ═══════════════════════════════════════════════════════════════════════════════

def _print_summary(df: pd.DataFrame, policies: list) -> None:
    W = 104
    print("\n" + "=" * W)
    print("  SUMMARY — means across all instances")
    print("=" * W)
    print(
        f"  {'Plan':<6}  {'ALNS TBC':>10}  {'ALNS EVx':>9}  "
        f"{'OTR tun TBC':>12}  {'OTR myo TBC':>12}  {'NoHO TBC':>10}  "
        f"{'saving%':>8}  {'fail%(tun)':>11}  {'HO%(tun)':>9}"
    )
    print("  " + "─" * (W - 2))
    for policy in policies:
        sub = df[df["Plan"] == policy]
        if sub.empty:
            continue
        print(
            f"  {policy:<6}  "
            f"{sub['ALNS_TBC'].mean():>10.0f}  "
            f"{sub['ALNS_EVx'].mean():>9.4f}  "
            f"{sub['OTR_tuned_TBC'].mean():>12.0f}  "
            f"{sub['OTR_myopic_TBC'].mean():>12.0f}  "
            f"{sub['NoHandoff_TBC'].mean():>10.0f}  "
            f"{sub['OTR_saving_pct'].mean():>8.1f}  "
            f"{sub['OTR_tuned_fail'].mean() * 100:>11.2f}  "
            f"{sub['OTR_tuned_HO_rate'].mean() * 100:>9.2f}"
        )
    print("\n  Notes:")
    print("    ALNS_EVx  = E[extra trucks/route] under headline stress mixture")
    print("    *_TBC     = Travel + omega_V × K_routes + *_exec  (total budget cost)")
    print("    saving%   = (NoHandoff_exec − OTR_tuned_exec) / NoHandoff_exec × 100")
    print(f"    omegaF/omegaV = {OMEGA_RATIO}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    data_dir     = _DEFAULT_DATA_DIR
    tlim         = _DEFAULT_TLIM
    n_train      = _DEFAULT_N_TRAIN
    n_test       = _DEFAULT_N_TEST
    cfail_ratio  = _DEFAULT_CFAIL_RATIO
    max_n        = None
    out_stem     = _DEFAULT_OUT_STEM
    policies     = list(_DEFAULT_POLICIES)
    n_workers    = os.cpu_count() or 1

    for arg in sys.argv[1:]:
        if   arg.startswith("dir="):       data_dir    = arg[4:]
        elif arg.startswith("tlim="):      tlim        = float(arg[5:])
        elif arg.startswith("n_train="):   n_train     = int(arg[8:])
        elif arg.startswith("n_test="):    n_test      = int(arg[7:])
        elif arg.startswith("cfail="):     cfail_ratio = float(arg[6:])
        elif arg.startswith("max="):       max_n       = int(arg[4:])
        elif arg.startswith("out="):       out_stem    = arg[4:]
        elif arg.startswith("workers="):   n_workers   = int(arg[8:])
        elif arg.startswith("policies="):  policies    = arg[9:].split(",")
        else:
            print(f"  Unknown argument '{arg}' — ignored")

    files = sorted(glob.glob(str(Path(data_dir) / "*.vrpspd")))
    if not files:
        files = sorted(glob.glob(str(Path(data_dir) / "*.txt")))
    if not files:
        print(f"ERROR: no instance files found in '{data_dir}'")
        return
    if max_n:
        files = files[:max_n]

    n_workers = min(n_workers, len(files))

    W = 90
    print("=" * W)
    print("  OTR EVALUATION — Online Threshold Reassignment on Dethloff SVRPSPD")
    print("=" * W)
    print(f"  instances={len(files)}  policies={policies}  workers={n_workers}/{os.cpu_count()}")
    print(f"  ALNS tlim={tlim}s/policy   n_train={n_train}   n_test={n_test}")
    print(f"  Cfail/omegaF={cfail_ratio}   alpha={ALPHA}   omegaF/omegaV={OMEGA_RATIO}")
    est_min = len(files) * len(policies) * tlim / 60 / n_workers
    print(f"  Estimated wall time: >{est_min:.0f} min (ALNS) + OTR (vectorized, fast)")
    print("-" * W)

    all_rows: list[dict] = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _run_instance, f, tlim, n_train, n_test, cfail_ratio, policies
            ): Path(f).stem
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
                print(log)
            except Exception as exc:
                print(f"\n[{done}/{len(files)}] {stem} — ERROR: {exc}")
                traceback.print_exc()

    total_min = (time.time() - t_start) / 60
    print(f"\n{'─' * W}")
    print(f"  Total wall time: {total_min:.1f} min")

    if not all_rows:
        print("  No results produced — nothing written.")
        return

    df = pd.DataFrame(all_rows)
    _print_summary(df, policies)

    csv_path = out_stem + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Wrote {csv_path}  ({len(df)} rows × {len(df.columns)} columns)")
    try:
        df.to_excel(out_stem + ".xlsx", index=False)
        print(f"  Wrote {out_stem}.xlsx")
    except Exception as exc:
        print(f"  (Excel skipped — pip install openpyxl to enable: {exc})")


if __name__ == "__main__":
    main()

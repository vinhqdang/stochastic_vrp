#!/usr/bin/env python3
"""
run_otr_eval.py -- Full evaluation of OTR on all Dethloff VRPSPD instances.

Corrected version. Changes vs the earlier harness:
  * Deterministic per-instance seed (hashlib.md5, not the PYTHONHASHSEED-salted hash()) -> the OTR
    numbers are now reproducible run to run and across worker processes.
  * OTR is TRAINED on the planning reference (gamma = DIST) but TESTED on the stress mixture
    (SHAPE_W), exactly like the W-DRO economics. This is the honest "calibrate on the reference,
    deploy on a reality that differs" test, and it makes ALNS_EVx and the OTR numbers share one
    test distribution. (Previously OTR was train+tested on gamma -- an in-distribution number.)
  * Uses the load-space core.otr (otr.py): passes per-route delivery/pickup draws and Q; no g/B.

For each of the Dethloff instances the script:
  1. Solves with ALNS under the planning policies (Det / Gounaris / Cui / MDRO / SAA / WDRO).
  2. Trains OTR on gamma reference scenarios; tests on stress-mixture scenarios.
  3. Simulates execution under three thresholds: tuned tau, myopic tau, no-handoff.
  4. Aggregates per-route stats to plan level and writes a CSV + Excel report.

Usage:
    python scripts/run_otr_eval.py [key=value ...]
Options:
    dir=<path>     Dethloff folder         (default: data/Dethloff)
    tlim=<s>       ALNS time limit/policy  (default: 60)
    n_train=<N>    Train scenarios/route   (default: 1000)   [gamma reference]
    n_test=<N>     Test  scenarios/route   (default: 2000)   [stress mixture]
    cfail=<ratio>  Cfail / omegaF          (default: 5.0)    [sweep this: 2, 5, 10]
    policies=p,..  ALNS policies           (default: Det,Gounaris,Cui,MDRO,SAA,WDRO)
    workers=<N>    Parallel processes      (default: all CPUs)
    max=<N>        Cap number of instances (default: all)
    out=<stem>     Output filename stem    (default: results_otr_eval)
"""

import os
import sys
import glob
import time
import hashlib
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# -- import paths (works from svrpspd_wdro/ or the repo root) ------------------
_SCRIPTS = Path(__file__).resolve().parent
_WDRO    = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_SCRIPTS))

from core.otr_endpoint import (
    fit_otr,
    tau_myopic,
    tune_tau_fast,
    simulate_fast,
)
from dethloff_runner import (
    sample_demands,
    solve_instance,
    eval_evextra,
    SHAPE_W,
    ALPHA, CV, DIST, SEED, OMEGA_RATIO, NO_IMPROVE,
)

# -- run-level defaults --------------------------------------------------------
_DEFAULT_DATA_DIR    = str(_WDRO / "data" / "Dethloff")
_DEFAULT_TLIM        = 60.0
_DEFAULT_N_TRAIN     = 1000
_DEFAULT_N_TEST      = 2000
_DEFAULT_CFAIL_RATIO = 5.0
_DEFAULT_POLICIES    = ["Det", "Gounaris", "Cui", "MDRO", "SAA", "WDRO"]
_DEFAULT_OUT_STEM    = "results_otr_eval"


# =============================================================================
# Scenario helpers
# =============================================================================
def _gen_reference(dbar: np.ndarray, pbar: np.ndarray, N: int, seed: int):
    """TRAIN scenarios: the planning reference (gamma = DIST). Returns (dsc, psc), shape (N, n)."""
    n = len(dbar)
    rng = np.random.default_rng(seed)
    dsc = sample_demands(dbar, n, N, CV, DIST, rng)
    psc = sample_demands(pbar, n, N, CV, DIST, rng)
    return dsc, psc


def _gen_test_mixture(dbar: np.ndarray, pbar: np.ndarray, N: int, seed: int):
    """TEST scenarios: pooled stress mixture (SHAPE_W). Each shape contributes ~round(N * w) rows,
    drawn with the same copula. The 'heavy' slot uses lognormal (the paper headline). Returns
    (dsc, psc) with ~N rows. Matches the distribution that eval_evextra / the economics use."""
    n = len(dbar)
    d_parts, p_parts = [], []
    for si, (shape, w) in enumerate(SHAPE_W.items()):
        ns = int(round(N * w))
        if ns <= 0:
            continue
        dist = "lognormal" if shape == "heavy" else shape
        rng = np.random.default_rng(seed + 7919 * (si + 1))
        d_parts.append(sample_demands(dbar, n, ns, CV, dist, rng))
        p_parts.append(sample_demands(pbar, n, ns, CV, dist, rng))
    return np.vstack(d_parts), np.vstack(p_parts)


# =============================================================================
# Per-route / per-plan OTR evaluation
# =============================================================================
def _eval_route(Q, d_tr, p_tr, d_te, p_te, omegaF, Cfail) -> dict:
    """Fit OTR on reference draws (d_tr,p_tr) for one route, simulate on mixture draws (d_te,p_te).
    All inputs already sliced to the route: shape (N, m)."""
    models  = fit_otr(d_tr, p_tr, Q)
    tau_myo = tau_myopic(omegaF, Cfail)
    tau_tun = tune_tau_fast(d_tr, p_tr, Q, models, omegaF, Cfail) if models else tau_myo
    result = {"tau_tuned": tau_tun}
    for label, tau in [("tuned", tau_tun), ("myopic", tau_myo), ("none", 1.0)]:
        result[label] = simulate_fast(d_te, p_te, Q, tau, omegaF, Cfail, models)
    return result


def _eval_plan(plan, dbar, pbar, Q, omegaF, Cfail, n_train, n_test, seed) -> dict:
    """Aggregate OTR over all routes of a plan. Train on gamma reference, test on stress mixture."""
    dsc_tr, psc_tr = _gen_reference(dbar, pbar, n_train, seed)
    dsc_te, psc_te = _gen_test_mixture(dbar, pbar, n_test, seed + 99_991)

    routes = [r for r in plan if r]
    keys = ("mean_cost", "handoff_rate", "fail_rate", "complete_rate", "emg_vehicles")
    agg = {lbl: {k: [] for k in keys} for lbl in ("tuned", "myopic", "none")}

    for route in routes:
        rr = np.asarray(route)
        res = _eval_route(Q, dsc_tr[:, rr], psc_tr[:, rr], dsc_te[:, rr], psc_te[:, rr], omegaF, Cfail)
        for lbl in ("tuned", "myopic", "none"):
            for k in keys:
                agg[lbl][k].append(res[lbl][k])

    out = {}
    for lbl, data in agg.items():
        if data["mean_cost"]:
            out[lbl] = {
                "total_exec_cost":          float(sum(data["mean_cost"])),
                "mean_exec_cost_per_route": float(np.mean(data["mean_cost"])),
                "handoff_rate":             float(np.mean(data["handoff_rate"])),
                "fail_rate":                float(np.mean(data["fail_rate"])),
                "complete_rate":            float(np.mean(data["complete_rate"])),
                "emg_vehicles":             float(np.mean(data["emg_vehicles"])),
            }
        else:
            out[lbl] = {k: 0.0 for k in
                        ("total_exec_cost", "mean_exec_cost_per_route", "handoff_rate",
                         "fail_rate", "complete_rate", "emg_vehicles")}
    return out


# =============================================================================
# Per-instance runner (inside worker processes)
# =============================================================================
def _run_instance(path, tlim, n_train, n_test, cfail_ratio, active_policies):
    log = []
    t_solve = time.time()
    sol = solve_instance(path, tlim, NO_IMPROVE, use_prune=True)
    solve_time = time.time() - t_solve

    dbar, pbar = sol["dbar"], sol["pbar"]
    Q, n       = sol["Q"], sol["n"]
    omega_V    = sol["omega_V"]
    omega_F    = OMEGA_RATIO * omega_V
    Cfail      = cfail_ratio * omega_F
    # deterministic per-instance seed (reproducible across processes / runs)  <-- fix
    inst_seed  = SEED + int(hashlib.md5(sol["name"].encode()).hexdigest(), 16) % 10_000

    rows = []
    for policy in active_policies:
        pdata = sol["res"][policy]
        plan, K, dist = pdata["plan"], pdata["K"], pdata["dist"]

        # reactive (ALNS) baseline, on the stress mixture
        _, evx    = eval_evextra(plan, dbar, pbar, n, Q)
        alns_exec = omega_F * evx
        alns_tbc  = dist + omega_V * K + alns_exec

        t_otr = time.time()
        otr   = _eval_plan(plan, dbar, pbar, Q, omega_F, Cfail, n_train, n_test, inst_seed)
        otr_time = time.time() - t_otr

        noh_exec   = otr["none"]["total_exec_cost"]
        tune_exec  = otr["tuned"]["total_exec_cost"]
        saving_pct = 100.0 * (noh_exec - tune_exec) / max(noh_exec, 1e-9)

        row = {
            "Instance": sol["name"], "Plan": policy, "N_cust": n - 1, "K_routes": K,
            "Travel": round(dist, 2), "omega_V": round(omega_V, 3), "omega_F": round(omega_F, 3),
            "Cfail": round(Cfail, 3),
            "ALNS_EVx": round(evx, 4), "ALNS_exec": round(alns_exec, 2), "ALNS_TBC": round(alns_tbc, 2),
        }
        for col_pfx, lbl in [("OTR_tuned", "tuned"), ("OTR_myopic", "myopic"), ("NoHandoff", "none")]:
            s = otr[lbl]
            row[f"{col_pfx}_exec"]    = round(s["total_exec_cost"], 2)
            row[f"{col_pfx}_TBC"]     = round(dist + omega_V * K + s["total_exec_cost"], 2)
            row[f"{col_pfx}_HO_rate"] = round(s["handoff_rate"], 4)
            row[f"{col_pfx}_fail"]    = round(s["fail_rate"], 4)
            row[f"{col_pfx}_emgV"]    = round(s["emg_vehicles"], 4)
            row[f"{col_pfx}_ok"]      = round(s["complete_rate"], 4)
        row["OTR_saving_pct"] = round(saving_pct, 2)
        row["solve_s"] = round(solve_time, 1)
        row["otr_s"]   = round(otr_time, 1)
        rows.append(row)

        log.append(
            f"  {policy:<8} K={K} dist={dist:.0f}  ALNS_EVx={evx:.3f} ALNS_TBC={alns_tbc:.0f}  "
            f"OTR(tun)={row['OTR_tuned_TBC']:.0f} OTR(myo)={row['OTR_myopic_TBC']:.0f} "
            f"NoHO={row['NoHandoff_TBC']:.0f}  saving={saving_pct:.1f}%  "
            f"fail%(tun)={otr['tuned']['fail_rate']*100:.1f}  t_otr={otr_time:.1f}s"
        )

    return rows, "\n".join(log)


# =============================================================================
# Summary printer
# =============================================================================
def _print_summary(df: pd.DataFrame, policies: list) -> None:
    W = 116
    print("\n" + "=" * W)
    print("  SUMMARY -- means across all instances   (OTR: train=gamma reference, test=stress mixture)")
    print("=" * W)
    print(f"  {'Plan':<9} {'ALNS TBC':>9} {'ALNS EVx':>9} {'OTRtun TBC':>11} {'OTRmyo TBC':>11} "
          f"{'NoHO TBC':>9} {'saving%':>8} {'fail%(tun)':>11} {'HO%(tun)':>9}")
    print("  " + "-" * (W - 2))
    for policy in policies:
        sub = df[df["Plan"] == policy]
        if sub.empty:
            continue
        print(f"  {policy:<9} {sub['ALNS_TBC'].mean():>9.0f} {sub['ALNS_EVx'].mean():>9.4f} "
              f"{sub['OTR_tuned_TBC'].mean():>11.0f} {sub['OTR_myopic_TBC'].mean():>11.0f} "
              f"{sub['NoHandoff_TBC'].mean():>9.0f} {sub['OTR_saving_pct'].mean():>8.1f} "
              f"{sub['OTR_tuned_fail'].mean()*100:>11.2f} {sub['OTR_tuned_HO_rate'].mean()*100:>9.2f}")
    print("\n  Notes:")
    print("    ALNS_EVx = E[extra trucks/route], reactive ceil model, stress mixture (this IS the OOS")
    print("               risk -- check it is comparable across policies, else baselines are miscalibrated)")
    print("    *_TBC    = Travel + omega_V*K + *_exec.  OTR exec uses ceil-vehicle emergencies * Cfail")
    print("               (set emergency_ceil=False in core.otr for the flat-per-event variant)")
    print("    saving%  = (NoHandoff_exec - OTR_tuned_exec) / NoHandoff_exec * 100  (within the Cfail world)")
    print(f"    omegaF/omegaV = {OMEGA_RATIO}   Cfail/omegaF swept via cfail=<ratio> (run 2, 5, 10)")
    print("    Compare OTR_tuned vs NoHandoff (same cost world); ALNS_TBC is a DIFFERENT cost model.")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    data_dir, tlim          = _DEFAULT_DATA_DIR, _DEFAULT_TLIM
    n_train, n_test         = _DEFAULT_N_TRAIN, _DEFAULT_N_TEST
    cfail_ratio, max_n      = _DEFAULT_CFAIL_RATIO, None
    out_stem, policies      = _DEFAULT_OUT_STEM, list(_DEFAULT_POLICIES)
    n_workers               = os.cpu_count() or 1

    for arg in sys.argv[1:]:
        if   arg.startswith("dir="):      data_dir    = arg[4:]
        elif arg.startswith("tlim="):     tlim        = float(arg[5:])
        elif arg.startswith("n_train="):  n_train     = int(arg[8:])
        elif arg.startswith("n_test="):   n_test      = int(arg[7:])
        elif arg.startswith("cfail="):    cfail_ratio = float(arg[6:])
        elif arg.startswith("max="):      max_n       = int(arg[4:])
        elif arg.startswith("out="):      out_stem    = arg[4:]
        elif arg.startswith("workers="):  n_workers   = int(arg[8:])
        elif arg.startswith("policies="): policies    = arg[9:].split(",")
        else: print(f"  Unknown argument '{arg}' -- ignored")

    files = sorted(glob.glob(str(Path(data_dir) / "*.vrpspd")))
    if not files:
        files = sorted(glob.glob(str(Path(data_dir) / "*.txt")))
    if not files:
        print(f"ERROR: no instance files found in '{data_dir}'"); return
    if max_n:
        files = files[:max_n]
    n_workers = min(n_workers, len(files))

    W = 96
    print("=" * W)
    print("  OTR EVALUATION -- Online Threshold Reassignment on Dethloff SVRPSPD  (load-space, corrected)")
    print("=" * W)
    print(f"  instances={len(files)}  policies={policies}  workers={n_workers}/{os.cpu_count()}")
    print(f"  ALNS tlim={tlim}s/policy   n_train={n_train} (gamma)   n_test={n_test} (mixture)")
    print(f"  Cfail/omegaF={cfail_ratio}   alpha={ALPHA}   omegaF/omegaV={OMEGA_RATIO}")
    est_min = len(files) * len(policies) * tlim / 60 / max(1, n_workers)
    print(f"  Estimated wall time: >{est_min:.0f} min (ALNS) + OTR (vectorized)")
    print("-" * W)

    all_rows, t_start = [], time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_instance, f, tlim, n_train, n_test, cfail_ratio, policies):
                   Path(f).stem for f in files}
        done = 0
        for fut in as_completed(futures):
            stem = futures[fut]; done += 1
            try:
                rows, log = fut.result()
                all_rows.extend(rows)
                print(f"\n[{done}/{len(files)}] {stem}\n{log}")
            except Exception as exc:
                print(f"\n[{done}/{len(files)}] {stem} -- ERROR: {exc}")
                traceback.print_exc()

    print(f"\n{'-' * W}\n  Total wall time: {(time.time() - t_start)/60:.1f} min")
    if not all_rows:
        print("  No results produced -- nothing written."); return

    df = pd.DataFrame(all_rows)
    _print_summary(df, policies)

    csv_path = out_stem + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Wrote {csv_path}  ({len(df)} rows x {len(df.columns)} columns)")
    try:
        df.to_excel(out_stem + ".xlsx", index=False)
        print(f"  Wrote {out_stem}.xlsx")
    except Exception as exc:
        print(f"  (Excel skipped -- pip install openpyxl: {exc})")


if __name__ == "__main__":
    main()
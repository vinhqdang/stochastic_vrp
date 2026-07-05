#!/usr/bin/env python3
"""
run_otr2_eval.py — Comparative evaluation of OTR-2.0 against OTR v1 and baselines.

Two modes:

  synthetic   Replicates the OTR-2.0 spec benchmark: collect-then-deliver,
              milk-run ramp + regime switching, and high-cost-ratio scenarios,
              5 seeds each. Compares per-route execution policies:
                  none      no handoff (reactive: pay Cfail at overflow)
                  v1_myo    OTR v1, endpoint label, myopic tau = omegaF/Cfail
                  v1_tun    OTR v1, endpoint label, grid-tuned tau
                  fb_tun    OTR-2.0 fallback: PEAK label + grid-tuned tau
                  v2_lsm    OTR-2.0: peak-aware Longstaff-Schwartz optimal stopping

  dethloff    Full pipeline on the 40 Dethloff VRPSPD instances: solve each
              with ALNS under Det/SAA/WDRO planning gates, then evaluate all
              five execution policies on every route (train scenarios fit the
              models, independent test scenarios score them).

Usage (from svrpspd_wdro/ or the repo root):
    python scripts/run_otr2_eval.py synthetic
    python scripts/run_otr2_eval.py dethloff [key=value ...]

Dethloff options (same defaults as run_otr_eval.py):
    dir=<path>     tlim=<s>      n_train=<N>   n_test=<N>
    cfail=<ratio>  policies=..   workers=<N>   max=<N>   out=<stem>
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

from core.otr import fit_otr, tau_myopic, tune_tau_fast, simulate_fast
from core.otr2 import (
    fit_lsm,
    fit_otr_peak,
    calibrate_B_empirical_peak,
    simulate_v2,
    validate,
)
from dethloff_runner import (
    sample_demands,
    solve_instance,
    eval_evextra,
    ALPHA, CV, DIST, SEED, OMEGA_RATIO, NO_IMPROVE,
)

RESULTS_DIR = _WDRO / "results"

POLICY_LABELS = ["none", "v1_myo", "v1_tun", "fb_tun", "v2_lsm"]
STAT_KEYS     = ["mean_cost", "handoff_rate", "fail_rate", "complete_rate"]


# ═══════════════════════════════════════════════════════════════════════════════
# Shared: evaluate all five execution policies on one (g_train, g_test) pair
# ═══════════════════════════════════════════════════════════════════════════════

def eval_policies(
    g_train: np.ndarray,
    g_test:  np.ndarray,
    B:       float,
    omegaF:  float,
    Cfail:   float,
) -> dict:
    """Fit every policy on g_train and score it on g_test.

    Returns {label: stats_dict} plus 'diag' with the v2 validation report.
    """
    m = g_train.shape[1]

    # ── OTR v1 (endpoint label) ────────────────────────────────────────────
    v1_models = fit_otr(g_train, B)
    tau_myo   = tau_myopic(omegaF, Cfail)
    tau_v1    = tune_tau_fast(g_train, B, v1_models, omegaF, Cfail) if v1_models else tau_myo

    # ── OTR-2.0 fallback (peak label + tuned tau) ──────────────────────────
    fb_models = fit_otr_peak(g_train, B)
    tau_fb    = tune_tau_fast(g_train, B, fb_models, omegaF, Cfail) if fb_models else tau_myo

    # ── OTR-2.0 (Longstaff-Schwartz optimal stopping, no tau) ──────────────
    cont_models = fit_lsm(g_train, B, omegaF, Cfail)

    out = {
        "none":   simulate_fast(g_test, B, 1.0,     omegaF, Cfail, v1_models),
        "v1_myo": simulate_fast(g_test, B, tau_myo, omegaF, Cfail, v1_models),
        "v1_tun": simulate_fast(g_test, B, tau_v1,  omegaF, Cfail, v1_models),
        "fb_tun": simulate_fast(g_test, B, tau_fb,  omegaF, Cfail, fb_models),
        "v2_lsm": simulate_v2(g_test, B, omegaF, Cfail, cont_models),
    }
    out["diag"] = validate(g_test, B, omegaF, Cfail, cont_models)
    out["diag"]["tau_v1"] = float(tau_v1)
    out["diag"]["tau_fb"] = float(tau_fb)
    return out


def saving_pct(none_cost: float, policy_cost: float) -> float:
    return 100.0 * (none_cost - policy_cost) / max(none_cost, 1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# Mode 1 — synthetic spec-replication benchmark
# ═══════════════════════════════════════════════════════════════════════════════

M_SYNTH   = 12
N_TRAIN_S = 10_000
N_TEST_S  = 12_000
SEEDS     = [0, 1, 2, 3, 4]


def gen_collect_then_deliver(N: int, m: int, rng) -> np.ndarray:
    """Pickups first, then the same goods delivered back: endpoint W_m == 0
    on every route, while the mid-route peak (= total picked up) varies."""
    half = m // 2
    pickups = rng.gamma(4.0, 0.25, (N, half))         # mean 1, cv 0.5
    return np.concatenate([pickups, -pickups[:, ::-1]], axis=1)


def gen_milk_run(N: int, m: int, rng) -> np.ndarray:
    """Milk-run ramp with regime switching: net increments mostly positive
    and growing along the route; 30% of days run a high-volume regime."""
    regime = rng.choice([0.7, 1.6], size=N, p=[0.7, 0.3])
    ramp   = 0.5 + np.arange(1, m + 1) / m            # 0.58 .. 1.5
    noise  = rng.gamma(4.0, 0.25, (N, m))             # mean 1, cv 0.5
    deliveries = 0.45                                  # constant drop-off per stop
    return regime[:, None] * ramp[None, :] * noise - deliveries


SCENARIOS_SYNTH = [
    ("collect_then_deliver", gen_collect_then_deliver, 5.0),
    ("milk_run_regime",      gen_milk_run,             5.0),
    ("high_cost_ratio",      gen_milk_run,             20.0),
]


def run_synthetic() -> None:
    omegaF = 1.0
    rows = []
    print("=" * 100)
    print("  OTR-2.0 SYNTHETIC BENCHMARK — spec-replication scenarios, "
          f"{len(SEEDS)} seeds x {N_TEST_S:,} test routes, m={M_SYNTH}")
    print("=" * 100)

    for name, gen, cfail_ratio in SCENARIOS_SYNTH:
        Cfail = cfail_ratio * omegaF
        for seed in SEEDS:
            rng_tr = np.random.default_rng(1_000 * seed + 1)
            rng_te = np.random.default_rng(1_000 * seed + 2)
            g_tr = gen(N_TRAIN_S, M_SYNTH, rng_tr)
            g_te = gen(N_TEST_S,  M_SYNTH, rng_te)
            B = calibrate_B_empirical_peak(g_tr, alpha=0.10)

            res = eval_policies(g_tr, g_te, B, omegaF, Cfail)
            none_cost = res["none"]["mean_cost"]
            row = {"scenario": name, "seed": seed, "B": round(B, 4),
                   "cfail_ratio": cfail_ratio,
                   "corr_endpoint_peak": round(res["diag"]["corr_endpoint_peak"], 4)
                       if np.isfinite(res["diag"]["corr_endpoint_peak"]) else float("nan"),
                   "peak_overflow_rate": round(res["diag"]["peak_overflow_rate"], 4)}
            for lbl in POLICY_LABELS:
                row[f"{lbl}_cost"]   = round(res[lbl]["mean_cost"], 5)
                row[f"{lbl}_fail"]   = round(res[lbl]["fail_rate"], 4)
                row[f"{lbl}_ho"]     = round(res[lbl]["handoff_rate"], 4)
                if lbl != "none":
                    row[f"{lbl}_saving"] = round(saving_pct(none_cost, res[lbl]["mean_cost"]), 2)
            rows.append(row)
            print(f"  {name:<22} seed={seed}  "
                  f"save%: v1_myo={row['v1_myo_saving']:>6.1f}  "
                  f"v1_tun={row['v1_tun_saving']:>6.1f}  "
                  f"fb_tun={row['fb_tun_saving']:>6.1f}  "
                  f"v2_lsm={row['v2_lsm_saving']:>6.1f}   "
                  f"fail%: none={row['none_fail']*100:.1f} v2={row['v2_lsm_fail']*100:.2f}")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "results_otr2_synthetic.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 100)
    print("  SUMMARY — mean saving vs no-handoff (%, over seeds)")
    print("=" * 100)
    print(f"  {'scenario':<24}{'v1 myopic':>11}{'v1 tuned':>10}{'fallback':>10}"
          f"{'v2 LSM':>8}{'v2 fail%':>10}{'corr(end,peak)':>16}")
    print("  " + "-" * 96)
    summary = []
    for name, _, _ in SCENARIOS_SYNTH:
        sub = df[df["scenario"] == name]
        line = {
            "scenario": name,
            "v1_myo_saving": sub["v1_myo_saving"].mean(),
            "v1_tun_saving": sub["v1_tun_saving"].mean(),
            "fb_tun_saving": sub["fb_tun_saving"].mean(),
            "v2_lsm_saving": sub["v2_lsm_saving"].mean(),
            "v2_fail_pct":   sub["v2_lsm_fail"].mean() * 100,
            "corr":          sub["corr_endpoint_peak"].mean(),
        }
        summary.append(line)
        print(f"  {name:<24}{line['v1_myo_saving']:>10.1f}%{line['v1_tun_saving']:>9.1f}%"
              f"{line['fb_tun_saving']:>9.1f}%{line['v2_lsm_saving']:>7.1f}%"
              f"{line['v2_fail_pct']:>9.2f}%{line['corr']:>16.3f}")

    with open(RESULTS_DIR / "results_otr2_synthetic_summary.json", "w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    print(f"\n  Wrote {csv_path}")
    print(f"  Wrote {RESULTS_DIR / 'results_otr2_synthetic_summary.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Mode 2 — Dethloff instances (ALNS plans + per-route execution policies)
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_DATA_DIR    = str(_WDRO / "data" / "Dethloff")
_DEFAULT_TLIM        = 60.0
_DEFAULT_N_TRAIN     = 1000
_DEFAULT_N_TEST      = 2000
_DEFAULT_CFAIL_RATIO = 5.0
_DEFAULT_POLICIES    = ["Det", "SAA", "WDRO"]
_DEFAULT_OUT_STEM    = "results_otr2_eval"


def _gen_scenarios(dbar, pbar, N, seed):
    n = len(dbar)
    rng = np.random.default_rng(seed)
    dsc = sample_demands(dbar, n, N, CV, DIST, rng)
    psc = sample_demands(pbar, n, N, CV, DIST, rng)
    return dsc, psc


def _route_g(dsc, psc, route):
    r = np.array(route)
    return psc[:, r] - dsc[:, r]


def _eval_plan(plan, dbar, pbar, Q, omegaF, Cfail, n_train, n_test, seed):
    """Evaluate all execution policies on every route of one ALNS plan."""
    dsc_tr, psc_tr = _gen_scenarios(dbar, pbar, n_train, seed)
    dsc_te, psc_te = _gen_scenarios(dbar, pbar, n_test,  seed + 99_991)

    routes = [r for r in plan if r]
    agg = {lbl: {k: [] for k in STAT_KEYS} for lbl in POLICY_LABELS}
    diags = {"corr_endpoint_peak": [], "peak_overflow_rate": [], "deploy_ok": []}

    for route in routes:
        g_train = _route_g(dsc_tr, psc_tr, route)
        g_test  = _route_g(dsc_te, psc_te, route)

        L0 = float(dbar[np.array(route)].sum())
        B  = float(Q - L0)
        if B <= 0.0:
            # v1 used the (buggy) endpoint quantile here; v2: peak quantile
            B = calibrate_B_empirical_peak(g_train, alpha=1.0 - ALPHA)

        res = eval_policies(g_train, g_test, B, omegaF, Cfail)
        for lbl in POLICY_LABELS:
            for k in STAT_KEYS:
                agg[lbl][k].append(res[lbl][k])
        c = res["diag"]["corr_endpoint_peak"]
        if np.isfinite(c):
            diags["corr_endpoint_peak"].append(c)
        diags["peak_overflow_rate"].append(res["diag"]["peak_overflow_rate"])
        diags["deploy_ok"].append(res["diag"]["deploy_ok"])

    out = {}
    for lbl, data in agg.items():
        if data["mean_cost"]:
            out[lbl] = {
                "total_exec_cost": float(sum(data["mean_cost"])),
                "handoff_rate":    float(np.mean(data["handoff_rate"])),
                "fail_rate":       float(np.mean(data["fail_rate"])),
                "complete_rate":   float(np.mean(data["complete_rate"])),
            }
        else:
            out[lbl] = {k: 0.0 for k in
                        ("total_exec_cost", "handoff_rate", "fail_rate", "complete_rate")}
    out["diag"] = {
        "corr_endpoint_peak": float(np.mean(diags["corr_endpoint_peak"]))
                              if diags["corr_endpoint_peak"] else float("nan"),
        "peak_overflow_rate": float(np.mean(diags["peak_overflow_rate"]))
                              if diags["peak_overflow_rate"] else 0.0,
        "deploy_ok_frac":     float(np.mean(diags["deploy_ok"]))
                              if diags["deploy_ok"] else 0.0,
    }
    return out


def _run_instance(path, tlim, n_train, n_test, cfail_ratio, active_policies):
    """Solve one instance + evaluate execution policies. Runs in a worker."""
    log = []
    t_solve = time.time()
    sol = solve_instance(path, tlim, NO_IMPROVE, use_prune=True)
    solve_time = time.time() - t_solve

    dbar, pbar, Q, n = sol["dbar"], sol["pbar"], sol["Q"], sol["n"]
    omega_V = sol["omega_V"]
    omega_F = OMEGA_RATIO * omega_V
    Cfail   = cfail_ratio * omega_F
    inst_seed = SEED + abs(hash(sol["name"])) % 10_000

    rows = []
    for policy in active_policies:
        pdata = sol["res"][policy]
        plan, K, dist = pdata["plan"], pdata["K"], pdata["dist"]

        _, evx    = eval_evextra(plan, dbar, pbar, n, Q)
        alns_exec = omega_F * evx

        t0  = time.time()
        agg = _eval_plan(plan, dbar, pbar, Q, omega_F, Cfail,
                         n_train, n_test, inst_seed)
        otr_time = time.time() - t0

        none_exec = agg["none"]["total_exec_cost"]
        fixed     = dist + omega_V * K

        row = {
            "Instance": sol["name"], "Plan": policy, "N_cust": n - 1,
            "K_routes": K, "Travel": round(dist, 2),
            "omega_V": round(omega_V, 3), "omega_F": round(omega_F, 3),
            "Cfail": round(Cfail, 3),
            "ALNS_EVx": round(evx, 4),
            "ALNS_TBC": round(fixed + alns_exec, 2),
            "corr_end_peak": round(agg["diag"]["corr_endpoint_peak"], 4),
            "peak_ovf_rate": round(agg["diag"]["peak_overflow_rate"], 4),
        }
        for lbl in POLICY_LABELS:
            s = agg[lbl]
            row[f"{lbl}_exec"] = round(s["total_exec_cost"], 2)
            row[f"{lbl}_TBC"]  = round(fixed + s["total_exec_cost"], 2)
            row[f"{lbl}_HO"]   = round(s["handoff_rate"], 4)
            row[f"{lbl}_fail"] = round(s["fail_rate"], 4)
            if lbl != "none":
                row[f"{lbl}_saving"] = round(saving_pct(none_exec, s["total_exec_cost"]), 2)
        row["solve_s"] = round(solve_time, 1)
        row["eval_s"]  = round(otr_time, 1)
        rows.append(row)

        log.append(
            f"  {policy:<5} K={K} dist={dist:.0f}  exec: "
            f"none={none_exec:.1f}  v1_tun={row['v1_tun_exec']:.1f}  "
            f"fb={row['fb_tun_exec']:.1f}  v2={row['v2_lsm_exec']:.1f}  "
            f"save%(v1/fb/v2)={row['v1_tun_saving']:.1f}/"
            f"{row['fb_tun_saving']:.1f}/{row['v2_lsm_saving']:.1f}  "
            f"corr={row['corr_end_peak']:.2f}  t={otr_time:.0f}s"
        )
    return rows, "\n".join(log)


def _print_summary_dethloff(df, policies):
    W = 118
    print("\n" + "=" * W)
    print("  SUMMARY — means across all instances (exec = expected execution cost per plan)")
    print("=" * W)
    print(f"  {'Plan':<6}{'ALNS TBC':>10}{'none TBC':>10}{'v1myo TBC':>11}"
          f"{'v1tun TBC':>11}{'fb TBC':>10}{'v2 TBC':>10}"
          f"{'sv% v1tun':>10}{'sv% fb':>8}{'sv% v2':>8}{'v2 fail%':>9}{'corr':>7}")
    print("  " + "-" * (W - 2))
    for policy in policies:
        sub = df[df["Plan"] == policy]
        if sub.empty:
            continue
        print(f"  {policy:<6}"
              f"{sub['ALNS_TBC'].mean():>10.0f}"
              f"{sub['none_TBC'].mean():>10.0f}"
              f"{sub['v1_myo_TBC'].mean():>11.0f}"
              f"{sub['v1_tun_TBC'].mean():>11.0f}"
              f"{sub['fb_tun_TBC'].mean():>10.0f}"
              f"{sub['v2_lsm_TBC'].mean():>10.0f}"
              f"{sub['v1_tun_saving'].mean():>10.1f}"
              f"{sub['fb_tun_saving'].mean():>8.1f}"
              f"{sub['v2_lsm_saving'].mean():>8.1f}"
              f"{sub['v2_lsm_fail'].mean() * 100:>9.2f}"
              f"{sub['corr_end_peak'].mean():>7.2f}")
    print("\n  saving% = (none_exec - policy_exec) / none_exec x 100, averaged over instances")
    print("  corr    = mean corr(endpoint W_m, peak max W_k) across routes "
          "(low => v1 endpoint label misleading)")


def run_dethloff(argv) -> None:
    data_dir, tlim        = _DEFAULT_DATA_DIR, _DEFAULT_TLIM
    n_train, n_test       = _DEFAULT_N_TRAIN, _DEFAULT_N_TEST
    cfail_ratio, max_n    = _DEFAULT_CFAIL_RATIO, None
    out_stem              = _DEFAULT_OUT_STEM
    policies              = list(_DEFAULT_POLICIES)
    n_workers             = os.cpu_count() or 1

    for arg in argv:
        if   arg.startswith("dir="):      data_dir    = arg[4:]
        elif arg.startswith("tlim="):     tlim        = float(arg[5:])
        elif arg.startswith("n_train="):  n_train     = int(arg[8:])
        elif arg.startswith("n_test="):   n_test      = int(arg[7:])
        elif arg.startswith("cfail="):    cfail_ratio = float(arg[6:])
        elif arg.startswith("max="):      max_n       = int(arg[4:])
        elif arg.startswith("out="):      out_stem    = arg[4:]
        elif arg.startswith("workers="):  n_workers   = int(arg[8:])
        elif arg.startswith("policies="): policies    = arg[9:].split(",")
        else:
            print(f"  Unknown argument '{arg}' — ignored")

    files = sorted(glob.glob(str(Path(data_dir) / "*.vrpspd"))) or \
            sorted(glob.glob(str(Path(data_dir) / "*.txt")))
    if not files:
        print(f"ERROR: no instance files found in '{data_dir}'")
        return
    if max_n:
        files = files[:max_n]
    n_workers = min(n_workers, len(files))

    W = 100
    print("=" * W)
    print("  OTR-2.0 EVALUATION — v2 (LSM) vs fallback vs OTR v1 vs no-handoff on Dethloff SVRPSPD")
    print("=" * W)
    print(f"  instances={len(files)}  policies={policies}  workers={n_workers}/{os.cpu_count()}")
    print(f"  ALNS tlim={tlim}s/policy  n_train={n_train}  n_test={n_test}  "
          f"Cfail/omegaF={cfail_ratio}  omegaF/omegaV={OMEGA_RATIO}")
    print("-" * W)

    all_rows = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_run_instance, f, tlim, n_train, n_test,
                        cfail_ratio, policies): Path(f).stem
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
        print("  No results produced — nothing written.")
        return

    df = pd.DataFrame(all_rows)
    _print_summary_dethloff(df, policies)

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / (out_stem + ".csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Wrote {csv_path}  ({len(df)} rows x {len(df.columns)} columns)")
    try:
        df.to_excel(RESULTS_DIR / (out_stem + ".xlsx"), index=False)
        print(f"  Wrote {RESULTS_DIR / (out_stem + '.xlsx')}")
    except Exception as exc:
        print(f"  (Excel skipped: {exc})")


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]
    mode = args[0] if args else "synthetic"
    if mode == "synthetic":
        run_synthetic()
    elif mode == "dethloff":
        run_dethloff(args[1:])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

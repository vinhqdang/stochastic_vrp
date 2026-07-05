#!/usr/bin/env python3
"""
run_otr2_sensitivity.py — Comprehensive sensitivity study for OTR-2.0.

Sweeps (synthetic generators, all policies incl. the clairvoyant oracle):

  cost_ratio   Cfail/omegaF in {2, 5, 10, 20}     x {ctd, milk_run} x seeds
  n_train      N_hist in {200, 500, 1000, 5000, 20000}   (LSM vs fallback)
  route_len    m in {6, 12, 24, 48}
  family       increment family in {gamma, lognormal, studentt}
  correlation  common-factor loading rho in {0.0, 0.3, 0.6, 0.9}

Policies per cell:
  none / v1_myo / v1_tun / fb_tun / v2_lsm / oracle
Reported per cell: mean cost, saving%, gap-to-oracle%, plus a paired
Wilcoxon signed-rank test of v2 vs the strongest competitor across seeds.

Output: results/results_otr2_sensitivity.csv (+ per-sweep summary JSON).

Usage:
    python scripts/run_otr2_sensitivity.py [sweep ...]
    (default: all sweeps)
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

_SCRIPTS = Path(__file__).resolve().parent
_WDRO    = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))

from core.otr import fit_otr, tau_myopic, tune_tau_fast, simulate_fast
from core.otr2 import (
    fit_lsm, fit_otr_peak, calibrate_B_empirical_peak,
    simulate_v2, simulate_oracle,
)

RESULTS_DIR = _WDRO / "results"
SEEDS       = list(range(5))
N_TRAIN     = 10_000
N_TEST      = 12_000
M_DEFAULT   = 12
OMEGA_F     = 1.0
POLICIES    = ["none", "v1_myo", "v1_tun", "fb_tun", "v2_lsm", "oracle"]


# ═══════════════════════════════════════════════════════════════════════════════
# Generators — g[s, k] = net increment (pickup - delivery) of stop k, scenario s
# ═══════════════════════════════════════════════════════════════════════════════

def _increments(N, m, rng, family="gamma", cv=0.5, mean=1.0):
    """Non-negative stochastic volumes with the requested tail family."""
    if family == "gamma":
        k = 1.0 / cv**2
        return rng.gamma(k, mean / k, (N, m))
    if family == "lognormal":
        s2 = np.log(1 + cv**2)
        return rng.lognormal(np.log(mean) - 0.5 * s2, np.sqrt(s2), (N, m))
    if family == "studentt":
        nu = 4.0
        s = cv * mean * np.sqrt((nu - 2) / nu)
        return np.clip(mean + s * rng.standard_t(nu, (N, m)), 0.0, None)
    raise ValueError(family)


def gen_ctd(N, m, rng, family="gamma", rho=0.0):
    """Collect-then-deliver: pickups first, identical volumes delivered after.
    Endpoint W_m == 0 always; peak = total collected."""
    half = m // 2
    p = _increments(N, half, rng, family)
    if rho > 0.0:
        f = rng.gamma(1.0 / rho**2, rho**2, (N, 1))    # mean-1 common factor
        p = p * (1 - rho) + p * rho * f
    return np.concatenate([p, -p[:, ::-1]], axis=1)


def gen_milk(N, m, rng, family="gamma", rho=0.0):
    """Milk-run ramp + regime switching; mostly positive drift."""
    regime = rng.choice([0.7, 1.6], size=N, p=[0.7, 0.3])
    ramp = 0.5 + np.arange(1, m + 1) / m
    x = _increments(N, m, rng, family)
    if rho > 0.0:
        f = rng.gamma(1.0 / rho**2, rho**2, (N, 1))
        x = x * (1 - rho) + x * rho * f
    return regime[:, None] * ramp[None, :] * x - 0.45


GENS = {"ctd": gen_ctd, "milk_run": gen_milk}


# ═══════════════════════════════════════════════════════════════════════════════
# One experimental cell
# ═══════════════════════════════════════════════════════════════════════════════

def run_cell(gen, seed, *, m=M_DEFAULT, n_train=N_TRAIN, n_test=N_TEST,
             cfail_ratio=5.0, family="gamma", rho=0.0) -> dict:
    Cfail = cfail_ratio * OMEGA_F
    rng_tr = np.random.default_rng(10_007 * seed + 1)
    rng_te = np.random.default_rng(10_007 * seed + 2)
    g_tr = gen(n_train, m, rng_tr, family=family, rho=rho)
    g_te = gen(n_test,  m, rng_te, family=family, rho=rho)
    B = calibrate_B_empirical_peak(g_tr, alpha=0.10)

    v1_models = fit_otr(g_tr, B)
    tau_myo   = tau_myopic(OMEGA_F, Cfail)
    tau_v1    = tune_tau_fast(g_tr, B, v1_models, OMEGA_F, Cfail) if v1_models else tau_myo
    fb_models = fit_otr_peak(g_tr, B)
    tau_fb    = tune_tau_fast(g_tr, B, fb_models, OMEGA_F, Cfail) if fb_models else tau_myo
    cm        = fit_lsm(g_tr, B, OMEGA_F, Cfail)

    res = {
        "none":   simulate_fast(g_te, B, 1.0,     OMEGA_F, Cfail, v1_models),
        "v1_myo": simulate_fast(g_te, B, tau_myo, OMEGA_F, Cfail, v1_models),
        "v1_tun": simulate_fast(g_te, B, tau_v1,  OMEGA_F, Cfail, v1_models),
        "fb_tun": simulate_fast(g_te, B, tau_fb,  OMEGA_F, Cfail, fb_models),
        "v2_lsm": simulate_v2(g_te, B, OMEGA_F, Cfail, cm),
        "oracle": simulate_oracle(g_te, B, OMEGA_F, Cfail),
    }
    none_cost = res["none"]["mean_cost"]
    orc_cost  = res["oracle"]["mean_cost"]
    out = {}
    for lbl in POLICIES:
        out[f"{lbl}_cost"] = res[lbl]["mean_cost"]
        out[f"{lbl}_fail"] = res[lbl]["fail_rate"]
        out[f"{lbl}_ho"]   = res[lbl]["handoff_rate"]
        if lbl not in ("none",):
            out[f"{lbl}_saving"] = 100.0 * (none_cost - res[lbl]["mean_cost"]) / max(none_cost, 1e-12)
        if lbl not in ("none", "oracle"):
            # optimality gap: how much of the oracle's achievable saving is missed
            denom = max(none_cost - orc_cost, 1e-12)
            out[f"{lbl}_gap"] = 100.0 * (res[lbl]["mean_cost"] - orc_cost) / denom
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Sweeps
# ═══════════════════════════════════════════════════════════════════════════════

def sweep_cost_ratio():
    for scen, gen in GENS.items():
        for r in (2.0, 5.0, 10.0, 20.0):
            for seed in SEEDS:
                yield {"sweep": "cost_ratio", "scenario": scen, "x": r, "seed": seed}, \
                      dict(cfail_ratio=r)


def sweep_n_train():
    for scen, gen in GENS.items():
        for n in (200, 500, 1000, 5000, 20000):
            for seed in SEEDS:
                yield {"sweep": "n_train", "scenario": scen, "x": n, "seed": seed}, \
                      dict(n_train=n)


def sweep_route_len():
    for scen, gen in GENS.items():
        for m in (6, 12, 24, 48):
            for seed in SEEDS:
                yield {"sweep": "route_len", "scenario": scen, "x": m, "seed": seed}, \
                      dict(m=m)


def sweep_family():
    for scen, gen in GENS.items():
        for fam in ("gamma", "lognormal", "studentt"):
            for seed in SEEDS:
                yield {"sweep": "family", "scenario": scen, "x": fam, "seed": seed}, \
                      dict(family=fam)


def sweep_correlation():
    for scen, gen in GENS.items():
        for rho in (0.0, 0.3, 0.6, 0.9):
            for seed in SEEDS:
                yield {"sweep": "correlation", "scenario": scen, "x": rho, "seed": seed}, \
                      dict(rho=rho)


SWEEPS = {
    "cost_ratio":  sweep_cost_ratio,
    "n_train":     sweep_n_train,
    "route_len":   sweep_route_len,
    "family":      sweep_family,
    "correlation": sweep_correlation,
}


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    which = sys.argv[1:] or list(SWEEPS)
    rows = []
    t0 = time.time()
    for sw in which:
        print(f"── sweep: {sw} " + "─" * 60, flush=True)
        for meta, kw in SWEEPS[sw]():
            gen = GENS[meta["scenario"]]
            cell = run_cell(gen, meta["seed"], **kw)
            rows.append({**meta, **cell})
            if meta["seed"] == SEEDS[-1]:
                sub = [r for r in rows
                       if r["sweep"] == sw and r["scenario"] == meta["scenario"]
                       and r["x"] == meta["x"]]
                mean = lambda k: float(np.mean([r[k] for r in sub]))
                print(f"  {meta['scenario']:<10} x={str(meta['x']):<8} "
                      f"save%: v1_tun={mean('v1_tun_saving'):6.1f} "
                      f"fb={mean('fb_tun_saving'):6.1f} "
                      f"v2={mean('v2_lsm_saving'):6.1f} "
                      f"oracle={mean('oracle_saving'):6.1f}  "
                      f"gap%: v2={mean('v2_lsm_gap'):5.1f} fb={mean('fb_tun_gap'):5.1f}",
                      flush=True)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / "results_otr2_sensitivity.csv"
    df.to_csv(out_csv, index=False)

    # paired significance: v2 vs strongest competitor, per sweep x scenario x x
    sig_rows = []
    for (sw, scen, x), grp in df.groupby(["sweep", "scenario", "x"]):
        comp = min(("v1_tun", "fb_tun"), key=lambda l: grp[f"{l}_cost"].mean())
        d = grp[f"{comp}_cost"].values - grp["v2_lsm_cost"].values
        if np.allclose(d, 0):
            w_p = 1.0
        else:
            try:
                w_p = float(sps.wilcoxon(d, alternative="greater").pvalue)
            except ValueError:
                w_p = float("nan")
        sig_rows.append({
            "sweep": sw, "scenario": scen, "x": x,
            "competitor": comp,
            "v2_cost_mean":   float(grp["v2_lsm_cost"].mean()),
            "comp_cost_mean": float(grp[f"{comp}_cost"].mean()),
            "v2_saving_mean": float(grp["v2_lsm_saving"].mean()),
            "v2_gap_mean":    float(grp["v2_lsm_gap"].mean()),
            "wilcoxon_p_v2_better": w_p,
        })
    sig = pd.DataFrame(sig_rows)
    sig_csv = RESULTS_DIR / "results_otr2_sensitivity_tests.csv"
    sig.to_csv(sig_csv, index=False)

    n_sig = int((sig["wilcoxon_p_v2_better"] < 0.05).sum())
    print(f"\n  cells where v2 beats the strongest tuned competitor at p<0.05: "
          f"{n_sig}/{len(sig)}")
    print(f"  mean v2 optimality gap: {df['v2_lsm_gap'].mean():.1f}% "
          f"(fallback {df['fb_tun_gap'].mean():.1f}%, v1 tuned {df['v1_tun_gap'].mean():.1f}%)")
    print(f"  Wrote {out_csv}")
    print(f"  Wrote {sig_csv}")
    print(f"  wall time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

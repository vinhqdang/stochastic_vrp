#!/usr/bin/env python3
"""
run_baton_r1.py — supplementary experiments for the BATON manuscript
(first revision). Each mode writes one CSV under results/r1/.

Modes
-----
dependence  demand dependence: Gaussian-copula rho in {0,0.3,0.6,0.9}
            and a route-level multiplicative day factor (Dethloff,
            Det+SAA gates; plans fixed)
shape       monotonicity diagnostic of the continuation value under
            dependence (unconstrained binned means on 50k paths)
daytype     non-exchangeable days: normal vs promotion days; pooled,
            day-type-aware and stale fits
fresh       bias of the fresh-start value F_k: in-sample, out-of-sample
            and near-exact (Salhi-Nagy + Dethloff SAA)
reset       exact residual-capacity reset after a depot return
budget      training-data budget N in {100..20000}
timing      single-thread offline fit and online decision latency
pool        capped standby pool with reservation cost; shadow price
regret      over-triggering of the myopic rule vs near-exact optimum
synthetic   synthetic scenarios with a flat-priced depot return added
exact       exact grid-convolution DP under independent demands (rho = 0)
            against BATON and the plug-in references

Usage (from svrpspd_wdro/):
    python scripts/run_baton_r1.py <mode> [workers=4] [max=N]
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_WDRO = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_SCRIPTS))

from core.costs import (  # noqa: E402
    LastMileCosts, route_cost_schedules, restock_schedule,
    fit_lsm_general, simulate_v2_general, fit_lsm_actions, simulate_actions,
    _fresh_value, fit_rollout, _simulate_costs_general,
    tune_tau_general, simulate_tau_general, oracle_costs_general,
)
from core.otr2 import (  # noqa: E402
    fit_otr_peak, calibrate_B_empirical_peak, _overflow_step,
)
from core.dp_exec import fit_dp  # noqa: E402
from core.extra_policies import (  # noqa: E402
    fit_dp_actions, fit_lsm_actions_exact, simulate_actions_exact,
    fit_threshold_k, simulate_threshold_k, cvar,
    fit_dp_actions_cf, simulate_actions_cf, fit_lsm_actions_cf,
)
from dethloff_runner import parse_dethloff, sample_demands, CV, DIST, ALPHA  # noqa: E402
import run_realistic_eval as rre  # noqa: E402

RES = _WDRO / "results"
OUT = RES / "r1"
OUT.mkdir(parents=True, exist_ok=True)
DATA = _WDRO / "data"
COSTS = LastMileCosts()


# ═══════════════════════════════════════════════════════════════════════════
# shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def instances(fam: str) -> list[Path]:
    return sorted(Path(DATA / fam).glob("*.vrpspd")) or \
        sorted(Path(DATA / fam).glob("*.txt"))


def load(path: Path, plans_sub: str = "plans"):
    D, dem, Q, n, scale = parse_dethloff(str(path))
    sol = json.loads((RES / plans_sub / f"{path.stem}.json").read_text())
    dbar = np.array(sol["dbar"], float)
    pbar = np.array(sol["pbar"], float)
    return D, Q, n, scale, dbar, pbar, sol["res"]


def scen(dbar, pbar, N, seed, rho=0.6, dfac=0.0, dscale=1.0, pscale=1.0):
    """(d, p) scenario matrices. rho: copula equicorrelation within the
    delivery and within the pickup vector; dfac > 0 multiplies every
    demand of a day by a common lognormal factor (sd dfac, mean 1)."""
    n = len(dbar)
    rng = np.random.default_rng(seed)
    d = sample_demands(dbar * dscale, n, N, CV, DIST, rng, rho=rho)
    p = sample_demands(pbar * pscale, n, N, CV, DIST, rng, rho=rho)
    if dfac > 0:
        s2 = np.log(1 + dfac ** 2)
        z = rng.lognormal(-0.5 * s2, np.sqrt(s2), N)[:, None]
        d, p = d * z, p * z
    return d, p


def route_setup(route, dbar, Q, D, scale, g_train):
    r = np.array(route)
    B = float(Q - dbar[r].sum())
    if B <= 0:
        B = calibrate_B_empirical_peak(g_train, alpha=1 - ALPHA)
    H, E = route_cost_schedules(route, D, scale, COSTS)
    R = restock_schedule(route, D, scale, COSTS)
    return B, H, E, R


def baton_full(g_tr, B, H, E, R):
    """BATON with deployment selection; returns (policy_fn, use_actions)."""
    cm = fit_lsm_general(g_tr, B, H, E)
    am = fit_lsm_actions(g_tr, B, H, E, R)
    a = simulate_actions(g_tr, B, H, E, R, am)["mean_cost"]
    h = simulate_v2_general(g_tr, B, H, E, cm)["mean_cost"]
    return cm, am, a <= h


def run_pool(fn, jobs, workers):
    rows = []
    if workers <= 1:
        for j in jobs:
            rows.extend(fn(*j))
        return rows
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_star, [(fn, j) for j in jobs]):
            rows.extend(r)
    return rows


def _star(a):
    fn, j = a
    return fn(*j)


def _plan_eval(route_list, dbar, Q, D, scale, tr, te, xl):
    """Full policy roster (run_realistic_eval) summed over a plan's routes."""
    agg, daily = {}, {}
    for route in route_list:
        if not route:
            continue
        res, _, _ = rre._eval_route_realistic(route, dbar, Q, D, scale, COSTS,
                                              tr[0], tr[1], te[0], te[1],
                                              xl[0], xl[1])
        for k, v in res.items():
            agg[k] = agg.get(k, 0.0) + v["mean_cost"]
            daily[k] = daily.get(k, 0.0) + v["costs"]
    return agg, daily


def _sv(agg, lbl):
    return 100 * (agg["none"] - agg[lbl]) / max(agg["none"], 1e-9)


KEY = ["fb_tau", "thr_k", "thr2", "dp3_n", "ro_theta", "pi3", "restock",
       "v2_lsm", "v2_act", "v2_cf", "dp_n", "dp_xl", "dp_xl3", "oracle",
       "oracle3"]


# ═══════════════════════════════════════════════════════════════════════════
# dependence
# ═══════════════════════════════════════════════════════════════════════════

DEP_CFG = [("rho0", dict(rho=0.0)), ("rho03", dict(rho=0.3)),
           ("rho06", dict(rho=0.6)), ("rho09", dict(rho=0.9)),
           ("dayfac", dict(rho=0.0, dfac=0.25))]


def _dep_job(path, gates):
    D, Q, n, scale, dbar, pbar, res = load(path)
    seed = rre.stable_seed(path.stem)
    rows = []
    for tag, kw in DEP_CFG:
        tr = scen(dbar, pbar, 1000, seed, **kw)
        te = scen(dbar, pbar, 2000, seed + 99_991, **kw)
        xl = scen(dbar, pbar, 50_000, seed + 424_243, **kw)
        for g in gates:
            agg, daily = _plan_eval(res[g]["plan"], dbar, Q, D, scale, tr, te, xl)
            row = {"Instance": path.stem, "Plan": g, "cfg": tag,
                   "none_rec": agg["none"]}
            for lbl in KEY:
                row[f"{lbl}_saving"] = _sv(agg, lbl)
                row[f"{lbl}_cvar95"] = cvar(daily[lbl])
            row["none_cvar95"] = cvar(daily["none"])
            rows.append(row)
    return rows


def mode_dependence(workers, max_n):
    files = instances("Dethloff")[:max_n]
    rows = run_pool(_dep_job, [(f, ["Det", "SAA"]) for f in files], workers)
    pd.DataFrame(rows).to_csv(OUT / "dependence.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# shape: is E[future cost | W_k] monotone under dependence?
# ═══════════════════════════════════════════════════════════════════════════

SHAPE_CFG = [("0.0", dict(rho=0.0)), ("0.3", dict(rho=0.3)),
             ("0.6", dict(rho=0.6)), ("0.9", dict(rho=0.9)),
             ("dayfac", dict(rho=0.0, dfac=0.25))]
DIP = (0.10, 0.25)  # injected dips, share of the mean continuation cost


def _shape_job(path, gates):
    D, Q, n, scale, dbar, pbar, res = load(path)
    seed = rre.stable_seed(path.stem)
    rows = []
    for tag, kw in SHAPE_CFG:
        rho = tag
        d, p = scen(dbar, pbar, 50_000, seed + 7, **kw)
        for g in gates:
            for route in res[g]["plan"]:
                if len(route) < 3:
                    continue
                r = np.array(route)
                gx = p[:, r] - d[:, r]
                B, H, E, R = route_setup(route, dbar, Q, D, scale, gx)
                N, m = gx.shape
                cum = np.cumsum(gx, axis=1)
                ost = _overflow_step(cum, B)
                fut = np.zeros(N)
                br = ost <= m
                fut[br] = E[np.clip(ost[br] - 1, 0, m - 1)]
                n_pairs = n_viol = 0
                n_dip, n_hit = 0, [0] * len(DIP)
                for k in range(m - 1, 0, -1):
                    al = ost > k
                    if al.sum() < 500:
                        continue
                    W, y = cum[al, k - 1], fut[al]
                    edges = np.unique(np.quantile(W, np.linspace(0, 1, 26)[1:-1]))
                    idx = np.searchsorted(edges, W, side="right")
                    nb = len(edges) + 1
                    cnt = np.bincount(idx, minlength=nb)
                    mu = np.bincount(idx, weights=y, minlength=nb) / np.maximum(cnt, 1)
                    m2 = np.bincount(idx, weights=y * y, minlength=nb) / np.maximum(cnt, 1)
                    se = np.sqrt(np.maximum(m2 - mu ** 2, 0) / np.maximum(cnt, 1))
                    dif = mu[1:] - mu[:-1]
                    sd = np.sqrt(se[1:] ** 2 + se[:-1] ** 2)
                    ok = (cnt[1:] > 0) & (cnt[:-1] > 0) & (sd > 0)
                    n_pairs += int(ok.sum())
                    n_viol += int((ok & (dif < -1.96 * sd)).sum())
                    # power: inject a dip of DIP x the mean continuation cost
                    # into the middle bin and test the pair entering it
                    j = nb // 2
                    if j >= 1 and cnt[j] > 0 and cnt[j - 1] > 0 and y.mean() > 0:
                        n_dip += 1
                        for q, dp_ in enumerate(DIP):
                            dj = (mu[j] - dp_ * float(y.mean())) - mu[j - 1]
                            n_hit[q] += int(dj < -1.96 * sd[j - 1])
                    # continue the backward pass with the near-exact policy
                    from sklearn.isotonic import IsotonicRegression
                    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
                    iso.fit(W, y)
                    pred = iso.predict(cum[:, k - 1])
                    stop = al & (pred > H[k - 1])
                    fut[stop] = H[k - 1]
                rows.append({"Instance": path.stem, "Plan": g, "rho": rho,
                             "m": m, "pairs": n_pairs, "viol": n_viol,
                             "dip_tests": n_dip,
                             **{f"dip_hits_{int(100 * dp_)}": n_hit[q]
                                for q, dp_ in enumerate(DIP)}})
    return rows


def mode_shape(workers, max_n):
    files = instances("Dethloff")[:max_n]
    rows = run_pool(_shape_job, [(f, ["Det", "SAA"]) for f in files], workers)
    pd.DataFrame(rows).to_csv(OUT / "shape.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# daytype: non-exchangeable days
# ═══════════════════════════════════════════════════════════════════════════

PROMO = dict(dscale=1.15, pscale=1.35)
P_PROMO = 0.2


def _mix(dbar, pbar, N, seed):
    """Pooled history of N days, a Binomial(N, 0.2) share of promotion
    days. Returns the pooled matrices and the two typed subsets."""
    rng = np.random.default_rng(seed + 3)
    npro = rng.binomial(N, P_PROMO)
    a = scen(dbar, pbar, N - npro, seed)
    b = scen(dbar, pbar, npro, seed + 1, **PROMO)
    return (np.vstack([a[0], b[0]]), np.vstack([a[1], b[1]])), a, b


def _daytype_job(path, gates):
    D, Q, n, scale, dbar, pbar, res = load(path)
    s = rre.stable_seed(path.stem)
    # the day-type-aware fits see the SAME history as the pooled fit, split
    # by type (~800 normal / ~200 promotion days), so the comparison isolates
    # the day-type information rather than the sample size
    mix, sub_n, sub_p = _mix(dbar, pbar, 1000, s + 11)
    tr = {"normal": sub_n, "promo": sub_p, "mix": mix}
    te = {"normal": scen(dbar, pbar, 2000, s + 99_991),
          "promo": scen(dbar, pbar, 2000, s + 99_997, **PROMO)}
    xl = {"normal": scen(dbar, pbar, 50_000, s + 424_243),
          "promo": scen(dbar, pbar, 50_000, s + 424_249, **PROMO)}
    rows = []
    for g in gates:
        for trk, tek in (("mix", "normal"), ("mix", "promo"), ("normal", "normal"),
                         ("promo", "promo"), ("normal", "promo")):
            agg, _ = _plan_eval(res[g]["plan"], dbar, Q, D, scale,
                                tr[trk], te[tek], xl[tek])
            row = {"Instance": path.stem, "Plan": g, "train": trk, "test": tek,
                   "n_train": len(tr[trk][0])}
            for lbl in ["none"] + KEY:
                row[f"{lbl}_rec"] = agg[lbl]
            rows.append(row)
    return rows


def mode_daytype(workers, max_n):
    files = instances("Dethloff")[:max_n]
    rows = run_pool(_daytype_job, [(f, ["Det", "SAA"]) for f in files], workers)
    pd.DataFrame(rows).to_csv(OUT / "daytype.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# fresh-start value bias + exact reset
# ═══════════════════════════════════════════════════════════════════════════

def _replace_F(models, F):
    return {k: (mdl, float(F[k])) for k, (mdl, _) in models.items()}


def _fresh_job(path, gate, fam):
    D, Q, n, scale, dbar, pbar, res = load(path)
    s = rre.stable_seed(path.stem)
    tr = scen(dbar, pbar, 1000, s)
    te = scen(dbar, pbar, 2000, s + 99_991)
    xl = scen(dbar, pbar, 50_000, s + 424_243)
    ev = scen(dbar, pbar, 50_000, s + 777_777)
    rows = []
    for ri, route in enumerate(res[gate]["plan"]):
        if len(route) < 3:
            continue
        r = np.array(route)
        gtr, gte = tr[1][:, r] - tr[0][:, r], te[1][:, r] - te[0][:, r]
        gxl, gev = xl[1][:, r] - xl[0][:, r], ev[1][:, r] - ev[0][:, r]
        B, H, E, R = route_setup(route, dbar, Q, D, scale, gtr)
        m = len(route)
        am = fit_lsm_actions(gtr, B, H, E, R)
        dp3 = fit_dp_actions(gxl[:25_000], B, H, E, R, g_eval=gxl[25_000:])
        F_in = {k: am[k][1] for k in am}
        F_out = {k: _fresh_value(k, m, gev, B, H, E, R, am) for k in am}
        F_st = {k: dp3[k][1] for k in dp3}
        am_out = _replace_F(am, F_out)
        am_st = _replace_F(am, F_st)
        st = {nm: simulate_actions(gte, B, H, E, R, mm, return_actions=True)
              for nm, mm in (("baton3", am), ("baton3_Fout", am_out),
                             ("baton3_Fstar", am_st), ("dp3", dp3))}
        orc = float(oracle_costs_general(gte, B, H, E).mean())
        react = _simulate_costs_general(gte, B, H * 1e9, E, fit_lsm_general(gtr, B, H * 1e9, E))[0].mean()
        from core.extra_policies import fit_lsm_actions_cf, simulate_actions_cf
        cf = fit_lsm_actions_cf(gtr, B, H, E, R)
        c_cf = simulate_actions_cf(gte, B, H, E, R, cf)["mean_cost"]
        # exact reset
        amx = fit_lsm_actions_exact(gtr, tr[0][:, r], B, H, E, R)
        stx = simulate_actions_exact(gte, te[0][:, r], B, H, E, R, amx)
        # restock usage (number of days with >= 1 return)
        rs = {}
        for nm, mm in (("baton3", am), ("baton3_Fstar", am_st), ("dp3", dp3)):
            rs[nm] = _restock_rate(gte, B, H, E, R, mm)
        ks = sorted(am)
        row = {"fam": fam, "Instance": path.stem, "route": ri, "m": m,
               "reactive": react, "oracle_ho": orc,
               "F_in": np.mean([F_in[k] for k in ks]),
               "F_out": np.mean([F_out[k] for k in ks]),
               "F_star": np.mean([F_st[k] for k in ks]),
               "RF_in": np.mean([R[k - 1] + F_in[k] for k in ks]),
               "RF_star": np.mean([R[k - 1] + F_st[k] for k in ks]),
               "H_mean": float(H[:m - 1].mean()),
               "bias_in": np.mean([F_in[k] - F_st[k] for k in ks]),
               "bias_out": np.mean([F_out[k] - F_st[k] for k in ks]),
               "cost_baton3": st["baton3"][0]["mean_cost"],
               "cost_baton3_Fout": st["baton3_Fout"][0]["mean_cost"],
               "cost_baton3_Fstar": st["baton3_Fstar"][0]["mean_cost"],
               "cost_dp3": st["dp3"][0]["mean_cost"],
               "cost_exact_reset": stx["mean_cost"],
               "cost_cf": c_cf,
               "cost_dp3cf": simulate_actions_cf(gte, B, H, E, R, fit_dp_actions_cf(gxl[:25_000], B, H, E, R, g_eval=gxl[25_000:]))["mean_cost"],
               "cost_ho": simulate_v2_general(gte, B, H, E, fit_lsm_general(gtr, B, H, E))["mean_cost"],
               "rs_baton3": rs["baton3"], "rs_baton3_Fstar": rs["baton3_Fstar"],
               "rs_dp3": rs["dp3"], "rs_exact_reset": stx["restock_rate"]}
        rows.append(row)
    return rows


def _restock_rate(g, B, H, E, R, models):
    """Share of days with at least one depot return under a 3-action model."""
    N, m = g.shape
    W = np.zeros(N)
    stopped = np.zeros(N, bool)
    used = np.zeros(N, bool)
    for k_idx in range(m - 1):
        act = ~stopped
        W[act] += g[act, k_idx]
        em = act & (W > B)
        stopped |= em
        alive = act & ~em
        ent = models.get(k_idx + 1)
        if ent is None or not alive.any():
            continue
        mdl, F = ent
        idx = np.where(alive)[0]
        chat = np.asarray(mdl.predict(W[idx]))
        v_rs = R[k_idx] + F
        do_ho = (H[k_idx] < chat) & (H[k_idx] <= v_rs)
        do_rs = (v_rs < chat) & ~do_ho
        stopped[idx[do_ho]] = True
        W[idx[do_rs]] = 0.0
        used[idx[do_rs]] = True
    return float(used.mean())


def mode_fresh(workers, max_n):
    jobs = [(f, "Det", "SalhiNagy") for f in instances("SalhiNagy")[:max_n]]
    jobs += [(f, "SAA", "Dethloff") for f in instances("Dethloff")[:max_n]]
    jobs += [(f, "Det", "City") for f in instances("City")[:max_n]]
    rows = run_pool(_fresh_job, jobs, workers)
    pd.DataFrame(rows).to_csv(OUT / "fresh.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# training-data budget
# ═══════════════════════════════════════════════════════════════════════════

BUDGETS = [100, 250, 500, 1000, 2000, 5000, 20000]


def _budget_job(path, gate, fam, plans_sub):
    D, Q, n, scale, dbar, pbar, res = load(path, plans_sub)
    s = rre.stable_seed(path.stem)
    te = scen(dbar, pbar, 2000, s + 99_991)
    xl = scen(dbar, pbar, 50_000, s + 424_243)
    big = scen(dbar, pbar, max(BUDGETS), s + 13)
    rows = []
    acc = {}
    for route in res[gate]["plan"]:
        if not route:
            continue
        r = np.array(route)
        gte = te[1][:, r] - te[0][:, r]
        gxl = xl[1][:, r] - xl[0][:, r]
        gbig = big[1][:, r] - big[0][:, r]
        B, H, E, R = route_setup(route, dbar, Q, D, scale, gbig[:1000])
        base = {"none": _simulate_costs_general(gte, B, H * 1e9, E, None, tau=1.0,
                                                prob_models=fit_otr_peak(gbig[:1000], B))[0].mean(),
                "dp_xl": simulate_v2_general(gte, B, H, E, fit_dp(gxl, B, H, E))["mean_cost"],
                "dp_xl3": simulate_actions_cf(gte, B, H, E, R, fit_dp_actions_cf(gxl[:25_000], B, H, E, R, g_eval=gxl[25_000:]))["mean_cost"],
                "oracle": float(oracle_costs_general(gte, B, H, E).mean())}
        for N in BUDGETS:
            gtr = gbig[:N]
            fb = fit_otr_peak(gtr, B)
            tau = tune_tau_general(gtr, B, H, E, fb)
            cm, am, use = baton_full(gtr, B, H, E, R)
            v = {"fb_tau": simulate_tau_general(gte, B, H, E, fb, tau)["mean_cost"],
                 "v2_lsm": simulate_v2_general(gte, B, H, E, cm)["mean_cost"],
                 "v2_act": (simulate_actions(gte, B, H, E, R, am) if use else
                            simulate_v2_general(gte, B, H, E, cm))["mean_cost"],
                 "dp_n": simulate_v2_general(gte, B, H, E, fit_dp(gtr, B, H, E))["mean_cost"]}
            v.update(base)
            for lbl, c in v.items():
                acc[(N, lbl)] = acc.get((N, lbl), 0.0) + c
    for N in BUDGETS:
        none = acc[(N, "none")]
        row = {"fam": fam, "Instance": path.stem, "Plan": gate, "N": N}
        for lbl in ("fb_tau", "v2_lsm", "v2_act", "dp_n", "dp_xl", "dp_xl3", "oracle"):
            row[f"{lbl}_saving"] = 100 * (none - acc[(N, lbl)]) / max(none, 1e-9)
        rows.append(row)
    return rows


def mode_budget(workers, max_n):
    jobs = [(f, g, "Dethloff", "plans") for f in instances("Dethloff")[:max_n]
            for g in ("Det", "SAA")]
    jobs += [(f, "Det", "City", "plans") for f in instances("City")[:max_n]]
    rows = run_pool(_budget_job, jobs, workers)
    pd.DataFrame(rows).to_csv(OUT / "budget.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# timing (single thread)
# ═══════════════════════════════════════════════════════════════════════════

def mode_timing(workers, max_n):
    rows = []
    sel = [(f, "SAA", "plans") for f in instances("Dethloff")[:10]]
    sel += [(f, "Det", "plans") for f in instances("City")
            if f.stem.endswith(("-100-1", "-200-1"))]
    for path, gate, sub in sel:
        D, Q, n, scale, dbar, pbar, res = load(path, sub)
        s = rre.stable_seed(path.stem)
        tr = scen(dbar, pbar, 1000, s)
        te = scen(dbar, pbar, 2000, s + 99_991)
        xl = scen(dbar, pbar, 50_000, s + 424_243)
        for route in res[gate]["plan"]:
            if len(route) < 3:
                continue
            _, _, tfit = rre._eval_route_realistic(route, dbar, Q, D, scale, COSTS,
                                                   tr[0], tr[1], te[0], te[1],
                                                   xl[0], xl[1])
            r = np.array(route)
            gtr = tr[1][:, r] - tr[0][:, r]
            B, H, E, R = route_setup(route, dbar, Q, D, scale, gtr)
            am = fit_lsm_actions(gtr, B, H, E, R)
            # online latency of ONE decision for one vehicle: (a) through the
            # scikit-learn predict call, (b) through the fitted breakpoints
            k = max(1, len(route) // 2)
            iso, F = am[k]
            ws = np.random.default_rng(0).normal(0, B / 3, 2000)
            t0 = time.perf_counter()
            for w in ws:
                c = iso.predict([w])[0]
                _ = min(H[k - 1], R[k - 1] + F) < c
            lat_sk = (time.perf_counter() - t0) / len(ws)
            xs, ys = iso.X_thresholds_, iso.y_thresholds_
            t0 = time.perf_counter()
            for w in ws:
                c = np.interp(w, xs, ys)
                _ = min(H[k - 1], R[k - 1] + F) < c
            lat_np = (time.perf_counter() - t0) / len(ws)
            row = {"Instance": path.stem, "m": len(route),
                   "lat_sklearn_us": 1e6 * lat_sk, "lat_interp_us": 1e6 * lat_np}
            row.update({f"{kk}_fit_s": v for kk, v in tfit.items() if not kk.startswith("_")})
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "timing.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# capped standby pool with reservation (holding) cost
# ═══════════════════════════════════════════════════════════════════════════

def _run_primary(g, B, Hd, Hb, E, R, models, use_actions):
    """Execute the primary policy until its FIRST handoff request.

    Returns per-day arrays: k_req (1-indexed stop, m+1 if none), cost
    accrued before/at completion excluding a requested handoff, the
    state W at the request, and the billed handoff price."""
    N, m = g.shape
    W = np.zeros(N)
    cost = np.zeros(N)
    stopped = np.zeros(N, bool)
    k_req = np.full(N, m + 1)
    W_req = np.zeros(N)
    h_bill = np.zeros(N)
    for k_idx in range(m):
        k = k_idx + 1
        act = ~stopped
        if not act.any():
            break
        W[act] += g[act, k_idx]
        em = act & (W > B)
        cost[em] += E[k_idx]
        stopped |= em
        if k == m:
            break
        alive = act & ~em
        ent = models.get(k)
        if ent is None or not alive.any():
            continue
        idx = np.where(alive)[0]
        if use_actions:
            mdl, F = ent
            chat = np.asarray(mdl.predict(W[idx]))
            v_rs = R[k_idx] + F
        else:
            mdl = ent
            chat = np.asarray(mdl.predict(W[idx]))
            v_rs = np.inf
        do_ho = (Hd[k_idx] < chat) & (Hd[k_idx] <= v_rs)
        do_rs = (v_rs < chat) & ~do_ho
        h = idx[do_ho]
        k_req[h] = k
        W_req[h] = W[h]
        h_bill[h] = Hb[k_idx]
        stopped[h] = True
        rsi = idx[do_rs]
        cost[rsi] += R[k_idx]
        W[rsi] = 0.0
    return k_req, cost, W_req, h_bill


def _run_fallback(g, B, E, R, fb_models, k_start, W0, use_actions):
    """Cost of the rest of the day for a route whose handoff request at
    stop k_start was refused, from state W0, under the fallback menu
    {continue, depot return}. The fallback may act at the refusal stop
    itself (a return there), then continues from stop k_start+1."""
    N, m = g.shape
    W = W0.copy()
    cost = np.zeros(N)
    stopped = k_start >= m
    if use_actions:
        # decision at the refusal stop
        for k in np.unique(k_start[~stopped]):
            sel = np.where((~stopped) & (k_start == k))[0]
            mdl, F = fb_models[int(k)]
            chat = np.asarray(mdl.predict(W[sel]))
            rs = (R[int(k) - 1] + F) < chat
            cost[sel[rs]] += R[int(k) - 1]
            W[sel[rs]] = 0.0
    for k_idx in range(m):
        k = k_idx + 1
        live = (~stopped) & (k > k_start)
        if not live.any():
            continue
        W[live] += g[live, k_idx]
        em = live & (W > B)
        cost[em] += E[k_idx]
        stopped = stopped | em
        if k == m:
            break
        alive = live & ~em
        if not use_actions or not alive.any():
            continue
        mdl, F = fb_models[k]
        idx = np.where(alive)[0]
        chat = np.asarray(mdl.predict(W[idx]))
        do_rs = (R[k_idx] + F) < chat
        cost[idx[do_rs]] += R[k_idx]
        W[idx[do_rs]] = 0.0
    return cost


def _prep_lambda(routes_data, lam, split):
    """Per-route request times, pre-request costs, billed handoff prices and
    refusal costs for one shadow price (independent of the pool size)."""
    Ts, base, hbs, cfbs = [], 0.0, [], []
    for rd in routes_data:
        g = rd[split]
        mdl, use = rd["models"][lam]
        k_req, c_pre, W_req, hb = _run_primary(g, rd["B"], rd["Hm"] + lam, rd["Hm"],
                                               rd["E"], rd["R"], mdl, use)
        c_fb = _run_fallback(g, rd["B"], rd["E"], rd["R"], rd["fb"], k_req, W_req, True)
        t = np.where(k_req <= g.shape[1], rd["t"][np.minimum(k_req, g.shape[1]) - 1], np.inf)
        Ts.append(t)
        base = base + c_pre
        hbs.append(hb)
        cfbs.append(c_fb)
    return np.vstack(Ts), base, np.vstack(hbs), np.vstack(cfbs)


def _eval_pool(prep, S):
    """Daily recourse under a pool of S reserved vehicles, first come first
    served by request time."""
    T, base, HB, CFB = prep
    order = np.argsort(T, axis=0, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(T.shape[0])[:, None].repeat(T.shape[1], 1), 0)
    fin = np.isfinite(T)
    granted = (rank < S) & fin
    denied = (rank >= S) & fin
    tot = base + np.where(granted, HB, 0.0).sum(0) + np.where(denied, CFB, 0.0).sum(0)
    return tot, granted.sum(0), denied.sum(0)


LAMS = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 1e6]      # 20 = F_sb (pay-per-use pricing)
HOLD = [0.0, 1.0, 2.5, 5.0, 10.0, 20.0]             # reserved holding cost per vehicle-day


def _route_data(path, gate, sub):
    D, Q, n, scale, dbar, pbar, res = load(path, sub)
    s = rre.stable_seed(path.stem)
    tr = scen(dbar, pbar, 1000, s)
    te = scen(dbar, pbar, 2000, s + 99_991)
    rds = []
    ppu = np.zeros(2000)
    for route in res[gate]["plan"]:
        if len(route) < 2:
            continue
        r = np.array(route)
        gtr, gte = tr[1][:, r] - tr[0][:, r], te[1][:, r] - te[0][:, r]
        B, H, E, R = route_setup(route, dbar, Q, D, scale, gtr)
        Hm = H - COSTS.F_standby                    # per-use part of a handoff
        cm, am, use = baton_full(gtr, B, H, E, R)   # pay-per-use benchmark
        ppu += (simulate_actions(gte, B, H, E, R, am) if use else
                simulate_v2_general(gte, B, H, E, cm))["costs"]
        models = {}
        for lam in LAMS:
            cm_l, am_l, use_l = baton_full(gtr, B, Hm + lam, E, R)
            models[lam] = (am_l, True) if use_l else (cm_l, False)
        fb = fit_lsm_actions(gtr, B, np.full_like(H, 1e9), E, R)
        pos = [0] + list(route)
        t = np.cumsum([D[pos[i], pos[i + 1]] / scale for i in range(len(route))])
        rds.append(dict(tr=gtr, te=gte, B=B, Hm=Hm, E=E, R=R, models=models, fb=fb, t=t))
    return rds, ppu


def _pool_rows(rds, ppu, fam, name, gate, S_max):
    prep_tr = {L: _prep_lambda(rds, L, "tr") for L in LAMS}
    prep_te = {L: _prep_lambda(rds, L, "te") for L in LAMS}
    rows = []
    for S in range(0, S_max + 1):
        best = min(LAMS, key=lambda L: _eval_pool(prep_tr[L], S)[0].mean())
        shad, g_s, d_s = _eval_pool(prep_te[best], S)
        naive, _, d_n = _eval_pool(prep_te[0.0], S)
        ppuP, _, _ = _eval_pool(prep_te[20.0], S)
        fbk, _, _ = _eval_pool(prep_te[1e6], S)
        rows.append({"fam": fam, "Instance": name, "Plan": gate, "K": len(rds), "S": S,
                     "shadow_rec": shad.mean(), "naive_rec": naive.mean(),
                     "ppuprice_rec": ppuP.mean(), "fallback_rec": fbk.mean(),
                     "lambda": best, "denied_naive": d_n.mean(),
                     "denied_shadow": d_s.mean(), "granted_shadow": g_s.mean(),
                     "ppu_rec": ppu.mean()})
    return rows


def _pool_job(path, gate, fam, sub):
    rds, ppu = _route_data(path, gate, sub)
    return _pool_rows(rds, ppu, fam, path.stem, gate, len(rds))


def _metro_job(cls, gate):
    rds, ppu = [], np.zeros(2000)
    for f in instances("Dethloff"):
        if f.stem.startswith(cls + "-"):
            r, p_ = _route_data(f, gate, "plans")
            rds += r
            ppu += p_
    return _pool_rows(rds, ppu, "Metro", cls, gate, min(len(rds), 40))


def mode_pool(workers, max_n):
    jobs = [(f, g, "Dethloff", "plans") for f in instances("Dethloff")[:max_n]
            for g in ("Det", "SAA")]
    jobs += [(f, "Det", "City", "plans") for f in instances("City")[:max_n]]
    rows = run_pool(_pool_job, jobs, workers)
    rows += run_pool(_metro_job, [(c, "Det") for c in ("CON3", "CON8", "SCA3", "SCA8")], workers)
    pd.DataFrame(rows).to_csv(OUT / "pool.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# regret of the myopic comparison (Proposition on over-triggering)
# ═══════════════════════════════════════════════════════════════════════════

def _regret_job(path, gate, fam, sub):
    D, Q, n, scale, dbar, pbar, res = load(path, sub)
    s = rre.stable_seed(path.stem)
    te = scen(dbar, pbar, 2000, s + 99_991)
    xl = scen(dbar, pbar, 50_000, s + 424_243)
    tr = scen(dbar, pbar, 1000, s)
    rows = []
    for route in res[gate]["plan"]:
        if len(route) < 3:
            continue
        r = np.array(route)
        gte, gxl, gtr = (te[1][:, r] - te[0][:, r], xl[1][:, r] - xl[0][:, r],
                         tr[1][:, r] - tr[0][:, r])
        B, H, E, R = route_setup(route, dbar, Q, D, scale, gtr)
        m = len(route)
        myo = fit_rollout(gxl, B, H, E)                 # C_fail*p analogue
        opt = fit_lsm_general(gxl, B, H, E)             # near-exact C_k
        cm, am, a_m = _stop_info(gte, B, H, E, myo)
        co, ao, a_o = _stop_info(gte, B, H, E, opt)
        cr, _, _ = _stop_info(gte, B, H * 1e9, E, opt)
        ost = _overflow_step(np.cumsum(gte, axis=1), B)
        prem = a_m < a_o
        clean = ost > m
        # pathwise bound: regret <= E[(H_sigma - L) 1{premature}], L the
        # clairvoyant cost-to-go after sigma (0 on clean days, else the
        # cheapest later handoff before the breach or the breach itself)
        hs = H[np.minimum(a_m, m) - 1]
        L = np.zeros(len(ost))
        for i in np.where(prem & ~clean)[0]:
            sg, T = a_m[i], ost[i]
            later = H[sg:T - 1]              # handoff after stops sg+1..T-1
            L[i] = min(later.min() if len(later) else np.inf, E[T - 1])
        bound = np.where(prem, hs - L, 0.0)
        bound_clean = np.where(prem & clean, hs, 0.0)
        # tuned global threshold (1000 paths) vs BATON-ho (1000 paths)
        fb = fit_otr_peak(gtr, B)
        tau = tune_tau_general(gtr, B, H, E, fb)
        c_thr = simulate_tau_general(gte, B, H, E, fb, tau)["mean_cost"]
        c_ho = simulate_v2_general(gte, B, H, E, fit_lsm_general(gtr, B, H, E))["mean_cost"]
        rows.append({"fam": fam, "Instance": path.stem, "Plan": gate, "m": m,
                     "reactive": cr.mean(), "myopic": cm.mean(), "opt": co.mean(),
                     "regret": cm.mean() - co.mean(),
                     "bound": bound.mean(),
                     "bound_clean": bound_clean.mean(),
                     "p_premature": prem.mean(),
                     "p_premature_clean": (prem & clean).mean(),
                     "p_contained": float((a_m <= a_o).mean()),
                     "H_spread": float(H[0] / H[m - 2]),
                     "thr_cost": c_thr, "ho_cost": c_ho})
    return rows


def _stop_info(g, B, H, E, models):
    c, a = _simulate_costs_general(g, B, H, E, models)
    # recover the stopping stop: first k where the rule fires (handoff) or
    # m+1; breaches do not count as a stop of the rule
    N, m = g.shape
    cum = np.cumsum(g, axis=1)
    stop = np.full(N, m + 1)
    alive = np.ones(N, bool)
    for k in range(1, m):
        Wk = cum[:, k - 1]
        alive &= Wk <= B
        mdl = models.get(k)
        if mdl is None:
            continue
        fire = alive & (stop > m) & (np.asarray(mdl.predict(Wk)) > H[k - 1])
        stop[fire] = k
    return c, a, stop


def mode_regret(workers, max_n):
    jobs = [(f, g, "Dethloff", "plans") for f in instances("Dethloff")[:max_n]
            for g in ("Det", "SAA")]
    jobs += [(f, "Det", "City", "plans") for f in instances("City")[:max_n]]
    rows = run_pool(_regret_job, jobs, workers)
    pd.DataFrame(rows).to_csv(OUT / "regret.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# synthetic scenarios with a flat depot return
# ═══════════════════════════════════════════════════════════════════════════

def mode_synthetic(workers, max_n):
    import run_otr2_eval as roe
    rows = []
    for name, gen, ratio in roe.SCENARIOS_SYNTH:
        for seed in roe.SEEDS:
            g_tr = gen(roe.N_TRAIN_S, roe.M_SYNTH, np.random.default_rng(1_000 * seed + 1))
            g_te = gen(roe.N_TEST_S, roe.M_SYNTH, np.random.default_rng(1_000 * seed + 2))
            B = calibrate_B_empirical_peak(g_tr, alpha=0.10)
            m = roe.M_SYNTH
            H = np.ones(m)
            E = np.full(m, ratio)
            R = np.full(m, 0.5)                      # depot return at half a handoff
            none = float(oracle_costs_general(g_te, B, H * 1e9, E).mean())
            cm, am, use = baton_full(g_tr, B, H, E, R)
            full = (simulate_actions(g_te, B, H, E, R, am) if use else
                    simulate_v2_general(g_te, B, H, E, cm))["mean_cost"]
            ho = simulate_v2_general(g_te, B, H, E, cm)["mean_cost"]
            rows.append({"scenario": name, "seed": seed,
                         "none": none, "baton_ho": ho, "baton": full,
                         "baton_ho_saving": 100 * (none - ho) / none,
                         "baton_saving": 100 * (none - full) / none,
                         "use_actions": use})
    pd.DataFrame(rows).to_csv(OUT / "synthetic_actions.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# exact: grid-convolution dynamic program under independent demands (rho = 0)
# ═══════════════════════════════════════════════════════════════════════════

def _gamma_pmf(mu, h, jmax):
    """Probability mass of a Gamma(1/CV^2, mu CV^2) demand on the cells
    [(j-1/2)h, (j+1/2)h), j = 0..jmax (last cell absorbs the tail)."""
    from scipy import stats as st
    if mu <= 0:
        out = np.zeros(jmax + 1)
        out[0] = 1.0
        return out
    k = 1.0 / CV ** 2
    edges = (np.arange(jmax + 2) - 0.5) * h
    edges[0] = 0.0
    c = st.gamma.cdf(edges, k, scale=mu / k)
    c[-1] = 1.0
    return np.diff(c)


def exact_values(dbar_r, pbar_r, B, H, E, R, G=1500):
    """Exact optimal expected recourse of one route under independent Gamma
    demands, on a W-grid of G cells over [L, B]: no action, handoff only,
    and the three-action menu with the conservative reset W <- 0.
    Returns (none, handoff-only, three-action)."""
    m = len(dbar_r)
    sd = CV * np.sqrt((dbar_r ** 2 + pbar_r ** 2).sum())
    L = min(0.0, -float(np.cumsum(dbar_r - pbar_r).max()) - 6 * sd)
    h = (B - L) / (G - 1)
    x = L + h * np.arange(G)
    i0 = int(round(-L / h))                        # grid index of W = 0
    pmfs = []
    for i in range(m):
        jd = int(np.ceil(dbar_r[i] * (1 + 12 * CV) / h)) + 1
        jp = int(np.ceil(pbar_r[i] * (1 + 12 * CV) / h)) + 1
        pd_ = _gamma_pmf(dbar_r[i], h, jd)
        pp_ = _gamma_pmf(pbar_r[i], h, jp)
        pmfs.append((np.convolve(pp_, pd_[::-1]), jd))   # offsets -jd..jp

    def cont(V, k):
        """C(x_i) = E[cost | W_k = x_i, continue]; stop k+1 increment."""
        pmf, jd = pmfs[k]
        jp = len(pmf) - 1 - jd
        ext = np.concatenate([np.full(jd, V[0]), V, np.full(jp, E[k])])
        return np.convolve(ext, pmf[::-1], mode="valid")

    out = []
    for menu in ("none", "ho", "ho+rs"):
        V = np.zeros(G)                            # alive after stop m
        for k in range(m - 1, 0, -1):              # decision after stop k
            C = cont(V, k)
            if menu == "none":
                V = C
            elif menu == "ho":
                V = np.minimum(C, H[k - 1])
            else:
                V = np.minimum(np.minimum(C, H[k - 1]), R[k - 1] + C[i0])
        C0 = cont(V, 0)
        out.append(float(C0[i0]))
    return tuple(out)


EXACT_KEY = ["fb_tau", "thr_k", "thr2", "v2_lsm", "v2_act", "v2_cf",
             "dp_n", "dp3_n", "dp_xl", "dp_xl3"]


def _exact_job(path, gates):
    D, Q, n, scale, dbar, pbar, res = load(path)
    seed = rre.stable_seed(path.stem)
    tr = scen(dbar, pbar, 1000, seed, rho=0.0)
    te = scen(dbar, pbar, 20_000, seed + 99_991, rho=0.0)
    xl = scen(dbar, pbar, 50_000, seed + 424_243, rho=0.0)
    rows = []
    for g in gates:
        for route in res[g]["plan"]:
            if not route:
                continue
            r = np.array(route)
            B, H, E, R = route_setup(route, dbar, Q, D, scale, tr[1][:, r] - tr[0][:, r])
            ex_none, ex_ho, ex_3 = exact_values(dbar[r], pbar[r], B, H, E, R)
            out, _, _ = rre._eval_route_realistic(route, dbar, Q, D, scale, COSTS,
                                                  tr[0], tr[1], te[0], te[1],
                                                  xl[0], xl[1])
            row = {"Instance": path.stem, "Plan": g, "m": len(route),
                   "exact_none": ex_none, "exact_ho": ex_ho, "exact_3": ex_3,
                   "none_rec": out["none"]["mean_cost"]}
            for lbl in EXACT_KEY:
                row[f"{lbl}_rec"] = out[lbl]["mean_cost"]
            rows.append(row)
    return rows


def mode_exact(workers, max_n):
    files = instances("Dethloff")[:max_n]
    rows = run_pool(_exact_job, [(f, ["Det", "SAA"]) for f in files], workers)
    pd.DataFrame(rows).to_csv(OUT / "exact.csv", index=False)


MODES = {"exact": mode_exact, "dependence": mode_dependence, "shape": mode_shape,
         "daytype": mode_daytype, "fresh": mode_fresh, "budget": mode_budget,
         "timing": mode_timing, "pool": mode_pool, "regret": mode_regret,
         "synthetic": mode_synthetic}


if __name__ == "__main__":
    mode = sys.argv[1]
    workers, max_n = 4, None
    for a in sys.argv[2:]:
        if a.startswith("workers="):
            workers = int(a[8:])
        elif a.startswith("max="):
            max_n = int(a[4:])
    t0 = time.time()
    MODES[mode](workers, max_n)
    print(f"{mode} done in {(time.time() - t0) / 60:.1f} min")

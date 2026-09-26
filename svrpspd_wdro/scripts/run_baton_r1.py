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


KEY = ["fb_tau", "thr_k", "ro_theta", "pi3", "restock", "v2_lsm", "v2_act",
       "v2_cf", "dp_n", "dp_xl", "dp_xl3", "oracle"]


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

def _shape_job(path, gates):
    D, Q, n, scale, dbar, pbar, res = load(path)
    seed = rre.stable_seed(path.stem)
    rows = []
    for rho in (0.0, 0.6, 0.9):
        d, p = scen(dbar, pbar, 50_000, seed + 7, rho=rho)
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
                gain = []
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
                    # continue the backward pass with the near-exact policy
                    from sklearn.isotonic import IsotonicRegression
                    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
                    iso.fit(W, y)
                    pred = iso.predict(cum[:, k - 1])
                    stop = al & (pred > H[k - 1])
                    fut[stop] = H[k - 1]
                rows.append({"Instance": path.stem, "Plan": g, "rho": rho,
                             "m": m, "pairs": n_pairs, "viol": n_viol})
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
    rng = np.random.default_rng(seed + 3)
    npro = rng.binomial(N, P_PROMO)
    a = scen(dbar, pbar, N - npro, seed)
    b = scen(dbar, pbar, npro, seed + 1, **PROMO)
    return np.vstack([a[0], b[0]]), np.vstack([a[1], b[1]])


def _daytype_job(path, gates):
    D, Q, n, scale, dbar, pbar, res = load(path)
    s = rre.stable_seed(path.stem)
    tr = {"normal": scen(dbar, pbar, 1000, s), "promo": scen(dbar, pbar, 1000, s + 5, **PROMO),
          "mix": _mix(dbar, pbar, 1000, s + 11)}
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
            row = {"Instance": path.stem, "Plan": g, "train": trk, "test": tek}
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
    """Cost of the remainder k_start+1..m from state W0 (per day) under the
    fallback menu {continue, depot return} (or pure continuation)."""
    N, m = g.shape
    W = W0.copy()
    cost = np.zeros(N)
    stopped = k_start >= m
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


def _fleet_day_costs(routes_data, S, lam, split):
    """Plan-level daily recourse under a pool of S standby vehicles, FCFS
    by request time. routes_data[i] carries fitted models per lambda."""
    per_route = []
    for rd in routes_data:
        g = rd[split]
        mdl, use = rd["models"][lam]
        k_req, c_pre, W_req, hb = _run_primary(g, rd["B"], rd["Hm"] + lam, rd["Hm"],
                                               rd["E"], rd["R"], mdl, use)
        c_fb = _run_fallback(g, rd["B"], rd["E"], rd["R"], rd["fb"], k_req, W_req, True)
        t = np.where(k_req <= g.shape[1], rd["t"][np.minimum(k_req, g.shape[1]) - 1], np.inf)
        per_route.append((t, c_pre, hb, c_fb))
    T = np.vstack([p[0] for p in per_route])            # routes x days
    order = np.argsort(T, axis=0, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(T.shape[0])[:, None].repeat(T.shape[1], 1), 0)
    granted = (rank < S) & np.isfinite(T)
    denied = (rank >= S) & np.isfinite(T)
    tot = np.zeros(T.shape[1])
    for i, (t, c_pre, hb, c_fb) in enumerate(per_route):
        tot += c_pre + np.where(granted[i], hb, 0.0) + np.where(denied[i], c_fb, 0.0)
    return tot, granted.sum(axis=0), denied.sum(axis=0)


LAMS = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 1e6]


def _pool_job(path, gate, fam, sub):
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
        Hm = H - COSTS.F_standby                    # marginal (reserved pool)
        # pay-per-use benchmark: unlimited pool, day rate billed per use
        cm, am, use = baton_full(gtr, B, H, E, R)
        ppu += (simulate_actions(gte, B, H, E, R, am) if use else
                simulate_v2_general(gte, B, H, E, cm))["costs"]
        models = {}
        for lam in LAMS:
            cm_l, am_l, use_l = baton_full(gtr, B, Hm + lam, E, R)
            models[lam] = (am_l, True) if use_l else (cm_l, False)
        fb = fit_lsm_actions(gtr, B, np.full_like(H, 1e9), E, R)
        pos = [0] + list(route)
        t = np.cumsum([D[pos[i], pos[i + 1]] / scale for i in range(len(route))])
        rds.append(dict(tr=gtr, te=gte, B=B, Hm=Hm, E=E, R=R, models=models,
                        fb=fb, t=t))
    K = len(rds)
    rows = []
    for S in range(0, K + 1):
        # naive: decide at the marginal price, ignore the cap
        naive, g_n, d_n = _fleet_day_costs(rds, S, 0.0, "te")
        # shadow price chosen on TRAINING days for this pool size
        best = min(LAMS, key=lambda L: _fleet_day_costs(rds, S, L, "tr")[0].mean())
        shad, g_s, d_s = _fleet_day_costs(rds, S, best, "te")
        hold = COSTS.F_standby * S
        rows.append({"fam": fam, "Instance": path.stem, "Plan": gate, "K": K,
                     "S": S, "hold": hold,
                     "naive_rec": naive.mean(), "shadow_rec": shad.mean(),
                     "naive_total": hold + naive.mean(),
                     "shadow_total": hold + shad.mean(),
                     "lambda": best, "denied_naive": d_n.mean(),
                     "denied_shadow": d_s.mean(), "granted_shadow": g_s.mean(),
                     "ppu_rec": ppu.mean()})
    return rows


def mode_pool(workers, max_n):
    jobs = [(f, "SAA", "Dethloff", "plans") for f in instances("Dethloff")[:max_n]]
    jobs += [(f, "Det", "City", "plans") for f in instances("City")[:max_n]]
    rows = run_pool(_pool_job, jobs, workers)
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


MODES = {"dependence": mode_dependence, "shape": mode_shape,
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

"""RARECERT revision round 1, task A: exact conditional reference and classical rivals.

Spec: rarecert/baselines.py (exact conditional DP reference, Glasserman-Li two-step IS,
conditional Monte Carlo, subset simulation) and the referee report items 3.1-3.4.

Routes and thresholds come from results/quasar/rare_routes.csv (120 routes, gamma in
1e-3..1e-6, threshold c per route and gamma). Route means are rebuilt with the QUASAR
sampler (seed 21), as in run_rarecert_smis.py, and the route ids are checked.

For each (route, gamma):
* reference: exact_reference (discrete DP, V = 256 and 512, trapezoid on [-6, 9]^2 with
  n_grid nodes per axis, Richardson extrapolation) with its error estimate and wall time;
* SMIS (n_arc = 0, N = 1e6) and CE mixture and single shift: from the CSVs (smis_routes.csv,
  rare_routes.csv), with their build/pilot costs;
* Glasserman-Li two-step IS, crude and shift variants, N_gl simulator evaluations;
* conditional Monte Carlo over the factors, N_factor conditional DP calls;
* subset simulation, n_rep replications at N_level inputs per level.

Cost accounting (95 per cent): samples needed for relative error eps is 1.96^2 RV / eps^2
in the estimator's own cost unit (simulator evaluations, or conditional DP calls for
cond_mc); the CE pilot is added; the wall time is build + samples x seconds per unit
(seconds per unit measured on this machine: the simulator for the simulator-based rivals,
the conditional DP for cond_mc).

For the 20 smallest hubs of results/rarecert/plan_hubs.csv at their threshold c (gamma 1e-3
and 1e-5): exact_reference of P_hub against the hub SMIS estimate in plan_hubs.csv.

Usage (from svrpspd_wdro/):
    python3 scripts/run_rarecert_baselines.py [--workers 3] [--routes N] [--gammas 1e-3,1e-6]
        [--n-grid 41] [--n-factor 20000] [--n-gl 20000] [--n-level 2000] [--n-rep 10]
        [--hubs 20] [--hub-gammas 1e-3,1e-5] [--out-dir DIR]

Outputs (default results/rarecert/): baselines_routes.csv (one row per route and gamma),
baselines_hubs.csv (one row per hub and gamma), baselines_summary.md, baselines.log.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_quasar_rare as rq  # noqa: E402  (CV, RHO)
import run_rarecert_smis as rs  # noqa: E402  (csv loaders, route means, bands)
from quantum.rare import RouteSim  # noqa: E402
from rarecert.baselines import (  # noqa: E402
    cond_mc,
    cond_shift,
    dominant_point,
    exact_reference,
    gl_two_step,
    subset_sim,
)
from rarecert.plan import HubSim  # noqa: E402
from rarecert.smis import stage_scales  # noqa: E402

PLANS_REGION = ROOT / "results" / "plans_Region"
RARE_CSV = ROOT / "results" / "quasar" / "rare_routes.csv"
SMIS_CSV = ROOT / "results" / "rarecert" / "smis_routes.csv"
HUBS_IN_CSV = ROOT / "results" / "rarecert" / "plan_hubs.csv"
OUT_DIR = ROOT / "results" / "rarecert"
GATE = "SAA"
ROUTES_CSV = "baselines_routes.csv"
HUBS_CSV = "baselines_hubs.csv"
SUMMARY_MD = "baselines_summary.md"
LOG_FILE = "baselines.log"
Z2 = 1.96 ** 2
EPS = (0.1, 0.01)
SEED = 53
N_GRID_DEFAULT = 41
N_FACTOR_DEFAULT = 20_000
N_GL_DEFAULT = 20_000
N_LEVEL_DEFAULT = 2_000
N_REP_DEFAULT = 10
HUBS_DEFAULT = 20
GAMMAS_DEFAULT = (1e-3, 1e-4, 1e-5, 1e-6)
HUB_GAMMAS_DEFAULT = (1e-3, 1e-5)
BANDS = ((0.0, 1.0 / 3.0, "[0,1/3]"), (1.0 / 3.0, 0.5, "(1/3,0.5]"),
         (0.5, 0.75, "(0.5,0.75]"), (0.75, 1.0, "(0.75,1]"))
ESTIMATORS = ("smis0", "ce", "single", "gl_crude", "gl_shift", "gl_tuned", "cond_mc", "subset")
EST_LABEL = {"smis0": "SMIS", "ce": "CE mixture", "single": "single shift", "gl_crude": "GL crude",
             "gl_shift": "GL shift", "gl_tuned": "GL tuned shift", "cond_mc": "cond MC",
             "subset": "subset sim"}

ROUTE_COLUMNS = [
    "route_id", "instance", "region", "route_index", "L", "L_tercile", "gamma", "ratio", "rho_band",
    "c", "p_ref", "ref_V_lo", "ref_V_hi", "ref_p_lo", "ref_p_hi", "ref_gap", "ref_quad_err", "ref_err_est",
    "ref_wall_s",
    "ref_dp_calls", "t_dp_s", "t_sim_s", "dominant_E", "build_s",
]
for _e in ESTIMATORS:
    ROUTE_COLUMNS += [f"{_e}_p", f"{_e}_relerr", f"{_e}_RV", f"{_e}_evals", f"{_e}_wall_s",
                      f"{_e}_t_unit_s", f"{_e}_n01", f"{_e}_n001", f"{_e}_t01_s", f"{_e}_t001_s"]
ROUTE_COLUMNS += ["cond_mu1", "cond_mu2", "cond_pilot_RV", "cond_shift_s", "subset_relerr_med",
                  "subset_p_sd_rel", "error"]
HUB_COLUMNS = [
    "name", "region", "n_customers", "n_routes", "R_reach", "gamma", "c", "p_ref", "p_lo", "p_hi",
    "err_est", "p_smis", "rse_smis", "rel_smis", "z_smis", "ref_wall_s", "error",
]


# ============================================================
# Inputs
# ============================================================


def load_routes_ref() -> dict:
    """(route_id, gamma) -> CSV fields of rare_routes.csv (first row per key)."""
    out: dict = {}
    with open(RARE_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["route_id"], float(r["gamma"]))
            if key in out:
                continue
            out[key] = {
                "instance": r["instance"], "route_index": int(r["route_index"]), "L": int(r["L"]),
                "c": float(r["c"]), "p_is": float(r["p_is"]), "RV_is": float(r["RV_is"]),
                "pilot_draws": float(r["pilot_draws"]), "p_single": float(r["p_single"]),
                "RV_single": float(r["RV_single"]),
            }
    return out


def load_smis_ref() -> dict:
    """(route_id, gamma) -> n_arc = 0 SMIS fields of smis_routes.csv."""
    out: dict = {}
    with open(SMIS_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["route_id"], float(r["gamma"]))
            out[key] = {"p": float(r["smis0_p"]), "RV": float(r["smis0_RV"]),
                        "build_s": float(r["smis0_build_s"]), "N": 1_000_000}
    return out


def sim_seconds_per_eval(sim: RouteSim, n: int = 20_000) -> float:
    """Wall time of one simulator evaluation (RouteSim.peak on n standard normal rows)."""
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((n, sim.dim))
    t0 = time.perf_counter()
    sim.peak(x)
    return (time.perf_counter() - t0) / n


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng([SEED] + [int(p) for p in parts])


def _gamma_key(g: float) -> int:
    return int(round(-math.log10(g) * 1000))


def _rho_band(ratio: float) -> str:
    for lo, hi, name in BANDS:
        if (lo <= ratio <= hi) if lo == 0.0 else (lo < ratio <= hi):
            return name
    return "n/a"


# ============================================================
# Per (route, gamma) work
# ============================================================


def _est(p, RV, evals, wall, build, t_unit, p_ref, pilot=0.0):
    """Common estimator summary: relative error, samples for eps, times (seconds)."""
    out = {"p": p, "relerr": abs(p / p_ref - 1.0) if (p_ref > 0 and p == p) else math.nan,
           "RV": RV, "evals": evals, "wall_s": wall, "t_unit_s": t_unit}
    for eps, key in zip(EPS, ("n01", "n001")):
        n_need = (Z2 * RV / (eps * eps) + pilot) if (RV == RV) else math.nan
        out[key] = n_need
    out["t01_s"] = build + out["n01"] * t_unit if out["n01"] == out["n01"] else math.nan
    out["t001_s"] = build + out["n001"] * t_unit if out["n001"] == out["n001"] else math.nan
    return out


def run_route_task(task: dict) -> dict:
    t0 = time.time()
    rid, gamma = task["rid"], task["gamma"]
    ref_meta = task["ref"]
    row = {k: "" for k in ROUTE_COLUMNS}
    try:
        sim = RouteSim(task["dbar"], task["pbar"], cv=rq.CV, rho=rq.RHO)
        c = ref_meta["c"]
        st = stage_scales(sim)
        t_sim = sim_seconds_per_eval(sim)
        ref = exact_reference(sim, c, n=task["n_grid"], V_pair=task["v_pair"])
        p_ref = ref["p_ext"]
        t_dp = ref["wall_s"] / max(ref["dp_calls"], 1)
        tb = time.perf_counter()
        x_star, E_star = dominant_point(sim, c)
        build = time.perf_counter() - tb
        row.update({
            "route_id": rid, "instance": ref_meta["instance"], "region": ref_meta["instance"].split("-")[0],
            "route_index": ref_meta["route_index"], "L": ref_meta["L"], "gamma": gamma,
            "ratio": st["ratio"], "rho_band": _rho_band(st["ratio"]), "c": c,
            "p_ref": p_ref, "ref_V_lo": task["v_pair"][0], "ref_V_hi": task["v_pair"][1],
            "ref_p_lo": ref["p_lo"], "ref_p_hi": ref["p_hi"], "ref_gap": ref["gap"],
            "ref_quad_err": ref["quad_err"], "ref_err_est": ref["err_est"], "ref_wall_s": ref["wall_s"],
            "ref_dp_calls": ref["dp_calls"], "t_dp_s": t_dp, "t_sim_s": t_sim, "dominant_E": E_star,
            "build_s": build,
        })
        ests = {}
        # SMIS, CE and single shift from the CSVs (same settings as run_rarecert_smis.py)
        sm = task["smis"]
        ests["smis0"] = _est(sm["p"], sm["RV"], sm["N"], sm["build_s"] + sm["N"] * t_sim, sm["build_s"],
                             t_sim, p_ref)
        ce = task["ce"]
        ests["ce"] = _est(ce["p_is"], ce["RV_is"], ce["pilot_draws"], ce["pilot_draws"] * t_sim, 0.0,
                          t_sim, p_ref, pilot=ce["pilot_draws"])
        ests["single"] = _est(ce["p_single"], ce["RV_single"], 0.0, 0.0, 0.0, t_sim, p_ref)  # no pilot counted
        # tuned factor shift (cond_shift pilot), shared by GL tuned and cond MC
        tm = time.perf_counter()
        mu_t, rv_pilot, _ = cond_shift(sim, c, V=task["v_cond"], mu_dom=x_star[:2])
        shift_s = time.perf_counter() - tm
        row.update({"cond_mu1": float(mu_t[0]), "cond_mu2": float(mu_t[1]), "cond_pilot_RV": rv_pilot,
                    "cond_shift_s": shift_s})
        # GL two-step: dominant-point factor shift (crude and shift variants) and the tuned shift
        for variant, key, mu_arg in (("crude", "gl_crude", None), ("shift", "gl_shift", None),
                                     ("shift", "gl_tuned", mu_t)):
            tg = time.perf_counter()
            p, RV, _ = gl_two_step(sim, c, task["n_gl"],
                                   _rng(task["idx"], _gamma_key(gamma), {"gl_crude": 1, "gl_shift": 2, "gl_tuned": 4}[key]),
                                   variant=variant, x_star=x_star, mu=mu_arg)
            wall = time.perf_counter() - tg
            n_eval = (task["n_gl"] // 8) * 8
            ests[key] = _est(p, RV, n_eval, wall, build, wall / n_eval, p_ref)
        # conditional MC over the factors (one conditional DP per draw), tuned factor shift
        tc = time.perf_counter()
        p, RV, _ = cond_mc(sim, c, task["n_factor"], _rng(task["idx"], _gamma_key(gamma), 3), shift=True,
                           V=task["v_cond"], mu=mu_t)
        wall = time.perf_counter() - tc
        ests["cond_mc"] = _est(p, RV, task["n_factor"], wall, build + shift_s, wall / task["n_factor"], p_ref)
        # subset simulation: replications
        ps, evs = [], []
        ts = time.perf_counter()
        for k in range(task["n_rep"]):
            p_k, ev_k = subset_sim(sim, c, task["n_level"], _rng(task["idx"], _gamma_key(gamma), 10 + k))
            ps.append(p_k)
            evs.append(ev_k)
        wall_s = time.perf_counter() - ts
        ps = np.asarray(ps)
        evs = np.asarray(evs, dtype=float)
        p_bar = float(ps.mean())
        ev_bar = float(evs.mean())
        var_rel = float(ps.var(ddof=1)) / (p_bar * p_bar) if p_bar > 0 else math.nan
        RV_s = var_rel * ev_bar
        ests["subset"] = _est(p_bar, RV_s, ev_bar, wall_s / task["n_rep"], 0.0,
                              (wall_s / max(evs.sum(), 1.0)), p_ref)
        rel_reps = np.abs(ps / p_ref - 1.0)
        row["subset_relerr_med"] = float(np.median(rel_reps))
        row["subset_p_sd_rel"] = math.sqrt(var_rel) if var_rel == var_rel else math.nan
        for e in ESTIMATORS:
            for k, v in ests[e].items():
                row[f"{e}_{k}"] = v
        row["error"] = ""
    except Exception as exc:  # keep the pool alive; the failure is recorded in the row
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["route_id"] = rid
    row["gamma"] = gamma
    row["_wall"] = time.time() - t0
    return {"kind": "route", "key": (rid, gamma), "row": row, "error": row["error"]}


def run_hub_task(task: dict) -> dict:
    t0 = time.time()
    row = {k: "" for k in HUB_COLUMNS}
    row.update({"name": task["name"], "region": task["region"], "n_customers": task["n_customers"],
                "n_routes": task["n_routes"], "gamma": task["gamma"], "c": task["c"]})
    try:
        hub = HubSim(task["plan"], cv=rq.CV, rho=rq.RHO)
        c = task["c"]
        ref = exact_reference(hub, c, n=task["n_grid"], V_pair=task["v_pair"])
        row.update({"R_reach": ref["R"], "p_ref": ref["p_ext"], "p_lo": ref["p_lo"], "p_hi": ref["p_hi"],
                    "err_est": ref["err_est"], "ref_wall_s": ref["wall_s"]})
        p_s = task["p_smis"]
        rse_s = task["rse_smis"]
        row.update({"p_smis": p_s, "rse_smis": rse_s})
        if ref["p_ext"] > 0 and p_s == p_s:
            row["rel_smis"] = p_s / ref["p_ext"] - 1.0
            tot = math.sqrt((rse_s * p_s) ** 2 + ref["err_est"] ** 2)
            row["z_smis"] = (p_s - ref["p_ext"]) / tot if tot > 0 else math.nan
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return {"kind": "hub", "key": (task["name"], task["gamma"]), "row": row, "error": row["error"]}


def run_task(task: dict) -> dict:
    if task["kind"] == "hub":
        return run_hub_task(task)
    return run_route_task(task)


# ============================================================
# Summary
# ============================================================


def _fl(v):
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return math.nan
    return fv if math.isfinite(fv) else math.nan


def _med(vals):
    a = np.asarray([v for v in (_fl(x) for x in vals) if math.isfinite(v)], dtype=float)
    return float(np.median(a)) if a.size else math.nan


def _mx(vals):
    a = np.asarray([v for v in (_fl(x) for x in vals) if math.isfinite(v)], dtype=float)
    return float(np.max(a)) if a.size else math.nan


def _f(x, fmt=".3g"):
    return "n/a" if not math.isfinite(_fl(x)) else format(_fl(x), fmt)


def _tercile_edges(rows):
    Ls = np.asarray([int(r["L"]) for r in rows], dtype=float)
    return float(np.quantile(Ls, 1 / 3)), float(np.quantile(Ls, 2 / 3))


def _tercile_label(L, edges):
    t1, t2 = edges
    if L <= t1:
        return f"short (L <= {t1:g})"
    if L <= t2:
        return f"medium (L <= {t2:g})"
    return f"long (L > {t2:g})"


def _accuracy_table(lines, groups):
    lines.append("| group | n | " + " | ".join(f"{EST_LABEL[e]} med / max" for e in ESTIMATORS) + " | ref wall s med |")
    lines.append("|---|---|" + "---|" * len(ESTIMATORS) + "---|")
    for name, sub in groups:
        if not sub:
            continue
        cells = [f"{_f(_med([r[e + '_relerr'] for r in sub]), '.2%')} / {_f(_mx([r[e + '_relerr'] for r in sub]), '.2%')}"
                 for e in ESTIMATORS]
        lines.append(f"| {name} | {len(sub)} | " + " | ".join(cells)
                     + f" | {_f(_med([r['ref_wall_s'] for r in sub]), '.3g')} |")


def _cost_table(lines, groups):
    lines.append("| group | " + " | ".join(f"{EST_LABEL[e]} RV / n(0.1) / n(0.01)" for e in ESTIMATORS) + " |")
    lines.append("|---|" + "---|" * len(ESTIMATORS))
    for name, sub in groups:
        if not sub:
            continue
        cells = []
        for e in ESTIMATORS:
            cells.append(f"{_f(_med([r[e + '_RV'] for r in sub]), '.3g')} / "
                         f"{_f(_med([r[e + '_n01'] for r in sub]), '.3g')} / "
                         f"{_f(_med([r[e + '_n001'] for r in sub]), '.3g')}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")


def _time_table(lines, groups):
    lines.append("| group | " + " | ".join(f"{EST_LABEL[e]} t(0.1) / t(0.01) s" for e in ESTIMATORS) + " |")
    lines.append("|---|" + "---|" * len(ESTIMATORS))
    for name, sub in groups:
        if not sub:
            continue
        cells = [f"{_f(_med([r[e + '_t01_s'] for r in sub]), '.3g')} / "
                 f"{_f(_med([r[e + '_t001_s'] for r in sub]), '.3g')}" for e in ESTIMATORS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")


def summary_markdown(rows, hub_rows, n_routes, elapsed, n_fail, args, ref_time_total, v_pair) -> str:
    lines = ["# RARECERT revision task A: exact conditional reference and classical rivals\n"]
    gammas = sorted({r["gamma"] for r in rows}, reverse=True)
    lines.append(
        f"{n_routes} routes from results/quasar/rare_routes.csv, gamma in {', '.join(f'{g:g}' for g in gammas)} "
        f"({len(rows)} route-gamma rows, {n_fail} failed). Wall time {elapsed / 60:.1f} min.\n")
    lines.append(
        "Reference: exact conditional DP (rarecert.baselines.exact_reference). P(F > c | W) is the discrete "
        f"DP at V = {v_pair[0]} and {v_pair[1]} levels (the lower and upper of the pair); the factor integral is a product trapezoid on [-6, 9]^2 "
        f"with {args.n_grid} nodes per axis; the two V values are combined by Richardson extrapolation of order 2. "
        "The reference error estimate is |p_hi - p_ext| + quad_err, where quad_err = |p - p_coarse| is the "
        "difference to the nested coarse grid (every second node), a conservative bound on the quadrature "
        "error. "
        "Relative errors are |p/p_ref - 1|. The reference itself is the discrete model at V levels "
        "extrapolated, not the continuum; its estimated error is reported per row.\n")
    lines.append(
        f"Estimators: SMIS n_arc = 0 (N = 1e6, from smis_routes.csv); CE mixture and single shift "
        f"(from rare_routes.csv, the CE pilot counted in the sample cost, none for the single shift); "
        f"GL two-step (N = {args.n_gl} simulator evaluations, inner draws {8}): crude and shift with the "
        "factor shift at the dominant point (Glasserman-Li), and shift with the factor shift tuned as "
        "below; conditional MC over the factors (N_factor = "
        f"{args.n_factor} conditional DP calls at V = {args.v_cond}) with the factor shift tuned by "
        "cond_shift (per-draw relative variance on a pilot grid of DP values); subset simulation "
        f"({args.n_rep} replications, N_level = {args.n_level}, p0 = 0.1, modified Metropolis width 1.0; "
        "relative error of the replicate mean, RV from the replicate spread).\n")
    lines.append(
        "RV is the relative variance per unit of cost (simulator evaluation, or conditional DP call "
        "for cond MC); n(eps) = 1.96^2 RV / eps^2 units (CE adds its pilot). t(eps) = build + n(eps) x "
        "seconds per unit, with seconds per unit measured on this machine (simulator for the "
        "simulator-based rivals, one conditional DP for cond MC).\n")

    rows_b = {}
    for g in gammas:
        sub_g = [r for r in rows if r["gamma"] == g]
        groups = []
        for lo, hi, name in BANDS:
            groups.append((f"gamma {g:g}, rho band {name}", [r for r in sub_g if r["rho_band"] == name]))
        groups.append((f"gamma {g:g}, all", sub_g))
        rows_b[g] = groups
    all_groups = []
    for g in gammas:
        all_groups += rows_b[g]

    lines.append("## Accuracy: relative error vs the exact reference (median / max)\n")
    _accuracy_table(lines, all_groups)
    lines.append("")
    lines.append("## Cost: RV per unit / samples for eps 0.1 / 0.01\n")
    _cost_table(lines, all_groups)
    lines.append("")
    lines.append("## Wall time: build + samples x time per unit (s), medians\n")
    _time_table(lines, all_groups)
    lines.append("")

    edges = _tercile_edges(rows) if rows else (0.0, 0.0)
    lines.append(f"## By route length tercile (cut points L = {edges[0]:g}, {edges[1]:g})\n")
    tg = []
    for g in gammas:
        for t_lab in ("short", "medium", "long"):
            sub = [r for r in rows if r["gamma"] == g and r["L_tercile"].startswith(t_lab)]
            tg.append((f"gamma {g:g}, {t_lab}", sub))
    _accuracy_table(lines, tg)
    lines.append("")
    _cost_table(lines, tg)
    lines.append("")

    lines.append("## Reference cost\n")
    lines.append(f"Total reference wall time over all routes: {ref_time_total / 60:.1f} min. "
                 f"Median per route-gamma: {_f(_med([r['ref_wall_s'] for r in rows]), '.3g')} s; "
                 f"median estimated quadrature error relative to p_ref: "
                 f"{_f(_med([_fl(r['ref_quad_err']) / _fl(r['p_ref']) if _fl(r['p_ref']) > 0 else math.nan for r in rows]), '.2e')}; "
                 f"median |p_hi - p_lo| / p_ref: "
                 f"{_f(_med([_fl(r['ref_gap']) / _fl(r['p_ref']) if _fl(r['p_ref']) > 0 else math.nan for r in rows]), '.2e')}.\n")

    lines.append("## Hubs: exact P_hub (reference) vs the hub SMIS estimate\n")
    if hub_rows:
        lines.append("| hub | region | customers | routes (reachable) | gamma | c | p_ref | p_hi - p_lo | "
                     "p_SMIS | rse SMIS | SMIS/ref - 1 | z | ref wall s |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in hub_rows:
            if r["error"]:
                lines.append(f"| {r['name']} | {r['region']} | {r['n_customers']} | | {r['gamma']:g} | | "
                             f"FAILED: {r['error']} | | | | | | |")
                continue
            lines.append(
                f"| {r['name']} | {r['region']} | {r['n_customers']} | {r['n_routes']} ({r['R_reach']}) | "
                f"{r['gamma']:g} | {_f(r['c'], '.6g')} | {_f(r['p_ref'], '.4g')} | "
                f"{_f(_fl(r['p_hi']) - _fl(r['p_lo']), '.3g')} | {_f(r['p_smis'], '.4g')} | "
                f"{_f(r['rse_smis'], '.2%')} | {_f(_fl(r['rel_smis']), '.2%')} | {_f(r['z_smis'], '.2f')} | "
                f"{_f(r['ref_wall_s'], '.3g')} |")
    else:
        lines.append("No hub rows.\n")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================


def _hub_tasks(args, v_pair) -> list[dict]:
    rows = list(csv.DictReader(open(HUBS_IN_CSV, newline="")))
    hub_gammas = [float(x) for x in args.hub_gammas.split(",")]
    names = {}
    for r in rows:
        if r["status"] != "secant" or r["error"]:
            continue
        g = float(r["gamma"])
        if g not in hub_gammas:
            continue
        names.setdefault(r["name"], {"meta": r, "by_gamma": {}})["by_gamma"][g] = r
    ordered = sorted(names.items(), key=lambda kv: int(kv[1]["meta"]["n_customers"]))[: args.hubs]
    tasks = []
    for name, info in ordered:
        path = PLANS_REGION / f"{name}.json"
        doc = json.loads(path.read_text())
        plan = {"dbar": doc["dbar"], "pbar": doc["pbar"], "routes": doc["res"][GATE]["plan"],
                "Q": float(doc["Q"])}
        for g in hub_gammas:
            r = info["by_gamma"].get(g)
            if r is None:
                continue
            tasks.append({"kind": "hub", "name": name, "region": name.split("-")[0],
                          "n_customers": int(r["n_customers"]), "n_routes": int(r["n_routes"]),
                          "gamma": g, "c": float(r["c"]), "plan": plan, "n_grid": args.n_grid,
                          "v_pair": v_pair,
                          "p_smis": float(r["p_hub"]), "rse_smis": float(r["rse"])})
    return tasks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--routes", type=int, default=None, help="first N routes in the QUASAR order")
    ap.add_argument("--gammas", default=",".join(f"{g:g}" for g in GAMMAS_DEFAULT))
    ap.add_argument("--n-grid", type=int, default=N_GRID_DEFAULT)
    ap.add_argument("--n-factor", type=int, default=N_FACTOR_DEFAULT)
    ap.add_argument("--n-gl", type=int, default=N_GL_DEFAULT)
    ap.add_argument("--n-level", type=int, default=N_LEVEL_DEFAULT)
    ap.add_argument("--n-rep", type=int, default=N_REP_DEFAULT)
    ap.add_argument("--hubs", type=int, default=HUBS_DEFAULT, help="smallest hubs to check (0 to skip)")
    ap.add_argument("--hub-gammas", default=",".join(f"{g:g}" for g in HUB_GAMMAS_DEFAULT))
    ap.add_argument("--v-pair", default="256,512", help="two level counts V of the Richardson pair")
    ap.add_argument("--v-cond", type=int, default=256, help="level count of the conditional DP (cond MC, shift pilot)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    v_pair = tuple(int(x) for x in args.v_pair.split(","))
    if len(v_pair) != 2 or v_pair[0] >= v_pair[1]:
        raise SystemExit("--v-pair needs two increasing level counts, e.g. 256,512")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / LOG_FILE, mode="w"), logging.StreamHandler()])
    log = logging.getLogger("baselines")
    t_start = time.time()

    ref = load_routes_ref()
    smis = load_smis_ref()
    means = rs.route_means()
    rids = sorted({k[0] for k in ref}, key=lambda r: means.get(r, {"idx": 1 << 30})["idx"])
    missing = [r for r in rids if r not in means]
    if missing:
        raise SystemExit(f"route ids in the CSV not rebuilt by the sampler: {missing[:5]} ...")
    if args.routes is not None:
        rids = rids[: max(0, args.routes)]
    gammas = [float(x) for x in args.gammas.split(",")]
    tasks = []
    for rid in rids:
        for g in gammas:
            if (rid, g) not in ref or (rid, g) not in smis:
                continue
            tasks.append({"kind": "route", "rid": rid, "gamma": g, "idx": means[rid]["idx"],
                          "ref": ref[(rid, g)], "ce": ref[(rid, g)], "smis": smis[(rid, g)],
                          "dbar": means[rid]["dbar"], "pbar": means[rid]["pbar"],
                          "n_grid": args.n_grid, "n_factor": args.n_factor, "n_gl": args.n_gl,
                          "n_level": args.n_level, "n_rep": args.n_rep, "v_pair": v_pair,
                          "v_cond": args.v_cond})
    hub_tasks = _hub_tasks(args, v_pair) if args.hubs > 0 else []
    log.info("routes %d, route-gamma tasks %d, hub tasks %d, workers %d, grid %d, n_factor %d",
             len(rids), len(tasks), len(hub_tasks), args.workers, args.n_grid, args.n_factor)

    rows, hub_rows, fails = [], [], []
    ctx = mp.get_context("fork")
    all_tasks = tasks + hub_tasks
    with ctx.Pool(processes=max(1, args.workers)) as pool_:
        for k, res in enumerate(pool_.imap_unordered(run_task, all_tasks, chunksize=1), 1):
            if res["kind"] == "route":
                rows.append(res["row"])
                tag = f"{res['key'][0]} gamma={res['key'][1]:g}"
            else:
                hub_rows.append(res["row"])
                tag = f"hub {res['key'][0]} gamma={res['key'][1]:g}"
            if res["error"]:
                fails.append(tag)
                log.error("[%d/%d] %s FAILED: %s", k, len(all_tasks), tag, res["error"])
            else:
                r = res["row"]
                if res["kind"] == "route":
                    log.info("[%d/%d] %s done in %.0fs: ref=%.4g relerr smis=%.2g ce=%.2g gl_s=%.2g cond=%.2g sub=%.2g",
                             k, len(all_tasks), tag, r["_wall"], r["p_ref"], _fl(r["smis0_relerr"]),
                             _fl(r["ce_relerr"]), _fl(r["gl_shift_relerr"]), _fl(r["cond_mc_relerr"]),
                             _fl(r["subset_relerr_med"]))
                else:
                    log.info("[%d/%d] %s done: p_ref=%.4g smis=%.4g z=%.2f", k, len(all_tasks), tag,
                             _fl(r["p_ref"]), _fl(r["p_smis"]), _fl(r["z_smis"]))

    rows.sort(key=lambda r: (r["route_id"], -float(r["gamma"])))
    edges = _tercile_edges(rows) if rows else (0.0, 0.0)
    for r in rows:
        r["L_tercile"] = _tercile_label(int(r["L"]), edges)
    with open(out / ROUTES_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=ROUTE_COLUMNS + ["L_tercile"], extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in ROUTE_COLUMNS + ["L_tercile"]})
    hub_rows.sort(key=lambda r: (r["name"], -float(r["gamma"])))
    with open(out / HUBS_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=HUB_COLUMNS + ["rel_smis"], extrasaction="ignore")
        wr.writeheader()
        for r in hub_rows:
            wr.writerow({k: r.get(k, "") for k in HUB_COLUMNS + ["rel_smis"]})

    elapsed = time.time() - t_start
    ref_total = float(sum(_fl(r["ref_wall_s"]) for r in rows if _fl(r["ref_wall_s"]) == _fl(r["ref_wall_s"])))
    md = summary_markdown(rows, hub_rows, len(rids), elapsed, len(fails), args, ref_total, v_pair)
    (out / SUMMARY_MD).write_text(md)
    log.info("wrote %s, %s and %s in %.1f min; failed: %d %s", ROUTES_CSV, HUBS_CSV, SUMMARY_MD,
             elapsed / 60, len(fails), fails)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

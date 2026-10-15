"""RARECERT plan level: certified overflow probability of every region hub.

Spec: papers/rarecert/THEORY.md section 5 and rarecert/plan.py. Each hub is one region
instance with its SAA plan (results/plans_Region/{name}.json, gate SAA): all routes, one
capacity c common to the routes, the hub overflow event = some route overflows.

For every hub and gamma in {1e-3, 1e-5}:
* c0 from capacity_for: Laplace-type proxy log S(c) = log sum_r exp(-E_r(c)), E_r(c) the
  smallest dominating-point energy of route r (route SMIS build, no draws); c0 solves
  log S(c) = log gamma by a safeguarded secant on [Q, 3Q] in at most 6 evaluations. S is a
  proxy, not a bound, and is used only to start the search. Proxy buffer b0 = c0/Q - 1;
* calibrate_capacity: the exact hub SMIS estimate is matched to gamma. Starting from the
  pair (c0, 0.97 c0), up to 3 secant steps on f(c) = log P_hub(c) - log gamma, each P_hub
  from hub_estimate with N = 2e4 draws (proposal rebuilt at each c), stop when |f| < log 1.25.
  The final c is c_final; buffer b_final = c_final/Q - 1;
* hub_estimate at c_final: P_hub by SMIS with N = 1e5 hub draws (eta_hub = 1e-4), with rse,
  RV, ESS, components and the risk attribution from the same draws. P_hub is the result and
  need not equal gamma (the ratio P_hub/gamma is reported);
* route_baselines at c_final: p_r by smis_estimate (N_r = 2e4), the bounds
  max_r p_r <= P_hub <= sum_r p_r, and the independence approximation 1 - prod_r (1 - p_r).
  A route with no SMIS candidate at c has p_r = 0 (see rarecert/plan.py NoSMISCandidate);
  a hub with no contributing route has P_hub = 0.
For the 20 smallest hubs at gamma = 1e-3 also a crude Monte Carlo check with N = 2e6 at c_final.
Region totals: P_region = 1 - prod_h (1 - P_h) over the hubs of a region (exact in the
model, since hubs have separate factors), and b_final by region (median, q90).

--scope all runs the 148 hubs. --scope sample runs the 26 VNN hubs plus 40 GD hubs chosen
stratified by number of customers: GD hubs sorted by customers, split into 40 equal-size
groups, one hub drawn per group with seed 31.

Usage (from svrpspd_wdro/):
    python3 scripts/run_rarecert_plan.py [--workers 3] [--hubs GLOB] [--max-hubs N]
        [--gammas 1e-3,1e-5] [--n-hub 200000] [--n-check 2000000] [--check-smallest 20]
        [--out-dir DIR]

Outputs (default results/rarecert/): plan_hubs.csv (one row per hub and gamma),
plan_attribution.csv (one row per hub, gamma and route), plan_summary.md, plan.log.
"""

from __future__ import annotations

import argparse
import csv
import glob
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

from rarecert.checkpoint import resumable_map  # noqa: E402
from rarecert.plan import (  # noqa: E402
    CAL_STEPS,
    HubSim,
    calibrate_capacity,
    capacity_for,
    hub_estimate,
    route_baselines,
)

PLANS = ROOT / "results" / "plans_Region"
OUT_DIR = ROOT / "results" / "rarecert"
GATE = "SAA"
CV = 0.30
RHO = 0.60
SEED = 2026
GAMMAS_DEFAULT = (1e-3, 1e-5)
MAX_EVALS = 6         # capacity evaluations (log S secant on [Q, 3Q]), no draws
N_BASE = 20_000       # draws per route for the route baselines, at the final c only
N_HUB = 100_000       # hub SMIS draws at the final c
N_CAL = 20_000        # hub SMIS draws per calibration evaluation (secant on log P_hub)
SAMPLE_VNN_ALL = True
SAMPLE_GD_BINS = 40
SAMPLE_SEED = 31
N_CHECK = 2_000_000
CHECK_SMALLEST = 20
CHECK_GAMMA = 1e-3
REGIONS = ("VNN", "GD")
CKPT = "plan_checkpoint.jsonl"
HUBS_CSV = "plan_hubs.csv"
ATTR_CSV = "plan_attribution.csv"
SUMMARY_MD = "plan_summary.md"
LOG_FILE = "plan.log"

HUB_COLUMNS = [
    "name", "region", "n_customers", "n_routes", "max_L", "Q", "gamma",
    "status", "c0", "b0", "logS_at_c", "cap_evals", "cap_residual",
    "cal_status", "cal_steps", "cal_evals", "cal_residual", "cal_trace",
    "c", "b_final", "p_hub", "ratio_gamma", "RV", "rse", "ESS", "n_components", "n_arc_fallback",
    "nocand_routes", "build_seconds", "sample_seconds",
    "p_upper_sum", "p_lower_max", "p_indep", "bounds_ok", "multi_frac", "top5_routes",
    "crude_N", "crude_p", "crude_rse", "crude_z", "seconds", "error",
]
ATTR_COLUMNS = [
    "name", "region", "gamma", "route", "L", "p_route_hubdraws", "p_route_smis", "rse_route_smis",
    "attr_unique", "modes_route",
]


# ============================================================
# Inputs
# ============================================================


def load_hub(path: Path) -> dict:
    doc = json.loads(path.read_text())
    routes = doc["res"][GATE]["plan"]
    return {
        "name": doc["name"],
        "region": doc["name"].split("-")[0],
        "Q": float(doc["Q"]),
        "plan": {"dbar": doc["dbar"], "pbar": doc["pbar"], "routes": routes, "Q": float(doc["Q"])},
        "n_customers": int(sum(len(r) for r in routes)),
        "n_routes": len(routes),
        "max_L": int(max(len(r) for r in routes)),
    }


def _seed(*parts) -> np.random.Generator:
    return np.random.default_rng([SEED] + [int(p) for p in parts])


def _gamma_key(g: float) -> int:
    return int(round(-math.log10(g) * 1000))


# ============================================================
# Per (hub, gamma) work
# ============================================================


def _nan_row(meta, gamma, err):
    row = {k: "" for k in HUB_COLUMNS}
    row.update({"name": meta["name"], "region": meta["region"], "n_customers": meta["n_customers"],
                "n_routes": meta["n_routes"], "max_L": meta["max_L"], "Q": meta["Q"],
                "gamma": gamma, "error": err})
    return row


def run_task(task: dict) -> dict:
    """One (hub, gamma): capacity, hub SMIS estimate, route baselines, optional crude check."""
    t0 = time.time()
    meta, gamma = task["meta"], task["gamma"]
    attr_rows = []
    try:
        hub = HubSim(meta["plan"], cv=CV, rho=RHO)
        Q = hub.Q
        cap = capacity_for(hub, gamma, max_evals=MAX_EVALS)
        c0 = cap["c"]
        cal = calibrate_capacity(
            hub, gamma, c0, seed_of=lambda k: _seed(task["idx"], _gamma_key(gamma), 10 + k), N=N_CAL)
        c = cal["c"]
        est = hub_estimate(hub, c, N_HUB, _seed(task["idx"], _gamma_key(gamma), 1))
        base = route_baselines(hub, c, _seed(task["idx"], _gamma_key(gamma), 2), N_r=N_BASE)
        p_r = base["route_p_smis"]
        rse_r = base["route_rse"]
        rel_max = float(np.max(rse_r[p_r > 0])) if np.any(p_r > 0) else 0.0
        p = est["p_hub"]
        if p > 0.0:
            slack = 4.0 * (est["rse"] + rel_max) if math.isfinite(est["rse"]) else math.inf
            bounds_ok = bool(p * (1.0 + slack) >= base["lower_max"] and p * (1.0 - slack) <= base["upper_sum"])
        else:   # P_hub = 0 is consistent only if every route probability is 0 too
            bounds_ok = bool(base["lower_max"] <= 0.0)
        top = np.argsort(-est["route_attr"])[:5]
        top5 = "; ".join(f"r{int(r)}(L{len(hub.routes[r])}) attr={est['route_attr'][r]:.3f} p={p_r[r]:.3g}"
                         for r in top)
        trace = " ".join(f"{q['c']:.6g}:{q['p_hub']:.3g}" for q in cal["pts"])
        row = _nan_row(meta, gamma, "")
        row.update({
            "status": cap["status"], "c0": c0, "b0": cap["buffer"], "logS_at_c": cap["logS_at_c"],
            "cap_evals": cap["evals"], "cap_residual": cap["residual"],
            "cal_status": cal["status"], "cal_steps": cal["steps"], "cal_evals": cal["evals"],
            "cal_residual": cal["residual"], "cal_trace": trace,
            "c": c, "b_final": c / Q - 1.0, "ratio_gamma": p / gamma,
            "p_hub": p, "RV": est["RV"], "rse": est["rse"], "ESS": est["ESS"],
            "n_components": est["n_components"], "n_arc_fallback": est["n_arc_fallback"],
            "nocand_routes": " ".join(str(r) for r in est["routes_nocand"]),
            "build_seconds": est["build_seconds"],
            "sample_seconds": est["sample_seconds"], "p_upper_sum": base["upper_sum"],
            "p_lower_max": base["lower_max"], "p_indep": base["indep"], "bounds_ok": bounds_ok,
            "multi_frac": est["multi_frac"], "top5_routes": top5,
        })
        for r in range(hub.R):
            attr_rows.append({
                "name": meta["name"], "region": meta["region"], "gamma": gamma, "route": r,
                "L": len(hub.routes[r]), "p_route_hubdraws": est["route_p"][r],
                "p_route_smis": p_r[r], "rse_route_smis": rse_r[r],
                "attr_unique": est["route_attr"][r], "modes_route": int(base["route_modes"][r]),
            })
        if task["check"] and gamma == CHECK_GAMMA:
            crng = _seed(task["idx"], 99)
            hits = 0
            done = 0
            chunk = max(1000, int(1e8 // (8 * hub.dim)))
            while done < N_CHECK:
                n = min(chunk, N_CHECK - done)
                x = crng.standard_normal((n, hub.dim))
                hits += int(np.count_nonzero(hub.fail(x, c)))
                done += n
            cp = hits / N_CHECK
            crse = math.sqrt((1.0 - cp) / (cp * N_CHECK)) if cp > 0 else math.inf
            sd = math.sqrt((est["rse"] * p) ** 2 + (crse * cp) ** 2) if p > 0 and cp > 0 else math.nan
            row.update({"crude_N": N_CHECK, "crude_p": cp, "crude_rse": crse,
                        "crude_z": (p - cp) / sd if math.isfinite(sd) and sd > 0 else math.nan})
        row["seconds"] = time.time() - t0
        return {"key": (meta["name"], gamma), "row": row, "attr": attr_rows, "error": ""}
    except Exception as exc:  # keep the pool alive; the failure is recorded
        msg = f"{type(exc).__name__}: {exc}"
        return {"key": (meta["name"], gamma), "row": _nan_row(meta, gamma, msg), "attr": [], "error": msg}


# ============================================================
# Summary
# ============================================================


def _f(x, fmt=".3g"):
    try:
        xv = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(xv) else format(xv, fmt)


def _fl(v):
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return math.nan
    return fv if math.isfinite(fv) else math.nan


def _vals(rows, key):
    vals = [_fl(r.get(key, "")) for r in rows]
    return np.asarray([v for v in vals if math.isfinite(v)], dtype=float)


def _tuples(rows, *keys):
    """Array (n, len(keys)) of the rows where every key is finite."""
    out = []
    for r in rows:
        vals = [_fl(r.get(k, "")) for k in keys]
        if all(math.isfinite(v) for v in vals):
            out.append(vals)
    return np.asarray(out, dtype=float).reshape(-1, len(keys))


def region_prob(ps) -> float:
    ps = np.asarray(ps, dtype=float)
    if ps.size == 0:
        return math.nan
    return float(-np.expm1(np.sum(np.log1p(-np.minimum(ps, 1.0 - 1e-16)))))


def summary_markdown(rows, attr_rows, gammas, n_hubs, elapsed, n_fail, n_check) -> str:
    ok = [r for r in rows if not r.get("error")]
    L = []
    L.append("# RARECERT plan level: certified overflow of the region hubs\n")
    L.append(
        f"{n_hubs} hubs (results/plans_Region, gate SAA), gamma in {', '.join(f'{g:g}' for g in gammas)} "
        f"({len(ok)} hub-gamma rows ok, {n_fail} failed). Proxy c0: Laplace-type log S(c) = log sum_r "
        f"exp(-E_r(c)) (a proxy, not a bound), secant on [Q, 3Q] within {MAX_EVALS} evaluations. Calibration: "
        f"exact hub SMIS with N = {N_CAL:,} draws per evaluation, secant on log P_hub - log gamma from "
        f"(c0, 0.97 c0), at most {CAL_STEPS} steps, stop at |f| < log 1.25; final c_final by hub SMIS with "
        f"N = {N_HUB:,} draws, eta_hub = 1e-4. Route baselines N_r = {N_BASE:,} at c_final. "
        f"Wall time {elapsed / 60:.1f} min.\n")
    L.append("P_region = 1 - prod over the hubs of the region of (1 - P_h), with P_h the hub SMIS estimate at "
             "the calibrated c_final (exact in the model: hubs have separate factors). b_final = c_final/Q - 1 "
             "is the calibrated buffer; b0 = c0/Q - 1 is the proxy buffer, kept for comparison.\n")
    L.append("## Region totals and buffers\n")
    L.append("| gamma | region | hubs | P_region (SMIS) | median b_final | q90 b_final | median b0 (proxy) | "
             "median P_h | calibration converged / max_steps / stalled | median steps |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for g in gammas:
        for reg in REGIONS + ("all",):
            sub = [r for r in ok if r["gamma"] == g and (reg == "all" or r["region"] == reg)]
            if not sub:
                continue
            ps = _vals(sub, "p_hub")
            bs = _vals(sub, "b_final")
            b0s = _vals(sub, "b0")
            steps = _vals(sub, "cal_steps")
            st = {}
            for r in sub:
                st[r["cal_status"]] = st.get(r["cal_status"], 0) + 1
            st_s = " / ".join(f"{st.get(k, 0)}" for k in ("converged", "max_steps", "stalled"))
            med_bs = float(np.median(bs)) if bs.size else math.nan
            q90_bs = float(np.quantile(bs, 0.9)) if bs.size else math.nan
            med_b0 = float(np.median(b0s)) if b0s.size else math.nan
            L.append(f"| {g:g} | {reg} | {len(sub)} | {_f(region_prob(ps))} | {_f(med_bs, '.4f')} | "
                     f"{_f(q90_bs, '.4f')} | {_f(med_b0, '.4f')} | "
                     f"{_f(float(np.median(ps)) if ps.size else math.nan)} | {st_s} | "
                     f"{_f(float(np.median(steps)) if steps.size else math.nan, '.1f')} |")
    L.append("")
    L.append("## Baselines and bounds\n")
    L.append("| gamma | bounds hold (hubs) | P_h / max p_r (median) | P_h / sum p_r (median) | "
             "indep / P_h (median) | indep / P_h (q90) | multi-route failure share (median) |")
    L.append("|---|---|---|---|---|---|---|")
    for g in gammas:
        sub = [r for r in ok if r["gamma"] == g]
        if not sub:
            continue
        nb = sum(1 for r in sub if str(r["bounds_ok"]) == "True")
        T = _tuples(sub, "p_hub", "p_lower_max", "p_upper_sum", "p_indep")
        mf = _vals(sub, "multi_frac")
        ratio_max = T[:, 0] / np.maximum(T[:, 1], 1e-300)
        ratio_sum = T[:, 0] / np.maximum(T[:, 2], 1e-300)
        ind = T[:, 3] / np.maximum(T[:, 0], 1e-300)
        L.append(f"| {g:g} | {nb} / {len(sub)} | {_f(np.median(ratio_max))} | {_f(np.median(ratio_sum))} | "
                 f"{_f(np.median(ind))} | {_f(np.quantile(ind, 0.9))} | {_f(np.median(mf) if mf.size else math.nan)} |")
    L.append("")
    L.append("Bounds are checked with slack 4 (rse_hub + max rse_r) on both sides. Calibration stops at "
             f"|log(P_h/gamma)| < log 1.25 (or after {CAL_STEPS} secant steps, N = {N_CAL:,} per evaluation); "
             "the final P_h (N = 1e5 at c_final) is then compared with gamma below.\n")
    L.append("## P_h / gamma distribution (final estimate at c_final)\n")
    L.append("| gamma | region | hubs | median P_h / gamma | q10 | q90 | min | max | outside log 1.25 band |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for g in gammas:
        for reg in REGIONS + ("all",):
            sub = [r for r in ok if r["gamma"] == g and (reg == "all" or r["region"] == reg)]
            rat = _vals(sub, "ratio_gamma")
            if rat.size:
                off = int(np.sum(np.abs(np.log(np.maximum(rat, 1e-300))) > math.log(1.25)))
                L.append(f"| {g:g} | {reg} | {len(sub)} | {_f(np.median(rat))} | {_f(np.quantile(rat, 0.1))} | "
                         f"{_f(np.quantile(rat, 0.9))} | {_f(rat.min())} | {_f(rat.max())} | {off} / {rat.size} |")
    L.append("")
    nocand = [r for r in ok if str(r.get("nocand_routes", "")).strip()]
    L.append("## Routes without an SMIS candidate (P_r = 0 at c_final)\n")
    if nocand:
        L.append(f"{len(nocand)} hub-gamma rows have at least one route with no SMIS candidate at c_final "
                 "(treated as P_r = 0; excluded from the mixture and from S(c)).\n")
        for r in nocand:
            L.append(f"- {r['name']} gamma={float(r['gamma']):g}: routes {r['nocand_routes']} of {r['n_routes']}")
    else:
        L.append("None.")
    L.append("")
    if n_check:
        L.append(f"## Crude Monte Carlo check (gamma = {CHECK_GAMMA:g}, {N_CHECK:,} draws, 20 smallest hubs)\n")
        L.append("| hub | customers | routes | c | P_hub SMIS | P_crude | rse crude | z (combined) |")
        L.append("|---|---|---|---|---|---|---|---|")
        chk = sorted([r for r in ok if r["gamma"] == CHECK_GAMMA and r.get("crude_p") not in ("", None)],
                     key=lambda r: int(r["n_customers"]))
        for r in chk:
            L.append(f"| {r['name']} | {r['n_customers']} | {r['n_routes']} | {_f(r['c'], '.6g')} | "
                     f"{_f(r['p_hub'])} | {_f(r['crude_p'])} | {_f(r['crude_rse'], '.2%')} | {_f(r['crude_z'], '.2f')} |")
        L.append("")
    if n_fail:
        L.append("## Failures\n")
        for r in rows:
            if r.get("error"):
                L.append(f"- {r['name']} gamma={r['gamma']:g}: {r['error']}")
        L.append("")
    return "\n".join(L)


# ============================================================
# Main
# ============================================================


def select_files(files: list, scope: str) -> list:
    """scope 'all': every hub. scope 'sample': all VNN hubs plus 40 GD hubs, stratified by
    number of customers (GD sorted by customers, 40 equal groups, one per group, seed 31)."""
    if scope == "all":
        return files
    vnn = [f for f in files if Path(f).name.startswith("VNN-")]
    gd = [f for f in files if Path(f).name.startswith("GD-")]
    if scope != "sample":
        raise ValueError(f"unknown scope {scope!r}")
    n_of = {f: load_hub(Path(f))["n_customers"] for f in gd}
    order = sorted(gd, key=lambda f: (n_of[f], f))
    rng = np.random.default_rng(SAMPLE_SEED)
    chosen = [grp[int(rng.integers(len(grp)))] for grp in np.array_split(np.arange(len(order)), SAMPLE_GD_BINS)
              if len(grp)]
    picked = [order[i] for i in chosen]
    return sorted(vnn + picked)


def _retype(rows):
    """CSV rows back to numbers where possible (merge path)."""
    out = []
    for r in rows:
        q = {}
        for k, v in r.items():
            if v in ("", None):
                q[k] = ""
                continue
            try:
                q[k] = int(v) if v.lstrip("-").isdigit() else float(v)
            except ValueError:
                q[k] = v
        out.append(q)
    return out


def merge_shards(dirs, out: Path, gammas, log) -> int:
    rows, attrs = [], []
    for d in dirs:
        with open(d / HUBS_CSV, newline="") as fh:
            rows.extend(csv.DictReader(fh))
        with open(d / ATTR_CSV, newline="") as fh:
            attrs.extend(csv.DictReader(fh))
    rows, attrs = _retype(rows), _retype(attrs)
    for r in rows:
        r["name"], r["region"] = str(r["name"]), str(r["region"])
    for r in attrs:
        r["name"] = str(r["name"])
    rows.sort(key=lambda r: (r["name"], -float(r["gamma"])))
    attrs.sort(key=lambda r: (r["name"], -float(r["gamma"]), int(r["route"])))
    for fname, cols, data in ((HUBS_CSV, HUB_COLUMNS, rows), (ATTR_CSV, ATTR_COLUMNS, attrs)):
        with open(out / fname, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            for r in data:
                wr.writerow({k: r.get(k, "") for k in cols})
    n_fail = sum(1 for r in rows if r.get("error"))
    n_check = sum(1 for r in rows if r.get("crude_p") not in ("", None))
    n_hubs = len({r["name"] for r in rows})
    md = summary_markdown(rows, attrs, gammas, n_hubs, 0.0, n_fail, n_check)
    (out / SUMMARY_MD).write_text(md)
    log.info("merged %d hub rows and %d attribution rows from %d shards", len(rows), len(attrs), len(dirs))
    return 0


def task_key(t: dict):
    return (t["meta"]["name"], t["gamma"])


def main(argv=None) -> int:
    global N_HUB, N_BASE, MAX_EVALS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--hubs", default="*.json", help="glob of plan file names in results/plans_Region")
    ap.add_argument("--max-hubs", type=int, default=None, help="quick run: first N hubs by name")
    ap.add_argument("--gammas", default=",".join(f"{g:g}" for g in GAMMAS_DEFAULT))
    ap.add_argument("--n-hub", type=int, default=N_HUB)
    ap.add_argument("--n-base", type=int, default=N_BASE)
    ap.add_argument("--max-evals", type=int, default=MAX_EVALS)
    ap.add_argument("--scope", default="all", choices=("all", "sample"))
    ap.add_argument("--n-check", type=int, default=N_CHECK)
    ap.add_argument("--check-smallest", type=int, default=CHECK_SMALLEST)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--shard", default="0/1", help="i/n: run only the tasks with index %% n == i")
    ap.add_argument("--merge-from", default=None,
                    help="comma-separated shard output dirs: merge their CSVs into --out-dir and rebuild the summary")
    args = ap.parse_args(argv)
    N_HUB, N_BASE, MAX_EVALS = args.n_hub, args.n_base, args.max_evals
    shard_i, shard_n = (int(v) for v in args.shard.split("/"))
    if not 0 <= shard_i < shard_n:
        raise SystemExit(f"bad --shard {args.shard!r}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / LOG_FILE, mode="w"), logging.StreamHandler()])
    log = logging.getLogger("plan")
    t_start = time.time()
    gammas = tuple(float(g) for g in args.gammas.split(",") if g.strip())

    if args.merge_from:
        return merge_shards([Path(d) for d in args.merge_from.split(",") if d.strip()], out, gammas, log)

    files = sorted(glob.glob(str(PLANS / args.hubs)))
    if not files:
        raise SystemExit(f"no hub plans match {PLANS / args.hubs}")
    files = select_files(files, args.scope)
    if args.max_hubs is not None:
        files = files[: max(0, args.max_hubs)]
    hubs = [load_hub(Path(f)) for f in files]
    smallest = {h["name"] for h in sorted(hubs, key=lambda h: h["n_customers"])[: args.check_smallest]}
    tasks = []
    for idx, h in enumerate(hubs):
        for g in gammas:
            tasks.append({"meta": h, "gamma": g, "idx": idx,
                          "check": h["name"] in smallest and args.n_check > 0})
    # largest hubs first, so the long tasks do not all end at the end of the run
    tasks.sort(key=lambda t: -t["meta"]["n_customers"])
    tasks = [t for k, t in enumerate(tasks) if k % shard_n == shard_i]
    log.info("shard %d/%d", shard_i, shard_n)
    log.info("scope %s, hubs %d, gammas %s, tasks %d, workers %d, N_hub %d, N_base %d, max_evals %d, "
             "crude check hubs %d (N %d)", args.scope, len(hubs), gammas, len(tasks), args.workers, N_HUB,
             N_BASE, MAX_EVALS, len(smallest), args.n_check)

    rows, attrs, fails = [], [], []
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=max(1, args.workers)) as pool_:
        for k, res in enumerate(resumable_map(run_task, tasks, task_key, out / CKPT, pool_), 1):
            rows.append(res["row"])
            attrs.extend(res["attr"])
            name, g = res["key"]
            if res["error"]:
                fails.append((name, g))
                log.error("[%d/%d] %s gamma=%g FAILED: %s", k, len(tasks), name, g, res["error"])
            else:
                r = res["row"]
                log.info("[%d/%d] %s n=%s gamma=%g done in %.1fs: c0=%.1f c=%.1f b0=%.4f b_final=%.4f "
                         "P_hub=%.3g ratio=%.3g rse=%.3g comps=%s cal=%s steps=%s nocand=%s",
                         k, len(tasks), name, r["n_customers"], g, r["seconds"], r["c0"], r["c"], r["b0"],
                         r["b_final"], r["p_hub"], r["ratio_gamma"], r["rse"], r["n_components"],
                         r["cal_status"], r["cal_steps"], r["nocand_routes"] or "-")

    rows.sort(key=lambda r: (r["name"], -float(r["gamma"])))
    attrs.sort(key=lambda r: (r["name"], -float(r["gamma"]), int(r["route"])))
    with open(out / HUBS_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=HUB_COLUMNS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in HUB_COLUMNS})
    with open(out / ATTR_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=ATTR_COLUMNS)
        wr.writeheader()
        for r in attrs:
            wr.writerow({k: r.get(k, "") for k in ATTR_COLUMNS})
    elapsed = time.time() - t_start
    n_check = sum(1 for t in tasks if t["check"])
    md = summary_markdown(rows, attrs, gammas, len(hubs), elapsed, len(fails), n_check)
    (out / SUMMARY_MD).write_text(md)
    log.info("wrote %s, %s, %s in %.1f min; failed rows: %d %s", HUBS_CSV, ATTR_CSV, SUMMARY_MD,
             elapsed / 60, len(fails), fails)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

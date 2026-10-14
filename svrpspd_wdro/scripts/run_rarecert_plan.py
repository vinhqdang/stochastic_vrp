"""RARECERT plan level: certified overflow probability of every region hub.

Spec: papers/rarecert/THEORY.md section 5 and rarecert/plan.py. Each hub is one region
instance with its SAA plan (results/plans_Region/{name}.json, gate SAA): all routes, one
capacity c common to the routes, the hub overflow event = some route overflows.

For every hub and gamma in {1e-3, 1e-5}:
* c from capacity_for: Laplace-type proxy log S(c) = log sum_r exp(-E_r(c)), E_r(c) the
  smallest dominating-point energy of route r (route SMIS build, no draws); c solves
  log S(c) = log gamma by a safeguarded secant on [Q, 3Q] in at most 6 evaluations. S is a
  proxy, not a bound, and is used only to locate c. Buffer b = c/Q - 1, Q the instance
  capacity;
* hub_estimate: P_hub by SMIS with N = 1e5 hub draws at that c (eta_hub = 1e-4), with rse,
  RV, ESS, components and the risk attribution from the same draws. P_hub is the result and
  need not equal gamma;
* route_baselines at the same c: p_r by smis_estimate (N_r = 2e4), the bounds
  max_r p_r <= P_hub <= sum_r p_r, and the independence approximation 1 - prod_r (1 - p_r).
For the 20 smallest hubs at gamma = 1e-3 also a crude Monte Carlo check with N = 2e6.
Region totals: P_region = 1 - prod_h (1 - P_h) over the hubs of a region (exact in the
model, since hubs have separate factors), and buffers by region (median, q90).

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

from rarecert.plan import (  # noqa: E402
    HubSim,
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
N_HUB = 100_000       # hub SMIS draws
SAMPLE_VNN_ALL = True
SAMPLE_GD_BINS = 40
SAMPLE_SEED = 31
N_CHECK = 2_000_000
CHECK_SMALLEST = 20
CHECK_GAMMA = 1e-3
REGIONS = ("VNN", "GD")
HUBS_CSV = "plan_hubs.csv"
ATTR_CSV = "plan_attribution.csv"
SUMMARY_MD = "plan_summary.md"
LOG_FILE = "plan.log"

HUB_COLUMNS = [
    "name", "region", "n_customers", "n_routes", "max_L", "Q", "gamma", "status", "c", "buffer",
    "logS_at_c", "cap_evals", "cap_residual", "p_hub", "RV", "rse", "ESS", "n_components", "n_arc_fallback", "build_seconds", "sample_seconds",
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
        cap = capacity_for(hub, gamma, max_evals=MAX_EVALS)
        c = cap["c"]
        est = hub_estimate(hub, c, N_HUB, _seed(task["idx"], _gamma_key(gamma), 1))
        base = route_baselines(hub, c, _seed(task["idx"], _gamma_key(gamma), 2), N_r=N_BASE)
        p_r = base["route_p_smis"]
        rse_r = base["route_rse"]
        rel_max = float(np.max(rse_r[p_r > 0])) if np.any(p_r > 0) else 0.0
        p = est["p_hub"]
        slack = 4.0 * (est["rse"] + rel_max) if math.isfinite(est["rse"]) else math.inf
        bounds_ok = bool(p * (1.0 + slack) >= base["lower_max"] and p * (1.0 - slack) <= base["upper_sum"])
        top = np.argsort(-est["route_attr"])[:5]
        top5 = "; ".join(f"r{int(r)}(L{len(hub.routes[r])}) attr={est['route_attr'][r]:.3f} p={p_r[r]:.3g}"
                         for r in top)
        row = _nan_row(meta, gamma, "")
        row.update({
            "status": cap["status"], "c": c, "buffer": cap["buffer"], "logS_at_c": cap["logS_at_c"],
            "cap_evals": cap["evals"], "cap_residual": cap["residual"],
            "p_hub": p, "RV": est["RV"], "rse": est["rse"], "ESS": est["ESS"],
            "n_components": est["n_components"], "n_arc_fallback": est["n_arc_fallback"],
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
        f"({len(ok)} hub-gamma rows ok, {n_fail} failed). Capacity: Laplace-type proxy log S(c) = log sum_r "
        f"exp(-E_r(c)) (a proxy, not a bound), secant on [Q, 3Q] within {MAX_EVALS} evaluations. Hub SMIS: "
        f"N = {N_HUB:,} draws at that c, eta_hub = 1e-4. Route baselines "
        f"N_r = {N_BASE:,}. Wall time {elapsed / 60:.1f} min.\n")
    L.append("P_region = 1 - prod over the hubs of the region of (1 - P_h), with P_h the hub SMIS estimate at "
             "its own c (exact in the model: hubs have separate factors). Buffer b = c/Q - 1.\n")
    L.append("## Region totals and buffers\n")
    L.append("| gamma | region | hubs | P_region (SMIS) | median buffer | q90 buffer | median P_h | "
             "capacity status (bisect / capped / below Q) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for g in gammas:
        for reg in REGIONS + ("all",):
            sub = [r for r in ok if r["gamma"] == g and (reg == "all" or r["region"] == reg)]
            if not sub:
                continue
            ps = _vals(sub, "p_hub")
            bs = _vals(sub, "buffer")
            st = {}
            for r in sub:
                st[r["status"]] = st.get(r["status"], 0) + 1
            st_s = " / ".join(f"{st.get(k, 0)}" for k in ("bisect", "capped_3Q", "at_or_below_Q"))
            med_bs = float(np.median(bs)) if bs.size else math.nan
            q90_bs = float(np.quantile(bs, 0.9)) if bs.size else math.nan
            L.append(f"| {g:g} | {reg} | {len(sub)} | {_f(region_prob(ps))} | {_f(med_bs, '.4f')} | "
                     f"{_f(q90_bs, '.4f')} | {_f(float(np.median(ps)) if ps.size else math.nan)} | {st_s} |")
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
    L.append("Bounds are checked with slack 4 (rse_hub + max rse_r) on both sides. The capacity proxy "
             "is not a bound, so P_h at the chosen c can sit above or below gamma; the ratio is reported below.\n")
    L.append("| gamma | median P_h / gamma | q90 P_h / gamma | min P_h / gamma |")
    L.append("|---|---|---|---|")
    for g in gammas:
        sub = [r for r in ok if r["gamma"] == g]
        rat = _vals(sub, "p_hub") / g
        if rat.size:
            L.append(f"| {g:g} | {_f(np.median(rat))} | {_f(np.quantile(rat, 0.9))} | {_f(rat.min())} |")
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
    args = ap.parse_args(argv)
    N_HUB, N_BASE, MAX_EVALS = args.n_hub, args.n_base, args.max_evals

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / LOG_FILE, mode="w"), logging.StreamHandler()])
    log = logging.getLogger("plan")
    t_start = time.time()
    gammas = tuple(float(g) for g in args.gammas.split(",") if g.strip())

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
    log.info("scope %s, hubs %d, gammas %s, tasks %d, workers %d, N_hub %d, N_base %d, max_evals %d, "
             "crude check hubs %d (N %d)", args.scope, len(hubs), gammas, len(tasks), args.workers, N_HUB,
             N_BASE, MAX_EVALS, len(smallest), args.n_check)

    rows, attrs, fails = [], [], []
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=max(1, args.workers)) as pool_:
        for k, res in enumerate(pool_.imap_unordered(run_task, tasks, chunksize=1), 1):
            rows.append(res["row"])
            attrs.extend(res["attr"])
            name, g = res["key"]
            if res["error"]:
                fails.append((name, g))
                log.error("[%d/%d] %s gamma=%g FAILED: %s", k, len(tasks), name, g, res["error"])
            else:
                r = res["row"]
                log.info("[%d/%d] %s n=%s gamma=%g done in %.1fs: c=%.1f b=%.4f P_hub=%.3g rse=%.3g "
                         "comps=%s evals=%s status=%s", k, len(tasks), name, r["n_customers"], g, r["seconds"],
                         r["c"], r["buffer"], r["p_hub"], r["rse"], r["n_components"], r["cap_evals"], r["status"])

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

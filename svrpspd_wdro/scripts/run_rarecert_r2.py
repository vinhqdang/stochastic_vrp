"""RARECERT R2: mode-decomposed IS on a route-day simulator (travel and service times)
with no exact DP.

Spec: rarecert/sim2.py (RouteDay), rarecert/modes.py (MDIS, CE baseline, thresholds).
Routes are the 120 QUASAR direction-A routes of results/quasar/rare_routes.csv (rebuilt
with the same sampler, seed 21, and checked against the CSV ids). Legs come from the
instance matrix in data/Region (parse_dethloff, km = D / 1e4).

For each (route, gamma) in (1e-3, 1e-4, 1e-5):
* thresholds: Qc at gamma/2 (demand-only) and slack by bisection so that the time
  modes also fail with probability gamma/2 (rarecert.modes.choose_thresholds);
* MDIS (rarecert.modes.mdis_mixture + is_generic) with N = 1e6 draws;
* generic CE mixture (one CE shift per mode, mode-weighted) with N = 1e6 draws, plus
  its pilot draws;
* single-shift CE (the shift of the most probable mode) with N = 1e6 draws;
* at gamma = 1e-3 also crude Monte Carlo with N = 1e7 as the reference.

Usage (from svrpspd_wdro/):
    python3 scripts/run_rarecert_r2.py [--workers 3] [--routes N] [--out-dir DIR]

Outputs (default results/rarecert/): r2_routes.csv (one row per route and gamma),
r2_summary.md, r2.log.
"""

from __future__ import annotations

import argparse
import csv
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

import run_quasar_rare as rq  # noqa: E402  (route sampler, CV, RHO, SEED)
from dethloff_runner import parse_dethloff  # noqa: E402
from rarecert import modes as MD  # noqa: E402
from rarecert.sim2 import RouteDay, legs_from_matrix  # noqa: E402

RARE_CSV = ROOT / "results" / "quasar" / "rare_routes.csv"
DATA_DIR = ROOT / "data" / "Region"
OUT_DIR = ROOT / "results" / "rarecert"
ROUTES_CSV = "r2_routes.csv"
SUMMARY_MD = "r2_summary.md"
LOG_FILE = "r2.log"
GAMMAS = (1e-3, 1e-4, 1e-5)
N_DRAWS = 1_000_000
N_CRUDE = 10_000_000
SEED_R2 = 53
EPS = (0.1, 0.01)
Z2 = 1.96 ** 2
CLASS_HI = 0.9
CLASS_LO = 0.1

COLUMNS = [
    "route_id", "instance", "route_index", "L", "gamma", "cls", "Qc", "slack", "p_time_thr",
    "thr_seconds", "thr_converged", "thr_hi_expanded",
    "p_mdis", "rse_mdis", "RV_mdis", "ESS_mdis", "p_over_gamma", "load_mass_share",
    "share_fail_load", "n_modes_mdis", "n_cand_mdis", "n_unreach_mdis", "build_s_mdis",
    "p_ce", "rse_ce", "RV_ce", "ESS_ce", "pilot_ce", "n_modes_ce", "skipped_ce", "build_s_ce",
    "p_single", "rse_single", "RV_single", "ESS_single", "single_mode",
    "p_crude", "rse_crude",
    "err_mdis", "err_ce", "err_single", "dis_mdis_ce", "dis_mdis_single",
    "cost_mdis_0.1", "cost_mdis_0.01", "cost_ce_0.1", "cost_ce_0.01",
    "seconds", "error",
]


# ============================================================
# Inputs
# ============================================================


def load_csv_route_ids(path: Path) -> dict:
    """route_id -> the QUASAR CSV's instance and route_index (first row per route)."""
    out: dict = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["route_id"], {"instance": r["instance"], "route_index": int(r["route_index"])})
    return out


def route_records() -> list[dict]:
    """The QUASAR stratified route sample in its round-robin order, with nodes and means."""
    pool = rq.collect_routes(rq.PLANS)
    routes = rq.order_round_robin(rq.sample_routes(pool, rq.SEED))
    out = []
    for idx, r in enumerate(routes):
        out.append({"idx": idx, "route_id": f"{r['instance']}#{r['route_index']}",
                    "instance": r["instance"], "route_index": r["route_index"],
                    "nodes": r["nodes"], "dbar": r["dbar"], "pbar": r["pbar"]})
    return out


def legs_of(instance: str, nodes) -> np.ndarray:
    D, _dem, _Q, _n, scale = parse_dethloff(str(DATA_DIR / f"{instance}.vrpspd"))
    return legs_from_matrix(D, nodes, units_per_km=float(scale))


# ============================================================
# Per-(route, gamma) work
# ============================================================


def _rng(idx: int, gamma: float, k: int) -> np.random.Generator:
    return np.random.default_rng([SEED_R2, idx, int(round(gamma * 1e9)), k])


def _class(load_share: float) -> str:
    if not math.isfinite(load_share):
        return "n/a"
    if load_share >= CLASS_HI:
        return "load only"
    if load_share <= CLASS_LO:
        return "time only"
    return "mixed"


def _rel(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or b <= 0 or a <= 0:
        return math.nan
    return abs(a / b - 1.0)


def run_task(task: dict) -> dict:
    t0 = time.time()
    rid, gamma, idx = task["rid"], task["gamma"], task["idx"]
    dbar, pbar, legs = task["dbar"], task["pbar"], task["legs"]
    row = {k: math.nan for k in COLUMNS}
    row.update({"route_id": rid, "instance": task["instance"], "route_index": task["route_index"],
                "L": len(dbar), "gamma": gamma, "cls": "", "single_mode": "", "error": ""})
    try:
        thr = MD.choose_thresholds(dbar, pbar, legs, gamma, _rng(idx, gamma, 0))
        Qc, slack = thr["Qc"], thr["slack"]
        sim = RouteDay(dbar, pbar, legs, cv=rq.CV, rho=rq.RHO, Qc=Qc, slack=slack)
        row.update({"Qc": Qc, "slack": slack, "p_time_thr": thr["p_time"],
                    "thr_seconds": thr["seconds"], "thr_converged": int(thr["converged"]),
                    "thr_hi_expanded": int(thr["hi_expanded"])})

        mix, diag = MD.mdis_mixture(sim)
        est = MD.is_generic(sim, mix.thetas, mix.log_a, N_DRAWS, _rng(idx, gamma, 1))
        row.update({"p_mdis": est["p"], "rse_mdis": est["rse"], "RV_mdis": est["RV"],
                    "ESS_mdis": est["ESS"], "p_over_gamma": est["p"] / gamma,
                    "load_mass_share": diag["load_mass_share"], "share_fail_load": est["share_load"],
                    "n_modes_mdis": diag["n_modes"], "n_cand_mdis": diag["n_candidates"],
                    "n_unreach_mdis": diag["n_unreachable"], "build_s_mdis": diag["build_seconds"]})
        row["cls"] = _class(diag["load_mass_share"])

        tc = time.perf_counter()
        ce_mix, cinfo = MD.ce_generic_mixture(sim, _rng(idx, gamma, 2))
        ce_est = MD.is_generic(sim, ce_mix.thetas, ce_mix.log_a, N_DRAWS, _rng(idx, gamma, 3))
        row.update({"p_ce": ce_est["p"], "rse_ce": ce_est["rse"], "RV_ce": ce_est["RV"],
                    "ESS_ce": ce_est["ESS"], "pilot_ce": cinfo["draws"], "n_modes_ce": cinfo["n_modes"],
                    "skipped_ce": len(cinfo["skipped"]), "build_s_ce": time.perf_counter() - tc})

        theta1 = ce_mix.thetas[1]                       # most probable mode (kept first)
        single_mode = ce_mix.stages[1]
        s_est = MD.is_generic(sim, theta1[None, :], np.zeros(1), N_DRAWS, _rng(idx, gamma, 4))
        row.update({"p_single": s_est["p"], "rse_single": s_est["rse"], "RV_single": s_est["RV"],
                    "ESS_single": s_est["ESS"], "single_mode": sim.mode_names[single_mode]})

        p_ref = math.nan
        if abs(gamma - 1e-3) < 1e-15:
            cr = MD.crude_generic(sim, N_CRUDE, _rng(idx, gamma, 5))
            p_ref = cr["p"]
            row.update({"p_crude": cr["p"], "rse_crude": cr["rse"]})
            row["err_mdis"] = _rel(est["p"], p_ref)
            row["err_ce"] = _rel(ce_est["p"], p_ref)
            row["err_single"] = _rel(s_est["p"], p_ref)
        row["dis_mdis_ce"] = _rel(est["p"], ce_est["p"])
        row["dis_mdis_single"] = _rel(est["p"], s_est["p"])
        for eps in EPS:
            key = f"{eps:g}"
            row[f"cost_mdis_{key}"] = Z2 * est["RV"] / eps ** 2
            row[f"cost_ce_{key}"] = cinfo["draws"] + Z2 * ce_est["RV"] / eps ** 2
        row["seconds"] = time.time() - t0
        return {"rid": rid, "gamma": gamma, "row": row, "error": ""}
    except Exception as exc:  # keep the pool alive; the failure is recorded in the row
        msg = f"{type(exc).__name__}: {exc}"
        row["error"] = msg
        row["seconds"] = time.time() - t0
        return {"rid": rid, "gamma": gamma, "row": row, "error": msg}


# ============================================================
# Summary
# ============================================================


def _finite(vals):
    return np.asarray([v for v in vals if v is not None and math.isfinite(v)], dtype=float)


def _med(vals):
    a = _finite(vals)
    return float(np.median(a)) if a.size else math.nan


def _q90(vals):
    a = _finite(vals)
    return float(np.quantile(a, 0.9)) if a.size else math.nan


def _f(x, fmt=".3g"):
    return "n/a" if x is None or not math.isfinite(x) else format(x, fmt)


def _pair(vals, fmt=".2%"):
    return f"{_f(_med(vals), fmt)} / {_f(_q90(vals), fmt)}"


def summary_markdown(rows: list[dict], n_routes: int, elapsed: float, n_fail: int) -> str:
    ok = [r for r in rows if not r["error"]]
    lines = ["# RARECERT R2: mode-decomposed IS on the route-day simulator\n"]
    lines.append(
        f"{n_routes} routes (QUASAR sample, results/quasar/rare_routes.csv), gamma in "
        f"{', '.join(f'{g:g}' for g in GAMMAS)}: {len(rows)} route-gamma rows, {n_fail} failed. "
        f"MDIS, generic CE mixture and single-shift CE with N = {N_DRAWS:,} draws; crude MC with "
        f"N = {N_CRUDE:,} at gamma = 1e-3 (reference). Wall time {elapsed / 60:.1f} min.\n")
    lines.append(
        "Class = share of the MDIS mixture mass on load modes: load only (>= 0.9), time only "
        "(<= 0.1), mixed. Errors are |estimate / reference - 1|; 'vs crude' uses crude MC "
        "(gamma = 1e-3 only); 'MDIS vs CE' and 'MDIS vs single' are disagreements (all gammas). "
        "Costs are sample counts at 95 per cent: MDIS = 1.96^2 RV / eps^2; CE = pilot draws + "
        "1.96^2 RV / eps^2. Threshold bisection cost is common to all methods and not counted.\n")

    lines.append("## Per gamma and class (median / q90)\n")
    lines.append("| gamma | class | n | err MDIS vs crude | err CE vs crude | err single vs crude | "
                 "MDIS vs CE | MDIS vs single | RV MDIS med | RV CE med | RV single med | "
                 "modes MDIS med | modes CE med | build s MDIS med | build s CE med | "
                 "share fail load med |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in GAMMAS:
        for cls in ("all", "load only", "time only", "mixed"):
            sub = [r for r in ok if r["gamma"] == g and (cls == "all" or r["cls"] == cls)]
            if not sub:
                lines.append(f"| {g:g} | {cls} | 0 |" + " |" * 14)
                continue
            lines.append(
                f"| {g:g} | {cls} | {len(sub)} | {_pair([r['err_mdis'] for r in sub])} | "
                f"{_pair([r['err_ce'] for r in sub])} | {_pair([r['err_single'] for r in sub])} | "
                f"{_pair([r['dis_mdis_ce'] for r in sub])} | {_pair([r['dis_mdis_single'] for r in sub])} | "
                f"{_f(_med([r['RV_mdis'] for r in sub]))} | {_f(_med([r['RV_ce'] for r in sub]))} | "
                f"{_f(_med([r['RV_single'] for r in sub]))} | "
                f"{_f(_med([r['n_modes_mdis'] for r in sub]), '.3g')} | "
                f"{_f(_med([r['n_modes_ce'] for r in sub]), '.3g')} | "
                f"{_f(_med([r['build_s_mdis'] for r in sub]), '.2f')} | "
                f"{_f(_med([r['build_s_ce'] for r in sub]), '.2f')} | "
                f"{_f(_med([r['share_fail_load'] for r in sub]), '.3f')} |")
    lines.append("")

    lines.append("## Costs and thresholds per gamma\n")
    lines.append("Totals are sums over the routes of the gamma (sample counts). CE / MDIS is the "
                 "ratio of the totals.\n")
    lines.append("| gamma | eps_rel | rows | MDIS total | CE total | CE / MDIS | "
                 "share of failure on load modes (med) | p_MDIS / gamma med | threshold s med | "
                 "unreachable modes med | not converged | slack expanded |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in GAMMAS:
        sub = [r for r in ok if r["gamma"] == g]
        for eps in EPS:
            key = f"{eps:g}"
            tm = sum(r[f"cost_mdis_{key}"] for r in sub if math.isfinite(r[f"cost_mdis_{key}"]))
            tc = sum(r[f"cost_ce_{key}"] for r in sub if math.isfinite(r[f"cost_ce_{key}"]))
            lines.append(
                f"| {g:g} | {eps:g} | {len(sub)} | {tm:.4g} | {tc:.4g} | "
                f"{_f(tc / tm if tm > 0 else math.nan, '.3g')} | "
                f"{_f(_med([r['share_fail_load'] for r in sub]), '.3f')} | "
                f"{_f(_med([r['p_over_gamma'] for r in sub]), '.3f')} | "
                f"{_f(_med([r['thr_seconds'] for r in sub]), '.3g')} | "
                f"{_f(_med([r['n_unreach_mdis'] for r in sub]), '.3g')} | "
                f"{sum(1 for r in sub if not r['thr_converged'])} | "
                f"{sum(1 for r in sub if r['thr_hi_expanded'])} |")
    lines.append("")
    lines.append("Single-shift mode chosen (count over rows): " + ", ".join(
        f"{g:g}: {_count_modes([r for r in ok if r['gamma'] == g])}" for g in GAMMAS) + ".\n")
    return "\n".join(lines)


def _count_modes(sub) -> str:
    tally: dict = {}
    for r in sub:
        tally[r["single_mode"]] = tally.get(r["single_mode"], 0) + 1
    return "; ".join(f"{k} {v}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1]))


# ============================================================
# Main
# ============================================================


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--routes", type=int, default=None,
                    help="quick run: first N routes in the QUASAR stratum order")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / LOG_FILE, mode="w"), logging.StreamHandler()])
    log = logging.getLogger("r2")
    t_start = time.time()

    csv_ids = load_csv_route_ids(RARE_CSV)
    recs = route_records()
    by_id = {r["route_id"]: r for r in recs}
    missing = [rid for rid in csv_ids if rid not in by_id]
    if missing:
        raise SystemExit(f"route ids in the CSV not rebuilt by the sampler: {missing[:5]} ...")
    order = sorted(csv_ids, key=lambda rid: by_id[rid]["idx"])
    if args.routes is not None:
        order = order[: max(0, args.routes)]

    tasks = []
    for rid in order:
        r = by_id[rid]
        legs = legs_of(r["instance"], r["nodes"])
        for g in GAMMAS:
            tasks.append({"rid": rid, "gamma": g, "idx": r["idx"], "instance": r["instance"],
                          "route_index": r["route_index"], "dbar": r["dbar"], "pbar": r["pbar"],
                          "legs": legs})
    log.info("routes %d, route-gamma tasks %d, workers %d", len(order), len(tasks), args.workers)

    rows, fails = [], []
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=max(1, args.workers)) as pool_:
        for k, res in enumerate(pool_.imap_unordered(run_task, tasks, chunksize=1), 1):
            rows.append(res["row"])
            if res["error"]:
                fails.append((res["rid"], res["gamma"]))
                log.error("[%d/%d] %s gamma=%g FAILED: %s", k, len(tasks), res["rid"], res["gamma"], res["error"])
            else:
                r = res["row"]
                log.info("[%d/%d] %s gamma=%g %s done in %.1fs: p_mdis=%.3g rse=%.3g RV=%.3g "
                         "p_ce=%.3g dis=%.3g err_mdis=%s modes=%d/%d",
                         k, len(tasks), res["rid"], res["gamma"], r["cls"], r["seconds"], r["p_mdis"],
                         r["rse_mdis"], r["RV_mdis"], r["p_ce"], r["dis_mdis_ce"],
                         _f(r["err_mdis"], ".3g"), r["n_modes_mdis"], r["n_modes_ce"])

    rows.sort(key=lambda r: (r["route_id"], -r["gamma"]))
    with open(out / ROUTES_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in COLUMNS})
    elapsed = time.time() - t_start
    md = summary_markdown(rows, len(order), elapsed, len(fails))
    (out / SUMMARY_MD).write_text(md)
    log.info("wrote %s and %s in %.1f min; failed rows: %d %s", ROUTES_CSV, SUMMARY_MD, elapsed / 60,
             len(fails), fails)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""QUASAR direction A: rare-event route overflow and quantum-versus-classical cost.

Design: papers/quasar/DESIGN_RARE.md. For routes of the SAA region plans (routes
with L >= 4, stratified by region and by L tercile, seed 21), and for each target
overflow level gamma in (1e-3, 1e-4, 1e-5, 1e-6):

* c is the capacity with P(F > c) = gamma (quantum.rare.threshold_for);
* theta is the cross-entropy mean shift at c (quantum.rare.ce_shift);
* p and RV are the importance sampling ground truth with N = 1e6 draws;
* for L <= 8 and gamma >= 1e-4 the exact discretised probability p_dp is computed
  (quantum.rare.dp_check) as a cross-check of the IS value;
* per relative error eps in (0.1, 0.03, 0.01) the classical sample counts C_crude
  and C_IS and the quantum amplitude estimation costs Q_crude and Q_IS are
  computed (ideal M known, and with the doubling search), with the best fixed
  k in {1, 3, ..., 20} per variant.

Usage (from svrpspd_wdro/):
    python3 scripts/run_quasar_rare.py [--workers 3] [--routes N] [--out-dir DIR]

Outputs (default results/quasar/): rare_routes.csv (one row per route, gamma and
eps), rare_summary.md (per gamma and eps summary and the DP cross-check table),
rare.log (progress).
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

from quantum.rare import (  # noqa: E402
    RouteSim,
    aqe_cost,
    aqe_is_cost,
    best_aqe,
    c_crude,
    c_is,
    ce_shift,
    dp_check,
    is_shift,
    threshold_for,
)

PLANS = ROOT / "results" / "plans_Region"
OUT_DIR = ROOT / "results" / "quasar"
GATE = "SAA"
MIN_LEN = 4
SEED = 21
DELTA = 0.05
CV = 0.30
RHO = 0.6
GAMMAS = (1e-3, 1e-4, 1e-5, 1e-6)
EPS = (0.1, 0.03, 0.01)
N_TRUTH = 1_000_000
DP_MAX_L = 8
DP_MIN_GAMMA = 1e-4
DP_V = 256
DP_NODES = 40
SAMPLE_PER_REGION = {"VNN": 40, "GD": 80}
REGIONS = ("VNN", "GD")
ROUTES_CSV = "rare_routes.csv"
SUMMARY_MD = "rare_summary.md"
LOG_FILE = "rare.log"

COLUMNS = [
    "route_id", "instance", "region", "route_index", "L", "gamma", "c", "theta_norm",
    "p_is", "rse_is", "RV_is", "ess_is", "p_dp", "dp_rel_diff", "eps_rel",
    "C_crude", "C_is", "Q_crude_ideal", "Q_crude_doubling", "Q_is_ideal", "Q_is_doubling",
    "Qc_over_Cc_ideal", "Qc_over_Cc_doubling", "QIS_over_CIS_ideal", "QIS_over_CIS_doubling",
    "QIS_over_minC_ideal", "QIS_over_minC_doubling", "k_crude_ideal", "k_crude_doubling",
    "k_is_ideal", "k_is_doubling", "seconds",
]


# ============================================================
# Route sample
# ============================================================


def collect_routes(plans: Path, gate: str = GATE, min_len: int = MIN_LEN) -> dict:
    """All routes of the gate with at least min_len stops, grouped by region."""
    pool = {reg: [] for reg in REGIONS}
    for f in sorted(glob.glob(str(Path(plans) / "*.json"))):
        doc = json.loads(Path(f).read_text())
        inst = doc["name"]
        reg = inst.split("-")[0]
        if reg not in pool:
            continue
        for j, route in enumerate(doc["res"][gate]["plan"]):
            if len(route) >= min_len:
                pool[reg].append({
                    "instance": inst, "region": reg, "route_index": j,
                    "nodes": [int(i) for i in route],
                    "dbar": [float(doc["dbar"][i]) for i in route],
                    "pbar": [float(doc["pbar"][i]) for i in route],
                })
    return pool


def sample_routes(pool: dict, seed: int = SEED) -> list[dict]:
    """Stratified sample: per region, equal shares of the L terciles (remainder to
    the lowest terciles). Returns routes in a fixed order, with a stratum tag."""
    rng = np.random.default_rng(seed)
    out = []
    for reg in REGIONS:
        recs = pool[reg]
        n_reg = SAMPLE_PER_REGION[reg]
        if not recs:
            continue
        L = np.array([len(r["nodes"]) for r in recs], dtype=float)
        cuts = np.quantile(L, [1.0 / 3.0, 2.0 / 3.0])
        tert = np.digitize(L, cuts, right=True)
        for t in range(3):
            members = [i for i in range(len(recs)) if tert[i] == t]
            want = n_reg // 3 + (1 if t < n_reg % 3 else 0)
            if len(members) < want:
                logging.warning("stratum %s tercile %d has %d < %d routes", reg, t, len(members), want)
                want = len(members)
            chosen = rng.choice(members, size=want, replace=False) if want else []
            for i in chosen:
                rec = dict(recs[int(i)])
                rec["stratum"] = (reg, t)
                out.append(rec)
    return out


def order_round_robin(routes: list[dict]) -> list[dict]:
    """Interleave the strata so that any prefix covers all strata evenly."""
    strata: dict = {}
    for r in routes:
        strata.setdefault(r["stratum"], []).append(r)
    keys = sorted(strata)
    out = []
    i = 0
    while True:
        added = False
        for key in keys:
            if i < len(strata[key]):
                out.append(strata[key][i])
                added = True
        if not added:
            break
        i += 1
    return out


# ============================================================
# Per-route work
# ============================================================


def _ratio(a: float, b: float) -> float:
    if not (a > 0 and math.isfinite(a) and b > 0 and math.isfinite(b)):
        return float("nan")
    return a / b


def run_route(task: dict) -> dict:
    """Ground truth and costs for one route; a failure is recorded, not raised."""
    t0 = time.time()
    try:
        return _run_route(task, t0)
    except Exception as exc:  # keep the pool alive; the failure is reported in the log
        return {"rid": task["rid"], "rows": [], "dp": [], "seconds": time.time() - t0,
                "error": f"{type(exc).__name__}: {exc}"}


def _run_route(task: dict, t0: float) -> dict:
    sim = RouteSim(task["dbar"], task["pbar"], cv=CV, rho=RHO)
    rows = []
    dp_rows = []
    for gi, gamma in enumerate(GAMMAS):
        rng = np.random.default_rng([SEED, task["idx"], gi])
        c = threshold_for(sim, gamma, rng)
        theta = ce_shift(sim, c, rng)
        p, RV, rse, ess = is_shift(sim, c, theta, N_TRUTH, rng)
        p_dp = float("nan")
        rel = float("nan")
        if sim.L <= DP_MAX_L and gamma >= DP_MIN_GAMMA and p > 0:
            p_dp = dp_check(sim, c, V=DP_V, n_gh=DP_NODES)
            rel = p_dp / p - 1.0
            dp_rows.append({"route_id": task["rid"], "L": sim.L, "gamma": gamma, "c": c,
                            "p_is": p, "rse_is": rse, "p_dp": p_dp, "rel_diff": rel})
        for eps in EPS:
            if p <= 0 or not math.isfinite(RV):
                rows.append(_nan_row(task, gamma, c, theta, p, RV, rse, ess, p_dp, rel, eps))
                continue
            Cc = c_crude(p, eps)
            Ci = c_is(RV, eps)
            qc = best_aqe(lambda k, d: aqe_cost(p, eps, d, k), delta=DELTA)
            qi = best_aqe(lambda k, d: aqe_is_cost(RV, p, eps, d, k), delta=DELTA)
            minC = min(Cc, Ci)                 # classical side has no doubling variant
            rows.append({
                "route_id": task["rid"], "instance": task["instance"], "region": task["region"],
                "route_index": task["route_index"], "L": sim.L, "gamma": gamma, "c": c,
                "theta_norm": float(np.linalg.norm(theta)),
                "p_is": p, "rse_is": rse, "RV_is": RV, "ess_is": ess,
                "p_dp": p_dp, "dp_rel_diff": rel, "eps_rel": eps,
                "C_crude": Cc, "C_is": Ci,
                "Q_crude_ideal": qc["ideal"], "Q_crude_doubling": qc["doubling"],
                "Q_is_ideal": qi["ideal"], "Q_is_doubling": qi["doubling"],
                "Qc_over_Cc_ideal": _ratio(qc["ideal"], Cc),
                "Qc_over_Cc_doubling": _ratio(qc["doubling"], Cc),
                "QIS_over_CIS_ideal": _ratio(qi["ideal"], Ci),
                "QIS_over_CIS_doubling": _ratio(qi["doubling"], Ci),
                "QIS_over_minC_ideal": _ratio(qi["ideal"], minC),
                "QIS_over_minC_doubling": _ratio(qi["doubling"], minC),
                "k_crude_ideal": qc["k_ideal"], "k_crude_doubling": qc["k_doubling"],
                "k_is_ideal": qi["k_ideal"], "k_is_doubling": qi["k_doubling"],
                "seconds": float("nan"),
            })
    elapsed = time.time() - t0
    for r in rows:
        r["seconds"] = elapsed / max(1, len(rows))
    return {"rid": task["rid"], "rows": rows, "dp": dp_rows, "seconds": elapsed}


def _nan_row(task, gamma, c, theta, p, RV, rse, ess, p_dp, rel, eps):
    row = {k: float("nan") for k in COLUMNS}
    row.update({"route_id": task["rid"], "instance": task["instance"], "region": task["region"],
                "route_index": task["route_index"], "L": len(task["nodes"]), "gamma": gamma,
                "c": c, "theta_norm": float(np.linalg.norm(theta)), "p_is": p, "rse_is": rse,
                "RV_is": RV, "ess_is": ess, "p_dp": p_dp, "dp_rel_diff": rel, "eps_rel": eps})
    return row


# ============================================================
# Summary
# ============================================================


def _q(vals, qs=(0.1, 0.5, 0.9)):
    arr = np.asarray([v for v in vals if v is not None and math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return [float("nan")] * len(qs)
    return [float(np.quantile(arr, q)) for q in qs]


def _fmt(x: float) -> str:
    if not math.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-2:
        return f"{x:.3g}"
    return f"{x:.3f}"


def summary_markdown(rows: list[dict], dp_rows: list[dict], n_routes: int, elapsed: float) -> str:
    lines = []
    lines.append("# QUASAR rare-event route overflow (direction A)\n")
    lines.append(
        f"{n_routes} routes (SAA gate, L >= {MIN_LEN}), gamma in {GAMMAS}, eps_rel in {EPS}. "
        f"Ground truth p and RV from importance sampling with {N_TRUTH:,} draws per (route, gamma). "
        f"Quantiles are 10%, 50% and 90% over routes. Ratios below 1 mean the quantum "
        f"schedule costs less. 'ideal' assumes M known; 'doubling' includes the search for M.\n"
    )
    lines.append(f"Wall time {elapsed / 60:.1f} min.\n")
    lines.append("## Per gamma and eps\n")
    lines.append("| gamma | eps | n | RV q10 | RV q50 | RV q90 | Qc/Cc ideal q10/50/90 | "
                 "QIS/CIS ideal q10/50/90 | QIS/minC ideal q10/50/90 | QIS/minC doubling q10/50/90 | "
                 "share QIS<minC ideal | share QIS<minC doubling |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for gamma in GAMMAS:
        for eps in EPS:
            sub = [r for r in rows if r["gamma"] == gamma and r["eps_rel"] == eps]
            if not sub:
                continue
            rv = _q([r["RV_is"] for r in sub])
            qcc = _q([r["Qc_over_Cc_ideal"] for r in sub])
            qis = _q([r["QIS_over_CIS_ideal"] for r in sub])
            qmin_i = _q([r["QIS_over_minC_ideal"] for r in sub])
            qmin_d = _q([r["QIS_over_minC_doubling"] for r in sub])
            sh_i = _share(sub, "QIS_over_minC_ideal")
            sh_d = _share(sub, "QIS_over_minC_doubling")
            lines.append(
                f"| {gamma:g} | {eps:g} | {len(sub)} | {_fmt(rv[0])} | {_fmt(rv[1])} | {_fmt(rv[2])} | "
                f"{_triple(qcc)} | {_triple(qis)} | {_triple(qmin_i)} | {_triple(qmin_d)} | "
                f"{sh_i:.2f} | {sh_d:.2f} |"
            )
    lines.append("")
    lines.append("## Exact DP cross-check (L <= 8, gamma >= 1e-4)\n")
    if dp_rows:
        lines.append("| route | L | gamma | c | p_IS | rse_IS | p_dp | p_dp/p_IS - 1 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for d in dp_rows:
            lines.append(
                f"| {d['route_id']} | {d['L']} | {d['gamma']:g} | {d['c']:.1f} | {d['p_is']:.4e} | "
                f"{d['rse_is']:.4f} | {d['p_dp']:.4e} | {d['rel_diff']:+.4f} |"
            )
        rels = np.array([d["rel_diff"] for d in dp_rows])
        lines.append("")
        lines.append(f"DP versus IS: median relative difference {np.median(rels):+.4f}, "
                     f"max |difference| {np.max(np.abs(rels)):.4f} over {len(rels)} routes.")
    else:
        lines.append("No routes in the sample qualified.")
    lines.append("")
    return "\n".join(lines)


def _triple(vals):
    return "/".join(_fmt(v) for v in vals)


def _share(sub, key):
    vals = [r[key] for r in sub if r[key] is not None and math.isfinite(r[key])]
    if not vals:
        return float("nan")
    return float(np.mean([v < 1.0 for v in vals]))


# ============================================================
# Main
# ============================================================


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", default=str(PLANS))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--routes", type=int, default=None,
                    help="subsample: first N routes in stratum round-robin order (quick run)")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / LOG_FILE, mode="w"), logging.StreamHandler()])
    log = logging.getLogger("rare")

    t_start = time.time()
    pool = collect_routes(Path(args.plans))
    sampled = sample_routes(pool, SEED)
    routes = order_round_robin(sampled)
    if args.routes is not None:
        routes = routes[: max(0, args.routes)]
    log.info("eligible routes: %s; sampled %d; running %d",
             {k: len(v) for k, v in pool.items()}, len(sampled), len(routes))
    tasks = []
    for idx, r in enumerate(routes):
        tasks.append({"rid": f"{r['instance']}#{r['route_index']}", "idx": idx,
                      "instance": r["instance"], "region": r["region"],
                      "route_index": r["route_index"], "nodes": r["nodes"],
                      "dbar": r["dbar"], "pbar": r["pbar"]})

    rows, dp_rows, failures = [], [], []
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=max(1, args.workers)) as pool_:
        for k, res in enumerate(pool_.imap_unordered(run_route, tasks, chunksize=1), 1):
            rows.extend(res["rows"])
            dp_rows.extend(res["dp"])
            if res.get("error"):
                failures.append(res["rid"])
                log.error("[%d/%d] %s FAILED: %s", k, len(tasks), res["rid"], res["error"])
            else:
                log.info("[%d/%d] %s done in %.1fs", k, len(tasks), res["rid"], res["seconds"])

    dp_rows.sort(key=lambda d: (d["route_id"], d["gamma"]))
    rows.sort(key=lambda r: (r["route_id"], r["gamma"], r["eps_rel"]))
    with open(out / ROUTES_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in COLUMNS})
    elapsed = time.time() - t_start
    md = summary_markdown(rows, dp_rows, len(tasks), elapsed)
    (out / SUMMARY_MD).write_text(md)
    log.info("wrote %s and %s in %.1f min; failed routes: %d %s", ROUTES_CSV, SUMMARY_MD,
             elapsed / 60, len(failures), failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

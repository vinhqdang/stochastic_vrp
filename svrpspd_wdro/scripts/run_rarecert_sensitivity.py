"""RARECERT R1 task B: sensitivity of the rare route estimators, their coverage, and the
single-shift diagnostic (referee items 3.5, 3.6, 3.10).

Part 1 (sensitivity). The first N routes of results/quasar/rare_routes.csv (QUASAR
round-robin order) are re-estimated at gamma = 1e-4 under one-factor-at-a-time
changes of the base model (rho 0.6, cv 0.3, pickup means as in the plan, gamma
marginals):

    rho in {0.3, 0.6, 0.8}; cv in {0.2, 0.3, 0.5}; pickup scale s in {0.5, 1, 2}
    (all pickup means of the route multiplied by s, which moves the end-mode ratio
    varrho independently of the region); marginal family in {gamma, lognormal}.

For every (route, setting) the threshold c is found with quantum.rare.threshold_for
under that setting. Estimators at c: SMIS (rarecert.smis, eta = 1e-3, n_arc = 0,
N draws), a single cross-entropy mean shift (N draws, pilot counted), and a
cross-entropy mixture (N draws, pilot counted). The reference is the exact
conditional DP (baselines.exact_reference, V = 256 and 512 Richardson) for gamma
marginals with L <= 8; otherwise the CE mixture re-estimated with --ref-draws fresh
draws, flagged ref_kind = "ce". The DP is not defined for the lognormal family, so
those rows are flagged "ce" as well.

Also recorded: the closed-form varrho (rarecert.smis.stage_scales ratio) and the
exact-energy mode mass m_hat = exp(-(E_other - E_min)) of the weaker end mode, where
the end modes are the dominating points of the delivery block of stage 0 and of the
pickup block of stage L, E_other is the larger of the two energies and E_min the
smallest candidate energy of the SMIS build.

Part 2 (coverage). The first --cov-routes routes at the base setting: the SMIS
mixture and the single shift are built once (both deterministic given the seed), then
--reps independent importance-sampling replications with --cov-draws draws each. A
replication covers the reference when |p - ref| <= 1.96 * p * rse (nominal 95 per
cent CLT interval p (1 +- 1.96 rse)).

Usage (from svrpspd_wdro/):
    python3 scripts/run_rarecert_sensitivity.py [--part all|sensitivity|coverage]
        [--routes 30] [--cov-routes 10] [--draws 200000] [--reps 300] [--workers 3]

Outputs (default results/rarecert/): sensitivity_routes.csv (one row per route and
setting), coverage.csv, sensitivity_summary.md (sensitivity and coverage sections of
this run), sensitivity.log. A "--part coverage" run rewrites sensitivity_summary.md
with the coverage section only.
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

import run_rarecert_smis as rs  # noqa: E402  (QUASAR route sampler, CV, RHO, route means)
from quantum.rare import ce_mixture, ce_shift, is_mixture, is_shift, threshold_for  # noqa: E402
from rarecert.baselines import exact_reference  # noqa: E402
from rarecert.checkpoint import resumable_map  # noqa: E402
from rarecert.family import FamilyRouteSim  # noqa: E402
from rarecert.smis import (  # noqa: E402
    Unreachable,
    block_direction,
    dominating_point,
    smis_mixture,
    stage_scales,
)

RARE_CSV = rs.RARE_CSV
OUT_DIR = ROOT / "results" / "rarecert"
SENS_CSV = "sensitivity_routes.csv"
COV_CSV = "coverage.csv"
CKPT = "sensitivity_checkpoint.jsonl"
SUMMARY_MD = "sensitivity_summary.md"
LOG_FILE = "sensitivity.log"

GAMMA = 1e-4
ETA = 1e-3
N_MAIN = 200_000
N_REF_CE = 1_000_000
V_PAIR = (256, 512)
DP_MAX_L = 8
N_ROUTES = 30
N_COV_ROUTES = 10
N_COV = 10_000
N_REP = 300
SEED = 23
Z95 = 1.96

# (factor, level label, rho, cv, pickup scale s, marginal family); the base model first.
SETTINGS = (
    ("base", "base", 0.6, 0.3, 1.0, "gamma"),
    ("rho", "rho=0.3", 0.3, 0.3, 1.0, "gamma"),
    ("rho", "rho=0.8", 0.8, 0.3, 1.0, "gamma"),
    ("cv", "cv=0.2", 0.6, 0.2, 1.0, "gamma"),
    ("cv", "cv=0.5", 0.6, 0.5, 1.0, "gamma"),
    ("pickup", "s=0.5", 0.6, 0.3, 0.5, "gamma"),
    ("pickup", "s=2", 0.6, 0.3, 2.0, "gamma"),
    ("family", "lognormal", 0.6, 0.3, 1.0, "lognormal"),
)

SENS_COLUMNS = [
    "route_id", "instance", "route_index", "L", "setting", "factor", "level", "rho", "cv", "s", "family",
    "gamma", "c", "c_csv", "varrho", "lam_star", "ref_kind", "p_ref", "rse_ref", "dp_gap", "N_main", "N_ref",
    "p_smis", "rse_smis", "RV_smis", "ESS_smis", "n_modes", "build_s",
    "p_single", "rse_single", "RV_single", "pilot_single",
    "p_ce", "rse_ce", "RV_ce", "pilot_ce",
    "e_smis", "e_single", "e_ce", "m_hat", "E_min", "E_other", "seconds", "error",
]
COV_COLUMNS = [
    "route_id", "instance", "L", "c", "ref_kind", "p_ref", "method", "N", "reps",
    "coverage", "median_rse", "n_zero", "seconds", "error",
]


# ============================================================
# Inputs
# ============================================================


def quasar_order(path: Path) -> list[dict]:
    """Distinct routes at gamma = 1e-4 in the row order of rare_routes.csv."""
    out, seen = [], set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if not math.isclose(float(r["gamma"]), GAMMA, rel_tol=1e-9):
                continue
            rid = r["route_id"]
            if rid in seen:
                continue
            seen.add(rid)
            out.append({"route_id": rid, "instance": r["instance"], "route_index": int(r["route_index"]),
                        "L": int(r["L"]), "c_csv": float(r["c"])})
    return out


def build_route_list(n_routes: int) -> list[dict]:
    means = rs.route_means()
    order = quasar_order(RARE_CSV)[:n_routes]
    out = []
    for pos, r in enumerate(order):
        if r["route_id"] not in means:
            raise SystemExit(f"route {r['route_id']} not rebuilt by the QUASAR sampler")
        m = means[r["route_id"]]
        if m["L"] != r["L"]:
            raise SystemExit(f"route {r['route_id']}: L {m['L']} != CSV L {r['L']}")
        out.append({**r, "pos": pos, "dbar": np.asarray(m["dbar"], dtype=np.float64),
                    "pbar": np.asarray(m["pbar"], dtype=np.float64)})
    return out


# ============================================================
# Estimators shared by the parts
# ============================================================


def end_energies(sim: FamilyRouteSim, c: float) -> tuple[float, float]:
    """Energies E = |x|^2 / 2 of the dominating points of the two end modes: the
    delivery block of stage 0 and the pickup block of stage L. An unreachable or empty
    block gives inf (its mode is absent, so its mass is zero)."""
    out = []
    for k, block in ((0, "d"), (sim.L, "p")):
        try:
            _, E = dominating_point(sim, c, k, block_direction(sim, k, block))
        except (Unreachable, ValueError):
            E = math.inf
        out.append(float(E))
    return out[0], out[1]


def coverage_rate(estimate, truth: float, n_rep: int, rng: np.random.Generator, z: float = Z95):
    """Empirical coverage of the nominal (1 - alpha) CLT interval p (1 +- z rse).

    estimate(rng) returns (p, rse) with rse the relative standard error of p. A
    replication covers the truth when |p - truth| <= z p rse; p = 0 or a non-finite rse
    never covers. Returns (coverage, median rse over replications with finite rse,
    number of replications with p = 0)."""
    hits, rses, zeros = 0, [], 0
    for _ in range(int(n_rep)):
        p, rse = estimate(rng)
        if p <= 0.0:
            zeros += 1
            continue
        if math.isfinite(rse):
            rses.append(rse)
        if math.isfinite(rse) and abs(p - truth) <= z * p * rse:
            hits += 1
    cov = hits / float(n_rep)
    med = float(np.median(rses)) if rses else math.nan
    return cov, med, zeros


# ============================================================
# Part 1: sensitivity
# ============================================================


def run_setting(task: dict) -> dict:
    t0 = time.time()
    route, (factor, level, rho, cv, s, family) = task["route"], task["setting"]
    sidx = task["sidx"]
    row = {k: math.nan for k in SENS_COLUMNS}
    row.update({"route_id": route["route_id"], "instance": route["instance"], "route_index": route["route_index"],
                "L": route["L"], "setting": level, "factor": factor, "level": level, "rho": rho, "cv": cv,
                "s": s, "family": family, "gamma": GAMMA, "c_csv": route["c_csv"] if level == "base" else math.nan,
                "ref_kind": "", "N_main": task["n_main"], "N_ref": math.nan, "error": ""})
    try:
        rng = np.random.default_rng([SEED, route["pos"], sidx])
        sim = FamilyRouteSim(route["dbar"], route["pbar"] * s, cv=cv, rho=rho, family=family)
        st = stage_scales(sim)
        row["varrho"], row["lam_star"] = st["ratio"], st["lam_star"]
        c = threshold_for(sim, GAMMA, rng)
        row["c"] = c

        mix, diag = smis_mixture(sim, c, eta=ETA, n_arc=0)
        p_s, RV_s, rse_s, ess_s = is_mixture(sim, c, mix, task["n_main"], rng)
        row.update({"p_smis": p_s, "rse_smis": rse_s, "RV_smis": RV_s, "ESS_smis": ess_s,
                    "n_modes": diag["n_modes"], "build_s": diag["build_seconds"]})
        E_d0, E_pL = end_energies(sim, c)
        E_min = diag["E_min"]
        E_other = max(E_d0, E_pL)
        row.update({"E_min": E_min, "E_other": E_other,
                    "m_hat": math.exp(-(E_other - E_min)) if math.isfinite(E_other) else 0.0})

        info: dict = {}
        theta = ce_shift(sim, c, rng, info=info)
        p_1, RV_1, rse_1, _ = is_shift(sim, c, theta, task["n_main"], rng)
        row.update({"p_single": p_1, "rse_single": rse_1, "RV_single": RV_1, "pilot_single": info["draws"]})

        cem = ce_mixture(sim, c, rng)
        p_c, RV_c, rse_c, _ = is_mixture(sim, c, cem, task["n_main"], rng)
        row.update({"p_ce": p_c, "rse_ce": rse_c, "RV_ce": RV_c, "pilot_ce": cem.draws})

        if family == "gamma" and sim.L <= DP_MAX_L:
            ref = exact_reference(sim, c, V_pair=V_PAIR)
            p_ref = ref["p_ext"]
            row.update({"ref_kind": "dp", "p_ref": p_ref, "dp_gap": ref["gap"]})
        else:
            p_ref, _, rse_ref, _ = is_mixture(sim, c, cem, task["n_ref"], rng)
            row.update({"ref_kind": "ce", "p_ref": p_ref, "rse_ref": rse_ref, "N_ref": task["n_ref"]})

        if p_ref > 0:
            row["e_smis"] = p_s / p_ref - 1.0
            row["e_single"] = p_1 / p_ref - 1.0
            row["e_ce"] = p_c / p_ref - 1.0
    except Exception as exc:  # keep the pool alive; the failure is recorded in the row
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.time() - t0
    return row


# ============================================================
# Part 2: coverage
# ============================================================


def run_coverage_route(task: dict) -> list[dict]:
    t0 = time.time()
    route, n_cov, reps, n_ref = task["route"], task["n_cov"], task["reps"], task["n_ref"]
    base = SETTINGS[0]
    _, _, rho, cv, s, family = base
    rows = []

    def base_row(method):
        return {"route_id": route["route_id"], "instance": route["instance"], "L": route["L"], "method": method,
                "N": n_cov, "reps": reps, "c": math.nan, "ref_kind": "", "p_ref": math.nan,
                "coverage": math.nan, "median_rse": math.nan, "n_zero": 0, "seconds": 0.0, "error": ""}

    try:
        rng = np.random.default_rng([SEED + 1, route["pos"]])
        sim = FamilyRouteSim(route["dbar"], route["pbar"] * s, cv=cv, rho=rho, family=family)
        c = threshold_for(sim, GAMMA, rng)
        mix, _ = smis_mixture(sim, c, eta=ETA, n_arc=0)
        info: dict = {}
        theta = ce_shift(sim, c, rng, info=info)
        cem = ce_mixture(sim, c, rng)
        if sim.L <= DP_MAX_L:
            p_ref = exact_reference(sim, c, V_pair=V_PAIR)["p_ext"]
            kind = "dp"
        else:
            p_ref, _, _, _ = is_mixture(sim, c, cem, n_ref, rng)
            kind = "ce"

        def smis_est(r):
            p, _, rse, _ = is_mixture(sim, c, mix, n_cov, r)
            return p, rse

        def single_est(r):
            p, _, rse, _ = is_shift(sim, c, theta, n_cov, r)
            return p, rse

        for method, est in (("smis", smis_est), ("single", single_est)):
            row = base_row(method)
            cov, med, zeros = coverage_rate(est, p_ref, reps, rng)
            row.update({"c": c, "ref_kind": kind, "p_ref": p_ref, "coverage": cov,
                        "median_rse": med, "n_zero": zeros})
            rows.append(row)
    except Exception as exc:
        for method in ("smis", "single"):
            row = base_row(method)
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    for row in rows:
        row["seconds"] = time.time() - t0
    return rows


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


def _abs(rows, key):
    return [abs(r[key]) if math.isfinite(r[key]) else math.nan for r in rows]


def summary_markdown(sens: list[dict], cov: list[dict], cfg: dict, elapsed: float) -> str:
    lines = ["# RARECERT R1 task B: sensitivity, coverage and the single-shift diagnostic\n"]
    lines.append(
        f"Gamma = {GAMMA:g}. Sensitivity: {cfg['routes']} routes (QUASAR order), {cfg['draws']:,} draws per "
        f"SMIS, single shift and CE mixture; reference = exact DP (V = 256/512 Richardson) for gamma marginals "
        f"with L <= {DP_MAX_L}, else CE mixture with {cfg['ref_draws']:,} fresh draws (flagged). "
        f"Errors are signed relative errors p/p_ref - 1 (medians of absolute values, q90 of absolute values). "
        f"Wall time {elapsed / 60:.1f} min.\n")
    if sens:
        ok = [r for r in sens if not r["error"]]
        lines.append(f"Rows: {len(sens)} route-setting pairs, {len(sens) - len(ok)} failed.\n")
        lines.append("## Sensitivity by factor level\n")
        lines.append("| factor | level | rows | ref dp / ce | varrho med | n_modes med | m_hat med | "
                     "err SMIS med / q90 | err single med / q90 | err CE med / q90 | "
                     "single share err<0 | RV SMIS med | RV single med | RV CE med |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, level, *_ in SETTINGS:
            sub = [r for r in ok if r["level"] == level]
            if not sub:
                continue
            ndp = sum(1 for r in sub if r["ref_kind"] == "dp")
            neg = [r["e_single"] for r in sub if math.isfinite(r["e_single"])]
            share = sum(1 for v in neg if v < 0) / len(neg) if neg else math.nan
            lines.append(
                f"| {sub[0]['factor']} | {level} | {len(sub)} | {ndp} / {len(sub) - ndp} | "
                f"{_f(_med([r['varrho'] for r in sub]), '.3f')} | {_f(_med([r['n_modes'] for r in sub]))} | "
                f"{_f(_med([r['m_hat'] for r in sub]), '.3g')} | "
                f"{_f(_med(_abs(sub, 'e_smis')), '.2%')} / {_f(_q90(_abs(sub, 'e_smis')), '.2%')} | "
                f"{_f(_med(_abs(sub, 'e_single')), '.2%')} / {_f(_q90(_abs(sub, 'e_single')), '.2%')} | "
                f"{_f(_med(_abs(sub, 'e_ce')), '.2%')} / {_f(_q90(_abs(sub, 'e_ce')), '.2%')} | "
                f"{_f(share, '.2f')} | {_f(_med([r['RV_smis'] for r in sub]))} | "
                f"{_f(_med([r['RV_single'] for r in sub]))} | {_f(_med([r['RV_ce'] for r in sub]))} |")
        lines.append("")
        lines.append("Per route-setting detail (signed errors) is in sensitivity_routes.csv.\n")
    if cov:
        lines.append("## Coverage of the nominal 95% CLT interval (base setting)\n")
        lines.append(f"{cfg['cov_routes']} routes, gamma = {GAMMA:g}, {cfg['reps']} replications with "
                     f"N = {cfg['cov_draws']:,} draws each; the proposal is built once per route. "
                     "Coverage: share of replications with |p - ref| <= 1.96 p rse.\n")
        lines.append("| route | L | ref | method | coverage | median rse | zero estimates |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in cov:
            if r["error"]:
                lines.append(f"| {r['route_id']} | {r['L']} | | {r['method']} | ERROR {r['error']} | | |")
                continue
            lines.append(f"| {r['route_id']} | {r['L']} | {r['ref_kind']} | {r['method']} | "
                         f"{r['coverage']:.3f} | {_f(r['median_rse'], '.3f')} | {r['n_zero']} |")
        for method in ("smis", "single"):
            sub = [r for r in cov if r["method"] == method and not r["error"]]
            if sub:
                pooled = float(np.mean([r["coverage"] for r in sub]))
                lines.append(f"\nPooled mean coverage, {method}: {pooled:.3f} over {len(sub)} routes.")
        lines.append("")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================


def sens_key(t: dict):
    return ("sens", t["route"]["route_id"], t["sidx"])


def cov_key(t: dict):
    return ("cov", t["route"]["route_id"])


def _pool_map(fn, tasks, workers, log, label, keyfn, ckpt):
    out = []
    if workers <= 1:
        for k, res in enumerate(resumable_map(fn, tasks, keyfn, ckpt), 1):
            out.append(res)
            log.info("%s [%d/%d] done", label, k, len(tasks))
        return out
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        for k, res in enumerate(resumable_map(fn, tasks, keyfn, ckpt, pool), 1):
            out.append(res)
            log.info("%s [%d/%d] done", label, k, len(tasks))
    return out


def _retype(rows):
    """CSV rows back to numbers where possible (merge path)."""
    out = []
    for r in rows:
        q = {}
        for k, v in r.items():
            if v in ("", None):
                q[k] = None if k != "error" else ""
                continue
            try:
                q[k] = int(v) if v.lstrip("-").isdigit() else float(v)
            except ValueError:
                q[k] = v
        out.append(q)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("all", "sensitivity", "coverage"), default="all")
    ap.add_argument("--routes", type=int, default=N_ROUTES)
    ap.add_argument("--cov-routes", type=int, default=N_COV_ROUTES)
    ap.add_argument("--draws", type=int, default=N_MAIN)
    ap.add_argument("--ref-draws", type=int, default=N_REF_CE)
    ap.add_argument("--cov-draws", type=int, default=N_COV)
    ap.add_argument("--reps", type=int, default=N_REP)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--reverse", action="store_true",
                    help="run the (shard's) tasks last first, so a second machine can work a shard from the other end; concatenate the two checkpoints into one out-dir and rerun to collect")
    ap.add_argument("--shard", default="0/1", help="i/n: run only the tasks with index %% n == i")
    ap.add_argument("--merge-from", default=None,
                    help="comma-separated shard output dirs: merge their CSVs into --out-dir and rebuild the summary")
    args = ap.parse_args(argv)
    shard_i, shard_n = (int(v) for v in args.shard.split("/"))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / LOG_FILE, mode="w"), logging.StreamHandler()])
    log = logging.getLogger("sensitivity")
    t_start = time.time()
    cfg = {"routes": args.routes, "cov_routes": args.cov_routes, "draws": args.draws,
           "ref_draws": args.ref_draws, "cov_draws": args.cov_draws, "reps": args.reps}

    if args.merge_from:
        sens_rows, cov_rows = [], []
        for d in args.merge_from.split(","):
            d = Path(d)
            if (d / SENS_CSV).exists():
                with open(d / SENS_CSV) as fh:
                    sens_rows += list(csv.DictReader(fh))
            if (d / COV_CSV).exists():
                with open(d / COV_CSV) as fh:
                    cov_rows += list(csv.DictReader(fh))
        for rows, cols, name in ((sens_rows, SENS_COLUMNS, SENS_CSV), (cov_rows, COV_COLUMNS, COV_CSV)):
            with open(out / name, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=cols)
                wr.writeheader()
                for r in rows:
                    wr.writerow({k: r.get(k, "") for k in cols})
        (out / SUMMARY_MD).write_text(summary_markdown(_retype(sens_rows), _retype(cov_rows), cfg, 0.0))
        log.info("merged %d sensitivity rows and %d coverage rows", len(sens_rows), len(cov_rows))
        return

    routes = build_route_list(max(args.routes, args.cov_routes))
    log.info("routes loaded: %d (sensitivity %d, coverage %d)", len(routes), args.routes, args.cov_routes)

    sens_rows, cov_rows = [], []
    if args.part in ("all", "sensitivity"):
        tasks = []
        for r in routes[: args.routes]:
            for sidx, st in enumerate(SETTINGS):
                tasks.append({"route": r, "setting": st, "sidx": sidx, "n_main": args.draws,
                              "n_ref": args.ref_draws})
        tasks = [t for k, t in enumerate(tasks) if k % shard_n == shard_i]
        if args.reverse:
            tasks = tasks[::-1]
        log.info("sensitivity tasks: %d (shard %d/%d)", len(tasks), shard_i, shard_n)
        sens_rows = _pool_map(run_setting, tasks, args.workers, log, "sens", sens_key, out / CKPT)
        for r in sens_rows:
            if r["error"]:
                log.error("%s %s FAILED: %s", r["route_id"], r["setting"], r["error"])
        sens_rows.sort(key=lambda r: (r["route_index"], r["instance"], r["setting"]))
        with open(out / SENS_CSV, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=SENS_COLUMNS)
            wr.writeheader()
            for r in sens_rows:
                wr.writerow({k: r.get(k, "") for k in SENS_COLUMNS})
        log.info("wrote %s", SENS_CSV)

    if args.part in ("all", "coverage"):
        tasks = [{"route": r, "n_cov": args.cov_draws, "reps": args.reps, "n_ref": args.ref_draws}
                 for r in routes[: args.cov_routes]]
        tasks = [t for k, t in enumerate(tasks) if k % shard_n == shard_i]
        if args.reverse:
            tasks = tasks[::-1]
        log.info("coverage routes: %d (shard %d/%d)", len(tasks), shard_i, shard_n)
        nested = _pool_map(run_coverage_route, tasks, args.workers, log, "cov", cov_key, out / CKPT)
        cov_rows = [row for rows in nested for row in rows]
        with open(out / COV_CSV, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=COV_COLUMNS)
            wr.writeheader()
            for r in cov_rows:
                wr.writerow({k: r.get(k, "") for k in COV_COLUMNS})
        log.info("wrote %s", COV_CSV)

    elapsed = time.time() - t_start
    (out / SUMMARY_MD).write_text(summary_markdown(sens_rows, cov_rows, cfg, elapsed))
    log.info("wrote %s in %.1f min", SUMMARY_MD, elapsed / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

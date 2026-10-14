"""RARECERT R1: closed-form stage-mixture IS (SMIS) on the QUASAR rare-event routes.

Spec: papers/rarecert/THEORY.md section 4 and rarecert/smis.py. Routes and thresholds
come from results/quasar/rare_routes.csv (QUASAR direction A, 120 routes, gamma in
1e-3..1e-6). The route means are rebuilt with the same sampler as run_quasar_rare.py
(seed 21, SAA gate, L >= 4), and the route ids are checked against the CSV.

For each (route, gamma) with the threshold c of the CSV:
* SMIS with N = 1e6 draws, eta = 1e-3, n_arc = 0 and n_arc = 3 (quantum.rare.is_mixture);
* the stage-scale ratio min/max of the two end blocks (rarecert.smis.stage_scales).
Comparison with the CE mixture (p_is, RV_is, pilot_draws) and the single shift
(p_single, RV_single) from the CSV, and with the exact DP p_dp where present (else
p_is) as the reference.

Usage (from svrpspd_wdro/):
    python3 scripts/run_rarecert_smis.py [--workers 3] [--routes N] [--out-dir DIR]

Outputs (default results/rarecert/): smis_routes.csv (one row per route and gamma),
smis_summary.md, smis.log.
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
from quantum.rare import RouteSim, is_mixture  # noqa: E402
from rarecert.smis import smis_mixture, stage_scales  # noqa: E402

RARE_CSV = ROOT / "results" / "quasar" / "rare_routes.csv"
OUT_DIR = ROOT / "results" / "rarecert"
ROUTES_CSV = "smis_routes.csv"
SUMMARY_MD = "smis_summary.md"
LOG_FILE = "smis.log"
N_DRAWS = 1_000_000
ETA = 1e-3
N_ARCS = (0, 3)
EPS = (0.1, 0.01)
Z2 = 1.96 ** 2
BANDS = ((0.0, 1.0 / 3.0, "[0,1/3]"), (1.0 / 3.0, 0.5, "(1/3,0.5]"),
         (0.5, 0.75, "(0.5,0.75]"), (0.75, 1.0, "(0.75,1]"))
SEED_SMIS = 41

COLUMNS = [
    "route_id", "instance", "route_index", "L", "gamma", "c", "ratio", "lam_star",
    "p_ref", "p_ref_kind", "p_is", "RV_is", "pilot_draws", "p_single", "RV_single", "p_dp",
    "smis0_p", "smis0_rse", "smis0_RV", "smis0_ESS", "smis0_modes", "smis0_build_s", "smis0_unreach",
    "smis3_p", "smis3_rse", "smis3_RV", "smis3_ESS", "smis3_modes", "smis3_build_s", "smis3_unreach",
    "rel_smis0", "rel_smis3", "rel_is", "rel_single", "seconds", "error",
]


# ============================================================
# Inputs
# ============================================================


def load_csv_routes(path: Path) -> dict:
    """(route_id, gamma) -> CSV fields (c, p_is, RV_is, pilot_draws, p_single, RV_single, p_dp)."""
    out: dict = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["route_id"], float(r["gamma"]))
            if key in out:
                continue
            out[key] = {
                "instance": r["instance"], "route_index": int(r["route_index"]), "L": int(r["L"]),
                "c": float(r["c"]), "p_is": float(r["p_is"]), "RV_is": float(r["RV_is"]),
                "pilot_draws": float(r["pilot_draws"]), "p_single": float(r["p_single"]),
                "RV_single": float(r["RV_single"]), "p_dp": float(r["p_dp"]) if r["p_dp"] else math.nan,
            }
    return out


def route_means() -> dict:
    """route_id -> (dbar, pbar), rebuilt with the QUASAR sampler (seed 21)."""
    pool = rq.collect_routes(rq.PLANS)
    routes = rq.order_round_robin(rq.sample_routes(pool, rq.SEED))
    out = {}
    for idx, r in enumerate(routes):
        rid = f"{r['instance']}#{r['route_index']}"
        out[rid] = {"idx": idx, "dbar": r["dbar"], "pbar": r["pbar"], "instance": r["instance"],
                    "route_index": r["route_index"], "L": len(r["nodes"])}
    return out


# ============================================================
# Per-(route, gamma) work
# ============================================================


def _nan_row(rid, gamma, meta, err):
    row = {k: math.nan for k in COLUMNS}
    row.update({"route_id": rid, "gamma": gamma, "p_ref_kind": "", "error": err})
    if meta:
        row.update({"instance": meta["instance"], "route_index": meta["route_index"], "L": meta["L"],
                    "c": meta["c"], "p_is": meta["p_is"], "RV_is": meta["RV_is"],
                    "pilot_draws": meta["pilot_draws"], "p_single": meta["p_single"],
                    "RV_single": meta["RV_single"], "p_dp": meta["p_dp"]})
    return row


def run_task(task: dict) -> dict:
    t0 = time.time()
    rid, gamma, meta = task["rid"], task["gamma"], task["csv"]
    try:
        sim = RouteSim(task["dbar"], task["pbar"], cv=rq.CV, rho=rq.RHO)
        c = meta["c"]
        st = stage_scales(sim)
        ratio = st["ratio"]
        p_dp = meta["p_dp"]
        if math.isfinite(p_dp):
            p_ref, kind = p_dp, "dp"
        else:
            p_ref, kind = meta["p_is"], "is"
        row = {k: math.nan for k in COLUMNS}
        row.update({"route_id": rid, "instance": meta["instance"], "route_index": meta["route_index"],
                    "L": meta["L"], "gamma": gamma, "c": c, "ratio": ratio, "lam_star": st["lam_star"],
                    "p_ref": p_ref, "p_ref_kind": kind, "p_is": meta["p_is"], "RV_is": meta["RV_is"],
                    "pilot_draws": meta["pilot_draws"], "p_single": meta["p_single"],
                    "RV_single": meta["RV_single"], "p_dp": p_dp, "error": ""})
        rel_is = meta["p_is"] / p_ref - 1.0 if p_ref > 0 else math.nan
        row["rel_is"] = abs(rel_is)
        row["rel_single"] = abs(meta["p_single"] / p_ref - 1.0) if p_ref > 0 else math.nan
        for n_arc in N_ARCS:
            tag = f"smis{n_arc}"
            rng = np.random.default_rng([SEED_SMIS, task["idx"], int(round(gamma * 1e9)), n_arc])
            mix, diag = smis_mixture(sim, c, eta=ETA, n_arc=n_arc)
            p, RV, rse, ess = is_mixture(sim, c, mix, N_DRAWS, rng)
            row[f"{tag}_p"] = p
            row[f"{tag}_rse"] = rse
            row[f"{tag}_RV"] = RV
            row[f"{tag}_ESS"] = ess
            row[f"{tag}_modes"] = diag["n_modes"]
            row[f"{tag}_build_s"] = diag["build_seconds"]
            row[f"{tag}_unreach"] = len(diag["unreachable"])
            row[f"rel_{tag}"] = abs(p / p_ref - 1.0) if p_ref > 0 else math.nan
        row["seconds"] = time.time() - t0
        return {"rid": rid, "gamma": gamma, "row": row, "error": ""}
    except Exception as exc:  # keep the pool alive; the failure is recorded in the row
        return {"rid": rid, "gamma": gamma, "row": _nan_row(rid, gamma, meta, f"{type(exc).__name__}: {exc}"),
                "error": f"{type(exc).__name__}: {exc}"}


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


def _band_rows(rows, lo, hi, first):
    out = []
    for r in rows:
        ratio = r["ratio"]
        if not math.isfinite(ratio):
            continue
        ok = (lo <= ratio <= hi) if first else (lo < ratio <= hi)
        if ok:
            out.append(r)
    return out


def summary_markdown(rows: list[dict], n_routes: int, elapsed: float, n_fail: int) -> str:
    lines = ["# RARECERT R1: closed-form SMIS on the QUASAR routes\n"]
    lines.append(
        f"{n_routes} routes from results/quasar/rare_routes.csv, gamma in 1e-3..1e-6 "
        f"({len(rows)} route-gamma rows, {n_fail} failed). SMIS with N = {N_DRAWS:,} draws, eta = {ETA:g}, "
        f"n_arc = 0 and 3. Reference p_ref = p_dp where present (L <= 8, gamma >= 1e-4), else the "
        f"CE mixture p_is. Wall time {elapsed / 60:.1f} min.\n")
    lines.append("Relative errors are |p/p_ref - 1|. RV is the relative variance per draw. "
                 "CE is the CSV mixture (RV_is, pilot_draws); single is the single CE shift.\n")
    lines.append("## Per gamma and ratio band\n")
    lines.append("| gamma | band | n | err SMIS0 med/q90 | err SMIS3 med/q90 | err CE med/q90 | "
                 "err single med/q90 | RV SMIS0 med | RV SMIS3 med | RV CE med | RV single med | "
                 "modes SMIS0 med | modes SMIS3 med | build s SMIS0 med | build s SMIS3 med |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    gammas = sorted({r["gamma"] for r in rows}, reverse=True)
    for g in gammas:
        for lo, hi, name in BANDS:
            sub = _band_rows([r for r in rows if r["gamma"] == g], lo, hi, first=(lo == 0.0))
            if not sub:
                lines.append(f"| {g:g} | {name} | 0 | | | | | | | | | | | | |")
                continue
            e0 = [r["rel_smis0"] for r in sub]
            e3 = [r["rel_smis3"] for r in sub]
            ec = [r["rel_is"] for r in sub]
            es = [r["rel_single"] for r in sub]
            lines.append(
                f"| {g:g} | {name} | {len(sub)} | {_f(_med(e0), '.2%')} / {_f(_q90(e0), '.2%')} | "
                f"{_f(_med(e3), '.2%')} / {_f(_q90(e3), '.2%')} | {_f(_med(ec), '.2%')} / {_f(_q90(ec), '.2%')} | "
                f"{_f(_med(es), '.2%')} / {_f(_q90(es), '.2%')} | "
                f"{_f(_med([r['smis0_RV'] for r in sub]))} | {_f(_med([r['smis3_RV'] for r in sub]))} | "
                f"{_f(_med([r['RV_is'] for r in sub]))} | {_f(_med([r['RV_single'] for r in sub]))} | "
                f"{_f(_med([r['smis0_modes'] for r in sub]))} | {_f(_med([r['smis3_modes'] for r in sub]))} | "
                f"{_f(_med([r['smis0_build_s'] for r in sub]), '.2f')} | "
                f"{_f(_med([r['smis3_build_s'] for r in sub]), '.2f')} |")
    lines.append("")
    lines.append("## Ratio band of the stage scales\n")
    lines.append("ratio = min(lam_d[0], lam_p[L]) / max(lam_d[0], lam_p[L]); bands as in the table.\n")
    lines.append("| band | route-gamma rows |")
    lines.append("|---|---|")
    for lo, hi, name in BANDS:
        n = len(_band_rows(rows, lo, hi, first=(lo == 0.0)))
        lines.append(f"| {name} | {n} |")
    lines.append("")
    lines.append("## Classical cost (95 per cent, sample counts)\n")
    lines.append("SMIS = 1.96^2 RV / eps^2 (no pilot). CE = pilot_draws + 1.96^2 RV_is / eps^2. "
                     "Totals are sums over routes (rows with finite RV only), per gamma.\n")
    lines.append("| gamma | eps_rel | rows | SMIS0 total | SMIS3 total | CE total | CE / SMIS0 | CE / SMIS3 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for g in gammas:
        for eps in EPS:
            sub = [r for r in rows if r["gamma"] == g]
            c0 = [Z2 * r["smis0_RV"] / eps ** 2 for r in sub if math.isfinite(r["smis0_RV"])]
            c3 = [Z2 * r["smis3_RV"] / eps ** 2 for r in sub if math.isfinite(r["smis3_RV"])]
            ce = [r["pilot_draws"] + Z2 * r["RV_is"] / eps ** 2 for r in sub
                  if math.isfinite(r["RV_is"]) and math.isfinite(r["pilot_draws"])]
            t0, t3, tc = sum(c0), sum(c3), sum(ce)
            lines.append(f"| {g:g} | {eps:g} | {len(sub)} | {t0:.4g} | {t3:.4g} | {tc:.4g} | "
                         f"{_f(tc / t0 if t0 > 0 else math.nan, '.3g')} | "
                         f"{_f(tc / t3 if t3 > 0 else math.nan, '.3g')} |")
    lines.append("")
    return "\n".join(lines)


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
    log = logging.getLogger("smis")
    t_start = time.time()

    csv_routes = load_csv_routes(RARE_CSV)
    means = route_means()
    rids = sorted({k[0] for k in csv_routes}, key=lambda r: means.get(r, {"idx": 1 << 30})["idx"])
    missing = [r for r in rids if r not in means]
    if missing:
        raise SystemExit(f"route ids in the CSV not rebuilt by the sampler: {missing[:5]} ...")
    if args.routes is not None:
        rids = rids[: max(0, args.routes)]
    gammas = sorted({k[1] for k in csv_routes}, reverse=True)
    tasks = []
    for rid in rids:
        for g in gammas:
            key = (rid, g)
            if key not in csv_routes:
                continue
            tasks.append({"rid": rid, "gamma": g, "idx": means[rid]["idx"], "csv": csv_routes[key],
                          "dbar": means[rid]["dbar"], "pbar": means[rid]["pbar"]})
    log.info("routes %d, route-gamma tasks %d, workers %d", len(rids), len(tasks), args.workers)

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
                log.info("[%d/%d] %s gamma=%g done in %.1fs: rel0=%.3g rel3=%.3g RV0=%.3g RV3=%.3g modes=%d/%d",
                         k, len(tasks), res["rid"], res["gamma"], r["seconds"], r["rel_smis0"], r["rel_smis3"],
                         r["smis0_RV"], r["smis3_RV"], r["smis0_modes"], r["smis3_modes"])

    rows.sort(key=lambda r: (r["route_id"], -r["gamma"]))
    with open(out / ROUTES_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in COLUMNS})
    elapsed = time.time() - t_start
    md = summary_markdown(rows, len(rids), elapsed, len(fails))
    (out / SUMMARY_MD).write_text(md)
    log.info("wrote %s and %s in %.1f min; failed rows: %d %s", ROUTES_CSV, SUMMARY_MD, elapsed / 60,
             len(fails), fails)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

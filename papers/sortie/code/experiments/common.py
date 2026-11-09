"""Shared helpers for the SORTIE experiment scripts (Module F).

Import this module before numpy so the thread limits take effect.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import csv
import json
import re
import sys
import time
import traceback
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PAPER_DIR = os.path.dirname(CODE_DIR)
DEFAULT_RESULTS = os.path.join(PAPER_DIR, "results")

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from sortie.instances import (  # noqa: E402
    CONG_SUFFIX, DATA_PATH, base_name, effective_H, load_x641, make_instance, noise_matrix,
    with_congestion,
)
from sortie.pboi import exante, expost_var, lam_vector, pboi  # noqa: E402
from sortie.search import SCEN_SEED, solve_fair  # noqa: E402
from sortie.sim import evaluate_plans  # noqa: E402
from sortie.tsp import is_tsp_optimal, tour_stats  # noqa: E402

RESULTS = os.path.abspath(os.environ.get("SORTIE_RESULTS") or DEFAULT_RESULTS)
PLANS = os.path.join(RESULTS, "plans")
DEFAULT_PLANS = os.path.abspath(os.environ.get("SORTIE_PLANS") or os.path.join(DEFAULT_RESULTS, "plans"))

V_DEF = 0.3
SY = 0.15
S_OOS = 20000
S_OOS_LARGE = 10000
SOLVE_SEED = 7
SCEN_S = 200
N_WORKERS = 4
TSP_CHECK_MAX = 15

MEASURES = ("max", "range", "gmd", "mad", "var")
ORDERED = ("max", "range", "gmd", "mad")

METHODS = {
    "det": dict(mode="det", kappa=0.0, screen="price", consistency="lower"),
    "saa": dict(mode="saa", kappa=0.0, screen="price", consistency="lower"),
    "sortie0": dict(mode="pboi", kappa=0.0, screen="price", consistency="lower"),
    "sortie1": dict(mode="pboi", kappa=1.0, screen="price", consistency="lower"),
    "det_lower": dict(mode="det", kappa=0.0, screen="price", consistency="lower"),
    "det_free": dict(mode="det", kappa=0.0, screen="price", consistency="free"),
    "pboi_lower": dict(mode="pboi", kappa=0.0, screen="price", consistency="lower"),
    "pboi_free": dict(mode="pboi", kappa=0.0, screen="price", consistency="free"),
    "sortie0_random": dict(mode="pboi", kappa=0.0, screen="random", consistency="lower"),
}

SEED_TIME = {15: 20.0, 20: 20.0, 25: 20.0, 30: 40.0, 35: 40.0, 50: 60.0, 75: 120.0, 100: 120.0}
TINY_SEED_TIME = 5.0
TINY_N = 10
TINY_M = 3
TINY_CAP = 2.4
MAIN_M = 5
MAIN_CAP = 4.0

MAIN_GRID = [(n, b) for n in (15, 20, 25, 30, 35) for b in range(5)]
LARGE_GRID = [(n, b) for n in (50, 75, 100) for b in range(6)]
E3_NS = (15, 20, 25, 30, 35)
QUICK_INST = (15, 0)


ROW_FIELDS = [
    "key", "experiment", "instance", "n", "b", "m", "measure", "B", "method", "v", "kappa",
    "time_limit", "seed", "eval_seed", "S", "sY", "Cstar", "sumL", "cost_ratio",
    "obj_insample", "pboi_pred", "exante",
    "E_max", "E_range", "E_gmd", "E_mad", "E_var",
    "E_max_X", "E_range_X", "E_gmd_X", "E_mad_X", "E_var_X",
    "CVaR90_range", "CVaR90_range_X", "oos_obj", "oos_obj_X",
    "nontsp_routes", "excess_len", "iters", "n_exact_evals", "runtime",
    "n_improve", "n_perturb", "n_accept", "solver_calls", "solver_hk", "solver_cache_hits",
    "screen", "consistency", "plan_file",
]


def inst_seed(n, b):
    """Deterministic per-instance evaluation seed 2026*1000 + n*10 + b (b < 10)."""
    return 2026 * 1000 + int(n) * 10 + int(b)


def time_limit_for(n):
    """Fair-run time limit by size (e3, e4, e5, e6)."""
    return 15.0 if n <= 25 else 25.0


@lru_cache(maxsize=1)
def _raw():
    return load_x641(DATA_PATH)


def load_instance(n, b, tiny=False, cong=False):
    """Block instance; cong=True applies the heterogeneous-noise multiplier (with_congestion)."""
    coords, demand = _raw()
    if tiny:
        inst = make_instance(coords, demand, n, b, m=TINY_M, cap_div=TINY_CAP)
    else:
        inst = make_instance(coords, demand, n, b, m=MAIN_M, cap_div=MAIN_CAP)
    return with_congestion(inst) if cong else inst


def instance_name(n, b, cong=False):
    return f"X641-n{n}-b{b}" + (CONG_SUFFIX if cong else "")


def ensure_dirs():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PLANS, exist_ok=True)


def _seed_path(name, plans_dir):
    return os.path.join(plans_dir, f"seed_{name}.json")


def _seed_file(name):
    for d in (PLANS, DEFAULT_PLANS):
        path = _seed_path(name, d)
        if os.path.exists(path):
            return path
    raise RuntimeError(f"missing seed plan for {name}; run experiments/e0_seed.py first")


def load_seed(name):
    """Seed record (routes, Cstar) for an instance, from RESULTS/plans or the default plans dir."""
    with open(_seed_file(name), "r", encoding="utf-8") as fh:
        rec = json.load(fh)
    rec["routes"] = [[int(x) for x in r] for r in rec["routes"]]
    return rec


def save_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, path)


def safe_name(key):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", key)[:200]


def plan_file_for(exp, key):
    return os.path.join(PLANS, exp, safe_name(key) + ".json")


def scen_tensor(N, v, S=SCEN_S, seed=SCEN_SEED, H=None):
    """Arc scenario tensor for the SAA method (independent of the evaluation seed)."""
    rng = np.random.default_rng(seed)
    if H is None:
        return rng.gamma(shape=1.0 / (v * v), scale=v * v, size=(S, N, N))
    vh = v * np.asarray(H, dtype=np.float64)
    return rng.gamma(shape=1.0 / (vh * vh), scale=vh * vh, size=(S, N, N))


def route_stats(routes, C, D=None):
    Ls, Ws = [], []
    for r in routes:
        L, W = tour_stats(r, C, D)
        Ls.append(L)
        Ws.append(W)
    return np.array(Ls), np.array(Ws)


def tsp_excess(routes, C):
    """(number of non-TSP-optimal routes, total excess length); NaN if any route has |S| > 15."""
    if max(len(r) for r in routes) > TSP_CHECK_MAX:
        return float("nan"), float("nan")
    nontsp = 0
    excess = 0.0
    for r in routes:
        ok, l_opt = is_tsp_optimal(r, C)
        L, _ = tour_stats(r, C)
        if not ok:
            nontsp += 1
        excess += max(L - l_opt, 0.0)
    return float(nontsp), float(excess)


def model_predictions(L, W, v, measure):
    """(PBOI-predicted ex-post value, ex-ante value) of a plan for one measure."""
    L = np.asarray(L, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    m = len(L)
    sigma = v * np.sqrt(np.maximum(W, 0.0))
    if measure == "var":
        return float(expost_var(L, sigma)), float(np.mean((L - L.mean()) ** 2))
    lam = lam_vector(measure, m)
    return float(pboi(L, sigma, lam)), float(exante(L, lam))


def oos_fields(ev):
    out = {}
    for name in ("max", "range", "gmd", "mad", "var"):
        out[f"E_{name}"] = float(ev.get(f"E_{name}", float("nan")))
        out[f"E_{name}_X"] = float(ev.get(f"E_{name}_X", float("nan")))
    out["CVaR90_range"] = float(ev["CVaR90_range"])
    out["CVaR90_range_X"] = float(ev["CVaR90_range_X"])
    return out


def build_row(exp, key, inst, routes, measure, B, method, v, kappa, time_limit, seed,
              eval_seed, S, Cstar, res, ev, screen, consistency, plan_file, sY=SY):
    """Flat result row shared by all solve and seed experiments."""
    C = inst.C
    L, W = route_stats(routes, C, noise_matrix(inst))
    sumL = float(L.sum())
    row = {
        "key": key,
        "experiment": exp,
        "instance": inst.name,
        "n": int(len(inst.demand) - 1),
        "b": None,
        "m": int(inst.m),
        "measure": measure,
        "B": B,
        "method": method,
        "v": v,
        "kappa": kappa,
        "time_limit": time_limit,
        "seed": seed,
        "eval_seed": eval_seed,
        "S": S,
        "sY": sY,
        "Cstar": Cstar,
        "sumL": sumL,
        "cost_ratio": sumL / Cstar if Cstar else float("nan"),
        "screen": screen,
        "consistency": consistency,
        "plan_file": plan_file,
    }
    if res is not None:
        row["obj_insample"] = float(res["value"])
        row["iters"] = res["iters"]
        row["n_exact_evals"] = res["n_exact_evals"]
        row["runtime"] = res["time"]
        row["n_improve"] = res["n_improve"]
        row["n_perturb"] = res["n_perturb"]
        row["n_accept"] = res["n_accept"]
        row["solver_calls"] = res["solver"]["n_calls"]
        row["solver_hk"] = res["solver"]["n_hk"]
        row["solver_cache_hits"] = res["solver"]["n_cache_hits"]
    else:
        row["obj_insample"] = sumL
        row["iters"] = 0
        row["n_exact_evals"] = 0
        row["runtime"] = 0.0
    if measure in MEASURES:
        pred, exa = model_predictions(L, W, v, measure)
    else:
        pred, exa = float("nan"), float("nan")
    row["pboi_pred"] = pred
    row["exante"] = exa
    row.update(oos_fields(ev))
    row["oos_obj"] = row.get(f"E_{measure}", float("nan")) if measure in MEASURES else float("nan")
    row["oos_obj_X"] = row.get(f"E_{measure}_X", float("nan")) if measure in MEASURES else float("nan")
    nontsp, excess = tsp_excess(routes, C)
    row["nontsp_routes"] = nontsp
    row["excess_len"] = excess
    return row


def evaluate_plan_row(exp, key, inst, routes, measure, B, method, v, kappa, time_limit, seed,
                      eval_seed, S, Cstar, res, screen, consistency, plan_file, b, sY=SY):
    ev = evaluate_plans(inst, [routes], v, sY, S=S, seed=eval_seed)[0]
    row = build_row(exp, key, inst, routes, measure, B, method, v, kappa, time_limit, seed,
                    eval_seed, S, Cstar, res, ev, screen, consistency, plan_file, sY=sY)
    row["b"] = int(b)
    return row


def run_unit(job):
    """One fair run: solve_fair from the seed cost plan, then out-of-sample evaluation.

    job keys: exp, n, b, tiny, cong, measure, B, method, v, time_limit, S, eval_seed, key.
    """
    exp = job["exp"]
    n, b = job["n"], job["b"]
    inst = load_instance(n, b, tiny=job.get("tiny", False), cong=job.get("cong", False))
    seed_rec = load_seed(base_name(inst.name))
    Cstar = float(seed_rec["Cstar"])
    spec = METHODS[job["method"]]
    v = float(job["v"])
    scen = None
    if spec["mode"] == "saa":
        scen = scen_tensor(inst.C.shape[0], v, H=effective_H(inst))
    budget = float(job["B"]) * Cstar
    res = solve_fair(inst, seed_rec["routes"], spec["mode"], job["measure"], v, spec["kappa"],
                     budget, float(job["time_limit"]), SOLVE_SEED, screen=spec["screen"],
                     consistency=spec["consistency"], scen=scen)
    routes = res["routes"]
    S = int(job["S"])
    pf = plan_file_for(exp, job["key"])
    save_json_atomic(pf, {
        "key": job["key"], "instance": inst.name, "method": job["method"],
        "measure": job["measure"], "B": job["B"], "v": v, "routes": routes,
        "L": res["L"], "W": res["W"], "value": res["value"], "budget": budget,
        "history": res["history"],
    })
    row = evaluate_plan_row(exp, job["key"], inst, routes, job["measure"], job["B"],
                            job["method"], v, spec["kappa"], job["time_limit"], SOLVE_SEED,
                            job["eval_seed"], S, Cstar, res, spec["screen"],
                            spec["consistency"], os.path.relpath(pf, RESULTS), b)
    return [row]


def run_cost_unit(job):
    """Out-of-sample evaluation of the e0 cost plan (no solve)."""
    inst = load_instance(job["n"], job["b"], tiny=job.get("tiny", False),
                         cong=job.get("cong", False))
    rec = load_seed(base_name(inst.name))
    routes = rec["routes"]
    cstar = float(rec["Cstar"])
    pf = os.path.relpath(_seed_file(base_name(inst.name)), RESULTS)
    row = evaluate_plan_row(job["exp"], job["key"], inst, routes, "cost", float("nan"), "cost",
                            V_DEF, 0.0, float(rec["seed_time"]), inst_seed(job["n"], job["b"]),
                            inst_seed(job["n"], job["b"]), int(job["S"]), cstar, None, "",
                            "lower", pf, job["b"], sY=SY)
    return [row]


def run_any(job):
    if job["method"] == "cost":
        return run_cost_unit(job)
    return run_unit(job)


def solve_key(exp, n, b, measure, B, method, v, tl, S, cong=False):
    return (f"{exp}|{instance_name(n, b, cong)}|{measure}|B{B:.2f}|{method}|v{v:g}|t{tl:g}|S{S}")


def make_job(exp, n, b, measure, B, method, v, tl, S, tiny=False, cong=False):
    return dict(exp=exp, n=n, b=b, tiny=tiny, cong=cong, measure=measure, B=B, method=method,
                v=v, time_limit=tl, S=S, eval_seed=inst_seed(n, b),
                key=solve_key(exp, n, b, measure, B, method, v, tl, S, cong))


def make_cost_job(exp, n, b, S, tiny=False, cong=False):
    return dict(exp=exp, n=n, b=b, tiny=tiny, cong=cong, measure="cost", B=float("nan"),
                method="cost", v=V_DEF, time_limit=0.0, S=S, eval_seed=inst_seed(n, b),
                key=f"{exp}|{instance_name(n, b, cong)}|cost|S{S}")


def _guard(args):
    """Run one job, returning ("ok", rows) or ("err", key, message)."""
    worker, job = args
    try:
        return ("ok", job["key"], worker(job))
    except Exception as exc:  # noqa: BLE001
        msg = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return ("err", job["key"], msg)


def _done_keys(csv_path):
    if not os.path.exists(csv_path):
        return set()
    keys = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("key"):
                keys.add(row["key"])
    return keys


def _ends_with_newline(path):
    with open(path, "rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) == b"\n"


def _clean(v):
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def run_jobs(exp, jobs, worker, csv_name, fields, n_workers=N_WORKERS):
    """Run jobs in a Pool with resume. Each job yields a list of rows (or none).

    Rows are appended to RESULTS/csv_name as each job finishes. Jobs whose key
    already appears in the CSV are skipped. Failures go to RESULTS/<exp>_failures.log.
    """
    ensure_dirs()
    csv_path = os.path.join(RESULTS, csv_name)
    done = _done_keys(csv_path)
    todo = [j for j in jobs if j["key"] not in done]
    print(f"[{exp}] jobs total={len(jobs)} already_done={len(jobs) - len(todo)} todo={len(todo)}",
          flush=True)
    fail_path = os.path.join(RESULTS, f"{exp}_failures.log")
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    t0 = time.perf_counter()
    n_ok = 0
    n_err = 0
    if not new_file and not _ends_with_newline(csv_path):
        with open(csv_path, "a", encoding="utf-8") as fh:
            fh.write("\n")
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", restval="")
        if new_file:
            writer.writeheader()
            fh.flush()
        if todo:
            with Pool(n_workers) as pool:
                for k, out in enumerate(pool.imap_unordered(_guard, [(worker, j) for j in todo]), 1):
                    if out[0] == "err":
                        n_err += 1
                        with open(fail_path, "a", encoding="utf-8") as ff:
                            ff.write(f"== {out[1]}\n{out[2]}\n")
                        print(f"[{exp}] FAILED {out[1]} ({k}/{len(todo)})", flush=True)
                        continue
                    n_ok += 1
                    for row in out[2]:
                        writer.writerow({kk: _clean(vv) for kk, vv in row.items()})
                    fh.flush()
                    os.fsync(fh.fileno())
                    print(f"[{exp}] {k}/{len(todo)} {out[1]} t={time.perf_counter() - t0:.0f}s",
                          flush=True)
    print(f"[{exp}] finished ok={n_ok} failed={n_err} elapsed={time.perf_counter() - t0:.0f}s",
          flush=True)
    return n_err


def parse_args(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke run: 1 instance, 1 measure, 2 s limits, S=2000")
    return ap.parse_args()


def quick_time(args, full):
    return 2.0 if args.quick else full


def quick_S(args, full):
    return 2000 if args.quick else full


def print_header(exp, args):
    print(f"== {exp} quick={args.quick} results={RESULTS} workers={N_WORKERS} "
          f"started={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

"""E17: second benchmark family. The 12 CVRPLIB X files in data/X, each reduced to its depot
plus the first 50 customers in file order (make_head_instance, m = 5, cap_div = 4), instance
name {file stem}-h50. Phase 1 computes the seed plans with solve_cost (60 s, 3 restarts) and
caches them as plans/seed_{stem}-h50.json (Cstar in e17_seeds.csv). Phase 2: measures
{range, gmd, max}, B = 1.10, methods det, saa, sortie0, detsig (z = 1), 120 s, solver seeds
{7, 11}, out-of-sample S = 10000 with CRN seed inst_seed(50, k), k the file index.
"""

import json
import os

import common as cm
from common import (  # noqa: E402
    EXT_FIELDS, ROW_FIELDS, S_OOS_LARGE, XDIR, inst_seed, load_job_instance, make_xjob,
    parse_args, print_header, quick_S, quick_time, run_any, run_jobs, save_json_atomic,
)
from sortie.search import solve_cost  # noqa: E402

EXP = "e17"
CSV = "e17_family.csv"
SEED_CSV = "e17_seeds.csv"
N = 50
M = 5
CAP_DIV = 4.0
MEASURES = ("range", "gmd", "max")
METHODS = ("det", "saa", "sortie0", "detsig")
SOLVE_SEEDS = (7, 11)
B = 1.10
TIME_LIMIT = 120.0
SEED_TIME = 60.0
RESTARTS = 3
FIELDS = ROW_FIELDS + EXT_FIELDS
SEED_FIELDS = ["key", "experiment", "instance", "n", "b", "m", "cap_div", "Cstar", "seed_time",
               "restarts", "solver_seed", "plan_file"]


def stems():
    return sorted(f[:-4] for f in os.listdir(XDIR) if f.endswith(".vrp"))


def seed_jobs(args):
    all_stems = stems()
    chosen = all_stems[:1] if args.quick else all_stems
    jobs = []
    for stem in chosen:
        k = all_stems.index(stem)
        label = f"{stem}-h{N}"
        jobs.append(dict(exp=EXP, key=f"e17seed|{label}", stem=stem, label=label, n=N, b=k,
                         m=M, cap_div=CAP_DIV, seed_time=quick_time(args, SEED_TIME),
                         solver_seed=inst_seed(N, k)))
    return jobs


def seed_worker(job):
    """Seed plan of one head instance: cached JSON if present, otherwise solve_cost."""
    inst = load_job_instance(job)
    path = cm.seed_file_path(inst.name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
    else:
        routes, cstar = solve_cost(inst, job["seed_time"], seed=job["solver_seed"],
                                   restarts=RESTARTS)
        rec = {
            "name": inst.name, "stem": job["stem"], "n": N, "m": M, "Q": int(inst.Q),
            "cap_div": CAP_DIV, "Cstar": float(cstar),
            "routes": [[int(x) for x in r] for r in routes], "seed_time": job["seed_time"],
            "solver_seed": job["solver_seed"], "restarts": RESTARTS,
        }
        save_json_atomic(path, rec)
    return [{
        "key": job["key"], "experiment": EXP, "instance": inst.name, "n": N, "b": job["b"],
        "m": M, "cap_div": CAP_DIV, "Cstar": rec["Cstar"], "seed_time": rec["seed_time"],
        "restarts": RESTARTS, "solver_seed": rec["solver_seed"],
        "plan_file": os.path.relpath(path, cm.RESULTS),
    }]


def solve_jobs(args):
    all_stems = stems()
    chosen = all_stems[:1] if args.quick else all_stems
    measures = ("range",) if args.quick else MEASURES
    seeds = SOLVE_SEEDS[:1] if args.quick else SOLVE_SEEDS
    tl = quick_time(args, TIME_LIMIT)
    S = quick_S(args, S_OOS_LARGE)
    jobs = []
    for stem in chosen:
        k = all_stems.index(stem)
        label = f"{stem}-h{N}"
        for meas in measures:
            for method in METHODS:
                for sd in seeds:
                    jobs.append(make_xjob(EXP, label, N, k, meas, B, method, cm.V_DEF, tl, S,
                                          stem=stem, m=M, cap_div=CAP_DIV, family="X",
                                          solve_seed=sd))
    return jobs


def main():
    args = parse_args("E17: second benchmark family (X set, head instances)")
    print_header(EXP, args)
    seeds = seed_jobs(args)
    n_err = run_jobs(EXP, seeds, seed_worker, SEED_CSV, SEED_FIELDS)
    jobs = solve_jobs(args)
    n_err += run_jobs(EXP, jobs, run_any, CSV, FIELDS)
    print(f"[{EXP}] seeds={len(seeds)} jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

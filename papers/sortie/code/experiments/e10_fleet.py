"""E10: fleet size. n = 35, blocks 0-4, m in {3, 5, 7} vehicles with cap_div = m - 1
(m = 5 reuses the E3 instances, cap_div 4), B = 1.10, measures {range, gmd, max}, methods
det, saa and sortie0, 25 s per solve.

Phase 1 computes the seed plans of the m = 3 and m = 7 instances with solve_cost (40 s, 3
restarts) and caches them as plans/seed_X641-n35-b{b}-m{m}.json (Cstar recorded in
e10_seeds.csv). The m = 5 seeds are the E0 seeds of the blocks. Phase 2 is the solve grid.
"""

import json
import os

import common as cm
from common import (  # noqa: E402
    EXT_FIELDS, ROW_FIELDS, S_OOS, inst_seed, make_xjob, load_job_instance,
    parse_args, print_header, quick_S, quick_time, run_any, run_jobs, save_json_atomic,
)
from sortie.search import solve_cost  # noqa: E402

EXP = "e10"
CSV = "e10_fleet.csv"
SEED_CSV = "e10_seeds.csv"
N = 35
BLOCKS = range(5)
MS = (3, 5, 7)
MEASURES = ("range", "gmd", "max")
METHODS = ("det", "saa", "sortie0")
B = 1.10
TIME_LIMIT = 25.0
SEED_TIME = 40.0
RESTARTS = 3
FIELDS = ROW_FIELDS + EXT_FIELDS
SEED_FIELDS = ["key", "experiment", "instance", "n", "b", "m", "cap_div", "Cstar", "seed_time",
               "restarts", "solver_seed", "plan_file"]


def seed_name(b, m):
    return f"X641-n{N}-b{b}" if m == 5 else f"X641-n{N}-b{b}-m{m}"


def seed_jobs(args):
    blocks = (0,) if args.quick else BLOCKS
    jobs = []
    for b in blocks:
        for m in MS:
            if m == 5:
                continue
            name = seed_name(b, m)
            jobs.append(dict(
                exp=EXP, key=f"e10seed|{name}", n=N, b=b, m=m, cap_div=float(m - 1),
                seed_time=quick_time(args, SEED_TIME), solver_seed=inst_seed(N, b), label=name,
            ))
    return jobs


def seed_worker(job):
    """Seed plan of one (instance, m): cached JSON if present, otherwise solve_cost."""
    inst = load_job_instance(job)
    path = cm.seed_file_path(inst.name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
    else:
        routes, cstar = solve_cost(inst, job["seed_time"], seed=job["solver_seed"],
                                   restarts=RESTARTS)
        rec = {
            "name": inst.name, "n": N, "b": job["b"], "m": int(inst.m), "Q": int(inst.Q),
            "cap_div": job["cap_div"], "Cstar": float(cstar),
            "routes": [[int(x) for x in r] for r in routes], "seed_time": job["seed_time"],
            "solver_seed": job["solver_seed"], "restarts": RESTARTS,
        }
        save_json_atomic(path, rec)
    return [{
        "key": job["key"], "experiment": EXP, "instance": inst.name, "n": N, "b": job["b"],
        "m": int(inst.m), "cap_div": job["cap_div"], "Cstar": rec["Cstar"],
        "seed_time": rec["seed_time"], "restarts": RESTARTS, "solver_seed": rec["solver_seed"],
        "plan_file": os.path.relpath(path, cm.RESULTS),
    }]


def solve_jobs(args):
    blocks = (0,) if args.quick else BLOCKS
    measures = ("range",) if args.quick else MEASURES
    jobs = []
    for b in blocks:
        tl = quick_time(args, TIME_LIMIT)
        S = quick_S(args, S_OOS)
        for m in MS:
            name = seed_name(b, m)
            for meas in measures:
                for method in METHODS:
                    jobs.append(make_xjob(EXP, name, N, b, meas, B, method, cm.V_DEF, tl, S,
                                          m=m, cap_div=float(m - 1)))
    return jobs


def main():
    args = parse_args("E10: fleet size m in {3, 5, 7}")
    print_header(EXP, args)
    seeds = seed_jobs(args)
    n_err = run_jobs(EXP, seeds, seed_worker, SEED_CSV, SEED_FIELDS)
    jobs = solve_jobs(args)
    n_err += run_jobs(EXP, jobs, run_any, CSV, FIELDS)
    print(f"[{EXP}] seeds={len(seeds)} jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E0: seed plans. C* per instance by solve_cost, cost-plan out-of-sample statistics.

Writes RESULTS/plans/seed_<instance>.json (routes, Cstar) and RESULTS/e0_seed.csv
(method "cost" rows, full out-of-sample statistics).
"""

import os

import common as cm
from common import (
    MAIN_GRID, LARGE_GRID, ROW_FIELDS, S_OOS, S_OOS_LARGE, SEED_TIME, TINY_N, TINY_SEED_TIME,
    V_DEF, SY, inst_seed, instance_name, load_instance, parse_args, print_header, quick_S,
    quick_time, run_jobs, save_json_atomic, evaluate_plan_row,
)
from sortie.search import solve_cost

EXP = "e0"
CSV = "e0_seed.csv"


def jobs_for(args):
    if args.quick:
        entries = [(TINY_N, 0, True)]
    else:
        entries = [(TINY_N, b, True) for b in range(10)]
        entries += [(n, b, False) for n, b in MAIN_GRID + LARGE_GRID]
    jobs = []
    for n, b, tiny in entries:
        name = instance_name(n, b)
        st = TINY_SEED_TIME if tiny else SEED_TIME[n]
        jobs.append(dict(
            exp=EXP, key=f"{EXP}|{name}", n=n, b=b, tiny=tiny,
            seed_time=quick_time(args, st),
            S=quick_S(args, S_OOS if n < 75 else S_OOS_LARGE),
        ))
    return jobs


def worker(job):
    n, b, tiny = job["n"], job["b"], job["tiny"]
    inst = load_instance(n, b, tiny=tiny)
    seed = inst_seed(n, b)
    routes, cstar = solve_cost(inst, job["seed_time"], seed=seed, restarts=3)
    routes = [[int(x) for x in r] for r in routes]
    seed_path = os.path.join(cm.PLANS, f"seed_{inst.name}.json")
    save_json_atomic(seed_path, {
        "name": inst.name, "n": n, "b": b, "m": int(inst.m), "Q": int(inst.Q), "tiny": tiny,
        "cap_div": cm.TINY_CAP if tiny else cm.MAIN_CAP, "Cstar": float(cstar),
        "routes": routes, "seed_time": job["seed_time"], "solver_seed": seed,
        "restarts": 3,
    })
    row = evaluate_plan_row(
        EXP, job["key"], inst, routes, "cost", float("nan"), "cost", V_DEF, 0.0,
        job["seed_time"], seed, seed, job["S"], float(cstar), None, "", "lower",
        os.path.relpath(seed_path, cm.RESULTS), b, sY=SY,
    )
    return [row]


def main():
    args = parse_args("E0: seed plans via solve_cost")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, worker, CSV, ROW_FIELDS)
    print(f"[{EXP}] instances={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

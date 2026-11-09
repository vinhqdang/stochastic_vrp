"""E13: risk attitude. n in {25, 35} x blocks 0-4, kappa in {0, 0.5, 1, 2, 4}, measures
{range, max}, mode pboi (method sortie0 with the kappa column set), B = 1.10, E3 time limits.

Extra columns, computed on the final plan:
  n_routes_nontsp  routes whose tour differs from the kappa = 0 TourSolver tour of the same
                   customer set (compared up to reversal);
  extra_len_vs_tsp total length minus the total length of those kappa = 0 tours;
  mean_risk        mean over routes of L + v sqrt(W) (W on the noise weights D = C o H).
"""

import json
import os

import numpy as np

import common as cm
from common import (  # noqa: E402
    EXT_FIELDS, ROW_FIELDS, S_OOS, QUICK_INST, make_xjob, parse_args, print_header, quick_S,
    quick_time, run_jobs, run_unit, time_limit_for,
)
from sortie.instances import noise_matrix  # noqa: E402
from sortie.tsp import TourSolver, tour_stats  # noqa: E402

EXP = "e13"
CSV = "e13_kappa.csv"
NS = (25, 35)
BLOCKS = range(5)
KAPPAS = (0.0, 0.5, 1.0, 2.0, 4.0)
MEASURES = ("range", "max")
METHOD = "sortie0"
B = 1.10
FIELDS = ROW_FIELDS + EXT_FIELDS + ["n_routes_nontsp", "extra_len_vs_tsp", "mean_risk"]


def jobs_for(args):
    if args.quick:
        insts = [QUICK_INST]
        measures = ("range",)
    else:
        insts = [(n, b) for n in NS for b in BLOCKS]
        measures = MEASURES
    jobs = []
    for n, b in insts:
        tl = quick_time(args, time_limit_for(n))
        S = quick_S(args, S_OOS)
        label = cm.instance_name(n, b)
        for meas in measures:
            for kappa in KAPPAS:
                jobs.append(make_xjob(EXP, label, n, b, meas, B, METHOD, cm.V_DEF, tl, S,
                                      kappa=kappa))
    return jobs


def kappa_extras(inst, routes, v):
    """Non-TSP route count, extra length against the kappa = 0 tours, and mean risk."""
    C = inst.C
    D = noise_matrix(inst)
    tsp_solver = TourSolver(C, 0.0, v, D=D)
    nontsp = 0
    len_plan = 0.0
    len_tsp = 0.0
    risks = []
    for r in routes:
        seq = [int(x) for x in r]
        tsp = [int(x) for x in tsp_solver.solve(seq)[0]]
        if seq != tsp and seq != tsp[::-1]:
            nontsp += 1
        L, W = tour_stats(seq, C, D)
        L_tsp, _ = tour_stats(tsp, C)
        len_plan += L
        len_tsp += L_tsp
        risks.append(L + v * np.sqrt(max(W, 0.0)))
    return {
        "n_routes_nontsp": nontsp,
        "extra_len_vs_tsp": len_plan - len_tsp,
        "mean_risk": float(np.mean(risks)),
    }


def worker(job):
    rows = run_unit(job)
    inst = cm.load_job_instance(job)
    with open(cm.plan_file_for(EXP, job["key"]), "r", encoding="utf-8") as fh:
        routes = json.load(fh)["routes"]
    extras = kappa_extras(inst, routes, job["v"])
    for row in rows:
        row.update(extras)
    return rows


def main():
    args = parse_args("E13: risk-consistent tours, kappa sweep")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, worker, CSV, FIELDS)
    print(f"[{EXP}] jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

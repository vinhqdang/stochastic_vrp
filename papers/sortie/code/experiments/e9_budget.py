"""E9: budget frontier. Blocks n in {25, 35} x 0-4, B in {1.02, 1.05, 1.10, 1.15, 1.20, 1.30},
measures {range, gmd, max}, methods det and sortie0 (PBOI, kappa 0). E3 time limits (15 s for
n = 25, 25 s for n = 35). Each row carries the cost ratio sum L / C* and the out-of-sample
ex-post value, so the frontier (cost ratio against value) can be plotted per measure.
"""

import common as cm
from common import (  # noqa: E402
    EXT_FIELDS, QUICK_INST, ROW_FIELDS, S_OOS, make_job, parse_args, print_header, quick_S,
    quick_time, run_any, run_jobs, time_limit_for,
)

EXP = "e9"
CSV = "e9_budget.csv"
NS = (25, 35)
BUDGETS = (1.02, 1.05, 1.10, 1.15, 1.20, 1.30)
MEASURES = ("range", "gmd", "max")
METHODS = ("det", "sortie0")
FIELDS = ROW_FIELDS + EXT_FIELDS


def jobs_for(args):
    if args.quick:
        insts = [QUICK_INST]
        measures = ("range",)
    else:
        insts = [(n, b) for n in NS for b in range(5)]
        measures = MEASURES
    jobs = []
    for n, b in insts:
        tl = quick_time(args, time_limit_for(n))
        S = quick_S(args, S_OOS)
        for meas in measures:
            for B in BUDGETS:
                for method in METHODS:
                    jobs.append(make_job(EXP, n, b, meas, B, method, cm.V_DEF, tl, S))
    return jobs


def main():
    args = parse_args("E9: budget frontier")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, run_any, CSV, FIELDS)
    print(f"[{EXP}] jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

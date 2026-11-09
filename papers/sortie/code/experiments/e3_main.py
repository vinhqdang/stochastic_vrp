"""E3: main comparison. det, saa, sortie0 (PBOI, kappa 0), sortie1 (PBOI, kappa 1) and the cost plan.

Grid: n in {15,...,35} x 5 blocks x measures {max,range,gmd,mad,var} x B in {1.05,1.10}.
Time limit 15 s (n <= 25) or 25 s (n >= 30). Every plan is evaluated out of sample on the
same arc scenarios (CRN, seed 2026*1000+n*10+b, S = 20000).
"""

import common as cm
from common import (  # noqa: E402
    E3_NS, MAIN_GRID, MEASURES, ROW_FIELDS, S_OOS, QUICK_INST, make_cost_job, make_job,
    parse_args, print_header, quick_S, quick_time, run_any, run_jobs, time_limit_for,
)

EXP = "e3"
CSV = "e3_main.csv"
BUDGETS = (1.05, 1.10)
METHODS = ("det", "saa", "sortie0", "sortie1")


def jobs_for(args):
    if args.quick:
        insts = [QUICK_INST]
        measures = ("range",)
    else:
        insts = [(n, b) for n, b in MAIN_GRID if n in E3_NS]
        measures = MEASURES
    jobs = []
    for n, b in insts:
        S = quick_S(args, S_OOS)
        jobs.append(make_cost_job(EXP, n, b, S))
        tl = quick_time(args, time_limit_for(n))
        for meas in measures:
            for B in BUDGETS:
                for method in METHODS:
                    jobs.append(make_job(EXP, n, b, meas, B, method, cm.V_DEF, tl, S))
    return jobs


def main():
    args = parse_args("E3: main comparison grid")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, run_any, CSV, ROW_FIELDS)
    print(f"[{EXP}] jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

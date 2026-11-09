"""E7: scale. n in {50, 75, 100} x 3 blocks, measures {range, gmd}, B = 1.10, methods det,
saa, sortie0, time limit 60 s, out-of-sample S = 10000 for every size.
"""

import common as cm
from common import (  # noqa: E402
    LARGE_GRID, ROW_FIELDS, S_OOS_LARGE, make_job, parse_args, print_header,
    quick_S, quick_time, run_any, run_jobs,
)

EXP = "e7"
CSV = "e7_scale.csv"
MEASURES = ("range", "gmd")
METHODS = ("det", "saa", "sortie0")
B = 1.10
TIME_LIMIT = 60.0


def jobs_for(args):
    if args.quick:
        insts = [(50, 0)]
        measures = ("range",)
    else:
        insts = LARGE_GRID
        measures = MEASURES
    jobs = []
    for n, b in insts:
        tl = quick_time(args, TIME_LIMIT)
        S = quick_S(args, S_OOS_LARGE)
        for meas in measures:
            for method in METHODS:
                jobs.append(make_job(EXP, n, b, meas, B, method, cm.V_DEF, tl, S))
    return jobs


def main():
    args = parse_args("E7: scaling to n = 100")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, run_any, CSV, ROW_FIELDS)
    print(f"[{EXP}] jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

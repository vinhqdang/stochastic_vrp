"""E7b: search-noise control at n = 100. Six blocks x measures {range, gmd, max}, B = 1.10,
methods det, saa, sortie0, three solver seeds each (11, 13, 17), time limit 240 s,
out-of-sample S = 10000. Comparisons use the per-instance mean over seeds, which averages
out the search noise that dominates single runs at this size (see E7).
"""

import common as cm
from common import (  # noqa: E402
    ROW_FIELDS, S_OOS_LARGE, make_job, parse_args, print_header, quick_S, quick_time,
    run_any, run_jobs,
)

EXP = "e7b"
CSV = "e7b_seeds.csv"
MEASURES = ("range", "gmd", "max")
METHODS = ("det", "saa", "sortie0")
SEEDS = (11, 13, 17)
B = 1.10
TIME_LIMIT = 240.0


def jobs_for(args):
    insts = [(100, 0)] if args.quick else [(100, b) for b in range(6)]
    measures = ("range",) if args.quick else MEASURES
    seeds = SEEDS[:1] if args.quick else SEEDS
    jobs = []
    for n, b in insts:
        tl = quick_time(args, TIME_LIMIT)
        S = quick_S(args, S_OOS_LARGE)
        for meas in measures:
            for method in METHODS:
                for sd in seeds:
                    job = make_job(EXP, n, b, meas, B, method, cm.V_DEF, tl, S)
                    job["solve_seed"] = sd
                    job["key"] = job["key"] + f"|seed{sd}"
                    jobs.append(job)
    return jobs


def main():
    args = parse_args("E7b: n = 100 with three solver seeds")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, run_any, CSV, ROW_FIELDS)
    print(f"[{EXP}] jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

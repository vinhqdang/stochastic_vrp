"""E5: screening ablation. Price-based screening (K = 8 best predictions) against K random
feasible moves, same time limit, PBOI kappa 0, measures {range, gmd}, B = 1.10,
n in {15,...,35} x 5 blocks.
"""

import common as cm
from common import (  # noqa: E402
    E3_NS, MAIN_GRID, ROW_FIELDS, S_OOS, QUICK_INST, make_job, parse_args, print_header,
    quick_S, quick_time, run_any, run_jobs, time_limit_for,
)

EXP = "e5"
CSV = "e5_ablation.csv"
MEASURES = ("range", "gmd")
METHODS = ("sortie0", "sortie0_random")
B = 1.10


def jobs_for(args):
    if args.quick:
        insts = [QUICK_INST]
        measures = ("range",)
    else:
        insts = [(n, b) for n, b in MAIN_GRID if n in E3_NS]
        measures = MEASURES
    jobs = []
    for n, b in insts:
        tl = quick_time(args, time_limit_for(n))
        S = quick_S(args, S_OOS)
        for meas in measures:
            for method in METHODS:
                jobs.append(make_job(EXP, n, b, meas, B, method, cm.V_DEF, tl, S))
    return jobs


def main():
    args = parse_args("E5: price screening versus random screening")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, run_any, CSV, ROW_FIELDS)
    print(f"[{EXP}] jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

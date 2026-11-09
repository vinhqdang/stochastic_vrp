"""E8: heterogeneous arc noise (congestion). Instances are the e3 blocks with the
congestion multiplier H of instances.with_congestion (seed 0): arc cv v * H_ij, H in
[1/3, 2], large near a random customer. n in {15,...,35} x blocks 0-4, measures
{range, gmd, max}, B = 1.10, methods det, saa, sortie0, sortie1 plus the e0 cost plan.
Start plans are the e0 cost plans of the base instances (the cost plan ignores noise).
Time limits as e3 (15 s n <= 25, 25 s n >= 30); out-of-sample S = 20000, v = 0.3, sY = 0.15.
"""

import common as cm
from common import (  # noqa: E402
    E3_NS, ROW_FIELDS, S_OOS, QUICK_INST, make_cost_job, make_job, parse_args, print_header,
    quick_S, quick_time, run_any, run_jobs, time_limit_for,
)

EXP = "e8"
CSV = "e8_congestion.csv"
BUDGET = 1.10
MEASURES = ("range", "gmd", "max")
METHODS = ("det", "saa", "sortie0", "sortie1")
BLOCKS = range(5)


def jobs_for(args):
    if args.quick:
        insts = [QUICK_INST]
        measures = ("range",)
    else:
        insts = [(n, b) for n in E3_NS for b in BLOCKS]
        measures = MEASURES
    jobs = []
    for n, b in insts:
        S = quick_S(args, S_OOS)
        jobs.append(make_cost_job(EXP, n, b, S, cong=True))
        tl = quick_time(args, time_limit_for(n))
        for meas in measures:
            for method in METHODS:
                jobs.append(make_job(EXP, n, b, meas, BUDGET, method, cm.V_DEF, tl, S, cong=True))
    return jobs


def main():
    args = parse_args("E8: heterogeneous arc noise (congestion) grid")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, run_any, CSV, ROW_FIELDS)
    print(f"[{EXP}] jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E4: route inconsistency. Free re-sequencing (consistency "free") against TSP-lower-level tours.

Measures {range, gmd, max}, B = 1.10, n in {15,...,35} x 5 blocks. Methods: pboi_free,
pboi_lower, det_free, det_lower. Reports non-TSP-optimal route counts, excess length,
in-sample objective and out-of-sample objective (per row). Time limit as in E3.
"""

import common as cm
from common import (  # noqa: E402
    E3_NS, MAIN_GRID, ROW_FIELDS, S_OOS, QUICK_INST, make_job, parse_args, print_header,
    quick_S, quick_time, run_any, run_jobs, time_limit_for,
)

EXP = "e4"
CSV = "e4_inconsistency.csv"
MEASURES = ("range", "gmd", "max")
METHODS = ("pboi_free", "pboi_lower", "det_free", "det_lower")
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
    args = parse_args("E4: free versus lower-level route consistency")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, run_any, CSV, ROW_FIELDS)
    print(f"[{EXP}] jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E16: risk-adjusted deterministic baseline and marginal family. The E3 grid (n in {15,...,35}
x blocks 0-4, all five measures) at B = 1.10 with methods detsig (deterministic ordered measure
of L + z sigma, z = 1) and pboi_normal (PBOI with normal marginals), E3 time limits.
"""

import common as cm
from common import (  # noqa: E402
    E3_NS, EXT_FIELDS, MAIN_GRID, ROW_FIELDS, S_OOS, QUICK_INST, make_job, parse_args,
    print_header, quick_S, quick_time, run_any, run_jobs, time_limit_for,
)

EXP = "e16"
CSV = "e16_baselines.csv"
MEASURES = ("max", "range", "gmd", "mad", "var")
METHODS = ("detsig", "pboi_normal")
B = 1.10
FIELDS = ROW_FIELDS + EXT_FIELDS


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
                if method == "pboi_normal" and meas == "var":
                    continue  # expost_var does not depend on the marginal family
                jobs.append(make_job(EXP, n, b, meas, B, method, cm.V_DEF, tl, S))
    return jobs


def main():
    args = parse_args("E16: risk-adjusted deterministic and normal-marginal baselines")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, run_any, CSV, FIELDS)
    print(f"[{EXP}] jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

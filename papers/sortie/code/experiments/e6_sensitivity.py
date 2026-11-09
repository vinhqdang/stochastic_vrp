"""E6: sensitivity to the stochastic level v. v in {0.1, 0.3, 0.5} used both to optimise and
to evaluate. n = 25, 5 blocks, measures {range, gmd}, B = 1.10, methods det and sortie0.
"""

from common import (  # noqa: E402
    ROW_FIELDS, S_OOS, QUICK_INST, make_job, parse_args, print_header, quick_S, quick_time,
    run_any, run_jobs, time_limit_for,
)

EXP = "e6"
CSV = "e6_sensitivity.csv"
VS = (0.1, 0.3, 0.5)
MEASURES = ("range", "gmd")
METHODS = ("det", "sortie0")
B = 1.10


def jobs_for(args):
    if args.quick:
        insts = [QUICK_INST]
        measures = ("range",)
    else:
        insts = [(25, b) for b in range(5)]
        measures = MEASURES
    jobs = []
    for n, b in insts:
        tl = quick_time(args, time_limit_for(n))
        S = quick_S(args, S_OOS)
        for v in VS:
            for meas in measures:
                for method in METHODS:
                    jobs.append(make_job(EXP, n, b, meas, B, method, v, tl, S))
    return jobs


def main():
    args = parse_args("E6: sensitivity to v")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, run_any, CSV, ROW_FIELDS)
    print(f"[{EXP}] jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

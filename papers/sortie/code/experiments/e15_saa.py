"""E15: SAA sample size. n in {25, 35} x blocks 0-4, measures {range, gmd, max}, B = 1.10,
SAA with S in {50, 200, 1000, 5000} scenarios (scenario seed fixed at SCEN_SEED), E3 time
limits. The sortie0 and det rows of the comparison are joined from E3 at analysis time.
"""

import common as cm
from common import (  # noqa: E402
    EXT_FIELDS, QUICK_INST, ROW_FIELDS, S_OOS, make_xjob, parse_args, print_header, quick_S,
    quick_time, run_any, run_jobs, time_limit_for,
)

EXP = "e15"
CSV = "e15_saa.csv"
NS = (25, 35)
BLOCKS = range(5)
MEASURES = ("range", "gmd", "max")
SAMPLE_SIZES = (50, 200, 1000, 5000)
B = 1.10
FIELDS = ROW_FIELDS + EXT_FIELDS


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
            for sS in SAMPLE_SIZES:
                jobs.append(make_xjob(EXP, label, n, b, meas, B, "saa", cm.V_DEF, tl, S,
                                      saa_S=sS))
    return jobs


def main():
    args = parse_args("E15: SAA sample size")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, run_any, CSV, FIELDS)
    print(f"[{EXP}] jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

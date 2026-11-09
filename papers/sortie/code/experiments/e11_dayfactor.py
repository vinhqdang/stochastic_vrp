"""E11: day factor. Evaluation only. Every E3 plan with B = 1.10 (methods det, saa, sortie0,
sortie1; measures max, range, gmd, mad, var) is re-evaluated with sY in {0, 0.15, 0.3, 0.5}.
The arc draws are the same for every sY (the day factors are drawn first from the chunk
generator and the arc draws follow), so only the day factor changes. Plans are read from the
real results (SORTIE_SOURCE_RESULTS, default papers/sortie/results).
"""

import common as cm
from common import (  # noqa: E402
    EXT_FIELDS, ROW_FIELDS, S_OOS, QUICK_INST, inst_seed, parse_args, print_header, quick_S,
    run_jobs,
)

EXP = "e11"
CSV = "e11_dayfactor.csv"
SYS = (0.0, 0.15, 0.3, 0.5)
B = 1.10
METHODS = ("det", "saa", "sortie0", "sortie1")
FIELDS = ROW_FIELDS + EXT_FIELDS


def source_plans(quick):
    out = []
    for r in cm.source_rows("e3_main.csv"):
        if r["experiment"] != "e3" or r["method"] not in METHODS or r["B"] in ("", "nan"):
            continue
        if abs(float(r["B"]) - B) > 1e-9:
            continue
        if quick and (int(r["n"]), int(r["b"]), r["measure"]) != (QUICK_INST[0], QUICK_INST[1], "range"):
            continue
        out.append(r)
    return out


def src_fields(r):
    return dict(src_method=r["method"], src_measure=r["measure"], src_instance=r["instance"],
                src_plan_file=r["plan_file"], src_time_limit=float(r["time_limit"]),
                src_seed=int(r["seed"]))


def jobs_for(args):
    S = quick_S(args, S_OOS)
    jobs = []
    for r in source_plans(args.quick):
        n, b = int(r["n"]), int(r["b"])
        for sY in SYS:
            key = (f"{EXP}|{r['instance']}|{r['measure']}|{r['method']}|"
                   f"t{float(r['time_limit']):g}|sY{sY:g}|S{S}")
            job = dict(exp=EXP, key=key, n=n, b=b, label=r["instance"], measure=r["measure"],
                       method=r["method"], sY=sY, S=S, eval_seed=inst_seed(n, b))
            job.update(src_fields(r))
            jobs.append(job)
    return jobs


def worker(job):
    inst = cm.load_instance(job["n"], job["b"])
    routes = cm.source_routes(job["src_plan_file"])
    cstar = float(cm.load_seed(cm.base_name(inst.name))["Cstar"])
    spec = cm.METHODS[job["method"]]
    row = cm.evaluate_routes_row(EXP, job["key"], inst, routes, job["measure"], B, job["method"],
                                 cm.V_DEF, spec["kappa"], job["src_time_limit"], job["src_seed"],
                                 job["eval_seed"], job["S"], job["sY"], cstar,
                                 job["src_plan_file"], job["b"])
    for name in ("src_method", "src_measure", "src_instance", "src_plan_file",
                 "src_time_limit", "src_seed"):
        row[name] = job[name]
    row["solve_seed"] = job["src_seed"]
    row["family"] = "X641"
    return [row]


def main():
    args = parse_args("E11: day factor sY re-evaluation of the E3 plans")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, worker, CSV, FIELDS)
    print(f"[{EXP}] jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E12: misspecification. Evaluation only. Every E3 plan with B = 1.10 (methods det, saa,
sortie0, sortie1; five measures; the X641 blocks n in {15,...,35}) is evaluated under the four
arc-noise families gamma, lognormal, incident and regional. Every E8 (congestion) plan with
B = 1.10 is evaluated under gamma and lognormal only: the incident and regional families raise
on the low-cv arcs of the congestion multiplier (1 + (vH)^2 below the family's floor), so they
are not defined there. Evaluation uses the same CRN seed inst_seed(n, b) for every family and
records the gamma PBOI prediction (pboi_pred) of each plan.
"""

import common as cm
from common import (  # noqa: E402
    EXT_FIELDS, ROW_FIELDS, S_OOS, QUICK_INST, inst_seed, parse_args, print_header, quick_S,
    run_jobs,
)

EXP = "e12"
CSV = "e12_misspec.csv"
B = 1.10
METHODS = ("det", "saa", "sortie0", "sortie1")
E3_FAMILIES = ("gamma", "lognormal", "incident", "regional")
E8_FAMILIES = ("gamma", "lognormal")
FIELDS = ROW_FIELDS + EXT_FIELDS + ["source", "cong"]


def source_plans(quick):
    """(source row, cong flag, source label, families) for every B = 1.10 plan of E3 and E8."""
    out = []
    for csv_name, exp_name, cong, fams in (("e3_main.csv", "e3", False, E3_FAMILIES),
                                           ("e8_congestion.csv", "e8", True, E8_FAMILIES)):
        for r in cm.source_rows(csv_name):
            if r["experiment"] != exp_name or r["method"] not in METHODS:
                continue
            if r["B"] in ("", "nan") or abs(float(r["B"]) - B) > 1e-9:
                continue
            if quick and (int(r["n"]), int(r["b"]), r["measure"]) != (QUICK_INST[0], QUICK_INST[1], "range"):
                continue
            out.append((r, cong, exp_name, fams))
    return out


def jobs_for(args):
    S = quick_S(args, S_OOS)
    jobs = []
    for r, cong, source, fams in source_plans(args.quick):
        n, b = int(r["n"]), int(r["b"])
        for fam in fams:
            key = (f"{EXP}|{r['instance']}|{r['measure']}|{r['method']}|"
                   f"t{float(r['time_limit']):g}|{fam}|S{S}|src{source}")
            jobs.append(dict(
                exp=EXP, key=key, n=n, b=b, cong=cong, label=r["instance"],
                measure=r["measure"], method=r["method"], dist=fam, source=source, S=S,
                eval_seed=inst_seed(n, b), src_method=r["method"], src_measure=r["measure"],
                src_instance=r["instance"], src_plan_file=r["plan_file"],
                src_time_limit=float(r["time_limit"]), src_seed=int(r["seed"]),
            ))
    return jobs


def worker(job):
    inst = cm.load_instance(job["n"], job["b"], cong=job["cong"])
    routes = cm.source_routes(job["src_plan_file"])
    cstar = float(cm.load_seed(cm.base_name(inst.name))["Cstar"])
    spec = cm.METHODS[job["method"]]
    row = cm.evaluate_routes_row(EXP, job["key"], inst, routes, job["measure"], B, job["method"],
                                 cm.V_DEF, spec["kappa"], job["src_time_limit"], job["src_seed"],
                                 job["eval_seed"], job["S"], cm.SY, cstar, job["src_plan_file"],
                                 job["b"], dist=job["dist"])
    for name in ("src_method", "src_measure", "src_instance", "src_plan_file", "src_time_limit",
                 "src_seed"):
        row[name] = job[name]
    row["solve_seed"] = job["src_seed"]
    row["family"] = "X641"
    row["source"] = job["source"]
    row["cong"] = job["cong"]
    return [row]


def main():
    args = parse_args("E12: misspecified arc-noise families")
    print_header(EXP, args)
    jobs = jobs_for(args)
    n_err = run_jobs(EXP, jobs, worker, CSV, FIELDS)
    print(f"[{EXP}] jobs={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E14: screening diagnostics.

(a) e14_moves.csv: move_diagnostics (mode pboi, kappa 0, budget 1.10 C*, max_moves 400) on
    the E0 cost plan and on the E3 sortie0 plan (B = 1.10) of n in {25, 35} x blocks 0-4 and
    measures {range, gmd, max}. One row per move. The route indices of move_diagnostics are
    written as route_a and route_b (b is the instance block, as in every other CSV).
(b) e14_ksweep.csv: sortie0 (B = 1.10) with the price-screen width K in {1, 2, 4, 8, 16, 32},
    n in {25, 35} x blocks 0-4, measures {range, gmd}, E3 time limits.
"""

import os

import common as cm
from common import (  # noqa: E402
    EXT_FIELDS, ROW_FIELDS, S_OOS, QUICK_INST, inst_seed, make_xjob, parse_args, print_header,
    quick_S, quick_time, run_any, run_jobs, time_limit_for,
)
from sortie.instances import base_name  # noqa: E402
from sortie.search import move_diagnostics  # noqa: E402

EXP = "e14"
CSV_MOVES = "e14_moves.csv"
CSV_K = "e14_ksweep.csv"
NS = (25, 35)
BLOCKS = range(5)
MEASURES_A = ("range", "gmd", "max")
MEASURES_B = ("range", "gmd")
KS = (1, 2, 4, 8, 16, 32)
B = 1.10
MAX_MOVES = 400
MAX_MOVES_QUICK = 50
PLAN_SOURCES = ("e0", "e3_sortie0")
FIELDS_A = ["key", "experiment", "plan_source", "instance", "n", "b", "m", "measure", "B",
            "method", "kappa", "mode", "budget", "Cstar", "max_moves", "diag_seed", "n_picked",
            "src_plan_file", "kind", "route_a", "route_b", "i", "j", "predicted", "exact",
            "pred_rank", "exact_rank"]
FIELDS_B = ROW_FIELDS + EXT_FIELDS


def e3_sortie0_rows():
    """(instance, measure) -> E3 row of the sortie0 plan at B = 1.10."""
    out = {}
    for r in cm.source_rows("e3_main.csv"):
        if r["experiment"] != "e3" or r["method"] != "sortie0" or r["B"] in ("", "nan"):
            continue
        if abs(float(r["B"]) - B) > 1e-9:
            continue
        out[(r["instance"], r["measure"])] = r
    return out


def diag_jobs(args):
    if args.quick:
        insts = [QUICK_INST]
        measures = ("range",)
    else:
        insts = [(n, b) for n in NS for b in BLOCKS]
        measures = MEASURES_A
    max_moves = MAX_MOVES_QUICK if args.quick else MAX_MOVES
    e3 = e3_sortie0_rows()
    jobs = []
    for n, b in insts:
        label = cm.instance_name(n, b)
        for meas in measures:
            for src in PLAN_SOURCES:
                job = dict(exp=EXP, n=n, b=b, label=label, measure=meas, plan_source=src,
                           max_moves=max_moves, diag_seed=inst_seed(n, b),
                           key=f"e14a|{label}|{src}|{meas}|B{B:.2f}|pboi|k0|mm{max_moves}")
                if src == "e3_sortie0":
                    row = e3.get((label, meas))
                    if row is None:
                        raise RuntimeError(f"no E3 sortie0 plan for {label} {meas} B={B}")
                    job["src_plan_file"] = row["plan_file"]
                    job["method"] = "sortie0"
                else:
                    job["src_plan_file"] = ""
                    job["method"] = "cost"
                jobs.append(job)
    return jobs


def diag_worker(job):
    inst = cm.load_instance(job["n"], job["b"])
    seed_rec = cm.load_seed(base_name(inst.name))
    cstar = float(seed_rec["Cstar"])
    if job["plan_source"] == "e0":
        routes = seed_rec["routes"]
        src_file = f"plans/seed_{inst.name}.json"
    else:
        routes = cm.source_routes(job["src_plan_file"])
        src_file = job["src_plan_file"]
    budget = B * cstar
    df = move_diagnostics(inst, routes, "pboi", job["measure"], cm.V_DEF, 0.0, budget,
                          max_moves=job["max_moves"], seed=job["diag_seed"])
    base = dict(key=job["key"], experiment=EXP, plan_source=job["plan_source"],
                instance=inst.name, n=int(job["n"]), b=int(job["b"]), m=int(inst.m),
                measure=job["measure"], B=B, method=job["method"], kappa=0.0, mode="pboi",
                budget=budget, Cstar=cstar, max_moves=job["max_moves"],
                diag_seed=job["diag_seed"], n_picked=len(df), src_plan_file=src_file)
    if len(df) == 0:
        return [dict(base, kind="none")]
    rows = []
    for rec in df.to_dict("records"):
        row = dict(base)
        row.update(kind=rec["kind"], route_a=int(rec["a"]), route_b=int(rec["b"]),
                   i=int(rec["i"]), j=int(rec["j"]), predicted=float(rec["predicted"]),
                   exact=float(rec["exact"]), pred_rank=int(rec["pred_rank"]),
                   exact_rank=float(rec["exact_rank"]))
        rows.append(row)
    return rows


def ksweep_jobs(args):
    if args.quick:
        insts = [QUICK_INST]
        measures = ("range",)
        ks = KS
    else:
        insts = [(n, b) for n in NS for b in BLOCKS]
        measures = MEASURES_B
        ks = KS
    jobs = []
    for n, b in insts:
        tl = quick_time(args, time_limit_for(n))
        S = quick_S(args, S_OOS)
        label = cm.instance_name(n, b)
        for meas in measures:
            for K in ks:
                jobs.append(make_xjob(EXP, label, n, b, meas, B, "sortie0", cm.V_DEF, tl, S,
                                      K=K))
    return jobs


def main():
    args = parse_args("E14: screening diagnostics and K sweep")
    print_header(EXP, args)
    diag = diag_jobs(args)
    n_err = run_jobs(EXP, diag, diag_worker, CSV_MOVES, FIELDS_A)
    ks = ksweep_jobs(args)
    n_err += run_jobs(EXP, ks, run_any, CSV_K, FIELDS_B)
    print(f"[{EXP}] diagnostics={len(diag)} ksweep={len(ks)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E1: accuracy of the PBOI (gamma) evaluator against arc-level Monte Carlo, and timing.

Plans: the e0 seed plans for n in {15,25,35,50} plus random feasible plans from
construct_initial (each route re-sequenced by the TSP lower level). Monte Carlo
uses X (no day factor) with S samples; per-sample standard errors give z scores.
Timing (serial, after the pool) compares pboi, the SAA adapter (S=200) and MC (S=20000)
on n=25 plans.
"""

import os

import common as cm
from common import (  # noqa: E402
    MAIN_GRID, SY, V_DEF, inst_seed, instance_name, load_instance, load_seed,
    parse_args, print_header, quick_S, route_stats, run_jobs,
)
import csv  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from sortie.pboi import expost_var, lam_vector, ordered, pboi  # noqa: E402
from sortie.search import construct_initial, make_objective  # noqa: E402
from sortie.sim import simulate  # noqa: E402
from sortie.tsp import TourSolver  # noqa: E402

EXP = "e1"
CSV = "e1_accuracy.csv"
TIMING_CSV = "e1_timing.csv"
N_RANDOM = 150
RANDOM_NS = (15, 25, 35, 50)
MEASURES_E1 = ("max", "range", "gmd", "mad", "var")
E1_FIELDS = [
    "key", "experiment", "plan_id", "plan_source", "instance", "n", "b", "m", "measure", "v",
    "S_mc", "mc_seed", "sumL", "pboi_pred", "mc_mean", "mc_se", "rel_err", "z_score",
]
TIMING_FIELDS = ["plan_id", "instance", "n", "b", "method", "repeats", "time_per_eval_s",
                 "ratio_to_pboi"]


def seed_plan_list(quick):
    entries = []
    if quick:
        names = [(15, 0), (15, 1)]
    else:
        names = [(n, b) for n, b in MAIN_GRID if n in (15, 25, 35)]
        names += [(50, b) for b in range(3)]
    for n, b in names:
        rec = load_seed(instance_name(n, b))
        entries.append(dict(plan_id=f"seed_{instance_name(n, b)}", plan_source="seed",
                            n=n, b=b, routes=rec["routes"]))
    return entries


def random_plan_list(quick):
    count = 1 if quick else N_RANDOM
    ns = (15,) if quick else RANDOM_NS
    entries = []
    for i in range(count):
        n = ns[i % len(ns)]
        b = (i // len(ns)) % (5 if n < 50 else 3)
        inst = load_instance(n, b)
        rng = np.random.default_rng(555000 + i)
        start = construct_initial(inst, rng)
        solver = TourSolver(inst.C, 0.0, V_DEF)
        routes = [[int(x) for x in solver.solve(r)[0]] for r in start]
        entries.append(dict(plan_id=f"rand_{i:03d}_{instance_name(n, b)}", plan_source="random",
                            n=n, b=b, routes=routes))
    return entries


def e1_worker(job):
    inst = load_instance(job["n"], job["b"])
    routes = job["routes"]
    C = inst.C
    m = len(routes)
    L, W = route_stats(routes, C)
    sigma = V_DEF * np.sqrt(W)
    S = job["S_mc"]
    seed = job["mc_seed"]
    X, _ = simulate(inst, routes, V_DEF, SY, S, seed)
    rows = []
    for meas in job["measures"]:
        if meas == "var":
            Z = X.var(axis=1)
            pred = expost_var(L, sigma)
        else:
            lam = lam_vector(meas, m)
            Z = ordered(X, lam)
            pred = pboi(L, sigma, lam)
        mc = float(Z.mean())
        se = float(Z.std(ddof=1) / np.sqrt(S))
        rows.append(dict(
            key=job["key"], experiment=EXP, plan_id=job["plan_id"],
            plan_source=job["plan_source"], instance=inst.name, n=job["n"], b=job["b"], m=m,
            measure=meas, v=V_DEF, S_mc=S, mc_seed=seed, sumL=float(L.sum()),
            pboi_pred=float(pred), mc_mean=mc, mc_se=se,
            rel_err=(pred - mc) / mc, z_score=(pred - mc) / se if se > 0 else float("nan"),
        ))
    return rows


def run_timing(plans, args):
    """Serial timing on n=25 plans (or n=15 in quick mode)."""
    tn = 15 if args.quick else 25
    sel = [p for p in plans if p["n"] == tn][:10]
    S_mc = quick_S(args, 20000)
    lam_name = "range"
    out_rows = []
    for p in sel:
        inst = load_instance(p["n"], p["b"])
        routes = p["routes"]
        C = inst.C
        L, W = route_stats(routes, C)
        seqs = routes
        adapter_p = make_objective("pboi", lam_name, V_DEF)
        scen = cm.scen_tensor(C.shape[0], V_DEF, S=cm.SCEN_S)
        adapter_s = make_objective("saa", lam_name, V_DEF, scen=scen, C=C)
        lam = lam_vector(lam_name, len(routes))

        def f_pboi():
            return adapter_p.value(L, W, seqs)

        def f_saa():
            adapter_s._xc.clear()
            return adapter_s.value(L, W, seqs)

        def f_mc():
            X, _ = simulate(inst, routes, V_DEF, SY, S_mc, inst_seed(p["n"], p["b"]))
            return float(np.mean(ordered(X, lam)))

        timings = []
        for name, fn, reps in (("pboi", f_pboi, 20), ("saa_S200", f_saa, 5),
                               (f"mc_S{S_mc}", f_mc, 3)):
            fn()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            timings.append((name, reps, (time.perf_counter() - t0) / reps))
        base = timings[0][2]
        for name, reps, t in timings:
            out_rows.append(dict(plan_id=p["plan_id"], instance=inst.name, n=p["n"], b=p["b"],
                                 method=name, repeats=reps, time_per_eval_s=t,
                                 ratio_to_pboi=t / base))
    path = os.path.join(cm.RESULTS, TIMING_CSV)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TIMING_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    os.replace(tmp, path)
    for name in sorted({r["method"] for r in out_rows}):
        vals = [r["time_per_eval_s"] for r in out_rows if r["method"] == name]
        print(f"[e1] timing {name}: mean {np.mean(vals):.6f} s over {len(vals)} plans", flush=True)


def main():
    args = parse_args("E1: PBOI vs Monte Carlo accuracy and timing")
    print_header(EXP, args)
    cm.ensure_dirs()
    plans = seed_plan_list(args.quick) + random_plan_list(args.quick)
    print(f"[e1] plans: {len(plans)} (seed={sum(p['plan_source'] == 'seed' for p in plans)})",
          flush=True)
    measures = ("range",) if args.quick else MEASURES_E1
    S_mc = quick_S(args, 50000)
    jobs = []
    for p in plans:
        jobs.append(dict(
            exp=EXP, key=f"{EXP}|{p['plan_id']}", plan_id=p["plan_id"],
            plan_source=p["plan_source"], n=p["n"], b=p["b"], routes=p["routes"],
            S_mc=S_mc, mc_seed=20260000 + len(jobs), measures=list(measures),
        ))
    n_err = run_jobs(EXP, jobs, e1_worker, CSV, E1_FIELDS)
    run_timing(plans, args)
    print(f"[{EXP}] done failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

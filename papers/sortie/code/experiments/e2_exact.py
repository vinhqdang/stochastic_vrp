"""E2: exact optimum on tiny instances (n = 10, m = 3) against solve_fair.

All assignments of 10 customers to 3 labelled vehicles are enumerated (3^10). Only
restricted growth strings are kept, which removes the label symmetry (S(10,3) = 9330
set partitions), and every block must be capacity-feasible. Tours come from TourSolver
with kappa = 0 (Held-Karp, memoised by customer set). For range and gmd (PBOI, gamma)
and for the deterministic objective the exact optimum under sum L <= 1.10 * C*_exact is
found by evaluating every budget-feasible partition. solve_fair (5 s, start from the e0
cost plan, budget 1.10 * C*_exact) is compared with the optimum through the gap in %.
"""

import itertools
import time

import common as cm
from common import (  # noqa: E402
    SOLVE_SEED, TINY_M, TINY_N, V_DEF, instance_name, load_instance, load_seed,
    parse_args, print_header, quick_time, run_jobs,
)
import numpy as np  # noqa: E402

from sortie.pboi import lam_vector, ordered, pboi  # noqa: E402
from sortie.search import solve_fair  # noqa: E402
from sortie.tsp import TourSolver  # noqa: E402

EXP = "e2"
CSV = "e2_exact.csv"
B = 1.10
OBJECTIVES = (("pboi", "range"), ("pboi", "gmd"), ("det", "range"), ("det", "gmd"))
E2_FIELDS = [
    "key", "experiment", "instance", "n", "b", "m", "measure", "method", "B", "v", "kappa",
    "time_limit", "seed", "Cstar_heur", "Cstar_exact", "budget", "sumL", "obj_insample",
    "opt_value", "gap_pct", "cost_ratio", "n_canonical", "n_feasible", "n_budget_feasible",
    "enum_time", "runtime", "iters", "n_exact_evals", "routes", "plan_file",
]


def enumerate_partitions(inst):
    """Canonical feasible partitions into m labelled-free blocks with their L, W and tours."""
    C = inst.C
    n = inst.demand.shape[0] - 1
    dem = inst.demand
    Q = int(inst.Q)
    solver = TourSolver(C, 0.0, V_DEF)
    cache = {}

    def block_info(mask):
        if mask in cache:
            return cache[mask]
        nodes = [i + 1 for i in range(n) if (mask >> i) & 1]
        if int(dem[nodes].sum()) > Q:
            res = (False, 0.0, 0.0, None)
        else:
            tour, L, W = solver.solve(nodes)
            res = (True, L, W, tour)
        cache[mask] = res
        return res

    t0 = time.perf_counter()
    parts = []
    n_canon = 0
    for a in itertools.product(range(TINY_M), repeat=n):
        seen = -1
        ok = True
        for x in a:
            if x > seen + 1:
                ok = False
                break
            if x > seen:
                seen = x
        if not ok or seen != TINY_M - 1:
            continue
        n_canon += 1
        masks = [0] * TINY_M
        for i, x in enumerate(a):
            masks[x] |= 1 << i
        infos = [block_info(mk) for mk in masks]
        if not all(f[0] for f in infos):
            continue
        Ls = np.array([f[1] for f in infos])
        Ws = np.array([f[2] for f in infos])
        parts.append(dict(sumL=float(Ls.sum()), L=Ls, W=Ws,
                          routes=[list(f[3]) for f in infos]))
    if n_canon != _stirling2(n, TINY_M):
        raise RuntimeError(f"canonical count {n_canon} differs from S({n},{TINY_M})")
    return parts, n_canon, time.perf_counter() - t0


def _stirling2(n, k):
    table = [[0] * (k + 1) for _ in range(n + 1)]
    table[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            table[i][j] = j * table[i - 1][j] + table[i - 1][j - 1]
    return table[n][k]


def objective_value(mode, measure, L, W):
    lam = lam_vector(measure, len(L))
    if mode == "pboi":
        return float(pboi(L, V_DEF * np.sqrt(W), lam))
    return float(ordered(L, lam))


def exact_optimum(parts, mode, measure, budget):
    best_val = np.inf
    best = None
    n_bud = 0
    for p in parts:
        if p["sumL"] > budget + 1e-9:
            continue
        n_bud += 1
        val = objective_value(mode, measure, p["L"], p["W"])
        if val < best_val:
            best_val = val
            best = p
    return best_val, best, n_bud


def worker(job):
    n, b = TINY_N, job["b"]
    tl = job["time_limit"]
    inst = load_instance(n, b, tiny=True)
    seed_rec = load_seed(inst.name)
    parts, n_canon, t_enum = enumerate_partitions(inst)
    cstar_exact = min(p["sumL"] for p in parts)
    cstar_heur = float(seed_rec["Cstar"])
    budget = B * cstar_exact
    rows = []
    for mode, measure in OBJECTIVES:
        t0 = time.perf_counter()
        opt_val, opt_part, n_bud = exact_optimum(parts, mode, measure, budget)
        t_opt = time.perf_counter() - t0
        base = dict(
            experiment=EXP, instance=inst.name, n=n, b=b, m=TINY_M, measure=measure, B=B,
            v=V_DEF, kappa=0.0, Cstar_heur=cstar_heur, Cstar_exact=cstar_exact, budget=budget,
            n_canonical=n_canon, n_feasible=len(parts), n_budget_feasible=n_bud,
            enum_time=t_enum,
        )
        rows.append(dict(base, key=job["key"], method=f"exact_{mode}", seed="",
                         time_limit="", sumL=opt_part["sumL"], obj_insample="",
                         opt_value=opt_val, gap_pct=0.0,
                         cost_ratio=opt_part["sumL"] / cstar_exact, runtime=t_opt,
                         iters="", n_exact_evals="", plan_file="",
                         routes=opt_part["routes"]))
        res = solve_fair(inst, seed_rec["routes"], mode, measure, V_DEF, 0.0, budget, tl,
                         SOLVE_SEED, screen="price", consistency="lower")
        gap = 100.0 * (res["value"] - opt_val) / opt_val if opt_val > 0 else float("nan")
        rows.append(dict(base, key=job["key"],
                         method=f"solve_{mode}", seed=SOLVE_SEED, time_limit=tl,
                         sumL=res["sumL"], obj_insample=res["value"], opt_value=opt_val,
                         gap_pct=gap, cost_ratio=res["sumL"] / cstar_exact,
                         runtime=res["time"], iters=res["iters"],
                         n_exact_evals=res["n_exact_evals"], plan_file="",
                         routes=res["routes"]))
    return rows


def main():
    args = parse_args("E2: exact enumeration on tiny instances versus solve_fair")
    print_header(EXP, args)
    cm.ensure_dirs()
    blocks = [0] if args.quick else list(range(10))
    tl = quick_time(args, 5.0)
    jobs = [dict(exp=EXP, key=f"{EXP}|{instance_name(TINY_N, b)}", b=b, time_limit=tl)
            for b in blocks]
    n_err = run_jobs(EXP, jobs, worker, CSV, E2_FIELDS)
    print(f"[{EXP}] instances={len(jobs)} failures={n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

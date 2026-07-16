"""experiment.py -- numerical illustration for the MWHED paper.

Compares four algorithms on random instances of Minimum Weighted
Hazard-Exposure Dispatch (MWHED):
  exact    -- the pseudo-polynomial DP of Theorem 2 (exact optimum)
  fptas    -- the value-scaled FPTAS of Theorem 3, at a given epsilon
  repair   -- a weighted Moore-Hodgson-style greedy: dispatch in
              earliest-deadline-first order, and whenever the running
              schedule becomes infeasible, repeatedly drop the
              currently-scheduled site with the smallest weight/time
              ratio until feasible again (a heuristic generalization of
              Moore and Hodgson's unweighted largest-processing-time
              removal rule; feasibility of the returned schedule is
              guaranteed by the same inductive argument as the
              unweighted case, but -- unlike Theorems 2/3 -- no
              optimality or approximation ratio is claimed for it)
  edd      -- naive baseline: dispatch in earliest-deadline-first order,
              count whichever sites happen to still be on time (no
              reconsideration at all -- the "do nothing clever" anchor
              of Proposition 5)

Also times exact-DP vs FPTAS as the processing-time range grows, to
illustrate the pseudo-polynomial-vs-polynomial gap of Theorems 2 and 3
in practice, not just asymptotically.

Self-contained, does not import anything from the BATON/TEMPO codebase.
"""
from __future__ import annotations

import json
import math
import random
import time


def solve_exact(p, d, w):
    """Theorem 2's DP. Sites pre-sorted by increasing deadline. Returns
    (optimal on-time weight, achieving subset as a set of indices)."""
    n = len(p)
    P = sum(p)
    NEG = float("-inf")
    f = [NEG] * (P + 1)
    f[0] = 0.0
    choice = [[False] * (P + 1) for _ in range(n)]
    for i in range(n):
        new_f = list(f)
        for t in range(P, p[i] - 1, -1):
            if f[t - p[i]] > NEG and t <= d[i]:
                cand = f[t - p[i]] + w[i]
                if cand > new_f[t]:
                    new_f[t] = cand
                    choice[i][t] = True
        f = new_f
    best_t = max(range(P + 1), key=lambda t: f[t])
    best_val = f[best_t]
    included = set()
    t = best_t
    for i in range(n - 1, -1, -1):
        if choice[i][t]:
            included.add(i)
            t -= p[i]
    return best_val, included


def solve_fptas(p, d, w, eps):
    """Theorem 3's FPTAS, implemented exactly as in the proof: scale
    weights by K = max(1, eps*max(w)/n), then run the VALUE-INDEXED dual
    DP g(i,v) = min dispatch time to reach scaled value v using a
    feasible subset of sites 1..i (Theorem 3's recursion), whose table
    has size O(n * sum(w')) = O(n^3/eps) -- independent of the
    processing-time sum P, unlike Theorem 2's time-indexed table.
    Reports the TRUE (unscaled) weight of the returned subset."""
    n = len(p)
    wmax = max(w)
    K = max(1.0, eps * wmax / n)
    wprime = [math.floor(wi / K) for wi in w]
    Vmax = sum(wprime)
    INF = float("inf")
    g = [INF] * (Vmax + 1)
    g[0] = 0
    choice = [[False] * (Vmax + 1) for _ in range(n)]
    for i in range(n):
        new_g = list(g)
        wi = wprime[i]
        for v in range(Vmax, wi - 1, -1):
            if g[v - wi] < INF:
                cand = g[v - wi] + p[i]
                if cand <= d[i] and cand < new_g[v]:
                    new_g[v] = cand
                    choice[i][v] = True
        g = new_g
    best_v = max((v for v in range(Vmax + 1) if g[v] < INF), default=0)
    included = set()
    v = best_v
    for i in range(n - 1, -1, -1):
        if choice[i][v]:
            included.add(i)
            v -= wprime[i]
    true_val = sum(w[i] for i in included)
    return true_val, included


def solve_edd_naive(p, d, w):
    """Baseline: dispatch in EDD order, no optimization -- just see who
    happens to still be on time under the resulting schedule."""
    n = len(p)
    c = 0
    val = 0.0
    included = set()
    for i in range(n):
        c += p[i]
        if c <= d[i]:
            val += w[i]
            included.add(i)
    return val, included


def solve_greedy_repair(p, d, w):
    """Weighted Moore-Hodgson-style greedy repair (see module docstring).
    Sites pre-sorted by increasing deadline, as required for the
    running-total feasibility argument to apply."""
    kept = []  # indices, in EDD order
    total = 0
    for i in range(len(p)):
        kept.append(i)
        total += p[i]
        while total > d[i] and kept:
            worst = min(kept, key=lambda j: w[j] / p[j])
            kept.remove(worst)
            total -= p[worst]
    val = sum(w[i] for i in kept)
    return val, set(kept)


def random_instance(n, rng, p_max=20, w_max=100, horizon_mult=0.6):
    p = [rng.randint(1, p_max) for _ in range(n)]
    w = [rng.randint(1, w_max) for _ in range(n)]
    total_p = sum(p)
    d_raw = [rng.randint(1, total_p) for _ in range(n)]
    idx = sorted(range(n), key=lambda i: d_raw[i])
    p2 = [p[i] for i in idx]
    w2 = [w[i] for i in idx]
    d2 = [d_raw[i] for i in idx]
    return p2, d2, w2


def random_instance_heavytail(n, rng, p_max=20):
    """Like random_instance, but weights are drawn from a heavy-tailed
    (exponential) distribution instead of uniform, to check that the
    accuracy comparison is not an artifact of uniform weights."""
    p = [rng.randint(1, p_max) for _ in range(n)]
    w = [max(1, int(rng.expovariate(1 / 20))) for _ in range(n)]
    total_p = sum(p)
    d_raw = [rng.randint(1, total_p) for _ in range(n)]
    idx = sorted(range(n), key=lambda i: d_raw[i])
    p2 = [p[i] for i in idx]
    w2 = [w[i] for i in idx]
    d2 = [d_raw[i] for i in idx]
    return p2, d2, w2


def random_instance_tight_deadlines(n, rng, p_max=20, w_max=100, horizon_frac=0.5):
    """Like random_instance, but deadlines are drawn from a narrower
    horizon (horizon_frac * sum(p)) instead of the full [1, sum(p)]
    range, producing more tightly constrained (harder) instances."""
    p = [rng.randint(1, p_max) for _ in range(n)]
    w = [rng.randint(1, w_max) for _ in range(n)]
    total_p = sum(p)
    horizon = max(1, int(horizon_frac * total_p))
    d_raw = [rng.randint(1, horizon) for _ in range(n)]
    idx = sorted(range(n), key=lambda i: d_raw[i])
    p2 = [p[i] for i in idx]
    w2 = [w[i] for i in idx]
    d2 = [d_raw[i] for i in idx]
    return p2, d2, w2


def run_robustness_check():
    """Repeats the accuracy comparison under two instance-generation
    regimes that depart from the base uniform/loose-horizon generator,
    to check the accuracy table's qualitative pattern is not an
    artifact of that specific choice."""
    rng = random.Random(20260618)
    n = 30
    trials = 20
    regimes = [
        ("heavy-tailed weights", lambda: random_instance_heavytail(n, rng)),
        ("tight deadlines (0.5x horizon)", lambda: random_instance_tight_deadlines(n, rng)),
    ]
    rows = []
    for label, gen in regimes:
        ratios = {"fptas": [], "repair": [], "naive": []}
        for _ in range(trials):
            p, d, w = gen()
            opt, _ = solve_exact(p, d, w)
            f, _ = solve_fptas(p, d, w, 0.1)
            rep, _ = solve_greedy_repair(p, d, w)
            nv, _ = solve_edd_naive(p, d, w)
            if opt > 0:
                ratios["fptas"].append(f / opt)
                ratios["repair"].append(rep / opt)
                ratios["naive"].append(nv / opt)
        row = dict(regime=label,
                   fptas_ratio=sum(ratios["fptas"]) / len(ratios["fptas"]),
                   repair_ratio=sum(ratios["repair"]) / len(ratios["repair"]),
                   naive_ratio=sum(ratios["naive"]) / len(ratios["naive"]))
        rows.append(row)
        print(f"{label:32s}  FPTAS/OPT={row['fptas_ratio']:.4f}  "
             f"Repair/OPT={row['repair_ratio']:.4f}  "
             f"Naive/OPT={row['naive_ratio']:.4f}")
    return rows


def run_accuracy_illustration():
    rng = random.Random(20260615)
    rows = []
    for n in (10, 15, 20, 25, 30, 50, 75, 100):
        for trial in range(20):
            p, d, w = random_instance(n, rng)
            opt, _ = solve_exact(p, d, w)
            f2, _ = solve_fptas(p, d, w, 0.2)
            f1, _ = solve_fptas(p, d, w, 0.1)
            rep, _ = solve_greedy_repair(p, d, w)
            naive, _ = solve_edd_naive(p, d, w)
            rows.append(dict(n=n, trial=trial, opt=opt,
                             fptas_eps02=f2, fptas_eps01=f1,
                             repair=rep, naive=naive))

    summary = {}
    for n in (10, 15, 20, 25, 30, 50, 75, 100):
        sub = [r for r in rows if r["n"] == n]
        opt_mean = sum(r["opt"] for r in sub) / len(sub)
        ratio02 = sum(r["fptas_eps02"] / r["opt"] for r in sub if r["opt"] > 0) / len(sub)
        ratio01 = sum(r["fptas_eps01"] / r["opt"] for r in sub if r["opt"] > 0) / len(sub)
        ratio_repair = sum(r["repair"] / r["opt"] for r in sub if r["opt"] > 0) / len(sub)
        ratio_naive = sum(r["naive"] / r["opt"] for r in sub if r["opt"] > 0) / len(sub)
        summary[n] = dict(opt_mean=opt_mean, fptas02_ratio=ratio02,
                          fptas01_ratio=ratio01, repair_ratio=ratio_repair,
                          naive_ratio=ratio_naive)
        print(f"n={n:3d}  opt={opt_mean:8.1f}  "
             f"FPTAS(eps=0.2)/OPT={ratio02:.3f}  "
             f"FPTAS(eps=0.1)/OPT={ratio01:.3f}  "
             f"Repair/OPT={ratio_repair:.3f}  "
             f"EDD-naive/OPT={ratio_naive:.3f}")
    return summary


def run_runtime_comparison():
    """Times exact DP vs FPTAS as the processing-time range p_max grows,
    at fixed n, to show the pseudo-polynomial (Theorem 2) vs polynomial
    (Theorem 3) gap in wall-clock terms, not just in the exponent of the
    complexity bound."""
    rng = random.Random(20260616)
    n = 40
    rows = []
    for p_max in (50, 200, 800, 3200, 12800):
        times_exact = []
        times_fptas = []
        for trial in range(5):
            p, d, w = random_instance(n, rng, p_max=p_max)
            t0 = time.perf_counter()
            solve_exact(p, d, w)
            t1 = time.perf_counter()
            solve_fptas(p, d, w, 0.1)
            t2 = time.perf_counter()
            times_exact.append(t1 - t0)
            times_fptas.append(t2 - t1)
        row = dict(p_max=p_max,
                   exact_ms=1000 * sum(times_exact) / len(times_exact),
                   fptas_ms=1000 * sum(times_fptas) / len(times_fptas))
        rows.append(row)
        print(f"p_max={p_max:6d}  exact={row['exact_ms']:9.2f} ms  "
             f"FPTAS(eps=0.1)={row['fptas_ms']:7.2f} ms")
    return rows


def run_epsilon_sensitivity():
    """Sweeps epsilon at a fixed instance size to show how the FPTAS's
    accuracy ratio and table-size (hence runtime) scale as epsilon
    shrinks, complementing the fixed-epsilon accuracy table."""
    rng = random.Random(20260617)
    n = 30
    eps_values = (0.5, 0.3, 0.2, 0.1, 0.05, 0.01)
    trials = 20
    instances = [random_instance(n, rng) for _ in range(trials)]
    rows = []
    for eps in eps_values:
        ratios = []
        times_ms = []
        for (p, d, w) in instances:
            opt, _ = solve_exact(p, d, w)
            t0 = time.perf_counter()
            val, _ = solve_fptas(p, d, w, eps)
            t1 = time.perf_counter()
            if opt > 0:
                ratios.append(val / opt)
            times_ms.append(1000 * (t1 - t0))
        row = dict(eps=eps,
                   mean_ratio=sum(ratios) / len(ratios),
                   mean_ms=sum(times_ms) / len(times_ms))
        rows.append(row)
        print(f"eps={eps:5.2f}  mean_ratio={row['mean_ratio']:.4f}  "
             f"mean_time={row['mean_ms']:7.2f} ms")
    return rows


def main():
    print("=== Accuracy illustration (Table 1) ===")
    summary = run_accuracy_illustration()
    print()
    print("=== Runtime comparison, exact DP vs FPTAS (Table 2) ===")
    runtime_rows = run_runtime_comparison()
    print()
    print("=== Epsilon sensitivity, FPTAS (Table 3) ===")
    eps_rows = run_epsilon_sensitivity()
    print()
    print("=== Robustness to instance-generation regime (Table 4) ===")
    robustness_rows = run_robustness_check()

    with open("results_illustration.json", "w") as f:
        json.dump(dict(accuracy=summary, runtime=runtime_rows,
                       epsilon_sensitivity=eps_rows,
                       robustness=robustness_rows), f, indent=2)


if __name__ == "__main__":
    main()

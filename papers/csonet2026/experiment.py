"""experiment.py -- small numerical illustration for the MWHED paper.

Compares three algorithms on random instances of Minimum Weighted
Hazard-Exposure Dispatch (MWHED):
  exact   -- the pseudo-polynomial DP of Theorem 2 (exact optimum)
  fptas   -- the value-scaled FPTAS of Theorem 3, at a given epsilon
  edd     -- naive baseline: dispatch in earliest-deadline-first order,
             count whichever sites happen to still be on time (no
             optimization at all -- the "do nothing clever" anchor)

Self-contained, does not import anything from the BATON/TEMPO codebase.
"""
from __future__ import annotations

import json
import random


def solve_exact(p, d, w):
    """Theorem 2's DP. Sites pre-sorted by increasing deadline. Returns
    (optimal on-time weight, achieving subset as a list of indices)."""
    n = len(p)
    P = sum(p)
    NEG = float("-inf")
    # f[t] = max on-time weight using a subset of sites processed so far
    # with total processing time exactly t; choice[i][t] records whether
    # site i was included to reach f[t] at step i.
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
    # backtrack
    included = set()
    t = best_t
    for i in range(n - 1, -1, -1):
        if choice[i][t]:
            included.add(i)
            t -= p[i]
    return best_val, included


def solve_fptas(p, d, w, eps):
    """Theorem 3's FPTAS: scale weights by K = eps*max(w)/n, run the
    exact DP (Theorem 2) on the scaled weights (this keeps the DP's
    table size polynomial in n/eps instead of pseudo-polynomial in
    sum(w)), then report the TRUE (unscaled) weight of the returned
    subset."""
    n = len(p)
    wmax = max(w)
    K = max(1.0, eps * wmax / n)
    w_scaled = [max(1, int(wi / K)) for wi in w]
    _, included = solve_exact(p, d, w_scaled)
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


def random_instance(n, rng, p_max=20, w_max=100, horizon_mult=0.6):
    p = [rng.randint(1, p_max) for _ in range(n)]
    w = [rng.randint(1, w_max) for _ in range(n)]
    total_p = sum(p)
    # deadlines spread over [0, total_p], correlated loosely with index
    d_raw = [rng.randint(1, total_p) for _ in range(n)]
    idx = sorted(range(n), key=lambda i: d_raw[i])
    p2 = [p[i] for i in idx]
    w2 = [w[i] for i in idx]
    d2 = [d_raw[i] for i in idx]
    return p2, d2, w2


def main():
    rng = random.Random(20260615)
    rows = []
    for n in (10, 15, 20, 25, 30):
        for trial in range(20):
            p, d, w = random_instance(n, rng)
            opt, _ = solve_exact(p, d, w)
            f2, _ = solve_fptas(p, d, w, 0.2)
            f1, _ = solve_fptas(p, d, w, 0.1)
            naive, _ = solve_edd_naive(p, d, w)
            rows.append(dict(n=n, trial=trial, opt=opt,
                             fptas_eps02=f2, fptas_eps01=f1, naive=naive))

    summary = {}
    for n in (10, 15, 20, 25, 30):
        sub = [r for r in rows if r["n"] == n]
        opt_mean = sum(r["opt"] for r in sub) / len(sub)
        ratio02 = sum(r["fptas_eps02"] / r["opt"] for r in sub if r["opt"] > 0) / len(sub)
        ratio01 = sum(r["fptas_eps01"] / r["opt"] for r in sub if r["opt"] > 0) / len(sub)
        ratio_naive = sum(r["naive"] / r["opt"] for r in sub if r["opt"] > 0) / len(sub)
        summary[n] = dict(opt_mean=opt_mean, fptas02_ratio=ratio02,
                          fptas01_ratio=ratio01, naive_ratio=ratio_naive)
        print(f"n={n:3d}  opt={opt_mean:7.1f}  "
             f"FPTAS(eps=0.2)/OPT={ratio02:.3f}  "
             f"FPTAS(eps=0.1)/OPT={ratio01:.3f}  "
             f"EDD-naive/OPT={ratio_naive:.3f}")

    with open("results_illustration.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

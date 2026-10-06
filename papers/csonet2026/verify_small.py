"""verify_small.py -- independent correctness checks for every algorithm
and structural claim in the MWHED paper, against exhaustive search.

  A  (n <= 7)  the DP optimum equals the optimum over ALL n! dispatch
               orders (checks Lemma 2 and Theorem 5 from the definition,
               without using the earliest-deadline reduction).
  B  (n <= 14) DP == best feasible subset (EDD-checked); the FPTAS
               returns >= (1-eps) * OPT and a feasible set, with and
               without the completion step, including instances whose
               weights are large enough that scaling is active and
               instances containing individually infeasible sites.
  C  (n <= 8)  equal-cost greedy == DP (m = 1) and == exhaustive search
               over assignments to m = 2, 3 identical vehicles (Thm 8).
  D            the greedy-repair heuristic's worst-case family
               (Proposition 7) has ratio 2/(2k-1).

Run: python3 verify_small.py   (writes verify_small_results.txt)
"""
import itertools
import random

from experiment import (edd_feasible, preprocess, solve_equal_cost,
                        solve_exact, solve_fptas, solve_greedy_repair)

LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


def best_over_permutations(p, d, w):
    n = len(p)
    best = 0
    for perm in itertools.permutations(range(n)):
        c = 0
        val = 0
        for i in perm:
            c += p[i]
            if c <= d[i]:
                val += w[i]
        best = max(best, val)
    return best


def best_over_subsets(p, d, w):
    n = len(p)
    best = 0
    for mask in range(1 << n):
        S = [i for i in range(n) if mask >> i & 1]
        if edd_feasible(S, p, d):
            best = max(best, sum(w[i] for i in S))
    return best


def check_A(rng, trials=400):
    bad = 0
    for _ in range(trials):
        n = rng.randint(1, 7)
        p = [rng.randint(1, 6) for _ in range(n)]
        d = [rng.randint(1, 20) for _ in range(n)]
        w = [rng.randint(1, 30) for _ in range(n)]
        pp, dd, ww, _ = preprocess(p, d, w)
        # permutations over the ORIGINAL instance (infeasible sites included)
        if best_over_permutations(p, d, w) != solve_exact(pp, dd, ww)[0]:
            bad += 1
    log(f"A  permutations vs DP:                 {trials - bad}/{trials} agree")
    return bad


def check_B(rng, trials=1500):
    bad_dp = bad_f = bad_c = 0
    scaled = 0
    for t in range(trials):
        n = rng.randint(1, 14)
        big = t % 3 == 0
        p = [rng.randint(1, 8) for _ in range(n)]
        d = [rng.randint(1, 4 * sum(p)) for _ in range(n)]
        w = [rng.randint(1, 10 ** 6 if big else 40) for _ in range(n)]
        pp, dd, ww, _ = preprocess(p, d, w)
        opt, _ = solve_exact(pp, dd, ww)
        if opt != best_over_subsets(pp, dd, ww):
            bad_dp += 1
        for eps in (0.5, 0.2, 0.05):
            info = {}
            v, S = solve_fptas(pp, dd, ww, eps, info=info)
            scaled += info.get("scaling_active", False) if pp else 0
            vc, Sc = solve_fptas(pp, dd, ww, eps, complete=True)
            if pp and (v < (1 - eps) * opt - 1e-9 or not edd_feasible(S, pp, dd)):
                bad_f += 1
            if pp and (vc < v - 1e-9 or not edd_feasible(Sc, pp, dd)):
                bad_c += 1
    log(f"B  DP vs best feasible subset:         {trials - bad_dp}/{trials} agree")
    log(f"B  FPTAS >= (1-eps)OPT and feasible:   violations = {bad_f} "
        f"({scaled} runs had active scaling)")
    log(f"B  completion step never hurts:        violations = {bad_c}")
    return bad_dp + bad_f + bad_c


def best_assignment(d, w, p, m):
    n = len(d)
    best = 0
    for mask in range(1 << n):
        ch = [i for i in range(n) if mask >> i & 1]
        ok = False
        for assign in itertools.product(range(m), repeat=len(ch)):
            good = True
            for v in range(m):
                ds = sorted(d[i] for i, a in zip(ch, assign) if a == v)
                if any(j * p > di for j, di in enumerate(ds, 1)):
                    good = False
                    break
            if good:
                ok = True
                break
        if ok:
            best = max(best, sum(w[i] for i in ch))
    return best


def check_C(rng, trials=300):
    bad1 = bad_m = 0
    for _ in range(trials):
        n = rng.randint(1, 8)
        p0 = rng.randint(1, 4)
        d = [rng.randint(p0, 6 * p0) for _ in range(n)]
        w = [rng.randint(1, 20) for _ in range(n)]
        pp, dd, ww, _ = preprocess([p0] * n, d, w)
        g1, _ = solve_equal_cost(dd, ww, p0, m=1)
        if g1 != solve_exact(pp, dd, ww)[0]:
            bad1 += 1
        for m in (2, 3):
            if n <= 7 and solve_equal_cost(dd, ww, p0, m=m)[0] != \
                    best_assignment(dd, ww, p0, m):
                bad_m += 1
    log(f"C  equal-cost greedy vs DP (m=1):      {trials - bad1}/{trials} agree")
    log(f"C  equal-cost greedy vs exhaustive m=2,3: mismatches = {bad_m}")
    return bad1 + bad_m


def check_D():
    ok = True
    for k in (3, 10, 100):
        p, d, w = [1, k], [1, k], [2, 2 * k - 1]
        opt, _ = solve_exact(p, d, w)
        g, _ = solve_greedy_repair(p, d, w)
        ratio = g / opt
        ok &= abs(ratio - 2 / (2 * k - 1)) < 1e-12
        log(f"D  greedy repair on family, k={k:3d}: ratio {ratio:.5f} "
            f"= 2/(2k-1) = {2 / (2 * k - 1):.5f}")
    return 0 if ok else 1


if __name__ == "__main__":
    rng = random.Random(20260701)
    fails = check_A(rng) + check_B(rng) + check_C(rng) + check_D()
    log("ALL CHECKS PASSED" if fails == 0 else f"FAILURES: {fails}")
    with open("verify_small_results.txt", "w") as f:
        f.write("\n".join(LOG) + "\n")

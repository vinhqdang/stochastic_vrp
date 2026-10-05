"""H2 part 2: chained route on a LINE with deadlines only (arrival reading).

  L1  interval DP (Pareto in (time, weight)) == exhaustive search          [validation]
  L2  equal weights: O(n^3) DP over (l, r, side, k) with value = min time    [validation]
  L3  weighted line problem is NP-hard: reduction from PARTITION, checked by
      (a) the interval DP and (b) an independent Held-Karp Pareto DP on the
      reduction instances, against the exact subset-sum answer.
  L4  one ray (all sites on one side): optimum = sum of individually feasible weights.
"""
import itertools
import random
from h2_common import *


# ---------------------------------------------------------------- interval DP
def line_split(pos):
    """pos[1..n] positions (pos[0] = 0 depot).  returns left/right lists of
    (distance, site index) sorted by distance."""
    n = len(pos) - 1
    left = sorted((-pos[i], i) for i in range(1, n + 1) if pos[i] < 0)
    right = sorted((pos[i], i) for i in range(1, n + 1) if pos[i] > 0)
    zero = [i for i in range(1, n + 1) if pos[i] == 0]
    assert not zero, "sites at the depot position excluded"
    return left, right


def pareto_insert(lst, t, wt):
    """keep Pareto set sorted by time asc with strictly increasing weight"""
    lst.append((t, wt))


def pareto_prune(lst):
    lst.sort(key=lambda x: (x[0], -x[1]))
    out = []
    bw = -1
    for t, wt in lst:
        if wt > bw:
            out.append((t, wt)); bw = wt
    return out


def line_weighted_dp(pos, h, w):
    """Exact optimum, pseudo-polynomial (Pareto front over (time, weight)).
    State (i, j, side): first i left / first j right sites covered, crew at the
    outermost covered site on `side` (side=0 left, 1 right; at the depot if i=j=0)."""
    left, right = line_split(pos)
    nl, nr = len(left), len(right)
    F = {}                                   # (i,j,side) -> pareto list
    F[(0, 0, 0)] = [(0, 0)]
    best = 0
    stats = 0
    # process in order of i+j
    for tot in range(0, nl + nr + 1):
        for i in range(0, min(tot, nl) + 1):
            j = tot - i
            if j > nr:
                continue
            for side in (0, 1):
                key = (i, j, side)
                if key not in F:
                    continue
                pl = pareto_prune(F[key])
                F[key] = pl
                stats = max(stats, len(pl))
                best = max(best, pl[-1][1])
                # current position
                if i == 0 and j == 0:
                    cur = 0
                elif side == 0:
                    cur = -left[i - 1][0]
                else:
                    cur = right[j - 1][0]
                if i < nl:
                    d, s = left[i]
                    for (t, wt) in pl:
                        t2 = t + abs(cur - (-d))
                        F.setdefault((i + 1, j, 0), []).append(
                            (t2, wt + (w[s] if t2 <= h[s] else 0)))
                if j < nr:
                    d, s = right[j]
                    for (t, wt) in pl:
                        t2 = t + abs(cur - d)
                        F.setdefault((i, j + 1, 1), []).append(
                            (t2, wt + (w[s] if t2 <= h[s] else 0)))
    return best, stats


def line_count_dp(pos, h):
    """Equal weights: dp[(i,j,side)][k] = MIN time to have covered the first i left and
    first j right sites with exactly k of them on time.  O(n^3) states x O(1)."""
    left, right = line_split(pos)
    nl, nr = len(left), len(right)
    INF = float("inf")
    dp = {}
    dp[(0, 0, 0)] = {0: 0}
    best = 0
    for tot in range(0, nl + nr + 1):
        for i in range(0, min(tot, nl) + 1):
            j = tot - i
            if j > nr:
                continue
            for side in (0, 1):
                key = (i, j, side)
                if key not in dp:
                    continue
                cur = 0 if (i == 0 and j == 0) else (-left[i - 1][0] if side == 0 else right[j - 1][0])
                for k, t in dp[key].items():
                    best = max(best, k)
                    if i < nl:
                        d, s = left[i]
                        t2 = t + abs(cur + d)
                        k2 = k + (1 if t2 <= h[s] else 0)
                        dd = dp.setdefault((i + 1, j, 0), {})
                        if t2 < dd.get(k2, INF):
                            dd[k2] = t2
                    if j < nr:
                        d, s = right[j]
                        t2 = t + abs(cur - d)
                        k2 = k + (1 if t2 <= h[s] else 0)
                        dd = dp.setdefault((i, j + 1, 1), {})
                        if t2 < dd.get(k2, INF):
                            dd[k2] = t2
    return best


# ---------------------------------------------------------------- validation
def validate_dp(seed=21, trials=1500):
    rng = random.Random(seed)
    n_ok = 0
    max_front = 0
    for tr in range(trials):
        n = rng.randint(1, 7)
        used = set()
        pos = [0]
        while len(pos) < n + 1:
            x = rng.randint(-9, 9)
            if x != 0 and x not in used:
                used.add(x); pos.append(x)
        # allow duplicates sometimes? (kept distinct positions)
        h = [0] + [rng.randint(abs(pos[i]), 3 * 9 + 6) if rng.random() < 0.8
                   else rng.randint(0, 3 * 9 + 6) for i in range(1, n + 1)]
        eq = rng.random() < 0.4
        w = rand_weights(n, rng, equal=eq, wmax=12)
        D = line_metric(pos)
        ref = chain_opt(D, h, w)
        ref2 = chain_opt_ordered_subsets(D, h, w)
        got, front = line_weighted_dp(pos, h, w)
        assert ref == ref2 == got, (pos, h, w, ref, got)
        if eq:
            assert line_count_dp(pos, h) == ref, ("count dp", pos, h)
        max_front = max(max_front, front)
        n_ok += 1
    return n_ok, max_front


def validate_one_ray(seed=22, trials=500):
    rng = random.Random(seed)
    for tr in range(trials):
        n = rng.randint(1, 7)
        xs = rng.sample(range(1, 30), n)
        pos = [0] + xs
        h = [0] + [rng.randint(0, 35) for _ in range(n)]
        w = rand_weights(n, rng, wmax=15)
        D = line_metric(pos)
        ref = chain_opt(D, h, w)
        formula = sum(w[i] for i in range(1, n + 1) if pos[i] <= h[i])
        assert ref == formula
    return trials


# ---------------------------------------------------------------- reduction
def partition_to_line(b):
    """Skeleton sites S_1..S_K (K=n+1) alternating right/left at scales X_k,
    option site O_j at X_j + b_j beyond S_j (j=1..n), same side.
    weights: skeleton M=A+1, option b_j.  Returns pos,h,w,K,M,A and the target."""
    n = len(b)
    A = sum(b)
    assert A % 2 == 0
    K = n + 1
    M = A + 1
    X = []
    for k in range(K):
        X.append(4 * A + 4 if k == 0 else 4 * sum(X) + 4 * A + 4)
    tau = []
    for k in range(K):
        tau.append(2 * sum(X[:k]) + X[k])
    pos = [0]; h = [0]; w = [0]
    for k in range(K):
        sgn = 1 if k % 2 == 0 else -1
        pos.append(sgn * X[k]); h.append(tau[k] + A); w.append(M)
    for j in range(n):
        sgn = 1 if j % 2 == 0 else -1
        pos.append(sgn * (X[j] + b[j])); h.append(tau[j] + b[j] + A); w.append(b[j])
    return pos, h, w, K, M, A


def subset_sums_leq(b, cap):
    best = 0
    sums = {0}
    for x in b:
        sums |= {s + x for s in sums}
    return max(s for s in sums if s <= cap)


def validate_reduction(seed=23, trials=60):
    rng = random.Random(seed)
    yes = no = 0
    for tr in range(trials):
        n = rng.randint(3, 6)
        while True:
            b = [rng.randint(1, 12) for _ in range(n)]
            if sum(b) % 2 == 0:
                break
        pos, h, w, K, M, A = partition_to_line(b)
        best, front = line_weighted_dp(pos, h, w)
        expect = K * M + subset_sums_leq(b, A // 2)
        assert best == expect, (b, best, expect)
        is_yes = subset_sums_leq(b, A // 2) == A // 2
        assert (best == K * M + A // 2) == is_yes
        yes += is_yes; no += (not is_yes)
        # independent exact solver (Held-Karp Pareto over (mask,last)) on small ones
        if len(pos) - 1 <= 11:
            D = line_metric(pos)
            ref = chain_pareto_dp(D, h, w)
            assert ref == best, ("heldkarp", b, ref, best)
    return yes, no


if __name__ == "__main__":
    print("L1/L2 interval DP + count DP vs exhaustive  (instances, max Pareto size):",
          validate_dp())
    print("L4 one-ray formula checks:", validate_one_ray())
    print("L3 reduction (yes, no) instances all consistent:", validate_reduction())


# ------------------------------------------------ front-induced deadlines version
def partition_to_line_front(b):
    """Same sites as partition_to_line, but EVERY deadline is h(x) = (3/2)|x| - (A+2),
    i.e. a radial hazard front centred at the depot, speed v = 2/3 of the crew's, and a
    dispatch delay Delta = (2A+4)/3 (time units of crew distance).  Uses
    tau_k = (3/2) X_k - 2A - 2 which holds for X_k = 4 sum_{j<k} X_j + 4A + 4."""
    pos, h, w, K, M, A = partition_to_line(b)
    n = len(b)
    for i in range(1, len(pos)):
        hv = Fraction(3, 2) * abs(pos[i]) - (A + 2)
        h[i] = hv
    return pos, h, w, K, M, A


from fractions import Fraction


def validate_reduction_front(seed=24, trials=80):
    rng = random.Random(seed)
    yes = no = 0
    for tr in range(trials):
        n = rng.randint(3, 6)
        while True:
            b = [rng.randint(1, 12) for _ in range(n)]
            if sum(b) % 2 == 0:
                break
        pos, h, w, K, M, A = partition_to_line_front(b)
        # check S_k deadlines coincide with tau_k + A
        pos0, h0, *_ = partition_to_line(b)
        for k in range(1, K + 1):
            assert h[k] == h0[k], (k, h[k], h0[k])
        for j in range(K + 1, len(pos)):
            assert h[j] >= h0[j]
        # integer scaling by 2 so the DP stays in integers
        h2 = [int(2 * x) for x in h]; pos2 = [2 * x for x in pos]
        best, front = line_weighted_dp(pos2, h2, w)
        expect = K * M + subset_sums_leq(b, A // 2)
        assert best == expect, (b, best, expect)
        is_yes = subset_sums_leq(b, A // 2) == A // 2
        yes += is_yes; no += (not is_yes)
        if len(pos) - 1 <= 11:
            D = line_metric(pos2)
            assert chain_pareto_dp(D, h2, w) == best
    return yes, no


if __name__ == "__main__":
    print("L3' reduction with front-induced deadlines h=(3/2)|x|-(A+2):", validate_reduction_front())

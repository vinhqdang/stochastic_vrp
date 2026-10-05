"""Shared library for the H1 (proportional-deadline MWHED) verification.

Model: site i has dispatch time p_i, weight w_i, deadline d_i = rho * p_i,
rho = u/v a positive rational.  A set S is feasible iff, served in
non-decreasing p (= non-decreasing d), every site completes by its deadline,
i.e.  P_{k-1} <= mu * p_k  for every k in S, mu = rho - 1.
All arithmetic is exact (integers / Fractions).
"""
from fractions import Fraction
from itertools import permutations
import math


def frac(x):
    return x if isinstance(x, Fraction) else Fraction(x)


# --------------------------------------------------------------------------
# feasibility tests
# --------------------------------------------------------------------------
def feasible_by_definition(S, p, d):
    """EDD order on S (ties by index), literal completion-time test."""
    order = sorted(S, key=lambda i: (d[i], i))
    t = 0
    for i in order:
        t += p[i]
        if t > d[i]:
            return False
    return True


def feasible_prefix(S, p, rho):
    """Claimed closed form: sorted by p, P_{k-1} <= (rho-1) p_k."""
    mu = frac(rho) - 1
    order = sorted(S, key=lambda i: (p[i], i))
    P = 0
    for i in order:
        if P > mu * p[i]:
            return False
        P += p[i]
    return True


def best_over_permutations(p, w, d):
    """max total on-time weight over ALL n! dispatch orders (definition)."""
    n = len(p)
    best = 0
    for perm in permutations(range(n)):
        t = 0
        val = 0
        for i in perm:
            t += p[i]
            if t <= d[i]:
                val += w[i]
        best = max(best, val)
    return best


# --------------------------------------------------------------------------
# exact solvers
# --------------------------------------------------------------------------
def brute_opt(p, w, rho, want_set=False):
    """Exhaustive search over subsets by DFS in non-decreasing p.

    Feasibility is hereditary and an item only constrains itself against the
    items before it, so the DFS enumerates exactly the feasible sets.
    Exponential, intended for n <= ~18 and pruned instances.
    """
    mu = frac(rho) - 1
    idx = sorted(range(len(p)), key=lambda i: (p[i], i))
    n = len(idx)
    suffix = [0] * (n + 1)
    for j in range(n - 1, -1, -1):
        suffix[j] = suffix[j + 1] + w[idx[j]]
    best = [0, ()]
    nodes = [0]

    def dfs(j, P, val, chosen):
        nodes[0] += 1
        if val > best[0]:
            best[0] = val
            best[1] = tuple(chosen)
        if j == n or val + suffix[j] <= best[0]:
            return
        i = idx[j]
        if P <= mu * p[i]:
            chosen.append(i)
            dfs(j + 1, P + p[i], val + w[i], chosen)
            chosen.pop()
        dfs(j + 1, P, val, chosen)

    dfs(0, 0, 0, [])
    return (best[0], best[1]) if want_set else best[0]


def count_feasible_sets(p, rho):
    mu = frac(rho) - 1
    idx = sorted(range(len(p)), key=lambda i: (p[i], i))
    n = len(idx)
    cnt = [0]

    def dfs(j, P):
        if j == n:
            cnt[0] += 1
            return
        i = idx[j]
        if P <= mu * p[i]:
            dfs(j + 1, P + p[i])
        dfs(j + 1, P)

    dfs(0, 0)
    return cnt[0]


def max_card_feasible(p, rho):
    """Largest feasible set cardinality (exhaustive)."""
    mu = frac(rho) - 1
    idx = sorted(range(len(p)), key=lambda i: (p[i], i))
    n = len(idx)
    best = [0]

    def dfs(j, P, c):
        best[0] = max(best[0], c)
        if j == n or c + (n - j) <= best[0]:
            return
        i = idx[j]
        if P <= mu * p[i]:
            dfs(j + 1, P + p[i], c + 1)
        dfs(j + 1, P, c)

    dfs(0, 0, 0)
    return best[0]


def dp_lawler_moore(p, w, d):
    """Pseudo-polynomial DP (Lawler-Moore) for integer p, d: O(nP)."""
    n = len(p)
    order = sorted(range(n), key=lambda i: (d[i], i))
    P = sum(p)
    NEG = -1
    f = [NEG] * (P + 1)
    f[0] = 0
    for i in order:
        for t in range(min(d[i], P), p[i] - 1, -1):
            if f[t - p[i]] >= 0 and f[t - p[i]] + w[i] > f[t]:
                f[t] = f[t - p[i]] + w[i]
    return max(f)


def moore_hodgson(p, d):
    """Max number of on-time jobs, Moore-Hodgson, O(n log n)."""
    import heapq
    order = sorted(range(len(p)), key=lambda i: (d[i], i))
    heap = []
    t = 0
    for i in order:
        heapq.heappush(heap, -p[i])
        t += p[i]
        if t > d[i]:
            t += heapq.heappop(heap)  # remove largest p
    return len(heap)


# --------------------------------------------------------------------------
# knapsack / subset-sum helpers for the reductions
# --------------------------------------------------------------------------
def knapsack_opt(c, v, C):
    best = 0
    q = len(c)
    for mask in range(1 << q):
        s = 0
        val = 0
        for j in range(q):
            if mask >> j & 1:
                s += c[j]
                val += v[j]
        if s <= C and val > best:
            best = val
    return best


def scale_to_integers(p, rho):
    """Multiply all sizes by a common factor so that p_i and d_i=rho*p_i are
    integers.  Returns (p_int, factor)."""
    rho = frac(rho)
    den = 1
    for x in list(p) + [rho * pi for pi in p]:
        den = den * frac(x).denominator // math.gcd(den, frac(x).denominator)
    return [int(frac(x) * den) for x in p], den


def reachable_sums_below(p_items, rho):
    """All achievable total sizes of feasible sets drawn from the given sizes (deduplicated DP over
    prefix sums in non-decreasing p).  Used where an exact-target question is asked (w = p)."""
    mu = frac(rho) - 1
    reach = {0}
    for x in sorted(p_items):
        new = {P + x for P in reach if P <= mu * x}
        reach |= new
    return reach

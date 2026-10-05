"""Shared brute-force utilities for the H2 (spoke-vs-chain) study.

Conventions: depot = node 0, sites = 1..n, metric D (n+1)x(n+1), crew speed 1,
hazard-arrival times h[1..n] (index 0 unused), weights w[1..n].
Arrival reading: site i is protected iff the crew ARRIVES at i by h[i].
All routines here are deliberately naive (exhaustive over orders/subsets) so that
they are independent of the structural lemmas (EDD, interval DP) they are used
to validate.
"""
import itertools
import math
import random
from fractions import Fraction


# ---------------------------------------------------------------- metrics
def euclid_metric(pts):
    n = len(pts)
    return [[math.dist(pts[i], pts[j]) for j in range(n)] for i in range(n)]


def graph_metric(n_nodes, rng, edge_prob=0.5, wmax=10):
    """Random connected weighted graph, shortest-path metric (Floyd-Warshall)."""
    INF = float("inf")
    d = [[INF] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        d[i][i] = 0
    # random spanning tree for connectivity
    for i in range(1, n_nodes):
        j = rng.randrange(i)
        w = rng.randint(1, wmax)
        d[i][j] = d[j][i] = w
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                w = rng.randint(1, wmax)
                d[i][j] = d[j][i] = min(d[i][j], w)
    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


def star_metric(a):
    """a[0..n] with a[0] = 0 is depot; delta(i,j)=a_i+a_j (distinct rays)."""
    n = len(a)
    return [[0 if i == j else a[i] + a[j] for j in range(n)] for i in range(n)]


def line_metric(pos):
    n = len(pos)
    return [[abs(pos[i] - pos[j]) for j in range(n)] for i in range(n)]


def is_metric(D, tol=1e-9):
    n = len(D)
    for i in range(n):
        if abs(D[i][i]) > tol:
            return False
        for j in range(n):
            if abs(D[i][j] - D[j][i]) > tol or (i != j and D[i][j] <= 0):
                return False
            for k in range(n):
                if D[i][j] > D[i][k] + D[k][j] + tol:
                    return False
    return True


# ---------------------------------------------------------------- spoke
def spoke_opt(D, h, w, speed=1.0):
    """Exhaustive over ALL permutations (no EDD lemma used).
    Arrival at the j-th site = sum_{l<j} 2 a_l/speed + a_j/speed."""
    n = len(D) - 1
    a = [D[0][i] / speed for i in range(n + 1)]
    best = 0
    for perm in itertools.permutations(range(1, n + 1)):
        t = 0.0
        tot = 0
        for i in perm:
            if t + a[i] <= h[i] + 1e-9:
                tot += w[i]
            t += 2 * a[i]
        best = max(best, tot)
    return best


# ---------------------------------------------------------------- chain
def chain_opt(D, h, w, speed=1.0):
    """Exhaustive over all permutations of all sites with direct travel between
    consecutive sites; sites visited late simply do not count.  (Appending the
    unprotected sites at the end of an ordered subset does not change any
    arrival, so this equals the optimum over ordered subsets.)"""
    n = len(D) - 1
    best = 0
    for perm in itertools.permutations(range(1, n + 1)):
        t = 0.0
        cur = 0
        tot = 0
        for i in perm:
            t += D[cur][i] / speed
            cur = i
            if t <= h[i] + 1e-9:
                tot += w[i]
        best = max(best, tot)
    return best


def chain_opt_ordered_subsets(D, h, w, speed=1.0):
    """Second independent implementation: DFS over ordered subsets, only
    sites that are on time are listed (greedy pruning is NOT used: a site is
    listed iff the DFS chooses to)."""
    n = len(D) - 1
    best = [0]

    def rec(cur, t, used, tot):
        best[0] = max(best[0], tot)
        for i in range(1, n + 1):
            if used >> i & 1:
                continue
            ti = t + D[cur][i] / speed
            if ti <= h[i] + 1e-9:
                rec(i, ti, used | (1 << i), tot + w[i])

    rec(0, 0.0, 0, 0)
    return best[0]


def chain_pareto_dp(D, h, w, speed=1.0):
    """Held-Karp style Pareto DP over (mask,last): exact for any metric, used
    for larger n (<= ~14).  State keeps Pareto set of (time, weight)."""
    n = len(D) - 1
    from collections import defaultdict
    layer = {(0, 0): [(0.0, 0)]}
    best = 0
    # only on-time sites are listed in the route
    frontier = dict(layer)
    allstates = {}
    while frontier:
        new = defaultdict(list)
        for (mask, last), plist in frontier.items():
            for i in range(1, n + 1):
                if mask >> i & 1:
                    continue
                for (t, wt) in plist:
                    ti = t + D[last][i] / speed
                    if ti <= h[i] + 1e-9:
                        new[(mask | (1 << i), i)].append((ti, wt + w[i]))
        nf = {}
        for key, lst in new.items():
            lst.sort(key=lambda x: (x[0], -x[1]))
            pr = []
            bw = -1
            for t, wt in lst:
                if wt > bw:
                    pr.append((t, wt))
                    bw = wt
            nf[key] = pr
            best = max(best, pr[-1][1])
        frontier = nf
    return best


def kappa_rho(D, sites=None):
    n = len(D) - 1
    sites = sites or list(range(1, n + 1))
    a = {i: D[0][i] for i in sites}
    kap = min((D[i][j] / (a[i] + a[j]) for i in sites for j in sites if i < j),
              default=1.0)
    rho = max(a.values()) / min(a.values())
    return kap, rho


def rand_weights(n, rng, equal=False, wmax=9):
    return [0] + ([1] * n if equal else [rng.randint(1, wmax) for _ in range(n)])

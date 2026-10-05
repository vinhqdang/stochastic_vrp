#!/usr/bin/env python3
"""ITEM 5 (Remark 4): the hard instances of Theorem 3 are realised on the
network K_{2,n}: depot D and hazard origin H, edge D-s_i of length a_i,
edge H-s_i of length A, unit speeds, return reading. Floyd-Warshall gives
the shortest paths; check p_i = 2 a_i, hazard time = A for every site, and
that the induced MWHED instance has W* >= A/2 (after scaling by 2) iff the
Partition instance is a yes-instance."""
import itertools
import random


def floyd(n, edges):
    INF = 10 ** 9
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for u, v, w in edges:
        d[u][v] = d[v][u] = min(d[u][v], w)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


def partition_yes(a):
    A = sum(a)
    return A % 2 == 0 and any(2 * sum(c) == A for r in range(len(a) + 1)
                              for c in itertools.combinations(a, r))


def opt_common_deadline(p, w, D):
    best = 0
    for r in range(len(p) + 1):
        for c in itertools.combinations(range(len(p)), r):
            if sum(p[i] for i in c) <= D:
                best = max(best, sum(w[i] for i in c))
    return best


rng = random.Random(20261005)
bad = checked = 0
for _ in range(300):
    n = rng.randint(1, 8)
    a = [rng.randint(1, 12) for _ in range(n)]
    A = sum(a)
    if A % 2 or max(a) > A // 2:
        continue
    checked += 1
    D_, H_ = 0, 1
    edges = [(D_, 2 + i, a[i]) for i in range(n)] + [(H_, 2 + i, A) for i in range(n)]
    d = floyd(n + 2, edges)
    p = [2 * d[D_][2 + i] for i in range(n)]
    hz = [d[H_][2 + i] for i in range(n)]
    ok = p == [2 * x for x in a] and all(h == A for h in hz)
    wstar = opt_common_deadline(p, a, A)            # w_i = a_i, return reading
    # yes iff the maximum on-time weight reaches A/2
    ok &= (wstar >= A // 2) == partition_yes(a)
    bad += not ok
print(f"K_(2,n) network realisation: {checked} Partition instances, mismatches: {bad}")
raise SystemExit(1 if bad else 0)

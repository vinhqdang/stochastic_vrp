"""Module B: exact and heuristic tours, and the Theorem 6 dichotomic search."""

import numpy as np
from numba import njit

HK_LIMIT = 15
HK_MAX_K = 16


@njit(cache=True)
def _held_karp_nb(W):
    n = W.shape[0]
    k = n - 1
    full = 1 << k
    dp = np.full((full, k), np.inf)
    par = np.full((full, k), -1, dtype=np.int64)
    for j in range(k):
        dp[1 << j, j] = W[0, j + 1]
    for mask in range(1, full):
        for j in range(k):
            if (mask >> j) & 1 == 0:
                continue
            cur = dp[mask, j]
            if cur == np.inf:
                continue
            for nx in range(k):
                if (mask >> nx) & 1:
                    continue
                nm = mask | (1 << nx)
                val = cur + W[j + 1, nx + 1]
                if val < dp[nm, nx]:
                    dp[nm, nx] = val
                    par[nm, nx] = j
    last = full - 1
    best = np.inf
    bj = -1
    for j in range(k):
        val = dp[last, j] + W[j + 1, 0]
        if val < best:
            best = val
            bj = j
    order = np.empty(k, dtype=np.int64)
    mask = last
    j = bj
    for pos in range(k - 1, -1, -1):
        order[pos] = j + 1
        pj = par[mask, j]
        mask ^= 1 << j
        j = pj
    return order, best


def held_karp(Wmat):
    """Exact closed-tour minimum on a (k+1)x(k+1) matrix, node 0 = depot.

    Returns (order, cost) with order a list of the indices 1..k.
    """
    W = np.ascontiguousarray(Wmat, dtype=np.float64)
    k = W.shape[0] - 1
    if k < 0:
        raise ValueError("empty matrix")
    if k == 0:
        return [], 0.0
    if k > HK_MAX_K:
        raise ValueError(f"held_karp supports k <= {HK_MAX_K}, got {k}")
    order, cost = _held_karp_nb(W)
    return [int(x) for x in order], float(cost)


def tour_stats(tour, C):
    """(L, W) of the closed tour depot -> tour -> depot."""
    if len(tour) == 0:
        return 0.0, 0.0
    seq = np.concatenate(([0], np.asarray(tour, dtype=np.int64), [0]))
    c = C[seq[:-1], seq[1:]]
    return float(c.sum()), float((c * c).sum())


@njit(cache=True)
def _cycle_cost(seq, W):
    n = seq.shape[0]
    s = 0.0
    for i in range(n):
        s += W[seq[i], seq[(i + 1) % n]]
    return s


@njit(cache=True)
def _local_search_nb(seq, W):
    cur = _cycle_cost(seq, W)
    improved = True
    while improved:
        improved = False
        n = seq.shape[0]
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a = seq[i - 1]
                b = seq[i]
                c = seq[j]
                d = seq[(j + 1) % n]
                delta = W[a, c] + W[b, d] - W[a, b] - W[c, d]
                if delta < -1e-12:
                    lo = i
                    hi = j
                    while lo < hi:
                        tmp = seq[lo]
                        seq[lo] = seq[hi]
                        seq[hi] = tmp
                        lo += 1
                        hi -= 1
                    cur += delta
                    improved = True
        for L in range(1, 4):
            if n - L < 1:
                break
            i = 1
            while i <= n - L:
                seg0 = seq[i]
                segL = seq[i + L - 1]
                m_rest = n - L
                rest = np.empty(m_rest, dtype=np.int64)
                rest[0] = seq[0]
                idx = 1
                for t in range(i + L, n):
                    rest[idx] = seq[t]
                    idx += 1
                for t in range(1, i):
                    rest[idx] = seq[t]
                    idx += 1
                R = _cycle_cost(rest, W)
                internal = 0.0
                for t in range(i, i + L - 1):
                    internal += W[seq[t], seq[t + 1]]
                best = np.inf
                bp = -1
                brev = False
                for p in range(m_rest):
                    x = rest[p]
                    y = rest[(p + 1) % m_rest]
                    fwd = W[x, seg0] + W[segL, y] - W[x, y]
                    rev = W[x, segL] + W[seg0, y] - W[x, y]
                    if fwd < best:
                        best = fwd
                        bp = p
                        brev = False
                    if rev < best:
                        best = rev
                        bp = p
                        brev = True
                if R + internal + best < cur - 1e-12:
                    new = np.empty(n, dtype=np.int64)
                    k2 = 0
                    for t in range(bp + 1):
                        new[k2] = rest[t]
                        k2 += 1
                    if brev:
                        for t in range(L - 1, -1, -1):
                            new[k2] = seq[i + t]
                            k2 += 1
                    else:
                        for t in range(L):
                            new[k2] = seq[i + t]
                            k2 += 1
                    for t in range(bp + 1, m_rest):
                        new[k2] = rest[t]
                        k2 += 1
                    seq = new
                    cur = R + internal + best
                    improved = True
                i += 1
    return seq, cur


def local_search_tour(nodes, Wmat_full, starts=8, rng=None):
    """2-opt + Or-opt (segments 1..3) on the weight matrix, best of several starts.

    `nodes` are customer ids in the full matrix; `Wmat_full` is indexed by
    node ids with node 0 the depot. Start 0 is nearest neighbour, the others
    are random permutations. Returns (tour, cost): tour is a list of customer
    ids, cost is its closed weight.
    """
    nodes = [int(x) for x in nodes]
    if len(nodes) == 0:
        return [], 0.0
    if rng is None:
        rng = np.random.default_rng(0)
    W = np.ascontiguousarray(Wmat_full, dtype=np.float64)
    arr = np.array(nodes, dtype=np.int64)
    starts_list = []
    remaining = set(nodes)
    cur = 0
    nn = []
    while remaining:
        nxt = min(remaining, key=lambda j: (W[cur, j], j))
        nn.append(nxt)
        remaining.discard(nxt)
        cur = nxt
    starts_list.append(np.array(nn, dtype=np.int64))
    for _ in range(max(0, int(starts) - 1)):
        starts_list.append(rng.permutation(arr))
    best_seq = None
    best_cost = np.inf
    for st in starts_list:
        seq = np.concatenate((np.array([0], dtype=np.int64), st))
        seq, cost = _local_search_nb(seq, W)
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_seq = seq
    return [int(x) for x in best_seq[1:]], float(best_cost)


class TourSolver:
    """Risk-consistent tour minimiser of L + kappa * v * sqrt(W) for a customer set.

    kappa == 0 returns the TSP-optimal tour (ties broken by smaller W).
    kappa > 0 runs the dichotomic search of Theorem 6 over arc weights
    c + theta*c^2 and returns the enumerated tour with the smallest objective.
    Results are memoised on frozenset(S).
    """

    def __init__(self, C, kappa, v):
        self.C = np.ascontiguousarray(C, dtype=np.float64)
        self.C2 = self.C * self.C
        self.kappa = float(kappa)
        self.v = float(v)
        self._cache = {}
        self.n_calls = 0
        self.n_hk = 0
        self.n_cache_hits = 0
        self.n_ls = 0
        self.ls_starts = 8

    def _opt_tour(self, nodes, Wfull):
        k = len(nodes)
        if k <= HK_LIMIT:
            idx = np.array([0] + nodes, dtype=np.int64)
            order, _ = held_karp(Wfull[np.ix_(idx, idx)])
            self.n_hk += 1
            return [nodes[o - 1] for o in order]
        self.n_ls += 1
        rng = np.random.default_rng([7919, k] + list(nodes))
        tour, _ = local_search_tour(nodes, Wfull, starts=self.ls_starts, rng=rng)
        return tour

    def _supported_tours(self, nodes):
        C = self.C
        C2 = self.C2
        found = {}

        def add(t):
            key = tuple(t)
            if key not in found:
                L, W = tour_stats(t, C)
                found[key] = (list(t), L, W)
            return found[key]

        a_tour = self._opt_tour(nodes, C + 1e-9 * C2)
        b_tour = self._opt_tour(nodes, C2 + 1e-9 * C)
        a = add(a_tour)
        b = add(b_tour)
        stack = [(a, b)]
        while stack:
            pa, pb = stack.pop()
            _, La, Wa = pa
            _, Lb, Wb = pb
            if not (La < Lb and Wa > Wb):
                continue
            theta = (Lb - La) / (Wa - Wb)
            t = add(self._opt_tour(nodes, C + theta * C2))
            Lt, Wt = t[1], t[2]
            if Lt + theta * Wt < La + theta * Wa - 1e-9:
                stack.append((pa, t))
                stack.append((t, pb))
        return list(found.values()), a

    def solve(self, S):
        """Return (tour, L, W) for customer set S (iterable of node ids)."""
        self.n_calls += 1
        key = frozenset(int(x) for x in S)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]
        nodes = sorted(key)
        if len(nodes) == 0:
            res = ([], 0.0, 0.0)
        elif self.kappa == 0.0:
            tour = self._opt_tour(nodes, self.C + 1e-9 * self.C2)
            L, W = tour_stats(tour, self.C)
            res = (tour, L, W)
        else:
            cands, _ = self._supported_tours(nodes)
            best = None
            best_val = np.inf
            for tour, L, W in cands:
                val = L + self.kappa * self.v * np.sqrt(W)
                if val < best_val - 1e-12:
                    best_val = val
                    best = (tour, L, W)
            res = best
        self._cache[key] = res
        return res


def is_tsp_optimal(tour, C):
    """(is_optimal, L_opt): compares the closed length of tour with Held-Karp (|S| <= 15)."""
    nodes = [int(x) for x in tour]
    if len(nodes) > HK_LIMIT:
        raise ValueError("is_tsp_optimal supports |S| <= 15")
    if len(nodes) == 0:
        return True, 0.0
    idx = np.array([0] + nodes, dtype=np.int64)
    Wsub = np.ascontiguousarray(C, dtype=np.float64)[np.ix_(idx, idx)]
    order, l_opt = held_karp(Wsub)
    L, _ = tour_stats(nodes, C)
    return bool(L <= l_opt + 1e-9), float(l_opt)

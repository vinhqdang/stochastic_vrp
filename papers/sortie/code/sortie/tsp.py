"""Module B: exact and heuristic tours, and the Theorem 6 dichotomic search."""

import numpy as np
from numba import njit

HK_LIMIT = 15
HK_MAX_K = 16
FAST_CACHE_MAX = 200000
FAST_MAX_TOURS = 24
FAST_LS_ROUNDS = 100
LS_MAX_ROUNDS = 1000


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


def tour_stats(tour, C, D=None):
    """(L, W) of the closed tour depot -> tour -> depot.

    L = sum of C over the arcs; W = sum of D^2 over the arcs (D = C o H for
    heterogeneous arc noise, D = C when omitted).
    """
    if len(tour) == 0:
        return 0.0, 0.0
    seq = np.concatenate(([0], np.asarray(tour, dtype=np.int64), [0]))
    c = C[seq[:-1], seq[1:]]
    if D is None:
        return float(c.sum()), float((c * c).sum())
    d = D[seq[:-1], seq[1:]]
    return float(c.sum()), float((d * d).sum())


@njit(cache=True)
def _cycle_cost(seq, W):
    n = seq.shape[0]
    s = 0.0
    for i in range(n):
        s += W[seq[i], seq[(i + 1) % n]]
    return s


@njit(cache=True)
def _local_search_nb(seq, W, max_rounds):
    """2-opt and Or-opt descent; stops after max_rounds sweeps (round-off can cycle at large weights)."""
    cur = _cycle_cost(seq, W)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        rounds += 1
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


@njit(cache=True)
def _nn_order_nb(nodes, W):
    """Nearest-neighbour order of `nodes` from depot 0 (ties by smaller node id)."""
    m = nodes.shape[0]
    out = np.empty(m, dtype=np.int64)
    used = np.zeros(m, dtype=np.bool_)
    cur = 0
    for pos in range(m):
        best = -1
        bw = np.inf
        for t in range(m):
            if used[t]:
                continue
            j = nodes[t]
            w = W[cur, j]
            if best < 0 or w < bw or (w == bw and j < nodes[best]):
                best = t
                bw = w
        used[best] = True
        out[pos] = nodes[best]
        cur = nodes[best]
    return out


def local_search_tour(nodes, Wmat_full, starts=8, rng=None, init=None):
    """2-opt + Or-opt (segments 1..3) on the weight matrix, best of several starts.

    `nodes` are customer ids in the full matrix; `Wmat_full` is indexed by
    node ids with node 0 the depot. Start 0 is nearest neighbour, the others
    are random permutations. If `init` (a permutation of `nodes`) is given it
    is run first, as an extra start before the nearest-neighbour one. Returns
    (tour, cost): tour is a list of customer ids, cost is its closed weight.
    """
    nodes = [int(x) for x in nodes]
    if len(nodes) == 0:
        return [], 0.0
    if rng is None:
        rng = np.random.default_rng(0)
    W = np.ascontiguousarray(Wmat_full, dtype=np.float64)
    arr = np.array(nodes, dtype=np.int64)
    starts_list = []
    if init is not None:
        init_arr = np.array([int(x) for x in init], dtype=np.int64)
        if sorted(init_arr.tolist()) != sorted(nodes):
            raise ValueError("init must be a permutation of nodes")
        starts_list.append(init_arr)
    starts_list.append(_nn_order_nb(arr, W))
    for _ in range(max(0, int(starts) - 1)):
        starts_list.append(rng.permutation(arr))
    best_seq = None
    best_cost = np.inf
    for st in starts_list:
        seq = np.concatenate((np.array([0], dtype=np.int64), st))
        seq, cost = _local_search_nb(seq, W, LS_MAX_ROUNDS)
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_seq = seq
    return [int(x) for x in best_seq[1:]], float(best_cost)


def cheapest_insertion_hint(seq, nodes_set, C):
    """Warm-start sequence for the customer set `nodes_set` built from the parent sequence `seq`.

    Customers of seq outside nodes_set are dropped (order kept); each missing customer
    is then inserted, in sorted id order, at its cheapest position (depot 0 at both ends).
    """
    target = set(int(x) for x in nodes_set)
    tour = []
    seen = set()
    for x in seq:
        x = int(x)
        if x in target and x not in seen:
            tour.append(x)
            seen.add(x)
    C = np.asarray(C, dtype=np.float64)
    for x in sorted(target - seen):
        full = np.array([0] + tour + [0], dtype=np.int64)
        a = full[:-1]
        b = full[1:]
        dl = C[a, x] + C[x, b] - C[a, b]
        tour.insert(int(np.argmin(dl)), x)
    return tour


class TourSolver:
    """Risk-consistent tour minimiser of L + kappa * v * sqrt(W) for a customer set.

    kappa == 0 returns the TSP-optimal tour (ties broken by smaller W).
    kappa > 0 runs the dichotomic search of Theorem 6 (THEORY.md; "risk-consistent tours
    are supported" in sortie.tex) over arc weights c + theta*D^2 and returns the
    enumerated tour with the smallest objective.
    Results are memoised on frozenset(S). W and the dichotomic weights use
    D (the arc noise weights C o H); D defaults to C.

    solve is the exact lower level. solve_fast is a screening lower level: one
    warm-start 2-opt/Or-opt run per tour subproblem, no Held-Karp, no random starts.
    It reads the exact memo when the set is already solved exactly.
    """

    def __init__(self, C, kappa, v, D=None):
        self.C = np.ascontiguousarray(C, dtype=np.float64)
        self.D = self.C if D is None else np.ascontiguousarray(D, dtype=np.float64)
        self.D2 = self.D * self.D
        self.kappa = float(kappa)
        self.v = float(v)
        self._cache = {}
        self._fast_cache = {}
        self.n_calls = 0
        self.n_hk = 0
        self.n_cache_hits = 0
        self.n_ls = 0
        self.n_fast = 0
        self.n_fast_hits = 0
        self.ls_starts = 8

    def _opt_tour(self, nodes, Wfull, init=None):
        k = len(nodes)
        if k <= HK_LIMIT:
            idx = np.array([0] + nodes, dtype=np.int64)
            order, _ = held_karp(Wfull[np.ix_(idx, idx)])
            self.n_hk += 1
            return [nodes[o - 1] for o in order]
        self.n_ls += 1
        rng = np.random.default_rng([7919, k] + list(nodes))
        tour, _ = local_search_tour(nodes, Wfull, starts=self.ls_starts, rng=rng, init=init)
        return tour

    def _warm_tour(self, nodes, Wfull, init):
        """One 2-opt/Or-opt run from the warm start `init` (NN order when None)."""
        if init is None:
            init = _nn_order_nb(np.asarray(nodes, dtype=np.int64), self.C)
        seq = np.concatenate((np.array([0], dtype=np.int64), np.asarray(init, dtype=np.int64)))
        seq, _ = _local_search_nb(seq, np.ascontiguousarray(Wfull, dtype=np.float64),
                                  FAST_LS_ROUNDS)
        return [int(x) for x in seq[1:]]

    def _supported_tours(self, nodes, tour_fn, max_tours=None):
        """Dichotomic search over the weights; tour_fn(Wfull) gives the tour of one subproblem.

        A pair of tours is expanded at most once (a revisit would recompute the same
        subproblem). max_tours, when given, stops the search after that many distinct tours.
        """
        C = self.C
        D = self.D
        D2 = self.D2
        found = {}

        def add(t):
            key = tuple(t)
            if key not in found:
                L, W = tour_stats(t, C, D)
                found[key] = (list(t), L, W)
            return found[key]

        a_tour = tour_fn(C + 1e-9 * D2)
        b_tour = tour_fn(D2 + 1e-9 * C)
        a = add(a_tour)
        b = add(b_tour)
        stack = [(a, b)]
        expanded = set()
        while stack:
            if max_tours is not None and len(found) >= max_tours:
                break
            pa, pb = stack.pop()
            pair = (tuple(pa[0]), tuple(pb[0]))
            if pair in expanded:
                continue
            expanded.add(pair)
            _, La, Wa = pa
            _, Lb, Wb = pb
            if not (La < Lb and Wa > Wb):
                continue
            theta = (Lb - La) / (Wa - Wb)
            t = add(tour_fn(C + theta * D2))
            Lt, Wt = t[1], t[2]
            if Lt + theta * Wt < La + theta * Wa - 1e-9:
                stack.append((pa, t))
                stack.append((t, pb))
        return list(found.values()), a

    def _pick(self, cands):
        best = None
        best_val = np.inf
        for tour, L, W in cands:
            val = L + self.kappa * self.v * np.sqrt(W)
            if val < best_val - 1e-12:
                best_val = val
                best = (tour, L, W)
        return best

    def solve(self, S, hint=None):
        """Return (tour, L, W) for customer set S (iterable of node ids), exact lower level.

        hint (a parent sequence) warm-starts the local search for sets above HK_LIMIT.
        Cache hits ignore the hint, so the first computation of a set wins.
        """
        self.n_calls += 1
        key = frozenset(int(x) for x in S)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]
        nodes = sorted(key)
        if len(nodes) == 0:
            res = ([], 0.0, 0.0)
        else:
            init = None
            if hint is not None and len(nodes) > HK_LIMIT:
                init = cheapest_insertion_hint(hint, key, self.C)
            if self.kappa == 0.0:
                tour = self._opt_tour(nodes, self.C + 1e-9 * self.D2, init)
                L, W = tour_stats(tour, self.C, self.D)
                res = (tour, L, W)
            else:
                cands, _ = self._supported_tours(
                    nodes, lambda Wf: self._opt_tour(nodes, Wf, init))
                res = self._pick(cands)
        self._cache[key] = res
        return res

    def solve_fast(self, S, hint=None):
        """Return (tour, L, W) for S from the screening lower level (never Held-Karp).

        An exact result for S, when present, is returned instead. Otherwise the result
        is memoised in _fast_cache, which is cleared once it exceeds FAST_CACHE_MAX.
        """
        key = frozenset(int(x) for x in S)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]
        if key in self._fast_cache:
            self.n_fast_hits += 1
            return self._fast_cache[key]
        self.n_fast += 1
        nodes = sorted(key)
        if len(nodes) == 0:
            res = ([], 0.0, 0.0)
        else:
            if hint is not None:
                init = cheapest_insertion_hint(hint, key, self.C)
            else:
                init = _nn_order_nb(np.asarray(nodes, dtype=np.int64), self.C).tolist()
            if self.kappa == 0.0:
                tour = self._warm_tour(nodes, self.C + 1e-9 * self.D2, init)
                L, W = tour_stats(tour, self.C, self.D)
                res = (tour, L, W)
            else:
                cands, _ = self._supported_tours(
                    nodes, lambda Wf: self._warm_tour(nodes, Wf, init),
                    max_tours=FAST_MAX_TOURS)
                res = self._pick(cands)
        if len(self._fast_cache) > FAST_CACHE_MAX:
            self._fast_cache.clear()
        self._fast_cache[key] = res
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

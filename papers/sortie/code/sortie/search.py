"""Module E: upper-level search for ordered route fairness (SORTIE).

Plan state: m nonempty tours (customer sequences, depot implicit). Each
iteration screens every relocate and swap move with first-order prices
(THEORY Prop. 3), evaluates the best K exactly through the lower level,
and applies the best budget-feasible improving move. Without an improving
move a ruin-and-recreate perturbation is proposed and accepted by simulated
annealing on the objective.
"""

import math
import time

import numpy as np

from .instances import effective_H, noise_matrix
from .pboi import expost_var, lam_vector, ordered, pboi, rank_prices, sigma_prices
from .tsp import TourSolver, tour_stats

FEAS_TOL = 1e-9
MOVE_TOL = 1e-9
BUDGET_SLACK = 0.02
SCEN_S = 200
SCEN_SEED = 4242
FREE_PASSES = 4
REPAIR_STEPS = 200
RUIN_MAX = 12
ORDERED_MEASURES = ("max", "range", "gmd", "mad")
MEASURES = ORDERED_MEASURES + ("var",)
MODES = ("cost", "det", "pboi", "saa", "detsig", "pboi_normal")


# ---------------------------------------------------------------------------
# Objective adapters
# ---------------------------------------------------------------------------

class CostAdapter:
    """Total expected length: value = sum L, prices (1, 0)."""

    mode = "cost"
    measure = None

    def __init__(self):
        self.v = 0.0

    def value(self, L, W, seqs):
        return float(np.sum(L))

    def prices(self, L, W, seqs):
        m = len(L)
        return np.ones(m), np.zeros(m)


def _rank_prices_det(L, lam):
    m = len(L)
    order = np.argsort(L, kind="stable")
    rank = np.empty(m, dtype=np.int64)
    rank[order] = np.arange(m)
    return lam[rank]


class DetAdapter:
    """Ex-ante fairness of the expected lengths (deterministic objective)."""

    mode = "det"

    def __init__(self, measure, v):
        self.measure = measure
        self.v = float(v)

    def value(self, L, W, seqs):
        if self.measure == "var":
            return float(np.mean((L - L.mean()) ** 2))
        return float(ordered(L, lam_vector(self.measure, len(L))))

    def prices(self, L, W, seqs):
        m = len(L)
        zeros = np.zeros(m)
        if self.measure == "var":
            return 2.0 * (L - L.mean()) / m, zeros
        return _rank_prices_det(L, lam_vector(self.measure, m)), zeros


class DetSigAdapter:
    """Deterministic ordered measure of the risk-adjusted lengths L + z sigma.

    sigma_k = v sqrt(W_k). The measure is applied to y = L + z sigma (or its
    empirical variance). Prices are the deterministic prices of y, and
    psi = z varpi, the first-order response to sigma.
    """

    mode = "detsig"

    def __init__(self, measure, v, z=1.0):
        self.measure = measure
        self.v = float(v)
        self.z = float(z)
        self._det = DetAdapter(measure, v)

    def _risk(self, L, W):
        return L + self.z * self.v * np.sqrt(np.maximum(W, 0.0))

    def value(self, L, W, seqs):
        return self._det.value(self._risk(L, W), W, seqs)

    def prices(self, L, W, seqs):
        varpi, _ = self._det.prices(self._risk(L, W), W, seqs)
        return varpi, self.z * varpi


class PboiAdapter:
    """Ex-post fairness E l_lambda(X) (Theorem 2) or expost_var.

    family selects the marginal law of the driver lengths: "gamma" (the
    default, mode pboi) or "normal" (mode pboi_normal, the ablation).
    """

    mode = "pboi"

    def __init__(self, measure, v, family="gamma"):
        self.measure = measure
        self.v = float(v)
        self.family = family

    def _sigma(self, W):
        return self.v * np.sqrt(np.maximum(W, 0.0))

    def value(self, L, W, seqs):
        sigma = self._sigma(W)
        if self.measure == "var":
            return float(expost_var(L, sigma))
        return float(pboi(L, sigma, lam_vector(self.measure, len(L)), family=self.family))

    def prices(self, L, W, seqs):
        sigma = self._sigma(W)
        m = len(L)
        if self.measure == "var":
            varpi = 2.0 * (L - L.mean()) / m
            psi = sigma_prices(L, sigma, "var")
        else:
            lam = lam_vector(self.measure, m)
            varpi = rank_prices(L, sigma, lam, family=self.family)
            psi = sigma_prices(L, sigma, lam, family=self.family)
        return varpi, psi


class SAAAdapter:
    """Sample-average approximation over a fixed arc-scenario tensor."""

    mode = "saa"

    def __init__(self, measure, v, eps, C):
        self.measure = measure
        self.v = float(v)
        self.eps = eps
        self.C = C
        self._xc = {}

    def _route_X(self, seq):
        key = tuple(seq)
        x = self._xc.get(key)
        if x is None:
            if len(self._xc) > 50000:
                self._xc.clear()
            full = np.array([0] + list(seq) + [0], dtype=np.int64)
            src = full[:-1]
            dst = full[1:]
            x = self.eps[:, src, dst] @ self.C[src, dst]
            self._xc[key] = x
        return x

    def _matrix(self, seqs):
        return np.stack([self._route_X(s) for s in seqs], axis=1)

    def value(self, L, W, seqs):
        X = self._matrix(seqs)
        if self.measure == "var":
            return float(np.mean(X.var(axis=1)))
        lam = lam_vector(self.measure, X.shape[1])
        return float(np.mean(np.sort(X, axis=1) @ lam))

    def prices(self, L, W, seqs):
        X = self._matrix(seqs)
        m = X.shape[1]
        if self.measure == "var":
            mu = X.mean(axis=1, keepdims=True)
            varpi = np.mean(2.0 * (X - mu) / m, axis=0)
        else:
            lam = lam_vector(self.measure, m)
            ranks = np.argsort(np.argsort(X, axis=1, kind="stable"), axis=1)
            varpi = np.mean(lam[ranks], axis=0)
        return varpi, np.zeros(m)


def make_objective(mode, measure, v, scen=None, C=None, S=SCEN_S, seed=SCEN_SEED, H=None,
                   z=1.0):
    """Objective adapter with value(L, W, routes) and prices(L, W, routes).

    `mode` is one of cost, det, pboi, saa, detsig, pboi_normal. For saa the arc
    scenario tensor eps (S, N, N) is `scen` or, if absent, drawn from
    Gamma(1/v^2, v^2) with `seed` (requires C for the node count and the arc
    costs). With a multiplier H the arc (i, j) is drawn with cv v * H_ij instead.
    z is the risk weight of detsig (ordered measure of L + z sigma).
    """
    if mode == "cost":
        return CostAdapter()
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if measure not in MEASURES:
        raise ValueError(f"unknown measure {measure!r}")
    if mode == "det":
        return DetAdapter(measure, v)
    if mode == "detsig":
        return DetSigAdapter(measure, v, z)
    if mode == "pboi":
        if v <= 0:
            raise ValueError("pboi needs v > 0 (use mode 'det' for v = 0)")
        return PboiAdapter(measure, v)
    if mode == "pboi_normal":
        if v <= 0:
            raise ValueError("pboi_normal needs v > 0 (use mode 'det' for v = 0)")
        return PboiAdapter(measure, v, family="normal")
    if C is None:
        raise ValueError("saa needs the distance matrix C")
    C = np.ascontiguousarray(C, dtype=np.float64)
    if scen is None:
        N = C.shape[0]
        if v == 0:
            eps = np.ones((S, N, N))
        elif H is None:
            rng = np.random.default_rng(seed)
            eps = rng.gamma(shape=1.0 / (v * v), scale=v * v, size=(S, N, N))
        else:
            rng = np.random.default_rng(seed)
            vh = v * np.asarray(H, dtype=np.float64)
            eps = rng.gamma(shape=1.0 / (vh * vh), scale=vh * vh, size=(S, N, N))
    else:
        eps = np.asarray(scen, dtype=np.float64)
    return SAAAdapter(measure, v, eps, C)


# ---------------------------------------------------------------------------
# Plan state and per-route precomputation
# ---------------------------------------------------------------------------

class _Ctx:
    def __init__(self, inst, solver, consistency, v, rng, K, screen, deadline):
        self.C = np.ascontiguousarray(inst.C, dtype=np.float64)
        self.D = np.ascontiguousarray(noise_matrix(inst), dtype=np.float64)
        self.D2 = self.D * self.D
        self.N = self.C.shape[0]
        self.n = self.N - 1
        self.m = int(inst.m)
        self.Q = int(inst.Q)
        self.demand = np.asarray(inst.demand, dtype=np.int64)
        self.solver = solver
        self.free = consistency == "free"
        self.v = float(v)
        self.rng = rng
        self.K = int(K)
        self.screen = screen
        self.deadline = deadline
        self.n_exact = 0
        self.n_confirm = 0
        self.t_screen = 0.0
        self.t_exact = 0.0
        self.t_perturb = 0.0


class _Plan:
    __slots__ = ("seqs", "L", "W", "load", "pre")

    def __init__(self, seqs, L, W, pre, ctx):
        self.seqs = seqs
        self.L = L
        self.W = W
        self.load = np.array(
            [int(ctx.demand[np.asarray(s, dtype=np.int64)].sum()) for s in seqs],
            dtype=np.int64,
        )
        self.pre = pre


def _build_plan(seqs, ctx):
    m = len(seqs)
    L = np.empty(m)
    W = np.empty(m)
    for k, s in enumerate(seqs):
        L[k], W[k] = tour_stats(s, ctx.C, ctx.D)
    return _Plan(seqs, L, W, [None] * m, ctx)


class _RoutePre:
    __slots__ = ("cust", "Lrem", "Wrem", "Lbest", "Wbest", "Lsw", "Wsw")


def _route_pre(seq, ctx):
    """Removal deltas, best insertion deltas and swap-insertion deltas of one route.

    Lrem[p], Wrem[p]: removal of the customer at position p (0-based).
    Lbest[j], Wbest[j]: best insertion of node j into this route (L-optimal edge).
    Lsw[p, j], Wsw[p, j]: insertion of node j into the route after removing
    position p (edges touching p are replaced by the bridging edge).
    """
    C = ctx.C
    D2 = ctx.D2
    N = ctx.N
    n = len(seq)
    full = np.array([0] + list(seq) + [0], dtype=np.int64)
    prev = full[0:n]
    cur = full[1:n + 1]
    nxt = full[2:n + 2]
    Lrem = C[prev, cur] + C[cur, nxt] - C[prev, nxt]
    Wrem = D2[prev, cur] + D2[cur, nxt] - D2[prev, nxt]
    X = full[:-1]
    Y = full[1:]
    Lins = C[X, :] + C[:, Y].T - C[X, Y][:, None]
    Wins = D2[X, :] + D2[:, Y].T - D2[X, Y][:, None]
    cols = np.arange(N)
    eb = np.argmin(Lins, axis=0)
    Lbest = Lins[eb, cols]
    Wbest = Wins[eb, cols]
    Lbr = C[prev, :] + C[:, nxt].T - C[prev, nxt][:, None]
    Wbr = D2[prev, :] + D2[:, nxt].T - D2[prev, nxt][:, None]
    rows = np.arange(n)
    Lm = np.broadcast_to(Lins, (n, n + 1, N)).copy()
    Wm = np.broadcast_to(Wins, (n, n + 1, N)).copy()
    Lm[rows, rows, :] = np.inf
    Lm[rows, rows + 1, :] = np.inf
    Lm = np.concatenate([Lm, Lbr[:, None, :]], axis=1)
    Wm = np.concatenate([Wm, Wbr[:, None, :]], axis=1)
    ch = np.argmin(Lm, axis=1)[:, None, :]
    Lsw = np.take_along_axis(Lm, ch, axis=1)[:, 0, :]
    Wsw = np.take_along_axis(Wm, ch, axis=1)[:, 0, :]
    pre = _RoutePre()
    pre.cust = np.asarray(seq, dtype=np.int64)
    pre.Lrem = Lrem
    pre.Wrem = Wrem
    pre.Lbest = Lbest
    pre.Wbest = Wbest
    pre.Lsw = Lsw
    pre.Wsw = Wsw
    return pre


def _get_pre(plan, k, ctx):
    pre = plan.pre[k]
    if pre is None:
        pre = _route_pre(plan.seqs[k], ctx)
        plan.pre[k] = pre
    return pre


def _pred_terms(varpi, psi, a, b, Wa, Wb, dLa, dLb, dWa, dWb, v):
    sa = v * (np.sqrt(np.maximum(Wa + dWa, 0.0)) - math.sqrt(Wa))
    sb = v * (np.sqrt(np.maximum(Wb + dWb, 0.0)) - math.sqrt(Wb))
    return varpi[a] * dLa + varpi[b] * dLb + psi[a] * sa + psi[b] * sb


# ---------------------------------------------------------------------------
# First-order screening and exact evaluation
# ---------------------------------------------------------------------------

def _move_blocks(plan, varpi, psi, budget, ctx):
    """First-order prediction of every relocate and swap move, in blocks.

    A block is (kind, a, b, pred, ca, cb) with one entry per move of that route
    pair (relocate: pred has one entry per customer of a; swap: a grid over the
    customers of a and b). pred is +inf where the move is infeasible for the
    screen (capacity, or sum L above budget * (1 + BUDGET_SLACK)).
    """
    m = ctx.m
    pre = [_get_pre(plan, k, ctx) for k in range(m)]
    thr = budget * (1.0 + BUDGET_SLACK) if np.isfinite(budget) else np.inf
    W = plan.W
    load = plan.load
    sumL = float(plan.L.sum())
    v = ctx.v
    Q = ctx.Q
    dem = ctx.demand
    blocks = []
    for a in range(m):
        Ra = pre[a]
        ca = Ra.cust
        na = len(ca)
        qa = dem[ca]
        if na >= 2:
            for b in range(m):
                if b == a:
                    continue
                Rb = pre[b]
                dLa = -Ra.Lrem
                dWa = -Ra.Wrem
                dLb = Rb.Lbest[ca]
                dWb = Rb.Wbest[ca]
                pred = _pred_terms(varpi, psi, a, b, W[a], W[b], dLa, dLb, dWa, dWb, v)
                ok = (load[b] + qa <= Q) & (sumL + dLa + dLb <= thr)
                blocks.append((0, a, b, np.where(ok, pred, np.inf), ca, None))
    for a in range(m):
        Ra = pre[a]
        ca = Ra.cust
        qa = dem[ca]
        for b in range(a + 1, m):
            Rb = pre[b]
            cb = Rb.cust
            qb = dem[cb]
            dLa = -Ra.Lrem[:, None] + Ra.Lsw[:, cb]
            dWa = -Ra.Wrem[:, None] + Ra.Wsw[:, cb]
            dLb = -Rb.Lrem[None, :] + Rb.Lsw[:, ca].T
            dWb = -Rb.Wrem[None, :] + Rb.Wsw[:, ca].T
            pred = _pred_terms(varpi, psi, a, b, W[a], W[b], dLa, dLb, dWa, dWb, v)
            ok = ((load[a] - qa[:, None] + qb[None, :] <= Q)
                  & (load[b] - qb[None, :] + qa[:, None] <= Q)
                  & (sumL + dLa + dLb <= thr))
            blocks.append((1, a, b, np.where(ok, pred, np.inf), ca, cb))
    return blocks


def _flatten_blocks(blocks):
    """Concatenated predictions (allp) and the start offset of every block."""
    sizes = np.array([blk[3].size for blk in blocks], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
    allp = np.concatenate([blk[3].ravel() for blk in blocks])
    return allp, starts


def _decode_move(blocks, starts, g):
    """Move tuple (kind, a, b, i, j) of flat index g; relocate has j = -1."""
    g = int(g)
    bi = int(np.searchsorted(starts, g, side="right")) - 1
    kind, a, b, _, ca, cb = blocks[bi]
    loc = g - int(starts[bi])
    if kind == 0:
        return (0, a, b, int(ca[loc]), -1)
    p, q = divmod(loc, len(cb))
    return (1, a, b, int(ca[p]), int(cb[q]))


def _feasible_moves(plan, varpi, psi, budget, ctx):
    """Every move the screen accepts (finite prediction) and its predicted change."""
    blocks = _move_blocks(plan, varpi, psi, budget, ctx)
    if not blocks:
        return [], np.empty(0)
    allp, starts = _flatten_blocks(blocks)
    fin = np.flatnonzero(np.isfinite(allp))
    return [_decode_move(blocks, starts, g) for g in fin], allp[fin]


def _screen(plan, varpi, psi, budget, ctx):
    """Predicted-best K moves (relocate and swap) from the first-order model."""
    blocks = _move_blocks(plan, varpi, psi, budget, ctx)
    if not blocks:
        return []
    allp, starts = _flatten_blocks(blocks)
    K = ctx.K
    if ctx.screen == "random":
        fin = np.flatnonzero(np.isfinite(allp))
        if fin.size == 0:
            return []
        pick = ctx.rng.choice(fin, size=min(K, fin.size), replace=False)
    else:
        order = np.argsort(allp, kind="stable")[:K]
        pick = order[np.isfinite(allp[order])]
    return [_decode_move(blocks, starts, g) for g in pick]


def _intra_neighbours(seq):
    """2-opt and Or-opt (segments 1..3, both orientations) neighbours of a sequence."""
    n = len(seq)
    for i in range(n):
        for j in range(i + 1, n):
            yield seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
    for seg_len in (1, 2, 3):
        if n - seg_len < 1:
            break
        for i in range(n - seg_len + 1):
            seg = seq[i:i + seg_len]
            rest = seq[:i] + seq[i + seg_len:]
            for k in range(len(rest) + 1):
                if k == i:
                    if seg_len == 1:
                        continue
                    yield rest[:k] + seg[::-1] + rest[k:]
                    continue
                yield rest[:k] + seg + rest[k:]
                if seg_len > 1:
                    yield rest[:k] + seg[::-1] + rest[k:]


def _free_descent(seqs, L, W, idxs, adapter, budget, ctx):
    """Intra-route re-sequencing that lowers the global objective and keeps the budget."""
    seqs = list(seqs)
    L = L.copy()
    W = W.copy()
    cur = adapter.value(L, W, seqs)
    for idx in idxs:
        for _ in range(FREE_PASSES):
            if time.perf_counter() >= ctx.deadline:
                return seqs, L, W
            improved = False
            for cand in _intra_neighbours(seqs[idx]):
                Lc, Wc = tour_stats(cand, ctx.C, ctx.D)
                Ln = L.copy()
                Wn = W.copy()
                Ln[idx] = Lc
                Wn[idx] = Wc
                if Ln.sum() > budget + FEAS_TOL:
                    continue
                sq = list(seqs)
                sq[idx] = cand
                val = adapter.value(Ln, Wn, sq)
                if val < cur - MOVE_TOL * max(1.0, abs(cur)):
                    seqs, L, W, cur = sq, Ln, Wn, val
                    improved = True
                    break
            if not improved:
                break
    return seqs, L, W


def _resolve_move(plan, mv, ctx, free, fast=False):
    """Apply a move and re-solve its two routes; returns (seqs, L, W) with no budget test.

    fast=True re-solves with the screening lower level (solve_fast).
    """
    kind, a, b, i, j = mv
    sa = plan.seqs[a]
    sb = plan.seqs[b]
    if kind == 0:
        new_a = [x for x in sa if x != i]
        new_b = sb + [i]
    else:
        new_a = [j if x == i else x for x in sa]
        new_b = [i if x == j else x for x in sb]
    solve = ctx.solver.solve_fast if fast else ctx.solver.solve
    # Free mode keeps the original hint-free lower level: its parent sequences are
    # free-descent sequences, not solver tours.
    ta, La, Wa = solve(new_a, hint=None if free else sa)
    tb, Lb, Wb = solve(new_b, hint=None if free else sb)
    seqs = list(plan.seqs)
    seqs[a] = list(ta)
    seqs[b] = list(tb)
    L = plan.L.copy()
    W = plan.W.copy()
    L[a] = La
    W[a] = Wa
    L[b] = Lb
    W[b] = Wb
    return seqs, L, W


def _exact_move(plan, mv, adapter, budget, ctx, free, fast=False):
    """Apply a move, re-solve the two changed routes, return (plan, value) or None.

    fast=True re-solves with the screening lower level (solve_fast); the result
    must then be confirmed with _confirm before it is applied.
    """
    a, b = mv[1], mv[2]
    seqs, L, W = _resolve_move(plan, mv, ctx, free, fast)
    if free:
        seqs, L, W = _free_descent(seqs, L, W, (a, b), adapter, budget, ctx)
    if L.sum() > budget + FEAS_TOL:
        return None
    pre = list(plan.pre)
    pre[a] = None
    pre[b] = None
    child = _Plan(seqs, L, W, pre, ctx)
    return child, adapter.value(L, W, seqs)


def _confirm(plan, routes, adapter, budget, ctx):
    """Re-solve the given routes of a screened child with the exact lower level.

    Returns (child, value), or None when the confirmed plan exceeds the budget.
    """
    ctx.n_confirm += 1
    seqs = list(plan.seqs)
    L = plan.L.copy()
    W = plan.W.copy()
    for r in routes:
        t, Lr, Wr = ctx.solver.solve(seqs[r], hint=seqs[r])
        seqs[r] = list(t)
        L[r] = Lr
        W[r] = Wr
    if L.sum() > budget + FEAS_TOL:
        return None
    pre = list(plan.pre)
    for r in routes:
        pre[r] = None
    child = _Plan(seqs, L, W, pre, ctx)
    return child, adapter.value(L, W, seqs)


def _improve_step(plan, adapter, budget, cur_val, ctx, varpi, psi):
    """Screen all moves, evaluate the top K exactly, return the best improving (plan, value).

    Outside free mode the K moves are evaluated with the fast lower level; every
    candidate that improves on cur_val is confirmed exactly, in order of its fast
    value, and the first one that still improves is returned.
    """
    t0 = time.perf_counter()
    moves = _screen(plan, varpi, psi, budget, ctx)
    t1 = time.perf_counter()
    ctx.t_screen += t1 - t0
    tol = MOVE_TOL * max(1.0, abs(cur_val))
    if ctx.free:
        free = adapter.mode != "cost"
        best = None
        best_val = cur_val
        for mv in moves:
            ctx.n_exact += 1
            res = _exact_move(plan, mv, adapter, budget, ctx, free)
            if res is None:
                continue
            child, val = res
            if val < best_val - MOVE_TOL * max(1.0, abs(best_val)):
                best = (child, val)
                best_val = val
        ctx.t_exact += time.perf_counter() - t1
        return best
    cands = []
    for mv in moves:
        ctx.n_exact += 1
        res = _exact_move(plan, mv, adapter, budget, ctx, False, fast=True)
        if res is None:
            continue
        child, val = res
        if val < cur_val - tol:
            cands.append((val, mv, child))
    cands.sort(key=lambda c: c[0])
    best = None
    for _, mv, child in cands:
        res = _confirm(child, (mv[1], mv[2]), adapter, budget, ctx)
        if res is not None and res[1] < cur_val - tol:
            best = res
            break
    ctx.t_exact += time.perf_counter() - t1
    return best


_COST = CostAdapter()


def _repair(plan, budget, ctx):
    """Cost-only descent until sum L <= budget; None if impossible."""
    ones = np.ones(ctx.m)
    zeros = np.zeros(ctx.m)
    steps = 0
    while plan.L.sum() > budget + FEAS_TOL:
        if steps >= REPAIR_STEPS:
            return None
        res = _improve_step(plan, _COST, np.inf, float(plan.L.sum()), ctx, ones, zeros)
        if res is None:
            return None
        plan = res[0]
        steps += 1
    return plan


# ---------------------------------------------------------------------------
# Construction, regret insertion and ruin-and-recreate
# ---------------------------------------------------------------------------

def _regret_insert(seqs, todo, varpi, psi, C, D, dem, Q, v, extra):
    """Regret-2 insertion of `todo` into the sequences (capacity-feasible first).

    Insertion score of i into route b: varpi_b dL + psi_b dsigma + extra dL,
    with sigma = v sqrt(W) and W built from the noise weights D (D^2 per arc).
    A customer with no feasible route goes to the route of largest residual
    capacity and the result is flagged as violated.
    """
    m = len(seqs)
    seqs = [list(s) for s in seqs]
    D2 = D * D
    loads = [int(dem[np.asarray(s, dtype=np.int64)].sum()) for s in seqs]
    W = [tour_stats(s, C, D)[1] for s in seqs]
    cache = [dict() for _ in range(m)]
    touched = set()
    violated = False
    remaining = [int(x) for x in todo]

    def info(b, i):
        hit = cache[b].get(i)
        if hit is None:
            full = np.array([0] + seqs[b] + [0], dtype=np.int64)
            X = full[:-1]
            Y = full[1:]
            dl = C[X, i] + C[i, Y] - C[X, Y]
            e = int(np.argmin(dl))
            x = X[e]
            y = Y[e]
            dw = float(D2[x, i] + D2[i, y] - D2[x, y])
            dLb = float(dl[e])
            dsig = v * (math.sqrt(max(W[b] + dw, 0.0)) - math.sqrt(W[b]))
            score = float(varpi[b]) * dLb + float(psi[b]) * dsig + extra * dLb
            hit = (score, e)
            cache[b][i] = hit
        return hit

    while remaining:
        best_key = None
        pick = None
        for i in remaining:
            qi = int(dem[i])
            scored = [(info(b, i)[0], b) for b in range(m) if loads[b] + qi <= Q]
            if not scored:
                key = (math.inf, 0.0)
                b_choice = int(np.argmax([Q - x for x in loads]))
            else:
                scored.sort()
                regret = scored[1][0] - scored[0][0] if len(scored) > 1 else math.inf
                key = (regret, -scored[0][0])
                b_choice = scored[0][1]
            if best_key is None or key > best_key:
                best_key = key
                pick = (i, b_choice)
        i, b = pick
        qi = int(dem[i])
        if loads[b] + qi > Q:
            violated = True
        e = info(b, i)[1]
        seqs[b].insert(e, i)
        loads[b] += qi
        W[b] = tour_stats(seqs[b], C, D)[1]
        cache[b] = {}
        touched.add(b)
        remaining.remove(i)
    return seqs, violated, touched


def construct_initial(inst, rng):
    """Random-angle sweep seeds plus regret insertion into m nonempty feasible routes."""
    xy = np.asarray(inst.coords, dtype=np.float64)
    N = xy.shape[0]
    m = int(inst.m)
    C = np.ascontiguousarray(inst.C, dtype=np.float64)
    D = np.ascontiguousarray(noise_matrix(inst), dtype=np.float64)
    dem = np.asarray(inst.demand, dtype=np.int64)
    Q = int(inst.Q)
    cust = np.arange(1, N)
    ang = np.arctan2(xy[1:, 1] - xy[0, 1], xy[1:, 0] - xy[0, 0])
    ones = np.ones(m)
    zeros = np.zeros(m)
    for _ in range(500):
        phi = rng.uniform(0.0, 2.0 * math.pi)
        rel = np.mod(ang - phi, 2.0 * math.pi)
        order = cust[np.argsort(rel, kind="stable")]
        pos = (np.arange(m) * len(order)) // m
        seeds = [int(s) for s in order[pos]]
        seed_set = set(seeds)
        todo = [int(c) for c in order if int(c) not in seed_set]
        seqs = [[s] for s in seeds]
        seqs, violated, _ = _regret_insert(seqs, todo, ones, zeros, C, D, dem, Q, 0.0, 1e-3)
        if violated:
            continue
        if all(int(dem[np.asarray(s, dtype=np.int64)].sum()) <= Q for s in seqs):
            return seqs
    raise RuntimeError("could not construct a capacity-feasible initial plan")


def _perturb(plan, varpi, psi, adapter, budget, ctx):
    """Ruin (price- or uniform-route, related or random) and regret-2 recreate.

    Returns a budget-feasible child plan, or None if discarded.
    """
    rng = ctx.rng
    m = ctx.m
    n = ctx.n
    N = ctx.N
    C = ctx.C
    seqs = [list(s) for s in plan.seqs]
    sizes = [len(s) for s in seqs]
    route_of = np.empty(N, dtype=np.int64)
    for r, s in enumerate(seqs):
        route_of[np.asarray(s, dtype=np.int64)] = r
    qmax = max(3, min(RUIN_MAX, int(math.ceil(0.25 * n))))
    q = int(rng.integers(2, qmax + 1))
    price_branch = rng.random() < 0.5
    related = rng.random() < 0.5
    if price_branch:
        w = np.abs(varpi) + 1e-9
        order = rng.choice(m, size=m, replace=False, p=w / w.sum())
        pool = []
        for r in order:
            pool.extend(seqs[int(r)])
            if len(pool) >= q:
                break
        pool = np.asarray(pool, dtype=np.int64)
    else:
        pool = np.arange(1, N, dtype=np.int64)
    if related:
        seed = int(pool[rng.integers(pool.size)])
        ordr = pool[np.argsort(C[seed, pool], kind="stable")]
    else:
        ordr = pool[rng.permutation(pool.size)]
    removed = []
    touched = set()
    for c in ordr:
        if len(removed) >= q:
            break
        c = int(c)
        r = int(route_of[c])
        if sizes[r] <= 1:
            continue
        sizes[r] -= 1
        removed.append(c)
        touched.add(r)
    rem = set(removed)
    for r in touched:
        seqs[r] = [x for x in seqs[r] if x not in rem]
    seqs, violated, touched2 = _regret_insert(
        seqs, removed, varpi, psi, C, ctx.D, ctx.demand, ctx.Q, ctx.v, 1e-3)
    if violated:
        return None
    changed = sorted(touched | touched2)
    L = plan.L.copy()
    W = plan.W.copy()
    for r in changed:
        if ctx.free:
            t, Lr, Wr = ctx.solver.solve(seqs[r])
        else:
            t, Lr, Wr = ctx.solver.solve_fast(seqs[r], hint=plan.seqs[r])
        seqs[r] = list(t)
        L[r] = Lr
        W[r] = Wr
    changed_set = set(changed)
    pre = [None if k in changed_set else plan.pre[k] for k in range(m)]
    child = _Plan(seqs, L, W, pre, ctx)
    child = _repair(child, budget, ctx)
    if child is None:
        return None
    if ctx.free:
        seqs2, L2, W2 = _free_descent(child.seqs, child.L, child.W, changed, adapter, budget, ctx)
        pre2 = list(child.pre)
        for k in changed:
            pre2[k] = None
        child = _Plan(seqs2, L2, W2, pre2, ctx)
    else:
        diff = [k for k in range(m) if tuple(child.seqs[k]) != tuple(plan.seqs[k])]
        res = _confirm(child, diff, adapter, budget, ctx)
        if res is None:
            return None
        child = res[0]
    return child


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def solve_fair(inst, start_routes, mode, measure, v, kappa, budget, time_limit, seed,
               screen="price", consistency="lower", scen=None, max_iters=None, K=8,
               solver=None, z=1.0, saa_S=None):
    """Upper-level search for the SORTIE problem.

    budget is an absolute bound on sum L (use np.inf for none). kappa is the
    lower-level risk weight of TourSolver. consistency "lower" always re-solves
    tours with the lower level; "free" additionally re-sequences changed routes
    by intra-route 2-opt/Or-opt on the global objective. screen "price" takes
    the K best first-order predictions, "random" takes K random feasible moves.
    z is the risk weight of mode detsig; saa_S the SAA sample size when mode is
    saa and no scen tensor is given (default SCEN_S).
    Returns a dict with routes, value, history [(elapsed, best value)], iters,
    n_exact_evals, solver counters and phase timings.
    """
    t_start = time.perf_counter()
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if consistency not in ("lower", "free"):
        raise ValueError("consistency must be 'lower' or 'free'")
    if screen not in ("price", "random"):
        raise ValueError("screen must be 'price' or 'random'")
    C = np.ascontiguousarray(inst.C, dtype=np.float64)
    N = C.shape[0]
    m = int(inst.m)
    if m < 2:
        raise ValueError("the search needs m >= 2")
    if mode != "cost" and measure == "mad" and m % 2 == 0:
        raise ValueError("mad needs odd m")
    cover = sorted(int(x) for r in start_routes for x in r)
    if len(start_routes) != m or any(len(r) == 0 for r in start_routes) or cover != list(range(1, N)):
        raise ValueError("start routes must be m nonempty tours covering every customer once")
    budget = float(budget)
    H = effective_H(inst)
    D = noise_matrix(inst)
    if solver is None:
        solver = TourSolver(C, kappa, v, D=D)
    snap = (solver.n_calls, solver.n_hk, solver.n_cache_hits, solver.n_ls,
            solver.n_fast, solver.n_fast_hits)
    S_saa = SCEN_S if saa_S is None else int(saa_S)
    adapter = make_objective(mode, measure, v, scen=scen, C=C, H=H, S=S_saa, z=z)
    rng = np.random.default_rng(seed)
    deadline = t_start + float(time_limit)
    ctx = _Ctx(inst, solver, consistency, v, rng, K, screen, deadline)

    if consistency == "lower":
        seqs = [list(solver.solve(r)[0]) for r in start_routes]
    else:
        seqs = [[int(x) for x in r] for r in start_routes]
    plan = _build_plan(seqs, ctx)
    if consistency == "free":
        seqs, L, W = _free_descent(plan.seqs, plan.L, plan.W, range(m), adapter, budget, ctx)
        plan = _Plan(seqs, L, W, [None] * m, ctx)
    plan = _repair(plan, budget, ctx)
    if plan is None:
        raise ValueError("start plan cannot be made budget-feasible")

    cur = plan
    cur_val = adapter.value(cur.L, cur.W, cur.seqs)
    best = cur
    best_val = cur_val
    history = [(0.0, best_val)]
    # Common temperature scale for every mode: 2% of the model-based ex-post
    # value of the start plan (cost mode: of its total length), so det, saa and
    # pboi runs accept uphill moves of the same absolute size.
    if mode == "cost":
        scale = abs(cur_val)
    else:
        scale = abs(make_objective("pboi", measure, v, C=C).value(cur.L, cur.W, cur.seqs))
    T0 = 0.02 * scale + 1e-9
    Tend = T0 * 1e-3
    it = 0
    n_improve = 0
    n_pert = 0
    n_acc = 0
    varpi = psi = None
    prices_for = None
    stuck = None
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break
        if max_iters is not None and it >= max_iters:
            break
        it += 1
        if prices_for is not cur:
            varpi, psi = adapter.prices(cur.L, cur.W, cur.seqs)
            prices_for = cur
        if screen == "price" and stuck is cur:
            res = None
        else:
            res = _improve_step(cur, adapter, budget, cur_val, ctx, varpi, psi)
            if res is None:
                stuck = cur
        if res is not None:
            cur, cur_val = res
            n_improve += 1
            if cur_val < best_val - MOVE_TOL * max(1.0, abs(best_val)):
                best, best_val = cur, cur_val
                history.append((time.perf_counter() - t_start, best_val))
            continue
        t_p = time.perf_counter()
        cand = _perturb(cur, varpi, psi, adapter, budget, ctx)
        ctx.t_perturb += time.perf_counter() - t_p
        n_pert += 1
        if cand is None:
            continue
        cand_val = adapter.value(cand.L, cand.W, cand.seqs)
        if max_iters is not None:
            progress = it / float(max_iters)
        else:
            progress = elapsed / float(time_limit)
        progress = min(max(progress, 0.0), 1.0)
        T = T0 * (Tend / T0) ** progress
        delta = cand_val - cur_val
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            cur, cur_val = cand, cand_val
            n_acc += 1
            if cur_val < best_val - MOVE_TOL * max(1.0, abs(best_val)):
                best, best_val = cur, cur_val
                history.append((time.perf_counter() - t_start, best_val))

    total = time.perf_counter() - t_start
    return {
        "routes": [list(s) for s in best.seqs],
        "value": best_val,
        "sumL": float(best.L.sum()),
        "L": best.L.tolist(),
        "W": best.W.tolist(),
        "history": history,
        "iters": it,
        "n_exact_evals": ctx.n_exact,
        "n_confirm": ctx.n_confirm,
        "n_improve": n_improve,
        "n_perturb": n_pert,
        "n_accept": n_acc,
        "solver": {
            "n_calls": solver.n_calls - snap[0],
            "n_hk": solver.n_hk - snap[1],
            "n_cache_hits": solver.n_cache_hits - snap[2],
            "n_ls": solver.n_ls - snap[3],
            "n_fast": solver.n_fast - snap[4],
            "n_fast_hits": solver.n_fast_hits - snap[5],
        },
        "time": total,
        "time_screen": ctx.t_screen,
        "time_exact": ctx.t_exact,
        "time_perturb": ctx.t_perturb,
        "budget": budget,
        "mode": mode,
        "measure": measure,
        "v": float(v),
        "kappa": float(kappa),
        "consistency": consistency,
        "screen": screen,
    }


def solve_cost(inst, time_limit, seed, restarts=3):
    """Minimum total expected length plan (mode cost, kappa 0), best of restarts.

    Returns (routes, C*) with C* = sum of tour lengths.
    """
    rng = np.random.default_rng(seed)
    best_routes = None
    best_c = math.inf
    per = float(time_limit) / max(1, int(restarts))
    for _ in range(max(1, int(restarts))):
        start = construct_initial(inst, rng)
        sub_seed = int(rng.integers(0, 2 ** 31 - 1))
        res = solve_fair(inst, start, "cost", "max", 0.0, 0.0, np.inf, per, sub_seed,
                         consistency="lower")
        if res["sumL"] < best_c - 1e-12:
            best_c = res["sumL"]
            best_routes = res["routes"]
    return best_routes, float(best_c)


# ---------------------------------------------------------------------------
# Move diagnostics (predicted versus exact change of every feasible move)
# ---------------------------------------------------------------------------

def move_diagnostics(inst, routes, mode, measure, v, kappa, budget, max_moves=None,
                     seed=0, z=1.0):
    """Predicted and exact change of the feasible relocate and swap moves of one plan.

    Consistency is "lower". The routes are first solved with the exact lower
    level and the prices of that plan are computed. Every move the screen accepts
    (capacity, and sum L within the screen's budget threshold) gets its
    first-order prediction (the one _screen ranks by). For all such moves, or a
    uniform random subset of max_moves of them drawn with rng(seed), the exact
    change of the objective is computed by re-solving the two changed routes with
    the exact lower level. A move whose re-solved plan exceeds the budget has
    exact = NaN.

    Returns a pandas DataFrame with columns kind ("relocate" or "swap"), a, b
    (route indices), i, j (customers; j = -1 for a relocate), predicted and exact
    (changes of the objective), pred_rank and exact_rank (0 = most decreasing
    change; exact_rank is NaN where exact is).
    """
    import pandas as pd

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if measure not in MEASURES:
        raise ValueError(f"unknown measure {measure!r}")
    C = np.ascontiguousarray(inst.C, dtype=np.float64)
    N = C.shape[0]
    m = int(inst.m)
    if m < 2:
        raise ValueError("the diagnostics need m >= 2")
    if mode != "cost" and measure == "mad" and m % 2 == 0:
        raise ValueError("mad needs odd m")
    cover = sorted(int(x) for r in routes for x in r)
    if len(routes) != m or any(len(r) == 0 for r in routes) or cover != list(range(1, N)):
        raise ValueError("routes must be m nonempty tours covering every customer once")
    budget = float(budget)
    H = effective_H(inst)
    D = noise_matrix(inst)
    solver = TourSolver(C, kappa, v, D=D)
    adapter = make_objective(mode, measure, v, C=C, H=H, z=z)
    ctx = _Ctx(inst, solver, "lower", v, np.random.default_rng(seed), 1, "price", np.inf)

    seqs = [list(solver.solve(r)[0]) for r in routes]
    plan = _build_plan(seqs, ctx)
    if plan.L.sum() > budget + FEAS_TOL:
        raise ValueError("the solved routes exceed the budget")
    cur_val = adapter.value(plan.L, plan.W, plan.seqs)
    varpi, psi = adapter.prices(plan.L, plan.W, plan.seqs)
    moves, preds = _feasible_moves(plan, varpi, psi, budget, ctx)

    total = len(moves)
    if max_moves is not None and int(max_moves) < total:
        pick = np.sort(np.random.default_rng(seed).choice(total, size=int(max_moves),
                                                          replace=False))
    else:
        pick = np.arange(total)
    rows = []
    for g in pick:
        mv = moves[int(g)]
        seqs_n, L_n, W_n = _resolve_move(plan, mv, ctx, free=False)
        if L_n.sum() > budget + FEAS_TOL:
            exact = math.nan
        else:
            exact = adapter.value(L_n, W_n, seqs_n) - cur_val
        kind = "relocate" if mv[0] == 0 else "swap"
        rows.append((kind, mv[1], mv[2], mv[3], mv[4], float(preds[int(g)]), float(exact)))
    df = pd.DataFrame(rows, columns=["kind", "a", "b", "i", "j", "predicted", "exact"])
    df["pred_rank"] = (df["predicted"].rank(method="first") - 1).astype("int64")
    df["exact_rank"] = df["exact"].rank(method="first") - 1
    return df

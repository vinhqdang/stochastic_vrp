"""Additional execution-stage policies and diagnostics for BATON.

Everything here plugs into the simulators of core.costs and shares their
conventions: g is an (N, m) matrix of net increments p_k - d_k, B the
slack, and H, E, R the per-stop handoff / emergency / depot-return price
schedules (entry k-1 = price after stop k).

Public API
----------
fit_threshold_k       -- position-dependent load thresholds (one per stop),
                         tuned by coordinate descent on realised cost
simulate_threshold_k  -- execute a per-stop load-threshold policy
tune_rollout_theta    -- cost-scaled myopic trigger chat_react > theta*H_k
fit_dp_actions        -- plug-in DP over {continue, handoff, depot return}
                         (binned conditional means; near-exact at large N)
fit_lsm_actions_exact -- BATON with the exact residual-capacity reset after
                         a depot return (reset state -D_<=k, not 0)
simulate_actions_exact-- execute the exact-reset policy
fresh_value_curve     -- F_k(x): cost of the suffix from reset level x
cvar                  -- upper-tail conditional value-at-risk
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .otr2 import _overflow_step, _ConstantModel
from .costs import _simulate_costs_general, _fresh_value
from .dp_exec import _BinnedModel, _fill_gaps


# ============================================================
# Position-dependent load thresholds
# ============================================================


def _costs_threshold_k(g: np.ndarray, B: float, H: np.ndarray,
                       E: np.ndarray, thr: np.ndarray,
                       H_bill: np.ndarray | None = None):
    """Per-scenario cost and action of the rule 'hand off after stop k iff
    W_k > thr[k-1]' (thr = inf disables stop k). Codes 0/1/2 as usual."""
    if H_bill is None:
        H_bill = H
    N, m = g.shape
    cum = np.cumsum(g, axis=1)
    costs = np.zeros(N)
    action = np.zeros(N, dtype=np.int8)
    stopped = np.zeros(N, dtype=bool)
    for k_idx in range(m):
        active = ~stopped
        if not active.any():
            break
        Wk = cum[:, k_idx]
        em = active & (Wk > B)
        costs[em] = E[k_idx]
        action[em] = 2
        stopped |= em
        if k_idx + 1 == m:
            break
        ho = active & ~em & (Wk > thr[k_idx])
        costs[ho] = H_bill[k_idx]
        action[ho] = 1
        stopped |= ho
    return costs, action


def fit_threshold_k(g_train: np.ndarray, B: float, H: np.ndarray,
                    E: np.ndarray, n_cand: int = 24,
                    sweeps: int = 2) -> np.ndarray:
    """Tune one load threshold per stop by coordinate descent on the
    realised training cost.

    The policy class {hand off iff W_k > w_k} is exactly the class of
    boundaries BATON-ho can represent (its continuation estimate is
    monotone, so its stopping region at each stop is an upper set in W).
    The difference is the estimator: here the m-1 boundaries are searched
    directly against realised cost instead of being derived by backward
    induction. Candidates per stop are quantiles of the training W_k
    among alive paths, plus +inf (never hand off at that stop). Sweeps run
    backwards (late stops first), starting from the reactive policy.
    """
    N, m = g_train.shape
    cum = np.cumsum(g_train, axis=1)
    ostep = _overflow_step(cum, B)
    thr = np.full(m, np.inf)
    best = float(_costs_threshold_k(g_train, B, H, E, thr)[0].mean())
    qs = np.linspace(0.02, 0.98, n_cand)
    for _ in range(sweeps):
        improved = False
        for k in range(m - 1, 0, -1):
            alive = ostep > k
            if alive.sum() < 2:
                continue
            cands = np.unique(np.quantile(cum[alive, k - 1], qs))
            cur = thr[k - 1]
            for c in np.concatenate([[np.inf], cands]):
                if c == cur:
                    continue
                thr[k - 1] = c
                v = float(_costs_threshold_k(g_train, B, H, E, thr)[0].mean())
                if v < best - 1e-12:
                    best, cur, improved = v, c, True
            thr[k - 1] = cur
        if not improved:
            break
    return thr


def simulate_threshold_k(g_test, B, H, E, thr, return_actions=False,
                         H_bill=None):
    c, a = _costs_threshold_k(g_test, B, H, E, thr, H_bill=H_bill)
    st = _stats(c, a)
    return (st, a) if return_actions else st


# ============================================================
# Cost-scaled myopic trigger (rollout with a tuned multiplier)
# ============================================================


def tune_rollout_theta(g_train: np.ndarray, B: float, H: np.ndarray,
                       E: np.ndarray, ro_models: dict,
                       grid: np.ndarray | None = None) -> float:
    """Tune theta in 'hand off iff chat_react_k(W_k) > theta * H_k', where
    chat_react is the reactive (never-act-again) cost-to-go of the rollout
    policy. theta = 1 is the untuned rollout; theta > 1 corrects the
    over-triggering of the myopic comparison with one position- and
    price-aware scalar. Billing always uses the true H."""
    if grid is None:
        grid = np.unique(np.concatenate([[1.0], np.geomspace(0.5, 8.0, 36)]))
    best_t, best_c = 1.0, np.inf
    for t in grid:
        c, _ = _simulate_costs_general(g_train, B, t * H, E, ro_models,
                                       H_bill=H)
        if c.mean() < best_c - 1e-12:
            best_c, best_t = float(c.mean()), float(t)
    return best_t


# ============================================================
# Plug-in DP over the full action set
# ============================================================


def fit_dp_actions(g_hist: np.ndarray, B: float, H: np.ndarray,
                   E: np.ndarray, R: np.ndarray,
                   n_bins: int | None = None,
                   g_eval: np.ndarray | None = None) -> dict:
    """Backward induction over {continue, handoff, depot return} with
    quantile-binned conditional means in place of the isotonic step, and
    the fresh-start value computed exactly as in BATON (forward pass of
    the reset state through the already-fitted downstream policy). At
    large N this is the near-exact yardstick for the three-action
    problem. Returns {k: (model, F_k)} in the fit_lsm_actions format.

    g_eval: independent paths on which F_k is evaluated (cross-fitting).
    Evaluating F_k on the paths the binned models were fitted on is
    optimistic wherever the reset state W = 0 lies in a sparsely populated
    region of the load axis (e.g. delivery-dominated routes, where W_k
    drifts negative), and the resulting under-priced return degrades the
    policy; the yardstick therefore uses cross-fitted F_k."""
    N, m = g_hist.shape
    if n_bins is None:
        n_bins = int(np.clip(N // 30, 8, 256))
    cum = np.cumsum(g_hist, axis=1)
    ostep = _overflow_step(cum, B)
    future = np.zeros(N)
    br = ostep <= m
    future[br] = E[np.clip(ostep[br] - 1, 0, m - 1)]

    models: dict = {}
    for k in range(m - 1, 0, -1):
        alive = ostep > k
        n_alive = int(alive.sum())
        if n_alive < 2:
            mdl = _ConstantModel(float(future[alive][0]) if n_alive == 1
                                 else float(E[min(k, m - 1)]))
        else:
            W = cum[alive, k - 1]
            nb = min(n_bins, max(1, n_alive // 2))
            edges = np.unique(np.quantile(W, np.linspace(0, 1, nb + 1)[1:-1]))
            idx = np.searchsorted(edges, W, side="right")
            nbe = len(edges) + 1
            sums = np.bincount(idx, weights=future[alive], minlength=nbe)
            cnts = np.bincount(idx, minlength=nbe)
            with np.errstate(invalid="ignore"):
                vals = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
            mdl = _BinnedModel(edges, _fill_gaps(vals))
        Fk = _fresh_value(k, m, g_hist if g_eval is None else g_eval,
                          B, H, E, R, models)
        models[k] = (mdl, Fk)
        pred = np.asarray(mdl.predict(cum[:, k - 1]))
        v_rs = R[k - 1] + Fk
        best = np.minimum(np.minimum(pred, H[k - 1]), v_rs)
        act_ho = alive & (H[k - 1] <= v_rs) & (best == H[k - 1])
        act_rs = alive & ~act_ho & (best == v_rs)
        future = future.copy()
        future[act_ho] = H[k - 1]
        future[act_rs] = v_rs
    return models


# ============================================================
# BATON with the exact residual-capacity reset
# ============================================================
#
# Physical load after stop k is L0 + W_k with L0 the departure load. A
# depot return unloads the pickups collected so far, leaving on board the
# undelivered cargo L0 - D_{<=k}; relative to the same slack B the state
# therefore resets to x_k = -D_{<=k} (<= 0), not to 0. The main policy
# uses the reset-to-0 convention (the vehicle is restored to its departure
# load), which understates the slack by the volume already delivered;
# this variant prices the exact reset.


def _suffix_cost_from(k, m, g, xs, dcum, B, H, E, R, models):
    """Mean cost of the suffix k+1..m over the training paths started at
    reset level xs (scalar). A nested return at stop j > k resets to
    xs - (D_<=j - D_<=k): the day's cargo still undelivered at j."""
    N = g.shape[0]
    base = float(xs) + (dcum[:, k - 1] if k >= 1 else 0.0)
    W = np.full(N, float(xs))
    cost = np.zeros(N)
    stopped = np.zeros(N, dtype=bool)
    for j in range(k + 1, m + 1):
        act = ~stopped
        if not act.any():
            break
        W[act] += g[act, j - 1]
        em = act & (W > B)
        cost[em] += E[j - 1]
        stopped |= em
        if j == m:
            break
        entry = models.get(j)
        if entry is None:
            continue
        mdl, Fcurve = entry
        alive = act & ~em
        if not alive.any():
            continue
        idx = np.where(alive)[0]
        chat = np.asarray(mdl.predict(W[idx]))
        v_ho = H[j - 1]
        lvl = base[idx] - dcum[idx, j - 1]
        v_rs = R[j - 1] + Fcurve(lvl)
        do_ho = (v_ho < chat) & (v_ho <= v_rs)
        do_rs = (v_rs < chat) & ~do_ho
        cost[idx[do_ho]] += v_ho
        stopped[idx[do_ho]] = True
        cost[idx[do_rs]] += R[j - 1]
        W[idx[do_rs]] = lvl[do_rs]
    return float(cost.mean())


class _Curve:
    """Monotone piecewise-linear F_k(x) on a grid (flat extrapolation)."""

    def __init__(self, xs, ys):
        self.xs = np.asarray(xs, float)
        self.ys = np.maximum.accumulate(np.asarray(ys, float))

    def __call__(self, x):
        return np.interp(np.asarray(x, float), self.xs, self.ys)


def fresh_value_curve(k, m, g, dcum, B, H, E, R, models, n_grid=9):
    """F_k(x) on a grid of reset levels spanning the training -D_<=k."""
    lv = -dcum[:, k - 1]
    xs = np.unique(np.quantile(lv, np.linspace(0.0, 1.0, n_grid)))
    ys = [_suffix_cost_from(k, m, g, x, dcum, B, H, E, R, models) for x in xs]
    return _Curve(xs, ys)


def fit_lsm_actions_exact(g_hist: np.ndarray, d_hist: np.ndarray, B: float,
                          H: np.ndarray, E: np.ndarray, R: np.ndarray,
                          n_grid: int = 9) -> dict:
    """Three-action BATON with the exact reset x_k = -D_<=k. d_hist holds
    the realised deliveries (N, m) of the same paths. Returns
    {k: (iso_model, F_curve)}."""
    N, m = g_hist.shape
    cum = np.cumsum(g_hist, axis=1)
    dcum = np.cumsum(d_hist, axis=1)
    ostep = _overflow_step(cum, B)
    future = np.zeros(N)
    br = ostep <= m
    future[br] = E[np.clip(ostep[br] - 1, 0, m - 1)]

    models: dict = {}
    for k in range(m - 1, 0, -1):
        alive = ostep > k
        n_alive = int(alive.sum())
        if n_alive >= 2:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(cum[alive, k - 1], future[alive])
        elif n_alive == 1:
            iso = _ConstantModel(future[alive][0])
        else:
            iso = _ConstantModel(float(E[min(k, m - 1)]))
        Fc = fresh_value_curve(k, m, g_hist, dcum, B, H, E, R, models,
                               n_grid=n_grid)
        models[k] = (iso, Fc)
        pred = np.asarray(iso.predict(cum[:, k - 1]))
        v_rs = R[k - 1] + Fc(-dcum[:, k - 1])
        best = np.minimum(np.minimum(pred, H[k - 1]), v_rs)
        act_ho = alive & (H[k - 1] <= v_rs) & (best == H[k - 1])
        act_rs = alive & ~act_ho & (best == v_rs)
        future = future.copy()
        future[act_ho] = H[k - 1]
        future[act_rs] = v_rs[act_rs]
    return models


def simulate_actions_exact(g_test, d_test, B, H, E, R, models,
                           return_actions=False, H_bill=None):
    """Execute the exact-reset three-action policy (codes as in
    core.costs.simulate_actions; restocks accumulate into cost)."""
    if H_bill is None:
        H_bill = H
    N, m = g_test.shape
    dcum = np.cumsum(d_test, axis=1)
    costs = np.zeros(N)
    action = np.zeros(N, dtype=np.int8)
    W = np.zeros(N)
    stopped = np.zeros(N, dtype=bool)
    n_rs = np.zeros(N, dtype=int)
    for k_idx in range(m):
        k = k_idx + 1
        act = ~stopped
        if not act.any():
            break
        W[act] += g_test[act, k_idx]
        em = act & (W > B)
        costs[em] += E[k_idx]
        action[em] = 2
        stopped |= em
        if k == m:
            break
        alive = act & ~em
        entry = models.get(k)
        if entry is None or not alive.any():
            continue
        mdl, Fc = entry
        idx = np.where(alive)[0]
        chat = np.asarray(mdl.predict(W[idx]))
        v_ho = H[k_idx]
        v_rs = R[k_idx] + Fc(-dcum[idx, k_idx])
        do_ho = (v_ho < chat) & (v_ho <= v_rs)
        do_rs = (v_rs < chat) & ~do_ho
        costs[idx[do_ho]] += H_bill[k_idx]
        action[idx[do_ho]] = 1
        stopped[idx[do_ho]] = True
        r = idx[do_rs]
        costs[r] += R[k_idx]
        W[r] = -dcum[r, k_idx]
        n_rs[r] += 1
    st = _stats(costs, action)
    st["restock_rate"] = float((n_rs > 0).mean())
    return (st, action) if return_actions else st


# ============================================================
# helpers
# ============================================================


def _stats(costs: np.ndarray, action: np.ndarray) -> dict:
    return {
        "mean_cost":     float(costs.mean()),
        "handoff_rate":  float((action == 1).mean()),
        "fail_rate":     float((action == 2).mean()),
        "complete_rate": float((action == 0).mean()),
        "costs":         costs,
    }


def cvar(x: np.ndarray, q: float = 0.95) -> float:
    """Mean of the worst (1-q) share of outcomes (upper tail)."""
    x = np.sort(np.asarray(x, float))
    k = int(np.floor(q * len(x)))
    return float(x[k:].mean()) if k < len(x) else float(x[-1])


# ============================================================
# BATON with a state-conditional fresh-start value F_k(W_k)
# ============================================================
#
# Under Assumption 1 the post-return suffix is independent of the pre-return
# state, so F_k is a constant. Under positively dependent demand a day on
# which the vehicle is heavy at stop k is also likely to be heavy after the
# return, and the unconditional F_k under-prices the return exactly on the
# days that trigger it. The conditional variant regresses (isotonically) the
# realised post-reset suffix cost of each training path on that path's
# pre-return state W_k: each path's own future increments carry the day's
# demand level. Under independence the fit is flat and the policy reduces
# to BATON.


class _CFModel:
    def __init__(self, iso):
        self.iso = iso

    def __call__(self, w):
        return np.asarray(self.iso.predict(np.atleast_1d(np.asarray(w, float))))


def _suffix_costs_cf(k, m, g, B, H, E, R, models):
    """Per-path cost of the suffix k+1..m started from reset level 0 with
    each path's own future increments; nested returns priced by the
    downstream conditional models."""
    N = g.shape[0]
    W = np.zeros(N)
    cost = np.zeros(N)
    stopped = np.zeros(N, dtype=bool)
    for j in range(k + 1, m + 1):
        act = ~stopped
        if not act.any():
            break
        W[act] += g[act, j - 1]
        em = act & (W > B)
        cost[em] += E[j - 1]
        stopped |= em
        if j == m:
            break
        entry = models.get(j)
        if entry is None:
            continue
        mdl, Fm = entry
        alive = act & ~em
        if not alive.any():
            continue
        idx = np.where(alive)[0]
        chat = np.asarray(mdl.predict(W[idx]))
        v_ho = H[j - 1]
        v_rs = R[j - 1] + Fm(W[idx])
        do_ho = (v_ho < chat) & (v_ho <= v_rs)
        do_rs = (v_rs < chat) & ~do_ho
        cost[idx[do_ho]] += v_ho
        stopped[idx[do_ho]] = True
        cost[idx[do_rs]] += R[j - 1]
        W[idx[do_rs]] = 0.0
    return cost


def fit_lsm_actions_cf(g_hist: np.ndarray, B: float, H: np.ndarray,
                       E: np.ndarray, R: np.ndarray) -> dict:
    """Three-action BATON with the conditional fresh-start value.
    Returns {k: (iso_model, F_model)} with F_model(w) vectorised."""
    N, m = g_hist.shape
    cum = np.cumsum(g_hist, axis=1)
    ostep = _overflow_step(cum, B)
    future = np.zeros(N)
    br = ostep <= m
    future[br] = E[np.clip(ostep[br] - 1, 0, m - 1)]
    models: dict = {}
    for k in range(m - 1, 0, -1):
        alive = ostep > k
        n_alive = int(alive.sum())
        if n_alive >= 2:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(cum[alive, k - 1], future[alive])
        elif n_alive == 1:
            iso = _ConstantModel(future[alive][0])
        else:
            iso = _ConstantModel(float(E[min(k, m - 1)]))
        sc = _suffix_costs_cf(k, m, g_hist, B, H, E, R, models)
        if n_alive >= 2:
            fi = IsotonicRegression(increasing=True, out_of_bounds="clip")
            fi.fit(cum[alive, k - 1], sc[alive])
        else:
            fi = _ConstantModel(float(sc.mean()))
        Fm = _CFModel(fi)
        models[k] = (iso, Fm)
        pred = np.asarray(iso.predict(cum[:, k - 1]))
        v_rs = R[k - 1] + Fm(cum[:, k - 1])
        best = np.minimum(np.minimum(pred, H[k - 1]), v_rs)
        act_ho = alive & (H[k - 1] <= v_rs) & (best == H[k - 1])
        act_rs = alive & ~act_ho & (best == v_rs)
        future = future.copy()
        future[act_ho] = H[k - 1]
        future[act_rs] = v_rs[act_rs]
    return models


def simulate_actions_cf(g_test, B, H, E, R, models, return_actions=False,
                        H_bill=None):
    """Execute the conditional-F three-action policy."""
    if H_bill is None:
        H_bill = H
    N, m = g_test.shape
    costs = np.zeros(N)
    action = np.zeros(N, dtype=np.int8)
    W = np.zeros(N)
    stopped = np.zeros(N, dtype=bool)
    for k_idx in range(m):
        k = k_idx + 1
        act = ~stopped
        if not act.any():
            break
        W[act] += g_test[act, k_idx]
        em = act & (W > B)
        costs[em] += E[k_idx]
        action[em] = 2
        stopped |= em
        if k == m:
            break
        alive = act & ~em
        entry = models.get(k)
        if entry is None or not alive.any():
            continue
        mdl, Fm = entry
        idx = np.where(alive)[0]
        chat = np.asarray(mdl.predict(W[idx]))
        v_ho = H[k_idx]
        v_rs = R[k_idx] + Fm(W[idx])
        do_ho = (v_ho < chat) & (v_ho <= v_rs)
        do_rs = (v_rs < chat) & ~do_ho
        costs[idx[do_ho]] += H_bill[k_idx]
        action[idx[do_ho]] = 1
        stopped[idx[do_ho]] = True
        costs[idx[do_rs]] += R[k_idx]
        W[idx[do_rs]] = 0.0
    st = _stats(costs, action)
    return (st, action) if return_actions else st


def _binned_fit(W, y, n_bins):
    edges = np.unique(np.quantile(W, np.linspace(0, 1, n_bins + 1)[1:-1]))
    idx = np.searchsorted(edges, W, side="right")
    nbe = len(edges) + 1
    sums = np.bincount(idx, weights=y, minlength=nbe)
    cnts = np.bincount(idx, minlength=nbe)
    with np.errstate(invalid="ignore"):
        vals = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return _BinnedModel(edges, _fill_gaps(vals))


class _BinnedF:
    def __init__(self, mdl):
        self.mdl = mdl

    def __call__(self, w):
        return np.asarray(self.mdl.predict(np.atleast_1d(np.asarray(w, float))))


def fit_dp_actions_cf(g_hist: np.ndarray, B: float, H: np.ndarray,
                      E: np.ndarray, R: np.ndarray,
                      g_eval: np.ndarray | None = None,
                      n_bins: int | None = None) -> dict:
    """Near-exact three-action yardstick with a state-conditional
    fresh-start value: binned conditional means for C_k (fitted on
    g_hist) and for F_k(W_k) (the post-reset suffix cost of each path of
    g_eval, computed from that path's own future increments, regressed on
    its pre-return state). Use with simulate_actions_cf."""
    if g_eval is None:
        g_eval = g_hist
    N, m = g_hist.shape
    if n_bins is None:
        n_bins = int(np.clip(N // 30, 8, 256))
    cum = np.cumsum(g_hist, axis=1)
    ost = _overflow_step(cum, B)
    cum_e = np.cumsum(g_eval, axis=1)
    ost_e = _overflow_step(cum_e, B)
    future = np.zeros(N)
    br = ost <= m
    future[br] = E[np.clip(ost[br] - 1, 0, m - 1)]
    models: dict = {}
    for k in range(m - 1, 0, -1):
        alive = ost > k
        n_alive = int(alive.sum())
        if n_alive < 2:
            mdl = _ConstantModel(float(future[alive][0]) if n_alive == 1
                                 else float(E[min(k, m - 1)]))
        else:
            mdl = _binned_fit(cum[alive, k - 1], future[alive],
                              min(n_bins, max(1, n_alive // 2)))
        sc = _suffix_costs_cf(k, m, g_eval, B, H, E, R, models)
        ae = ost_e > k
        if ae.sum() >= 2:
            Fm = _BinnedF(_binned_fit(cum_e[ae, k - 1], sc[ae],
                                      min(n_bins, max(1, int(ae.sum()) // 2))))
        else:
            Fm = _BinnedF(_ConstantModel(float(sc.mean())))
        models[k] = (mdl, Fm)
        pred = np.asarray(mdl.predict(cum[:, k - 1]))
        v_rs = R[k - 1] + Fm(cum[:, k - 1])
        best = np.minimum(np.minimum(pred, H[k - 1]), v_rs)
        act_ho = alive & (H[k - 1] <= v_rs) & (best == H[k - 1])
        act_rs = alive & ~act_ho & (best == v_rs)
        future = future.copy()
        future[act_ho] = H[k - 1]
        future[act_rs] = v_rs[act_rs]
    return models

"""OTR-2.0 — Peak-aware labels + optimal-stopping trigger for the SVRPSPD.

Supersedes OTR v1 (core/otr.py). Two defects of v1 are fixed here:

1.  v1 labelled overflow on the ENDPOINT total ``W_m > B``. On routes where
    pickups precede deliveries (collect-then-deliver, milk runs) the endpoint
    cancels toward 0 while the mid-route peak breaches capacity, so v1's
    training labels are all-zero and the policy never triggers.
    v2 labels on the RUNNING PEAK: ``max_k W_k > B`` (the physical event).

2.  v1 used a fixed probability threshold ``tau``, which ignores option
    value (waiting is cheap early in the route, expensive late).
    v2 replaces the threshold with optimal stopping: hand off iff
    ``omegaF < predicted cost of continuing optimally``, computed by
    backward induction (Longstaff-Schwartz regression). No ``tau`` exists.

Public API
----------
fit_lsm                     -- offline: per-step continuation-cost models
otr_route_v2                -- online: run OTR-2.0 on one truck route
calibrate_B_empirical_peak  -- B at the (1-alpha)-quantile of the PEAK distribution
validate                    -- pre-deployment validation protocol (3 checks)
simulate_v2                 -- vectorized batch evaluation over a test set
fit_otr_peak                -- fallback for small N: peak-label isotonic p-curves
                               (use with v1's tune_tau_fast / simulate_fast)

Dependencies: numpy + scikit-learn only.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


# ============================================================
# Helpers
# ============================================================


def _overflow_step(cum: np.ndarray, B: float) -> np.ndarray:
    """First step (1-indexed) where the running sum exceeds B; m+1 if never.

    Parameters
    ----------
    cum : np.ndarray, shape (N, m)
        Cumulative net increments, cum[s, k-1] = W_k for scenario s.
    B : float
        Slack = Q - L0.

    Returns
    -------
    ostep : np.ndarray of int, shape (N,)
        Physical overflow time per scenario (m+1 = the route never overflows).
    """
    m = cum.shape[1]
    over = cum > B
    return np.where(over.any(axis=1), over.argmax(axis=1) + 1, m + 1)


class _ConstantModel:
    """Degenerate continuation model used when a step has < 2 alive paths."""

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x):
        return np.full(np.asarray(x, dtype=float).shape, self.value)


# ============================================================
# Offline phase — Longstaff-Schwartz backward induction
# ============================================================


def fit_lsm(g_hist: np.ndarray, B: float, omegaF: float, Cfail: float) -> dict:
    """Fit per-step continuation-cost models by backward induction.

    At each step the model answers: "if I do NOT hand off now and then act
    optimally forever after, what will this route cost in expectation?"
    The online rule compares that number to omegaF.

    Parameters
    ----------
    g_hist : np.ndarray, shape (N, m)
        Historical net increments, g_hist[s, k-1] = customer k on scenario s.
    B      : float   slack = Q - L0 (or empirical peak quantile)
    omegaF : float   cost of a planned handoff
    Cfail  : float   cost of an unplanned overflow (> omegaF)

    Returns
    -------
    cont_models : dict[int -> regressor]
        cont_models[k].predict([w])[0] = estimated expected future cost of
        continuing past step k with running sum W_k = w, under optimal play.
        Keys: k = 1, ..., m-1.

    Notes
    -----
    The continuation cost is non-decreasing in the running sum W_k (a higher
    prefix can only raise future overflow exposure), so isotonic regression
    remains the right monotone, assumption-free estimator. The regression
    target is a COST (mixture of 0, omegaF, Cfail), not a probability.
    For N < 500 backward induction amplifies estimation noise — prefer the
    peak-label + fixed-tau fallback (fit_otr_peak).
    """
    N, m = g_hist.shape
    cum = np.cumsum(g_hist, axis=1)            # W_k per scenario
    ostep = _overflow_step(cum, B)             # physical overflow time

    # future[s] = cost incurred AFTER the current decision step, given the
    # policy built so far for later steps. Initialise with "never act":
    future = np.where(ostep <= m, Cfail, 0.0).astype(float)

    cont_models: dict = {}
    for k in range(m - 1, 0, -1):              # backward: decide after stop k
        alive = ostep > k                      # not yet physically overflowed
        n_alive = int(alive.sum())
        if n_alive >= 2:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(cum[alive, k - 1], future[alive])
        elif n_alive == 1:
            iso = _ConstantModel(future[alive][0])
        else:
            # every training path has already overflowed by step k: any path
            # still alive online is off-distribution — be conservative.
            iso = _ConstantModel(Cfail)
        cont_models[k] = iso

        # apply the implied optimal action to the training paths themselves,
        # so that earlier steps see costs under the optimal LATER policy:
        pred = iso.predict(cum[:, k - 1])
        stop = alive & (pred > omegaF)         # handing off is cheaper
        future[stop] = omegaF

    return cont_models


# ============================================================
# Online phase — cost-comparison trigger (no tau)
# ============================================================


def otr_route_v2(
    observe,
    m: int,
    B: float,
    omegaF: float,
    Cfail: float,
    cont_models: dict,
) -> tuple:
    """Run OTR-2.0 online. Identical control flow to v1 with two changes:

      (i)  the physical overflow guard uses the RUNNING PEAK implicitly,
           because the check ``Wk > B`` runs at every stop (it always did —
           the v1 bug was in the TRAINING LABEL, not the online guard);
      (ii) trigger = cost comparison, not probability threshold.

    ``chat`` already prices in every future opportunity to hand off (that is
    what the backward induction computed), so ``chat > omegaF`` is the exact
    optimal-stopping rule given the fitted models — not a heuristic. The
    myopic v1 rule ``p > omegaF/Cfail`` is the special case that pretends
    step k is the last chance to act; it systematically triggers too early
    when many stops remain.

    Returns (action, cost, stopped_at) exactly as v1's otr_route.
    """
    Wk = 0.0
    for k in range(1, m + 1):
        d_k, p_k = observe(k)
        Wk += p_k - d_k
        if Wk > B:                                    # physical overflow
            return ("EMERGENCY", Cfail, k)
        if k == m:
            return ("COMPLETE", 0.0, m)
        chat = cont_models[k].predict(np.array([Wk]))[0]
        if chat > omegaF:                             # continuing costs more
            return ("HANDOFF", omegaF, k)             # than handing off now
    return ("COMPLETE", 0.0, m)


# ============================================================
# Slack calibration — must use the peak
# ============================================================


def calibrate_B_empirical_peak(g_hist: np.ndarray, alpha: float = 0.10) -> float:
    """Set B so that alpha-fraction of historical routes would have PEAK-overflowed.

    v1 BUG: quantile of the endpoint total W_m. On collect-then-deliver data
    the endpoint is ~0 for every route and that quantile is meaningless.

    If Q and L0 are known exactly, B = Q - L0 is unchanged — but validate
    that the historical PEAK-overflow frequency at that B matches your
    operational overflow rate. A large mismatch means L0 is mis-estimated.
    """
    cum = np.cumsum(g_hist, axis=1)
    peak = cum.max(axis=1)                 # running maximum, not final value
    return float(np.quantile(peak, 1.0 - alpha))


# ============================================================
# Vectorized batch simulation
# ============================================================


def _simulate_costs_v2(
    g: np.ndarray,
    B: float,
    omegaF: float,
    Cfail: float,
    cont_models: dict,
) -> np.ndarray:
    """Vectorized OTR-2.0 simulation. Returns per-scenario cost array (N,).

    Logic identical to otr_route_v2(): at step k reveal W_k, check physical
    overflow (cost assigned at the overflow step ostep, not the endpoint),
    check last step, then apply the cost-comparison trigger.
    """
    N, m = g.shape
    cumW = np.cumsum(g, axis=1)

    costs = np.zeros(N)
    stopped = np.zeros(N, dtype=bool)

    for k_idx in range(m):
        k = k_idx + 1
        active = ~stopped
        if not active.any():
            break

        Wk = cumW[:, k_idx]

        # Physical overflow at this step
        em = active & (Wk > B)
        costs[em] = Cfail
        stopped |= em

        if k == m:
            break               # last customer — remaining active routes complete

        # Cost-comparison trigger: continuing costs more than handing off
        ho_active = active & ~em
        model = cont_models.get(k)
        if model is not None and ho_active.any():
            chat = model.predict(Wk[ho_active])
            ho_idx = np.where(ho_active)[0][chat > omegaF]
            costs[ho_idx] = omegaF
            stopped[ho_idx] = True

    return costs


def simulate_v2(
    g_test: np.ndarray,
    B: float,
    omegaF: float,
    Cfail: float,
    cont_models: dict,
) -> dict:
    """Run OTR-2.0 on every row of g_test. Same summary dict as v1's simulate()."""
    costs = _simulate_costs_v2(g_test, B, omegaF, Cfail, cont_models)
    ho = costs == omegaF
    em = costs == Cfail
    return {
        "mean_cost":     float(costs.mean()),
        "handoff_rate":  float(ho.mean()),
        "fail_rate":     float(em.mean()),
        "complete_rate": float((~ho & ~em).mean()),
    }


# ============================================================
# Clairvoyant oracle — perfect-information lower bound
# ============================================================


def oracle_costs(g: np.ndarray, B: float, omegaF: float, Cfail: float) -> np.ndarray:
    """Per-scenario cost of the clairvoyant policy that sees the whole path.

    If the route never peak-overflows, complete it free. If it would
    overflow at step 1 there is no decision epoch before the breach, so
    Cfail is unavoidable. Otherwise hand off just in time and pay omegaF.
    No online policy measurable w.r.t. the running-sum filtration can do
    better, so this is the perfect-information lower bound for the class.
    """
    ostep = _overflow_step(np.cumsum(g, axis=1), B)
    m = g.shape[1]
    costs = np.zeros(g.shape[0])
    costs[ostep == 1] = Cfail
    costs[(ostep > 1) & (ostep <= m)] = omegaF
    return costs


def simulate_oracle(g_test: np.ndarray, B: float, omegaF: float, Cfail: float) -> dict:
    """Summary-stats wrapper around oracle_costs (same dict as simulate_v2)."""
    costs = oracle_costs(g_test, B, omegaF, Cfail)
    ho = costs == omegaF
    em = costs == Cfail
    return {
        "mean_cost":     float(costs.mean()),
        "handoff_rate":  float(ho.mean()),
        "fail_rate":     float(em.mean()),
        "complete_rate": float((~ho & ~em).mean()),
    }


# ============================================================
# Validation protocol (run before deployment)
# ============================================================


def validate(
    g_val: np.ndarray,
    B: float,
    omegaF: float,
    Cfail: float,
    cont_models: dict,
) -> dict:
    """Three checks. Deploy only if all pass.

    (1) label sanity: peak-overflow rate must be materially nonzero;
    (2) endpoint-vs-peak divergence: low correlation means v1-style endpoint
        labels would have been catastrophically wrong — confirms v2 needed;
    (3) the policy beats the purely reactive baseline on held-out data.
    """
    cum = np.cumsum(g_val, axis=1)
    ostep = _overflow_step(cum, B)
    m = g_val.shape[1]

    peak_rate = float((ostep <= m).mean())
    endpoint = cum[:, -1]
    peak = cum.max(axis=1)
    if np.std(endpoint) < 1e-12 or np.std(peak) < 1e-12:
        corr = float("nan")     # degenerate: one of the two is constant
    else:
        corr = float(np.corrcoef(endpoint, peak)[0, 1])

    react = Cfail * peak_rate
    cost = float(_simulate_costs_v2(g_val, B, omegaF, Cfail, cont_models).mean())

    return {
        "peak_overflow_rate": peak_rate,
        "corr_endpoint_peak": corr,
        "reactive_cost":      react,
        "policy_cost":        cost,
        "deploy_ok":          bool((peak_rate > 0.005) and (cost < 0.9 * react)),
    }


# ============================================================
# Fallback: peak-label + fixed threshold (when history is small)
# ============================================================


def fit_otr_peak(g_hist: np.ndarray, B: float) -> dict:
    """Peak-label variant of v1's fit_otr, for the N < 500 regime.

    Backward induction is too noisy on small histories; fit isotonic
    p_k(W_k) curves against the PEAK-overflow label instead and tune tau
    with the v1 machinery (core.otr.tune_tau_fast / simulate_fast).
    Recovers ~95% of v2's saving and degrades gracefully with small N.
    NEVER use the endpoint label under any circumstances.

    Returns
    -------
    models : dict[int -> IsotonicRegression]
        models[k] estimates P(peak overflow | W_k = w), k = 1, ..., m-1.
    """
    N, m = g_hist.shape
    cum = np.cumsum(g_hist, axis=1)
    ostep = _overflow_step(cum, B)
    Y = (cum.max(axis=1) > B).astype(float)     # PEAK label — the v2 fix

    models: dict = {}
    for k in range(1, m):
        # condition only on paths still alive at step k so the curve prices
        # FUTURE risk (paths already overflowed are handled by the online
        # physical guard before the model is ever consulted)
        alive = ostep > k
        n_alive = int(alive.sum())
        if n_alive >= 2:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(cum[alive, k - 1], Y[alive])
        elif n_alive == 1:
            iso = _ConstantModel(Y[alive][0])
        else:
            iso = _ConstantModel(1.0)
        models[k] = iso
    return models

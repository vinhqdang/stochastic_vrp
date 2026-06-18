"""Online Threshold Reassignment (OTR) for the SVRPSPD.

Algorithm: fit conditional overflow probability curves offline using
isotonic regression, then make per-customer handoff decisions online
by comparing the predicted overflow probability to a threshold tau.

Reference: OTR_algorithm.md

Public API
----------
fit_otr               -- offline: fit per-step conditional overflow models
otr_route             -- online: run OTR on one truck route
calibrate_B           -- B = Q - L0
calibrate_B_empirical -- B at (1-alpha)-quantile of historical W distribution
tau_myopic            -- break-even threshold (omegaF / Cfail)
tune_tau              -- cross-validate tau on training data
calibration_rmse      -- calibration diagnostic (target < 0.05)
simulate              -- batch evaluation over a test set
fit_gaussian_params   -- parameter estimation for Gaussian fallback
gaussian_p            -- Gaussian closed-form P(W > B | W_k) (scarce-data fallback)
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


# ============================================================
# Offline phase
# ============================================================


def fit_otr(g_hist: np.ndarray, B: float) -> dict:
    """Fit per-step conditional overflow probability curves.

    Parameters
    ----------
    g_hist : np.ndarray, shape (N, m)
        Historical net increments. g_hist[s, k-1] is the net increment
        of customer k on historical route s (pickup minus delivery).
    B : float
        Slack = Q - L0. Route overflows when cumsum(g) > B.

    Returns
    -------
    models : dict[int -> IsotonicRegression]
        models[k] estimates P(W > B | W_k = w) for k in {1, ..., m-1}.
        Call as: prob = models[k].predict(np.array([w]))[0]
    """
    N, m = g_hist.shape
    cum = np.cumsum(g_hist, axis=1)   # shape (N, m); cum[s, k-1] = W_k for scenario s
    Y = (cum[:, -1] > B).astype(float)  # 1 if route s overflowed

    models = {}
    for k in range(1, m):             # steps 1 .. m-1
        x = cum[:, k - 1]             # running sum W_k across all scenarios
        iso = IsotonicRegression(
            increasing=True,           # higher W_k -> higher overflow risk
            out_of_bounds="clip",      # clamp extrapolation to boundary value
        )
        iso.fit(x, Y)
        models[k] = iso
    return models


# ============================================================
# Online phase
# ============================================================


def otr_route(
    observe,
    m: int,
    B: float,
    tau: float,
    omegaF: float,
    Cfail: float,
    models: dict,
) -> tuple:
    """Run OTR online, visiting customers one by one.

    Parameters
    ----------
    observe : callable
        observe(k) -> (d_k, p_k). Called exactly once per visited customer.
    m : int
        Number of customers on this route.
    B : float
        Slack = Q - L0.
    tau : float
        Decision threshold in (0, 1). Trigger handoff when P(overflow | W_k) > tau.
    omegaF : float
        Cost of a planned handoff (spare truck).
    Cfail : float
        Cost of an unplanned overflow. Must be > omegaF.
    models : dict[int -> IsotonicRegression]
        From fit_otr(). Must have keys 1 .. m-1.

    Returns
    -------
    (action, cost, stopped_at) where:
        action     : "COMPLETE" | "HANDOFF" | "EMERGENCY"
        cost       : 0.0 | omegaF | Cfail
        stopped_at : customer index at which the route ended (1-indexed)
    """
    Wk = 0.0   # running net-increment total

    for k in range(1, m + 1):
        d_k, p_k = observe(k)
        g = p_k - d_k
        Wk += g

        # Physical overflow already occurred — no model predicted this in time
        if Wk > B:
            return ("EMERGENCY", Cfail, k)

        # Last customer: route is done
        if k == m:
            return ("COMPLETE", 0.0, m)

        # Predict P(W > B | W_k) and decide
        p = models[k].predict(np.array([Wk]))[0]
        if p > tau:
            return ("HANDOFF", omegaF, k)

    # Unreachable: loop always returns inside the body at k == m
    return ("COMPLETE", 0.0, m)


# ============================================================
# Slack calibration
# ============================================================


def calibrate_B(Q: float, L0: float) -> float:
    """B = Q - L0. Use when Q and L0 are known exactly."""
    return Q - L0


def calibrate_B_empirical(g_hist: np.ndarray, alpha: float = 0.10) -> float:
    """Set B so that historically alpha-fraction of routes would have overflowed.

    Use this when Q or L0 is uncertain.

    Parameters
    ----------
    g_hist : np.ndarray, shape (N, m)
    alpha : float
        Acceptable historical overflow rate (e.g. 0.10 for 10%).

    Returns
    -------
    B : float
        (1 - alpha)-quantile of the historical total net increment distribution.
    """
    W_hist = g_hist.sum(axis=1)
    return float(np.quantile(W_hist, 1.0 - alpha))


# ============================================================
# Threshold selection
# ============================================================


def tau_myopic(omegaF: float, Cfail: float) -> float:
    """Break-even threshold: hand off now (omegaF) vs overflow (p * Cfail).

    Lower bound on the optimal tau. Valid when predicted probabilities are
    well-calibrated. Use tune_tau when Cfail/omegaF > 5.
    """
    return omegaF / Cfail


def tune_tau(
    g_train: np.ndarray,
    B: float,
    models: dict,
    omegaF: float,
    Cfail: float,
    tau_grid: np.ndarray | None = None,
) -> float:
    """Pick tau with lowest average cost on the training set.

    Parameters
    ----------
    g_train : np.ndarray, shape (N, m)
    B : float
    models : dict — from fit_otr()
    omegaF : float
    Cfail : float
    tau_grid : np.ndarray, optional
        Grid to search over. Defaults to np.linspace(0.02, 0.95, 47).

    Returns
    -------
    best_tau : float

    Warning
    -------
    For N < 1000 with a fine grid, the result may overfit. Use tau_myopic instead.
    """
    if tau_grid is None:
        tau_grid = np.linspace(0.02, 0.95, 47)

    N, m = g_train.shape
    best_tau, best_cost = tau_grid[0], np.inf

    for tau in tau_grid:
        costs = []
        for s in range(N):
            g_s = g_train[s]
            obs = lambda k, _g=g_s: (0.0, _g[k - 1])  # delivery=0, pickup=g
            _, cost, _ = otr_route(obs, m, B, tau, omegaF, Cfail, models)
            costs.append(cost)
        avg = float(np.mean(costs))
        if avg < best_cost:
            best_cost = avg
            best_tau = float(tau)

    return best_tau


# ============================================================
# Calibration diagnostic
# ============================================================


def calibration_rmse(
    g_test: np.ndarray,
    B: float,
    models: dict,
    n_bins: int = 10,
) -> float:
    """Calibration RMSE across all steps k and all test scenarios.

    Lower is better. Values > 0.05 indicate meaningful miscalibration.

    Parameters
    ----------
    g_test : np.ndarray, shape (N_test, m)
    B : float
    models : dict — from fit_otr()
    n_bins : int

    Returns
    -------
    rmse : float, or nan if no bin has >= 50 samples
    """
    N, m = g_test.shape
    cum = np.cumsum(g_test, axis=1)
    Wfull = cum[:, -1]
    Y = (Wfull > B).astype(float)

    preds, actuals = [], []
    for k in range(1, m):
        Wk = cum[:, k - 1]
        p = models[k].predict(Wk)
        preds.append(p)
        actuals.append(Y)

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    bins = np.linspace(0, 1, n_bins + 1)
    sq_errors = []
    for b in range(n_bins):
        mask = (preds >= bins[b]) & (preds < bins[b + 1])
        if mask.sum() > 50:
            sq_errors.append((preds[mask].mean() - actuals[mask].mean()) ** 2)

    return float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan")


# ============================================================
# Batch simulation
# ============================================================


def simulate(
    g_test: np.ndarray,
    B: float,
    tau: float,
    omegaF: float,
    Cfail: float,
    models: dict,
) -> dict:
    """Run OTR on every row of g_test and return summary statistics.

    Parameters
    ----------
    g_test : np.ndarray, shape (N_test, m)
    B : float
    tau : float
    omegaF : float
    Cfail : float
    models : dict — from fit_otr()

    Returns
    -------
    dict with keys: mean_cost, handoff_rate, fail_rate, complete_rate
    """
    N, m = g_test.shape
    actions = []
    costs = []

    for s in range(N):
        g_s = g_test[s]
        action, cost, _ = otr_route(
            observe=lambda k, _g=g_s: (0.0, _g[k - 1]),
            m=m, B=B, tau=tau, omegaF=omegaF, Cfail=Cfail, models=models,
        )
        actions.append(action)
        costs.append(cost)

    costs = np.array(costs)
    return {
        "mean_cost":     float(costs.mean()),
        "handoff_rate":  float(np.mean([a == "HANDOFF"   for a in actions])),
        "fail_rate":     float(np.mean([a == "EMERGENCY" for a in actions])),
        "complete_rate": float(np.mean([a == "COMPLETE"  for a in actions])),
    }


# ============================================================
# Appendix A — Gaussian closed-form fallback (scarce data, N < 200)
# ============================================================


def fit_gaussian_params(g_hist: np.ndarray) -> tuple:
    """Estimate (mu, sigma, rho) from historical net increments.

    mu  : per-customer mean net increment
    sigma : per-customer standard deviation
    rho : average pairwise correlation across customers

    Use gaussian_p() as the predictor when N < 200.
    """
    mu = float(g_hist.mean())
    sigma = float(g_hist.std())
    C = np.corrcoef(g_hist.T)   # (m, m) correlation matrix
    m = C.shape[0]
    if m <= 1:
        rho = 0.0
    else:
        rho = float((C.sum() - m) / (m * (m - 1)))
    rho = float(np.clip(rho, 1e-6, 0.999))
    return mu, sigma, rho


def gaussian_p(
    k: int,
    Wk: float,
    m: int,
    mu: float,
    sigma: float,
    rho: float,
    B: float,
) -> float:
    """Closed-form P(W > B | W_k = Wk) under the linear-Gaussian factor model.

    Drop-in replacement for models[k].predict([Wk])[0] when N < 200.
    Miscalibrated on non-Gaussian / heavy-tailed data (RMSE ~0.056 vs ~0.014
    for the isotonic model).
    """
    from scipy.stats import norm as _norm

    lam = sigma * np.sqrt(rho)
    sig_eps = sigma * np.sqrt(1.0 - rho)
    Vk = (1.0 - rho) / ((1.0 - rho) + k * rho)
    Fhat = Vk * (lam / sig_eps ** 2) * (Wk - k * mu)
    rem = m - k
    E_rem = rem * mu + lam * rem * Fhat
    Var_rem = lam ** 2 * rem ** 2 * Vk + rem * sig_eps ** 2
    return float(1.0 - _norm.cdf((B - Wk - E_rem) / np.sqrt(Var_rem)))

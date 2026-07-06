"""OTR-2.1 — feature-enriched continuation models (experimental).

OTR-2.0 conditions on the scalar running sum W_k. Within a fixed stop k
that discards path information: two days with the same W_k can carry very
different signals about what is still coming. Under the correlated-demand
generator (equicorrelated Gaussian copula) the natural extra statistic is
the PRECISION-WEIGHTED residual

    fhat_k = sum_{j<=k} (g_j - mu_j) / sigma_j^2   (normalized)

— the posterior location of today's common demand factor, which predicts
the level of the remaining increments and is NOT a function of W_k when
customer means/variances are heterogeneous. We also add the last increment
g_k (regime recency) and the running max increment (spike detector).

The per-step regressor becomes a small gradient-boosted tree ensemble with
a monotone constraint on W_k (continuation cost still rises with the
prefix). Everything else — the peak-aware labels, the backward induction,
the cost-comparison trigger — is unchanged from OTR-2.0.

This module deliberately mirrors core/costs.py's general-schedule API:
fit on (g_hist, B, H, E), simulate with per-stop prices.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .otr2 import _overflow_step, _ConstantModel


def _feature_arrays(g: np.ndarray, mu: np.ndarray, sig: np.ndarray):
    """Per-step feature tensors, each shape (N, m).

    Features at step k (columns k-1):
      0: W_k        running net-increment sum (monotone risk driver)
      1: fhat_k     precision-weighted residual = common-factor posterior
      2: g_k        last increment
      3: gmax_k     running max increment
    """
    cum = np.cumsum(g, axis=1)
    w = 1.0 / np.maximum(sig, 1e-9) ** 2
    resid = (g - mu[None, :]) * w[None, :]
    denom = np.cumsum(w)
    fhat = np.cumsum(resid, axis=1) / denom[None, :]
    gmax = np.maximum.accumulate(g, axis=1)
    return cum, fhat, g, gmax


def _stack(k_idx: int, cum, fhat, glast, gmax) -> np.ndarray:
    return np.column_stack([cum[:, k_idx], fhat[:, k_idx],
                            glast[:, k_idx], gmax[:, k_idx]])


def fit_lsm_rich(g_hist: np.ndarray, mu: np.ndarray, sig: np.ndarray,
                 B: float, H: np.ndarray, E: np.ndarray,
                 min_alive: int = 50) -> dict:
    """Backward induction with boosted-tree continuation models on the
    enriched statistic. Same recursion as costs.fit_lsm_general.

    mu, sig: per-customer mean and std of the net increment g (route order),
    used only to build features — no distributional assumption enters the
    labels or the recursion.
    """
    N, m = g_hist.shape
    cum, fhat, glast, gmax = _feature_arrays(g_hist, mu, sig)
    ostep = _overflow_step(cum, B)

    future = np.zeros(N)
    breached = ostep <= m
    future[breached] = E[np.clip(ostep[breached] - 1, 0, m - 1)]

    models: dict = {}
    for k in range(m - 1, 0, -1):
        alive = ostep > k
        n_alive = int(alive.sum())
        if n_alive < min_alive:
            # too few survivors for trees: constant (conservative) estimate
            models[k] = _ConstantModel(float(future[alive].mean())
                                       if n_alive else float(E[min(k, m - 1)]))
        else:
            X = _stack(k - 1, cum, fhat, glast, gmax)[alive]
            reg = HistGradientBoostingRegressor(
                max_iter=120, max_depth=3, learning_rate=0.1,
                l2_regularization=1.0, min_samples_leaf=20,
                monotonic_cst=[1, 0, 0, 0], early_stopping=False,
                random_state=0)
            reg.fit(X, future[alive])
            models[k] = reg
        Xall = _stack(k - 1, cum, fhat, glast, gmax)
        pred = models[k].predict(Xall) if not isinstance(models[k], _ConstantModel) \
            else models[k].predict(Xall[:, 0])
        stop = alive & (pred > H[k - 1])
        future = future.copy()
        future[stop] = H[k - 1]
    return models


def simulate_rich(g_test: np.ndarray, mu: np.ndarray, sig: np.ndarray,
                  B: float, H: np.ndarray, E: np.ndarray, models: dict,
                  return_actions: bool = False, H_bill=None):
    """Vectorized execution of the OTR-2.1 policy under per-stop prices."""
    if H_bill is None:
        H_bill = H
    N, m = g_test.shape
    cum, fhat, glast, gmax = _feature_arrays(g_test, mu, sig)

    costs = np.zeros(N)
    action = np.zeros(N, dtype=np.int8)
    stopped = np.zeros(N, dtype=bool)
    for k_idx in range(m):
        k = k_idx + 1
        active = ~stopped
        if not active.any():
            break
        Wk = cum[:, k_idx]
        em = active & (Wk > B)
        costs[em] = E[k_idx]
        action[em] = 2
        stopped |= em
        if k == m:
            break
        ho_active = active & ~em
        mdl = models.get(k)
        if mdl is None or not ho_active.any():
            continue
        if isinstance(mdl, _ConstantModel):
            chat = mdl.predict(Wk[ho_active])
        else:
            X = _stack(k_idx, cum, fhat, glast, gmax)[ho_active]
            chat = mdl.predict(X)
        trig = chat > H[k_idx]
        idx = np.where(ho_active)[0][trig]
        costs[idx] = H_bill[k_idx]
        action[idx] = 1
        stopped[idx] = True

    stats = {
        "mean_cost":     float(costs.mean()),
        "handoff_rate":  float((action == 1).mean()),
        "fail_rate":     float((action == 2).mean()),
        "complete_rate": float((action == 0).mean()),
    }
    return (stats, action) if return_actions else stats

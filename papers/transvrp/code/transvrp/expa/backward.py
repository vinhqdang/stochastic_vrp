"""Generic Longstaff-Schwartz backward driver (module M3).

EXPERIMENT_A.md sections 2.2 and 2.3. For k = m-1 down to 1:

1. fit the regressor(s) of epoch k on the alive rows (tau > k) with target
   ``future``;
2. set ``future`` to H[k-1] on the alive rows whose predicted continuation
   cost exceeds H[k-1].

``folds=None`` is BATON's mode: one predictor, fitted and applied to the same
rows. With ``folds=(A, B)`` (fitting rows split by parity) and validation rows
``val_idx``, net A is fitted on A's alive rows and applied to B's rows, net B
the reverse, and validation rows use the mean of both (section 2.3). Every
row a net is fitted on is never a row it decides on at the same epoch.

The fit function owns the small-sample branches (for isotonic fits, the
>= 2 / == 1 / 0 alive cases of ``costs.fit_lsm_general``).
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..baton.otr2 import _ConstantModel, _overflow_step
from .features import RouteCache


class Predictor(Protocol):
    def predict(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray: ...


FitFn = Callable[[int, RouteCache, np.ndarray, np.ndarray, np.ndarray,
                  np.ndarray, "Predictor | None"], Predictor]
#  (k, cache, fit_idx, y_fit, val_idx, y_val, warm_start) -> predictor for epoch k


def split_80_20(N: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Fitting rows are the first floor(0.8 N) paths, split by parity into
    folds A and B; validation rows are the last ceil(0.2 N) paths."""
    n_fit = (4 * N) // 5
    fit = np.arange(n_fit, dtype=np.int64)
    val = np.arange(n_fit, N, dtype=np.int64)
    return (fit[0::2], fit[1::2]), val


class IsoPredictor:
    """Wraps an isotonic regressor or _ConstantModel applied to W_k."""

    def __init__(self, model: object):
        self.model = model

    def predict_w(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        if w.size == 0:
            return np.zeros(0)
        return np.asarray(self.model.predict(w), dtype=float)

    def predict(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
        return self.predict_w(cache.W[np.asarray(idx, dtype=np.int64), k - 1])


def iso_fit_fn() -> FitFn:
    """Exact replica of the per-k branches of ``costs.fit_lsm_general``."""

    def fit(k: int, cache: RouteCache, fit_idx: np.ndarray, y_fit: np.ndarray,
            val_idx: np.ndarray, y_val: np.ndarray,
            warm: "Predictor | None") -> Predictor:
        m = cache.m
        n = int(np.asarray(fit_idx).size)
        if n >= 2:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(cache.W[fit_idx, k - 1], y_fit)
            model: object = iso
        elif n == 1:
            model = _ConstantModel(float(y_fit[0]))
        else:
            model = _ConstantModel(float(cache.spec.E[min(k, m - 1)]))
        return IsoPredictor(model)

    return fit


DISAGREE_ROWS = 20_000   # fixed evenly spaced subsample of alive rows for the diagnostic


def _disagreement(netA: Predictor, netB: Predictor, cache: RouteCache, k: int,
                  alive_rows: np.ndarray, h: float) -> float:
    """Fraction of alive rows on which the two fold predictors disagree about
    stopping at epoch k. Computed on at most DISAGREE_ROWS evenly spaced alive
    rows, so the diagnostic costs two forward passes on a bounded set."""
    if alive_rows.size == 0:
        return 0.0
    if alive_rows.size > DISAGREE_ROWS:
        pos = np.linspace(0, alive_rows.size - 1, DISAGREE_ROWS).round().astype(np.int64)
        rows = alive_rows[pos]
    else:
        rows = alive_rows
    a = netA.predict(cache, k, rows) > h
    b = netB.predict(cache, k, rows) > h
    return float(np.mean(a != b))


def _log_entry(log: list, k: int, fold: str, fit_rows: np.ndarray,
               applied_rows: np.ndarray, disagree: float) -> None:
    log.append({"k": k, "fold": fold, "fit_rows": fit_rows,
                "applied_rows": applied_rows, "disagree_rate": disagree})


def ls_backward(cache: RouteCache, B: float, H: np.ndarray, E: np.ndarray,
                fit_fn: FitFn,
                folds: tuple[np.ndarray, np.ndarray] | None,
                val_idx: np.ndarray | None,
                log: list | None = None) -> dict[int, list[Predictor]]:
    """Backward induction over k. Returns {k: [predictor]} in BATON mode and
    {k: [net_A, net_B]} with folds (the deployed value is their mean)."""
    W = cache.W
    N, m = W.shape
    H = np.asarray(H, dtype=float)
    E = np.asarray(E, dtype=float)
    if H.shape != (m,) or E.shape != (m,):
        raise ValueError("H and E must have length m")

    tau = _overflow_step(W, B)
    future = np.zeros(N)
    breached = tau <= m
    future[breached] = E[tau[breached] - 1]
    empty = np.zeros(0, dtype=np.int64)
    models: dict[int, list[Predictor]] = {}

    if folds is None:
        prev: Predictor | None = None
        every = np.arange(N, dtype=np.int64)
        for k in range(m - 1, 0, -1):
            alive = tau > k
            fit_rows = np.flatnonzero(alive)
            pred = fit_fn(k, cache, fit_rows, future[fit_rows], empty,
                          np.zeros(0), prev)
            models[k] = [pred]
            prev = pred
            cont = pred.predict(cache, k, every)
            stop = alive & (cont > H[k - 1])
            future[stop] = H[k - 1]
            if log is not None:
                _log_entry(log, k, "all", fit_rows, fit_rows, 0.0)
        return models

    fA, fB = (np.asarray(folds[0], dtype=np.int64), np.asarray(folds[1], dtype=np.int64))
    val = (np.zeros(0, dtype=np.int64) if val_idx is None
           else np.asarray(val_idx, dtype=np.int64))
    if np.intersect1d(fA, fB).size or np.intersect1d(np.r_[fA, fB], val).size:
        raise ValueError("folds and validation rows must be disjoint")

    prevA: Predictor | None = None
    prevB: Predictor | None = None
    for k in range(m - 1, 0, -1):
        alive = tau > k
        Aa = fA[alive[fA]]
        Ba = fB[alive[fB]]
        Va = val[alive[val]]
        y_A = future[Aa].copy()
        y_B = future[Ba].copy()
        y_V = future[Va].copy()
        # both nets are fitted before any stop update of this epoch
        netA = fit_fn(k, cache, Aa, y_A, Va, y_V, prevA)
        netB = fit_fn(k, cache, Ba, y_B, Va, y_V, prevB)

        pA_on_B = netA.predict(cache, k, Ba)
        pB_on_A = netB.predict(cache, k, Aa)
        pA_val = netA.predict(cache, k, Va)
        pB_val = netB.predict(cache, k, Va)
        h = H[k - 1]
        future[Aa[pB_on_A > h]] = h
        future[Ba[pA_on_B > h]] = h
        future[Va[0.5 * (pA_val + pB_val) > h]] = h

        if log is not None:
            dis = _disagreement(netA, netB, cache, k, np.flatnonzero(alive), h)
            _log_entry(log, k, "A", Aa, Ba, dis)
            _log_entry(log, k, "B", Ba, Aa, dis)
            _log_entry(log, k, "val", np.r_[Aa, Ba], Va, dis)

        models[k] = [netA, netB]
        prevA, prevB = netA, netB
    return models

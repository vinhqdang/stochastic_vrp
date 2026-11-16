"""Gradient-boosted continuation values on the FULL globals (module M11).

EXPERIMENT_A.md section 9, M11 and section 1.5 (M11 contrast). Run only in the
WIN branch, outside the Holm family, with its own alpha of 0.05.

``HistGradientBoostingRegressor`` with the OTR-2.1 hyperparameters (max_iter
120, max_depth 3, learning_rate 0.1, l2_regularization 1.0, min_samples_leaf
20, random_state 0) and **no monotone constraint** (section 4.2). The input is
the 12 FULL globals g1-g12 of ``features.stage_batch`` (token sequence unused).
It is plugged into ``backward.ls_backward`` with the same folds as the neural
arms.

Deviation: ``early_stopping=False`` is set as in OTR-2.1. Without it,
scikit-learn's default ('auto') enables early stopping above 10,000 rows, which
the section 9 list does not mention.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .backward import FitFn, ls_backward, split_80_20
from .features import RouteCache, RouteStats, stage_batch
from .paths import Paths, RouteSpec
from .train import ConstantPredictor

BOOST_PARAMS = dict(max_iter=120, max_depth=3, learning_rate=0.1,
                    l2_regularization=1.0, min_samples_leaf=20,
                    early_stopping=False, random_state=0)
MIN_ROWS = 20


class BoostPredictor:
    """Row-wise predictor on the FULL globals, bound to the cache passed in."""

    def __init__(self, model: HistGradientBoostingRegressor):
        self.model = model

    def predict(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return np.zeros(0)
        glob, _ = stage_batch(cache, k, idx, "full")
        return np.asarray(self.model.predict(np.asarray(glob, dtype=np.float64)),
                          dtype=float)


def boost_fit_fn() -> FitFn:
    """FitFn for ls_backward. Fewer than MIN_ROWS fitting rows give the constant
    fallback of the neural arms (mean target, or E[min(k, m-1)] if none)."""

    def fit(k, cache, fit_idx, y_fit, val_idx, y_val, warm):
        m = cache.m
        E = cache.spec.E
        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        y_fit = np.asarray(y_fit, dtype=np.float64)
        n = int(fit_idx.size)
        if n < MIN_ROWS:
            value = float(y_fit.mean()) if n else float(E[min(k, m - 1)])
            return ConstantPredictor(value)
        glob, _ = stage_batch(cache, k, fit_idx, "full")
        model = HistGradientBoostingRegressor(**BOOST_PARAMS)
        model.fit(np.asarray(glob, dtype=np.float64), y_fit)
        return BoostPredictor(model)

    return fit


def fit_boost(spec: RouteSpec, paths: Paths, N: int, rep: int
              ) -> tuple[dict, RouteStats, list]:
    """Cross-fitted boosted models for one route, same split and statistics as
    ``train.fit_neural``. ``rep`` is accepted for symmetry (not used)."""
    if paths.p.shape[0] != N:
        raise ValueError("paths must hold exactly N rows")
    if np.any(spec.H > spec.E0 + 1e-12) or np.any(spec.E > spec.E0 + 1e-12):
        raise ValueError("H and E must not exceed E0")
    folds, val = split_80_20(N)
    fit_all = np.concatenate(folds)
    stats = RouteStats.fit(Paths(p=paths.p[fit_all], d=paths.d[fit_all]))
    cache = RouteCache(spec, paths, stats)
    log: list = []
    models = ls_backward(cache, spec.B, spec.H, spec.E, boost_fit_fn(),
                         folds, val, log=log)
    diag = [dict(kind="ls", k=e["k"], fold=e["fold"],
                 n_fit=int(e["fit_rows"].size), n_applied=int(e["applied_rows"].size),
                 disagree_rate=float(e["disagree_rate"])) for e in log]
    return models, stats, diag

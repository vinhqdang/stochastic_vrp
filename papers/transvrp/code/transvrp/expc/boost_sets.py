"""BOOST for Q1 feature sets (EXPERIMENT_C.md 1.3, module MC2).

Same learner, hyperparameters, folds and statistics as ``expa.boost`` (M11), but
the input is ``select_columns(mlp_batch, set_name)``: the 15 mlp inputs restricted
to the set, token summaries s1..s3 included for every set. The predictions are
clipped to [0, E0] (as RLSM's are); A's BOOST is not clipped and is not changed.

Deviation from A's BOOST (finding S4, stated in the paper): predictions are clipped to
[0, E0]. The clip is the only change of the learner; the parameters and folds are A's.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..expa.backward import FitFn, ls_backward, split_80_20
from ..expa.boost import BOOST_PARAMS, MIN_ROWS
from ..expa.features import RouteCache, RouteStats, mlp_batch
from ..expa.paths import Paths, RouteSpec
from ..expa.train import ConstantPredictor
from .featuresets import columns_of, select_columns


class BoostSetPredictor:
    """Boosted predictor on the set's columns of mlp_batch, bound to the cache passed in."""

    def __init__(self, model: HistGradientBoostingRegressor, set_name: str):
        self.model = model
        self.set_name = set_name

    def predict(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return np.zeros(0)
        x = select_columns(mlp_batch(cache, k, idx), self.set_name)
        raw = np.asarray(self.model.predict(np.asarray(x, dtype=np.float64)), dtype=float)
        return np.clip(raw, 0.0, cache.spec.E0)


def boost_set_fit_fn(set_name: str) -> FitFn:
    """FitFn for ls_backward on one feature set. Fewer than MIN_ROWS fitting rows give
    the constant fallback of the neural arms (as in ``expa.boost.boost_fit_fn``)."""
    columns_of(set_name)

    def fit(k, cache, fit_idx, y_fit, val_idx, y_val, warm):
        m = cache.m
        E = cache.spec.E
        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        y_fit = np.asarray(y_fit, dtype=np.float64)
        n = int(fit_idx.size)
        if n < MIN_ROWS:
            value = float(y_fit.mean()) if n else float(E[min(k, m - 1)])
            return ConstantPredictor(value)
        x = select_columns(mlp_batch(cache, k, fit_idx), set_name)
        model = HistGradientBoostingRegressor(**BOOST_PARAMS)
        model.fit(np.asarray(x, dtype=np.float64), y_fit)
        return BoostSetPredictor(model, set_name)

    return fit


def fit_boost_set(spec: RouteSpec, paths: Paths, N: int, rep: int, set_name: str
                  ) -> tuple[dict, RouteStats, list]:
    """Cross-fitted boosted models for one route and feature set. Mirrors
    ``expa.boost.fit_boost`` (same split, statistics and checks); ``rep`` is unused."""
    if paths.p.shape[0] != N:
        raise ValueError("paths must hold exactly N rows")
    if np.any(spec.H > spec.E0 + 1e-12) or np.any(spec.E > spec.E0 + 1e-12):
        raise ValueError("H and E must not exceed E0")
    folds, val = split_80_20(N)
    fit_all = np.concatenate(folds)
    stats = RouteStats.fit(Paths(p=paths.p[fit_all], d=paths.d[fit_all]))
    cache = RouteCache(spec, paths, stats)
    log: list = []
    models = ls_backward(cache, spec.B, spec.H, spec.E, boost_set_fit_fn(set_name),
                         folds, val, log=log)
    diag = [dict(kind="ls", k=e["k"], fold=e["fold"],
                 n_fit=int(e["fit_rows"].size), n_applied=int(e["applied_rows"].size),
                 disagree_rate=float(e["disagree_rate"])) for e in log]
    return models, stats, diag

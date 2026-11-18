"""RLSM: randomized neural networks for optimal stopping (EXPERIMENT_B.md 3.3; M15;
Herrera, Krach, Ruyssen & Teichmann 2024).

A FitFn for ``backward.ls_backward`` with A's folds. At stage k and fold f the network
is a random layer h = LeakyReLU(x A + b) (A in R^{15 x K}, b in R^K, i.i.d. N(0, 1),
drawn per stage and fold, the per-date variant of the paper) followed by a least-squares
fit of the continuation cost on [h, 1]. The ridge penalty is lambda = ridge * n_fit (the
grid of 3.3 is {1e-6, 1e-3} times n). Predictions are clipped to [0, E0]. Fewer than
``min_rows`` fitting rows give A's constant fallback.

Bug-detector diagnostics (gate B-RLSM): the relative normal-equation residual and the
validation MSE against the constant predictor. They are not fidelity measures.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

import numpy as np

from ..expa.backward import FitFn, ls_backward, split_80_20
from ..expa.features import MLP_DIM, RouteCache, RouteStats, mlp_batch
from ..expa.paths import Paths, RouteSpec
from ..expa.train import ConstantPredictor, _fold_label

PRED_CHUNK = 8192


@dataclass(frozen=True)
class RLSMConfig:
    K: int = 100
    slope: float = 0.5
    ridge: float = 1e-6      # lambda = ridge * n_fit
    min_rows: int = 20


def leaky(z: np.ndarray, slope: float) -> np.ndarray:
    return np.where(z > 0, z, slope * z)


def draw_layer(seed_key: str, fold: str, k: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Random layer (A (15, K), b (K,)) for stage k and fold, i.i.d. standard normal."""
    rng = np.random.default_rng(zlib.crc32(f"{seed_key}|{fold}|{k}".encode("utf-8")))
    A = rng.standard_normal((MLP_DIM, K))
    b = rng.standard_normal(K)
    return A, b


def design(x: np.ndarray, A: np.ndarray, b: np.ndarray, slope: float) -> np.ndarray:
    """[LeakyReLU(xA + b), 1]: the regressors of the least-squares step."""
    h = leaky(np.asarray(x, dtype=float) @ A + b, slope)
    return np.hstack([h, np.ones((h.shape[0], 1))])


def ridge_lstsq(D: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """argmin ||D theta - y||^2 + lam ||theta||^2 in one least-squares solve, on the
    augmented system [D; sqrt(lam) I] theta = [y; 0] (no normal equations formed)."""
    p = D.shape[1]
    if lam > 0:
        aug = np.vstack([D, np.sqrt(lam) * np.eye(p)])
        rhs = np.concatenate([np.asarray(y, dtype=float), np.zeros(p)])
    else:
        aug, rhs = D, np.asarray(y, dtype=float)
    return np.linalg.lstsq(aug, rhs, rcond=None)[0]


def normal_residual_rel(D: np.ndarray, y: np.ndarray, theta: np.ndarray, lam: float) -> float:
    """Backward error of the ridge normal equations: ||M theta - D'y|| relative to
    ||M|| ||theta|| + ||D'y||, with M = D'D + lam I."""
    M = D.T @ D + lam * np.eye(D.shape[1])
    rhs = D.T @ np.asarray(y, dtype=float)
    num = np.linalg.norm(M @ theta - rhs)
    den = np.linalg.norm(M) * np.linalg.norm(theta) + np.linalg.norm(rhs)
    return float(num / den) if den > 0 else float(num)


class RLSMPredictor:
    """Fitted random-feature regression for one (stage, fold)."""

    def __init__(self, A: np.ndarray, b: np.ndarray, theta: np.ndarray, slope: float,
                 E0: float):
        self.A = A
        self.b = b
        self.theta = theta
        self.slope = float(slope)
        self.E0 = float(E0)

    def raw(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
        """Unclipped regression output (cost units)."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.zeros(idx.size)
        for s in range(0, idx.size, PRED_CHUNK):
            sl = idx[s:s + PRED_CHUNK]
            x = mlp_batch(cache, k, sl).astype(float)
            out[s:s + sl.size] = design(x, self.A, self.b, self.slope) @ self.theta
        return out

    def predict(self, cache: RouteCache, k: int, idx: np.ndarray) -> np.ndarray:
        return np.clip(self.raw(cache, k, idx), 0.0, self.E0)


def rlsm_fit_fn(seed_key: str, cfg: RLSMConfig, diag: list) -> FitFn:
    """FitFn for ls_backward; diagnostics are appended to ``diag``."""

    def fit(k, cache, fit_idx, y_fit, val_idx, y_val, warm):
        t0 = time.perf_counter()
        spec: RouteSpec = cache.spec
        m, E0 = cache.m, spec.E0
        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        y_fit = np.asarray(y_fit, dtype=float)
        y_val = np.asarray(y_val, dtype=float)
        if y_fit.size and (y_fit.max() > E0 + 1e-9 or y_fit.min() < -1e-9):
            raise ValueError("fitting targets must lie in [0, E0]")
        n = int(fit_idx.size)
        fold = _fold_label(fit_idx)
        if n < cfg.min_rows:
            value = float(y_fit.mean()) if n else float(spec.E[min(k, m - 1)])
            diag.append(dict(kind="stage", arm="rlsm", k=k, fold=fold, n_fit=n,
                             n_val=int(val_idx.size), action="constant", steps=0,
                             seconds=time.perf_counter() - t0))
            return ConstantPredictor(value)

        A, b = draw_layer(seed_key, fold, k, cfg.K)
        x = mlp_batch(cache, k, fit_idx).astype(float)
        D = design(x, A, b, cfg.slope)
        lam = cfg.ridge * n
        theta = ridge_lstsq(D, y_fit, lam)
        pred = RLSMPredictor(A, b, theta, cfg.slope, E0)
        entry = dict(kind="stage", arm="rlsm", k=k, fold=fold, n_fit=n,
                     n_val=int(val_idx.size), action="fit", steps=0,
                     ne_residual_rel=normal_residual_rel(D, y_fit, theta, lam),
                     seconds=time.perf_counter() - t0)
        if val_idx.size:
            pv = pred.predict(cache, k, val_idx)
            entry["val_mse"] = float(np.mean((pv - y_val) ** 2))
            entry["const_val_mse"] = float(np.mean((y_fit.mean() - y_val) ** 2))
        diag.append(entry)
        return pred

    return fit


def fit_rlsm(spec: RouteSpec, paths: Paths, N: int, rep: int, cfg: RLSMConfig
             ) -> tuple[dict, RouteStats, list, dict]:
    """Cross-fitted RLSM models for one route (A's split and statistics). Returns
    (models {k: [net_A, net_B]}, RouteStats, diag, budget)."""
    if paths.p.shape[0] != N:
        raise ValueError(f"expected {N} training paths, got {paths.p.shape[0]}")
    t0 = time.perf_counter()
    folds, val = split_80_20(N)
    fit_all = np.concatenate(folds)
    stats = RouteStats.fit(Paths(p=paths.p[fit_all], d=paths.d[fit_all]))
    cache = RouteCache(spec, paths, stats)
    seed_key = f"{spec.inst}|{spec.route_id}|{N}|{rep}|rlsm"
    diag: list = []
    log: list = []
    models = ls_backward(cache, spec.B, spec.H, spec.E,
                         rlsm_fit_fn(seed_key, cfg, diag), folds, val, log=log)
    for e in log:
        diag.append(dict(kind="ls", k=e["k"], fold=e["fold"],
                         n_fit=int(e["fit_rows"].size),
                         n_applied=int(e["applied_rows"].size),
                         disagree_rate=float(e["disagree_rate"])))
    fitted = sum(1 for d in diag if d.get("action") == "fit")
    budget = {"unique_paths": int(N), "path_stop_reads": int(N * spec.m),
              "val_reads": int(sum(d.get("n_val", 0) for d in diag if d.get("kind") == "stage")),
              "n_params": int(fitted * (cfg.K + 1)), "feature_seconds": 0.0,
              "train_seconds": time.perf_counter() - t0, "val_seconds": 0.0,
              "device": "cpu"}
    return models, stats, diag, budget

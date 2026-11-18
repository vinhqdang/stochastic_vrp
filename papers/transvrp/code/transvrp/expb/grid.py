"""Continuation values tabulated on a uniform load grid (EXPERIMENT_B.md 3.1, 3.2; M13).

``GridPredictor`` is the deployed form of THR and ALP-Z: a cost predictor C_k(W_k) read
by linear interpolation, clipped at the lowest grid load. The common tie rule (T1) is a
tolerance added to the prediction: ``simulate_policy`` hands off iff its strict test
``chat > H[k-1]`` holds, so returning C + tie_tol means hand off iff C > H - tie_tol,
which is "C >= H - tie_tol" except on the float-measure-zero set C = H - tie_tol.
"""
from __future__ import annotations

import numpy as np

TIE_REL = 1e-6          # T1: tie tolerance in units of E0
MIN_GRID = 2


def tie_tolerance(E0: float) -> float:
    return TIE_REL * float(E0)


def lin_interp(x: np.ndarray, w0: float, h: float, values: np.ndarray) -> np.ndarray:
    """Linear interpolation on the uniform grid w0 + i*h, clipped at w0 from below
    (loads below the lowest grid point take the first value)."""
    xc = np.maximum(np.asarray(x, dtype=float), w0)
    pos = (xc - w0) / h
    i = np.minimum(np.floor(pos).astype(np.intp), values.size - 2)
    frac = pos - i
    return values[i] + frac * (values[i + 1] - values[i])


class GridPredictor:
    """Predictor for ``simulate_policy``. ``c`` has shape (m-1, n_grid): row k-1 holds the
    continuation cost of decision epoch k on ``w_grid``."""

    def __init__(self, w_grid: np.ndarray, c: np.ndarray, tie_tol: float = 0.0):
        w = np.asarray(w_grid, dtype=float)
        c = np.asarray(c, dtype=float)
        if w.ndim != 1 or w.size < MIN_GRID:
            raise ValueError("w_grid must be a 1-D grid with at least two points")
        if c.ndim != 2 or c.shape[1] != w.size:
            raise ValueError("c must have shape (m-1, n_grid) matching w_grid")
        steps = np.diff(w)
        if not np.all(steps > 0) or not np.allclose(steps, steps[0], rtol=1e-9, atol=0.0):
            raise ValueError("w_grid must be strictly increasing and uniform")
        self.w_grid = w
        self.c = c
        self.tie_tol = float(tie_tol)
        self._w0 = float(w[0])
        self._h = float(steps[0])

    def predict(self, cache, k: int, idx: np.ndarray) -> np.ndarray:
        loads = cache.W[np.asarray(idx, dtype=np.intp), k - 1]
        return lin_interp(loads, self._w0, self._h, self.c[k - 1]) + self.tie_tol

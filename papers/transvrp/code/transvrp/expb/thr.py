"""THR: threshold DP under independent stops (EXPERIMENT_B.md 3.1; M13).

Yang, Mathur & Ballou (2000) and Venkateshan & Mathur (NRL 2026, doi 10.1002/nav.70103).
The stops are treated as independent: each stop's increments are resampled from its own
empirical marginal S_j, so the DP ignores the copula dependence of the generator.

Indexing (0-based H and E, 1-based decision epochs k = 1..m-1, as simulate.py):
- continuation after decision k: the next stop is k+1, its breach costs E[k], and a
  handoff at decision j costs H[j-1];
- C_k(w) = mean over g in S_{k+1} of [1{w+g > B} E[k] + 1{w+g <= B} V_{k+1}(w+g)];
- V_j(x) = min(H[j-1], C_j(x)) for j <= m-1 and V_m = 0.
The value at the first stop is E_{g1}[1{g1 > B} E[0] + 1{g1 <= B} V_1(g1)].
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

import numpy as np

from ..expa.features import RouteStats
from ..expa.paths import Paths, RouteSpec
from .grid import GridPredictor, lin_interp, tie_tolerance

BLOCK_ELEMS = 1 << 22      # grid x rows elements per block (bounds float64 temporaries)


@dataclass(frozen=True)
class THRConfig:
    n_grid: int = 2001      # T3: no tuning; 4,001 is a reported refinement


def _grid_bounds(g: np.ndarray, B: float, m: int) -> float:
    """w_lo = min over rows and k of W_k, minus max_j |min S_j| (section 3.1, step 2)."""
    W = np.cumsum(g, axis=1)
    return float(W[:, : m - 1].min() - np.max(np.abs(g.min(axis=0))))


def _continuation(w_grid: np.ndarray, g_next: np.ndarray, B: float, e_k: float,
                  v_next: np.ndarray) -> np.ndarray:
    """C_k on the grid: the mean over the next stop's increments, in grid blocks."""
    h = float(w_grid[1] - w_grid[0])
    w0 = float(w_grid[0])
    out = np.empty(w_grid.size)
    rows = max(1, BLOCK_ELEMS // g_next.size)
    for s in range(0, w_grid.size, rows):
        x = w_grid[s:s + rows, None] + g_next[None, :]
        vals = np.where(x > B, e_k, lin_interp(x, w0, h, v_next))
        out[s:s + rows] = vals.mean(axis=1)
    return out


def independent_dp(spec: RouteSpec, g: np.ndarray, n_grid: int = 2001
                   ) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Returns (w_grid, C, V): C[k] and V[k] on the grid for k = 1..m-1."""
    g = np.asarray(g, dtype=float)
    if g.ndim != 2 or g.shape[1] != spec.m:
        raise ValueError("g must have shape (N, m) matching the route")
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2")
    w_grid = np.linspace(_grid_bounds(g, spec.B, spec.m), spec.B, n_grid)
    C: dict[int, np.ndarray] = {}
    V: dict[int, np.ndarray] = {}
    v_next = np.zeros(n_grid)                  # V_m = 0: completion costs nothing
    for k in range(spec.m - 1, 0, -1):
        C[k] = _continuation(w_grid, g[:, k], spec.B, float(spec.E[k]), v_next)
        V[k] = np.minimum(float(spec.H[k - 1]), C[k])
        v_next = V[k]
    return w_grid, C, V


def thresholds(w_grid: np.ndarray, C: dict[int, np.ndarray], H: np.ndarray,
               tie_tol: float = 0.0) -> dict[int, float]:
    """W*_k = first grid load with C_k(w) >= H[k-1] - tie_tol; +inf if none."""
    out: dict[int, float] = {}
    for k, ck in C.items():
        hit = np.flatnonzero(np.asarray(ck) >= float(H[k - 1]) - tie_tol)
        out[k] = float(w_grid[hit[0]]) if hit.size else float("inf")
    return out


def permute_columns(g: np.ndarray, key: str) -> np.ndarray:
    """Independence resampling (3.8): each column is permuted by its own seeded
    permutation, seed crc32("perm|<key>|<column>"), key = "inst|route|purpose". Column
    marginals are preserved exactly; the dependence between columns is removed."""
    g = np.asarray(g)
    out = np.empty_like(g)
    for j in range(g.shape[1]):
        rng = np.random.default_rng(zlib.crc32(f"perm|{key}|{j}".encode("utf-8")))
        out[:, j] = g[rng.permutation(g.shape[0]), j]
    return out


def _monotone_gap(ck: np.ndarray) -> float:
    """Largest decrease of C_k along the grid (0 when nondecreasing)."""
    return float(max(0.0, -np.min(np.diff(ck)))) if ck.size > 1 else 0.0


def fit_thr(spec: RouteSpec, paths: Paths, N: int, rep: int,
            cfg: THRConfig | None = None) -> tuple[dict, RouteStats, list, dict]:
    """THR on all N training rows (A's baton_eq information class). ``rep`` is the
    replicate of the job and is recorded only in the diagnostics."""
    cfg = THRConfig() if cfg is None else cfg
    if paths.p.shape[0] != N:
        raise ValueError(f"expected {N} training paths, got {paths.p.shape[0]}")
    t0 = time.perf_counter()
    stats = RouteStats.fit(paths)
    tie_tol = tie_tolerance(spec.E0)
    w_grid, C, _ = independent_dp(spec, paths.g, cfg.n_grid)
    table = np.stack([C[k] for k in range(1, spec.m)]) if spec.m > 1 else np.zeros((0, 0))
    predictor = GridPredictor(w_grid, table, tie_tol) if spec.m > 1 else None
    models = {k: [predictor] for k in range(1, spec.m)}
    thr = thresholds(w_grid, C, spec.H, tie_tol)
    diag = [{"k": k, "rep": int(rep), "threshold": thr[k],
             "monotone_gap": _monotone_gap(C[k])} for k in sorted(C)]
    budget = {"unique_paths": int(N), "path_stop_reads": int(N * spec.m),
              "val_reads": 0, "n_grid": int(cfg.n_grid), "n_params": 0,
              "feature_seconds": 0.0, "train_seconds": time.perf_counter() - t0,
              "val_seconds": 0.0, "device": "cpu"}
    return models, stats, diag, budget

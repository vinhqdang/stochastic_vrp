"""ALP-Z: Zhu, Adulyasak & Rousseau's approximate linear programme on a fixed route
(EXPERIMENT_B.md 3.2; M14; METHODS_ALP_PPO.md section 1).

On a fixed route the basis of their ALP is a free value per (stage, load), so the LP is
exact: it is the independent-demand DP on a load lattice of n_w points. Loads are
lattice points w_i = w_lo + i h with w_{n_w-1} = B; each stop's increments are rounded
to multiples of h and binned into a pmf over integer offsets r.

Variables f_k(i), k = 1..m-1 (decision epochs), i = 0..n_w-1, with f_m = 0.
- handoff: f_k(i) <= H[k-1] (variable upper bound);
- continue: f_k(i) <= sum_r p_{k+1}(r) [E[k] 1{i+r > n-1} + 1{i+r <= n-1} f_{k+1}(max(i+r, 0))];
- objective 'initial' (their eq. 24.1): maximise sum_r p_1(r) f_1(i0 + r) over the
  non-breach r; the breach cost at stop 1 is a constant (reported as info["const"]);
- objective 'uniform' (gate B-ALPZ c): maximise the sum of all f_k(i).

Deployment (eq. 29.1): hand off iff L_k(w) >= H[k-1] - tau (T1), where L_k is the
one-step expectation of the LP's f_{k+1}. L_k is tabulated on THR's grid and returned
as a GridPredictor, which applies the common tie rule.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from ..expa.features import RouteStats
from ..expa.paths import Paths, RouteSpec
from .grid import GridPredictor, lin_interp, tie_tolerance
from .thr import THRConfig, _grid_bounds, thresholds

OBJECTIVES = ("initial", "uniform")


@dataclass(frozen=True)
class ALPZConfig:
    n_w: int = 401          # T2
    objective: str = "initial"

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if self.n_w < 2:
            raise ValueError("n_w must be at least 2")


@dataclass(frozen=True)
class Lattice:
    w_lo: float
    h: float
    n_w: int

    @property
    def w(self) -> np.ndarray:
        return self.w_lo + self.h * np.arange(self.n_w)

    @property
    def i0(self) -> int:
        """Lattice index of the initial load w = 0 (nearest node)."""
        return int(round((0.0 - self.w_lo) / self.h))


def make_lattice(spec: RouteSpec, g: np.ndarray, n_w: int) -> Lattice:
    """Lattice on [w_lo, B] with the THR bounds (3.1, step 2) and n_w points."""
    if n_w < 2:
        raise ValueError("n_w must be at least 2")
    w_lo = _grid_bounds(np.asarray(g, dtype=float), spec.B, spec.m)
    if not spec.B > w_lo:
        raise ValueError("B must exceed the lattice's lowest load")
    return Lattice(w_lo=w_lo, h=(spec.B - w_lo) / (n_w - 1), n_w=int(n_w))


def lattice_pmf(g: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Binned empirical pmfs of the rounded increments.

    Returns (offsets (R,), probs (m, R)): probs[j-1, r] = P(round(g_j / h) = offsets[r]),
    the empirical frequency over the rows of g. Offsets are the union over stops."""
    if h <= 0:
        raise ValueError("h must be positive")
    g = np.asarray(g, dtype=float)
    n, m = g.shape
    r = np.rint(g / h).astype(np.int64)
    offsets = np.unique(r)
    pos = np.searchsorted(offsets, r)
    probs = np.zeros((m, offsets.size))
    for j in range(m):
        probs[j] = np.bincount(pos[:, j], minlength=offsets.size) / n
    return offsets, probs


def _positive(pmf_row: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = pmf_row > 0
    return offsets[keep], pmf_row[keep]


def build_lp(spec: RouteSpec, pmfs, lat: Lattice, cfg: ALPZConfig
             ) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray, dict]:
    """LP in the form maximise c @ x, A x <= b, x <= ub (x = vec of f, stage-major).

    Returns (c, A, b, info); info holds the variable upper bounds ``ub``, the constant
    ``const`` (breach cost at stop 1, omitted from the initial objective), and n_vars."""
    offsets, probs = np.asarray(pmfs[0], dtype=np.int64), np.asarray(pmfs[1], dtype=float)
    K, n = spec.m - 1, lat.n_w
    idx = np.arange(n)
    rows, cols, vals = [], [], []
    b = np.zeros(K * n)
    for k in range(1, K + 1):
        off, pk = _positive(probs[k], offsets)
        J = idx[:, None] + off[None, :]
        breach = J > n - 1
        r0 = (k - 1) * n
        rows.append(r0 + idx)
        cols.append(r0 + idx)
        vals.append(np.ones(n))
        b[r0:r0 + n] = spec.E[k] * (breach * pk[None, :]).sum(axis=1)
        if k < K:
            ii, rr = np.nonzero(~breach)
            rows.append(r0 + ii)
            cols.append(k * n + np.maximum(J[ii, rr], 0))
            vals.append(-pk[rr])
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(K * n, K * n))
    c = np.zeros(K * n)
    const = 0.0
    if cfg.objective == "uniform":
        c[:] = 1.0
    else:
        off1, p1 = _positive(probs[0], offsets)
        J1 = lat.i0 + off1
        ok = J1 <= n - 1
        np.add.at(c, np.maximum(J1[ok], 0), p1[ok])
        const = float(spec.E[0] * p1[~ok].sum())
    ub = np.repeat(np.asarray(spec.H[:K], dtype=float), n)
    info = {"ub": ub, "const": const, "n_vars": K * n, "objective": cfg.objective}
    return c, A, b, info


def solve_lp(c: np.ndarray, A: sp.csr_matrix, b: np.ndarray, info: dict
             ) -> tuple[np.ndarray, int, float]:
    """HiGHS dual simplex (scipy linprog, presolve on). Returns (x, status, seconds).
    scipy 1.15.3 exposes no random_seed option; determinism is checked by re-solving."""
    t0 = time.perf_counter()
    res = linprog(-np.asarray(c, dtype=float), A_ub=A, b_ub=b,
                  bounds=[(None, float(u)) for u in info["ub"]], method="highs-ds",
                  options={"presolve": True})
    seconds = time.perf_counter() - t0
    x = res.x if res.x is not None else np.full(info["n_vars"], np.nan)
    return np.asarray(x, dtype=float), int(res.status), seconds


def _dp_values(spec: RouteSpec, offsets: np.ndarray, probs: np.ndarray, lat: Lattice
               ) -> np.ndarray:
    K, n = spec.m - 1, lat.n_w
    idx = np.arange(n)
    out = np.zeros((K, n))
    v_next = np.zeros(n)                       # f_m = 0
    for k in range(K, 0, -1):
        off, pk = _positive(probs[k], offsets)
        J = idx[:, None] + off[None, :]
        cont = np.where(J > n - 1, spec.E[k], v_next[np.clip(J, 0, n - 1)])
        v_next = np.minimum(spec.H[k - 1], cont @ pk)
        out[k - 1] = v_next
    return out


def lattice_dp(spec: RouteSpec, pmfs, lat: Lattice) -> np.ndarray:
    """Exact backward recursion on the lattice: (m-1, n_w) array of V_k(i)."""
    return _dp_values(spec, np.asarray(pmfs[0], dtype=np.int64),
                      np.asarray(pmfs[1], dtype=float), lat)


def initial_value_dp(spec: RouteSpec, pmfs, lat: Lattice) -> float:
    """DP value at the initial state, including the breach cost at stop 1."""
    offsets, probs = np.asarray(pmfs[0], dtype=np.int64), np.asarray(pmfs[1], dtype=float)
    V = _dp_values(spec, offsets, probs, lat)
    off1, p1 = _positive(probs[0], offsets)
    J1 = lat.i0 + off1
    n = lat.n_w
    return float(np.where(J1 > n - 1, spec.E[0], V[0][np.clip(J1, 0, n - 1)]) @ p1)


def continuation_table(spec: RouteSpec, pmfs, lat: Lattice, f: np.ndarray,
                       w_grid: np.ndarray) -> np.ndarray:
    """L_k(w) of eq. 29.1 on ``w_grid``: (m-1, len(w_grid)), row k-1 for epoch k.
    f is the LP vertex (m-1, n_w); f_{k+1} is read by linear interpolation."""
    offsets, probs = np.asarray(pmfs[0], dtype=np.int64), np.asarray(pmfs[1], dtype=float)
    K = spec.m - 1
    w = np.asarray(w_grid, dtype=float)
    out = np.zeros((K, w.size))
    for k in range(1, K + 1):
        off, pk = _positive(probs[k], offsets)
        X = w[:, None] + lat.h * off[None, :]
        if k < K:
            nxt = lin_interp(X, lat.w_lo, lat.h, f[k])
        else:
            nxt = np.zeros_like(X)
        out[k - 1] = np.where(X > spec.B, spec.E[k], nxt) @ pk
    return out


def deploy_predictor(spec: RouteSpec, pmfs, lat: Lattice, f: np.ndarray,
                     w_grid: np.ndarray) -> GridPredictor:
    """Deployed ALP-Z predictor: L_k tabulated on ``w_grid`` with the common tie rule T1."""
    table = continuation_table(spec, pmfs, lat, f, w_grid)
    return GridPredictor(w_grid, table, tie_tolerance(spec.E0))


def _vertex_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


def fit_alpz(spec: RouteSpec, paths: Paths, N: int, rep: int,
             cfg: ALPZConfig | None = None) -> tuple[dict, RouteStats, list, dict]:
    """ALP-Z on all N training rows (THR's information class). ``rep`` is recorded only
    in the diagnostics."""
    cfg = ALPZConfig() if cfg is None else cfg
    if paths.p.shape[0] != N:
        raise ValueError(f"expected {N} training paths, got {paths.p.shape[0]}")
    t0 = time.perf_counter()
    stats = RouteStats.fit(paths)
    g = paths.g
    K = spec.m - 1
    lat = make_lattice(spec, g, cfg.n_w)
    pmfs = lattice_pmf(g, lat.h)
    c, A, b, info = build_lp(spec, pmfs, lat, cfg)
    x, status, lp_seconds = solve_lp(c, A, b, info)
    if status != 0:
        raise RuntimeError(f"ALP-Z LP returned status {status}")
    f = x.reshape(K, lat.n_w)
    tie = tie_tolerance(spec.E0)
    V = lattice_dp(spec, pmfs, lat)
    thr_cfg = THRConfig()
    w_grid = np.linspace(lat.w_lo, spec.B, thr_cfg.n_grid)
    predictor = deploy_predictor(spec, pmfs, lat, f, w_grid)
    table = predictor.c
    models = {k: [predictor] for k in range(1, spec.m)}
    thr = thresholds(w_grid, {k: table[k - 1] for k in range(1, spec.m)}, spec.H, tie)
    vhash = _vertex_hash(x)
    diag = [{"k": k, "rep": int(rep), "threshold": thr[k], "lp_status": status,
             "vertex_sha256": vhash,
             "offpath_share": float(np.mean(f[k - 1] < V[k - 1] - tie))}
            for k in range(1, spec.m)]
    budget = {"unique_paths": int(N), "path_stop_reads": int(N * spec.m),
              "val_reads": 0, "lp_rows": int(A.shape[0]), "lp_cols": int(A.shape[1]),
              "lp_nnz": int(A.nnz), "lp_seconds": float(lp_seconds),
              "lp_objective": float(c @ x + info["const"]), "n_grid": int(thr_cfg.n_grid),
              "n_params": 0, "feature_seconds": 0.0,
              "train_seconds": time.perf_counter() - t0, "val_seconds": 0.0,
              "device": "cpu"}
    return models, stats, diag, budget

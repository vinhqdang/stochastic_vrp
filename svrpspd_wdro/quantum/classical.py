"""Exact peak-load distributions for SVRPSPD routes on discrete demands.

Model
-----
A route is a sequence of customer indices (i_1, ..., i_L); the depot is
implicit. Each customer i has an independent random delivery d_i and pickup
p_i, each with a finite pmf on integer levels 0..V-1 (level v means
v * unit demand units). The load profile is

    L_0 = sum_k d_{i_k},    L_k = L_{k-1} - d_{i_k} + p_{i_k},

and the peak is F = max_{0 <= k <= L} L_k.

Exact peak pmf
--------------
With A_k = sum_{l>k} d_{i_l} and G_k = max_{l>=k} (A_l + sum_{k<m<=l} p_{i_m})
one has F = G_0, and the backward recursion (A_L, G_L) = (0, 0),

    A_{k-1} = A_k + d_{i_k},
    G_{k-1} = max(d_{i_k} + A_k, p_{i_k} + G_k),

propagates the joint pmf of (A, G) from the last stop to the first. The
marginal of G_0 is the exact pmf of F.

Since d_{i_k} + A_k = A_{k-1}, the G-step reads G_{k-1} = max(A_{k-1},
p_{i_k} + G_k). ``peak_distribution`` applies the two sub-steps (delivery
shift of A, then pickup-and-clip of G) as shifted array additions and a
cumulative sum, which costs O(V * M^2) per stop instead of the
O(V^2 * M^2) of a direct enumeration of (d, p) pairs. The result is the
same joint pmf.

Insertion lemma
---------------
Inserting customer j into a route r at any position gives a route r' with
F_r <= F_{r'} <= F_r + max(d_j, p_j) scenario by scenario. Hence the event
{1[F_{r'} > Q] != 1[F_r > Q]} equals {F_r <= Q < F_{r'}}.

Conventions
-----------
Levels are rounded with ``np.rint``, which breaks ties to the nearest even
integer. CDF comparisons in :func:`cvar` use the tolerance ``_CDF_TOL``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

_MAX_ENUM = 2_000_000
_TOL = 1e-9
_CDF_TOL = 1e-12


# ============================================================
# Load profile
# ============================================================


def peak_load(route: Sequence[int], d, p):
    """Peak load F = max_k L_k of one route under one demand scenario.

    Parameters
    ----------
    route : sequence of int
        Customer indices in visiting order (depot implicit).
    d, p : indexable by customer
        Delivery and pickup realisations (same units as the result).

    Returns
    -------
    int or float
        Peak load, in the units of ``d`` and ``p``.
    """
    load = 0
    for c in route:
        load = load + d[c]
    peak = load
    for c in route:
        load = load - d[c] + p[c]
        if load > peak:
            peak = load
    if isinstance(peak, np.generic):
        return peak.item()
    return peak


# ============================================================
# Discrete demand model
# ============================================================


@dataclass
class DiscreteDemand:
    """Independent discrete delivery and pickup pmfs for every customer.

    Attributes
    ----------
    pmf_d : np.ndarray, shape (n_customers, V)
        pmf_d[i, v] = P(d_i = v * unit).
    pmf_p : np.ndarray, shape (n_customers, V)
        pmf_p[i, v] = P(p_i = v * unit).
    unit : float
        Demand units per level.
    """

    pmf_d: np.ndarray
    pmf_p: np.ndarray
    unit: float

    def __post_init__(self):
        self.pmf_d = np.asarray(self.pmf_d, dtype=np.float64)
        self.pmf_p = np.asarray(self.pmf_p, dtype=np.float64)
        if self.pmf_d.ndim != 2 or self.pmf_d.shape != self.pmf_p.shape:
            raise ValueError(
                "pmf_d and pmf_p must be 2-D with the same shape, got "
                f"{self.pmf_d.shape} and {self.pmf_p.shape}"
            )
        if self.pmf_d.shape[1] < 1:
            raise ValueError("pmfs must have at least one level")
        for name, pmf in (("pmf_d", self.pmf_d), ("pmf_p", self.pmf_p)):
            if np.any(pmf < 0):
                raise ValueError(f"{name} has negative entries")
            if not np.allclose(pmf.sum(axis=1), 1.0, atol=_TOL, rtol=0.0):
                raise ValueError(f"rows of {name} must sum to 1")
        if not self.unit > 0:
            raise ValueError(f"unit must be positive, got {self.unit}")

    @property
    def n_customers(self) -> int:
        return self.pmf_d.shape[0]

    @property
    def levels(self) -> int:
        """Number of levels V."""
        return self.pmf_d.shape[1]


def _check_samples(samples: np.ndarray) -> np.ndarray:
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("samples must be a non-empty 1-D array")
    if np.any(s < 0) or not np.all(np.isfinite(s)):
        raise ValueError("samples must be finite and non-negative")
    return s


def discretize_samples(
    samples: np.ndarray, b: int, unit: float | None = None
) -> tuple[np.ndarray, float]:
    """Empirical pmf over V = 2**b levels for one customer's demand samples.

    Each sample is divided by ``unit``, rounded to the nearest integer
    (``np.rint``: ties go to the nearest even integer) and clipped at V - 1.
    If ``unit`` is None it is set to max(samples)/(V-1) (or 1 if the maximum
    is 0), so the largest sample maps to level V - 1.

    Returns
    -------
    pmf : np.ndarray, shape (V,)
    unit : float
    """
    s = _check_samples(samples)
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    V = 2 ** b
    if unit is None:
        mx = float(s.max())
        unit = mx / (V - 1) if mx > 0 else 1.0
    elif not unit > 0:
        raise ValueError(f"unit must be positive, got {unit}")
    levels = np.minimum(np.rint(s / unit).astype(np.int64), V - 1)
    pmf = np.bincount(levels, minlength=V).astype(np.float64) / s.size
    return pmf, float(unit)


def discretize_instance(
    X: np.ndarray, n: int, b: int, unit: float | None = None
) -> DiscreteDemand:
    """Discretise an N x 2n scenario matrix with one common unit.

    Column convention (as in ``core.scenarios``): X[:, 2j] = d_j and
    X[:, 2j+1] = p_j. If ``unit`` is None it is set to
    max(X) / (V - 1) over the whole matrix (1 if the maximum is 0).
    Rounding follows :func:`discretize_samples` (ties to even).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 2 * n:
        raise ValueError(f"X must have shape (N, {2 * n}), got {X.shape}")
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    V = 2 ** b
    if unit is None:
        mx = float(X.max())
        unit = mx / (V - 1) if mx > 0 else 1.0
    pmf_d = np.empty((n, V), dtype=np.float64)
    pmf_p = np.empty((n, V), dtype=np.float64)
    for j in range(n):
        pmf_d[j], _ = discretize_samples(X[:, 2 * j], b, unit=unit)
        pmf_p[j], _ = discretize_samples(X[:, 2 * j + 1], b, unit=unit)
    return DiscreteDemand(pmf_d=pmf_d, pmf_p=pmf_p, unit=float(unit))


# ============================================================
# Exact peak-load pmf
# ============================================================


def peak_distribution(route: Sequence[int], dem: DiscreteDemand) -> np.ndarray:
    """Exact pmf of the peak load F, in level units (index = level).

    Backward dynamic program over the joint state (A, G); see the module
    docstring. The returned array has length L * (V - 1) + 1, where L is the
    route length, which bounds F.

    The joint array is kept at its active size: after j stops A and G are at
    most j * (V - 1), so the array grows by V - 1 rows and columns per stop.
    """
    seq = [int(c) for c in route]
    V = dem.levels
    step = V - 1
    P = np.ones((1, 1), dtype=np.float64)  # (A, G) = (0, 0) at k = L
    m = 1
    for c in reversed(seq):
        wd = dem.pmf_d[c]
        wp = dem.pmf_p[c]
        m_new = m + step
        Pn = np.zeros((m_new, m_new), dtype=np.float64)
        Pn[:m, :m] = P

        # Delivery sub-step: A' = A + d.
        Q = np.zeros((m_new, m_new), dtype=np.float64)
        for dv in range(V):
            if wd[dv] != 0.0:
                Q[dv:, :] += wd[dv] * Pn[: m_new - dv, :]

        # Pickup sub-step: G' = max(A', G + p).
        # Entries with G + p > A' move to G + p; entries with G + p <= A' are
        # clipped onto the diagonal G' = A' (cumulative-sum term below).
        R_off = np.zeros((m_new, m_new), dtype=np.float64)
        for pv in range(V):
            if wp[pv] != 0.0:
                R_off[:, pv:] += wp[pv] * Q[:, : m_new - pv]
        idx = np.arange(m_new)
        R = np.where(idx[None, :] > idx[:, None], R_off, 0.0)
        C = np.cumsum(Q, axis=1)
        for pv in range(V):
            if wp[pv] != 0.0 and pv < m_new:
                rows = idx[pv:]
                R[rows, rows] += wp[pv] * C[rows, rows - pv]
        P = R
        m = m_new
    return P.sum(axis=0)


def chance(route: Sequence[int], dem: DiscreteDemand, Q_level: int) -> float:
    """P(F > Q_level) for the peak load F, with Q_level in level units.

    A non-integer Q_level is handled as floor(Q_level), which is exact because
    F takes integer levels.
    """
    pmf = peak_distribution(route, dem)
    start = int(math.floor(Q_level)) + 1
    if start <= 0:
        return float(pmf.sum())
    return float(pmf[start:].sum())


def _cvar_from_pmf(pmf: np.ndarray, alpha: float) -> float:
    """CVaR_alpha of a pmf on levels 0..len(pmf)-1 (level units).

    VaR_alpha is the smallest level x with P(F <= x) >= alpha, up to the
    tolerance ``_CDF_TOL``; if no level qualifies (round-off in the total mass)
    the last level is used. CVaR = VaR + E[(F - VaR)^+] / (1 - alpha).
    """
    cdf = np.cumsum(pmf)
    hits = np.flatnonzero(cdf >= alpha - _CDF_TOL)
    var = int(hits[0]) if hits.size else int(pmf.size - 1)
    levels = np.arange(pmf.size)
    tail = float(np.sum(pmf * np.maximum(levels - var, 0)))
    return float(var + tail / (1.0 - alpha))


def cvar(route: Sequence[int], dem: DiscreteDemand, alpha: float) -> float:
    """CVaR_alpha of the peak load F, in level units.

    VaR_alpha is the smallest level x with P(F <= x) >= alpha, and
    CVaR_alpha = VaR_alpha + E[(F - VaR_alpha)^+] / (1 - alpha).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    return _cvar_from_pmf(peak_distribution(route, dem), alpha)


# ============================================================
# Single-insertion window (small cases, exact enumeration)
# ============================================================


def _peak_vec(seq: Sequence[int], d_lv: Mapping, p_lv: Mapping, n_rows: int):
    """Vectorised peak over a batch of scenarios (integer level arrays)."""
    load = np.zeros(n_rows, dtype=np.int64)
    for c in seq:
        load = load + d_lv[c]
    peak = load.copy()
    for c in seq:
        load = load - d_lv[c] + p_lv[c]
        peak = np.maximum(peak, load)
    return peak


def insertion_window(
    route: Sequence[int], j: int, pos: int, dem: DiscreteDemand, Q_level: float
) -> float:
    """Exact P(1[F_{r'} > Q] != 1[F_r > Q]) for inserting customer j.

    r' is ``route`` with customer j inserted so that it occupies 1-indexed
    position ``pos`` in the new route (same convention as ``Route.insert``).
    The probability is computed by enumerating every demand combination of
    the customers in r', so it is restricted to small cases:
    V ** (2 * (len(route) + 1)) must not exceed 2,000,000.
    """
    seq = [int(c) for c in route]
    j = int(j)
    if j in seq:
        raise ValueError(f"Customer {j} already in route")
    if not 1 <= pos <= len(seq) + 1:
        raise IndexError(f"Insertion position {pos} out of range [1, {len(seq) + 1}]")
    V = dem.levels
    if V ** (2 * (len(seq) + 1)) > _MAX_ENUM:
        raise ValueError(
            f"insertion_window enumeration too large: V={V}, "
            f"route length={len(seq)} exceeds {_MAX_ENUM} scenarios"
        )
    new_seq = seq[: pos - 1] + [j] + seq[pos - 1 :]
    custs = sorted(set(seq) | {j})
    K = 2 * len(custs)
    grid = np.indices((V,) * K, dtype=np.int64).reshape(K, -1).T
    n_rows = grid.shape[0]
    weight = np.ones(n_rows, dtype=np.float64)
    d_lv: dict[int, np.ndarray] = {}
    p_lv: dict[int, np.ndarray] = {}
    for t, c in enumerate(custs):
        d_lv[c] = grid[:, 2 * t]
        p_lv[c] = grid[:, 2 * t + 1]
        weight *= dem.pmf_d[c][d_lv[c]] * dem.pmf_p[c][p_lv[c]]
    F_r = _peak_vec(seq, d_lv, p_lv, n_rows)
    F_new = _peak_vec(new_seq, d_lv, p_lv, n_rows)
    flip = (F_new > Q_level) != (F_r > Q_level)
    return float(np.sum(weight[flip]))

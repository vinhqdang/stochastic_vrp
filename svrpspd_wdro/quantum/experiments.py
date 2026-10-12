"""Experiment helpers for QUASAR M3 (design: papers/quasar/DESIGN_M3.md, Revisions 2 and 3).

M3a (v2): insertion, relocate, swap, 2-opt and alternative-position moves of
every cached plan, evaluated on common scenarios of the BATON demand model
(one-factor Gaussian copula, rho = 0.6, gamma marginals, cv = 0.3, drawn by
``scripts/dethloff_runner.make_scenarios``). Per move: p, p', P(D = 1),
P(D = -1), Var(D), rho_q, rho_c with delta-method standard errors, and the
query counts of canonical amplitude estimation and classical Monte Carlo.

Discretisation (for the circuits): unit u = Q / (K + 1/2) with K the smallest
integer such that K + 1/2 >= Q (V - 1) / q_0.999^max, threshold level K, and
nearest-level rounding. The copula enters the discrete model through its
conditional pmfs given the two factor values (Gauss-Hermite nodes); the window
probability is the exact double sum over node pairs of ``classical.chance``.

M3b helpers: maximum-likelihood amplitude estimation with a configurable
Grover schedule (``mlae_estimate``) and move statistics on the discrete
independent model (``move_stats``, used by part b).

Demand model
------------
Gamma marginals with shape 1/cv^2 and scale mean*cv^2 (as in the runner).
Deliveries and pickups each carry one common factor (independent of each
other); given the factor z, customers are independent with pmf
P(D_i <= x | z) = Phi((Phi^{-1}(F_i(x)) - sqrt(rho) z) / sqrt(1 - rho)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.stats as st
from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator
from qiskit_algorithms import EstimationProblem, MaximumLikelihoodAmplitudeEstimation

from quantum.classical import DiscreteDemand, chance
from quantum.oracle import PeakOracle, _make_sampler

# Defaults of scripts/dethloff_runner.py (CV = 0.30, DIST = "gamma", RHO = 0.6).
DEFAULT_CV = 0.30
DEFAULT_DIST = "gamma"
DEFAULT_RHO = 0.6

_RHO_TOL = 1e-12
_Q999 = 0.999

# Query-count constants (design, Revision 2).
AE_REPS = 5          # repetitions of canonical AE; the median is taken
Z_95 = 1.96          # normal quantile for the classical sample count


# ============================================================
# Demand discretisation
# ============================================================


def _gamma_shape_scale(mean: float, cv: float) -> tuple[float, float]:
    k = 1.0 / (cv * cv)
    return k, mean * cv * cv


def gamma_level_pmf(mean: float, cv: float, unit: float, V: int) -> np.ndarray:
    """Level pmf of a gamma demand with the given mean and cv on V levels.

    Level v >= 1 carries [(v - 1/2) u, (v + 1/2) u); level 0 carries [0, u/2);
    the top level V - 1 carries the whole upper tail [(V - 3/2) u, inf). The
    masses are differences of the exact gamma cdf, so they telescope to one.
    """
    V = int(V)
    if V < 2:
        raise ValueError(f"V must be >= 2, got {V}")
    if not unit > 0:
        raise ValueError(f"unit must be positive, got {unit}")
    if not cv > 0:
        raise ValueError(f"cv must be positive, got {cv}")
    mean = float(mean)
    if mean < 0:
        raise ValueError(f"mean must be non-negative, got {mean}")
    pmf = np.zeros(V, dtype=np.float64)
    if mean == 0.0:
        pmf[0] = 1.0
        return pmf
    k, scale = _gamma_shape_scale(mean, cv)
    bounds = np.empty(V + 1, dtype=np.float64)
    bounds[0] = 0.0
    bounds[1:V] = (np.arange(1, V) - 0.5) * unit
    bounds[V] = np.inf
    cdf = st.gamma.cdf(bounds, k, scale=scale)
    cdf[0], cdf[-1] = 0.0, 1.0
    pmf = np.clip(np.diff(cdf), 0.0, None)
    return pmf / pmf.sum()


def _plan_arrays(plan_json: dict) -> tuple[np.ndarray, np.ndarray, int, float]:
    dbar = np.asarray(plan_json["dbar"], dtype=np.float64)
    pbar = np.asarray(plan_json["pbar"], dtype=np.float64)
    n = int(plan_json["n"])
    Q = float(plan_json["Q"])
    if dbar.shape != (n,) or pbar.shape != (n,):
        raise ValueError(f"dbar/pbar must have length n={n}, got {dbar.shape}, {pbar.shape}")
    return dbar, pbar, n, Q


def q999_max(dbar: np.ndarray, pbar: np.ndarray, cv: float = DEFAULT_CV) -> float:
    """Largest 0.999 quantile over all deliveries and pickups of the instance."""
    q_max = 0.0
    for mu in np.concatenate([np.asarray(dbar, float), np.asarray(pbar, float)]):
        if mu > 0:
            k, scale = _gamma_shape_scale(float(mu), cv)
            q_max = max(q_max, float(st.gamma.ppf(_Q999, k, scale=scale)))
    return q_max


LEVEL_RULES = ("design", "nearest")


def level_rule(Q: float, V: int, q_max: float, rule: str = "design") -> tuple[int, float]:
    """Threshold level K and unit u = Q / (K + 1/2) (Revision 2).

    rule = "design" (default): K is the smallest integer with
    K + 1/2 >= Q (V - 1) / q_max. rule = "nearest" (review variant):
    K = round(Q (V - 1) / q_max - 1/2), the nearest half-level. In both cases
    the threshold sits on a half-level boundary and K is at least 1.
    """
    if rule not in LEVEL_RULES:
        raise ValueError(f"rule must be one of {LEVEL_RULES}, got {rule!r}")
    if not Q > 0:
        raise ValueError(f"Q must be positive, got {Q}")
    if not q_max > 0:
        raise ValueError(f"q_max must be positive, got {q_max}")
    x = Q * (V - 1) / q_max
    if rule == "design":
        K = max(1, math.ceil(x - 0.5))
    else:
        K = max(1, int(round(x - 0.5)))
    return K, Q / (K + 0.5)


def instance_demand(
    plan_json: dict, b: int, cv: float = DEFAULT_CV
) -> tuple[DiscreteDemand, float, int]:
    """Independent discrete demand model of one instance with V = 2**b levels.

    Unit and threshold follow :func:`level_rule`; the nearest-level binning
    of :func:`gamma_level_pmf` is unchanged.

    Returns
    -------
    dem : DiscreteDemand
        Rows indexed by node id 0..n-1; the depot row is the point mass at 0.
    unit : float
        Demand units per level, Q / (Q_level + 1/2).
    Q_level : int
        Threshold level K (the window is F_level > K).
    """
    dbar, pbar, n, Q = _plan_arrays(plan_json)
    b = int(b)
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    V = 2**b
    q_max = q999_max(dbar, pbar, cv)
    Q_level, unit = level_rule(Q, V, q_max) if q_max > 0 else (1, Q / 1.5)
    pmf_d = np.stack([gamma_level_pmf(m, cv, unit, V) for m in dbar])
    pmf_p = np.stack([gamma_level_pmf(m, cv, unit, V) for m in pbar])
    dem = DiscreteDemand(pmf_d=pmf_d, pmf_p=pmf_p, unit=float(unit))
    return dem, float(unit), int(Q_level)


# ============================================================
# Copula: conditional pmfs given the factor value
# ============================================================


def gh_nodes(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes and weights for a standard normal factor.

    Weights are normalised to sum to one, so sum_t w_t g(z_t) approximates
    E[g(Z)] for Z ~ N(0, 1).
    """
    n_nodes = int(n_nodes)
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    z, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    return z.astype(np.float64), (w / w.sum()).astype(np.float64)


def equal_mass_grid(T: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-probability factor grid with T bins (Revision 3).

    Bin t carries probability 1/T and is represented by its probability
    midpoint u_t = (t + 1/2) / T, so the factor value is z_t = Phi^{-1}(u_t)
    with equal weights w_t = 1/T. For T = 2^f the register is a uniform
    superposition over the f qubits (Hadamards only).
    """
    T = int(T)
    if T < 1:
        raise ValueError(f"T must be positive, got {T}")
    u = (np.arange(T, dtype=np.float64) + 0.5) / T
    return st.norm.ppf(u).astype(np.float64), np.full(T, 1.0 / T, dtype=np.float64)


EM_AVG_INNER = 32     # inner Gauss-Legendre nodes per bin (16 gives 2.5e-12 at 128 bins; 32 gives 2e-15)
EM_AVG_TRUNC = 9.0    # outer bins are cut at |z| = 9 (mass beyond it is below 1e-18)


def bin_averaged_nodes(T: int, m: int = EM_AVG_INNER) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inner quadrature of the bin averages over T equal-probability bins (em_avg).

    For bin t, the average of a function g(z) over the bin under the standard
    normal law is (T * integral over bin t of g(z) phi(z) dz). It is evaluated
    with m-point Gauss-Legendre in z on the bin [z_t, z_{t+1}], z_t = Phi^{-1}(t / T),
    with the outer bins cut at +-EM_AVG_TRUNC, and the weights are normalised
    within the bin (they sum to one). The rule is taken in z rather than in
    u = Phi(z): the integrand in z is smooth on each finite bin, whereas in u it
    has algebraic endpoint behaviour in the outer bins.

    Returns
    -------
    z_mid : (T,) probability-midpoint values Phi^{-1}((t + 1/2) / T) (reporting only)
    z_nodes : (T, m) inner nodes
    w_nodes : (T, m) weights, each row summing to one
    """
    T = int(T)
    m = int(m)
    if T < 1 or m < 1:
        raise ValueError("T and m must be positive")
    edges = st.norm.ppf(np.arange(T + 1, dtype=np.float64) / T)
    edges[0], edges[-1] = -EM_AVG_TRUNC, EM_AVG_TRUNC
    x, wgl = np.polynomial.legendre.leggauss(m)
    lo, hi = edges[:-1, None], edges[1:, None]
    z_nodes = 0.5 * (lo + hi) + 0.5 * (hi - lo) * x[None, :]
    raw = wgl[None, :] * 0.5 * (hi - lo) * st.norm.pdf(z_nodes)
    w_nodes = raw / raw.sum(axis=1, keepdims=True)
    z_mid = st.norm.ppf((np.arange(T) + 0.5) / T)
    return z_mid.astype(np.float64), z_nodes.astype(np.float64), w_nodes.astype(np.float64)


FACTOR_GRIDS = ("gh", "em", "em_avg")


def factor_grid(kind: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Factor nodes and weights.

    "gh" = Gauss-Hermite nodes; "em" = equal-mass bins at their midpoints;
    "em_avg" = the same bins, returned with their midpoints for reference only
    (the pmfs of an em_avg model are bin averages, see :func:`copula_demand`).
    """
    if kind == "gh":
        return gh_nodes(size)
    if kind in ("em", "em_avg"):
        return equal_mass_grid(size)
    raise ValueError(f"grid must be one of {FACTOR_GRIDS}, got {kind!r}")


def conditional_level_pmf(
    mean: float, cv: float, unit: float, V: int, z: np.ndarray, rho: float
) -> np.ndarray:
    """pmf of one gamma demand on V levels conditional on the copula factor.

    Under the one-factor Gaussian copula U_i = Phi(sqrt(rho) Z0 + sqrt(1-rho)
    eps_i), the cdf of D_i given Z0 = z is
        P(D_i <= x | z) = Phi((Phi^{-1}(F_i(x)) - sqrt(rho) z) / sqrt(1 - rho)),
    and the level bins are those of :func:`gamma_level_pmf`.

    Returns
    -------
    np.ndarray, shape (len(z), V); each row sums to one.
    """
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    V = int(V)
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must lie in [0, 1), got {rho}")
    out = np.zeros((z.size, V), dtype=np.float64)
    if float(mean) == 0.0:
        out[:, 0] = 1.0
        return out
    k, scale = _gamma_shape_scale(float(mean), cv)
    interior = (np.arange(1, V) - 0.5) * unit
    F = st.gamma.cdf(interior, k, scale=scale)
    a = st.norm.ppf(np.clip(F, 0.0, 1.0))  # -inf / +inf at the extremes
    s, t = math.sqrt(rho), math.sqrt(1.0 - rho)
    G = np.zeros((z.size, V + 1), dtype=np.float64)
    G[:, 1:V] = st.norm.cdf((a[None, :] - s * z[:, None]) / t)
    G[:, V] = 1.0
    pmf = np.clip(np.diff(G, axis=1), 0.0, None)
    return pmf / pmf.sum(axis=1, keepdims=True)


@dataclass
class CopulaDemand:
    """Discrete copula model of one instance for the exact node-pair sum.

    pmf_d[t] and pmf_p[s] are the independent per-customer pmfs given the
    delivery factor z_t and the pickup factor z_s; w are the (normalised)
    Gauss-Hermite weights shared by both factors.
    """

    pmf_d: np.ndarray  # (T, n, V)
    pmf_p: np.ndarray  # (T, n, V)
    z: np.ndarray      # (T,)
    w: np.ndarray      # (T,)
    unit: float
    Q_level: int
    rho: float
    b: int

    def demand(self, t: int, s: int) -> DiscreteDemand:
        return DiscreteDemand(pmf_d=self.pmf_d[t], pmf_p=self.pmf_p[s], unit=self.unit)


def copula_demand(plan_json: dict, b: int, rho: float = DEFAULT_RHO, n_nodes: int = 10,
                  cv: float = DEFAULT_CV, rule: str = "design", grid: str = "gh") -> CopulaDemand:
    """Copula discrete model of an instance at V = 2**b levels (see module doc).

    grid = "gh": n_nodes Gauss-Hermite nodes per factor; grid = "em": n_nodes
    equal-probability bins per factor, conditional pmfs at the bin midpoints
    (:func:`equal_mass_grid`); grid = "em_avg": the same bins with the
    conditional pmfs averaged over each bin (:func:`bin_averaged_nodes`),
    P(level | t) = T * integral over bin t of P(level | z) phi(z) dz.
    """
    dbar, pbar, n, Q = _plan_arrays(plan_json)
    V = 2 ** int(b)
    q_max = q999_max(dbar, pbar, cv)
    Q_level, unit = level_rule(Q, V, q_max, rule)
    if grid == "em_avg":
        z, nodes, W = bin_averaged_nodes(n_nodes)
        w = np.full(z.size, 1.0 / z.size, dtype=np.float64)
        pmf_d = _bin_averaged_pmfs(dbar, cv, unit, V, nodes, W, rho)
        pmf_p = _bin_averaged_pmfs(pbar, cv, unit, V, nodes, W, rho)
    else:
        z, w = factor_grid(grid, n_nodes)
        pmf_d = np.stack([conditional_level_pmf(m, cv, unit, V, z, rho) for m in dbar], axis=1)
        pmf_p = np.stack([conditional_level_pmf(m, cv, unit, V, z, rho) for m in pbar], axis=1)
    return CopulaDemand(pmf_d=pmf_d, pmf_p=pmf_p, z=z, w=w, unit=float(unit),
                        Q_level=int(Q_level), rho=float(rho), b=int(b))


def _bin_averaged_pmfs(means: np.ndarray, cv: float, unit: float, V: int,
                       nodes: np.ndarray, W: np.ndarray, rho: float) -> np.ndarray:
    """(T, n, V): per-customer conditional pmfs averaged over each factor bin.

    nodes, W: (T, m) inner nodes and bin-normalised weights from
    :func:`bin_averaged_nodes`.
    """
    T, m = nodes.shape
    cond = np.stack([conditional_level_pmf(mu, cv, unit, V, nodes.ravel(), rho).reshape(T, m, V)
                     for mu in means], axis=1)          # (T, n, m, V)
    return np.einsum("tnjv,tj->tnv", cond, W)


def tail_over_pickups(route: Sequence[int], pmf_d: np.ndarray, pmf_p_batch: np.ndarray,
                      K: int) -> np.ndarray:
    """P(F > K) for one delivery pmf set and a batch of pickup pmf sets.

    pmf_d: (n, V) delivery pmfs of the route's customers; pmf_p_batch: (T, n, V)
    pickup pmfs, one set per batch element; returns the T tail probabilities.
    Exact, no truncation error. F is the peak load in level units, so the
    window is F > K.

    Recursion (backward over the route). With A = sum of the deliveries of the
    processed suffix and G = max over the suffix of the load minus the current
    load, the final peak is F = A + G, and one customer c updates
        A <- A + d_c,   G <- max(0, G + p_c - d_c).
    On the event F <= K every partial sum is at most K, so states with
    S = A + G > K are absorbed as overflow (S never decreases). The state is
    kept as (A, S) for the delivery step, which is then a Toeplitz product in
    A, and as (A, G~ = S - A) for the pickup step, which is row independent.
    Both steps are batched over the pickup sets with matrix products.
    """
    K = int(K)
    pmf_p_batch = np.asarray(pmf_p_batch, dtype=np.float64)
    T = pmf_p_batch.shape[0]
    if K < 0:
        return np.ones(T)
    V = pmf_p_batch.shape[2]
    K1 = K + 1
    idx = np.arange(K1)
    Af, Sf = np.meshgrid(idx, idx, indexing="ij")     # axes (A, S)
    Jf = Sf - Af + K                                  # G~ = S - A in [-K, K] -> [0, 2K]
    Gf = Sf - Af                                      # G after pickup, valid where >= 0
    valid = Gf >= 0
    jj = np.arange(2 * K + 1)
    P = np.zeros((T, K1, K1))                         # [pickup set, A, S]
    P[:, 0, 0] = 1.0
    for c in reversed([int(x) for x in route]):
        pd = np.asarray(pmf_d[c], dtype=np.float64)
        Td = np.zeros((K1, K1))                        # Td[A + v, A] = pd[v]
        for v in np.flatnonzero(pd):
            if v <= K:
                Td[np.arange(v, K1), np.arange(0, K1 - v)] += pd[v]
        Qs = np.matmul(Td, P)
        Gt = np.zeros((T, K1, 2 * K + 1))
        Gt[:, Af, Jf] = Qs[:, Af, Sf]
        Mp = np.zeros((T, 2 * K + 1, K1))              # G~ -> max(0, G~ + p) per pickup set
        pp = pmf_p_batch[:, c, :]
        for p in range(V):
            if not np.any(pp[:, p]):
                continue
            g = np.maximum(jj - K + p, 0)
            ok = g <= K
            Mp[:, jj[ok], g[ok]] += pp[:, p][:, None]
        R = np.matmul(Gt, Mp)                          # [pickup set, A, G]
        P = np.zeros((T, K1, K1))
        P[:, Af[valid], Sf[valid]] = R[:, Af[valid], Gf[valid]]
    return 1.0 - P.sum(axis=(1, 2))


def chance_copula(route: Sequence[int], cd: CopulaDemand) -> float:
    """P(F_r > Q_level) under the discrete copula model.

    Exact identity P = sum_t w_t sum_s w_s P(F > Q | z_t, z_s), with the inner
    sum over s computed by :func:`tail_over_pickups` (validated against
    ``classical.chance`` on the conditional independent model).
    """
    seq = [int(c) for c in route]
    total = 0.0
    for t in range(cd.z.size):
        tails = tail_over_pickups(seq, cd.pmf_d[t], cd.pmf_p, cd.Q_level)
        total += cd.w[t] * float(np.dot(cd.w, tails))
    return float(total)


def _sample_levels(pmf: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """K independent draws of every customer's level from its pmf row; (n, K) floats."""
    n = pmf.shape[0]
    cdf = np.cumsum(pmf, axis=1)
    cdf[:, -1] = 1.0
    U = rng.random((n, K))
    out = np.empty((n, K), dtype=np.float64)
    for c in range(n):
        out[c] = np.searchsorted(cdf[c], U[c], side="right")
    return out


def crn_factor_grid(pairs: Sequence[tuple[Sequence[int], Sequence[int]]], cd: CopulaDemand,
                    K: int, seed: int) -> dict:
    """Factor-grid estimates of P(D = +1), P(D = -1) for pairs of routes (common draws).

    For a pair (a, b) of routes, D = 1[F_b > Q] - 1[F_a > Q] with both routes
    evaluated on the SAME draws (common random numbers). Alternative-position
    moves have a different base route for each move, hence the list of pairs.
    The quantity is the sum over factor pairs (t, s) of w_t w_s times the
    conditional probability, evaluated in the discrete copula model of ``cd``
    (the model that :func:`chance_copula` evaluates exactly for one route).
    Each factor pair draws its own K scenarios, in which the customers are
    independent with pmfs pmf_d[t], pmf_p[s]: a conditional Monte Carlo over
    the pairs, not an exact computation. The standard error of each grid sum is
    sqrt(sum_ts (w_t w_s)^2 p_ts (1 - p_ts) / K) over the independent pairs.
    Per draw, P(D = +1) - P(D = -1) = P(F_b > Q) - P(F_a > Q) exactly, so the
    two estimates satisfy that identity without error.

    Returns arrays over the pairs: P_D1, P_Dm1 (with standard errors se_D1,
    se_Dm1), p_a and p_b (tails of the two routes), all as floats.
    """
    K = int(K)
    if K < 1:
        raise ValueError("K must be positive")
    pairs = [([int(c) for c in a], [int(c) for c in b]) for a, b in pairs]
    if not pairs:
        raise ValueError("pairs must be non-empty")
    routes = []
    index: dict[tuple[int, ...], int] = {}
    for a, b in pairs:
        for r in (a, b):
            key = tuple(r)
            if key not in index:
                index[key] = len(routes)
                routes.append(list(r))
    ia = np.array([index[tuple(a)] for a, _ in pairs])
    ib = np.array([index[tuple(b)] for _, b in pairs])
    M = len(pairs)
    Q = cd.Q_level
    T = cd.z.size
    P_D1 = np.zeros(M)
    P_Dm1 = np.zeros(M)
    var_D1 = np.zeros(M)
    var_Dm1 = np.zeros(M)
    p_a = np.zeros(M)
    p_b = np.zeros(M)
    for t in range(T):
        for s in range(T):
            rng = np.random.default_rng([int(seed), t, s])
            dsc = _sample_levels(cd.pmf_d[t], K, rng)
            psc = _sample_levels(cd.pmf_p[s], K, rng)
            w = float(cd.w[t] * cd.w[s])
            flags = [route_peak(r, dsc, psc) > Q for r in routes]
            for m in range(M):
                x = flags[ib[m]]
                y = flags[ia[m]]
                d1 = float(np.mean(x & ~y))
                dm = float(np.mean(~x & y))
                P_D1[m] += w * d1
                P_Dm1[m] += w * dm
                p_a[m] += w * float(np.mean(y))
                p_b[m] += w * float(np.mean(x))
                var_D1[m] += w * w * d1 * (1.0 - d1) / K
                var_Dm1[m] += w * w * dm * (1.0 - dm) / K
    return {"P_D1": P_D1, "P_Dm1": P_Dm1, "se_D1": np.sqrt(var_D1), "se_Dm1": np.sqrt(var_Dm1),
            "p_a": p_a, "p_b": p_b}


# ============================================================
# Scenario-level load profiles and insertion peaks
# ============================================================


def load_profile(route: Sequence[int], dsc: np.ndarray, psc: np.ndarray) -> np.ndarray:
    """Load profile L_0..L_L of a route on every scenario.

    dsc, psc have shape (n, N) (customer-major). Returns an (L + 1, N) array
    with L_0 = sum of deliveries and L_k = L_{k-1} - d_{i_k} + p_{i_k}
    (the peak is the column maximum). Computed in place, one stop at a time.
    """
    idx = [int(c) for c in route]
    L = len(idx)
    if L == 0:
        raise ValueError("route must be non-empty")
    N = dsc.shape[1]
    total = np.zeros(N, dtype=np.float64)
    for c in idx:
        total += dsc[c]
    prof = np.empty((L + 1, N), dtype=np.float64)
    prof[0] = total
    run = total.copy()
    for k, c in enumerate(idx):
        np.subtract(run, dsc[c], out=run)
        np.add(run, psc[c], out=run)
        prof[k + 1] = run
    return prof


def route_peak(route: Sequence[int], dsc: np.ndarray, psc: np.ndarray) -> np.ndarray:
    """Peak per scenario, max_k L_k, computed as load_profile without storing it."""
    idx = [int(c) for c in route]
    if not idx:
        raise ValueError("route must be non-empty")
    N = dsc.shape[1]
    total = np.zeros(N, dtype=np.float64)
    for c in idx:
        total += dsc[c]
    peak = total.copy()
    run = total.copy()
    for c in idx:
        np.subtract(run, dsc[c], out=run)
        np.add(run, psc[c], out=run)
        np.maximum(peak, run, out=peak)
    return peak


def insertion_envelopes(prof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prefix maxima PM[k] = max_{m<=k} L_m and suffix maxima SM[k] = max_{m>=k} L_m."""
    PM = np.maximum.accumulate(prof, axis=0)
    SM = np.maximum.accumulate(prof[::-1], axis=0)[::-1]
    return PM, SM


def insertion_peak(PM: np.ndarray, SM: np.ndarray, pos: int, dj: np.ndarray,
                   pj: np.ndarray) -> np.ndarray:
    """Peak per scenario after inserting a customer (d_j, p_j) at 1-indexed pos.

    The loads before the insertion gain d_j (the delivery is on board from the
    start) and the loads from the insertion on gain p_j, so
        F' = max( PM[pos-1] + d_j , SM[pos-1] + p_j ).
    """
    k = int(pos) - 1
    return np.maximum(PM[k] + dj, SM[k] + pj)


def intra_candidates(route: Sequence[int], cap: int, rng: np.random.Generator,
                     all_upto: int = 12) -> tuple[list[tuple[str, int, int, tuple[int, ...]]], int]:
    """Distinct intra-route neighbours of a route: relocate, swap and 2-opt.

    Returns (moves, n_total). Each move is (kind, i, k, new_route) with
    new_route != route and duplicates removed (first occurrence kept). All
    neighbours are returned when len(route) <= all_upto; otherwise a uniform
    random subset of size cap. n_total is the number before subsampling.
    """
    base = tuple(int(c) for c in route)
    L = len(base)
    seen = {base}
    out: list[tuple[str, int, int, tuple[int, ...]]] = []
    for i in range(L):
        for k in range(L):
            if k == i:
                continue
            s = list(base)
            c = s.pop(i)
            s.insert(k, c)
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                out.append(("relocate", i, k, t))
    for i in range(L):
        for k in range(i + 1, L):
            s = list(base)
            s[i], s[k] = s[k], s[i]
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                out.append(("swap", i, k, t))
    for i in range(L):
        for k in range(i + 1, L):
            s = list(base)
            s[i:k + 1] = s[i:k + 1][::-1]
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                out.append(("2opt", i, k, t))
    n_total = len(out)
    if L <= all_upto or n_total <= cap:
        return out, n_total
    idx = np.sort(rng.choice(n_total, size=cap, replace=False))
    return [out[int(i)] for i in idx], n_total


def insertion_candidates(pool: Sequence[int], route: Sequence[int], cap: int,
                         rng: np.random.Generator) -> list[tuple[int, int]]:
    """Insertion moves (j, pos) of customers in ``pool`` (not on the route).

    pos runs over 1..len(route)+1. More than ``cap`` candidates are reduced to
    a uniform random subset of size cap (without replacement), in candidate order.
    """
    on_route = set(int(c) for c in route)
    L = len(route)
    cands = [(int(j), pos) for j in sorted(set(int(c) for c in pool) - on_route)
             for pos in range(1, L + 2)]
    if len(cands) <= cap:
        return cands
    idx = np.sort(rng.choice(len(cands), size=cap, replace=False))
    return [cands[int(i)] for i in idx]


# ============================================================
# Window statistics and delta-method errors
# ============================================================


def window_stats(y: np.ndarray, x: np.ndarray) -> dict:
    """Statistics of D = 1[x] - 1[y] over common scenarios.

    y: flags of the base route (F > Q), x: flags of the new route. Returns
    p, p_new, P(D=1), P(D=-1), delta = E[D], var_D = Var(D), rho_q, rho_c and
    their delta-method standard errors. rho_q = sqrt(var_D)/sqrt(p'(1-p')),
    rho_c = var_D/(p'(1-p')). Ratios are NaN when p' is 0 or 1; the SE of
    rho_q is NaN when var_D = 0 (square-root kink).
    """
    y = np.asarray(y, dtype=bool)
    x = np.asarray(x, dtype=bool)
    if y.shape != x.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("y and x must be non-empty 1-D arrays of equal length")
    N = y.size
    D = x.astype(np.int8) - y.astype(np.int8)
    n_pos = int(np.count_nonzero(D > 0))
    n_neg = int(np.count_nonzero(D < 0))
    n_y = int(np.count_nonzero(y))
    n_x = int(np.count_nonzero(x))
    p = n_y / N
    pn = n_x / N
    P1 = n_pos / N
    Pm1 = n_neg / N
    delta = (n_pos - n_neg) / N
    M2 = (n_pos + n_neg) / N
    var = max(M2 - delta * delta, 0.0)
    out = {"p": p, "p_new": pn, "P_D1": P1, "P_Dm1": Pm1, "delta": delta, "var_D": var,
           "zero": n_pos + n_neg == 0,
           "rho_q": float("nan"), "se_rho_q": float("nan"),
           "rho_c": float("nan"), "se_rho_c": float("nan")}
    q = pn * (1.0 - pn)
    if q <= 0.0:
        return out
    out["rho_c"] = var / q
    out["rho_q"] = math.sqrt(var) / math.sqrt(q)
    Df = D.astype(np.float64)
    xf = x.astype(np.float64)
    phi_pn = xf - pn
    phi_var = (Df * Df - M2) - 2.0 * delta * (Df - delta)
    if var > 0.0:
        g_v = 1.0 / (2.0 * math.sqrt(var * q))
        g_p = -math.sqrt(var) * (1.0 - 2.0 * pn) / (2.0 * q ** 1.5)
        IF = g_v * phi_var + g_p * phi_pn
        out["se_rho_q"] = math.sqrt(float(np.mean(IF * IF)) / N)
    g_v = 1.0 / q
    g_p = -var * (1.0 - 2.0 * pn) / (q * q)
    IF = g_v * phi_var + g_p * phi_pn
    out["se_rho_c"] = math.sqrt(float(np.mean(IF * IF)) / N)
    return out


# ============================================================
# Query counts (design, Revision 2)
# ============================================================


def canonical_ae_M(var: float, eps: float) -> int:
    """Smallest power of two M with 2 pi sqrt(var) / M + pi^2 / M^2 <= eps.

    This is the error bound of canonical amplitude estimation with M Grover
    iterates (holds with probability at least 8/pi^2); its cost is 2M + 1
    applications of A.
    """
    if not eps > 0:
        raise ValueError(f"eps must be positive, got {eps}")
    s = math.sqrt(max(float(var), 0.0))
    M = 1
    while 2.0 * math.pi * s / M + math.pi ** 2 / (M * M) > eps:
        M *= 2
    return M


def quantum_queries(var: float, eps: float, reps: int = AE_REPS) -> float:
    """A-applications for an estimate at error eps: reps * (2M + 1)."""
    M = canonical_ae_M(var, eps)
    return float(reps * (2 * M + 1))


def classical_queries(var: float, eps: float, z: float = Z_95) -> float:
    """Monte Carlo samples for error eps at confidence z: z^2 var / eps^2."""
    if not eps > 0:
        raise ValueError(f"eps must be positive, got {eps}")
    return z * z * max(float(var), 0.0) / (eps * eps)


CERT_FLOOR = 3.0  # rule-of-three constant of the classical certificate floor (Revision 3)


def classical_queries_certified(var: float, eps: float, z: float = Z_95) -> float:
    """Classical samples certifying error eps: max(z^2 var / eps^2, 3 / eps).

    The first term is the CLT count. The floor is what a classical estimator
    needs before it can certify a probability below eps when no event has
    been seen (rule of three: about 3/eps samples); it binds when var is small
    (for example a move with Var(D) = 0 in the sample).
    """
    if not eps > 0:
        raise ValueError(f"eps must be positive, got {eps}")
    return max(z * z * max(float(var), 0.0) / (eps * eps), CERT_FLOOR / eps)


def classical_queries_certified_array(var, eps, z: float = Z_95) -> np.ndarray:
    """Vectorised :func:`classical_queries_certified` (broadcasts var and eps)."""
    var = np.maximum(np.asarray(var, dtype=np.float64), 0.0)
    eps = np.asarray(eps, dtype=np.float64)
    if np.any(~(eps > 0)):
        raise ValueError("eps must be positive")
    return np.maximum(z * z * var / (eps * eps), CERT_FLOOR / eps)


def signed_costs(P1: float, Pm1: float, eps: float) -> tuple[float, float]:
    """(quantum, classical) cost of a signed D: two estimates, each at eps/2."""
    v1 = P1 * (1.0 - P1)
    v2 = Pm1 * (1.0 - Pm1)
    q = quantum_queries(v1, eps / 2.0) + quantum_queries(v2, eps / 2.0)
    c = classical_queries(v1, eps / 2.0) + classical_queries(v2, eps / 2.0)
    return q, c


# ============================================================
# Discrete move statistics (independent model; used by M3b part b)
# ============================================================


def move_stats(
    route: Sequence[int],
    j: int,
    pos: int,
    dem: DiscreteDemand,
    Q_level: float,
    p_base: float | None = None,
) -> dict:
    """Window statistics of inserting customer j into ``route`` at ``pos``.

    p = P(F_r > Q), p_new = P(F_{r'} > Q) and delta = p_new - p, each from
    ``classical.chance`` on the independent discrete model. By the insertion
    lemma delta >= 0 (checked, and a violation raises ``ValueError``).

    Returns
    -------
    dict with keys p, p_new, delta, rho_q, rho_c (NaN when p_new is 0 or 1).
    """
    seq = [int(c) for c in route]
    j = int(j)
    if j in seq:
        raise ValueError(f"Customer {j} already in route")
    if not 1 <= pos <= len(seq) + 1:
        raise IndexError(f"Insertion position {pos} out of range [1, {len(seq) + 1}]")
    new_seq = seq[: pos - 1] + [j] + seq[pos - 1:]
    p = float(chance(seq, dem, Q_level)) if p_base is None else float(p_base)
    p_new = float(chance(new_seq, dem, Q_level))
    delta = p_new - p
    if delta < -_RHO_TOL:
        raise ValueError(
            f"Insertion lemma violated: route={seq}, j={j}, pos={pos}, "
            f"p={p!r}, p_new={p_new!r}, delta={delta!r}"
        )
    if p_new <= 0.0 or p_new >= 1.0:
        rho_q = float("nan")
        rho_c = float("nan")
    else:
        d = min(max(delta, 0.0), 1.0)
        dv = d * (1.0 - d)
        pv = p_new * (1.0 - p_new)
        rho_q = math.sqrt(dv) / math.sqrt(pv)
        rho_c = dv / pv
    return {"p": p, "p_new": p_new, "delta": delta, "rho_q": rho_q, "rho_c": rho_c}


# ============================================================
# M3b helpers
# ============================================================


def mlae_estimate(oracle: PeakOracle, schedule: int, seed: int = 7) -> dict:
    """Maximum-likelihood amplitude estimation with a given Grover schedule.

    ``schedule = m`` uses the Grover powers [0, 1, 2, 4, ..., 2**(m-1)] (qiskit's
    integer convention). The query accounting is the same as in
    ``quantum.oracle.estimate_probability``: shots * sum(2k + 1) applications
    of A over the rounds.
    """
    sampler = _make_sampler(seed)
    phase = QuantumCircuit(oracle.num_qubits)
    phase.z(oracle.objective)
    grover_q = grover_operator(phase, state_preparation=oracle.circuit)
    problem = EstimationProblem(
        state_preparation=oracle.circuit,
        objective_qubits=[oracle.objective],
        grover_operator=grover_q,
    )
    estimator = MaximumLikelihoodAmplitudeEstimation(evaluation_schedule=int(schedule), sampler=sampler)
    res = estimator.estimate(problem)
    rounds = [int(k) for k in res.evaluation_schedule]
    shots = int(sampler.default_shots)
    lo, hi = res.confidence_interval
    return {
        "estimate": float(res.estimation),
        "ci": (float(lo), float(hi)),
        "num_oracle_queries": int(shots * sum(2 * k + 1 for k in rounds)),
        "num_oracle_queries_library": int(res.num_oracle_queries),
        "rounds": rounds,
    }


def classical_samples_for_error(p: float, err: float, z: float = 1.96) -> float:
    """Monte Carlo samples for absolute error err at confidence z, using p.

    Returns NaN when err is 0 (no finite sample count reproduces it).
    """
    if err <= 0.0:
        return float("nan")
    return z * z * p * (1.0 - p) / (err * err)


# ============================================================
# M3a v3: interpretation layer (rule of three, end-load lemma, costs,
# paired level check, route bootstrap). Design: DESIGN_M3.md, status after
# review of revision 2.
# ============================================================

GATE_LEVEL = 0.10  # per-route failure target 1 - ALPHA of dethloff_runner (ALPHA = 0.90)
PPRIME_BANDS = ((0.0, 0.02, "<0.02"), (0.02, 0.2, "0.02-0.2"), (0.2, 0.5, "0.2-0.5"),
                (0.5, 1.0000001, ">0.5"))
GATE_BANDS = ((0.0, 0.02, "<0.02"), (0.02, 0.1, "0.02-0.1"), (0.1, 0.4, "0.1-0.4"),
              (0.4, 1.0000001, ">0.4"))
ALT_DIST_LABELS = ("1", "2-3", ">3")
BOOT_REPS = 1000


def band_label(x: float, bands: Sequence[tuple[float, float, str]]) -> str:
    """Label of the half-open band [lo, hi) that contains x ("NaN" if none)."""
    if x is None or not np.isfinite(x):
        return "NaN"
    for lo, hi, lab in bands:
        if lo <= x < hi:
            return lab
    return "NaN"


def gate_distance(pprime: float, gate: float = GATE_LEVEL) -> float:
    """Absolute distance |p' - gate| of a move's new window probability to the gate level."""
    return abs(float(pprime) - gate)


LENGTH_BANDS = ((0, 8, "<=8"), (9, 15, "9-15"), (16, 25, "16-25"), (26, 10**9, ">25"))


def length_label(L: int) -> str:
    """Route-length band of the paired level check: <= 8, 9-15, 16-25, > 25."""
    L = int(L)
    if L < 1:
        raise ValueError(f"route length must be >= 1, got {L}")
    for lo, hi, lab in LENGTH_BANDS:
        if lo <= L <= hi:
            return lab
    raise AssertionError("unreachable")


def alt_distance_label(d: int) -> str:
    """Position distance |pos2 - pos| for alternative-position moves: 1, 2-3, > 3."""
    d = int(d)
    if d < 1:
        raise ValueError(f"position distance must be >= 1, got {d}")
    if d == 1:
        return "1"
    return "2-3" if d <= 3 else ">3"


def rule_of_three_bound(N: int) -> float:
    """95 % upper bound on P(D != 0) when no scenario out of N shows D != 0."""
    if int(N) < 1:
        raise ValueError("N must be positive")
    return 3.0 / float(N)


def plug_in_variance(var: float, N: int) -> float:
    """Variance used for costs and ratios: max(var, 3 / N) (rule-of-three plug-in)."""
    return max(float(var), rule_of_three_bound(N))


def rho_q_upper(var: float, pn: float, N: int) -> float:
    """rho_q with the rule-of-three plug-in for Var(D); NaN when p' is 0 or 1."""
    q = float(pn) * (1.0 - float(pn))
    if q <= 0.0:
        return float("nan")
    return math.sqrt(plug_in_variance(var, N)) / math.sqrt(q)


# ---------------------------------------------------------------- per-class costs

def class_costs(cls: int, P1: float, Pm1: float, pn: float, eps: float, N: int) -> dict:
    """Query costs of one move at absolute error eps, both paradigms, with the
    rule-of-three plug-in (Var -> max(Var, 3/N) before any cost is computed).

    Stand-alone (per-move) estimate of p': quantum and classical at eps.
    Class 1 (D in {0, 1}): one amplitude P(D = 1); quantum CRN cost at eps,
    classical CRN cost with var = max(P1 (1 - P1), 3/N).
    Classes 2-3 (D in {-1, 0, 1}): quantum two-estimate cost, P(D = 1) and
    P(D = -1) each at eps/2 with variance max(P (1 - P), 3/N); classical DIRECT
    signed-mean cost with var = max(E D^2 - (E D)^2, 3/N).
    Every classical cost is the certified count max(z^2 var / eps^2, 3 / eps)
    (:func:`classical_queries_certified`); the quantum cost is the canonical
    AE count of :func:`quantum_queries`.
    """
    if cls not in (1, 2, 3):
        raise ValueError(f"cls must be 1, 2 or 3, got {cls}")
    if not eps > 0:
        raise ValueError(f"eps must be positive, got {eps}")
    P1 = float(P1)
    Pm1 = float(Pm1)
    if cls == 1 and Pm1 != 0.0:
        raise ValueError("class 1 moves have D in {0, 1}; P(D = -1) must be 0")
    delta = P1 - Pm1
    var_dir = max(P1 + Pm1 - delta * delta, 0.0)
    sa_var = float(pn) * (1.0 - float(pn))
    out = {
        "Nq_sa": quantum_queries(sa_var, eps),
        "Nc_sa": classical_queries_certified(sa_var, eps),
        "Nc": classical_queries_certified(plug_in_variance(var_dir, N), eps),
    }
    if cls == 1:
        out["Nq"] = quantum_queries(plug_in_variance(var_dir, N), eps)
    else:
        v1 = plug_in_variance(P1 * (1.0 - P1), N)
        v2 = plug_in_variance(Pm1 * (1.0 - Pm1), N)
        out["Nq"] = quantum_queries(v1, eps / 2.0) + quantum_queries(v2, eps / 2.0)
    return out


# ---------------------------------------------------------------- end-load lemma

def end_load_bound(end_ok: np.ndarray, base_over: np.ndarray,
                   move_over: Sequence[np.ndarray] = ()) -> dict:
    """Route-level end-load bounds on P(D != 0) for the intra moves of one route.

    end_ok: bool, sum of deliveries <= Q and sum of pickups <= Q (E).
    base_over: bool, F_r > Q. move_over: bool arrays F_{r'} > Q for the moves.

    B_spec = P(E and F_r > Q) (the quantity in the design note).
    B_union = P(E and (F_r > Q or any F_{r'} > Q)). Under E, D != 0 forces
    max(F_r, F_{r'}) > Q, so B_union bounds P(D != 0) of every listed move.
    """
    E = np.asarray(end_ok, dtype=bool)
    base = np.asarray(base_over, dtype=bool)
    if E.shape != base.shape or E.ndim != 1:
        raise ValueError("end_ok and base_over must be 1-D arrays of equal length")
    union = E & base
    for x in move_over:
        x = np.asarray(x, dtype=bool)
        if x.shape != base.shape:
            raise ValueError("move flags must have the same shape as base_over")
        union = union | (E & x)
    return {"B_spec": float(np.mean(E & base)), "B_union": float(np.mean(union))}


def end_load_violations(D: np.ndarray, end_ok: np.ndarray) -> int:
    """Scenarios with D != 0 where an end load exceeds Q. The lemma says 0."""
    D = np.asarray(D)
    E = np.asarray(end_ok, dtype=bool)
    if D.shape != E.shape:
        raise ValueError("D and end_ok must have the same shape")
    return int(np.count_nonzero((D != 0) & ~E))


def sum_loads(route: Sequence[int], dsc: np.ndarray) -> np.ndarray:
    """Sum of the given demands over a route, accumulated as in load_profile."""
    total = np.zeros(dsc.shape[1], dtype=np.float64)
    for c in route:
        total += dsc[int(c)]
    return total


def reconstruct_intra(route: Sequence[int], kind: str, i: int, k: int) -> list[int]:
    """Route after an intra move as produced by intra_candidates (kind, i, k)."""
    s = [int(c) for c in route]
    if kind == "relocate":
        c = s.pop(int(i))
        s.insert(int(k), c)
    elif kind == "swap":
        s[int(i)], s[int(k)] = s[int(k)], s[int(i)]
    elif kind == "2opt":
        s[int(i):int(k) + 1] = s[int(i):int(k) + 1][::-1]
    else:
        raise ValueError(f"unknown intra move kind {kind!r}")
    return s


# ---------------------------------------------------------------- paired level check

def quantise_levels(x: np.ndarray, unit: float, V: int) -> np.ndarray:
    """Nearest-level binning of continuous demands on the grid of level_rule.

    Level v carries [(v - 1/2) u, (v + 1/2) u); the top level V - 1 absorbs the
    upper tail. Returns float64 levels (integer valued).
    """
    if not unit > 0:
        raise ValueError(f"unit must be positive, got {unit}")
    lev = np.floor(np.asarray(x, dtype=np.float64) / unit + 0.5)
    return np.clip(lev, 0.0, float(int(V) - 1))


def paired_level_check(route: Sequence[int], dsc: np.ndarray, psc: np.ndarray,
                       Q: float, unit: float, V: int, K: int) -> dict:
    """Level-only discretisation error on the same continuous scenarios.

    p_cont = P(F > Q) on the continuous scenarios; p_lev = P(F_level > K) on
    the quantised scenarios (same draws). bias = p_lev - p_cont; se is the
    paired standard error of the scenario-wise difference of the two flags.
    """
    y = route_peak(route, dsc, psc) > Q
    x = route_peak(route, quantise_levels(dsc, unit, V),
                   quantise_levels(psc, unit, V)) > float(K)
    diff = x.astype(np.float64) - y.astype(np.float64)
    N = y.size
    se = float(np.std(diff, ddof=1)) / math.sqrt(N) if N > 1 else float("nan")
    p_cont = float(y.mean())
    p_lev = float(x.mean())
    return {"p_cont": p_cont, "p_lev": p_lev, "bias": p_lev - p_cont, "se": se, "N": N}


# ---------------------------------------------------------------- bootstrap

def weighted_quantile_sorted(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Lower weighted q-quantile of values already sorted ascending."""
    cw = np.cumsum(weights)
    total = cw[-1]
    if not total > 0:
        return float("nan")
    idx = int(np.searchsorted(cw, q * total, side="left"))
    return float(values[min(idx, values.size - 1)])


def route_bootstrap_quantile(values: np.ndarray, clusters: np.ndarray, q: float = 0.5,
                             reps: int = BOOT_REPS, seed: int = 0,
                             level: float = 0.95) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap interval of a quantile, resampling
    clusters (routes) with replacement.

    Each replicate draws R clusters from the R distinct clusters (multinomial
    counts) and takes the weighted lower q-quantile of the moves, each move
    weighted by the count of its route. Non-finite values are dropped.
    Returns (point, lo, hi) with the point estimate from the same definition.
    """
    v = np.asarray(values, dtype=np.float64)
    c = np.asarray(clusters)
    if v.shape != c.shape or v.ndim != 1:
        raise ValueError("values and clusters must be 1-D arrays of equal length")
    keep = np.isfinite(v)
    v, c = v[keep], c[keep]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    uniq, inv = np.unique(c, return_inverse=True)
    R = uniq.size
    order = np.argsort(v, kind="stable")
    vs = v[order]
    cs = inv[order]
    point = weighted_quantile_sorted(vs, np.ones(vs.size), q)
    rng = np.random.default_rng(seed)
    probs = np.full(R, 1.0 / R)
    draws = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        cnt = rng.multinomial(R, probs)
        draws[b] = weighted_quantile_sorted(vs, cnt[cs].astype(np.float64), q)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [a, 1.0 - a])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------- vectorised costs

def canonical_ae_M_array(var, eps) -> np.ndarray:
    """Vectorised :func:`canonical_ae_M` (broadcasts var and eps)."""
    var = np.maximum(np.asarray(var, dtype=np.float64), 0.0)
    eps = np.broadcast_to(np.asarray(eps, dtype=np.float64), var.shape)
    if np.any(~(eps > 0)):
        raise ValueError("eps must be positive")
    s = np.sqrt(var)
    M = np.full(var.shape, np.nan)
    done = np.zeros(var.shape, dtype=bool)
    Mk = 1.0
    for _ in range(80):
        ok = (~done) & (2.0 * math.pi * s / Mk + math.pi ** 2 / (Mk * Mk) <= eps)
        M[ok] = Mk
        done |= ok
        if done.all():
            break
        Mk *= 2.0
    if not done.all():
        raise RuntimeError("canonical_ae_M_array: no power of two reaches the target")
    return M


def quantum_queries_array(var, eps, reps: int = AE_REPS) -> np.ndarray:
    """Vectorised :func:`quantum_queries`."""
    return reps * (2.0 * canonical_ae_M_array(var, eps) + 1.0)


def classical_queries_array(var, eps, z: float = Z_95) -> np.ndarray:
    """Vectorised :func:`classical_queries`."""
    var = np.maximum(np.asarray(var, dtype=np.float64), 0.0)
    eps = np.asarray(eps, dtype=np.float64)
    if np.any(~(eps > 0)):
        raise ValueError("eps must be positive")
    return z * z * var / (eps * eps)


def class_costs_array(cls, P1, Pm1, pn, eps, N: int) -> dict:
    """Vectorised :func:`class_costs` over moves (cls, P1, Pm1, pn, eps arrays)."""
    cls = np.asarray(cls)
    P1 = np.asarray(P1, dtype=np.float64)
    Pm1 = np.asarray(Pm1, dtype=np.float64)
    pn = np.asarray(pn, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    floor = rule_of_three_bound(N)
    delta = P1 - Pm1
    var_dir = np.maximum(P1 + Pm1 - delta * delta, 0.0)
    v_dir = np.maximum(var_dir, floor)
    sa_var = pn * (1.0 - pn)
    v1 = np.maximum(P1 * (1.0 - P1), floor)
    v2 = np.maximum(Pm1 * (1.0 - Pm1), floor)
    Nq_single = quantum_queries_array(v_dir, eps)
    Nq_signed = quantum_queries_array(v1, eps / 2.0) + quantum_queries_array(v2, eps / 2.0)
    return {
        "Nq_sa": quantum_queries_array(sa_var, eps),
        "Nc_sa": classical_queries_certified_array(sa_var, eps),
        "Nq": np.where(cls == 1, Nq_single, Nq_signed),
        "Nc": classical_queries_certified_array(v_dir, eps),
    }

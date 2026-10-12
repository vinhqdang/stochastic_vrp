"""Experiment helpers for QUASAR M3 (design: papers/quasar/DESIGN_M3.md, Revision 2).

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
                  cv: float = DEFAULT_CV, rule: str = "design") -> CopulaDemand:
    """Copula discrete model of an instance at V = 2**b levels (see module doc)."""
    dbar, pbar, n, Q = _plan_arrays(plan_json)
    V = 2 ** int(b)
    q_max = q999_max(dbar, pbar, cv)
    Q_level, unit = level_rule(Q, V, q_max, rule)
    z, w = gh_nodes(n_nodes)
    pmf_d = np.stack([conditional_level_pmf(m, cv, unit, V, z, rho) for m in dbar], axis=1)
    pmf_p = np.stack([conditional_level_pmf(m, cv, unit, V, z, rho) for m in pbar], axis=1)
    return CopulaDemand(pmf_d=pmf_d, pmf_p=pmf_p, z=z, w=w, unit=float(unit),
                        Q_level=int(Q_level), rho=float(rho), b=int(b))


def chance_copula(route: Sequence[int], cd: CopulaDemand) -> float:
    """P(F_r > Q_level) under the discrete copula model.

    Exact identity P = sum_t w_t sum_s w_s P(F > Q | z_t, z_s), each term a
    ``classical.chance`` call on the conditional independent model.
    """
    seq = [int(c) for c in route]
    total = 0.0
    T = cd.z.size
    for t in range(T):
        for s in range(T):
            total += cd.w[t] * cd.w[s] * chance(seq, cd.demand(t, s), cd.Q_level)
    return float(total)


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

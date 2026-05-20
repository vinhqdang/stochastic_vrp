"""Phase 2 Hoeffding-Brown filter for ALNS-W-DRO.

Implements the cheap proxy + safety margin filter from Section 6 of the
manuscript (M14-M18). Per Brown (2007, OR Letters), the empirical CVaR over
n_0 iid sub-samples concentrates around the population CVaR exponentially fast
under bounded support, giving the per-iteration false-prune bound
delta(T_k) <= 6 * exp(-kappa/T_k) for adaptive n_0(k) = ceil(lambda/T_k).

Core objects:
    - FilterConfig: kappa (tolerance contract), lambda_ (sub-sample scaling)
    - cheap_proxy_*: O(n_0 log n_0) proxy evaluation via cache row sub-sampling
    - safety_margin: route-independent constant Gamma* (Definition M16)
    - filter_passes: rule -- True if candidate should be evaluated exactly
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.cache import RouteCache
from core.instance import Instance
from core.wdro_exact import empirical_cvar


# ============================================================
# Configuration
# ============================================================


@dataclass
class FilterConfig:
    """Phase 2 filter hyperparameters."""
    kappa: float = 2.0           # absolute tolerance contract (M18)
    lambda_: float = 100.0       # sub-sample scaling, n_0(k) = ceil(lambda/T_k)
    n0_min: int = 10             # minimum sub-sample size, safety floor
    n0_max: int | None = None    # cap on n_0 (default None = up to scenarios N)
    enabled: bool = True         # toggle for ablation comparisons


# ============================================================
# Safety margin Gamma* (Definition M16)
# ============================================================


def safety_margin(
    C_max: float,
    alpha: float,
    kappa: float,
    lambda_: float,
) -> float:
    """Gamma* = C_max * sqrt(kappa / ((1 - alpha) * lambda)).

    Route- and iteration-independent constant. Engineered so that with
    n_0(k) = lambda/T_k and Brown's bound, per-iteration false-prune
    probability decays as 6*exp(-kappa/T_k) (Proposition M-false-prune).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if kappa <= 0 or lambda_ <= 0:
        raise ValueError(f"kappa, lambda must be > 0; got {kappa}, {lambda_}")
    return C_max * np.sqrt(kappa / ((1.0 - alpha) * lambda_))


# ============================================================
# Adaptive sub-sample schedule
# ============================================================


def adaptive_n0(
    T_k: float,
    cfg: FilterConfig,
    N_total: int,
) -> int:
    """n_0(k) = clamp( ceil(lambda / T_k), n0_min, n0_max ).

    As temperature T_k decreases over iterations, n_0(k) grows; this is the
    'cooler = more samples' schedule that keeps Brown bound tight.
    """
    if T_k <= 0:
        return min(N_total, cfg.n0_max or N_total)
    n0 = int(np.ceil(cfg.lambda_ / T_k))
    n0 = max(n0, cfg.n0_min)
    if cfg.n0_max is not None:
        n0 = min(n0, cfg.n0_max)
    n0 = min(n0, N_total)
    return n0


# ============================================================
# Cheap proxy evaluators (sub-sampled cache rows)
# ============================================================


def cheap_proxy_phi_at_peaks(
    peak_loads_sub: np.ndarray,
    inst: Instance,
    alpha: float,
    epsilon: float,
) -> float:
    """Phi proxy given a sub-sampled vector of peak loads.

    Decoupled from cache structure to allow flexible callers.
    """
    violations = np.maximum(0.0, peak_loads_sub - inst.Q)
    cvar = empirical_cvar(violations, alpha)
    return cvar + epsilon / (1.0 - alpha)


def cheap_proxy_insertion_phi(
    cache: RouteCache,
    j: int,
    position: int,
    inst: Instance,
    alpha: float,
    epsilon: float,
    sub_indices: np.ndarray,
) -> float:
    """O(n_0 log n_0) proxy for Phi(r') under insertion of customer j at
    position k = `position`.

    Uses the M11 closed-form
        f_{r'}(xi^(s)) = max(Omega_r[s, k-1] + d_j^(s), Psi_r[s, k-1] + p_j^(s))
    evaluated only at the n_0 = |sub_indices| sub-sampled scenarios.

    Parameters
    ----------
    cache : RouteCache (host route r)
    j : int
        Customer to insert (must not be in cache.route).
    position : int (1-indexed, in {1, ..., m+1})
    sub_indices : np.ndarray, shape (n_0,)
        Indices into [0, N). Typically rng.integers(0, N, size=n_0).
    """
    m = cache.m
    if not (1 <= position <= m + 1):
        raise ValueError(f"position must be in [1, {m+1}], got {position}")

    k = position
    # Cache rows at sub-sampled scenarios
    omega_sub = cache.Omega[sub_indices, k - 1]
    psi_sub   = cache.Psi[sub_indices,   k - 1]
    d_j_sub   = cache.scenarios[sub_indices, 2 * j]
    p_j_sub   = cache.scenarios[sub_indices, 2 * j + 1]

    new_peaks = np.maximum(omega_sub + d_j_sub, psi_sub + p_j_sub)
    return cheap_proxy_phi_at_peaks(new_peaks, inst, alpha, epsilon)


def cheap_proxy_removal_phi(
    cache: RouteCache,
    position: int,
    inst: Instance,
    alpha: float,
    epsilon: float,
    sub_indices: np.ndarray,
) -> float:
    """O(n_0 log n_0) proxy for Phi(r^-) under removal at position k.

    Uses M9 symmetric closed-form
        f_{r^-}(xi^(s)) = max(Omega_r[s, k-1] - d_j^(s), Psi_r[s, k+1] - p_j^(s))
    at sub-sampled scenarios.
    """
    m = cache.m
    if m == 0:
        raise ValueError("Cannot remove from empty route")
    if not (1 <= position <= m):
        raise ValueError(f"position must be in [1, {m}], got {position}")
    if m == 1:
        return epsilon / (1.0 - alpha)  # r^- = empty, Phi = +eps/(1-alpha)? actually 0 for empty

    k = position
    j = cache.route.customers[k - 1]
    d_j_sub = cache.scenarios[sub_indices, 2 * j]
    p_j_sub = cache.scenarios[sub_indices, 2 * j + 1]

    prefix_peak = cache.Omega[sub_indices, k - 1] - d_j_sub
    if k < m:
        suffix_peak = cache.Psi[sub_indices, k + 1] - p_j_sub
        new_peaks = np.maximum(prefix_peak, suffix_peak)
    else:
        new_peaks = prefix_peak
    return cheap_proxy_phi_at_peaks(new_peaks, inst, alpha, epsilon)


# ============================================================
# Filter rule (Definition def:filter)
# ============================================================


def filter_passes(
    proxy_total: float,
    current_best_total: float,
    gamma_star: float,
) -> bool:
    """Filter retention rule.

    Returns True if the candidate should be promoted to exact evaluation,
    False if it can be safely pruned.

    Pruned candidate has proxy_total > current_best_total + gamma_star.
    """
    return proxy_total <= current_best_total + gamma_star


# ============================================================
# Diagnostic: false-prune tracking
# ============================================================


@dataclass
class FilterDiagnostics:
    """Counters logged across an ALNS run to verify the kappa contract."""
    n_proxy_evals: int = 0       # cheap proxy evals
    n_exact_evals: int = 0       # promoted to exact
    n_pruned: int = 0            # filtered out
    false_prunes: int = 0        # cases where pruned but would have improved
    deviations: list[tuple[float, float]] = None  # (proxy, exact) pairs sampled

    def __post_init__(self):
        if self.deviations is None:
            self.deviations = []

    @property
    def prune_rate(self) -> float:
        total = self.n_pruned + self.n_exact_evals
        return self.n_pruned / total if total > 0 else 0.0

    @property
    def false_prune_rate(self) -> float:
        return (
            self.false_prunes / self.n_pruned if self.n_pruned > 0 else 0.0
        )
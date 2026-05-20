"""Fast W-DRO evaluation under route perturbation, using Phase 1 cache.

Implements Theorem M11: per-candidate insertion evaluation in O(N log N),
independent of route length m.

Key formula (Lemma M5 + M11):
    f_{r'}(xi^(s)) = max(
        Omega_r[s, k-1] + d_j(xi^(s)),     # prefix max + delivery
        Psi_r[s, k-1]   + p_j(xi^(s))      # pivot/suffix max + pickup
    )
where (j, k) is the candidate insertion (customer j at position k).
"""

from __future__ import annotations

import numpy as np

from core.cache import RouteCache
from core.instance import Instance
from core.wdro_exact import empirical_cvar


def evaluate_insertion_peak_loads_via_cache(
    cache: RouteCache,
    j: int,
    position: int,
) -> np.ndarray:
    """Compute f_{r'}(xi^(s)) for all scenarios s, after inserting customer j
    at position k = `position`.

    Parameters
    ----------
    cache : RouteCache
        Cache for the host route r.
    j : int
        Customer to insert. Must not be in cache.route.
    position : int
        Insertion position k, 1-indexed. Valid range: 1 .. m+1.

    Returns
    -------
    peak_loads : np.ndarray, shape (N,)
        f_{r'}(xi^(s)) for each scenario.
    """
    m = cache.m
    if not (1 <= position <= m + 1):
        raise ValueError(
            f"position must be in [1, {m+1}], got {position}"
        )
    if j in cache.route.customers:
        raise ValueError(f"customer {j} already in route")

    k = position
    # d_j, p_j across all scenarios
    d_j = cache.scenarios[:, 2 * j]      # shape (N,)
    p_j = cache.scenarios[:, 2 * j + 1]  # shape (N,)

    # Prefix peak through stage k-1 of r (stages of r are 0..m, so k-1 valid for k in 1..m+1)
    omega_kminus1 = cache.Omega[:, k - 1]  # shape (N,)
    psi_kminus1   = cache.Psi[:,   k - 1]  # shape (N,)

    # New peak: max of prefix-shift and pivot/suffix-shift
    new_peaks = np.maximum(omega_kminus1 + d_j, psi_kminus1 + p_j)
    return new_peaks


def evaluate_phi_insertion_via_cache(
    cache: RouteCache,
    j: int,
    position: int,
    inst: Instance,
    alpha: float,
    epsilon: float,
) -> float:
    """Full Phi(r') evaluation under insertion via cache.

    Phi(r') = CVaR_alpha^{F_0}( max(0, f_{r'}(xi) - Q) ) + epsilon / (1 - alpha)

    Cost: O(N log N) — independent of m. (M11)
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")

    new_peaks = evaluate_insertion_peak_loads_via_cache(cache, j, position)
    violations = np.maximum(0.0, new_peaks - inst.Q)
    cvar = empirical_cvar(violations, alpha)
    return cvar + epsilon / (1.0 - alpha)


def best_insertion_via_cache(
    cache: RouteCache,
    j: int,
    inst: Instance,
    alpha: float,
    epsilon: float,
) -> tuple[int, float]:
    """Find the best position to insert customer j into the cached route.

    Evaluates all m+1 positions in O((m+1) * N log N).

    Returns
    -------
    (best_position, best_phi)
        1-indexed position; best Phi(r').
    """
    m = cache.m
    best_pos = 1
    best_phi = np.inf
    for pos in range(1, m + 2):
        phi = evaluate_phi_insertion_via_cache(
            cache, j, pos, inst, alpha, epsilon
        )
        if phi < best_phi:
            best_phi = phi
            best_pos = pos
    return best_pos, best_phi


# ============================================================
# Removal evaluation (M9 symmetric to M11)
# ============================================================


def evaluate_removal_peak_loads_via_cache(
    cache: RouteCache,
    position: int,
) -> np.ndarray:
    """Compute f_{r^-}(xi^(s)) for all scenarios after removing customer at
    position k from cache.route.

    By the inverse of Lemma M5:
        L_{r^-, k'} = L_{r, k'}   - d_j     for k' in {0, ..., k-1}
        L_{r^-, k'} = L_{r, k'+1} - p_j     for k' in {k, ..., m-1}

    Peak load:
        f_{r^-} = max( Omega_r[k-1] - d_j,  Psi_r[k+1] - p_j )

    Parameters
    ----------
    cache : RouteCache
        Cache for the host route r.
    position : int
        Position of customer to remove (1-indexed). Valid: 1 .. m.

    Returns
    -------
    peak_loads : np.ndarray, shape (N,)
    """
    m = cache.m
    if m == 0:
        raise ValueError("Cannot remove from empty route")
    if not (1 <= position <= m):
        raise ValueError(
            f"position must be in [1, {m}], got {position}"
        )

    N = cache.N
    if m == 1:
        # Removing the only customer -> empty route -> f === 0
        return np.zeros(N)

    k = position
    j = cache.route.customers[k - 1]  # 1-indexed -> 0-indexed Python list

    d_j = cache.scenarios[:, 2 * j]      # shape (N,)
    p_j = cache.scenarios[:, 2 * j + 1]  # shape (N,)

    # Prefix part: stages 0..k-1 of r^- correspond to 0..k-1 of r (minus d_j)
    prefix_peak = cache.Omega[:, k - 1] - d_j

    if k < m:
        # Suffix part: stages k..m-1 of r^- correspond to k+1..m of r (minus p_j)
        suffix_peak = cache.Psi[:, k + 1] - p_j
        return np.maximum(prefix_peak, suffix_peak)
    else:
        # k == m: no suffix, only prefix
        return prefix_peak


def evaluate_phi_removal_via_cache(
    cache: RouteCache,
    position: int,
    inst: Instance,
    alpha: float,
    epsilon: float,
) -> float:
    """Phi(r^-) after removal at position k, via cache. O(N log N)."""
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")

    new_peaks = evaluate_removal_peak_loads_via_cache(cache, position)
    violations = np.maximum(0.0, new_peaks - inst.Q)
    cvar = empirical_cvar(violations, alpha)
    return cvar + epsilon / (1.0 - alpha)


def best_removal_via_cache(
    cache: RouteCache,
    inst: Instance,
    alpha: float,
    epsilon: float,
) -> tuple[int, float]:
    """Find the position whose removal gives the smallest Phi(r^-).

    Returns (best_position_1indexed, best_phi). Useful for destroy operators.
    """
    m = cache.m
    if m == 0:
        raise ValueError("Cannot remove from empty route")

    best_pos = 1
    best_phi = np.inf
    for k in range(1, m + 1):
        phi = evaluate_phi_removal_via_cache(cache, k, inst, alpha, epsilon)
        if phi < best_phi:
            best_phi = phi
            best_pos = k
    return best_pos, best_phi
"""W-DRO exact evaluator (slow baseline, no caching).

Implements the projection-equivalent W-DRO objective for a single route r:

    Phi(r) = CVaR_alpha^{F_0}( max(0, f_r(xi) - Q) ) + epsilon / (1 - alpha)

where:
    - f_r(xi) = max-k beta_{r,k}^T xi  is the peak load function (Theorem M2),
    - F_0     = empirical distribution over N scenarios,
    - alpha   = CVaR confidence level (e.g., 0.95),
    - epsilon = Wasserstein ambiguity radius (model hyperparameter, fixed),
    - Q       = vehicle capacity.

This evaluator is O(N * m) per route call (full batch over scenarios). It is the
SLOW baseline used as ground truth in Days 4-5 (cache + fast insertion) and as
the production fallback when cache invalidates.

The closed-form +epsilon/(1-alpha) term is enabled by Huang's projection
equivalence theorem combined with our Universal Lipschitz Invariance
(Theorem M3): every route in R_n satisfies ||beta_{r,k}||_inf = 1, so the same
Wasserstein regularization applies uniformly across the route family.
"""

from __future__ import annotations

import numpy as np

from core.instance import Instance
from core.route import Route


# ============================================================
# Empirical CVaR (Rockafellar-Uryasev sample form)
# ============================================================


def empirical_cvar(losses: np.ndarray, alpha: float) -> float:
    """Compute empirical CVaR_alpha of a 1D sample of losses.

    Sample form: CVaR_alpha(L) = mean of the top ceil((1-alpha) * N) losses.

    Parameters
    ----------
    losses : np.ndarray, shape (N,)
        Loss realizations under F_0.
    alpha : float in [0, 1)
        Risk level. alpha=0 -> sample mean. alpha->1 -> sample max.

    Returns
    -------
    float
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    N = losses.shape[0]
    if N == 0:
        return 0.0

    k = max(1, int(np.ceil((1.0 - alpha) * N)))
    # Partial sort: only need top-k, not full sort. np.partition is O(N).
    if k < N:
        threshold_idx = N - k
        partitioned = np.partition(losses, threshold_idx)
        top_k = partitioned[threshold_idx:]
    else:
        top_k = losses
    return float(top_k.mean())


# ============================================================
# Per-route W-DRO evaluator
# ============================================================


def evaluate_phi_exact(
    route: Route,
    inst: Instance,
    scenarios: np.ndarray,
    alpha: float,
    epsilon: float,
) -> float:
    """Exact W-DRO penalty for a single route.

    Phi(r) = CVaR_alpha^{F_0}( max(0, f_r(xi) - Q) ) + epsilon / (1 - alpha)

    Parameters
    ----------
    route : Route
    inst : Instance
    scenarios : np.ndarray, shape (N, 2n)
        Empirical samples of xi under F_0.
    alpha : float in [0, 1)
    epsilon : float, >= 0
        Wasserstein radius. epsilon=0 collapses to pure empirical CVaR.

    Returns
    -------
    float, >= 0
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    if len(route) == 0:
        return 0.0  # empty route: f_r === 0, Lip = 0, no penalty.

    peak_loads = route.peak_loads_batch(scenarios, inst.n)        # shape (N,)
    violations = np.maximum(0.0, peak_loads - inst.Q)     # h(f_r)
    cvar = empirical_cvar(violations, alpha)
    return cvar + epsilon / (1.0 - alpha)


# ============================================================
# Solution-level aggregation
# ============================================================


def evaluate_phi_total(
    routes: list[Route],
    inst: Instance,
    scenarios: np.ndarray,
    alpha: float,
    epsilon: float,
) -> float:
    """Total W-DRO penalty over a multi-route solution.

    Per the SVRPSPD formulation, routes share the depot but operate independently
    (each route = one vehicle's tour). The total ambiguity penalty is the sum
    of per-route penalties.
    """
    return sum(
        evaluate_phi_exact(r, inst, scenarios, alpha, epsilon)
        for r in routes
    )


def evaluate_objective(
    routes: list[Route],
    inst: Instance,
    scenarios: np.ndarray,
    alpha: float,
    epsilon: float,
    penalty_lambda: float = 1.0,
) -> dict:
    """Full ALNS objective: travel cost + lambda * total W-DRO penalty.

    Returns a dict with breakdown for diagnostics.
    """
    D = inst.distances()
    travel = sum(r.travel_cost(D) for r in routes)
    penalty = evaluate_phi_total(routes, inst, scenarios, alpha, epsilon)
    return {
        "travel": travel,
        "wdro_penalty": penalty,
        "objective": travel + penalty_lambda * penalty,
    }
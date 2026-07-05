"""Phase 0 seeding for SVRPSPD with W-DRO awareness.

Implements Clarke-Wright savings with effective capacity Q_eff = Q - eps/(1-alpha)
(Definition M13.1) to guarantee the a priori bound of Proposition M13.
"""

from __future__ import annotations

import numpy as np

from core.instance import Instance
from core.route import Route


def clarke_wright_svrpspd(
    inst: Instance,
    capacity_buffer: float = 0.0,
    verbose: bool = False,
) -> list[Route]:
    """Clarke-Wright savings adapted for SVRPSPD.

    Capacity feasibility uses PEAK LOAD over nominal demand, not sum.
    
    Parameters
    ----------
    inst : Instance
    capacity_buffer : float
        Reduce effective capacity by this amount. Pass eps/(1-alpha) for
        Phase 0 W-DRO seeding.
    """
    n = inst.n
    Q_eff = inst.Q - capacity_buffer
    D = inst.distances()
    xi_nom = inst.nominal_xi()

    savings: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = D[0, i + 1] + D[0, j + 1] - D[i + 1, j + 1]
            savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])

    routes: list[list[int]] = [[c] for c in range(n)]
    route_of = list(range(n))

    def feasible(route_customers: list[int]) -> bool:
        if not route_customers:
            return True
        r = Route(route_customers)
        return r.peak_load(xi_nom, n) <= Q_eff + 1e-6

    n_merges = 0
    for s, i, j in savings:
        if s <= 0:
            break
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        route_i = routes[ri]
        route_j = routes[rj]
        merged = None
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        elif route_i[-1] == i and route_j[-1] == j:
            merged = route_i + route_j[::-1]
        elif route_i[0] == i and route_j[0] == j:
            merged = route_i[::-1] + route_j
        elif route_i[0] == i and route_j[-1] == j:
            merged = route_j + route_i
        if merged is None:
            continue
        if not feasible(merged):
            continue
        routes[ri] = merged
        routes[rj] = []
        for c in merged:
            route_of[c] = ri
        n_merges += 1

    final_routes = [Route(r) for r in routes if r]

    if verbose:
        print(f"[CW] {n_merges} merges, {len(final_routes)} routes formed")
        for k, r in enumerate(final_routes):
            peak = r.peak_load(xi_nom, n)
            print(f"     Route {k}: {len(r)} customers, peak {peak:.0f} / Q_eff {Q_eff:.0f}")

    return final_routes


def phase0_wdro_seeding(
    inst: Instance,
    alpha: float,
    epsilon: float,
    verbose: bool = False,
) -> list[Route]:
    """Phase 0 W-DRO-aware seeding (M13).

    Reserves a buffer of eps/(1-alpha) below the vehicle capacity to absorb
    empirical demand variability. The resulting routes satisfy
        f_{r_0}(xi_bar) <= Q - eps/(1-alpha)
    by construction, enabling the a priori bound of Proposition M13.

    Parameters
    ----------
    inst : Instance
    alpha : float in (0, 1)
    epsilon : float >= 0
        Wasserstein ambiguity radius.

    Returns
    -------
    list[Route]
        Initial feasible routes under nominal demand with Q_eff buffer.

    Raises
    ------
    ValueError
        If Q_eff = Q - eps/(1-alpha) <= 0.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")

    buffer = epsilon / (1.0 - alpha)
    Q_eff = inst.Q - buffer
    if Q_eff <= 0:
        raise ValueError(
            f"Q_eff = Q - eps/(1-alpha) = {Q_eff:.4f} must be > 0. "
            f"Either decrease epsilon (current {epsilon}) or alpha (current {alpha})."
        )

    return clarke_wright_svrpspd(inst, capacity_buffer=buffer, verbose=verbose)
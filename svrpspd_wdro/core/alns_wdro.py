"""W-DRO-aware ALNS-SA for SVRPSPD.

Objective:
    F(solution) = sum_r travel_cost(r) + lambda * sum_r Phi(r)

where Phi(r) is the W-DRO penalty from Proposition M4. Per-iteration
candidate evaluation uses the Phase 1 cache (M10) for O(N log N) cost
per insertion / O(N) per removal (Theorems M11 + M9).

Maintains nominal feasibility (peak load <= Q under xi_bar) as a hard
constraint during repair; W-DRO penalty captures stochastic robustness on top.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from core.instance import Instance
from core.route import Route
from core.cache import RouteCache
from core.wdro_exact import evaluate_phi_exact, empirical_cvar
from core.wdro_fast import (
    evaluate_phi_insertion_via_cache,
    evaluate_phi_removal_via_cache,
)


# ============================================================
# Configuration
# ============================================================


@dataclass
class WDROConfig:
    """Hyperparameters for W-DRO ALNS-SA."""
    alpha: float = 0.9
    epsilon: float = 0.5
    penalty_lambda: float = 1.0
    max_iters: int = 5000
    T_init_frac: float = 0.05
    alpha_cooling: float = 0.9997
    destroy_frac_min: float = 0.10
    destroy_frac_max: float = 0.30
    seed: int = 42
    verbose_every: int = 500
    time_limit_sec: float | None = None
    max_vehicles: int | None = None


@dataclass
class WDROALNSResult:
    best_solution: list[Route]
    best_objective: float
    best_breakdown: dict
    cost_history: list[float] = field(default_factory=list)
    n_iters: int = 0
    elapsed_sec: float = 0.0
    n_accepted: int = 0
    n_improved: int = 0


# ============================================================
# Objective evaluation
# ============================================================


def _phi_from_cache(
    cache: RouteCache,
    inst: Instance,
    alpha: float,
    epsilon: float,
) -> float:
    """Compute Phi(r) directly from cache.peak_loads (avoids rebuild)."""
    if cache.m == 0:
        return 0.0
    peak_loads = cache.peak_loads
    violations = np.maximum(0.0, peak_loads - inst.Q)
    return empirical_cvar(violations, alpha) + epsilon / (1.0 - alpha)


def solution_breakdown(
    routes: list[Route],
    caches: list[RouteCache],
    inst: Instance,
    alpha: float,
    epsilon: float,
    penalty_lambda: float,
) -> dict:
    """Decompose objective into travel + W-DRO penalty + total."""
    D = inst.distances()
    travel = sum(r.travel_cost(D) for r in routes)
    phi_total = sum(_phi_from_cache(c, inst, alpha, epsilon) for c in caches)
    return {
        "travel": travel,
        "wdro_penalty": phi_total,
        "objective": travel + penalty_lambda * phi_total,
    }


# ============================================================
# Destroy operators
# ============================================================


def destroy_random(
    routes: list[Route],
    n_remove: int,
    rng: np.random.Generator,
) -> tuple[list[Route], list[int]]:
    """Uniformly remove n_remove customers."""
    all_cust = [(ri, c) for ri, r in enumerate(routes) for c in r]
    if not all_cust:
        return [r.copy() for r in routes], []
    n_remove = min(n_remove, len(all_cust))
    indices = rng.choice(len(all_cust), size=n_remove, replace=False)
    to_remove = {all_cust[i] for i in indices}
    new_routes = [
        Route([c for c in r if (ri, c) not in to_remove])
        for ri, r in enumerate(routes)
    ]
    new_routes = [r for r in new_routes if len(r) > 0]
    removed = [c for (_, c) in to_remove]
    return new_routes, removed


def destroy_worst_wdro(
    routes: list[Route],
    caches: list[RouteCache],
    inst: Instance,
    alpha: float,
    epsilon: float,
    n_remove: int,
    rng: np.random.Generator,
    noise: float = 0.3,
) -> tuple[list[Route], list[int]]:
    """Remove customers whose removal reduces Phi most (with noise).

    Uses cache-based O(N) removal evaluation per (route, position).
    """
    contributions = []  # (-reduction, route_idx, customer)
    for ri, (r, cache) in enumerate(zip(routes, caches)):
        if len(r) == 0:
            continue
        phi_now = _phi_from_cache(cache, inst, alpha, epsilon)
        for k in range(1, len(r) + 1):
            phi_after = evaluate_phi_removal_via_cache(
                cache, k, inst, alpha, epsilon
            )
            reduction = phi_now - phi_after
            reduction *= 1.0 + (rng.random() - 0.5) * 2 * noise
            contributions.append((-reduction, ri, r.customers[k - 1]))

    contributions.sort()
    to_remove = {(ri, c) for _, ri, c in contributions[:n_remove]}
    new_routes = [
        Route([c for c in r if (ri, c) not in to_remove])
        for ri, r in enumerate(routes)
    ]
    new_routes = [r for r in new_routes if len(r) > 0]
    removed = [c for (_, c) in to_remove]
    return new_routes, removed


# ============================================================
# Repair operator
# ============================================================


def repair_greedy_wdro(
    routes: list[Route],
    to_insert: list[int],
    inst: Instance,
    scenarios: np.ndarray,
    alpha: float,
    epsilon: float,
    penalty_lambda: float,
    rng: np.random.Generator,
    max_vehicles: int | None = None,
) -> tuple[list[Route], list[RouteCache]] | None:
    """Greedy insertion: each pending customer placed at minimum total-delta position.

    Total delta = travel_delta + lambda * Phi_delta.
    Maintains nominal feasibility (peak <= Q under xi_bar).

    Returns (routes, caches) on success, None if some customer cannot be placed.
    """
    D = inst.distances()
    xi_nom = inst.nominal_xi()
    Q = inst.Q

    sol = [r.copy() for r in routes]
    caches = [RouteCache(r, scenarios, inst.n) for r in sol]
    phi_per_route = [_phi_from_cache(c, inst, alpha, epsilon) for c in caches]

    pending = list(to_insert)
    rng.shuffle(pending)

    for c in pending:
        best = None  # (total_delta, ri, pos, phi_after)

        for ri, (r, cache) in enumerate(zip(sol, caches)):
            for pos in range(1, len(r) + 2):
                # Travel delta
                prev = 0 if pos == 1 else r.customers[pos - 2] + 1
                nxt = 0 if pos == len(r) + 1 else r.customers[pos - 1] + 1
                travel_delta = D[prev, c + 1] + D[c + 1, nxt] - D[prev, nxt]

                # Nominal feasibility check
                trial = r.copy()
                trial.insert(c, pos=pos)
                peak_nom = trial.peak_load(xi_nom, inst.n)
                if peak_nom > Q + 1e-6:
                    continue

                # Phi after insertion (via cache, O(N log N))
                phi_after = evaluate_phi_insertion_via_cache(
                    cache, c, pos, inst, alpha, epsilon
                )
                phi_delta = phi_after - phi_per_route[ri]

                total_delta = travel_delta + penalty_lambda * phi_delta

                if best is None or total_delta < best[0]:
                    best = (total_delta, ri, pos, phi_after)

        if best is not None:
            _, ri, pos, phi_new = best
            sol[ri].insert(c, pos=pos)
            caches[ri] = RouteCache(sol[ri], scenarios, inst.n)
            phi_per_route[ri] = phi_new
        else:
            # Open new route if allowed
            if max_vehicles is None or len(sol) < max_vehicles:
                new_r = Route([c])
                if new_r.peak_load(xi_nom, inst.n) > Q + 1e-6:
                    return None
                sol.append(new_r)
                new_cache = RouteCache(new_r, scenarios, inst.n)
                caches.append(new_cache)
                phi_per_route.append(_phi_from_cache(new_cache, inst, alpha, epsilon))
            else:
                return None

    return sol, caches


# ============================================================
# Main ALNS-SA loop
# ============================================================


def alns_sa_wdro(
    initial_solution: list[Route],
    inst: Instance,
    scenarios: np.ndarray,
    config: WDROConfig,
) -> WDROALNSResult:
    """W-DRO-aware ALNS-SA main loop."""
    rng = np.random.default_rng(config.seed)

    current = [r.copy() for r in initial_solution]
    current_caches = [RouteCache(r, scenarios, inst.n) for r in current]
    breakdown = solution_breakdown(
        current, current_caches, inst,
        config.alpha, config.epsilon, config.penalty_lambda,
    )
    current_obj = breakdown["objective"]

    best = [r.copy() for r in current]
    best_obj = current_obj
    best_breakdown = breakdown.copy()

    T = max(config.T_init_frac * current_obj, 1e-6)
    history = [current_obj]
    n_accepted = 0
    n_improved = 0
    n_customers_total = sum(len(r) for r in current)

    t0 = time.time()
    last_iter = 0

    for k in range(1, config.max_iters + 1):
        last_iter = k
        if config.time_limit_sec is not None:
            if time.time() - t0 > config.time_limit_sec:
                break

        frac = rng.uniform(config.destroy_frac_min, config.destroy_frac_max)
        n_remove = max(2, int(frac * n_customers_total))

        # Destroy
        if rng.random() < 0.5:
            partial, removed = destroy_random(current, n_remove, rng)
        else:
            partial, removed = destroy_worst_wdro(
                current, current_caches, inst,
                config.alpha, config.epsilon, n_remove, rng,
            )

        # Repair
        repair_result = repair_greedy_wdro(
            partial, removed, inst, scenarios,
            config.alpha, config.epsilon, config.penalty_lambda, rng,
            max_vehicles=config.max_vehicles,
        )
        if repair_result is None:
            continue
        candidate, cand_caches = repair_result

        # Verify all customers served
        served = set()
        for r in candidate:
            served.update(r.customers)
        if served != set(range(inst.n)):
            continue

        cand_bd = solution_breakdown(
            candidate, cand_caches, inst,
            config.alpha, config.epsilon, config.penalty_lambda,
        )
        cand_obj = cand_bd["objective"]
        delta = cand_obj - current_obj

        accept = False
        if delta < 0:
            accept = True
            if cand_obj < best_obj:
                best = [r.copy() for r in candidate]
                best_obj = cand_obj
                best_breakdown = cand_bd.copy()
                n_improved += 1
        else:
            if T > 1e-12 and rng.random() < np.exp(-delta / T):
                accept = True

        if accept:
            current = candidate
            current_caches = cand_caches
            current_obj = cand_obj
            n_accepted += 1

        history.append(current_obj)
        T *= config.alpha_cooling

        if config.verbose_every and k % config.verbose_every == 0:
            elapsed = time.time() - t0
            print(
                f"[iter {k:5d}] T={T:.2f} cur={current_obj:.0f} "
                f"best={best_obj:.0f} "
                f"travel={best_breakdown['travel']:.0f} "
                f"wdro={best_breakdown['wdro_penalty']:.2f} "
                f"t={elapsed:.1f}s"
            )

    return WDROALNSResult(
        best_solution=best,
        best_objective=best_obj,
        best_breakdown=best_breakdown,
        cost_history=history,
        n_iters=last_iter,
        elapsed_sec=time.time() - t0,
        n_accepted=n_accepted,
        n_improved=n_improved,
    )

# ============================================================
# Phase 2 FILTERED variants (Day 8)
# ============================================================

from core.filter import (
    FilterConfig,
    FilterDiagnostics,
    safety_margin,
    adaptive_n0,
    cheap_proxy_insertion_phi,
    filter_passes,
)


def repair_greedy_wdro_filtered(
    routes: list[Route],
    to_insert: list[int],
    inst: Instance,
    scenarios: np.ndarray,
    alpha: float,
    epsilon: float,
    penalty_lambda: float,
    rng: np.random.Generator,
    filter_cfg: FilterConfig,
    T_k: float,
    diagnostics: FilterDiagnostics | None = None,
    max_vehicles: int | None = None,
) -> tuple[list[Route], list[RouteCache]] | None:
    """Filtered greedy repair: prune candidates via cheap proxy before exact eval.

    Each candidate (route, position) is first scored by an O(n_0 log n_0)
    proxy. If proxy_total_delta exceeds best_so_far + Gamma*, the exact
    O(N log N) evaluation is SKIPPED. Otherwise, escalate to exact.

    Diagnostics object (if provided) tracks proxy / exact / prune counts.
    """
    D = inst.distances()
    xi_nom = inst.nominal_xi()
    Q = inst.Q
    N = scenarios.shape[0]
    C_max = Q  # Option alpha: M14

    gamma = safety_margin(C_max, alpha, filter_cfg.kappa, filter_cfg.lambda_)

    sol = [r.copy() for r in routes]
    caches = [RouteCache(r, scenarios, inst.n) for r in sol]
    phi_per_route = [_phi_from_cache(c, inst, alpha, epsilon) for c in caches]

    pending = list(to_insert)
    rng.shuffle(pending)

    for c in pending:
        best_total = None  # (total_delta, ri, pos, phi_after_exact)

        for ri, (r, cache) in enumerate(zip(sol, caches)):
            for pos in range(1, len(r) + 2):
                # 1. Exact travel delta (fast)
                prev = 0 if pos == 1 else r.customers[pos - 2] + 1
                nxt = 0 if pos == len(r) + 1 else r.customers[pos - 1] + 1
                travel_delta = D[prev, c + 1] + D[c + 1, nxt] - D[prev, nxt]

                # 2. Nominal feasibility check
                trial = r.copy()
                trial.insert(c, pos=pos)
                peak_nom = trial.peak_load(xi_nom, inst.n)
                if peak_nom > Q + 1e-6:
                    continue

                # 3. Filter step: cheap proxy
                if filter_cfg.enabled and best_total is not None:
                    n_0 = adaptive_n0(T_k, filter_cfg, N)
                    #sub_indices = rng.integers(0, N, size=n_0)
                    start = int(rng.integers(0, max(1, N - n_0 + 1)))
                    sub_indices = slice(start, start + n_0)
                    proxy_phi = cheap_proxy_insertion_phi(
                        cache, c, pos, inst, alpha, epsilon, sub_indices,
                    )
                    proxy_delta = proxy_phi - phi_per_route[ri]
                    proxy_total = travel_delta + penalty_lambda * proxy_delta
                    if diagnostics:
                        diagnostics.n_proxy_evals += 1

                    if not filter_passes(proxy_total, best_total[0], penalty_lambda * gamma):
                        if diagnostics:
                            diagnostics.n_pruned += 1
                        continue  # PRUNED

                # 4. Exact evaluation (only if filter passes or first candidate)
                phi_after = evaluate_phi_insertion_via_cache(
                    cache, c, pos, inst, alpha, epsilon,
                )
                if diagnostics:
                    diagnostics.n_exact_evals += 1

                phi_delta = phi_after - phi_per_route[ri]
                total_delta = travel_delta + penalty_lambda * phi_delta

                if best_total is None or total_delta < best_total[0]:
                    best_total = (total_delta, ri, pos, phi_after)

        if best_total is not None:
            _, ri, pos, phi_new = best_total
            sol[ri].insert(c, pos=pos)
            caches[ri] = RouteCache(sol[ri], scenarios, inst.n)
            phi_per_route[ri] = phi_new
        else:
            # Open new route
            if max_vehicles is None or len(sol) < max_vehicles:
                new_r = Route([c])
                if new_r.peak_load(xi_nom, inst.n) > Q + 1e-6:
                    return None
                sol.append(new_r)
                new_cache = RouteCache(new_r, scenarios, inst.n)
                caches.append(new_cache)
                phi_per_route.append(_phi_from_cache(new_cache, inst, alpha, epsilon))
            else:
                return None

    return sol, caches


@dataclass
class WDROFilteredConfig(WDROConfig):
    """Extends WDROConfig with filter sub-config."""
    filter_cfg: FilterConfig = field(default_factory=FilterConfig)
    enable_diagnostics: bool = False


def alns_sa_wdro_filtered(
    initial_solution: list[Route],
    inst: Instance,
    scenarios: np.ndarray,
    config: WDROFilteredConfig,
) -> tuple[WDROALNSResult, FilterDiagnostics]:
    """W-DRO ALNS-SA with Phase 2 filter integrated into repair."""
    rng = np.random.default_rng(config.seed)
    diagnostics = FilterDiagnostics() if config.enable_diagnostics else None

    current = [r.copy() for r in initial_solution]
    current_caches = [RouteCache(r, scenarios, inst.n) for r in current]
    breakdown = solution_breakdown(
        current, current_caches, inst,
        config.alpha, config.epsilon, config.penalty_lambda,
    )
    current_obj = breakdown["objective"]

    best = [r.copy() for r in current]
    best_obj = current_obj
    best_breakdown = breakdown.copy()

    T = max(config.T_init_frac * current_obj, 1e-6)
    history = [current_obj]
    n_accepted = 0
    n_improved = 0
    n_customers_total = sum(len(r) for r in current)

    t0 = time.time()
    last_iter = 0

    for k in range(1, config.max_iters + 1):
        last_iter = k
        if config.time_limit_sec is not None:
            if time.time() - t0 > config.time_limit_sec:
                break

        frac = rng.uniform(config.destroy_frac_min, config.destroy_frac_max)
        n_remove = max(2, int(frac * n_customers_total))

        # Destroy (un-filtered for now; could filter symmetrically later)
        if rng.random() < 0.5:
            partial, removed = destroy_random(current, n_remove, rng)
        else:
            partial, removed = destroy_worst_wdro(
                current, current_caches, inst,
                config.alpha, config.epsilon, n_remove, rng,
            )

        # Repair WITH FILTER
        repair_result = repair_greedy_wdro_filtered(
            partial, removed, inst, scenarios,
            config.alpha, config.epsilon, config.penalty_lambda, rng,
            filter_cfg=config.filter_cfg,
            T_k=T,
            diagnostics=diagnostics,
            max_vehicles=config.max_vehicles,
        )
        if repair_result is None:
            continue
        candidate, cand_caches = repair_result

        served = set()
        for r in candidate:
            served.update(r.customers)
        if served != set(range(inst.n)):
            continue

        cand_bd = solution_breakdown(
            candidate, cand_caches, inst,
            config.alpha, config.epsilon, config.penalty_lambda,
        )
        cand_obj = cand_bd["objective"]
        delta = cand_obj - current_obj

        accept = False
        if delta < 0:
            accept = True
            if cand_obj < best_obj:
                best = [r.copy() for r in candidate]
                best_obj = cand_obj
                best_breakdown = cand_bd.copy()
                n_improved += 1
        else:
            if T > 1e-12 and rng.random() < np.exp(-delta / T):
                accept = True

        if accept:
            current = candidate
            current_caches = cand_caches
            current_obj = cand_obj
            n_accepted += 1

        history.append(current_obj)
        T *= config.alpha_cooling

        if config.verbose_every and k % config.verbose_every == 0:
            elapsed = time.time() - t0
            diag_str = ""
            if diagnostics:
                pr = diagnostics.prune_rate
                diag_str = f" prune={pr:.2%}"
            print(f"[iter {k:5d}] T={T:.2f} cur={current_obj:.0f} "
                  f"best={best_obj:.0f}{diag_str} t={elapsed:.1f}s")

    result = WDROALNSResult(
        best_solution=best,
        best_objective=best_obj,
        best_breakdown=best_breakdown,
        cost_history=history,
        n_iters=last_iter,
        elapsed_sec=time.time() - t0,
        n_accepted=n_accepted,
        n_improved=n_improved,
    )
    return result, (diagnostics if diagnostics else FilterDiagnostics())
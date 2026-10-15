"""Tests for rarecert/baselines.py (RARECERT revision task A, referee items 3.1-3.4).

Covers: the factor quadrature weights; the exact conditional route probability against the
existing dp_check (trapezoid within 0.5 per cent, Gauss-Hermite identical to dp_check); the
exact hub probability of a two-route hub against crude Monte Carlo (1e6 draws, within 4 rse);
the rivals (SMIS, Glasserman-Li two-step crude and shift, conditional MC, subset simulation)
against the exact route reference at gamma about 1e-3 on an L = 6 route; the subset
simulation evaluation count; and the zero hub probability when no route can reach c.
The DP-based tests dominate the run time (about 2 minutes on one core).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from quantum.rare import RouteSim, dp_check
from rarecert.baselines import (
    cond_mc,
    cond_shift,
    dominant_point,
    exact_hub_prob,
    exact_reference,
    exact_route_prob,
    factor_rule,
    gl_two_step,
    subset_sim,
)
from rarecert.plan import HubSim
from rarecert.smis import smis_estimate

CV = 0.3
RHO = 0.6
GAMMA = 1e-3
N_TRAPZ = 41


def _route6() -> RouteSim:
    rng = np.random.default_rng(5)
    dbar = rng.uniform(800.0, 1600.0, 6)
    pbar = rng.uniform(800.0, 1600.0, 6)
    return RouteSim(dbar, pbar, cv=CV, rho=RHO)


def _crude_quantile(sim: RouteSim, gamma: float, n: int = 400_000, seed: int = 3) -> float:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, sim.dim))
    return float(np.quantile(sim.peak(x), 1.0 - gamma))


@lru_cache(maxsize=1)
def _route_case():
    """Shared L = 6 route, its threshold at gamma about 1e-3, and the trapezoid reference at
    V = 256 (computed once, reused by the estimator tests)."""
    sim = _route6()
    c = _crude_quantile(sim, GAMMA)
    p_ref = exact_route_prob(sim, c, V=256, rule="trapz", n=N_TRAPZ)["p"]
    return sim, c, p_ref


def _two_route_hub(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    dbar = np.concatenate([[0.0], rng.uniform(500.0, 1500.0, 6)])
    pbar = np.concatenate([[0.0], rng.uniform(500.0, 1500.0, 6)])
    return {"dbar": dbar.tolist(), "pbar": pbar.tolist(),
            "routes": [[1, 3, 5], [2, 4, 6]], "Q": 6000.0}


def test_factor_rules_integrate_gaussian_moments():
    z, w = factor_rule("trapz", N_TRAPZ)
    # the box [-6, 9] misses the Gaussian mass 1 - Phi(6) = 9.9e-10 below -6 (no weight: g ~ 0 there)
    assert abs(w.sum() - 1.0) < 2e-9
    assert abs(float(w @ (z * z)) - 1.0) < 1e-6
    zg, wg = factor_rule("gh", 40)
    assert abs(wg.sum() - 1.0) < 1e-12
    assert abs(float(wg @ (zg * zg)) - 1.0) < 1e-10


def test_exact_route_matches_dp_check_at_gamma_1e3():
    sim, c, p_trapz = _route_case()
    assert sim.L == 6
    p_dp = dp_check(sim, c, V=256, n_gh=40)
    assert p_dp > 0.0
    assert abs(p_trapz / p_dp - 1.0) < 5e-3, (p_trapz, p_dp)
    # the Gauss-Hermite rule of exact_route_prob is the dp_check computation
    p_gh = exact_route_prob(sim, c, V=256, rule="gh", n=40)["p"]
    assert abs(p_gh / p_dp - 1.0) < 1e-9, (p_gh, p_dp)


def test_exact_hub_two_routes_vs_crude_mc():
    plan = _two_route_hub()
    hub = HubSim(plan, cv=CV, rho=RHO)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((200_000, hub.dim))
    c = float(np.quantile(hub.peaks(x).max(axis=1), 0.99))   # P_hub about 1e-2
    rng = np.random.default_rng(12)
    N = 1_000_000
    hits, done = 0, 0
    while done < N:
        n = min(200_000, N - done)
        xs = rng.standard_normal((n, hub.dim))
        hits += int(np.count_nonzero(hub.peaks(xs).max(axis=1) > c))
        done += n
    p_mc = hits / N
    rse_mc = math.sqrt((1.0 - p_mc) / (p_mc * N))
    out = exact_hub_prob(hub, c, V=256, rule="trapz", n=N_TRAPZ)
    assert out["R"] == 2
    assert abs(out["p"] - p_mc) <= 4.0 * rse_mc * p_mc + 0.005 * p_mc, (out["p"], p_mc, rse_mc)


def test_estimators_agree_with_exact_route_at_gamma_1e3():
    sim, c, p_ref = _route_case()
    tol_disc = 0.005 * p_ref          # V = 256 discretisation allowance (see module docstring)

    res = smis_estimate(sim, c, 100_000, np.random.default_rng(21))
    assert abs(res["p_hat"] - p_ref) <= 4.0 * res["rse"] * p_ref + tol_disc, (res["p_hat"], p_ref)

    x_star, _ = dominant_point(sim, c)
    mu_t, _, _ = cond_shift(sim, c, V=256, mu_dom=x_star[:2])
    for variant in ("crude", "shift"):
        p, RV, rse = gl_two_step(sim, c, 200_000, np.random.default_rng(22), variant=variant,
                                 x_star=x_star, mu=mu_t)
        assert abs(p - p_ref) <= 4.0 * rse * p_ref + tol_disc, (variant, p, p_ref, rse)

    # loose check: at 2000 draws the relative error of cond MC is about 15 per cent
    p, RV, rse = cond_mc(sim, c, 2000, np.random.default_rng(23), shift=True, mu=mu_t)
    assert abs(p - p_ref) <= 4.0 * rse * p_ref + tol_disc, (p, p_ref, rse)

    ps = []
    for k in range(10):
        p_k, n_ev = subset_sim(sim, c, 2000, np.random.default_rng(100 + k))
        ps.append(p_k)
    ps = np.asarray(ps)
    mean, sd = float(ps.mean()), float(ps.std(ddof=1))
    assert abs(mean - p_ref) <= 3.0 * sd + tol_disc, (mean, sd, p_ref)


def test_subset_simulation_evaluation_count_and_range():
    sim = _route6()
    c = _crude_quantile(sim, 1e-2, n=100_000, seed=4)
    N = 2000
    n_seed = int(round(0.1 * N))
    p, n_ev = subset_sim(sim, c, N, np.random.default_rng(31))
    assert 0.0 < p < 1.0
    levels = (n_ev - N) / (N - n_seed)
    assert abs(levels - round(levels)) < 1e-12 and round(levels) >= 1
    crude = float(np.mean(sim.peak(np.random.default_rng(32).standard_normal((200_000, sim.dim))) > c))
    rse_c = math.sqrt((1.0 - crude) / (crude * 200_000))
    assert abs(p - crude) <= 4.0 * rse_c * crude + 0.2 * crude, (p, crude)


def test_exact_hub_zero_when_unreachable():
    hub = HubSim(_two_route_hub(), cv=CV, rho=RHO)
    out = exact_hub_prob(hub, 1e12, V=256, rule="trapz", n=N_TRAPZ)
    assert out["p"] == 0.0 and out["R"] == 0 and out["dp_calls"] == 0

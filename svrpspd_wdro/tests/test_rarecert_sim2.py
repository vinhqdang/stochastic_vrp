"""Tests for rarecert/sim2.py (RouteDay, route-day simulator with travel and service
times) and rarecert/modes.py (mode-decomposed IS, RARECERT R2). Runs in under three
minutes.

Covers: nominal arrivals and the arrival recursion against a direct loop, the load
part against quantum.rare.RouteSim.peak, the mean of the travel times, the boundary
property and energy bound of dominating_point, the agreement of MDIS with crude
Monte Carlo on a short route at gamma ~ 1e-2, and the mean one of the mixture
likelihood ratio."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from quantum.rare import RouteSim, _draws
from rarecert import modes as M
from rarecert.sim2 import RouteDay, S0, S1, SPEED_KMH, SIGMA_T

CV = 0.3
RHO = 0.6
DBAR = [1000.0, 800.0, 1200.0, 900.0]
PBAR = [900.0, 700.0, 1100.0, 800.0]
LEGS = [6.0, 2.0, 3.0, 1.5, 7.0]              # km: depot->c1, c1->c2, c2->c3, c3->c4, c4->depot


def _day(Qc=0.0, slack=0.0):
    return RouteDay(DBAR, PBAR, LEGS, cv=CV, rho=RHO, Qc=Qc, slack=slack)


# ------------------------------------------------------------------ simulator


def test_nominal_arrivals_and_recursion_match_direct_loop():
    sim = _day(Qc=2000.0, slack=30.0)
    L = sim.L
    arr0, ret0 = sim.arrivals_and_return(np.zeros((1, sim.dim)))
    assert np.allclose(arr0[0], sim.Abar)
    assert abs(ret0[0] - sim.Rbar) < 1e-12
    # direct loop at x = 0: T_l = tbar exp(-sigma^2/2), S_j from median demands
    d, p = sim.demand.demands(np.zeros((1, sim.n_dem)))
    S = S0 + S1 * (d[0] + p[0]) / 1000.0
    T = sim.tbar * math.exp(-0.5 * SIGMA_T * SIGMA_T)
    A_direct = [sum(T[l] for l in range(i)) + sum(S[j] for j in range(i - 1)) for i in range(1, L + 1)]
    R_direct = sum(T) + sum(S)
    assert np.allclose(sim.Abar, A_direct, rtol=1e-12)
    assert abs(sim.Rbar - R_direct) < 1e-9

    rng = np.random.default_rng(5)
    x = rng.standard_normal((6, sim.dim))
    arr, ret = sim.arrivals_and_return(x)
    d, p = sim.demand.demands(x[:, :sim.n_dem])
    T = sim.travel_times(x)
    S = S0 + S1 * (d + p) / 1000.0
    for n in range(x.shape[0]):
        for i in range(1, L + 1):
            a = sum(T[n, l] for l in range(i)) + sum(S[n, j] for j in range(i - 1))
            assert abs(arr[n, i - 1] - a) <= 1e-9 * max(1.0, a)
        r = sum(T[n]) + sum(S[n])
        assert abs(ret[n] - r) <= 1e-9 * max(1.0, r)


def test_load_part_equals_routesim_peak():
    Qc = 2500.0
    sim = _day(Qc=Qc)
    route = RouteSim(DBAR, PBAR, cv=CV, rho=RHO)
    rng = np.random.default_rng(6)
    x = rng.standard_normal((500, sim.dim))
    assert np.allclose(sim.loads(x).max(axis=1), route.peak(x[:, :route.dim]), rtol=0, atol=1e-9)
    g = sim.modes(x)
    assert np.allclose(g[:, :sim.L + 1], sim.loads(x) - Qc)
    fail_load = (g[:, :sim.L + 1].max(axis=1) > 0)
    assert np.array_equal(fail_load, route.peak(x[:, :route.dim]) > Qc)


def test_travel_time_mean_is_tbar():
    sim = _day()
    rng = np.random.default_rng(7)
    x = rng.standard_normal((200_000, sim.dim))
    T = sim.travel_times(x)
    ratio = T.mean(axis=0) / sim.tbar
    assert np.all(np.abs(ratio - 1.0) < 0.02), ratio
    assert np.allclose(sim.tbar, 60.0 * np.array(LEGS) / SPEED_KMH)


# ------------------------------------------------------------------ dominating points


def test_dominating_point_boundary_and_energy_below_ray():
    sim = _day(Qc=0.0, slack=40.0)
    m = sim.L + 2                                    # late_2
    fun = lambda X: sim.normalised_modes(X)[:, m]
    u0 = np.zeros(sim.dim)
    u0[sim.i_H + 0] = 1.0                            # leg 0 noise alone
    u0[sim.i_H + 1] = 0.5
    res = M.dominating_point(fun, u0)
    assert res is not None
    x, E = res
    assert abs(float(fun(x[None, :])[0])) < 1e-6
    assert abs(E - 0.5 * float(x @ x)) < 1e-12
    # ray point of the same direction
    uu = u0 / np.linalg.norm(u0)
    t0 = brentq(lambda t: float(fun((t * uu)[None, :])[0]), 0.0, 200.0)
    assert E <= 0.5 * t0 * t0 * (1.0 + 1e-9)


def test_mode_starts_and_mixture_defensive_weight():
    sim = _day(Qc=2500.0, slack=20.0)
    for m in range(sim.n_modes):
        starts = M.mode_starts(sim, m)
        assert len(starts) >= 1
        for _, u in starts:
            assert u.shape == (sim.dim,)
            assert abs(np.linalg.norm(u) - 1.0) < 1e-9 or np.linalg.norm(u) == 0.0
    mix, diag = M.mdis_mixture(sim)
    assert abs(mix.weights.sum() - 1.0) < 1e-12
    assert mix.weights[0] == 0.05
    assert diag["n_modes"] >= 1 and diag["n_modes"] <= M.MAX_COMP


# ------------------------------------------------------------------ estimators


def _short_route_thresholds(sim, gamma, rng, n=400_000):
    """Simple empirical thresholds: capacity and slack at the (1 - gamma/2) quantiles of
    the load peak and of the time-mode maximum M = max(max_i (A_i - Abar_i), (R - Rbar)/2)
    (time failure at slack s is M > s)."""
    x = rng.standard_normal((n, sim.dim))
    peak = sim.loads(x).max(axis=1)
    arr, ret = sim.arrivals_and_return(x)
    Mt = np.maximum((arr - sim.Abar[None, :]).max(axis=1), 0.5 * (ret - sim.Rbar))
    q = 1.0 - gamma / 2.0
    return float(np.quantile(peak, q)), float(max(np.quantile(Mt, q), 0.0))


def test_mdis_agrees_with_crude_on_short_route_at_gamma_1e2():
    gamma = 1e-2
    base = _day()
    rng = np.random.default_rng(11)
    Qc, slack = _short_route_thresholds(base, gamma, rng)
    sim = base.with_thresholds(Qc, slack)
    mix, _ = M.mdis_mixture(sim)
    est = M.is_generic(sim, mix.thetas, mix.log_a, 200_000, rng)
    crude_rng = np.random.default_rng(12)
    crude = M.crude_generic(sim, 1_000_000, crude_rng)
    rse_c = math.sqrt((1.0 - crude["p"]) / (crude["p"] * 1_000_000))
    comb = math.sqrt((est["rse"] * est["p"]) ** 2 + (rse_c * crude["p"]) ** 2)
    assert crude["p"] > 0.0 and est["p"] > 0.0
    assert abs(est["p"] - crude["p"]) <= 4.0 * comb, (est, crude)
    # the thresholds are near the target level
    assert 0.3 * gamma < crude["p"] < 3.0 * gamma


def test_mixture_likelihood_ratio_has_mean_one():
    sim = _day(Qc=2500.0, slack=20.0)
    mix, _ = M.mdis_mixture(sim)
    rng = np.random.default_rng(13)
    x, logw = _draws(mix.thetas, mix.log_a, 200_000, rng)
    w = np.exp(logw)
    assert abs(float(w.mean()) - 1.0) < 0.02, float(w.mean())

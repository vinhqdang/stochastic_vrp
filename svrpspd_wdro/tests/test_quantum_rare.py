"""Tests for quantum/rare.py (QUASAR direction A: rare-event route overflow).

Covers the route simulator (peak against experiments.route_peak, stage loads,
gamma marginals, copula correlation), crude and importance sampling, the
cross-entropy shift, the mixture proposal over stage events, the threshold search,
the exact discretised cross-check (against classical.chance on small instances,
against IS on an L = 4 route, and against the mixture IS on a two-mode L = 8 route),
and the cost functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from quantum.cert import h as cert_h, r_needed
from quantum.classical import DiscreteDemand, chance
from quantum.experiments import route_peak
from quantum.rare import (
    RouteSim,
    _draws,
    aqe_cost,
    aqe_is_cost,
    best_aqe,
    c_crude,
    c_is,
    ce_mixture,
    ce_shift,
    crude,
    dp_check,
    exceed_prob,
    is_mixture,
    is_shift,
    threshold_for,
)

DBAR = [1000.0, 800.0, 1200.0, 900.0]
PBAR = [600.0, 700.0, 500.0, 800.0]


def _sim():
    return RouteSim(DBAR, PBAR, cv=0.3, rho=0.6)


# ------------------------------------------------------------------ simulator


def test_peak_matches_route_peak():
    sim = _sim()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3000, sim.dim))
    d, p = sim.demands(x)
    ref = route_peak(list(range(sim.L)), d.T, p.T)
    assert np.allclose(sim.peak(x), ref, rtol=0, atol=1e-9)


def test_peak_chunking_is_consistent():
    sim = _sim()
    x = np.random.default_rng(1).standard_normal((500, sim.dim))
    assert np.array_equal(sim.peak(x, chunk=64), sim.peak(x, chunk=10_000))


def test_stage_loads_consistent_with_peak():
    sim = _sim()
    x = np.random.default_rng(14).standard_normal((2000, sim.dim))
    d, p = sim.demands(x)
    assert np.allclose(sim.peak(x, stage=0), d.sum(axis=1), rtol=0, atol=1e-9)
    stage_max = np.max(np.stack([sim.peak(x, stage=k) for k in range(sim.L + 1)]), axis=0)
    assert np.allclose(stage_max, sim.peak(x), rtol=0, atol=1e-9)
    with pytest.raises(ValueError):
        sim.peak(x, stage=sim.L + 1)


def test_demands_validate_shape():
    sim = _sim()
    with pytest.raises(ValueError):
        sim.demands(np.zeros((5, sim.dim + 1)))


def test_demands_marginal_mean_and_cv():
    sim = _sim()
    rng = np.random.default_rng(2)
    x = rng.standard_normal((200_000, sim.dim))
    d, p = sim.demands(x)
    for i in range(sim.L):
        for samples, mu in ((d[:, i], DBAR[i]), (p[:, i], PBAR[i])):
            assert abs(samples.mean() / mu - 1.0) < 0.03
            assert abs(samples.std() / samples.mean() / 0.3 - 1.0) < 0.03


def test_demands_copula_rank_correlation():
    sim = _sim()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((200_000, sim.dim))
    d, _ = sim.demands(x)
    rho_s = stats.spearmanr(d[:, 0], d[:, 1]).statistic
    implied = 6.0 / math.pi * math.asin(0.6 / 2.0)     # normal-score correlation rho = 0.6
    assert abs(rho_s - implied) < 0.02


# ------------------------------------------------------------------ sampling


def test_is_theta_zero_matches_crude():
    sim = _sim()
    rng = np.random.default_rng(4)
    c = threshold_for(sim, 1e-2, np.random.default_rng(5))
    N = 200_000
    p_c, rse_c = crude(sim, c, N, rng)
    p_i, RV, rse_i, _ = is_shift(sim, c, np.zeros(sim.dim), N, rng)
    combined = math.sqrt(rse_c ** 2 * p_c ** 2 + rse_i ** 2 * p_i ** 2) / p_c
    assert abs(p_i - p_c) <= 4 * combined * p_c
    assert abs(RV - (1 - p_c) / p_c) / ((1 - p_c) / p_c) < 0.1


def test_ce_shift_agrees_with_crude_and_reduces_variance():
    sim = _sim()
    rng = np.random.default_rng(6)
    c = threshold_for(sim, 1e-2, rng)
    theta = ce_shift(sim, c, rng)
    N = 200_000
    p_c, rse_c = crude(sim, c, N, rng)
    p_i, RV, rse_i, _ = is_shift(sim, c, theta, N, rng)
    combined = math.sqrt(rse_c ** 2 * p_c ** 2 + rse_i ** 2 * p_i ** 2) / p_c
    assert abs(p_i - p_c) <= 4 * combined * p_c
    assert RV < 0.1 * (1 - p_i) / p_i


def test_mixture_loglik_reduces_to_shift_for_equal_components():
    sim = _sim()
    rng = np.random.default_rng(15)
    theta = rng.normal(size=sim.dim) * 0.7
    thetas = np.vstack([theta, theta])
    log_a = np.log(np.array([0.5, 0.5]))
    x, logw_mix = _draws(thetas, log_a, 500, np.random.default_rng(16))
    logw_shift = -(x @ theta) + 0.5 * float(theta @ theta)
    assert np.allclose(logw_mix, logw_shift, rtol=0, atol=1e-9)


def test_ce_mixture_structure():
    sim = _sim()
    c = threshold_for(sim, 1e-3, np.random.default_rng(17))
    mix = ce_mixture(sim, c, np.random.default_rng(18))
    assert mix.weights.sum() == pytest.approx(1.0)
    assert mix.weights[0] == pytest.approx(0.05)
    assert mix.stages[0] is None and len(mix.stages) == mix.thetas.shape[0] <= 13
    assert np.all(mix.q > 0) and np.all(np.diff(mix.q) <= 0)
    assert mix.draws > 0


def test_is_rejects_wrong_theta_length():
    sim = _sim()
    with pytest.raises(ValueError):
        is_shift(sim, 5000.0, np.zeros(3), 100, np.random.default_rng(0))


def test_threshold_hits_gamma():
    sim = _sim()
    gamma = 1e-3
    c = threshold_for(sim, gamma, np.random.default_rng(7))
    theta = ce_shift(sim, c, np.random.default_rng(8))
    p, _, _, _ = is_shift(sim, c, theta, 200_000, np.random.default_rng(9))
    assert abs(p / gamma - 1.0) < 0.15


# ------------------------------------------------------------------ exact cross-check


def test_exceed_prob_matches_classical_chance():
    rng = np.random.default_rng(10)
    for L, V in ((1, 4), (3, 6), (4, 8)):
        pd = rng.dirichlet(np.ones(V), size=L)
        pp = rng.dirichlet(np.ones(V), size=L)
        dem = DiscreteDemand(pmf_d=pd, pmf_p=pp, unit=1.0)
        for T in range(0, L * (V - 1) + 1, 2):
            assert abs(exceed_prob(pd, pp, T) - chance(list(range(L)), dem, T)) < 1e-12


def test_dp_check_agrees_with_mixture_is_at_gamma_1e3():
    sim = _sim()
    c = threshold_for(sim, 1e-3, np.random.default_rng(11))
    mix = ce_mixture(sim, c, np.random.default_rng(12))
    p_is, _, rse, _ = is_mixture(sim, c, mix, 400_000, np.random.default_rng(13))
    p_dp = dp_check(sim, c, V=256, n_gh=40)
    assert abs(p_dp / p_is - 1.0) < max(4 * rse, 0.02)


# Two failure modes: heavy deliveries early (L_0 = sum d large) and heavy pickups late
# (L_L = sum p large). A single mean shift cannot cover both ends of the route.
TWO_MODE_D = [1000.0] * 4 + [100.0] * 4
TWO_MODE_P = [100.0] * 4 + [1000.0] * 4


def test_mixture_matches_dp_on_two_mode_route():
    sim = RouteSim(TWO_MODE_D, TWO_MODE_P, cv=0.3, rho=0.6)
    gamma = 1e-4
    c = threshold_for(sim, gamma, np.random.default_rng(19))
    mix = ce_mixture(sim, c, np.random.default_rng(20))
    p_mix, RV_mix, _, _ = is_mixture(sim, c, mix, 400_000, np.random.default_rng(21))
    theta = ce_shift(sim, c, np.random.default_rng(22))
    p_single, RV_single, _, _ = is_shift(sim, c, theta, 400_000, np.random.default_rng(23))
    p_dp = dp_check(sim, c, V=256, n_gh=40)
    assert abs(p_dp / p_mix - 1.0) < 0.03
    # The single shift covers one of the two modes. Its bias is seed-dependent on this
    # route (2% to 55% in different runs), so the stable signal is its relative variance:
    # the weights are heavy-tailed and RV_single is orders of magnitude above RV_mix.
    assert abs(p_dp / p_single - 1.0) > 0.03 or RV_single > 100.0 * RV_mix


def test_dp_check_rejects_long_routes():
    sim = RouteSim([100.0] * 9, [50.0] * 9)
    with pytest.raises(ValueError):
        dp_check(sim, 1000.0, V=64, n_gh=8)


# ------------------------------------------------------------------ costs


def test_classical_sample_counts_hand_values():
    assert c_crude(0.01, 0.1) == pytest.approx(1.96 ** 2 * 0.99 / (0.01 * 0.01))
    assert c_is(10.0, 0.1) == pytest.approx(1.96 ** 2 * 10.0 / 0.01)


def test_aqe_cost_hand_value_k1():
    p, eps, delta = 0.25, 0.5, 0.05
    out = aqe_cost(p, eps, delta, k=1)
    # h_{M,1}(0.1875) = 2 pi sqrt(0.1875)/M + pi^2/M^2 <= 0.125 first holds at M = 32
    assert out["M"] == 32
    r = r_needed(1, delta)
    assert out["ideal"] == r * (2 * 32 + 1)
    assert cert_h(16, 1, 0.1875) > eps * p >= cert_h(32, 1, 0.1875)


def test_aqe_cost_decreasing_in_eps():
    p = 1e-4
    costs = [aqe_cost(p, eps, 0.05, k=3) for eps in (0.1, 0.03, 0.01)]
    assert costs[0]["ideal"] < costs[1]["ideal"] < costs[2]["ideal"]
    assert costs[0]["doubling"] < costs[1]["doubling"] < costs[2]["doubling"]


def test_q_is_at_most_q_crude_when_rv_is_crude_variance():
    for p in (1e-3, 0.05, 0.3):
        RV = (1 - p) / p
        for eps in (0.1, 0.01):
            qc = aqe_cost(p, eps, 0.05, k=3)
            qi = aqe_is_cost(RV, p, eps, 0.05, k=3)
            assert qi["ideal"] <= qc["ideal"]
            assert qi["doubling"] <= qc["doubling"]


def test_q_is_constant_in_p_for_fixed_rv():
    RV = 25.0
    for eps in (0.1, 0.01):
        costs = [aqe_is_cost(RV, p, eps, 0.05, k=3) for p in (1e-3, 1e-5, 1e-6, 0.2)]
        assert all(c == costs[0] for c in costs)


def test_best_aqe_is_minimum_over_k():
    p, eps = 1e-3, 0.03
    best = best_aqe(lambda k, d: aqe_cost(p, eps, d, k), ks=(1, 3, 5))
    per_k = [aqe_cost(p, eps, 0.05, k)["ideal"] for k in (1, 3, 5)]
    assert best["ideal"] == min(per_k)

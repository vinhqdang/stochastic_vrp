"""Tests for W-DRO exact evaluator.

The critical test gate of Day 3 is COLLAPSE:
    variance = 0 (constant scenarios)
    epsilon  = 0
    -> Phi(r) = max(0, f_r(xi_bar) - Q)   (deterministic violation cost)

If this collapse fails, the W-DRO machinery has a bug somewhere.
"""

import numpy as np
import pytest

from core.instance import Instance
from core.route import Route
from core.scenarios import ScenarioConfig, generate_scenarios
from core.wdro_exact import (
    empirical_cvar,
    evaluate_phi_exact,
    evaluate_phi_total,
    evaluate_objective,
)


@pytest.fixture
def toy_instance(tmp_path):
    """5 customers, capacity 40."""
    p = tmp_path / "toy.txt"
    p.write_text(
        "5\n40\n"
        "50 50 0 0\n"
        "20 30 10 5\n"
        "80 70 8 12\n"
        "30 80 15 3\n"
        "70 20 6 14\n"
        "60 60 9 7\n"
    )
    return Instance.from_file(p)


# ============================================================
# A. Empirical CVaR sanity
# ============================================================


class TestEmpiricalCVaR:

    def test_cvar_alpha_zero_equals_mean(self):
        losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(empirical_cvar(losses, alpha=0.0) - 3.0) < 1e-9

    def test_cvar_alpha_high_equals_top_few(self):
        """N=5, alpha=0.8 -> k=ceil(0.2*5)=1 -> top-1 = max = 5."""
        losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(empirical_cvar(losses, alpha=0.8) - 5.0) < 1e-9

    def test_cvar_alpha_half_equals_top_half_mean(self):
        """N=4, alpha=0.5 -> k=2 -> mean of top 2 = (3+4)/2 = 3.5."""
        losses = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(empirical_cvar(losses, alpha=0.5) - 3.5) < 1e-9

    def test_cvar_constant_losses(self):
        """All losses identical -> CVaR = that constant for any alpha."""
        losses = np.full(100, 5.0)
        for alpha in (0.0, 0.5, 0.9, 0.99):
            assert abs(empirical_cvar(losses, alpha) - 5.0) < 1e-9

    def test_cvar_monotone_in_alpha(self):
        """CVaR is non-decreasing in alpha (more weight on tail)."""
        rng = np.random.default_rng(42)
        losses = rng.gamma(2.0, 1.0, size=1000)
        alphas = [0.0, 0.5, 0.9, 0.95, 0.99]
        cvars = [empirical_cvar(losses, a) for a in alphas]
        for i in range(len(cvars) - 1):
            assert cvars[i] <= cvars[i + 1] + 1e-9

    def test_cvar_invalid_alpha_rejected(self):
        with pytest.raises(ValueError):
            empirical_cvar(np.array([1.0]), alpha=1.0)
        with pytest.raises(ValueError):
            empirical_cvar(np.array([1.0]), alpha=-0.01)

    def test_cvar_empty_losses(self):
        """Edge case: empty array."""
        assert empirical_cvar(np.array([]), alpha=0.5) == 0.0


# ============================================================
# B. Phi collapse test (THE critical Day 3 gate)
# ============================================================


class TestPhiCollapse:
    """variance=0 + epsilon=0 -> Phi(r) = max(0, f_r(xi_bar) - Q)."""

    def test_collapse_feasible_route(self, toy_instance):
        """Feasible route under nominal -> Phi = 0."""
        route = Route([0, 1])
        nominal = toy_instance.nominal_xi()
        peak = route.peak_load(nominal, toy_instance.n)
        assert peak <= toy_instance.Q  # precondition: feasible

        cfg = ScenarioConfig(distribution="constant", cv=0.0)
        scenarios = generate_scenarios(toy_instance, 1000, cfg)

        phi = evaluate_phi_exact(
            route, toy_instance, scenarios, alpha=0.95, epsilon=0.0
        )
        assert abs(phi) < 1e-9

    def test_collapse_infeasible_route(self, toy_instance):
        """Infeasible route under nominal -> Phi = peak - Q (the violation)."""
        route = Route([0, 1, 2, 3, 4])  # all 5 customers; likely violates Q
        nominal = toy_instance.nominal_xi()
        peak = route.peak_load(nominal, toy_instance.n)
        expected = max(0.0, peak - toy_instance.Q)

        cfg = ScenarioConfig(distribution="constant", cv=0.0)
        scenarios = generate_scenarios(toy_instance, 1000, cfg)

        phi = evaluate_phi_exact(
            route, toy_instance, scenarios, alpha=0.95, epsilon=0.0
        )
        assert abs(phi - expected) < 1e-9

    def test_collapse_invariant_to_alpha(self, toy_instance):
        """With variance=0, all scenarios identical, so CVaR = that constant
        regardless of alpha."""
        route = Route([0, 1, 2, 3, 4])
        cfg = ScenarioConfig(distribution="constant", cv=0.0)
        scenarios = generate_scenarios(toy_instance, 500, cfg)

        phis = [
            evaluate_phi_exact(route, toy_instance, scenarios, a, epsilon=0.0)
            for a in (0.1, 0.5, 0.9, 0.99)
        ]
        for p in phis[1:]:
            assert abs(p - phis[0]) < 1e-9


# ============================================================
# C. Epsilon regularization (M4 + M3)
# ============================================================


class TestEpsilonRegularization:
    """The +epsilon/(1-alpha) term is linear in epsilon and route-independent."""

    def test_epsilon_linear(self, toy_instance):
        route = Route([0, 1, 2])
        cfg = ScenarioConfig(distribution="gamma", cv=0.2, seed=42)
        scenarios = generate_scenarios(toy_instance, 1000, cfg)
        alpha = 0.9

        phi_0 = evaluate_phi_exact(route, toy_instance, scenarios, alpha, epsilon=0.0)
        phi_1 = evaluate_phi_exact(route, toy_instance, scenarios, alpha, epsilon=1.0)
        phi_2 = evaluate_phi_exact(route, toy_instance, scenarios, alpha, epsilon=2.0)

        assert abs((phi_1 - phi_0) - 1.0 / (1 - alpha)) < 1e-9
        assert abs((phi_2 - phi_0) - 2.0 / (1 - alpha)) < 1e-9

    def test_epsilon_shift_route_independent(self, toy_instance):
        """The epsilon shift is the SAME for any route (M3 -> uniform Lip = 1).
        This is the structural fact enabling epsilon cancellation in Delta-Phi."""
        route1 = Route([0, 1])
        route2 = Route([2, 3, 4])

        cfg = ScenarioConfig(distribution="gamma", cv=0.2, seed=42)
        scenarios = generate_scenarios(toy_instance, 1000, cfg)
        alpha = 0.9
        epsilon = 1.5

        shift1 = (
            evaluate_phi_exact(route1, toy_instance, scenarios, alpha, epsilon)
            - evaluate_phi_exact(route1, toy_instance, scenarios, alpha, 0.0)
        )
        shift2 = (
            evaluate_phi_exact(route2, toy_instance, scenarios, alpha, epsilon)
            - evaluate_phi_exact(route2, toy_instance, scenarios, alpha, 0.0)
        )

        assert abs(shift1 - shift2) < 1e-9
        assert abs(shift1 - epsilon / (1 - alpha)) < 1e-9

    def test_epsilon_negative_rejected(self, toy_instance):
        route = Route([0])
        cfg = ScenarioConfig(distribution="constant", cv=0.0)
        scenarios = generate_scenarios(toy_instance, 100, cfg)
        with pytest.raises(ValueError):
            evaluate_phi_exact(route, toy_instance, scenarios, alpha=0.9, epsilon=-0.1)


# ============================================================
# D. Phi non-negativity and monotonicity
# ============================================================


class TestPhiProperties:

    def test_phi_nonneg_random(self, toy_instance):
        """Phi >= 0 for arbitrary routes."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            n_cust = int(rng.integers(1, 6))
            customers = list(rng.choice(toy_instance.n, size=n_cust, replace=False))
            route = Route(customers)

            cfg = ScenarioConfig(
                distribution="gamma", cv=0.3, seed=int(rng.integers(0, 10000))
            )
            scenarios = generate_scenarios(toy_instance, 500, cfg)

            phi = evaluate_phi_exact(
                route, toy_instance, scenarios, alpha=0.9, epsilon=0.5
            )
            assert phi >= 0.0

    def test_phi_monotone_in_alpha_when_eps_zero(self, toy_instance):
        """At epsilon=0, Phi = CVaR which is monotone in alpha."""
        route = Route([0, 1, 2, 3, 4])
        cfg = ScenarioConfig(distribution="gamma", cv=0.3, seed=42)
        scenarios = generate_scenarios(toy_instance, 5000, cfg)

        phis = [
            evaluate_phi_exact(route, toy_instance, scenarios, a, epsilon=0.0)
            for a in (0.0, 0.5, 0.9, 0.95)
        ]
        for i in range(len(phis) - 1):
            assert phis[i] <= phis[i + 1] + 1e-9

    def test_phi_empty_route_is_zero(self, toy_instance):
        """Empty route: f_r === 0, no customers -> no penalty."""
        cfg = ScenarioConfig(distribution="gamma", cv=0.2, seed=42)
        scenarios = generate_scenarios(toy_instance, 100, cfg)
        phi = evaluate_phi_exact(
            Route([]), toy_instance, scenarios, alpha=0.9, epsilon=1.0
        )
        assert phi == 0.0


# ============================================================
# E. Solution-level aggregation
# ============================================================


class TestPhiTotal:

    def test_total_equals_sum_over_routes(self, toy_instance):
        routes = [Route([0, 1]), Route([2, 3, 4])]
        cfg = ScenarioConfig(distribution="gamma", cv=0.2, seed=42)
        scenarios = generate_scenarios(toy_instance, 500, cfg)
        alpha, epsilon = 0.9, 0.5

        total = evaluate_phi_total(routes, toy_instance, scenarios, alpha, epsilon)
        per_route_sum = sum(
            evaluate_phi_exact(r, toy_instance, scenarios, alpha, epsilon)
            for r in routes
        )
        assert abs(total - per_route_sum) < 1e-9

    def test_objective_breakdown(self, toy_instance):
        """evaluate_objective returns dict with travel + penalty + total."""
        routes = [Route([0, 1, 2])]
        cfg = ScenarioConfig(distribution="constant", cv=0.0)
        scenarios = generate_scenarios(toy_instance, 100, cfg)

        out = evaluate_objective(
            routes, toy_instance, scenarios, alpha=0.9, epsilon=0.0,
            penalty_lambda=1.0,
        )
        assert "travel" in out
        assert "wdro_penalty" in out
        assert "objective" in out
        assert abs(out["objective"] - (out["travel"] + out["wdro_penalty"])) < 1e-9


# ============================================================
# F. Sanity vs manual computation
# ============================================================


class TestPhiManual:
    """Compute Phi by hand on a tiny example and compare."""

    def test_phi_eq_manual_on_random_scenarios(self, toy_instance):
        """Cross-check the function against a manual numpy computation."""
        route = Route([0, 1, 2])
        cfg = ScenarioConfig(distribution="gamma", cv=0.25, seed=7)
        scenarios = generate_scenarios(toy_instance, 1000, cfg)

        # Manual: compute peak loads, violations, CVaR by hand.
        peaks = route.peak_loads_batch(scenarios, toy_instance.n)
        violations = np.maximum(0.0, peaks - toy_instance.Q)
        sorted_v = np.sort(violations)
        # alpha = 0.9 -> k = ceil(0.1 * 1000) = 100
        manual_cvar = sorted_v[-100:].mean()

        phi = evaluate_phi_exact(
            route, toy_instance, scenarios, alpha=0.9, epsilon=0.0
        )
        assert abs(phi - manual_cvar) < 1e-9
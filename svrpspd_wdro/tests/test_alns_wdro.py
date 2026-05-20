"""Tests for W-DRO ALNS-SA integration (Day 7).

Critical test: COLLAPSE — variance=0 + epsilon=0 should reduce the W-DRO
objective to pure travel cost, making W-DRO ALNS behave like deterministic ALNS.
"""

import numpy as np
import pytest

from core.instance import Instance
from core.route import Route
from core.scenarios import ScenarioConfig, generate_scenarios
from core.seeding import phase0_wdro_seeding
from core.alns_wdro import (
    WDROConfig,
    alns_sa_wdro,
    solution_breakdown,
    repair_greedy_wdro,
    destroy_random,
    destroy_worst_wdro,
)
from core.cache import RouteCache


@pytest.fixture
def toy_instance(tmp_path):
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


class TestSolutionBreakdown:

    def test_breakdown_keys(self, toy_instance):
        routes = [Route([0, 1])]
        cfg = ScenarioConfig(distribution="constant", cv=0.0)
        scenarios = generate_scenarios(toy_instance, 50, cfg)
        caches = [RouteCache(r, scenarios, toy_instance.n) for r in routes]
        bd = solution_breakdown(routes, caches, toy_instance,
                                alpha=0.9, epsilon=0.0, penalty_lambda=1.0)
        assert "travel" in bd and "wdro_penalty" in bd and "objective" in bd
        assert abs(bd["objective"] - (bd["travel"] + bd["wdro_penalty"])) < 1e-9

    def test_collapse_phi_zero_for_feasible_with_eps_zero(self, toy_instance):
        """variance=0 + epsilon=0 + feasible -> wdro_penalty = 0."""
        routes = [Route([0, 1])]
        cfg = ScenarioConfig(distribution="constant", cv=0.0)
        scenarios = generate_scenarios(toy_instance, 50, cfg)
        caches = [RouteCache(r, scenarios, toy_instance.n) for r in routes]
        bd = solution_breakdown(routes, caches, toy_instance,
                                alpha=0.9, epsilon=0.0, penalty_lambda=1.0)
        # Route [0, 1]: peak <= Q likely. Verify Phi == 0.
        assert bd["wdro_penalty"] < 1e-9


class TestRepairWDRO:

    def test_repair_inserts_all_pending(self, toy_instance):
        cfg = ScenarioConfig(distribution="gamma", cv=0.2, seed=42)
        scenarios = generate_scenarios(toy_instance, 200, cfg)
        partial = [Route([0])]
        to_insert = [1, 2, 3, 4]
        rng = np.random.default_rng(0)
        result = repair_greedy_wdro(
            partial, to_insert, toy_instance, scenarios,
            alpha=0.9, epsilon=0.5, penalty_lambda=1.0, rng=rng,
            max_vehicles=toy_instance.n,
        )
        assert result is not None
        sol, _ = result
        served = set()
        for r in sol:
            served.update(r.customers)
        assert served == set(range(toy_instance.n))


class TestALNSCollapse:
    """variance=0 + epsilon=0: W-DRO objective = pure travel cost."""

    def test_collapse_run_produces_zero_phi(self, toy_instance):
        cfg = ScenarioConfig(distribution="constant", cv=0.0)
        scenarios = generate_scenarios(toy_instance, 100, cfg)

        initial = phase0_wdro_seeding(toy_instance, alpha=0.9, epsilon=0.0)
        config = WDROConfig(
            alpha=0.9, epsilon=0.0, penalty_lambda=1.0,
            max_iters=300, seed=42, verbose_every=0,
            max_vehicles=toy_instance.n,
        )
        result = alns_sa_wdro(initial, toy_instance, scenarios, config)
        assert result.best_breakdown["wdro_penalty"] < 1e-6
        # Objective == travel
        assert abs(result.best_objective - result.best_breakdown["travel"]) < 1e-6


class TestALNSReproducibility:

    def test_same_seed_same_result(self, toy_instance):
        cfg = ScenarioConfig(distribution="gamma", cv=0.2, seed=1)
        scenarios = generate_scenarios(toy_instance, 100, cfg)
        initial = phase0_wdro_seeding(toy_instance, alpha=0.9, epsilon=0.5)

        config = WDROConfig(
            alpha=0.9, epsilon=0.5, penalty_lambda=1.0,
            max_iters=200, seed=123, verbose_every=0,
            max_vehicles=toy_instance.n,
        )
        r1 = alns_sa_wdro(initial, toy_instance, scenarios, config)
        r2 = alns_sa_wdro(initial, toy_instance, scenarios, config)
        assert abs(r1.best_objective - r2.best_objective) < 1e-9


class TestALNSEpsilonEffect:
    """Higher epsilon -> objective shifts UP by approximately K * eps/(1-alpha)."""

    def test_objective_shift_with_epsilon(self, toy_instance):
        cfg = ScenarioConfig(distribution="constant", cv=0.0)
        scenarios = generate_scenarios(toy_instance, 100, cfg)
        initial = phase0_wdro_seeding(toy_instance, alpha=0.9, epsilon=0.0)

        config_eps0 = WDROConfig(
            alpha=0.9, epsilon=0.0, penalty_lambda=1.0,
            max_iters=100, seed=42, verbose_every=0,
            max_vehicles=toy_instance.n,
        )
        config_eps1 = WDROConfig(
            alpha=0.9, epsilon=1.0, penalty_lambda=1.0,
            max_iters=100, seed=42, verbose_every=0,
            max_vehicles=toy_instance.n,
        )
        # Phase 0 with eps=1 needs Q_eff. We assume toy can still fit.
        initial1 = phase0_wdro_seeding(toy_instance, alpha=0.9, epsilon=1.0)

        r0 = alns_sa_wdro(initial, toy_instance, scenarios, config_eps0)
        r1 = alns_sa_wdro(initial1, toy_instance, scenarios, config_eps1)

        # eps=1 result should have wdro_penalty ~ K * 1/(1-0.9) = K * 10
        K = len(r1.best_solution)
        expected_phi = K * 1.0 / (1 - 0.9)
        assert abs(r1.best_breakdown["wdro_penalty"] - expected_phi) < 1e-4
        # eps=0 result has zero wdro penalty
        assert r0.best_breakdown["wdro_penalty"] < 1e-6
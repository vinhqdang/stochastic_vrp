"""Tests for Phase 0 W-DRO seeding (M13 verification)."""

import numpy as np
import pytest

from core.instance import Instance
from core.route import Route
from core.scenarios import ScenarioConfig, generate_scenarios
from core.seeding import clarke_wright_svrpspd, phase0_wdro_seeding
from core.wdro_exact import evaluate_phi_exact, empirical_cvar


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


class TestPhase0Feasibility:
    """Phase 0 routes satisfy nominal feasibility with Q_eff."""

    def test_all_routes_within_Qeff(self, toy_instance):
        alpha, epsilon = 0.9, 1.0
        routes = phase0_wdro_seeding(toy_instance, alpha, epsilon)
        Q_eff = toy_instance.Q - epsilon / (1 - alpha)
        nominal = toy_instance.nominal_xi()
        for r in routes:
            peak = r.peak_load(nominal, toy_instance.n)
            assert peak <= Q_eff + 1e-6, (
                f"Route {r.customers}: peak {peak} > Q_eff {Q_eff}"
            )

    def test_all_customers_assigned(self, toy_instance):
        routes = phase0_wdro_seeding(toy_instance, alpha=0.9, epsilon=1.0)
        served = set()
        for r in routes:
            for c in r:
                served.add(c)
        assert served == set(range(toy_instance.n))


class TestPhase0ZeroEpsilon:
    """epsilon=0 should reduce to standard CW (no buffer)."""

    def test_zero_eps_equivalent_to_standard_CW(self, toy_instance):
        routes_wdro = phase0_wdro_seeding(toy_instance, alpha=0.9, epsilon=0.0)
        routes_std  = clarke_wright_svrpspd(toy_instance, capacity_buffer=0.0)
        # Compare route customer sets (order may differ)
        sets_wdro = {tuple(sorted(r.customers)) for r in routes_wdro}
        sets_std  = {tuple(sorted(r.customers)) for r in routes_std}
        assert sets_wdro == sets_std


class TestQeffValidation:

    def test_qeff_nonpositive_raises(self, toy_instance):
        """Q=40, alpha=0.5, eps=25 -> Q_eff = 40 - 50 = -10 < 0."""
        with pytest.raises(ValueError, match="Q_eff"):
            phase0_wdro_seeding(toy_instance, alpha=0.5, epsilon=25.0)

    def test_alpha_outside_range_raises(self, toy_instance):
        with pytest.raises(ValueError):
            phase0_wdro_seeding(toy_instance, alpha=1.0, epsilon=0.1)
        with pytest.raises(ValueError):
            phase0_wdro_seeding(toy_instance, alpha=0.0, epsilon=0.1)

    def test_negative_epsilon_raises(self, toy_instance):
        with pytest.raises(ValueError):
            phase0_wdro_seeding(toy_instance, alpha=0.9, epsilon=-0.1)


class TestM13APrioriBound:
    """Empirical verification of M13 a priori bound.
    
    Theorem M13: if f_{r_0}(xi_bar) <= Q_eff, then
        Phi(r_0) <= CVaR_alpha^{F_0}(max(0, f_{r_0}(xi) - f_{r_0}(xi_bar) - eps/(1-alpha)))
                    + eps/(1-alpha).
    """

    def _verify_bound_for_routes(
        self, routes, inst, scenarios, alpha, epsilon
    ):
        """Verify M13 bound for each route in routes."""
        nominal = inst.nominal_xi()
        buffer = epsilon / (1 - alpha)
        for r in routes:
            # Phase 0 condition: f_{r}(xi_bar) <= Q_eff
            peak_nom = r.peak_load(nominal, inst.n)
            assert peak_nom <= inst.Q - buffer + 1e-6, "Phase 0 condition violated"

            # LHS: Phi(r) per Proposition M4
            phi = evaluate_phi_exact(r, inst, scenarios, alpha, epsilon)

            # RHS: bound from M13
            peaks_scen = r.peak_loads_batch(scenarios, inst.n)
            excess = np.maximum(0.0, peaks_scen - peak_nom - buffer)
            rhs = empirical_cvar(excess, alpha) + buffer

            assert phi <= rhs + 1e-6, (
                f"M13 violated on route {r.customers}: "
                f"Phi={phi:.6f} > RHS={rhs:.6f}"
            )

    def test_M13_bound_holds_gamma(self, toy_instance):
        alpha, epsilon = 0.9, 0.5
        routes = phase0_wdro_seeding(toy_instance, alpha, epsilon)
        cfg = ScenarioConfig(distribution="gamma", cv=0.2, seed=42)
        scenarios = generate_scenarios(toy_instance, 1000, cfg)
        self._verify_bound_for_routes(routes, toy_instance, scenarios, alpha, epsilon)

    def test_M13_bound_holds_lognormal(self, toy_instance):
        alpha, epsilon = 0.95, 0.3
        routes = phase0_wdro_seeding(toy_instance, alpha, epsilon)
        cfg = ScenarioConfig(distribution="lognormal", cv=0.25, seed=7)
        scenarios = generate_scenarios(toy_instance, 1000, cfg)
        self._verify_bound_for_routes(routes, toy_instance, scenarios, alpha, epsilon)

    def test_M13_tight_when_variance_zero(self, toy_instance):
        """With variance=0, excess = 0 always, so Phi(r_0) = buffer exactly.
        (Or Phi = 0 if peak_nom + buffer >= Q.)"""
        alpha, epsilon = 0.9, 0.5
        routes = phase0_wdro_seeding(toy_instance, alpha, epsilon)
        cfg = ScenarioConfig(distribution="constant", cv=0.0)
        scenarios = generate_scenarios(toy_instance, 100, cfg)
        buffer = epsilon / (1 - alpha)

        for r in routes:
            phi = evaluate_phi_exact(r, toy_instance, scenarios, alpha, epsilon)
            peak_nom = r.peak_load(toy_instance.nominal_xi(), toy_instance.n)
            # If feasible under nominal Q, Phi = 0 + buffer (only the regularizer)
            assert phi == pytest.approx(buffer, abs=1e-6), (
                f"Route {r.customers}: expected Phi = buffer = {buffer}, got {phi}"
            )


class TestPhase0Reproducibility:

    def test_same_inputs_same_routes(self, toy_instance):
        """Phase 0 is deterministic (no randomness)."""
        r1 = phase0_wdro_seeding(toy_instance, alpha=0.9, epsilon=0.5)
        r2 = phase0_wdro_seeding(toy_instance, alpha=0.9, epsilon=0.5)
        for a, b in zip(r1, r2):
            assert a.customers == b.customers
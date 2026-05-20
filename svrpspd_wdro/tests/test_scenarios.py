"""Tests for scenario generators: moment matching, variance=0, truncation."""

import numpy as np
import pytest

from core.instance import Instance
from core.scenarios import (
    ScenarioConfig,
    generate_scenarios,
    empirical_moments,
)


@pytest.fixture
def toy_instance(tmp_path):
    """Create a small toy instance for scenario testing."""
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


class TestConstantDistribution:
    """variance = 0 should give N identical rows = nominal xi."""

    def test_constant_distribution_exact(self, toy_instance):
        cfg = ScenarioConfig(distribution="constant", cv=0.0)
        X = generate_scenarios(toy_instance, n_scenarios=100, config=cfg)
        nominal = toy_instance.nominal_xi()
        for s in range(100):
            np.testing.assert_array_equal(X[s], nominal)

    def test_cv_zero_via_gamma(self, toy_instance):
        """gamma + cv=0 should collapse to nominal."""
        cfg = ScenarioConfig(distribution="gamma", cv=0.0)
        X = generate_scenarios(toy_instance, n_scenarios=100, config=cfg)
        nominal = toy_instance.nominal_xi()
        np.testing.assert_array_almost_equal(X[0], nominal)
        np.testing.assert_array_almost_equal(X[-1], nominal)

    def test_bimodal_zero_spread_and_inner_cv_collapses(self, toy_instance):
        """bimodal with both spread=0 and bimodal_cv=0 collapses to nominal."""
        cfg = ScenarioConfig(
            distribution="bimodal",
            cv=0.0,
            seed=1,
            bimodal_spread=0.0,
            bimodal_cv=0.0,
        )
        X = generate_scenarios(toy_instance, n_scenarios=100, config=cfg)
        nominal = toy_instance.nominal_xi()
        np.testing.assert_array_almost_equal(X[0], nominal)
        np.testing.assert_array_almost_equal(X[-1], nominal)


class TestGammaMoments:
    """Empirical mean and CV match target within statistical tolerance."""

    def test_gamma_mean_matches_nominal(self, toy_instance):
        cfg = ScenarioConfig(distribution="gamma", cv=0.2, seed=42)
        N = 10000
        X = generate_scenarios(toy_instance, N, cfg)
        nominal = toy_instance.nominal_xi()
        means = X.mean(axis=0)
        sem = nominal * cfg.cv / np.sqrt(N)
        for k in range(len(nominal)):
            if nominal[k] > 0:
                assert abs(means[k] - nominal[k]) <= 3 * sem[k], (
                    f"Column {k}: mean {means[k]:.3f} vs target {nominal[k]:.3f}"
                )

    def test_gamma_cv_matches_target(self, toy_instance):
        cfg = ScenarioConfig(distribution="gamma", cv=0.25, seed=42)
        N = 10000
        X = generate_scenarios(toy_instance, N, cfg)
        nominal = toy_instance.nominal_xi()
        for k in range(len(nominal)):
            if nominal[k] > 0:
                emp_cv = X[:, k].std(ddof=1) / X[:, k].mean()
                assert abs(emp_cv - cfg.cv) / cfg.cv < 0.15, (
                    f"Column {k}: CV {emp_cv:.3f} vs target {cfg.cv}"
                )


class TestLogNormalMoments:
    def test_lognormal_mean(self, toy_instance):
        cfg = ScenarioConfig(distribution="lognormal", cv=0.3, seed=7)
        N = 10000
        X = generate_scenarios(toy_instance, N, cfg)
        nominal = toy_instance.nominal_xi()
        means = X.mean(axis=0)
        sem = nominal * cfg.cv / np.sqrt(N)
        for k in range(len(nominal)):
            if nominal[k] > 0:
                assert abs(means[k] - nominal[k]) <= 5 * sem[k]


class TestBimodal:
    def test_bimodal_mean_centered(self, toy_instance):
        cfg = ScenarioConfig(
            distribution="bimodal", cv=0.0, seed=1,
            bimodal_spread=0.5, bimodal_cv=0.1,
        )
        N = 20000
        X = generate_scenarios(toy_instance, N, cfg)
        nominal = toy_instance.nominal_xi()
        means = X.mean(axis=0)
        for k in range(len(nominal)):
            if nominal[k] > 0:
                assert abs(means[k] - nominal[k]) / nominal[k] < 0.05

    def test_bimodal_two_modes_visible(self, toy_instance):
        """For a single column, histogram should show two peaks."""
        cfg = ScenarioConfig(
            distribution="bimodal", cv=0.0, seed=2,
            bimodal_spread=0.5, bimodal_cv=0.05,
        )
        X = generate_scenarios(toy_instance, 10000, cfg)
        col = X[:, 2]  # d_1 column (customer 1, delivery)
        nominal_val = toy_instance.nominal_xi()[2]
        low = col[col < nominal_val]
        high = col[col >= nominal_val]
        assert len(low) > 0.3 * len(col)
        assert len(high) > 0.3 * len(col)


class TestTruncation:
    def test_clipping_enforced(self, toy_instance):
        """All scenario values should be in [0, clip_at]."""
        clip = 30.0
        cfg = ScenarioConfig(distribution="gamma", cv=0.5, seed=3, clip_at=clip)
        X = generate_scenarios(toy_instance, 5000, cfg)
        assert (X >= 0).all()
        assert (X <= clip + 1e-9).all()

    def test_default_clip_is_capacity(self, toy_instance):
        cfg = ScenarioConfig(distribution="gamma", cv=0.5, seed=4)
        X = generate_scenarios(toy_instance, 5000, cfg)
        assert (X <= toy_instance.Q + 1e-9).all()


class TestReproducibility:
    def test_same_seed_same_output(self, toy_instance):
        cfg = ScenarioConfig(distribution="gamma", cv=0.3, seed=99)
        X1 = generate_scenarios(toy_instance, 1000, cfg)
        X2 = generate_scenarios(toy_instance, 1000, cfg)
        np.testing.assert_array_equal(X1, X2)

    def test_different_seed_different_output(self, toy_instance):
        cfg1 = ScenarioConfig(distribution="gamma", cv=0.3, seed=1)
        cfg2 = ScenarioConfig(distribution="gamma", cv=0.3, seed=2)
        X1 = generate_scenarios(toy_instance, 1000, cfg1)
        X2 = generate_scenarios(toy_instance, 1000, cfg2)
        assert not np.array_equal(X1, X2)


class TestShape:
    def test_output_shape(self, toy_instance):
        cfg = ScenarioConfig(distribution="gamma", cv=0.2)
        X = generate_scenarios(toy_instance, 500, cfg)
        assert X.shape == (500, 2 * toy_instance.n)
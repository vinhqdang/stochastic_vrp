"""Tests for Phase 1 cache: M10 correctness."""

import numpy as np
import pytest

from core.instance import Instance
from core.route import Route
from core.scenarios import ScenarioConfig, generate_scenarios
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


@pytest.fixture
def toy_scenarios(toy_instance):
    cfg = ScenarioConfig(distribution="gamma", cv=0.2, seed=42)
    return generate_scenarios(toy_instance, 500, cfg)


class TestCacheShapes:

    def test_cache_shape_matches_route(self, toy_instance, toy_scenarios):
        route = Route([0, 1, 2])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        N, m = toy_scenarios.shape[0], len(route)
        assert cache.L.shape == (N, m + 1)
        assert cache.Omega.shape == (N, m + 1)
        assert cache.Psi.shape == (N, m + 1)

    def test_cache_empty_route(self, toy_instance, toy_scenarios):
        cache = RouteCache(Route([]), toy_scenarios, toy_instance.n)
        assert cache.L.shape == (toy_scenarios.shape[0], 1)
        assert (cache.L == 0).all()


class TestCacheLoads:
    """Cache loads must match Route.loads_at_stages_batch identically."""

    def test_L_matches_direct_computation(self, toy_instance, toy_scenarios):
        route = Route([0, 1, 2, 3])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        direct = route.loads_at_stages_batch(toy_scenarios, toy_instance.n)
        np.testing.assert_array_almost_equal(cache.L, direct)


class TestPrefixPeakOmega:
    """Omega = forward cumulative max."""

    def test_omega_monotone_nondecreasing(self, toy_instance, toy_scenarios):
        route = Route([0, 1, 2, 3, 4])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        diffs = np.diff(cache.Omega, axis=1)
        assert (diffs >= -1e-9).all()

    def test_omega_first_equals_L_first(self, toy_instance, toy_scenarios):
        route = Route([0, 1, 2])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        np.testing.assert_array_equal(cache.Omega[:, 0], cache.L[:, 0])

    def test_omega_last_equals_peak(self, toy_instance, toy_scenarios):
        route = Route([0, 1, 2, 3])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        peaks = route.peak_loads_batch(toy_scenarios, toy_instance.n)
        np.testing.assert_array_almost_equal(cache.Omega[:, -1], peaks)

    def test_omega_definition_manual(self, toy_instance, toy_scenarios):
        """Omega[s, k] = max_{k' <= k} L[s, k'] from definition."""
        route = Route([0, 1, 2])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        for k in range(len(route) + 1):
            manual = cache.L[:, :k + 1].max(axis=1)
            np.testing.assert_array_almost_equal(cache.Omega[:, k], manual)


class TestSuffixPeakPsi:
    """Psi = backward cumulative max."""

    def test_psi_monotone_nonincreasing(self, toy_instance, toy_scenarios):
        route = Route([0, 1, 2, 3, 4])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        diffs = np.diff(cache.Psi, axis=1)
        assert (diffs <= 1e-9).all()

    def test_psi_last_equals_L_last(self, toy_instance, toy_scenarios):
        route = Route([0, 1, 2])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        np.testing.assert_array_equal(cache.Psi[:, -1], cache.L[:, -1])

    def test_psi_first_equals_peak(self, toy_instance, toy_scenarios):
        route = Route([0, 1, 2, 3])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        peaks = route.peak_loads_batch(toy_scenarios, toy_instance.n)
        np.testing.assert_array_almost_equal(cache.Psi[:, 0], peaks)

    def test_psi_definition_manual(self, toy_instance, toy_scenarios):
        """Psi[s, k] = max_{k' >= k} L[s, k'] from definition."""
        route = Route([0, 1, 2])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        m = len(route)
        for k in range(m + 1):
            manual = cache.L[:, k:].max(axis=1)
            np.testing.assert_array_almost_equal(cache.Psi[:, k], manual)


class TestCachePeakLoadsProperty:

    def test_peak_loads_matches_direct(self, toy_instance, toy_scenarios):
        route = Route([0, 1, 2])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        direct = route.peak_loads_batch(toy_scenarios, toy_instance.n)
        np.testing.assert_array_almost_equal(cache.peak_loads, direct)


class TestCacheRebuild:

    def test_rebuild_idempotent(self, toy_instance, toy_scenarios):
        """Rebuilding without changing route gives identical tables."""
        route = Route([0, 1, 2])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        L_before = cache.L.copy()
        cache.rebuild()
        np.testing.assert_array_equal(cache.L, L_before)

    def test_rebuild_after_route_mutation(self, toy_instance, toy_scenarios):
        """Mutate route, rebuild, check new cache matches fresh build."""
        route = Route([0, 1, 2])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        # Mutate
        route.insert(3, pos=2)  # now route = [0, 3, 1, 2]
        cache.rebuild()
        # Verify
        fresh = RouteCache(route, toy_scenarios, toy_instance.n)
        np.testing.assert_array_almost_equal(cache.L, fresh.L)
        np.testing.assert_array_almost_equal(cache.Omega, fresh.Omega)
        np.testing.assert_array_almost_equal(cache.Psi, fresh.Psi)
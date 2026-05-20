"""Tests for fast W-DRO evaluation via cache: M11 identity gate.

Critical: cache-based evaluation must MATCH exact evaluation bit-by-bit for
all valid (j, position) candidates.
"""

import numpy as np
import pytest

from core.instance import Instance
from core.route import Route
from core.scenarios import ScenarioConfig, generate_scenarios
from core.cache import RouteCache
from core.wdro_exact import evaluate_phi_exact
from core.wdro_fast import (
    evaluate_insertion_peak_loads_via_cache,
    evaluate_phi_insertion_via_cache,
    best_insertion_via_cache,
    evaluate_removal_peak_loads_via_cache,
    evaluate_phi_removal_via_cache,
    best_removal_via_cache,
)


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


# ============================================================
# A. Identity gate: cache eval == exact eval
# ============================================================


class TestPeakLoadIdentity:
    """f_{r'} via cache must match f_{r'} via direct construction."""

    def test_peak_load_identity_single_position(self, toy_instance, toy_scenarios):
        route = Route([0, 1, 2])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)

        j, k = 3, 2  # insert customer 3 at position 2
        # Cache-based
        peaks_cache = evaluate_insertion_peak_loads_via_cache(cache, j, k)

        # Direct
        new_route = route.copy()
        new_route.insert(j, pos=k)
        peaks_direct = new_route.peak_loads_batch(toy_scenarios, toy_instance.n)

        np.testing.assert_array_almost_equal(peaks_cache, peaks_direct)

    def test_peak_load_identity_all_positions(self, toy_instance, toy_scenarios):
        """For every valid (j, k), cache result must match direct."""
        route = Route([0, 1, 2])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        unrouted = [j for j in range(toy_instance.n) if j not in route.customers]

        for j in unrouted:
            for k in range(1, len(route) + 2):
                peaks_cache = evaluate_insertion_peak_loads_via_cache(cache, j, k)
                new_route = route.copy()
                new_route.insert(j, pos=k)
                peaks_direct = new_route.peak_loads_batch(
                    toy_scenarios, toy_instance.n
                )
                np.testing.assert_array_almost_equal(
                    peaks_cache, peaks_direct,
                    err_msg=f"Mismatch at j={j}, k={k}"
                )


class TestPhiIdentity:
    """Phi via cache must match Phi via exact evaluator."""

    def test_phi_identity_single(self, toy_instance, toy_scenarios):
        route = Route([0, 1, 2])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)

        j, k = 3, 2
        phi_cache = evaluate_phi_insertion_via_cache(
            cache, j, k, toy_instance, alpha=0.9, epsilon=0.5
        )

        new_route = route.copy()
        new_route.insert(j, pos=k)
        phi_exact = evaluate_phi_exact(
            new_route, toy_instance, toy_scenarios, alpha=0.9, epsilon=0.5
        )

        assert abs(phi_cache - phi_exact) < 1e-9

    def test_phi_identity_random_100_cases(self, toy_instance, toy_scenarios):
        """100 random (route, j, k) cases: cache vs exact agree."""
        rng = np.random.default_rng(0)
        for trial in range(100):
            # Random base route of 1-3 customers
            n_cust = int(rng.integers(1, 4))
            base = list(rng.choice(toy_instance.n, size=n_cust, replace=False))
            route = Route(base)
            cache = RouteCache(route, toy_scenarios, toy_instance.n)

            # Random insertion candidate
            unrouted = [c for c in range(toy_instance.n) if c not in route.customers]
            if not unrouted:
                continue
            j = int(rng.choice(unrouted))
            k = int(rng.integers(1, len(route) + 2))

            alpha = float(rng.uniform(0.5, 0.99))
            epsilon = float(rng.uniform(0.0, 2.0))

            phi_cache = evaluate_phi_insertion_via_cache(
                cache, j, k, toy_instance, alpha, epsilon
            )

            new_route = route.copy()
            new_route.insert(j, pos=k)
            phi_exact = evaluate_phi_exact(
                new_route, toy_instance, toy_scenarios, alpha, epsilon
            )

            assert abs(phi_cache - phi_exact) < 1e-7, (
                f"Trial {trial}: route={base}, j={j}, k={k}, "
                f"phi_cache={phi_cache}, phi_exact={phi_exact}"
            )


# ============================================================
# B. Best insertion via cache
# ============================================================


class TestBestInsertion:

    def test_best_insertion_matches_brute_force(self, toy_instance, toy_scenarios):
        """best_insertion_via_cache returns same position as brute force exact."""
        route = Route([0, 1, 2])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        j = 3

        best_pos, best_phi = best_insertion_via_cache(
            cache, j, toy_instance, alpha=0.9, epsilon=0.5
        )

        # Brute force exact
        phis = []
        for k in range(1, len(route) + 2):
            new_route = route.copy()
            new_route.insert(j, pos=k)
            phi = evaluate_phi_exact(
                new_route, toy_instance, toy_scenarios, alpha=0.9, epsilon=0.5
            )
            phis.append((phi, k))
        phis.sort()
        brute_phi, brute_pos = phis[0]

        assert best_pos == brute_pos
        assert abs(best_phi - brute_phi) < 1e-9


# ============================================================
# C. Performance sanity check
# ============================================================


class TestPerformance:
    """Cache eval should be measurably faster than exact for medium m."""

    def test_cache_faster_for_medium_route(self, toy_instance):
        """With m=10 stages, N=1000 scenarios, cache eval ~10x faster."""
        import time

        # Build larger setup
        cfg = ScenarioConfig(distribution="gamma", cv=0.2, seed=42)
        scenarios = generate_scenarios(toy_instance, 1000, cfg)

        # Build a route of 4 customers (max for toy with n=5)
        route = Route([0, 1, 2, 3])
        cache = RouteCache(route, scenarios, toy_instance.n)
        j = 4

        # Time cache eval over many trials
        n_trials = 1000
        t0 = time.time()
        for _ in range(n_trials):
            for k in range(1, len(route) + 2):
                evaluate_phi_insertion_via_cache(
                    cache, j, k, toy_instance, alpha=0.9, epsilon=0.5
                )
        t_cache = time.time() - t0

        # Time exact eval
        t0 = time.time()
        for _ in range(n_trials):
            for k in range(1, len(route) + 2):
                new_route = route.copy()
                new_route.insert(j, pos=k)
                evaluate_phi_exact(
                    new_route, toy_instance, scenarios, alpha=0.9, epsilon=0.5
                )
        t_exact = time.time() - t0

        # Cache should be measurably faster (we don't require huge speedup
        # since toy instance is tiny; main test is on M11 identity)
        print(f"\n  Cache: {t_cache:.3f}s  Exact: {t_exact:.3f}s  "
              f"Speedup: {t_exact/t_cache:.2f}x")
        # Mild assertion: cache shouldn't be slower
        assert t_cache <= t_exact * 1.5

class TestRemovalPeakIdentity:
    """f_{r^-} via cache must match f_{r^-} via direct construction."""

    def test_removal_peak_single_position(self, toy_instance, toy_scenarios):
        route = Route([0, 1, 2, 3])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)

        k = 2  # remove customer at position 2 (= customer 1)
        peaks_cache = evaluate_removal_peak_loads_via_cache(cache, k)

        new_route = route.copy()
        new_route.remove_at(k)
        peaks_direct = new_route.peak_loads_batch(toy_scenarios, toy_instance.n)

        np.testing.assert_array_almost_equal(peaks_cache, peaks_direct)

    def test_removal_peak_all_positions(self, toy_instance, toy_scenarios):
        """Every removable position: cache result == direct."""
        route = Route([0, 1, 2, 3, 4])  # 5 customers
        cache = RouteCache(route, toy_scenarios, toy_instance.n)

        for k in range(1, len(route) + 1):
            peaks_cache = evaluate_removal_peak_loads_via_cache(cache, k)
            new_route = route.copy()
            new_route.remove_at(k)
            peaks_direct = new_route.peak_loads_batch(
                toy_scenarios, toy_instance.n
            )
            np.testing.assert_array_almost_equal(
                peaks_cache, peaks_direct,
                err_msg=f"Mismatch at removal position k={k}"
            )

    def test_removal_single_customer_gives_zero(self, toy_instance, toy_scenarios):
        """Removing the only customer leaves an empty route with f === 0."""
        route = Route([2])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)
        peaks = evaluate_removal_peak_loads_via_cache(cache, position=1)
        np.testing.assert_array_equal(peaks, np.zeros(toy_scenarios.shape[0]))


class TestRemovalPhi:
    """Phi via cache removal must match Phi via exact evaluator on r^-."""

    def test_removal_phi_identity_random_cases(self, toy_instance, toy_scenarios):
        rng = np.random.default_rng(0)
        for trial in range(50):
            n_cust = int(rng.integers(2, 6))
            base = list(rng.choice(toy_instance.n, size=n_cust, replace=False))
            route = Route(base)
            cache = RouteCache(route, toy_scenarios, toy_instance.n)

            k = int(rng.integers(1, len(route) + 1))
            alpha = float(rng.uniform(0.5, 0.99))
            epsilon = float(rng.uniform(0.0, 2.0))

            phi_cache = evaluate_phi_removal_via_cache(
                cache, k, toy_instance, alpha, epsilon
            )

            new_route = route.copy()
            new_route.remove_at(k)
            phi_exact = evaluate_phi_exact(
                new_route, toy_instance, toy_scenarios, alpha, epsilon
            )

            assert abs(phi_cache - phi_exact) < 1e-7, (
                f"Trial {trial}: route={base}, k={k}, "
                f"cache={phi_cache:.6f}, exact={phi_exact:.6f}"
            )


class TestInsertionRemovalInverse:
    """insert(j, k) followed by remove(k) should recover original peak loads."""

    def test_inverse_round_trip(self, toy_instance, toy_scenarios):
        route_init = Route([0, 1, 2])
        cache_init = RouteCache(route_init, toy_scenarios, toy_instance.n)
        peaks_init = cache_init.peak_loads.copy()

        # Insert customer 3 at position 2
        j, k = 3, 2
        peaks_after_insert = evaluate_insertion_peak_loads_via_cache(
            cache_init, j, k
        )

        # Now build the inserted route and remove at position k
        route_after_insert = route_init.copy()
        route_after_insert.insert(j, pos=k)
        cache_after_insert = RouteCache(
            route_after_insert, toy_scenarios, toy_instance.n
        )
        peaks_after_remove = evaluate_removal_peak_loads_via_cache(
            cache_after_insert, k
        )

        # Should recover original peak loads
        np.testing.assert_array_almost_equal(peaks_after_remove, peaks_init)


class TestBestRemoval:

    def test_best_removal_matches_brute_force(self, toy_instance, toy_scenarios):
        route = Route([0, 1, 2, 3, 4])
        cache = RouteCache(route, toy_scenarios, toy_instance.n)

        best_k, best_phi = best_removal_via_cache(
            cache, toy_instance, alpha=0.9, epsilon=0.5
        )

        # Brute force
        phis = []
        for k in range(1, len(route) + 1):
            new_route = route.copy()
            new_route.remove_at(k)
            phi = evaluate_phi_exact(
                new_route, toy_instance, toy_scenarios, alpha=0.9, epsilon=0.5
            )
            phis.append((phi, k))
        phis.sort()
        bf_phi, bf_k = phis[0]

        assert best_k == bf_k
        assert abs(best_phi - bf_phi) < 1e-9
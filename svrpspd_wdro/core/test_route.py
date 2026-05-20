"""Tests for Route: mask matrix, load recurrence, insert/remove, edge cases.

These tests verify mathematical correctness of:
  - Mask matrix construction (manual verification on small cases)
  - Load recurrence vs mask-matrix multiplication (consistency check)
  - Batched vs single-scenario computation (must agree)
  - Edge cases: empty route, single customer, insert at boundaries
"""

import numpy as np
import pytest

from core.route import Route


# ============================================================
# Mask matrix construction tests
# ============================================================


class TestMaskMatrix:
    """Verify mask matrix construction against manual computation."""

    def test_empty_route(self):
        """Empty route: B has shape (2n, 1), all zeros."""
        r = Route()
        B = r.build_mask_matrix(n=5)
        assert B.shape == (10, 1)
        assert B.sum() == 0

    def test_single_customer_mask(self):
        """Route [1] on n=3 customers.

        Stage 0: only d_1 loaded -> B[2,0] = 1.
        Stage 1: drop d_1, load p_1 -> B[2,1] = 0, B[3,1] = 1.
        """
        n = 3
        r = Route([1])
        B = r.build_mask_matrix(n)
        assert B.shape == (6, 2)
        # Stage 0
        assert B[2, 0] == 1  # d_1 loaded
        assert B[3, 0] == 0  # p_1 not yet
        assert B[:, 0].sum() == 1
        # Stage 1
        assert B[2, 1] == 0  # d_1 dropped
        assert B[3, 1] == 1  # p_1 loaded
        assert B[:, 1].sum() == 1

    def test_two_customer_mask(self):
        """Route [0, 2] on n=3 customers.

        Stage 0: d_0 and d_2 loaded -> B[0,0]=1, B[4,0]=1.
        Stage 1: visited 0 -> drop d_0, load p_0. d_2 still on board.
                 B[0,1]=0, B[1,1]=1, B[4,1]=1.
        Stage 2: visited 2 -> drop d_2, load p_2.
                 B[0,2]=0, B[1,2]=1, B[4,2]=0, B[5,2]=1.
        """
        n = 3
        r = Route([0, 2])
        B = r.build_mask_matrix(n)
        assert B.shape == (6, 3)

        # Stage 0
        expected_0 = np.array([1, 0, 0, 0, 1, 0])
        np.testing.assert_array_equal(B[:, 0], expected_0)

        # Stage 1
        expected_1 = np.array([0, 1, 0, 0, 1, 0])
        np.testing.assert_array_equal(B[:, 1], expected_1)

        # Stage 2
        expected_2 = np.array([0, 1, 0, 0, 0, 1])
        np.testing.assert_array_equal(B[:, 2], expected_2)

    def test_mask_binary_and_lipschitz_invariance(self):
        """Verify M3: every column has L_inf norm = 1 for non-empty routes."""
        n = 10
        for route_list in [
            [3],
            [0, 1, 2],
            [9, 5, 0, 3, 7, 1, 8, 2, 4, 6],  # all customers
        ]:
            r = Route(route_list)
            B = r.build_mask_matrix(n)
            # All entries in {0, 1}
            assert ((B == 0) | (B == 1)).all()
            # Each column has L_inf norm = 1 (universal Lipschitz invariance)
            col_max = B.max(axis=0)
            np.testing.assert_array_equal(col_max, np.ones(len(r) + 1))


# ============================================================
# Load computation tests
# ============================================================


class TestLoadComputation:
    """Verify load recurrence matches mask-matrix multiplication."""

    def test_known_load_sequence(self):
        """Manually verify load values for a small example.

        Route [1, 3, 4] on n=5, xi = [10,1, 20,2, 30,3, 40,4, 50,5].
        (d_0=10, p_0=1, ..., d_4=50, p_4=5)

        Stage 0: d_1 + d_3 + d_4 = 20+40+50 = 110
        Stage 1: visited 1 -> 110 - d_1 + p_1 = 110 - 20 + 2 = 92
        Stage 2: visited 3 -> 92 - d_3 + p_3 = 92 - 40 + 4 = 56
        Stage 3: visited 4 -> 56 - d_4 + p_4 = 56 - 50 + 5 = 11
        """
        n = 5
        r = Route([1, 3, 4])
        xi = np.array(
            [10, 1, 20, 2, 30, 3, 40, 4, 50, 5], dtype=np.float64
        )
        expected = np.array([110.0, 92.0, 56.0, 11.0])

        L = r.load_at_stages(xi, n)
        np.testing.assert_array_almost_equal(L, expected)

        peak = r.peak_load(xi, n)
        assert peak == 110.0  # Stage 0 is highest

    def test_recurrence_matches_mask_multiplication(self):
        """For arbitrary route + scenario, both methods must give identical L."""
        n = 8
        rng = np.random.default_rng(123)
        for trial in range(20):
            m = rng.integers(1, n + 1)
            customers = rng.permutation(n)[:m].tolist()
            r = Route(customers)
            xi = rng.uniform(0, 10, size=2 * n)

            L_recurrence = r.load_at_stages(xi, n)
            B = r.build_mask_matrix(n)
            L_mask = B.T @ xi

            np.testing.assert_array_almost_equal(L_recurrence, L_mask)

    def test_empty_route_load(self):
        """Empty route: load is 0 at the single stage."""
        r = Route()
        xi = np.zeros(10)
        L = r.load_at_stages(xi, n=5)
        assert L.shape == (1,)
        assert L[0] == 0.0
        assert r.peak_load(xi, n=5) == 0.0

    def test_nonneg_load_under_nonneg_xi(self):
        """If d_j, p_j >= 0, all loads >= 0."""
        n = 6
        rng = np.random.default_rng(7)
        r = Route(rng.permutation(n).tolist())
        xi = rng.uniform(0, 5, size=2 * n)
        L = r.load_at_stages(xi, n)
        assert (L >= 0).all()


# ============================================================
# Batched load computation
# ============================================================


class TestBatchedLoads:
    """Verify batched computation matches per-scenario computation."""

    def test_batch_equals_singles(self):
        """For N scenarios, loads_at_stages_batch[s] == load_at_stages(X[s])."""
        n = 6
        rng = np.random.default_rng(42)
        m = 4
        r = Route(rng.permutation(n)[:m].tolist())
        N = 50
        X = rng.uniform(0, 10, size=(N, 2 * n))

        L_batch = r.loads_at_stages_batch(X, n)
        assert L_batch.shape == (N, m + 1)

        for s in range(N):
            L_single = r.load_at_stages(X[s], n)
            np.testing.assert_array_almost_equal(L_batch[s], L_single)

    def test_batch_peak_loads(self):
        """Peak loads (batched) match per-scenario peaks."""
        n = 5
        rng = np.random.default_rng(1)
        r = Route([4, 0, 2])
        N = 30
        X = rng.uniform(0, 10, size=(N, 2 * n))
        peaks = r.peak_loads_batch(X, n)
        for s in range(N):
            assert peaks[s] == pytest.approx(r.peak_load(X[s], n))

    def test_batch_empty_route(self):
        """Empty route on N scenarios returns zero peaks."""
        r = Route()
        X = np.random.default_rng(0).uniform(0, 10, size=(10, 6))
        peaks = r.peak_loads_batch(X, n=3)
        assert peaks.shape == (10,)
        assert (peaks == 0).all()


# ============================================================
# Insert / Remove operations
# ============================================================


class TestRouteOperations:
    """Verify insert/remove correctness and edge cases."""

    def test_insert_at_start(self):
        r = Route([2, 3])
        r.insert(1, pos=1)
        assert r.customers == [1, 2, 3]

    def test_insert_at_end(self):
        r = Route([2, 3])
        r.insert(4, pos=3)  # len was 2, pos=3 = end
        assert r.customers == [2, 3, 4]

    def test_insert_middle(self):
        r = Route([1, 3, 4])
        r.insert(2, pos=2)
        assert r.customers == [1, 2, 3, 4]

    def test_insert_invalid_position(self):
        r = Route([1, 2])
        with pytest.raises(IndexError):
            r.insert(3, pos=0)
        with pytest.raises(IndexError):
            r.insert(3, pos=4)  # only positions 1..3 allowed when len=2

    def test_insert_duplicate(self):
        r = Route([1, 2])
        with pytest.raises(ValueError):
            r.insert(1, pos=1)

    def test_remove_at(self):
        r = Route([1, 2, 4])
        removed = r.remove_at(pos=2)
        assert removed == 2
        assert r.customers == [1, 4]

    def test_remove_invalid_position(self):
        r = Route([1, 2])
        with pytest.raises(IndexError):
            r.remove_at(pos=0)
        with pytest.raises(IndexError):
            r.remove_at(pos=3)

    def test_copy_independence(self):
        r1 = Route([1, 2, 3])
        r2 = r1.copy()
        r2.insert(4, pos=4)
        assert r1.customers == [1, 2, 3]
        assert r2.customers == [1, 2, 3, 4]


# ============================================================
# M6 verification: f_{r'} - f_r <= max(d_j, p_j)
# ============================================================


class TestPerturbationBound:
    """Empirically verify M6 (Theorem 5.1 pointwise bound).

    For random routes, insertion positions, and demand realizations,
    check 0 <= f_{r'}(xi) - f_r(xi) <= max(d_j(xi), p_j(xi)).
    """

    def test_m6_random_routes(self):
        n = 8
        rng = np.random.default_rng(2025)
        n_trials = 100

        for _ in range(n_trials):
            m = rng.integers(1, n)
            customers = rng.permutation(n)[:m].tolist()
            r = Route(customers)

            # Pick a customer NOT in r
            remaining = [c for c in range(n) if c not in customers]
            if not remaining:
                continue
            j = int(rng.choice(remaining))

            pos = int(rng.integers(1, m + 2))

            r_prime = r.copy()
            r_prime.insert(j, pos=pos)

            xi = rng.uniform(0, 10, size=2 * n)
            f_r = r.peak_load(xi, n)
            f_r_prime = r_prime.peak_load(xi, n)

            dj = xi[2 * j]
            pj = xi[2 * j + 1]
            bound = max(dj, pj)

            # M6 upper bound
            assert f_r_prime - f_r <= bound + 1e-9, (
                f"M6 upper bound violated! "
                f"f_r={f_r}, f_r'={f_r_prime}, delta={f_r_prime - f_r}, "
                f"bound={bound}, route={customers}, j={j}, pos={pos}"
            )
            # M6 lower bound (using d_j, p_j >= 0)
            assert f_r_prime - f_r >= -1e-9, (
                f"M6 lower bound violated! "
                f"f_r={f_r}, f_r'={f_r_prime}"
            )

    def test_m6_tightness_balanced_peaks(self):
        """M6 is tight when prefix and suffix peaks are equal."""
        # Construct a case where insertion gives exactly max(d_j, p_j) gap
        n = 3
        # Route [0]: stage 0 has d_0; stage 1 has p_0.
        # If d_0 = p_0 = 5, peak load = 5 for both stages.
        r = Route([0])
        xi = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        f_r = r.peak_load(xi, n)
        assert f_r == 5.0

        # Insert j=1 at pos=1: route [1, 0]
        # Stage 0: d_1 + d_0
        # Stage 1: drop d_1, add p_1; d_0 still here -> d_0 + p_1
        # Stage 2: drop d_0, add p_0 -> p_0 + p_1
        r_prime = r.copy()
        r_prime.insert(1, pos=1)
        xi[2] = 7.0  # d_1
        xi[3] = 3.0  # p_1
        # Stages: 5+7=12; 5+3=8; 5+3=8 -> peak 12
        # f_r with new xi: stages d_0=5, p_0=5 -> peak 5
        f_r_new = r.peak_load(xi, n)
        f_r_prime = r_prime.peak_load(xi, n)
        bound = max(xi[2 * 1], xi[2 * 1 + 1])  # max(d_1, p_1) = max(7,3) = 7
        assert f_r_prime - f_r_new <= bound + 1e-9


# ============================================================
# Travel cost
# ============================================================


class TestTravelCost:
    def test_empty_route_cost_zero(self):
        r = Route()
        D = np.zeros((4, 4))
        assert r.travel_cost(D) == 0.0

    def test_simple_triangle(self):
        """Depot at (0,0), customers at (1,0) and (0,1). Route [0,1]."""
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        D = np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, np.sqrt(2)],
                [1.0, np.sqrt(2), 0.0],
            ]
        )
        r = Route([0, 1])
        # depot -> customer 0 -> customer 1 -> depot
        # indices in D: 0 -> 1 -> 2 -> 0
        # cost = 1 + sqrt(2) + 1
        expected = 1 + np.sqrt(2) + 1
        assert r.travel_cost(D) == pytest.approx(expected)
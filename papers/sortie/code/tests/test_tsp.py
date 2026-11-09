import itertools
import math

import numpy as np
import pytest

from sortie.tsp import TourSolver, held_karp, is_tsp_optimal, local_search_tour, tour_stats


def _sym_matrix(rng, n):
    W = rng.uniform(1.0, 100.0, size=(n, n))
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    return W


def _euclid(rng, k, scale=100.0):
    pts = rng.uniform(0.0, scale, size=(k + 1, 2))
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _cycle_cost(order, W):
    seq = [0] + list(order) + [0]
    return sum(W[seq[i], seq[i + 1]] for i in range(len(seq) - 1))


def _brute_min(W):
    k = W.shape[0] - 1
    best = math.inf
    for perm in itertools.permutations(range(1, k + 1)):
        best = min(best, _cycle_cost(perm, W))
    return best


def test_held_karp_matches_brute_force():
    rng = np.random.default_rng(101)
    for case in range(20):
        k = 1 + case % 7
        W = _sym_matrix(rng, k + 1)
        order, cost = held_karp(W)
        assert sorted(order) == list(range(1, k + 1))
        assert cost == pytest.approx(_cycle_cost(order, W), rel=1e-12, abs=1e-9)
        assert cost == pytest.approx(_brute_min(W), rel=1e-12, abs=1e-9)


def test_held_karp_empty_and_single():
    assert held_karp(np.zeros((1, 1))) == ([], 0.0)
    W = np.array([[0.0, 3.0], [4.0, 0.0]])
    order, cost = held_karp(W)
    assert order == [1] and cost == pytest.approx(7.0)


def test_tour_stats_closed_tour():
    C = np.array([[0.0, 3.0, 4.0, 5.0],
                  [3.0, 0.0, 6.0, 7.0],
                  [4.0, 6.0, 0.0, 8.0],
                  [5.0, 7.0, 8.0, 0.0]])
    L, W = tour_stats([1, 2, 3], C)
    assert L == pytest.approx(3 + 6 + 8 + 5)
    assert W == pytest.approx(9 + 36 + 64 + 25)
    assert tour_stats([], C) == (0.0, 0.0)


def test_tour_solver_kappa0_is_tsp_optimal():
    rng = np.random.default_rng(202)
    for case in range(10):
        k = 2 + case % 6
        C = _euclid(rng, k)
        solver = TourSolver(C, kappa=0.0, v=0.3)
        tour, L, W = solver.solve(range(1, k + 1))
        best = min(tour_stats(list(p), C)[0] for p in itertools.permutations(range(1, k + 1)))
        assert L == pytest.approx(best, rel=1e-10, abs=1e-9)
        assert sorted(tour) == list(range(1, k + 1))


def test_tour_solver_kappa_positive_equals_brute_force():
    rng = np.random.default_rng(303)
    kappas = (0.5, 1.0, 2.0)
    v = 0.3
    for case in range(20):
        k = 1 + case % 7
        kappa = kappas[case % 3]
        C = _euclid(rng, k)
        solver = TourSolver(C, kappa=kappa, v=v)
        tour, L, W = solver.solve(range(1, k + 1))
        got = L + kappa * v * math.sqrt(W)
        best = math.inf
        for p in itertools.permutations(range(1, k + 1)):
            Lp, Wp = tour_stats(list(p), C)
            best = min(best, Lp + kappa * v * math.sqrt(Wp))
        assert got == pytest.approx(best, rel=1e-9, abs=1e-9), (case, k, kappa)
        assert sorted(tour) == list(range(1, k + 1))
        assert (L, W) == pytest.approx(tour_stats(tour, C))


def test_solver_small_sets_and_empty():
    rng = np.random.default_rng(404)
    C = _euclid(rng, 4)
    for kappa in (0.0, 1.0):
        solver = TourSolver(C, kappa=kappa, v=0.3)
        assert solver.solve([]) == ([], 0.0, 0.0)
        tour, L, W = solver.solve([2])
        assert tour == [2]
        assert L == pytest.approx(2 * C[0, 2])
        assert W == pytest.approx(2 * C[0, 2] ** 2)
        tour, L, W = solver.solve([3, 1])
        assert sorted(tour) == [1, 3]
        assert L == pytest.approx(C[0, 1] + C[1, 3] + C[3, 0])


def test_solver_memoisation_and_counters():
    rng = np.random.default_rng(505)
    C = _euclid(rng, 6)
    solver = TourSolver(C, kappa=1.0, v=0.3)
    first = solver.solve({1, 2, 3, 4})
    calls_after_first = solver.n_hk
    assert solver.n_calls == 1 and solver.n_cache_hits == 0
    second = solver.solve(frozenset([4, 3, 2, 1]))
    assert second == first
    assert solver.n_calls == 2 and solver.n_cache_hits == 1
    assert solver.n_hk == calls_after_first
    assert solver.n_hk >= 1


def test_solver_large_set_uses_local_search():
    rng = np.random.default_rng(606)
    C = _euclid(rng, 20)
    solver = TourSolver(C, kappa=1.0, v=0.3)
    tour, L, W = solver.solve(range(1, 21))
    assert sorted(tour) == list(range(1, 21))
    assert (L, W) == pytest.approx(tour_stats(tour, C))
    assert solver.n_ls >= 1


def test_is_tsp_optimal():
    rng = np.random.default_rng(707)
    for case in range(5):
        k = 5 + case
        C = _euclid(rng, k)
        nodes = list(range(1, k + 1))
        opt = min(tour_stats(list(p), C)[0] for p in itertools.permutations(nodes))
        ok, l_opt = is_tsp_optimal(nodes, C)
        assert l_opt == pytest.approx(opt, rel=1e-10, abs=1e-9)
        assert ok == (tour_stats(nodes, C)[0] <= opt + 1e-9)
        best_perm = min(itertools.permutations(nodes), key=lambda p: tour_stats(list(p), C)[0])
        assert is_tsp_optimal(list(best_perm), C)[0] is True


def test_local_search_returns_valid_tour_close_to_optimal():
    rng = np.random.default_rng(808)
    for case in range(3):
        k = 13 + case
        C = _euclid(rng, k)
        nodes = list(range(1, k + 1))
        tour, cost = local_search_tour(nodes, C, starts=8, rng=np.random.default_rng(case))
        assert sorted(tour) == nodes
        assert cost == pytest.approx(_cycle_cost(tour, C), rel=1e-12)
        _, opt = held_karp(C)
        assert cost <= 1.05 * opt

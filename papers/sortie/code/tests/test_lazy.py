"""Screened (fast) lower level: warm starts, solve_fast, and the exact-confirmation invariant."""

import numpy as np
import pytest

from sortie.instances import load_x641, make_instance, noise_matrix
from sortie.search import construct_initial, solve_fair
from sortie.tsp import (
    HK_LIMIT,
    LS_MAX_ROUNDS,
    TourSolver,
    _local_search_nb,
    _nn_order_nb,
    cheapest_insertion_hint,
    local_search_tour,
    tour_stats,
)


@pytest.fixture(scope="module")
def inst20():
    coords, demand = load_x641()
    return make_instance(coords, demand, n=20, b=0)


def _same_up_to_reversal(a, b):
    return list(a) == list(b) or list(a) == list(b)[::-1]


def _ref_nn(nodes, W):
    """The original pure-python nearest-neighbour loop."""
    remaining = set(nodes)
    cur = 0
    out = []
    while remaining:
        nxt = min(remaining, key=lambda j: (W[cur, j], j))
        out.append(nxt)
        remaining.discard(nxt)
        cur = nxt
    return out


def _ref_local_search_tour(nodes, W, starts, rng):
    """The original local_search_tour (NN start, then random permutations)."""
    arr = np.array(nodes, dtype=np.int64)
    starts_list = [np.array(_ref_nn(nodes, W), dtype=np.int64)]
    for _ in range(max(0, int(starts) - 1)):
        starts_list.append(rng.permutation(arr))
    best_seq, best_cost = None, np.inf
    for st in starts_list:
        seq = np.concatenate((np.array([0], dtype=np.int64), st))
        seq, cost = _local_search_nb(seq, W, LS_MAX_ROUNDS)
        if cost < best_cost - 1e-12:
            best_cost, best_seq = cost, seq
    return [int(x) for x in best_seq[1:]], float(best_cost)


def _cycle(tour, W):
    seq = [0] + list(tour) + [0]
    return sum(W[seq[i], seq[i + 1]] for i in range(len(seq) - 1))


def _int_sym_matrix(rng, n):
    """Integer weights, so that nearest-neighbour ties actually occur."""
    W = rng.integers(1, 6, size=(n, n)).astype(np.float64)
    W = np.triu(W, 1)
    W = W + W.T
    return W


def test_nn_order_nb_matches_python_loop_with_ties():
    rng = np.random.default_rng(31)
    for case in range(15):
        n = 3 + case
        W = _int_sym_matrix(rng, n + 1)
        nodes = rng.permutation(np.arange(1, n + 1))
        got = _nn_order_nb(nodes.astype(np.int64), W)
        assert got.tolist() == _ref_nn(sorted(nodes.tolist()), W)


def test_local_search_tour_init_none_matches_original_path():
    rng = np.random.default_rng(37)
    for case in range(6):
        k = 6 + 3 * case
        W = _int_sym_matrix(rng, k + 1)
        nodes = list(range(1, k + 1))
        tour, cost = local_search_tour(nodes, W, starts=8, rng=np.random.default_rng(case))
        ref_tour, ref_cost = _ref_local_search_tour(nodes, W, 8, np.random.default_rng(case))
        assert tour == ref_tour
        assert cost == ref_cost


def test_local_search_tour_init_is_first_start_and_validated():
    rng = np.random.default_rng(41)
    k = 18
    W = _int_sym_matrix(rng, k + 1)
    nodes = list(range(1, k + 1))
    init = list(rng.permutation(nodes))
    tour, cost = local_search_tour(nodes, W, starts=1, rng=np.random.default_rng(0), init=init)
    assert sorted(tour) == nodes
    # starts=1 runs init and the NN start; the best of the two is returned,
    # so the result can be no worse than the run from init alone
    _, c_init = _local_search_nb(np.array([0] + [int(x) for x in init], dtype=np.int64), W, LS_MAX_ROUNDS)
    assert cost <= c_init + 1e-9
    assert cost == pytest.approx(_cycle(tour, W))
    with pytest.raises(ValueError):
        local_search_tour(nodes, W, starts=1, init=[1, 2, 3])


def test_cheapest_insertion_hint_is_permutation_of_target():
    rng = np.random.default_rng(43)
    C = np.ascontiguousarray(_int_sym_matrix(rng, 16).astype(float))
    for case in range(10):
        S = sorted(rng.choice(np.arange(1, 16), size=5 + case % 6, replace=False).tolist())
        parent = rng.permutation(np.arange(1, 16)).tolist()
        hint = cheapest_insertion_hint(parent, S, C)
        assert sorted(hint) == S
        assert len(hint) == len(set(hint))


def test_cheapest_insertion_hint_keeps_parent_order_and_inserts_cheaply():
    # depot at the origin, customers on a plane; 3 is missing from the parent sequence
    pts = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 5.0]])
    diff = pts[:, None, :] - pts[None, :, :]
    C = np.sqrt((diff ** 2).sum(axis=2))
    # customers dropped from the parent (3 here) disappear, survivors keep their order
    assert cheapest_insertion_hint([3, 1, 2, 4], {1, 2, 4}, C) == [1, 2, 4]
    # inserting customer 2 into the route [1]: both slots cost 14.14 (a tie), and the
    # first slot (before 1) wins, so the hint is [2, 1]
    assert cheapest_insertion_hint([1], {1, 2}, C) == [2, 1]


def test_solve_fast_is_permutation_with_length_at_least_exact():
    coords, demand = load_x641()
    inst = make_instance(coords, demand, n=20, b=0)
    C = np.ascontiguousarray(inst.C, dtype=np.float64)
    D = noise_matrix(inst)
    rng = np.random.default_rng(47)
    for case in range(12):
        k = 3 + case % 13  # 3..15 customers, all within the Held-Karp range
        assert k <= HK_LIMIT
        S = sorted(rng.choice(np.arange(1, 21), size=k, replace=False).tolist())
        parent = rng.permutation(np.arange(1, 21)).tolist()
        fast = TourSolver(C, 0.0, 0.3, D=D)
        exact = TourSolver(C, 0.0, 0.3, D=D)
        tour_f, L_f, W_f = fast.solve_fast(S, hint=parent)
        tour_e, L_e, _ = exact.solve(S)
        assert sorted(tour_f) == S
        assert L_f == pytest.approx(tour_stats(tour_f, C, D)[0], rel=1e-12)
        assert L_f >= L_e - 1e-9
        assert fast.n_fast == 1
        # memoised: a second call is a fast hit and returns the same object
        assert fast.solve_fast(S) is fast.solve_fast(S, hint=None)
        assert fast.n_fast == 1 and fast.n_fast_hits >= 1


def test_solve_fast_reads_exact_cache_and_kappa_positive_is_permutation():
    coords, demand = load_x641()
    inst = make_instance(coords, demand, n=20, b=0)
    C = np.ascontiguousarray(inst.C, dtype=np.float64)
    D = noise_matrix(inst)
    S = list(range(1, 13))
    solver = TourSolver(C, 0.0, 0.3, D=D)
    exact = solver.solve(S)
    assert solver.solve_fast(S, hint=list(reversed(S))) is exact
    assert solver.n_fast == 0
    big = list(range(1, 21))
    solver1 = TourSolver(C, 1.0, 0.3, D=D)
    tour, L, W = solver1.solve_fast(big, hint=list(range(20, 0, -1)))
    assert sorted(tour) == big
    assert L == pytest.approx(tour_stats(tour, C, D)[0], rel=1e-12)
    assert W == pytest.approx(tour_stats(tour, C, D)[1], rel=1e-12)


def test_exact_solve_with_hint_is_permutation_for_large_sets():
    coords, demand = load_x641()
    inst = make_instance(coords, demand, n=20, b=0)
    C = np.ascontiguousarray(inst.C, dtype=np.float64)
    solver = TourSolver(C, 0.0, 0.3)
    S = list(range(1, 21))
    tour, L, _ = solver.solve(S, hint=list(range(20, 0, -1)))
    assert sorted(tour) == S
    assert L == pytest.approx(tour_stats(tour, C)[0], rel=1e-12)


def _sumL(routes, C):
    return sum(tour_stats(r, C)[0] for r in routes)


@pytest.mark.parametrize("kappa", [0.0, 1.0])
def test_solve_fair_plans_are_exact_lower_level(inst20, kappa):
    inst = inst20
    C = inst.C
    D = noise_matrix(inst)
    start = construct_initial(inst, np.random.default_rng(0))
    solver = TourSolver(C, kappa, 0.3, D=D)
    budget = 1.10 * sum(TourSolver(C, 0.0, 0.3).solve(r)[1] for r in start)
    res = solve_fair(inst, start, "pboi", "range", 0.3, kappa, budget,
                     time_limit=60.0, seed=5, max_iters=30, solver=solver)
    routes = res["routes"]
    assert sorted(x for r in routes for x in r) == list(range(1, C.shape[0]))
    for r in routes:
        tour = solver.solve(r)[0]
        assert _same_up_to_reversal(r, tour)
        if len(r) <= HK_LIMIT:
            fresh = TourSolver(C, kappa, 0.3).solve(r)[0]
            assert _same_up_to_reversal(r, fresh)
    assert res["sumL"] <= budget + 1e-6
    assert res["sumL"] == pytest.approx(_sumL(routes, C), abs=1e-6)
    # every accepted improvement came out of a confirmation with the exact lower level
    assert res["n_confirm"] >= res["n_improve"]
    assert "n_fast" in res["solver"] and "n_fast_hits" in res["solver"]

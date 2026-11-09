import itertools
import math

import numpy as np
import pytest

from sortie import search
from sortie.instances import Instance, load_x641, make_instance
from sortie.search import (
    _Ctx,
    _build_plan,
    _route_pre,
    construct_initial,
    make_objective,
    solve_cost,
    solve_fair,
)
from sortie.tsp import TourSolver, is_tsp_optimal, tour_stats


@pytest.fixture(scope="module")
def inst15():
    coords, demand = load_x641()
    return make_instance(coords, demand, n=15, b=0, m=5)


@pytest.fixture(scope="module")
def cost_plan(inst15):
    routes, cstar = solve_cost(inst15, time_limit=1.0, seed=1, restarts=1)
    return routes, cstar


def _check_partition(routes, inst):
    assert len(routes) == inst.m
    assert all(len(r) > 0 for r in routes)
    cover = sorted(x for r in routes for x in r)
    assert cover == list(range(1, inst.C.shape[0]))
    for r in routes:
        assert int(inst.demand[np.asarray(r)].sum()) <= inst.Q


def _sumL(routes, C):
    return sum(tour_stats(r, C)[0] for r in routes)


def test_solve_cost_partition_and_tsp(inst15, cost_plan):
    routes, cstar = cost_plan
    _check_partition(routes, inst15)
    for r in routes:
        assert is_tsp_optimal(r, inst15.C)[0]
    assert cstar == pytest.approx(_sumL(routes, inst15.C), abs=1e-9)


def test_solve_fair_det_range_not_worse_and_budget(inst15, cost_plan):
    routes, cstar = cost_plan
    budget = 1.10 * cstar
    start_val = make_objective("det", "range", 0.3).value(
        np.array([tour_stats(r, inst15.C)[0] for r in routes]),
        np.array([tour_stats(r, inst15.C)[1] for r in routes]),
        routes,
    )
    res = solve_fair(inst15, routes, "det", "range", 0.3, 0.0, budget,
                     time_limit=10.0, seed=3, max_iters=120)
    _check_partition(res["routes"], inst15)
    assert res["value"] <= start_val + 1e-9
    assert res["sumL"] <= budget + 1e-9
    assert res["sumL"] == pytest.approx(_sumL(res["routes"], inst15.C), abs=1e-6)


def test_solve_fair_pboi_budget_cstar(inst15, cost_plan):
    routes, cstar = cost_plan
    res = solve_fair(inst15, routes, "pboi", "range", 0.3, 0.0, cstar,
                     time_limit=10.0, seed=4, max_iters=80)
    _check_partition(res["routes"], inst15)
    assert res["sumL"] <= cstar + 1e-6
    assert _sumL(res["routes"], inst15.C) <= cstar + 1e-6


def test_free_mode_non_tsp_lower_tsp():
    C = np.zeros((5, 5))

    def setc(i, j, x):
        C[i, j] = C[j, i] = x

    setc(0, 1, 1)
    setc(1, 2, 1)
    setc(2, 3, 1)
    setc(3, 0, 1)
    setc(0, 2, 2)
    setc(1, 3, 2)
    for k in range(4):
        setc(k, 4, 5)
    inst = Instance(name="tiny", coords=np.zeros((5, 2)), demand=np.array([0, 1, 1, 1, 1]),
                    m=2, Q=10, C=C)
    start = [[1, 3, 2], [4]]
    free = solve_fair(inst, start, "det", "range", 0.3, 0.0, np.inf,
                      time_limit=5.0, seed=0, consistency="free", max_iters=0)
    lower = solve_fair(inst, start, "det", "range", 0.3, 0.0, np.inf,
                       time_limit=5.0, seed=0, consistency="lower", max_iters=0)
    assert free["routes"][0] == [1, 3, 2]
    assert not is_tsp_optimal(free["routes"][0], C)[0]
    assert free["value"] < lower["value"]
    for r in lower["routes"]:
        assert is_tsp_optimal(r, C)[0]


def test_determinism_same_seed_same_result(inst15, cost_plan):
    routes, cstar = cost_plan
    runs = [solve_fair(inst15, routes, "pboi", "gmd", 0.3, 0.0, 1.10 * cstar,
                       time_limit=120.0, seed=11, max_iters=40) for _ in range(2)]
    assert runs[0]["routes"] == runs[1]["routes"]
    assert runs[0]["value"] == runs[1]["value"]
    assert runs[0]["iters"] == runs[1]["iters"] == 40


def test_construct_initial_valid(inst15):
    rng = np.random.default_rng(5)
    routes = construct_initial(inst15, rng)
    _check_partition(routes, inst15)


def test_route_pre_matches_brute_force(inst15):
    C = inst15.C
    ctx = _Ctx(inst15, TourSolver(C, 0.0, 0.3), "lower", 0.3,
               np.random.default_rng(0), 8, "price", np.inf)
    routes = construct_initial(inst15, np.random.default_rng(7))
    routes = [list(TourSolver(C, 0.0, 0.3).solve(r)[0]) for r in routes]
    for a in range(len(routes)):
        seq = routes[a]
        pre = _route_pre(seq, ctx)
        La = tour_stats(seq, C)[0]
        for p, i in enumerate(seq):
            red = [x for x in seq if x != i]
            assert pre.Lrem[p] == pytest.approx(La - tour_stats(red, C)[0], abs=1e-9)
            for j in range(1, C.shape[0]):
                if j in seq:
                    continue
                best = min(tour_stats(red[:k] + [j] + red[k:], C)[0] for k in range(len(red) + 1))
                best_red = tour_stats(red, C)[0]
                assert pre.Lsw[p, j] == pytest.approx(best - best_red, abs=1e-9)
        for j in range(1, C.shape[0]):
            if j in seq:
                continue
            best = min(tour_stats(seq[:k] + [j] + seq[k:], C)[0] for k in range(len(seq) + 1))
            assert pre.Lbest[j] == pytest.approx(best - La, abs=1e-9)


def test_det_prices_match_finite_difference():
    L = np.array([10.0, 5.0, 7.0, 12.0, 9.0])
    W = np.ones(5)
    ad = make_objective("det", "range", 0.3)
    varpi, psi = ad.prices(L, W, [[1]] * 5)
    assert np.allclose(psi, 0.0)
    assert varpi.sum() == pytest.approx(0.0, abs=1e-12)
    h = 1e-3
    for k in range(5):
        Lp = L.copy()
        Lp[k] += h
        fd = (ad.value(Lp, W, None) - ad.value(L, W, None)) / h
        assert varpi[k] == pytest.approx(fd, abs=1e-6)


def test_saa_v0_equals_det_value(inst15):
    C = inst15.C
    routes = [list(TourSolver(C, 0.0, 0.0).solve(r)[0])
              for r in construct_initial(inst15, np.random.default_rng(9))]
    L = np.array([tour_stats(r, C)[0] for r in routes])
    W = np.array([tour_stats(r, C)[1] for r in routes])
    saa = make_objective("saa", "range", 0.0, C=C)
    det = make_objective("det", "range", 0.0)
    assert saa.value(L, W, routes) == pytest.approx(det.value(L, W, routes), rel=1e-12)


def test_random_screen_and_free_run_keep_feasibility(inst15, cost_plan):
    routes, cstar = cost_plan
    res = solve_fair(inst15, routes, "pboi", "gmd", 0.3, 0.0, 1.10 * cstar,
                     time_limit=10.0, seed=6, screen="random", consistency="free", max_iters=25)
    _check_partition(res["routes"], inst15)
    assert res["sumL"] <= 1.10 * cstar + 1e-9
    assert res["iters"] == 25

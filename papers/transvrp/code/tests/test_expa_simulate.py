"""M6: neural-style simulator (EXPERIMENT_A.md section 9, M6)."""
import functools

import numpy as np

from transvrp.baton.costs import (_simulate_costs_general, fit_lsm_general,
                                  oracle_costs_general, simulate_v2_general)
from transvrp.data import route_problem
from transvrp.expa.features import RouteCache, RouteStats
from transvrp.expa.paths import Paths, RouteSpec, route_specs
from transvrp.expa.simulate import baton_models, simulate_policy, summarize


@functools.lru_cache(maxsize=None)
def _specs():
    return {s.route_id: s for s in route_specs("CON3-0")}


@functools.lru_cache(maxsize=None)
def _con3_problems():
    return list(route_problem("CON3-0", n_train=500, n_test=2000))


def _plain_cache(spec: RouteSpec, g: np.ndarray) -> RouteCache:
    """Cache whose ``g`` is exactly ``g`` (d = 0). Statistics are unused by the
    BATON-style predictors."""
    zeros = np.zeros(spec.m)
    paths = Paths(p=g.copy(), d=np.zeros_like(g))
    return RouteCache(spec, paths, RouteStats(zeros, zeros, zeros, zeros))


def test_parity_with_baton_simulator_on_con3():
    specs = _specs()
    for p in _con3_problems():
        spec = specs[p["route_id"]]
        B, H, E = p["B"], p["H"], p["E"]
        g_tr, g_te = p["g_train"], p["g_test"]
        cache = _plain_cache(spec, g_te)

        cont = fit_lsm_general(g_tr, B, H, E)
        ref_c, ref_a = _simulate_costs_general(g_te, B, H, E, cont)
        got_c, got_a = simulate_policy(cache, B, H, E, baton_models(cont))
        assert np.array_equal(got_c, ref_c)
        assert np.array_equal(got_a, ref_a)
        ref_stats = simulate_v2_general(g_te, B, H, E, cont)
        assert summarize(got_c, got_a)["mean_cost"] == ref_stats["mean_cost"]

        # 'none': never hand off (decide with H = 1e12), bill H
        big = np.full_like(H, 1e12)
        cont_none = fit_lsm_general(g_tr, B, big, E)
        ref_c, ref_a = _simulate_costs_general(g_te, B, big, E, cont_none, H_bill=H)
        got_c, got_a = simulate_policy(cache, B, big, E, baton_models(cont_none),
                                       H_bill=H)
        assert np.array_equal(got_c, ref_c)
        assert np.array_equal(got_a, ref_a)


def test_mean_cost_is_at_least_oracle():
    specs = _specs()
    for p in _con3_problems():
        spec = specs[p["route_id"]]
        B, H, E = p["B"], p["H"], p["E"]
        g_te = p["g_test"]
        cache = _plain_cache(spec, g_te)
        cont = fit_lsm_general(p["g_train"], B, H, E)
        costs, _ = simulate_policy(cache, B, H, E, baton_models(cont))
        oracle = oracle_costs_general(g_te, B, H, E)
        assert costs.mean() >= oracle.mean()
        assert np.all(costs >= oracle - 1e-12)     # per path, not only on average


def test_toy_route_control_flow():
    """m = 3, B = 1. Row 0 breaches at stop 2; row 1 never breaches.
    A constant continuation value 10 > H[0] = 5 hands off at stop 1 for
    both rows, so row 0 is billed H_bill[0] and row 1 too (no emergency)."""
    spec = RouteSpec(inst="toy", route_id=0, route=(0, 1, 2), m=3, B=1.0,
                     H=np.array([5.0, 6.0, 7.0]), E=np.array([9.0, 9.0, 9.0]),
                     E0=9.0)
    g = np.array([[0.5, 1.0, 0.0],
                  [0.0, 0.0, 0.0]])
    cache = _plain_cache(spec, g)

    class Const:
        def predict(self, cache, k, idx):
            return np.full(np.asarray(idx).shape[0], 10.0)

    costs, act = simulate_policy(cache, 1.0, spec.H, spec.E, {1: [Const()]},
                                 H_bill=np.array([4.0, 6.0, 7.0]))
    assert np.array_equal(act, np.array([1, 1], dtype=np.int8))
    assert np.array_equal(costs, np.array([4.0, 4.0]))

    # without a decision at k = 1 the first row fails at stop 2
    costs, act = simulate_policy(cache, 1.0, spec.H, spec.E, {})
    assert np.array_equal(act, np.array([2, 0], dtype=np.int8))
    assert np.array_equal(costs, np.array([9.0, 0.0]))


def test_mean_of_predictor_list_is_used():
    spec = RouteSpec(inst="toy", route_id=0, route=(0, 1, 2), m=3, B=1.0,
                     H=np.array([5.0, 6.0, 7.0]), E=np.array([9.0, 9.0, 9.0]),
                     E0=9.0)
    g = np.zeros((2, 3))
    cache = _plain_cache(spec, g)

    class Const:
        def __init__(self, v):
            self.v = v

        def predict(self, cache, k, idx):
            return np.full(np.asarray(idx).shape[0], self.v)

    # mean(4, 8) = 6 > 5 hands off; mean(4, 4) = 4 does not
    costs, act = simulate_policy(cache, 1.0, spec.H, spec.E,
                                 {1: [Const(4.0), Const(8.0)]})
    assert np.all(act == 1)
    costs, act = simulate_policy(cache, 1.0, spec.H, spec.E,
                                 {1: [Const(4.0), Const(4.0)]})
    assert np.all(act == 0)

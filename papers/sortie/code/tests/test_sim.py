import numpy as np

from sortie.instances import load_x641, make_instance
from sortie.sim import evaluate_plans, simulate
from sortie.tsp import tour_stats

import os

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "X-n641-k35.vrp")


def _instance():
    coords, demand = load_x641(DATA)
    return make_instance(coords, demand, n=15, b=0, m=5)


def _plan_a():
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11], [12, 13, 14, 15]]


def _plan_b():
    return [[1, 2, 3], [9, 8], [4, 6, 5], [15, 14], [7, 10, 11, 12, 13]]


def test_crn_shared_route_gives_identical_columns():
    inst = _instance()
    XA, YA = simulate(inst, _plan_a(), v=0.3, sY=0.2, S=3000, seed=7)
    XB, YB = simulate(inst, _plan_b(), v=0.3, sY=0.2, S=3000, seed=7)
    assert np.array_equal(XA[:, 0], XB[:, 0])
    assert np.array_equal(YA, YB)
    assert not np.array_equal(XA[:, 1], XB[:, 1])


def test_crn_across_evaluate_plans_shared_route():
    inst = _instance()
    res = evaluate_plans(inst, [_plan_a(), _plan_b()], v=0.3, sY=0.2, S=3000, seed=7)
    assert res[0]["mean_driver_X"][0] == res[1]["mean_driver_X"][0]
    assert res[0]["sd_driver_X"][0] == res[1]["sd_driver_X"][0]
    assert res[0]["mean_driver"][0] == res[1]["mean_driver"][0]


def test_v0_sY0_gives_deterministic_lengths():
    inst = _instance()
    plan = _plan_a()
    X, Y = simulate(inst, plan, v=0.0, sY=0.0, S=50, seed=3)
    assert np.all(Y == 1.0)
    L = np.array([tour_stats(r, inst.C)[0] for r in plan])
    for s in range(X.shape[0]):
        assert np.array_equal(X[s], L)


def test_arc_moments_match_model():
    inst = _instance()
    plan = _plan_a()
    v = 0.3
    S = 20000
    X, Y = simulate(inst, plan, v=v, sY=0.0, S=S, seed=11)
    L = np.array([tour_stats(r, inst.C)[0] for r in plan])
    W = np.array([tour_stats(r, inst.C)[1] for r in plan])
    assert np.allclose(X.mean(axis=0), L, rtol=0.01)
    assert np.allclose(X.var(axis=0), v * v * W, rtol=0.05)


def test_day_factor_has_unit_mean_and_requested_variance():
    inst = _instance()
    sY = 0.2
    _, Y = simulate(inst, _plan_a(), v=0.0, sY=sY, S=20000, seed=5)
    assert abs(Y.mean() - 1.0) < 0.01
    expected_var = np.exp(sY * sY) - 1.0
    assert abs(Y.var() - expected_var) / expected_var < 0.08


def test_evaluate_plans_output_keys_and_suffix():
    inst = _instance()
    res = evaluate_plans(inst, [_plan_a()], v=0.3, sY=0.2, S=500, seed=1)[0]
    for key in ("E_max", "E_range", "E_gmd", "E_mad", "E_var", "CVaR90_range", "mean_total"):
        assert key in res and key + "_X" in res
    assert len(res["mean_driver"]) == 5
    assert len(res["sd_driver_X"]) == 5
    assert res["E_range"] >= 0.0
    assert res["E_max"] >= res["E_range"]

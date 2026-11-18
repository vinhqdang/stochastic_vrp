"""THR and the grid predictor (EXPERIMENT_B.md 3.1, M13). Toy routes are checked against
hand computations; the independent-world check runs simulate_policy on fresh paths."""
import numpy as np
import pytest

from transvrp.expa.features import RouteCache, RouteStats
from transvrp.expa.paths import Paths, RouteSpec
from transvrp.expa.simulate import simulate_policy
from transvrp.expb.grid import GridPredictor, lin_interp, tie_tolerance
from transvrp.expb.thr import (fit_thr, independent_dp, permute_columns,
                               thresholds, THRConfig)


def make_spec(B, H, E, m):
    return RouteSpec(inst="TOY", route_id=0, route=tuple(range(1, m + 1)), m=m,
                     B=float(B), H=np.asarray(H, float), E=np.asarray(E, float),
                     E0=float(E[0]))


def integer_grid(g, B, m):
    """Grid with unit spacing that starts at w_lo, so every w + g lands on a node."""
    lo = int(round(float(np.min(np.cumsum(g, axis=1)[:, : m - 1])
                         - np.max(np.abs(g.min(axis=0))))))
    return lo, int(B) - lo + 1


# ---------------------------------------------------------------- hand computations

def test_two_stop_route_matches_hand_computation():
    g = np.array([[1.0, 2.0], [1.0, -1.0], [0.0, 3.0], [2.0, 0.0]])
    B, H, E = 3.0, [1.5, 0.7], [0.7, 0.9]
    spec = make_spec(B, H, E, 2)
    w_grid, C, V = independent_dp(spec, g, n_grid=6)
    # the value of H[m-1] is never used by the policy, but V_m = H would change C_1
    # C_1(w) = mean over the second stop's increments of 1{w + g > B} * E[1]
    expected = np.array([np.mean((w + g[:, 1]) > B) * E[1] for w in w_grid])
    assert np.max(np.abs(C[1] - expected)) < 1e-12
    assert np.max(np.abs(V[1] - np.minimum(H[0], expected))) < 1e-12


def test_three_stop_route_matches_hand_recursion_on_integer_grid():
    g = np.array([[1, 2, 0], [0, -1, 2], [2, 1, -1], [-1, 0, 1]], dtype=float)
    B, H, E = 4.0, [2.5, 1.0, 0.6], [0.6, 0.8, 1.1]
    spec = make_spec(B, H, E, 3)
    lo, n = integer_grid(g, B, 3)
    w_grid, C, V = independent_dp(spec, g, n_grid=n)
    assert np.allclose(np.diff(w_grid), 1.0)

    # hand recursion on the lattice: C_2 from S_3 (breach E[2]); V_2 = min(H[1], C_2);
    # C_1 from S_2 (breach E[1], next value V_2 at w + g, clipped at the lowest node)
    c2 = {w: np.mean([E[2] if w + gg > B else 0.0 for gg in g[:, 2]]) for w in w_grid}
    v2 = {w: min(H[1], c2[w]) for w in w_grid}

    def v2_at(x):
        return v2[max(x, lo)]

    c1 = {w: np.mean([E[1] if w + gg > B else v2_at(w + gg) for gg in g[:, 1]])
          for w in w_grid}
    assert np.max(np.abs(C[2] - np.array([c2[w] for w in w_grid]))) < 1e-12
    assert np.max(np.abs(C[1] - np.array([c1[w] for w in w_grid]))) < 1e-12
    assert np.max(np.abs(V[2] - np.array([v2[w] for w in w_grid]))) < 1e-12


# ---------------------------------------------------------------- properties

def test_c_k_is_nondecreasing_in_the_load():
    rng = np.random.default_rng(3)
    g = rng.integers(-2, 4, size=(800, 5)).astype(float)
    spec = make_spec(6.0, [2.0, 1.5, 1.0, 0.5], [0.4, 0.6, 0.8, 0.9, 1.0], 5)
    _, C, _ = independent_dp(spec, g, n_grid=301)
    for k, ck in C.items():
        assert np.all(np.diff(ck) >= -1e-12), k


def test_thresholds_match_the_first_crossing():
    w = np.linspace(-1.0, 5.0, 61)
    C = {1: np.linspace(0.0, 2.0, 61), 2: np.concatenate([np.zeros(30), np.ones(31)])}
    H = [1.0, 0.5]
    got = thresholds(w, C, H, tie_tol=1e-9)
    for k in C:
        first = next((w[i] for i in range(w.size) if C[k][i] >= H[k - 1] - 1e-9), np.inf)
        assert got[k] == first
    assert thresholds(w, {1: np.zeros(61)}, [1.0], 0.0) == {1: np.inf}


def test_thresholds_use_the_tie_direction_of_t1():
    w = np.array([0.0, 1.0, 2.0])
    C = {1: np.array([0.0, 0.5, 1.0])}
    assert thresholds(w, C, [1.0], tie_tol=0.0) == {1: 2.0}   # C = H ties to handoff
    assert thresholds(w, C, [1.0], tie_tol=0.5) == {1: 1.0}


def test_permutation_preserves_column_marginals_exactly():
    rng = np.random.default_rng(11)
    g = rng.integers(0, 9, size=(500, 4)).astype(float)
    p = permute_columns(g, "CON3-0|3|test")
    assert p.shape == g.shape
    for j in range(g.shape[1]):
        assert np.array_equal(np.sort(p[:, j]), np.sort(g[:, j]))
    assert not np.array_equal(p, g)
    assert np.array_equal(permute_columns(g, "CON3-0|3|test"), p)
    assert not np.array_equal(permute_columns(g, "CON3-0|4|test"), p)


def test_independent_world_thr_agrees_with_simulation_within_three_se():
    # continuous increments: on a discrete support the linear grid interpolates across
    # the value-function jumps at lattice points and is biased (checked: a 4-atom toy
    # is off by ~0.03 at any grid size), while the generator's marginals are continuous
    rng = np.random.default_rng(2026)
    m = 4
    g_train = rng.normal(0.3, 1.5, size=(20_000, m))
    g_test = rng.normal(0.3, 1.5, size=(20_000, m))
    spec = make_spec(4.0, [2.0, 1.5, 1.0, 0.0], [0.5, 1.0, 1.5, 2.0], m)
    models, stats, _, _ = fit_thr(spec, Paths(p=g_train, d=np.zeros_like(g_train)),
                                  20_000, 0, THRConfig(n_grid=801))
    w_grid, _, V = independent_dp(spec, g_train, n_grid=801)
    dp_value = np.mean([spec.E[0] if g1 > spec.B else np.interp(g1, w_grid, V[1])
                        for g1 in g_train[:, 0]])
    test_cache = RouteCache(spec, Paths(p=g_test, d=np.zeros_like(g_test)), stats)
    costs, _ = simulate_policy(test_cache, spec.B, spec.H, spec.E, models, H_bill=spec.H)
    se = costs.std(ddof=1) / np.sqrt(costs.size)
    assert abs(costs.mean() - dp_value) <= 3 * se


def test_fit_thr_returns_one_predictor_per_epoch_and_a_budget():
    g = np.array([[1, 2, 0], [0, -1, 2], [2, 1, -1], [-1, 0, 1]] * 25, dtype=float)
    spec = make_spec(4.0, [2.5, 1.0, 0.0], [0.6, 0.8, 1.1], 3)
    models, stats, diag, budget = fit_thr(spec, Paths(p=g, d=np.zeros_like(g)), 100, 0,
                                          THRConfig(n_grid=41))
    assert sorted(models) == [1, 2]
    assert all(len(v) == 1 and isinstance(v[0], GridPredictor) for v in models.values())
    assert {d["k"] for d in diag} == {1, 2}
    assert budget["unique_paths"] == 100 and budget["path_stop_reads"] == 300
    assert isinstance(stats, RouteStats)
    with pytest.raises(ValueError, match="expected 99"):
        fit_thr(spec, Paths(p=g, d=np.zeros_like(g)), 99, 0)


# ---------------------------------------------------------------- grid predictor

def _cache_with_loads(loads):
    cache = RouteCache.__new__(RouteCache)
    cache.W = np.asarray(loads, dtype=float).reshape(-1, 1)
    return cache


def test_predict_is_linear_interpolation_and_clips_below_the_grid():
    w = np.array([0.0, 1.0, 2.0, 3.0])
    c = np.array([[0.0, 1.0, 1.5, 4.0]])
    pred = GridPredictor(w, c)
    out = pred.predict(_cache_with_loads([-5.0, 0.5, 1.25, 2.5, 3.0]), 1, np.arange(5))
    assert np.allclose(out, [0.0, 0.5, 1.125, 2.75, 4.0])


def test_tie_tolerance_is_added_to_the_prediction():
    pred = GridPredictor(np.array([0.0, 1.0]), np.array([[1.0, 2.0]]), tie_tol=0.25)
    assert pred.predict(_cache_with_loads([0.5]), 1, np.array([0])) == pytest.approx([1.75])


def test_grid_must_be_uniform_and_increasing():
    with pytest.raises(ValueError):
        GridPredictor(np.array([0.0, 1.0, 3.0]), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        GridPredictor(np.array([1.0, 0.0]), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        GridPredictor(np.array([0.0, 1.0]), np.zeros((1, 3)))


def test_lin_interp_matches_numpy_inside_the_grid():
    w = np.linspace(-2.0, 4.0, 13)
    v = np.sin(w)
    x = np.linspace(-2.0, 4.0, 97)
    assert np.allclose(lin_interp(x, w[0], w[1] - w[0], v), np.interp(x, w, v), atol=1e-12)


def test_tie_tolerance_is_t1():
    assert tie_tolerance(2.0) == pytest.approx(2e-6)

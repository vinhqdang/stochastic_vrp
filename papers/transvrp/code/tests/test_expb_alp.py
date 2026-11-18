"""ALP-Z (EXPERIMENT_B.md 3.2, 10 M14). Toy routes with known answers: an independent
loop-level recursion is the reference for the LP; the tie test pins the T1 rule."""
import numpy as np
import pytest

from transvrp.expa.features import RouteCache, RouteStats
from transvrp.expa.paths import Paths, RouteSpec
from transvrp.expa.simulate import simulate_policy
from transvrp.expb.alp import (ALPZConfig, Lattice, build_lp, continuation_table,
                               deploy_predictor, fit_alpz, initial_value_dp, lattice_dp,
                               lattice_pmf, make_lattice, solve_lp)
from transvrp.expb.grid import GridPredictor, tie_tolerance
from transvrp.expb.thr import _grid_bounds, independent_dp, thresholds


def make_spec(B, H, E, m):
    return RouteSpec(inst="TOY", route_id=0, route=tuple(range(1, m + 1)), m=m,
                     B=float(B), H=np.asarray(H, float), E=np.asarray(E, float),
                     E0=float(E[0]))


def dp_reference(spec, lat, offsets, probs):
    """Loop-level lattice recursion written from the formulas of 3.2 (independent of
    alp.lattice_dp): V_m = 0; C_k(i) = sum_r p_{k+1}(r)[E[k] 1{i+r>n-1} + ... ];
    V_k = min(H[k-1], C_k). Returns ({k: list}, initial value at the stop-1 state)."""
    m, n = spec.m, lat.n_w
    V = {m: [0.0] * n}
    for k in range(m - 1, 0, -1):
        row = []
        for i in range(n):
            c = 0.0
            for r, pr in zip(offsets, probs[k]):
                j = i + int(r)
                c += pr * (spec.E[k] if j > n - 1 else V[k + 1][max(j, 0)])
            row.append(min(spec.H[k - 1], c))
        V[k] = row
    i0 = lat.i0
    init = 0.0
    for r, pr in zip(offsets, probs[0]):
        j = i0 + int(r)
        init += pr * (spec.E[0] if j > n - 1 else V[1][max(j, 0)])
    return V, init


TOY_LAT = Lattice(w_lo=-3.0, h=1.0, n_w=7)               # w = -3..3, B = 3
TOY_OFF = np.array([-1, 0, 2])
TOY_PROB = np.array([[0.2, 0.3, 0.5],                    # stop 1
                     [0.5, 0.25, 0.25],                  # stop 2
                     [0.1, 0.6, 0.3]])                   # stop 3
TOY_SPEC = make_spec(3.0, [0.8, 0.6, 0.7], [1.0, 1.5, 2.0], 3)


# ---------------------------------------------------------------- lattice and pmfs

def test_lattice_pmf_rounds_increments_to_the_nearest_multiple():
    g = np.array([[0.4, -1.0, 2.0], [1.6, -0.9, 2.2]])
    offsets, probs = lattice_pmf(g, 1.0)
    assert list(offsets) == [-1, 0, 2]
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[0], [0.0, 0.5, 0.5])        # 0.4 -> 0, 1.6 -> 2
    assert np.allclose(probs[1], [1.0, 0.0, 0.0])        # both -> -1
    assert np.allclose(probs[2], [0.0, 0.0, 1.0])        # 2.0, 2.2 -> 2


def test_lattice_pmf_scales_with_h():
    g = np.array([[0.9, 0.2], [2.4, -0.7]])
    offsets, probs = lattice_pmf(g, 0.5)                 # 0.9 -> 2, 2.4 -> 5 at h = 0.5
    assert 2 in offsets and 5 in offsets
    assert probs[0][list(offsets).index(2)] == pytest.approx(0.5)


def test_make_lattice_uses_the_thr_bounds_and_ends_at_B():
    g = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 2.0], [2.0, 1.0, -1.0], [-1.0, 0.0, 1.0]])
    spec = make_spec(4.0, [2.5, 1.0, 0.6], [0.6, 0.8, 1.1], 3)
    lat = make_lattice(spec, g, 41)
    assert lat.w_lo == pytest.approx(_grid_bounds(g, spec.B, 3))
    assert lat.n_w == 41
    assert lat.h == pytest.approx((spec.B - lat.w_lo) / 40)
    assert lat.w[-1] == pytest.approx(spec.B, abs=1e-12)


def test_config_rejects_unknown_objective():
    with pytest.raises(ValueError):
        ALPZConfig(objective="weighted")


# ---------------------------------------------------------------- exact lattice DP

def test_lattice_dp_matches_the_loop_reference():
    V_ref, init_ref = dp_reference(TOY_SPEC, TOY_LAT, TOY_OFF, TOY_PROB)
    pmfs = (TOY_OFF, TOY_PROB)
    V = lattice_dp(TOY_SPEC, pmfs, TOY_LAT)
    for k in (1, 2):
        assert np.allclose(V[k - 1], V_ref[k], atol=1e-12)
    assert initial_value_dp(TOY_SPEC, pmfs, TOY_LAT) == pytest.approx(init_ref, abs=1e-12)


def test_lp_initial_objective_plus_breach_constant_is_the_dp_initial_value():
    _, init_ref = dp_reference(TOY_SPEC, TOY_LAT, TOY_OFF, TOY_PROB)
    c, A, b, info = build_lp(TOY_SPEC, (TOY_OFF, TOY_PROB), TOY_LAT, ALPZConfig())
    x, status, _ = solve_lp(c, A, b, info)
    assert status == 0
    # the published objective (24.1) omits the breach cost at stop 1, a constant
    assert c @ x + info["const"] == pytest.approx(init_ref, abs=1e-9)


def test_lp_uniform_objective_reproduces_the_dp_at_every_lattice_point():
    V_ref, _ = dp_reference(TOY_SPEC, TOY_LAT, TOY_OFF, TOY_PROB)
    c, A, b, info = build_lp(TOY_SPEC, (TOY_OFF, TOY_PROB), TOY_LAT,
                             ALPZConfig(n_w=7, objective="uniform"))
    x, status, _ = solve_lp(c, A, b, info)
    assert status == 0
    f = x.reshape(TOY_SPEC.m - 1, TOY_LAT.n_w)
    for k in (1, 2):
        assert np.max(np.abs(f[k - 1] - np.array(V_ref[k]))) < 1e-9


def test_lp_size_and_nonzeros_match_the_layout():
    c, A, b, info = build_lp(TOY_SPEC, (TOY_OFF, TOY_PROB), TOY_LAT, ALPZConfig())
    n_w, K = TOY_LAT.n_w, TOY_SPEC.m - 1
    assert c.size == K * n_w and b.size == K * n_w and A.shape == (K * n_w, K * n_w)
    expected = 0
    for k in range(1, K + 1):
        for i in range(n_w):
            cols = {(k - 1) * n_w + i}
            for r, pr in zip(TOY_OFF, TOY_PROB[k]):
                j = i + int(r)
                if pr > 0 and j <= n_w - 1 and k < K:
                    cols.add(k * n_w + max(j, 0))
            expected += len(cols)
    assert A.nnz == expected
    assert info["n_vars"] == K * n_w


def test_lp_solution_is_deterministic():
    c, A, b, info = build_lp(TOY_SPEC, (TOY_OFF, TOY_PROB), TOY_LAT, ALPZConfig())
    x1, s1, _ = solve_lp(c, A, b, info)
    x2, s2, _ = solve_lp(c, A, b, info)
    assert s1 == s2 == 0
    assert np.array_equal(x1, x2)


def test_handoff_bounds_are_respected_by_the_vertex():
    c, A, b, info = build_lp(TOY_SPEC, (TOY_OFF, TOY_PROB), TOY_LAT,
                             ALPZConfig(n_w=7, objective="uniform"))
    x, _, _ = solve_lp(c, A, b, info)
    n_w = TOY_LAT.n_w
    assert np.all(x[:n_w] <= TOY_SPEC.H[0] + 1e-9)
    assert np.all(x[n_w:] <= TOY_SPEC.H[1] + 1e-9)


# ---------------------------------------------------------------- deployment and ties

def test_deployed_rule_equals_thr_on_a_lattice_aligned_toy():
    rng = np.random.default_rng(7)
    g = rng.integers(-2, 4, size=(60, 3)).astype(float)
    spec = make_spec(4.0, [0.9, 0.6, 0.4], [2.0, 1.5, 1.0], 3)
    lo = _grid_bounds(g, spec.B, 3)
    n_w = int(round(spec.B - lo)) + 1                    # unit spacing, both ends on nodes
    lat = make_lattice(spec, g, n_w)
    assert lat.h == pytest.approx(1.0)
    pmfs = lattice_pmf(g, lat.h)
    c, A, b, info = build_lp(spec, pmfs, lat, ALPZConfig(n_w=n_w, objective="uniform"))
    x, status, _ = solve_lp(c, A, b, info)
    assert status == 0
    f = x.reshape(2, n_w)
    table = continuation_table(spec, pmfs, lat, f, lat.w)
    w_thr, C_thr, _ = independent_dp(spec, g, n_grid=n_w)
    assert np.allclose(w_thr, lat.w, atol=1e-12)
    tie = tie_tolerance(spec.E0)
    for k in (1, 2):
        assert np.max(np.abs(table[k - 1] - C_thr[k])) < 1e-7
        assert np.array_equal(table[k - 1] >= spec.H[k - 1] - tie,
                              C_thr[k] >= spec.H[k - 1] - tie)


def _two_stop_tie_toy():
    """Stop 1 never moves the load; stop 2 breaches with probability 1/2, so the
    continuation cost at w = 0 is 0.5 * E[1] = 0.5 = H[0]: an exact tie."""
    lat = Lattice(w_lo=-2.0, h=1.0, n_w=5)               # w = -2..2, B = 2
    offsets = np.array([-1, 0, 3])
    probs = np.array([[0.0, 1.0, 0.0],
                      [0.5, 0.0, 0.5]])
    spec = make_spec(2.0, [0.5, 0.7], [1.0, 1.0], 2)
    return spec, lat, (offsets, probs)


def test_tie_at_a_handoff_optimal_state_hands_off():
    spec, lat, pmfs = _two_stop_tie_toy()
    c, A, b, info = build_lp(spec, pmfs, lat, ALPZConfig(n_w=5))
    x, status, _ = solve_lp(c, A, b, info)
    assert status == 0
    f = x.reshape(1, 5)
    assert f[0, lat.i0] == pytest.approx(spec.H[0], abs=1e-9)   # the vertex sits at H
    L = continuation_table(spec, pmfs, lat, f, np.array([0.0]))
    assert L[0, 0] == pytest.approx(spec.H[0], abs=1e-12)       # L = H exactly
    paths = Paths(p=np.array([[0.0, 3.0], [0.0, 0.0], [0.0, 3.0], [0.0, 0.0]]),
                  d=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0]]))
    cache = RouteCache(spec, paths, RouteStats.fit(paths))
    models = {1: [GridPredictor(np.array([-2.0, 2.0]), np.repeat(L, 2, axis=1),
                                tie_tolerance(spec.E0))]}
    _, action = simulate_policy(cache, spec.B, spec.H, spec.E, models)
    assert np.all(action == 1)                           # every route hands off at 1


def test_deploy_predictor_hands_off_at_the_tie_on_the_lattice():
    spec, lat, pmfs = _two_stop_tie_toy()
    c, A, b, info = build_lp(spec, pmfs, lat, ALPZConfig(n_w=5))
    x, _, _ = solve_lp(c, A, b, info)
    predictor = deploy_predictor(spec, pmfs, lat, x.reshape(1, 5), lat.w)
    assert predictor.tie_tol == pytest.approx(tie_tolerance(spec.E0))
    paths = Paths(p=np.array([[0.0, 3.0], [0.0, 0.0], [0.0, 3.0], [0.0, 0.0]]),
                  d=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0]]))
    cache = RouteCache(spec, paths, RouteStats.fit(paths))
    _, action = simulate_policy(cache, spec.B, spec.H, spec.E, {1: [predictor]})
    assert np.all(action == 1)


def test_strict_rule_would_continue_at_the_tie():
    spec, lat, pmfs = _two_stop_tie_toy()
    L = np.array([[spec.H[0]]])
    strict = GridPredictor(np.array([-2.0, 2.0]), np.repeat(L, 2, axis=1), 0.0)
    cache = RouteCache.__new__(RouteCache)
    cache.W = np.zeros((1, 2))
    prediction = strict.predict(cache, 1, np.array([0]))[0]
    assert not (prediction > spec.H[0])                  # L > H would be false: continue


# ---------------------------------------------------------------- fit end to end

def test_fit_alpz_returns_one_grid_predictor_per_epoch_and_a_budget():
    rng = np.random.default_rng(5)
    g = rng.integers(-2, 4, size=(50, 3)).astype(float)
    paths = Paths(p=np.maximum(g, 0.0), d=np.maximum(-g, 0.0))
    spec = make_spec(4.0, [0.9, 0.6, 0.4], [2.0, 1.5, 1.0], 3)
    models, stats, diag, budget = fit_alpz(spec, paths, 50, 0, ALPZConfig(n_w=41))
    assert sorted(models) == [1, 2]
    assert all(isinstance(v[0], GridPredictor) for v in models.values())
    assert isinstance(stats, RouteStats)
    assert budget["lp_rows"] == 2 * 41 and budget["lp_cols"] == 2 * 41
    assert budget["lp_nnz"] > 0 and budget["unique_paths"] == 50
    assert {d["k"] for d in diag} == {1, 2}
    thr = thresholds(models[1][0].w_grid, {1: models[1][0].c[0], 2: models[1][0].c[1]},
                     spec.H, 0.0)
    assert set(thr) == {1, 2}


def test_fit_alpz_refuses_a_wrong_path_count():
    spec = make_spec(4.0, [0.9, 0.6, 0.4], [2.0, 1.5, 1.0], 3)
    paths = Paths(p=np.ones((10, 3)), d=np.zeros((10, 3)))
    with pytest.raises(ValueError, match="expected 11"):
        fit_alpz(spec, paths, 11, 0, ALPZConfig(n_w=11))

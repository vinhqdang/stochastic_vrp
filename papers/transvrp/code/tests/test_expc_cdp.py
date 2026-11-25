"""CDP (EXPERIMENT_C 1.7, 6, 9 MC3): posterior, tables, predictor, fit and gate helpers.

Toy routes use gamma marginals (CV 0.3) with a Gaussian copula written out here, so the
model is correctly specified. The real-route tests use CON3-0 with reduced grids (speed);
timing at the full configuration is measured separately (see the module report).
"""
import numpy as np
import pytest
import scipy.stats as st

from transvrp.data import gen_scenarios, load_plans, stable_seed
from transvrp.expa.features import RouteCache, RouteStats
from transvrp.expa.paths import Paths, RouteSpec, route_pool, route_specs
from transvrp.expa.simulate import simulate_policy
from transvrp.expb.grid import GridPredictor, lin_interp, tie_tolerance
from transvrp.expb.thr import THRConfig, _continuation, _grid_bounds, fit_thr
from transvrp.expc import cdp
from transvrp.expc.cdp import (CDPPredictor, build_tables, fit_cdp, fit_cdp_model,
                               g_cdp0_within, node_check, node_tables, posterior)

CV = 0.30


def make_spec(B, H, E, m, inst="TOY"):
    return RouteSpec(inst=inst, route_id=0, route=tuple(range(1, m + 1)), m=m,
                     B=float(B), H=np.asarray(H, float), E=np.asarray(E, float),
                     E0=float(E[0]))


def toy_spec():
    return make_spec(2.0, [0.9, 0.7, 0.5, 0.0], [1.0, 1.2, 1.4, 1.6], 4)


def copula_paths(mu_p, mu_d, n, rho, seed):
    """Gamma marginals (CV 0.3) with an equicorrelated Gaussian copula per side."""
    rng = np.random.default_rng(seed)

    def side(mu):
        m = len(mu)
        Sigma = np.full((m, m), rho)
        np.fill_diagonal(Sigma, 1.0)
        Z = rng.standard_normal((n, m)) @ np.linalg.cholesky(Sigma).T
        U = np.clip(st.norm.cdf(Z), 1e-12, 1 - 1e-12)
        k = 1.0 / CV ** 2
        return st.gamma.ppf(U, k, scale=np.asarray(mu) / k)

    return Paths(p=side(mu_p), d=side(mu_d))


def toy_paths(n, rho, seed):
    mu_p = [1.0, 1.2, 0.9, 1.1]
    mu_d = [0.4, 0.5, 0.4, 0.6]
    return copula_paths(mu_p, mu_d, n, rho, seed)


def _thr_c_at(models, cache, k, loads):
    return models[k][0].predict(cache, k, np.arange(loads.size))


# ------------------------------------------------------------------ (a) posterior

def _brute_posterior(z, rho):
    """Gaussian conditioning on the joint (Z, z_1..z_k) with the model's covariance."""
    k = z.size
    if k == 0:
        return 0.0, 1.0
    S = np.full((k, k), rho)
    np.fill_diagonal(S, 1.0)
    c = np.full(k, np.sqrt(rho))
    sol = np.linalg.solve(S, np.column_stack([z, c]))
    mu = float(c @ sol[:, 0])
    var = 1.0 - float(c @ sol[:, 1])
    return mu, np.sqrt(var)


def test_posterior_matches_brute_force_gaussian_conditioning():
    rng = np.random.default_rng(20261125)
    for _ in range(1000):
        rho = rng.uniform(0.0, 0.95)
        k = int(rng.integers(0, 16))
        z = rng.standard_normal(k)
        mu, s = posterior(z[None, :], rho)
        mu_b, s_b = _brute_posterior(z, rho)
        assert abs(mu[0] - mu_b) < 1e-9
        assert abs(s - s_b) < 1e-9


def test_posterior_k0_is_the_prior_and_rows_are_independent():
    mu, s = posterior(np.zeros((3, 0)), 0.6)
    assert np.array_equal(mu, np.zeros(3)) and s == pytest.approx(1.0)
    z = np.array([[0.5, -1.0], [2.0, 0.0]])
    mu2, s2 = posterior(z, 0.4)
    assert mu2[0] == pytest.approx(posterior(z[:1], 0.4)[0][0])
    assert s2 == pytest.approx(posterior(z[1:], 0.4)[1])


# ------------------------------------------------------------------ fit of the model

def test_fit_cdp_model_recovers_rho_on_gen_scenarios_at_rho_06():
    inst = "CON3-0"
    sol = load_plans(inst)
    base = stable_seed(inst)
    dsc, psc = gen_scenarios(sol["dbar"], sol["pbar"], 20_000, [base, 1, 0, 0], rho=0.6)
    paths = Paths(p=psc, d=dsc)
    model = fit_cdp_model(paths)
    assert abs(model.rho_d - 0.6) <= 0.03
    assert abs(model.rho_p - 0.6) <= 0.03
    zero = np.flatnonzero(sol["dbar"] == 0)
    assert not model.active_d[zero].any() and np.all(model.shape_d[zero] == 0)


def test_fit_cdp_model_clips_and_overrides():
    paths = toy_paths(5000, 0.0, 1)
    assert fit_cdp_model(paths).rho_d < 0.05
    model = fit_cdp_model(paths, rho_override=0.6)
    assert model.rho_d == model.rho_p == pytest.approx(0.6)
    with pytest.raises(ValueError):
        fit_cdp_model(paths, rho_override=1.0)


def test_inactive_stop_stays_degenerate_and_moments_are_recovered():
    rng = np.random.default_rng(3)
    d = rng.gamma(4.0, 0.25, size=(20000, 3))
    d[:, 1] = 0.0
    p = np.zeros_like(d)
    model = fit_cdp_model(Paths(p=p, d=d))
    assert model.active_d.tolist() == [True, False, True]
    assert model.shape_d[1] == 0.0 and model.scale_d[1] == 0.0
    assert model.shape_d[0] == pytest.approx(4.0, rel=0.05)
    assert model.scale_d[0] == pytest.approx(0.25, rel=0.05)


# ------------------------------------------------------------------ (b) monotone tables

def test_tables_are_nondecreasing_in_w_for_every_node_toy():
    spec = toy_spec()
    paths = toy_paths(4000, 0.6, 2)
    model = fit_cdp_model(paths)
    tables = build_tables(spec, model, paths, n_nodes=5, n_eps=256, n_grid=401, seed=5)
    assert np.all(np.diff(tables.C, axis=-1) >= -1e-9 * spec.E0)


def test_tables_are_nondecreasing_in_w_on_a_real_route():
    spec = route_specs("CON3-0")[0]
    pool = route_pool("CON3-0", "train", 0, 2000, routes=[spec.route_id])[spec.route_id]
    model = fit_cdp_model(pool)
    tables = build_tables(spec, model, pool, n_nodes=5, n_eps=256, n_grid=201, seed=7)
    assert np.all(np.diff(tables.C, axis=-1) >= -1e-9 * spec.E0)


# ------------------------------------------------------------------ (c) rho = 0 is THR

def test_rho_zero_tables_do_not_depend_on_the_node_and_match_the_thr_recursion():
    spec = toy_spec()
    paths = toy_paths(4000, 0.0, 4)
    model = fit_cdp_model(paths, rho_override=0.0)
    tables = build_tables(spec, model, paths, n_nodes=5, n_eps=256, n_grid=301, seed=9)
    for a in range(5):
        for b in range(5):
            assert np.max(np.abs(tables.C[:, a, b] - tables.C[:, 0, 0])) < 1e-12
    # same draws, same grid, THR's own recursion
    D = cdp._margin_draws(model.shape_d, model.scale_d, model.active_d, 0.0,
                          tables.zeta[:1], tables.eps_d)[0]
    P = cdp._margin_draws(model.shape_p, model.scale_p, model.active_p, 0.0,
                          tables.zeta[:1], tables.eps_p)[0]
    g = P - D
    v_next = np.zeros(tables.w_grid.size)
    for k in range(spec.m - 1, 0, -1):
        ck = _continuation(tables.w_grid, g[:, k], spec.B, float(spec.E[k]), v_next)
        assert np.max(np.abs(ck - tables.C[k - 1, 0, 0])) < 1e-12
        v_next = np.minimum(float(spec.H[k - 1]), ck)


def test_rho_zero_cdp_agrees_with_thr_on_a_toy_route_within_sampling_tolerance():
    spec = toy_spec()
    paths = toy_paths(20000, 0.0, 6)
    cfg = THRConfig(n_grid=401)
    thr_models, thr_stats, _, _ = fit_thr(spec, paths, 20000, 0, cfg)
    out = fit_cdp(spec, paths, 20000, 0, rho_override=0.0, n_nodes=5, n_eps=256, n_grid=401)
    test = toy_paths(20000, 0.0, 60)
    thr_cache = RouteCache(spec, test, thr_stats)
    cdp_cache = RouteCache(spec, test, out["stats"])
    thr_costs, thr_act = simulate_policy(thr_cache, spec.B, spec.H, spec.E, thr_models,
                                         H_bill=spec.H)
    cdp_costs, cdp_act = simulate_policy(cdp_cache, spec.B, spec.H, spec.E, out["models"],
                                         H_bill=spec.H)
    # documented tolerance: 256 quasi-random draws per stop versus 20,000 empirical ones
    assert abs(cdp_costs.mean() - thr_costs.mean()) <= 0.01 * spec.E0
    assert np.mean(cdp_act == thr_act) >= 0.98


# ------------------------------------------------------------------ (d) interpolation

def test_nodes_reproduce_the_table_and_edges_are_clamped():
    spec = toy_spec()
    paths = toy_paths(3000, 0.6, 8)
    model = fit_cdp_model(paths)
    tables = build_tables(spec, model, paths, n_nodes=5, n_eps=256, n_grid=201, seed=3)
    for a in range(5):
        for b in range(5):
            got = node_tables(tables, tables.zeta[a], tables.zeta[b])
            assert np.max(np.abs(got - tables.C[:, a, b])) < 1e-12
    lo, hi = tables.zeta[0], tables.zeta[-1]
    assert np.max(np.abs(node_tables(tables, 10.0, 0.0) - node_tables(tables, hi, 0.0))) < 1e-12
    assert np.max(np.abs(node_tables(tables, -10.0, 0.0) - node_tables(tables, lo, 0.0))) < 1e-12
    assert np.max(np.abs(node_tables(tables, 0.0, 10.0) - node_tables(tables, 0.0, hi))) < 1e-12


def test_locate_clamps_to_the_node_range():
    i, f = cdp._locate(np.array([-10.0, -4.0, 0.0, 4.0, 10.0]), -4.0, 1 / 3, 25)
    assert i.tolist() == [0, 0, 12, 23, 23]
    assert np.allclose(f[[0, 1]], 0.0) and f[4] == pytest.approx(1.0)


def test_predictor_reproduces_a_constant_table_regardless_of_posterior():
    spec = toy_spec()
    paths = toy_paths(2000, 0.0, 9)
    model = fit_cdp_model(paths, rho_override=0.0)
    tables = build_tables(spec, model, paths, n_nodes=5, n_eps=128, n_grid=201, seed=4)
    pred = CDPPredictor(model, tables)
    cache = RouteCache(spec, toy_paths(50, 0.0, 10), RouteStats.fit(paths))
    out = pred.predict(cache, 2, np.arange(50))
    grid_val = lin_interp(cache.W[:50, 1], float(tables.w_grid[0]),
                          float(tables.w_grid[1] - tables.w_grid[0]), tables.C[1, 0, 0])
    assert np.max(np.abs(out - (grid_val + tables.tie_tol))) < 1e-9


# ------------------------------------------------------------------ (e) predictions

def test_predictions_are_finite_and_within_zero_and_e0_plus_tau():
    spec = route_specs("CON3-0")[0]
    pool = route_pool("CON3-0", "train", 0, 2000, routes=[spec.route_id])[spec.route_id]
    out = fit_cdp(spec, pool, 2000, 0, n_nodes=5, n_eps=128, n_grid=201)
    test = route_pool("CON3-0", "train", 1, 2000, routes=[spec.route_id])[spec.route_id]
    cache = RouteCache(spec, test, out["stats"])
    tau = tie_tolerance(spec.E0)
    for k, preds in out["models"].items():
        # only non-breached rows reach a decision (simulate_policy); breached rows lie
        # outside the load grid, where the table is extrapolated
        live = np.flatnonzero(cache.W[:, k - 1] <= spec.B)
        val = preds[0].predict(cache, k, live)
        assert np.all(np.isfinite(val))
        assert val.min() >= 0.0 and val.max() <= spec.E0 + tau


def test_predictions_depend_on_the_observed_scores_when_rho_is_positive():
    spec = toy_spec()
    paths = toy_paths(3000, 0.6, 11)
    model = fit_cdp_model(paths)
    tables = build_tables(spec, model, paths, n_nodes=9, n_eps=128, n_grid=201, seed=6)
    pred = CDPPredictor(model, tables)
    # two rows with identical loads (g = 0 at every stop) but opposite observed deliveries
    # and pickups: the posterior of the factors must move the prediction
    high = np.full((1, spec.m), 2.0)
    low = np.full((1, spec.m), 0.2)
    test = Paths(p=np.vstack([high, low]), d=np.vstack([high, low]))
    cache = RouteCache(spec, test, RouteStats.fit(paths))
    assert np.allclose(cache.W[:, 2], 0.0)
    out = pred.predict(cache, 3, np.arange(2))
    assert abs(out[0] - out[1]) > 1e-3 * spec.E0
    assert not np.allclose(tables.C[2, 0, 0], tables.C[2, -1, -1])


def test_predictor_matches_an_explicit_gauss_hermite_reference():
    """Row by row: posterior of each factor from its own scores, 9 x 9 Gauss-Hermite
    nodes, bilinear table at each node, linear in w. Ignoring either factor fails this."""
    spec = toy_spec()
    paths = toy_paths(3000, 0.6, 15)
    model = fit_cdp_model(paths)
    tables = build_tables(spec, model, paths, n_nodes=9, n_eps=128, n_grid=201, seed=8)
    pred = CDPPredictor(model, tables)
    test = toy_paths(30, 0.6, 16)
    cache = RouteCache(spec, test, RouteStats.fit(paths))
    k = 2
    out = pred.predict(cache, k, np.arange(30))
    x, wq = np.polynomial.hermite_e.hermegauss(9)
    wq = wq / wq.sum()
    w0, h = float(tables.w_grid[0]), float(tables.w_grid[1] - tables.w_grid[0])
    for r in range(30):
        ref = 0.0
        p_obs = np.diff(cache.P[r, :k], prepend=0.0)
        d_obs = np.diff(cache.D[r, :k], prepend=0.0)
        zd = st.norm.ppf(st.gamma.cdf(d_obs[model.active_d[:k]],
                                      model.shape_d[:k][model.active_d[:k]],
                                      scale=model.scale_d[:k][model.active_d[:k]]))
        zp = st.norm.ppf(st.gamma.cdf(p_obs[model.active_p[:k]],
                                      model.shape_p[:k][model.active_p[:k]],
                                      scale=model.scale_p[:k][model.active_p[:k]]))
        mu_d, s_d = posterior(zd[None, :], model.rho_d)
        mu_p, s_p = posterior(zp[None, :], model.rho_p)
        for i in range(9):
            for j in range(9):
                zdn = np.clip(mu_d[0] + s_d * x[i], -4.0, 4.0)
                zpn = np.clip(mu_p[0] + s_p * x[j], -4.0, 4.0)
                row = node_tables(tables, zdn, zpn)[k - 1]
                # loads are clamped to the grid (finding S5), as the predictor does
                w_load = np.clip(np.array([cache.W[r, k - 1]]), w0, float(tables.w_grid[-1]))
                ref += wq[i] * wq[j] * lin_interp(w_load, w0, h, row)[0]
        assert out[r] == pytest.approx(ref + tables.tie_tol, abs=1e-9)


def test_tables_depend_on_the_node_when_rho_is_positive():
    spec = toy_spec()
    paths = toy_paths(3000, 0.6, 13)
    model = fit_cdp_model(paths)
    tables = build_tables(spec, model, paths, n_nodes=5, n_eps=128, n_grid=201, seed=2)
    assert np.max(np.abs(tables.C[1, 0, 0] - tables.C[1, -1, -1])) > 1e-3


# ------------------------------------------------------------------ gate helpers

def test_node_check_runs_and_reports_a_finite_comparison():
    spec = toy_spec()
    paths = toy_paths(3000, 0.6, 14)
    model = fit_cdp_model(paths)
    # production node count: the off-grid node (0.5, -0.5) is interpolated, and with 9 nodes
    # the bilinear error is ~0.036 (13 SE); with 25 nodes it is ~0.006 (checked in the report)
    tables = build_tables(spec, model, paths, n_nodes=25, n_eps=256, n_grid=401, seed=1)
    res = node_check(spec, model, tables, zeta=(0.5, -0.5), n=20000, seed=3)
    assert np.isfinite(res["mean"]) and np.isfinite(res["table"]) and res["se"] > 0
    assert res["ok"], res


def test_g_cdp0_accepts_identical_and_rejects_shifted_costs():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.0, 1.0, 5000)
    assert g_cdp0_within(base, base, 1.0)
    assert not g_cdp0_within(base + 0.05, base, 1.0)


# ------------------------------------------------------------ (e) grid bound (S2, S5)

def test_grid_lower_bound_on_con3_0_is_thrs_data_driven_bound():
    spec = route_specs("CON3-0")[0]
    pool = route_pool("CON3-0", "train", 0, 2000, routes=[spec.route_id])[spec.route_id]
    model = fit_cdp_model(pool)
    tables = build_tables(spec, model, pool, n_nodes=5, n_eps=64, n_grid=1001, seed=11)
    assert tables.w_grid[0] == _grid_bounds(pool.g, spec.B, spec.m)
    assert tables.w_grid[-1] == pytest.approx(spec.B)


def test_predictor_grid_position_is_clamped_to_the_grid_for_any_load():
    spec = toy_spec()
    paths = toy_paths(1000, 0.6, 12)
    model = fit_cdp_model(paths)
    tables = build_tables(spec, model, paths, n_nodes=3, n_eps=32, n_grid=51, seed=2)
    pred = CDPPredictor(model, tables)
    W = np.array([-1e9, tables.w_grid[0] - 1.0, tables.w_grid[7] + 1e-9, tables.w_grid[-1],
                  1e9])
    i0, frac = pred._w_index(W)
    assert np.all((i0 >= 0) & (i0 <= tables.w_grid.size - 2))
    assert np.all((frac >= 0.0) & (frac <= 1.0))
    assert i0[0] == 0 and frac[0] == 0.0                          # below: first value
    assert i0[-1] == tables.w_grid.size - 2 and frac[-1] == 1.0   # above: last value


def test_recursion_takes_the_first_grid_value_for_loads_below_the_grid():
    from transvrp.expc.cdp import _pair_continuation
    w_grid = np.linspace(0.0, 2.0, 21)
    h = float(w_grid[1] - w_grid[0])
    v_next = (np.arange(21, dtype=float) * 0.1)[None, :]
    g = np.array([[-5.0]])                       # every w + g lies below the grid
    out = _pair_continuation(w_grid, g, 10.0, 1.0, v_next, float(w_grid[0]), h)
    assert np.allclose(out, v_next[0, 0])


def test_rebuilt_tables_are_deterministic_and_equal_the_fit_tables():
    from transvrp.expc.stages import cdp_tables_of
    spec = toy_spec()
    paths = toy_paths(2000, 0.6, 21)
    out = fit_cdp(spec, paths, 2000, 0, n_nodes=5, n_eps=32, n_grid=101, seed=13)
    blob = {"cdp_model": out["cdp_model"],
            "params": dict(n_nodes=5, n_eps=32, n_grid=101, seed=13)}
    t1 = cdp_tables_of(blob, spec, paths)
    t2 = cdp_tables_of(blob, spec, paths)
    assert np.array_equal(t1.C, t2.C) and np.array_equal(t1.w_grid, t2.w_grid)
    assert np.array_equal(t1.C, out["tables"].C)
    assert np.array_equal(t1.w_grid, out["tables"].w_grid)


# ------------------------------------------------------------ (a) hand-computed DP, m = 3

def _hand_dp(w_grid, draws, B, E, H):
    """Backward recursion written out for m = 3 with explicit loops and np.interp (no thr
    or cdp code). C_k(w) = mean over draws of [1{w+g>B} E[k] + 1{w+g<=B} V_{k+1}(w+g)],
    V_j = min(H[j-1], C_j), V_m = 0, g = column k of the draws (stop k+1)."""
    m = draws.shape[1]
    V_next = np.zeros(w_grid.size)
    out = {}
    for k in range(m - 1, 0, -1):
        C = np.empty(w_grid.size)
        for i, w in enumerate(w_grid):
            vals = []
            for g in draws[:, k]:
                x = w + g
                vals.append(E[k] if x > B else np.interp(x, w_grid, V_next))
            C[i] = float(np.mean(vals))
        out[k] = C
        V_next = np.minimum(H[k - 1], C)
    return out


def test_cdp_tables_match_a_hand_computed_dp_for_three_stops_with_rho_zero(monkeypatch):
    B, H, E = 2.0, np.array([0.9, 0.7, 0.5]), np.array([1.0, 1.2, 1.4])
    spec = make_spec(B, H, E, 3)
    draws = np.array([[0.0, 0.4, -0.3], [0.0, -0.2, 0.5]])     # S = 2 deterministic draws
    paths = Paths(p=np.ones((4, 3)), d=np.zeros((4, 3)))       # w_lo = 0 by the THR bound
    model = fit_cdp_model(toy_paths(400, 0.0, 30), rho_override=0.0)

    def fake_gaps(model_, eps_d, eps_p, zeta):
        return np.broadcast_to(draws[None, None], (zeta.size, zeta.size, 2, 3)).copy()

    monkeypatch.setattr(cdp, "_pair_gaps", fake_gaps)
    tables = build_tables(spec, model, paths, n_nodes=3, n_eps=2, n_grid=21, seed=0)
    assert tables.w_grid[0] == 0.0 and tables.w_grid[-1] == pytest.approx(B)
    expected = _hand_dp(tables.w_grid, draws, B, E, H)
    for k in (1, 2):
        for a in range(3):
            for b in range(3):
                assert np.allclose(tables.C[k - 1, a, b], expected[k], atol=1e-12), (k, a, b)


def test_hand_dp_is_sensitive_to_the_breach_cost_and_handoff_price_of_the_stop_named():
    """The indexing guard: moving E[1] or H[1] (the values of stop 2) changes C_1, so the
    hand-computed test above would catch an index shift."""
    B, H, E = 2.0, np.array([0.9, 0.7, 0.5]), np.array([1.0, 1.2, 1.4])
    draws = np.array([[0.0, 0.4, -0.3], [0.0, -0.2, 0.5]])
    grid = np.linspace(0.0, B, 21)
    base = _hand_dp(grid, draws, B, E, H)
    shifted_H = _hand_dp(grid, draws, B, E, np.array([0.9, 0.5, 0.7]))
    shifted_E = _hand_dp(grid, draws, B, np.array([1.0, 2.0, 1.4]), H)
    assert not np.allclose(base[1], shifted_H[1])
    assert not np.allclose(base[1], shifted_E[1])


# ------------------------------------------------------------ (b) recovered increments

def test_recover_increments_returns_the_known_per_stop_p_and_d():
    from transvrp.expc.cdp import recover_increments
    rng = np.random.default_rng(41)
    spec = make_spec(2.0, [0.9, 0.7, 0.5, 0.0], [1.0, 1.2, 1.4, 1.6], 4)
    p = rng.uniform(0.1, 1.5, (50, 4))
    d = rng.uniform(0.1, 1.5, (50, 4))
    cache = RouteCache(spec, Paths(p=p, d=d), RouteStats.fit(Paths(p=p, d=d)))
    idx = np.array([0, 7, 49])
    for k in range(1, 5):
        pr, dr = recover_increments(cache, idx, k)
        assert pr.shape == (3, k) and dr.shape == (3, k)
        assert np.allclose(pr, p[idx, :k], atol=1e-12)
        assert np.allclose(dr, d[idx, :k], atol=1e-12)


# ------------------------------------------------------------ (c) zero-mean stop, end to end

def test_predictor_with_a_zero_mean_stop_is_finite_and_ignores_that_stop():
    from transvrp.expc.stages import cdp_tables_of
    spec = toy_spec()
    paths = toy_paths(1500, 0.6, 33)
    paths.p[:, 1] = 0.0
    paths.d[:, 1] = 0.0                                   # stop 2 is inactive
    out = fit_cdp(spec, paths, 1500, 0, n_nodes=5, n_eps=32, n_grid=101, seed=17)
    model = out["cdp_model"]
    assert not model.active_d[1] and not model.active_p[1]
    tables = cdp_tables_of({"cdp_model": model,
                            "params": dict(n_nodes=5, n_eps=32, n_grid=101, seed=17)},
                           spec, paths)
    pred = CDPPredictor(model, tables)
    cache = RouteCache(spec, paths, RouteStats.fit(paths))
    idx = np.arange(300)
    for k in range(1, spec.m):
        vals = pred.predict(cache, k, idx)
        assert np.all(np.isfinite(vals))
        bound = max(float(np.max(spec.E)), float(np.max(spec.H))) + tables.tie_tol
        assert np.all(vals >= -1e-9) and np.all(vals <= bound + 1e-9)
    # the posterior uses only the active stops: the inactive stop's observed value is ignored
    x_zero = paths.d[:, :3].copy()
    x_changed = x_zero.copy()
    x_changed[:, 1] = 123.0
    mu0, s0 = pred._factor_posterior(x_zero, pred._act_d, model.shape_d, model.scale_d,
                                     model.rho_d, 3)
    mu1, s1 = pred._factor_posterior(x_changed, pred._act_d, model.shape_d, model.scale_d,
                                     model.rho_d, 3)
    assert np.array_equal(mu0, mu1) and s0 == s1

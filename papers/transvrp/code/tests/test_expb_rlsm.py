"""RLSM (EXPERIMENT_B.md 3.3, 10 M15): random-feature least squares for ls_backward."""
import numpy as np
import pytest

from transvrp.expa.backward import split_80_20
from transvrp.expa.features import MLP_DIM, RouteCache, RouteStats, mlp_batch
from transvrp.expa.paths import Paths, RouteSpec
from transvrp.expa.train import ConstantPredictor
from transvrp.expb import rlsm
from transvrp.expb.rlsm import (RLSMConfig, RLSMPredictor, design, draw_layer, fit_rlsm,
                                leaky, ridge_lstsq, rlsm_fit_fn)


def make_spec(B, H, E, m):
    return RouteSpec(inst="TOY", route_id=0, route=tuple(range(1, m + 1)), m=m,
                     B=float(B), H=np.asarray(H, float), E=np.asarray(E, float),
                     E0=float(E[0]))


def make_cache(N=200, m=4, seed=1):
    spec = make_spec(4.0, [0.6] * m, [1.0] * m, m)
    rng = np.random.default_rng(seed)
    paths = Paths(p=rng.uniform(0.5, 2.0, (N, m)), d=rng.uniform(0.5, 2.0, (N, m)))
    stats = RouteStats.fit(paths)
    return spec, paths, RouteCache(spec, paths, stats)


# ---------------------------------------------------------------- building blocks

def test_leaky_relu_uses_the_slope_on_the_negative_branch():
    assert np.allclose(leaky(np.array([-2.0, 3.0]), 0.5), [-1.0, 3.0])
    assert np.allclose(leaky(np.array([-2.0, 3.0]), 1.0), [-2.0, 3.0])


def test_draw_layer_is_seeded_and_differs_by_stage_and_fold():
    A1, b1 = draw_layer("k", "A", 2, 7)
    A2, b2 = draw_layer("k", "A", 2, 7)
    assert A1.shape == (MLP_DIM, 7) and b1.shape == (7,)
    assert np.array_equal(A1, A2) and np.array_equal(b1, b2)
    assert not np.array_equal(draw_layer("k", "B", 2, 7)[0], A1)
    assert not np.array_equal(draw_layer("k", "A", 3, 7)[0], A1)
    assert not np.array_equal(draw_layer("other", "A", 2, 7)[0], A1)


def test_draw_layer_is_standard_normal():
    A, b = draw_layer("moments", "A", 1, 2000)
    sample = np.r_[A.ravel(), b]
    assert abs(sample.mean()) < 0.02
    assert abs(sample.std() - 1.0) < 0.02


def test_ridge_lstsq_matches_the_closed_form_and_uses_one_solve(monkeypatch):
    rng = np.random.default_rng(3)
    D = np.hstack([rng.normal(size=(300, 5)), np.ones((300, 1))])
    y = rng.normal(size=300)
    lam = 0.7
    calls = []
    real = np.linalg.lstsq

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", counting)
    theta = ridge_lstsq(D, y, lam)
    assert len(calls) == 1
    closed = np.linalg.solve(D.T @ D + lam * np.eye(6), D.T @ y)
    assert np.allclose(theta, closed, atol=1e-10)


def test_k1_slope1_zero_ridge_is_ordinary_least_squares_on_a_constructed_target():
    spec, paths, cache = make_cache(N=200, m=4, seed=2)
    fit_idx = np.arange(0, 160, 2)                       # fold A rows
    seed_key = "toy|0|200|0"
    A, b = draw_layer(seed_key, "A", 1, 1)
    x = mlp_batch(cache, 1, fit_idx).astype(float)
    y = 0.01 * (x @ A[:, 0] + b[0]) + 0.5                # exactly linear in h
    cfg = RLSMConfig(K=1, slope=1.0, ridge=0.0, min_rows=20)
    diag = []
    fn = rlsm_fit_fn(seed_key, cfg, diag)
    pred = fn(1, cache, fit_idx, y, np.zeros(0, np.int64), np.zeros(0), None)
    assert isinstance(pred, RLSMPredictor)
    assert np.max(np.abs(pred.raw(cache, 1, fit_idx) - y)) < 1e-8
    D = design(x, A, b, 1.0)
    theta_ols = np.linalg.lstsq(D, y, rcond=None)[0]
    assert np.allclose(pred.theta, theta_ols, atol=1e-8)


def test_normal_equation_residual_is_within_1e8_relative_with_ridge():
    spec, paths, cache = make_cache(N=300, m=4, seed=4)
    fit_idx = np.arange(0, 240, 2)
    y = np.clip(np.random.default_rng(9).uniform(0.0, 1.0, fit_idx.size), 0, 1.0)
    diag = []
    rlsm_fit_fn("ne|0|300|0", RLSMConfig(K=20, ridge=1e-6), diag)(
        1, cache, fit_idx, y, np.zeros(0, np.int64), np.zeros(0), None)
    assert diag[-1]["ne_residual_rel"] <= 1e-8


def test_ridge_penalty_is_lambda_times_n_fit():
    spec, paths, cache = make_cache(N=200, m=3, seed=6)
    fit_idx = np.arange(0, 120, 2)
    seed_key = "rg|0|200|0"
    cfg = RLSMConfig(K=4, ridge=1e-2, slope=0.5)
    y = np.random.default_rng(1).uniform(0.0, 1.0, fit_idx.size)
    pred = rlsm_fit_fn(seed_key, cfg, [])(1, cache, fit_idx, y, np.zeros(0, np.int64),
                                          np.zeros(0), None)
    A, b = draw_layer(seed_key, "A", 1, 4)
    D = design(mlp_batch(cache, 1, fit_idx).astype(float), A, b, 0.5)
    lam = 1e-2 * fit_idx.size
    closed = np.linalg.solve(D.T @ D + lam * np.eye(5), D.T @ y)
    assert np.allclose(pred.theta, closed, atol=1e-9)


def test_stage_features_do_not_read_later_stops():
    spec, paths, cache = make_cache(N=150, m=4, seed=8)
    later = paths.p.copy(), paths.d.copy()
    later[0][:, 2:] += 3.0
    later[1][:, 2:] -= 1.0
    cache2 = RouteCache(spec, Paths(p=later[0], d=later[1]), cache.stats)
    idx = np.arange(150)
    assert np.array_equal(mlp_batch(cache, 1, idx), mlp_batch(cache2, 1, idx))


def test_too_few_rows_give_the_constant_fallback():
    spec, paths, cache = make_cache(N=100, m=4, seed=10)
    y = np.linspace(0.1, 0.5, 10)
    pred = rlsm_fit_fn("c|0|100|0", RLSMConfig(), [])(
        2, cache, np.arange(0, 20, 2), y, np.zeros(0, np.int64), np.zeros(0), None)
    assert isinstance(pred, ConstantPredictor)
    assert pred.value == pytest.approx(y.mean())
    empty = rlsm_fit_fn("c|0|100|0", RLSMConfig(), [])(
        2, cache, np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64), np.zeros(0), None)
    assert empty.value == pytest.approx(spec.E[min(2, spec.m - 1)])


def test_exactly_min_rows_is_fitted_not_constant():
    spec, paths, cache = make_cache(N=100, m=4, seed=11)
    idx = np.arange(0, 40, 2)                            # 20 rows
    y = np.random.default_rng(2).uniform(0.0, 1.0, idx.size)
    pred = rlsm_fit_fn("b|0|100|0", RLSMConfig(min_rows=20), [])(
        1, cache, idx, y, np.zeros(0, np.int64), np.zeros(0), None)
    assert isinstance(pred, RLSMPredictor)


def test_predictions_are_clipped_to_zero_and_E0():
    spec, paths, cache = make_cache(N=200, m=4, seed=12)
    idx = np.arange(0, 160, 2)
    y = np.where(np.arange(idx.size) % 2 == 0, 0.0, spec.E0)
    pred = rlsm_fit_fn("clip|0|200|0", RLSMConfig(K=50), [])(
        1, cache, idx, y, np.zeros(0, np.int64), np.zeros(0), None)
    out = pred.predict(cache, 1, np.arange(200))
    assert np.all(out >= 0.0) and np.all(out <= spec.E0)
    assert np.any(pred.raw(cache, 1, np.arange(200)) != out)   # the clip is active


def test_targets_outside_zero_E0_are_refused():
    spec, paths, cache = make_cache(N=100, m=3, seed=13)
    with pytest.raises(ValueError):
        rlsm_fit_fn("bad|0|100|0", RLSMConfig(), [])(
            1, cache, np.arange(0, 60, 2), np.full(30, 2.0), np.zeros(0, np.int64),
            np.zeros(0), None)


def test_seeded_fits_are_reproducible():
    spec, paths, cache = make_cache(N=200, m=4, seed=14)
    y = np.random.default_rng(5).uniform(0.0, 1.0, 60)
    idx = np.arange(0, 120, 2)
    p1 = rlsm_fit_fn("r|0|200|0", RLSMConfig(), [])(1, cache, idx, y, np.zeros(0, np.int64),
                                                   np.zeros(0), None)
    p2 = rlsm_fit_fn("r|0|200|0", RLSMConfig(), [])(1, cache, idx, y, np.zeros(0, np.int64),
                                                   np.zeros(0), None)
    assert np.array_equal(p1.predict(cache, 1, np.arange(200)),
                          p2.predict(cache, 1, np.arange(200)))


def test_fit_rlsm_end_to_end_gives_two_predictors_per_epoch_and_a_budget():
    spec, paths, _ = make_cache(N=200, m=4, seed=15)
    models, stats, diag, budget = fit_rlsm(spec, paths, 200, 0, RLSMConfig(K=20))
    assert sorted(models) == [1, 2, 3]
    assert all(len(v) == 2 for v in models.values())
    assert isinstance(stats, RouteStats)
    assert budget["unique_paths"] == 200 and budget["path_stop_reads"] == 800
    assert budget["n_params"] > 0
    assert any(d.get("kind") == "stage" for d in diag)
    assert any(d.get("kind") == "ls" for d in diag)
    assert all("ne_residual_rel" in d for d in diag if d.get("kind") == "stage" and d["action"] == "fit")


def test_fit_rlsm_refuses_a_wrong_path_count():
    spec, paths, _ = make_cache(N=50, m=3, seed=16)
    with pytest.raises(ValueError, match="expected 60"):
        fit_rlsm(spec, paths, 60, 0, RLSMConfig(K=5))


def test_config_defaults_match_the_spec():
    cfg = RLSMConfig()
    assert (cfg.K, cfg.slope, cfg.ridge, cfg.min_rows) == (100, 0.5, 1e-6, 20)
    assert rlsm.MLP_DIM == MLP_DIM
    assert split_80_20(10)[0][0].size == 4

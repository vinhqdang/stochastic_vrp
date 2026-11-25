"""MC2: nested feature sets for Q1 (EXPERIMENT_C.md 1.3 and 9, MC2)."""
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from transvrp.expa.backward import ls_backward, split_80_20
from transvrp.expa.boost import BOOST_PARAMS, MIN_ROWS
from transvrp.expa.features import MLP_DIM, RouteCache, RouteStats, mlp_batch
from transvrp.expa.paths import Paths, RouteSpec
from transvrp.expa.train import ConstantPredictor
from transvrp.expb.rlsm import (RLSMConfig, RLSMPredictor, design, draw_layer, fit_rlsm,
                                rlsm_seed_key, ridge_lstsq)
from transvrp.expc import boost_sets, rlsm_sets
from transvrp.expc.boost_sets import BoostSetPredictor, boost_set_fit_fn, fit_boost_set
from transvrp.expc.featuresets import FEATURE_SETS, mask_columns, select_columns
from transvrp.expc.rlsm_sets import fit_rlsm_set, rlsm_set_fit_fn

G_HISTORY_COL = 7          # g8 = P_k / B, 0-based global index 7 (column 7 of mlp_batch)
SETS = ("F0", "F0n", "F1", "F2", "F3")


def make_spec(B, H, E, m):
    return RouteSpec(inst="TOY", route_id=0, route=tuple(range(1, m + 1)), m=m,
                     B=float(B), H=np.asarray(H, float), E=np.asarray(E, float),
                     E0=float(E[0]))


def make_cache(N=200, m=4, seed=1):
    # B = 1 so that routes breach often enough for the continuation targets to vary
    spec = make_spec(1.0, [0.3] * m, [1.0] * m, m)
    rng = np.random.default_rng(seed)
    paths = Paths(p=rng.uniform(0.5, 2.0, (N, m)), d=rng.uniform(0.5, 2.0, (N, m)))
    stats = RouteStats.fit(paths)
    return spec, paths, RouteCache(spec, paths, stats)


def perturbed_mlp(real, col=G_HISTORY_COL, delta=1.0):
    """Wrap mlp_batch so one column is shifted (a change in a single input value)."""
    def wrapper(cache, k, idx):
        x = np.array(real(cache, k, idx), copy=True)
        x[:, col] += delta
        return x
    return wrapper


# ------------------------------------------------------------ (a) index sets

def test_feature_sets_match_the_spec_indices():
    # Spec 1.3 (0-based globals; tokens are columns 12..14).
    assert FEATURE_SETS["F0"] == (0, 2, 3, 4, 5, 6, 12, 13, 14)
    assert FEATURE_SETS["F0n"] == (0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14)
    assert FEATURE_SETS["F1"] == (0, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14)
    assert FEATURE_SETS["F2"] == (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14)
    assert FEATURE_SETS["F3"] == tuple(range(15))


def test_sets_are_nested_and_f3_is_all_fifteen_columns():
    s = {k: set(v) for k, v in FEATURE_SETS.items()}
    assert s["F0"] < s["F0n"]
    assert s["F0"] < s["F1"] < s["F2"] < s["F3"]
    assert s["F3"] == set(range(MLP_DIM)) and MLP_DIM == 15
    assert all(s["F0"] <= s[name] for name in SETS)


def test_every_set_contains_the_token_summaries():
    for name in SETS:
        assert {12, 13, 14} <= set(FEATURE_SETS[name]), name


def test_unknown_set_name_is_refused():
    x = np.zeros((3, MLP_DIM), dtype=np.float32)
    with pytest.raises(ValueError):
        mask_columns(x, "F9")
    with pytest.raises(ValueError):
        select_columns(x, "F9")
    with pytest.raises(ValueError):
        boost_set_fit_fn("F9")
    with pytest.raises(ValueError):
        rlsm_set_fit_fn("seed", RLSMConfig(), [], "F9")


# ------------------------------------------------ (b) F3 is the unmasked batch

def test_f3_mask_and_select_equal_the_unmasked_mlp_batch():
    spec, paths, cache = make_cache(N=120, m=4, seed=3)
    idx = np.arange(120)
    for k in range(1, spec.m):
        x = mlp_batch(cache, k, idx)
        masked = mask_columns(x, "F3")
        assert np.array_equal(masked, x) and masked is not x
        assert np.array_equal(select_columns(x, "F3"), x)


def test_mask_keeps_shape_and_zeroes_exactly_the_outside_columns():
    x = np.arange(4 * MLP_DIM, dtype=np.float32).reshape(4, MLP_DIM) + 1.0
    before = x.copy()
    out = mask_columns(x, "F0")
    assert out.shape == x.shape and out.dtype == x.dtype
    keep = list(FEATURE_SETS["F0"])
    drop = [c for c in range(MLP_DIM) if c not in keep]
    assert np.array_equal(out[:, keep], x[:, keep])
    assert np.all(out[:, drop] == 0.0)
    assert np.array_equal(x, before)                      # input not modified


def test_select_returns_only_the_set_columns_in_index_order():
    x = np.arange(3 * MLP_DIM, dtype=np.float32).reshape(3, MLP_DIM)
    out = select_columns(x, "F1")
    assert out.shape == (3, len(FEATURE_SETS["F1"]))
    assert np.array_equal(out, x[:, list(FEATURE_SETS["F1"])])


# ------------------------------- (c) masked columns do not affect predictions

def test_boost_f0_predictions_ignore_the_masked_column_g8(monkeypatch):
    spec, paths, cache = make_cache(N=200, m=4, seed=4)
    models, _, _ = fit_boost_set(spec, paths, 200, 0, "F0")
    idx = np.arange(200)
    before = {k: [p.predict(cache, k, idx) for p in models[k]] for k in models}
    monkeypatch.setattr(boost_sets, "mlp_batch", perturbed_mlp(boost_sets.mlp_batch))
    for k in models:
        for p, ref in zip(models[k], before[k]):
            assert np.array_equal(p.predict(cache, k, idx), ref), k


def test_boost_f3_does_see_the_column_g8(monkeypatch):
    """Guard against a vacuous test: the same change moves the F3 predictions."""
    spec, paths, cache = make_cache(N=200, m=4, seed=4)
    models, _, _ = fit_boost_set(spec, paths, 200, 0, "F3")
    idx = np.arange(200)
    k = 1
    before = [p.predict(cache, k, idx) for p in models[k]]
    monkeypatch.setattr(boost_sets, "mlp_batch", perturbed_mlp(boost_sets.mlp_batch))
    after = [p.predict(cache, k, idx) for p in models[k]]
    assert any(not np.array_equal(a, b) for a, b in zip(after, before))


def test_rlsm_f0_predictions_ignore_the_masked_column_g8(monkeypatch):
    spec, paths, cache = make_cache(N=200, m=4, seed=5)
    cfg = RLSMConfig(K=20)
    models, _, _, _ = fit_rlsm_set(spec, paths, 200, 0, cfg, "F0")
    idx = np.arange(200)
    before = {k: [p.predict(cache, k, idx) for p in models[k]] for k in models}
    monkeypatch.setattr(rlsm_sets, "mlp_batch", perturbed_mlp(rlsm_sets.mlp_batch))
    for k in models:
        for p, ref in zip(models[k], before[k]):
            assert np.array_equal(p.predict(cache, k, idx), ref), k


def test_rlsm_f3_predictions_see_the_column_g8(monkeypatch):
    spec, paths, cache = make_cache(N=200, m=4, seed=5)
    models, _, _, _ = fit_rlsm_set(spec, paths, 200, 0, RLSMConfig(K=20), "F3")
    idx = np.arange(200)
    k = 1
    before = [p.predict(cache, k, idx) for p in models[k]]
    monkeypatch.setattr(rlsm_sets, "mlp_batch", perturbed_mlp(rlsm_sets.mlp_batch))
    after = [p.predict(cache, k, idx) for p in models[k]]
    assert any(not np.array_equal(a, b) for a, b in zip(after, before))


def test_rlsm_set_fit_applies_the_mask_at_fit_time():
    """Fit-side check: theta equals the ridge fit on the zero-masked 15-column inputs."""
    spec, paths, cache = make_cache(N=200, m=4, seed=6)
    cfg = RLSMConfig(K=15, slope=0.5, ridge=1e-3)
    seed = rlsm_seed_key(spec, 200, 0)
    fit_idx = np.arange(0, 160, 2)
    y = np.random.default_rng(7).uniform(0.0, 1.0, fit_idx.size)
    pred = rlsm_set_fit_fn(seed, cfg, [], "F0")(1, cache, fit_idx, y,
                                                np.zeros(0, np.int64), np.zeros(0), None)
    assert isinstance(pred, RLSMPredictor)
    A, b = draw_layer(seed, "A", 1, cfg.K)
    x = mask_columns(mlp_batch(cache, 1, fit_idx), "F0").astype(float)
    theta = ridge_lstsq(design(x, A, b, cfg.slope), y, cfg.ridge * fit_idx.size)
    assert np.allclose(pred.theta, theta, rtol=1e-10, atol=1e-10)


def test_boost_predictor_clips_to_zero_and_E0():
    class _Raw:
        def predict(self, X):
            return np.array([-1.0, 0.5, 99.0])[: len(X)]

    spec, _, cache = make_cache(N=60, m=3, seed=8)
    pred = BoostSetPredictor(_Raw(), "F0")
    out = pred.predict(cache, 1, np.arange(3))
    assert np.all(out >= 0.0) and np.all(out <= spec.E0)
    assert out[0] == 0.0 and out[2] == spec.E0


# ------------------------------------------------ (d) random layer is shared

def test_rlsm_random_layer_is_identical_for_f0_and_f3():
    spec, paths, _ = make_cache(N=200, m=4, seed=9)
    cfg = RLSMConfig(K=20)
    m0, _, _, _ = fit_rlsm_set(spec, paths, 200, 0, cfg, "F0")
    m3, _, _, _ = fit_rlsm_set(spec, paths, 200, 0, cfg, "F3")
    seed = rlsm_seed_key(spec, 200, 0)
    compared = 0
    for k in m0:
        for net, (p0, p3) in enumerate(zip(m0[k], m3[k])):
            if not (isinstance(p0, RLSMPredictor) and isinstance(p3, RLSMPredictor)):
                continue
            fold = "A" if net == 0 else "B"
            A, b = draw_layer(seed, fold, k, cfg.K)
            assert np.array_equal(p0.A, p3.A) and np.array_equal(p0.b, p3.b)
            assert np.array_equal(p0.A, A) and np.array_equal(p0.b, b)
            compared += 1
    assert compared >= 2


# ------------------------------------------------- (e) F3 equals the unmasked fits

def _direct_boost_fit_fn():
    """Boosting on the 15 mlp columns, written without the expc selection code."""
    def fit(k, cache, fit_idx, y_fit, val_idx, y_val, warm):
        m, E = cache.m, cache.spec.E
        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        y_fit = np.asarray(y_fit, dtype=float)
        n = fit_idx.size
        if n < MIN_ROWS:
            return ConstantPredictor(float(y_fit.mean()) if n else float(E[min(k, m - 1)]))
        X = np.asarray(mlp_batch(cache, k, fit_idx), dtype=np.float64)
        model = HistGradientBoostingRegressor(**BOOST_PARAMS).fit(X, y_fit)

        class _Direct:
            def predict(self, cache, k, idx):
                Z = np.asarray(mlp_batch(cache, k, np.asarray(idx, dtype=np.int64)),
                               dtype=np.float64)
                return np.clip(model.predict(Z), 0.0, cache.spec.E0)

        return _Direct()
    return fit


def test_fit_boost_set_f3_equals_a_direct_fit_on_the_15_columns():
    N = 200
    spec, paths, _ = make_cache(N=N, m=4, seed=10)
    models, stats, _ = fit_boost_set(spec, paths, N, 0, "F3")
    folds, val = split_80_20(N)
    fit_all = np.concatenate(folds)
    s2 = RouteStats.fit(Paths(p=paths.p[fit_all], d=paths.d[fit_all]))
    assert np.allclose(stats.mu_p, s2.mu_p, rtol=1e-12, atol=0)
    cache = RouteCache(spec, paths, s2)
    direct = ls_backward(cache, spec.B, spec.H, spec.E, _direct_boost_fit_fn(), folds, val)
    idx = np.arange(N)
    assert sorted(models) == sorted(direct)
    for k in models:
        for p, q in zip(models[k], direct[k]):
            assert np.allclose(p.predict(cache, k, idx), q.predict(cache, k, idx),
                               rtol=1e-12, atol=1e-12), k


def test_fit_rlsm_set_f3_equals_the_unmasked_fit_rlsm():
    N = 200
    spec, paths, cache = make_cache(N=N, m=4, seed=11)
    cfg = RLSMConfig(K=20)
    m3, _, _, _ = fit_rlsm_set(spec, paths, N, 0, cfg, "F3")
    mu, _, _, _ = fit_rlsm(spec, paths, N, 0, cfg)
    idx = np.arange(N)
    assert sorted(m3) == sorted(mu)
    for k in m3:
        for p, q in zip(m3[k], mu[k]):
            assert np.allclose(p.predict(cache, k, idx), q.predict(cache, k, idx),
                               rtol=1e-12, atol=1e-12), k


# --------------------------------------------------- (f) finite and in [0, E0]

@pytest.mark.parametrize("set_name", ["F0", "F3"])
def test_predictions_are_finite_and_within_zero_and_E0(set_name):
    N = 200
    spec, paths, _ = make_cache(N=N, m=4, seed=12)
    boost, stats_b, _ = fit_boost_set(spec, paths, N, 0, set_name)
    rl, stats_r, _, _ = fit_rlsm_set(spec, paths, N, 0, RLSMConfig(K=20), set_name)
    cache_b = RouteCache(spec, paths, stats_b)
    cache_r = RouteCache(spec, paths, stats_r)
    idx = np.arange(N)
    for models, cache in ((boost, cache_b), (rl, cache_r)):
        for k in models:
            for p in models[k]:
                out = p.predict(cache, k, idx)
                assert np.all(np.isfinite(out))
                assert np.all(out >= 0.0) and np.all(out <= spec.E0)


# ------------------------------------------------------------ (f) picklable set predictors

def test_rlsm_set_predictor_pickles_and_predicts_bit_identically_after_reload():
    import pickle
    spec, paths, cache = make_cache(N=200, m=4, seed=9)
    A, b = draw_layer("pickle-key", "fold", 1, 20)
    theta = np.random.default_rng(3).standard_normal(21)
    pred = RLSMPredictor(A, b, theta, 0.5, spec.E0,
                         featurize=rlsm_sets.masked_featurize("F1"))
    back = pickle.loads(pickle.dumps(pred))
    idx = np.arange(200)
    for k in range(1, spec.m):
        assert np.array_equal(back.predict(cache, k, idx), pred.predict(cache, k, idx))


def test_masked_featurize_is_a_module_level_callable_with_the_set_name():
    from transvrp.expa.features import mlp_batch as real
    fz = rlsm_sets.masked_featurize("F0")
    assert fz.set_name == "F0"
    assert type(fz).__module__ == rlsm_sets.__name__
    spec, paths, cache = make_cache(N=50, m=4, seed=2)
    assert np.array_equal(fz(cache, 2, np.arange(50)),
                          mask_columns(real(cache, 2, np.arange(50)), "F0"))


def test_predictor_pickled_without_the_featurize_attribute_uses_mlp_batch():
    """Predictors written before the hook existed carry no ``featurize``: they predict with
    B's unmodified mlp_batch (expb.rlsm getattr fallback)."""
    import pickle
    spec, paths, cache = make_cache(N=120, m=4, seed=4)
    A, b = draw_layer("old-key", "fold", 2, 20)
    theta = np.random.default_rng(5).standard_normal(21)
    new = RLSMPredictor(A, b, theta, 0.5, spec.E0)
    old = pickle.loads(pickle.dumps(new))
    old.__dict__.pop("featurize", None)
    idx = np.arange(120)
    assert np.array_equal(old.predict(cache, 2, idx), new.predict(cache, 2, idx))


def test_codec_round_trip_keeps_the_feature_set_mask_of_an_rlsm_predictor():
    """The payload codec must rebuild a set's RLSM predictor with the same input mask it
    was fitted with (otherwise F0/F0n/F1/F2 are evaluated on unmasked inputs)."""
    from transvrp.expb import codecs
    spec, paths, cache = make_cache(N=160, m=4, seed=12)
    A, b = draw_layer("codec-key", "fold", 1, 20)
    theta = np.random.default_rng(8).standard_normal(21)
    pred = RLSMPredictor(A, b, theta, 0.5, spec.E0,
                         featurize=rlsm_sets.masked_featurize("F1"))
    back = codecs.unpack_models(codecs.pack_models({1: [pred]}, "rlsm", spec.E0))[1][0]
    idx = np.arange(160)
    assert np.array_equal(back.predict(cache, 1, idx), pred.predict(cache, 1, idx))
    plain = RLSMPredictor(A, b, theta, 0.5, spec.E0)
    back_plain = codecs.unpack_models(codecs.pack_models({1: [plain]}, "rlsm", spec.E0))[1][0]
    assert np.array_equal(back_plain.predict(cache, 1, idx), plain.predict(cache, 1, idx))

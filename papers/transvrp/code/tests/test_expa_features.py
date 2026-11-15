"""M2: state features (EXPERIMENT_A.md section 3)."""
import numpy as np
import pytest

from transvrp.expa.features import (G, F, HISTORY_IDX, RouteCache, RouteStats,
                                    mlp_batch, stage_batch)
from transvrp.expa.paths import Paths, RouteSpec, route_pool, route_specs

TOY_SPEC = RouteSpec(inst="toy", route_id=0, route=(1, 2, 3), m=3, B=10.0,
                     H=np.array([4.0, 3.0, 2.0]), E=np.array([8.0, 6.0, 5.0]),
                     E0=8.0)
TOY_STATS = RouteStats(mu_p=np.array([2.0, 3.0, 4.0]),
                       mu_d=np.array([1.0, 2.0, 1.0]),
                       sd_p=np.array([1.0, 2.0, 0.0]),
                       sd_d=np.array([0.5, 1.0, 2.0]))
TOY_P = np.array([[3.0, 4.0, 1.0]])
TOY_D = np.array([[1.0, 2.0, 2.0]])


def _toy_cache(p=TOY_P, d=TOY_D):
    return RouteCache(TOY_SPEC, Paths(p=p.copy(), d=d.copy()), TOY_STATS)


def test_shapes_dtype_and_range_on_real_route():
    spec_list = route_specs("CON3-0")
    pool = route_pool("CON3-0", "test_std")
    spec = spec_list[0]
    stats = RouteStats.fit(pool[spec.route_id])
    cache = RouteCache(spec, pool[spec.route_id], stats)
    idx = np.arange(200)
    for k in range(1, spec.m):
        for arm in ("full", "scalar"):
            glob, tok = stage_batch(cache, k, idx, arm)
            assert glob.shape == (200, G) and glob.dtype == np.float32
            assert tok.shape == (200, spec.m - k, F) and tok.dtype == np.float32
            assert np.all(np.abs(glob) <= 5) and np.all(np.abs(tok) <= 5)
        mb = mlp_batch(cache, k, idx)
        assert mb.shape == (200, 15) and mb.dtype == np.float32
        assert np.all(np.abs(mb) <= 5)


def test_no_peeking_after_stage_k():
    spec = route_specs("CON3-0")[3]
    pool = route_pool("CON3-0", "test_std")
    paths = pool[spec.route_id]
    stats = RouteStats.fit(paths)
    base = RouteCache(spec, paths, stats)
    idx = np.arange(300)
    for k in range(1, spec.m):
        p2 = paths.p.copy()
        d2 = paths.d.copy()
        p2[:, k:] = p2[:, k:] * 1.7 + 3.0           # stops k+1..m (0-based k..)
        d2[:, k:] = d2[:, k:] * 0.4
        alt = RouteCache(spec, Paths(p=p2, d=d2), stats)
        for arm in ("full", "scalar"):
            g1, t1 = stage_batch(base, k, idx, arm)
            g2, t2 = stage_batch(alt, k, idx, arm)
            assert np.array_equal(g1, g2) and np.array_equal(t1, t2)
        assert np.array_equal(mlp_batch(base, k, idx), mlp_batch(alt, k, idx))


def test_scalar_ignores_history_full_does_not():
    p = np.vstack([TOY_P, TOY_P + np.array([[1.0, 0.0, 0.0]])])
    d = np.vstack([TOY_D, TOY_D + np.array([[1.0, 0.0, 0.0]])])
    cache = _toy_cache(p, d)
    assert cache.W[0, 0] == cache.W[1, 0]          # equal running sum at k = 1
    idx = np.array([0, 1])
    gf, tf = stage_batch(cache, 1, idx, "full")
    gs, ts = stage_batch(cache, 1, idx, "scalar")
    assert np.array_equal(ts[0], ts[1])
    assert np.array_equal(gs[0], gs[1])
    assert not np.array_equal(gf[0], gf[1])
    assert np.all(gs[:, list(HISTORY_IDX)] == 0)


def test_hand_computed_toy_route_stage1():
    cache = _toy_cache()
    glob, tok = stage_batch(cache, 1, np.array([0]), "full")
    g = glob[0].astype(np.float64)
    expect = {0: 0.2, 1: 0.2, 2: 1 / 3, 3: 2 / 3, 4: 3 / 22, 5: 0.5, 6: 1.0,
              7: 0.3, 8: 0.1, 9: 1.0, 10: 0.0, 11: 0.2}
    for i, v in expect.items():
        assert abs(g[i] - v) < 1e-6, i
    t = tok[0].astype(np.float64)                  # tokens j = 2, 3
    assert t.shape == (2, F)
    exp_j2 = [0.3, 0.2, 0.2, 0.1, 1 / 3, 2 / 3, 0.375, 0.75, 0.0, 0.3,
              7 / np.sqrt(5)]
    exp_j3 = [0.4, 0.1, 0.0, 0.2, 2 / 3, 1.0, 0.0, 0.625, 1.0, 0.6, 4 / 3]
    assert np.max(np.abs(t[0] - np.array(exp_j2))) < 1e-6
    assert np.max(np.abs(t[1] - np.array(exp_j3))) < 1e-6


def test_hand_computed_toy_route_stage2_and_mlp():
    cache = _toy_cache()
    idx = np.array([0])
    glob, tok = stage_batch(cache, 2, idx, "full")
    g = glob[0].astype(np.float64)
    expect = {0: 0.4, 1: 0.4, 2: 2 / 3, 3: 1 / 3, 5: 0.375, 6: 0.75, 7: 0.7,
              8: 0.3, 9: 0.75, 10: 0.0, 11: 0.2}
    for i, v in expect.items():
        assert abs(g[i] - v) < 1e-6, i
    t = tok[0].astype(np.float64)                  # token j = 3 only
    exp_j3 = [0.4, 0.1, 0.0, 0.2, 1 / 3, 1.0, 0.0, 0.625, 1.0, 0.7, 1.5]
    assert np.max(np.abs(t[0] - np.array(exp_j3))) < 1e-6
    mb = mlp_batch(cache, 2, idx)[0].astype(np.float64)
    assert mb.shape == (15,)
    assert abs(mb[12] - 1.5) < 1e-6                # min t11 (only token)
    assert abs(mb[13] - 1 / 3) < 1e-6              # (j* - k)/m = (3-2)/3
    assert mb[14] == 0.0                           # no j with k < j < m


def test_mlp_summaries_stage1():
    cache = _toy_cache()
    mb = mlp_batch(cache, 1, np.array([0]))[0].astype(np.float64)
    assert abs(mb[12] - 4 / 3) < 1e-6              # min(7/sqrt5, 4/3)
    assert abs(mb[13] - 2 / 3) < 1e-6              # argmin is j = 3
    assert abs(mb[14] - 0.375) < 1e-6              # H[1]/E0 for j = 2


def test_route_stats_use_only_rows_passed():
    spec = route_specs("CON3-0")[0]
    paths = route_pool("CON3-0", "test_std")[spec.route_id]
    part = Paths(p=paths.p[:500], d=paths.d[:500])
    st = RouteStats.fit(part)
    assert np.allclose(st.mu_p, paths.p[:500].mean(axis=0))
    assert np.allclose(st.sd_d, paths.d[:500].std(axis=0, ddof=1))
    st_full = RouteStats.fit(paths)
    assert not np.allclose(st.mu_p, st_full.mu_p)

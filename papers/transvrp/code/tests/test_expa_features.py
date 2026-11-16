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


# Spec section 3.2, written out by hand (0-based columns of the 12 globals):
# history globals g2 (peak), g8 (P_k), g9 (D_k), g10 (zbar_p), g11 (zbar_d),
# g12 (p_k - d_k) are zeroed in SCALAR; g1, g3..g7 are kept.
SCALAR_ZEROED = [1, 7, 8, 9, 10, 11]
SCALAR_KEPT = [0, 2, 3, 4, 5, 6]


def _pd_rows(rows):
    """rows: list of (p, d) lists of length 3 -> cache on the toy route."""
    p = np.array([r[0] for r in rows], dtype=float)
    d = np.array([r[1] for r in rows], dtype=float)
    return _toy_cache(p, d)


def test_scalar_zeroes_the_spec_history_columns_and_nothing_else():
    p = np.array([[3.0, 1.0, 1.0], [2.0, 2.0, 1.0]])
    d = np.array([[1.0, 2.0, 2.0], [1.0, 1.0, 2.0]])
    cache = _toy_cache(p, d)
    idx = np.arange(2)
    for k in (1, 2):
        gf, _ = stage_batch(cache, k, idx, "full")
        gs, _ = stage_batch(cache, k, idx, "scalar")
        assert np.all(gs[:, SCALAR_ZEROED] == 0.0)
        assert np.array_equal(gs[:, SCALAR_KEPT], gf[:, SCALAR_KEPT])
        assert not np.array_equal(gf[:, SCALAR_ZEROED], gs[:, SCALAR_ZEROED])


def test_scalar_blind_to_split_of_equal_running_sum_at_stage_1():
    """Equal W_1 and equal peak (= W_1), different pickup/delivery split."""
    cache = _pd_rows([([3.0, 1.0, 1.0], [1.0, 2.0, 2.0]),     # p1 = 3, d1 = 1
                      ([2.0, 2.0, 1.0], [0.0, 2.0, 2.0])])    # p1 = 2, d1 = 0
    assert cache.W[0, 0] == cache.W[1, 0] == 2.0
    assert cache.P[0, 0] != cache.P[1, 0] and cache.D[0, 0] != cache.D[1, 0]
    idx = np.arange(2)
    gf, tf = stage_batch(cache, 1, idx, "full")
    gs, ts = stage_batch(cache, 1, idx, "scalar")
    assert np.array_equal(gs[0], gs[1]) and np.array_equal(ts[0], ts[1])
    assert not np.array_equal(gf[0], gf[1])


def test_scalar_blind_to_running_peak_at_stage_2():
    """Equal W_2 = 1, but the running peak is 2 in row 0 and 1 in row 1, and
    P_2, D_2 differ. SCALAR must not see the peak; FULL must."""
    cache = _pd_rows([([3.0, 1.0, 1.0], [1.0, 2.0, 2.0]),     # g = (2, -1, -1)
                      ([3.0, 2.0, 1.0], [2.0, 2.0, 2.0])])    # g = (1, 0, -1)
    assert cache.W[0, 1] == cache.W[1, 1] == 1.0
    assert cache.peak[0, 1] == 2.0 and cache.peak[1, 1] == 1.0
    assert cache.P[0, 1] != cache.P[1, 1] and cache.D[0, 1] != cache.D[1, 1]
    idx = np.arange(2)
    gf, tf = stage_batch(cache, 2, idx, "full")
    gs, ts = stage_batch(cache, 2, idx, "scalar")
    assert np.array_equal(gs[0], gs[1]) and np.array_equal(ts[0], ts[1])
    assert gf[0, 1] != gf[1, 1]                              # peak differs in FULL
    assert not np.array_equal(gf[0], gf[1])


def test_scalar_output_is_a_function_of_k_and_W_only():
    """Random routes with exactly representable values; every row shares W_k,
    so SCALAR outputs must be identical for every row at that stage."""
    rng = np.random.default_rng(7)
    n = 60
    for k in (1, 2):
        g = rng.integers(-16, 17, size=(n, 3)) / 8.0            # exact binary values
        g[:, k - 1] = 1.0 - g[:, :k - 1].sum(axis=1)            # W_k = 1 for all rows
        p = rng.integers(8, 40, size=(n, 3)) / 8.0
        d = p - g
        cache = _toy_cache(p, d)
        assert np.all(cache.W[:, k - 1] == 1.0)
        idx = np.arange(n)
        gs, ts = stage_batch(cache, k, idx, "scalar")
        gf, _ = stage_batch(cache, k, idx, "full")
        assert np.array_equal(gs, np.repeat(gs[:1], n, axis=0))
        assert np.array_equal(ts, np.repeat(ts[:1], n, axis=0))
        assert len(np.unique(gf[:, 7])) > 1                     # FULL sees P_k
        if k == 2:
            assert len(np.unique(gf[:, 1])) > 1                 # FULL sees the peak


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

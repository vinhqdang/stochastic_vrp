"""M1 and M2 memory and cache behaviour (EXPERIMENT_A.md section 5 and section 7.2
memory note): the on-disk chunk cache must reproduce the uncached path bit for bit,
keys must separate replicates, routes can be restricted, feature batches do not depend
on the block size, and the measurement helpers see real allocations."""
import numpy as np
import pytest

from transvrp.expa import paths
from transvrp.expa.features import RouteCache, RouteStats, mlp_batch, stage_batch
from transvrp.expa.measure import Stopwatch, process_peak_rss_mb, traced_peak
from transvrp.expa.paths import Paths, chunk_file, route_pool, route_specs

INST = "CON3-0"


def _assert_pools_equal(a, b):
    assert set(a) == set(b)
    for rid in a:
        assert np.array_equal(a[rid].p, b[rid].p)
        assert np.array_equal(a[rid].d, b[rid].d)
        assert a[rid].p.dtype == np.float64


def test_cached_chunks_are_bit_identical_to_generated_ones(tmp_path):
    plain = route_pool(INST, "train", rep=1, N=2000)
    first = route_pool(INST, "train", rep=1, N=2000, cache_dir=tmp_path)   # writes
    again = route_pool(INST, "train", rep=1, N=2000, cache_dir=tmp_path)   # reads
    _assert_pools_equal(plain, first)
    _assert_pools_equal(plain, again)
    assert any(tmp_path.glob("*.npy"))


def test_cache_keys_separate_replicates(tmp_path):
    # if the replicate were missing from the key, rep 1 would read rep 0's chunk
    r0 = route_pool(INST, "train", rep=0, N=1000, cache_dir=tmp_path)
    r1 = route_pool(INST, "train", rep=1, N=1000, cache_dir=tmp_path)
    _assert_pools_equal(route_pool(INST, "train", rep=0, N=1000), r0)
    _assert_pools_equal(route_pool(INST, "train", rep=1, N=1000), r1)
    rid = next(iter(r0))
    assert not np.array_equal(r0[rid].p, r1[rid].p)
    names = {p.name for p in tmp_path.glob("*.npy")}
    assert len(names) == 2


def test_chunk_is_reused_across_routes_and_calls(tmp_path, monkeypatch):
    calls = []
    real = paths._generate_chunk

    def counting(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(paths, "_generate_chunk", counting)
    route_pool(INST, "train", rep=0, N=1000, cache_dir=tmp_path)
    assert len(calls) == 1                          # generated once
    one_route = route_pool(INST, "train", rep=0, N=1000, cache_dir=tmp_path, routes=[3])
    assert len(calls) == 1                          # read from disk for the next route
    assert set(one_route) == {3}


def test_chunk_file_key_contains_all_five_parts(tmp_path):
    a = chunk_file(tmp_path, INST, "train", 0, 0, [1, 2, 3], 100000)
    b = chunk_file(tmp_path, INST, "train", 1, 0, [1, 2, 3], 100000)
    c = chunk_file(tmp_path, INST, "train", 0, 0, [1, 2, 4], 100000)
    d = chunk_file(tmp_path, INST, "train", 0, 1, [1, 2, 3], 100000)
    assert len({a, b, c, d}) == 4
    assert a.parent == tmp_path and a.suffix == ".npy"


def test_routes_restriction_returns_only_those_routes_and_matches_full_pool():
    full = route_pool(INST, "dev")
    only = route_pool(INST, "dev", routes=[3])
    assert set(only) == {3}
    assert np.array_equal(only[3].p, full[3].p)
    assert only[3].p.shape == (5000, full[3].p.shape[1])


def test_unknown_route_is_refused():
    with pytest.raises(ValueError):
        route_pool(INST, "dev", routes=[999])


def test_dev_pool_ignores_replicate_in_the_cache_key(tmp_path):
    a = route_pool(INST, "dev", rep=0, cache_dir=tmp_path)
    b = route_pool(INST, "dev", rep=2, cache_dir=tmp_path)
    _assert_pools_equal(a, b)
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_stage_batch_does_not_depend_on_block_size():
    pool = route_pool(INST, "test_std")
    spec = {s.route_id: s for s in route_specs(INST)}[3]
    p = pool[3]
    stats = RouteStats.fit(p)
    cache = RouteCache(spec, p, stats)
    idx = np.arange(0, 2000, 3, dtype=np.int64)
    for k in (1, 5, spec.m - 1):
        for arm in ("full", "scalar"):
            g1, t1 = stage_batch(cache, k, idx, arm)
            g2, t2 = stage_batch(cache, k, idx, arm, block_rows=7)
            assert g1.dtype == np.float32 and t1.dtype == np.float32
            assert np.array_equal(g1, g2) and np.array_equal(t1, t2)
        assert np.array_equal(mlp_batch(cache, k, idx),
                              mlp_batch(cache, k, idx, block_rows=5))


def test_route_cache_keeps_no_reference_to_the_paths_and_g_is_p_minus_d():
    pool = route_pool(INST, "test_std", routes=[3])[3]
    spec = {s.route_id: s for s in route_specs(INST)}[3]
    cache = RouteCache(spec, pool, RouteStats.fit(pool))
    assert not hasattr(cache, "paths")
    assert np.array_equal(cache.g, pool.p - pool.d)
    assert cache.zsum_p.dtype == np.float32 and cache.W.dtype == np.float64


def test_traced_peak_sees_a_numpy_allocation():
    with traced_peak() as box:
        buf = np.ones(4 * 2 ** 20 // 8)          # 4 MiB of float64
        buf.sum()
    assert box["peak_mb"] >= 3.9
    assert box["retained_mb"] >= 3.9             # buf is still alive at exit
    del buf


def test_traced_peak_is_zero_for_no_allocation():
    with traced_peak() as box:
        x = 1 + 1
    assert x == 2 and box["peak_mb"] < 0.5


def test_process_rss_and_stopwatch():
    assert process_peak_rss_mb() > 0
    sw = Stopwatch()
    with sw.lap("a"):
        pass
    with sw.lap("a"):
        pass
    assert set(sw.seconds) == {"a"} and sw.seconds["a"] >= 0.0


# ---------------------------------------------------------------- fold-disagreement diagnostic

class _Const:
    def __init__(self, v):
        self.v = v
        self.sizes = []

    def predict(self, cache, k, idx):
        self.sizes.append(len(idx))
        return np.full(len(idx), self.v)


def test_disagreement_hand_values_and_subsample_bound():
    from transvrp.expa.backward import DISAGREE_ROWS, _disagreement

    cache = None                      # the stubs ignore the cache
    rows = np.arange(100, dtype=np.int64)
    assert _disagreement(_Const(1.0), _Const(0.0), cache, 1, rows, 0.5) == 1.0
    assert _disagreement(_Const(1.0), _Const(0.9), cache, 1, rows, 0.5) == 0.0
    assert _disagreement(_Const(0.2), _Const(0.9), cache, 1, np.zeros(0, np.int64), 0.5) == 0.0
    big = np.arange(3 * DISAGREE_ROWS, dtype=np.int64)
    a, b = _Const(1.0), _Const(0.0)
    assert _disagreement(a, b, cache, 1, big, 0.5) == 1.0
    assert max(a.sizes) <= DISAGREE_ROWS and max(b.sizes) <= DISAGREE_ROWS


def test_fit_boost_refuses_costs_above_E0():
    from transvrp.expa.boost import fit_boost
    from transvrp.expa.paths import RouteSpec

    spec = RouteSpec(inst="T", route_id=0, route=(0, 1), m=2, B=5.0,
                     H=np.array([9.0, 1.0]), E=np.array([4.0, 4.0]), E0=4.0)
    rng = np.random.default_rng(0)
    pool = Paths(p=rng.uniform(0, 3, (100, 2)), d=rng.uniform(0, 3, (100, 2)))
    with pytest.raises(ValueError, match="E0"):
        fit_boost(spec, pool, 100, 0)

"""M1: route pools and route specs."""
import numpy as np
import pytest

from transvrp.data import route_problem
from transvrp.expa.paths import route_pool, route_specs


def test_test_std_matches_route_problem_exactly():
    specs = {s.route_id: s for s in route_specs("CON3-0")}
    pool = route_pool("CON3-0", "test_std")
    seen = 0
    for p in route_problem("CON3-0", n_train=500, n_test=2000):
        rid = p["route_id"]
        assert set(pool) >= {rid}
        assert np.array_equal(pool[rid].g, p["g_test"])
        assert pool[rid].g.dtype == np.float64
        assert specs[rid].B == p["B"]
        assert np.array_equal(specs[rid].H, p["H"])
        assert np.array_equal(specs[rid].E, p["E"])
        assert specs[rid].m == len(p["route"])
        assert specs[rid].route == tuple(p["route"])
        assert specs[rid].E0 == float(p["E"][0])
        seen += 1
    assert seen == len(specs) == 4


def test_route_specs_skip_empty_and_keep_order():
    specs = route_specs("CON3-0")
    ids = [s.route_id for s in specs]
    assert ids == sorted(ids)
    assert all(s.m >= 1 for s in specs)


@pytest.mark.slow
def test_train_prefix_is_nested_across_N():
    small = route_pool("CON3-0", "train", rep=0, N=100_000)
    big = route_pool("CON3-0", "train", rep=0, N=1_000_000)
    for rid in small:
        assert np.array_equal(big[rid].p[:100_000], small[rid].p)
        assert np.array_equal(big[rid].d[:100_000], small[rid].d)


def test_replicates_and_dev_differ():
    r0 = route_pool("CON3-0", "train", rep=0, N=1000)
    r1 = route_pool("CON3-0", "train", rep=1, N=1000)
    dev = route_pool("CON3-0", "dev")
    rid = next(iter(r0))
    assert not np.array_equal(r0[rid].p, r1[rid].p)
    assert not np.array_equal(r0[rid].p, dev[rid].p[:1000])
    assert dev[rid].p.shape[0] == 5000


def test_fixed_size_purposes_reject_other_N():
    with pytest.raises(ValueError):
        route_pool("CON3-0", "test_std", N=500)

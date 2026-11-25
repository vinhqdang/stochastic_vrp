"""C module MC1 (EXPERIMENT_C.md 1.4, 3, 5, 9): world-aware generator, world seeds,
the test_c gate and the chunk cache key.

The W6 generator and its seeds must reproduce A's and B's streams exactly. Test-stream
values (test, test_b, test_c) are never written anywhere: the tests only compare first
rows in memory, and compute them through the seed functions with an in-memory token."""
import json
import zlib

import numpy as np
import pytest
import scipy.stats as st

from transvrp.data import (CV, DIST, RHO, WORLD_SEED_CODE, WORLDS,
                           gen_scenarios, load_plans, sample_demands, stable_seed)
from transvrp.expa import paths
from transvrp.expa.paths import (CHUNK, GateToken, TestStreamGateError, chunk_file,
                                 route_pool, route_specs)

INST = "CON3-0"
N_SMALL = 4          # first rows used for the distinctness comparison
N_CORR = 20_000      # rows used for the correlation check


def _plan(inst=INST):
    sol = load_plans(inst)
    return sol["dbar"], sol["pbar"]


def _reference(dbar, pbar, N, seed):
    """The original generator, written out here (separate copulas, rho for both,
    deliveries drawn before pickups). Independent of data.gen_scenarios."""
    rng = np.random.default_rng(seed)
    dsc = sample_demands(dbar, len(dbar), N, CV, DIST, rng, rho=RHO)
    psc = sample_demands(pbar, len(pbar), N, CV, DIST, rng, rho=RHO)
    return dsc, psc


def _token(stream, stage=None):
    return GateToken(commit="a" * 40, stream=stream, stage=stage,
                     _issuer=paths._GATE_ISSUER)


def _stream_seed(world_code, purpose, rep=0):
    """Seed of a stream through the seed functions (test purposes get an in-memory token)."""
    if purpose in ("test", "test_b", "test_c"):
        gate = _token(purpose, "b_thr")
        return paths._stream(INST, purpose, 0, 0, gate, world_code)[0]
    if purpose == "train":
        return paths._stream(INST, purpose, rep, 0, None, world_code)[0]
    return paths._stream(INST, purpose, 0, 0, None, world_code)[0]


def _first_rows(world_code, purpose, rep=0):
    """First N_SMALL rows (deliveries then pickups) of one stream, in memory only."""
    seed = _stream_seed(world_code, purpose, rep)
    dbar, pbar = _plan()
    dsc, psc = gen_scenarios(dbar, pbar, N_SMALL, seed, world=WORLDS[world_code])
    return np.concatenate([dsc.ravel(), psc.ravel()])


# ---------------------------------------------------------------- (a) W6 bit identity

def test_world_none_and_w6_match_the_original_generator_bit_for_bit():
    dbar, pbar = _plan()
    seed = [stable_seed(INST), 1, 0, 0]
    ref_d, ref_p = _reference(dbar, pbar, 300, seed)
    d_none, p_none = gen_scenarios(dbar, pbar, 300, seed)
    d_w6, p_w6 = gen_scenarios(dbar, pbar, 300, seed, world=WORLDS["W6"])
    d_code, p_code = gen_scenarios(dbar, pbar, 300, seed, world="W6")
    for d, p in ((d_none, p_none), (d_w6, p_w6), (d_code, p_code)):
        assert np.array_equal(d, ref_d) and d.dtype == ref_d.dtype
        assert np.array_equal(p, ref_p) and p.dtype == ref_p.dtype


def test_default_rho_argument_is_still_the_generator_default():
    dbar, pbar = _plan()
    seed = 12345
    a = gen_scenarios(dbar, pbar, 50, seed)
    b = gen_scenarios(dbar, pbar, 50, seed, rho=RHO, world=None)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ---------------------------------------------------------------- (b) route pools of W6

def test_w6_route_pools_equal_the_original_seeds_and_default_call():
    base = stable_seed(INST)
    dbar, pbar = _plan()
    spec = route_specs(INST)[0]
    rid = spec.route_id
    route = np.array(spec.route)

    train = route_pool(INST, "train", rep=0, N=200)
    train_w6 = route_pool(INST, "train", rep=0, N=200, world="W6")
    # a train pool of N <= 10^5 is the first N rows of the full chunk 0: pickups are drawn
    # after the N*m deliveries, so the reference must be the full chunk, then sliced
    ref_d, ref_p = _reference(dbar, pbar, CHUNK, [base, 1, 0, 0])
    assert np.array_equal(train[rid].p, ref_p[:200][:, route])
    assert np.array_equal(train[rid].d, ref_d[:200][:, route])
    assert np.array_equal(train_w6[rid].p, train[rid].p)
    assert np.array_equal(train_w6[rid].d, train[rid].d)

    dev = route_pool(INST, "dev")
    dev_w6 = route_pool(INST, "dev", world="W6")
    ref_d, ref_p = _reference(dbar, pbar, 5000, [base, 3])
    assert np.array_equal(dev[rid].p, ref_p[:, route])
    assert np.array_equal(dev[rid].d, ref_d[:, route])
    assert np.array_equal(dev_w6[rid].p, dev[rid].p)


# ---------------------------------------------------------------- (c) distinct streams

def test_c_streams_are_distinct_from_each_other_and_from_a_and_b_draws():
    c_streams = [("train", w) for w in ("W0", "W3", "W9", "WS")]
    c_streams += [("dev", w) for w in ("W0", "W3", "W9", "WS")]
    c_streams += [("test_c", w) for w in WORLDS]
    ab_streams = [("train", "W6", 0), ("train", "W6", 1), ("dev", "W6", 0),
                  ("test", "W6", 0), ("test_b", "W6", 0)]

    flats = {}
    for purpose, w in c_streams:
        flats[f"C-{purpose}-{w}"] = _first_rows(w, purpose)
    for purpose, w, rep in ab_streams:
        flats[f"AB-{purpose}-{w}-r{rep}"] = _first_rows(w, purpose, rep)

    names = list(flats)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.array_equal(flats[a], flats[b]), (a, b)


def test_seed_layout_per_world():
    base = stable_seed(INST)
    for w, code in WORLD_SEED_CODE.items():
        assert paths._stream(INST, "test_c", 0, 0, _token("test_c", "b_thr"), w)[0] \
            == [base, 7, code]
        if w != "W6":
            assert paths._stream(INST, "train", 2, 3, None, w)[0] == [base, 1, 2, 3, code]
            assert paths._stream(INST, "dev", 0, 0, None, w)[0] == [base, 3, code]
    assert paths._stream(INST, "train", 2, 3, None, "W6")[0] == [base, 1, 2, 3]
    assert paths._stream(INST, "dev", 0, 0, None, "W6")[0] == [base, 3]


def test_ab_streams_are_refused_for_non_w6_worlds():
    with pytest.raises(ValueError):
        paths._stream(INST, "test", 0, 0, _token("test"), "W3")


# ---------------------------------------------------------------- (d) correlation of scores

def _normal_scores(x, mu, cv=CV):
    k = 1.0 / (cv * cv)
    return st.norm.ppf(st.gamma.cdf(x, k, scale=mu / k))


def _mean_pair_correlation(world_code):
    dbar, pbar = _plan()
    dsc, psc = gen_scenarios(dbar, pbar, N_CORR, 20261125, world=WORLDS[world_code])
    both = [i for i in range(1, len(dbar)) if dbar[i] > 0 and pbar[i] > 0]
    assert len(both) >= 10
    rs = []
    for i in both:
        zd = _normal_scores(dsc[:, i], dbar[i])
        zp = _normal_scores(psc[:, i], pbar[i])
        rs.append(np.corrcoef(zd, zp)[0, 1])
    return float(np.mean(rs))


def test_ws_delivery_pickup_scores_correlate_about_0_6():
    assert abs(_mean_pair_correlation("WS") - 0.6) < 0.03


def test_w0_delivery_pickup_scores_are_uncorrelated():
    assert abs(_mean_pair_correlation("W0")) < 0.03


def test_w6_separate_copulas_leave_delivery_pickup_uncorrelated():
    # separate copulas: a customer's delivery and pickup are independent
    assert abs(_mean_pair_correlation("W6")) < 0.03


# ---------------------------------------------------------------- (e) the test_c gate

def test_test_c_without_a_gate_token_is_refused(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no test_c data may be generated")
    monkeypatch.setattr(paths, "gen_scenarios", boom)
    with pytest.raises(TestStreamGateError):
        route_pool(INST, "test_c")
    with pytest.raises(TestStreamGateError):
        route_pool(INST, "test_c", frozen="a" * 40)          # no stage name
    with pytest.raises(TestStreamGateError):
        route_pool(INST, "test_c", frozen="a" * 40, stage="b_thr", world="W3")


def test_test_c_stream_needs_a_stage_and_fit_commit():
    with pytest.raises(TestStreamGateError, match="name of the stage"):
        paths.verify_test_c_gate(None, "a" * 40)
    with pytest.raises(TestStreamGateError, match="fit commit"):
        paths.verify_test_c_gate("b_thr", None)


def test_test_c_token_does_not_unlock_other_streams():
    with pytest.raises(TestStreamGateError):
        paths._require_gate("test", _token("test_c", "b_thr"))
    with pytest.raises(TestStreamGateError):
        paths._require_gate("test_b", _token("test_c", "b_thr"))
    with pytest.raises(TestStreamGateError):
        paths._require_gate("test_c", _token("test"))


def test_route_pool_uses_the_world_seed_and_copula():
    base = stable_seed(INST)
    dbar, pbar = _plan()
    spec = route_specs(INST)[0]
    rid, route = spec.route_id, np.array(spec.route)
    pool = route_pool(INST, "dev", world="W9")
    dsc, psc = gen_scenarios(dbar, pbar, 5000, [base, 3, WORLD_SEED_CODE["W9"]],
                             world=WORLDS["W9"])
    assert np.array_equal(pool[rid].p, psc[:, route])
    assert np.array_equal(pool[rid].d, dsc[:, route])
    assert not np.array_equal(pool[rid].p, route_pool(INST, "dev")[rid].p)


def test_test_c_gate_defers_to_the_stage_check_and_carries_its_stream(monkeypatch):
    from transvrp.expc import gates as c_gates

    def refuse(stage, fit_commit, repo=None):
        raise RuntimeError("stage not pinned")

    def boom(*a, **k):
        raise AssertionError("no test_c data may be generated")

    monkeypatch.setattr(c_gates, "verify_test_c_stage", refuse)
    monkeypatch.setattr(paths, "gen_scenarios", boom)
    with pytest.raises(RuntimeError, match="not pinned"):
        route_pool(INST, "test_c", frozen="a" * 40, stage="b_thr")
    monkeypatch.setattr(c_gates, "verify_test_c_stage",
                        lambda s, f, repo=None: ("a" * 40, "c" * 64))
    token = paths.verify_test_c_gate("b_thr", "a" * 40)
    assert token.stream == "test_c" and token.stage == "b_thr"
    assert token.family_sha256 == "c" * 64


# ---------------------------------------------------------------- (f) chunk file names

def _old_chunk_name(inst, purpose, rep, c, seed, n):
    tag = format(zlib.crc32(repr(seed).encode()), "08x")
    return f"{inst}__{purpose}__r{rep}__c{c}__n{n}__s{tag}.npy"


def test_w6_chunk_file_names_are_unchanged(tmp_path):
    base = stable_seed(INST)
    seed = [base, 1, 1, 0]
    got = chunk_file(tmp_path, INST, "train", 1, 0, seed, 100_000)
    assert got.name == _old_chunk_name(INST, "train", 1, 0, seed, 100_000)
    seed_dev = [base, 3]
    got = chunk_file(tmp_path, INST, "dev", 0, 0, seed_dev, 5000, "W6")
    assert got.name == _old_chunk_name(INST, "dev", 0, 0, seed_dev, 5000)


def test_non_w6_chunk_file_names_include_the_world(tmp_path):
    base = stable_seed(INST)
    seed = [base, 1, 0, 0, WORLD_SEED_CODE["W3"]]
    name = chunk_file(tmp_path, INST, "train", 0, 0, seed, 100_000, "W3").name
    assert name == _old_chunk_name(INST, "train", 0, 0, seed, 100_000)[:-4] + "__W3.npy"
    dev_w3 = chunk_file(tmp_path, INST, "dev", 0, 0, [base, 3, 13], 5000, "W3").name
    dev_w9 = chunk_file(tmp_path, INST, "dev", 0, 0, [base, 3, 13], 5000, "W9").name
    assert dev_w3.endswith("__W3.npy") and dev_w3 != dev_w9


def test_test_c_chunk_file_names_include_the_world_even_for_w6(tmp_path):
    base = stable_seed(INST)
    seed = [base, 7, WORLD_SEED_CODE["W6"]]
    name = chunk_file(tmp_path, INST, "test_c", 0, 0, seed, 20_000, "W6").name
    assert name.endswith("__W6.npy")
    assert name != _old_chunk_name(INST, "test_c", 0, 0, seed, 20_000)


# ---------------------------------------------------------------- rho with a world (item 5)

def test_explicit_rho_that_conflicts_with_the_world_is_refused():
    dbar, pbar = _plan()
    with pytest.raises(ValueError, match="conflicts"):
        gen_scenarios(dbar, pbar, 10, 1, rho=0.6, world=WORLDS["W0"])
    with pytest.raises(ValueError, match="conflicts"):
        gen_scenarios(dbar, pbar, 10, 1, rho=0.3, world=WORLDS["WS"])


def test_explicit_rho_equal_to_the_world_copula_is_accepted_and_changes_nothing():
    dbar, pbar = _plan()
    a = gen_scenarios(dbar, pbar, 50, 7, rho=0.6, world="W6")
    b = gen_scenarios(dbar, pbar, 50, 7, world="W6")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ---------------------------------------------------------------- WS dimension 2m' (item 4)

def test_every_plan_route_has_equal_active_delivery_and_pickup_counts():
    """WS draws one vector of 2m' scores (deliveries then pickups of the active customers).
    Checked on every route of every instance plan: the counts agree, so the code's
    len(act_d) + len(act_p) is 2m'."""
    from transvrp.data import PLANS_DIR
    plans = sorted(PLANS_DIR.glob("*.json"))
    assert len(plans) == 20
    checked = 0
    for path in plans:
        sol = json.loads(path.read_text(encoding="utf-8"))
        dbar, pbar = np.array(sol["dbar"], float), np.array(sol["pbar"], float)
        act_d = [i for i in range(1, len(dbar)) if dbar[i] > 0]
        act_p = [i for i in range(1, len(pbar)) if pbar[i] > 0]
        assert act_d == act_p, path.name
        for plan in sol["res"].values():
            for route in plan["plan"]:
                if route:
                    r = np.array(route)
                    assert int(np.sum(dbar[r] > 0)) == int(np.sum(pbar[r] > 0)), path.name
                    checked += 1
    assert checked == 802      # non-empty routes over the 8 plan methods of 20 instances


def test_ws_refuses_unequal_active_counts():
    dbar = np.array([0.0, 1.0, 2.0])
    pbar = np.array([0.0, 1.0, 0.0])
    with pytest.raises(AssertionError, match="2m'"):
        gen_scenarios(dbar, pbar, 5, 1, world=WORLDS["WS"])

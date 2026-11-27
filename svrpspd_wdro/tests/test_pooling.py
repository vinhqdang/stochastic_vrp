"""Tests for ev/pooling.py: zones, grid posteriors, decision factors and the
method-of-moments history estimate. Fast and deterministic."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ev.fleet import FleetPlan  # noqa: E402
from ev.pooling import (BG, CV, SCENARIOS, SHAPE, Scenario, build_structure,  # noqa: E402
                        customer_zones, draw_day, gauss, history_estimate,
                        hier_means, loglik, post_mean, weighted_factor,
                        zone_count)


def _euclid(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def test_zone_assignment_deterministic_and_in_range():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 10, size=(30, 2))
    D = np.rint(_euclid(pts) * 100).astype(int)
    customers = list(range(30))
    K = 9
    Z = zone_count(K)
    assert Z == 4
    z1 = customer_zones(D, customers, K)
    z2 = customer_zones(D, customers, K)
    assert np.array_equal(z1, z2)
    assert z1.shape == (30,)
    assert set(z1.tolist()) <= set(range(Z))


def test_zone_assignment_recovers_separated_clusters():
    centres = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    rng = np.random.default_rng(4)
    pts = np.vstack([c + rng.normal(0, 1.0, size=(8, 2)) for c in centres])
    D = np.rint(_euclid(pts) * 10).astype(int)
    customers = list(range(len(pts)))
    z = customer_zones(D, customers, K=6)      # Z = 3
    groups = [set(z[8 * i: 8 * (i + 1)].tolist()) for i in range(3)]
    assert all(len(g) == 1 for g in groups)
    assert len({next(iter(g)) for g in groups}) == 3


def test_zero_sd_hier_reduces_to_pooled_posteriors():
    rng = np.random.default_rng(5)
    Lz = rng.normal(0.0, 3.0, size=(3, len(BG)))
    # s_z -> 0: one shared factor with prior N(0, s_c^2) on log a.
    s_c = 0.3
    full_prior = -0.5 * (BG / s_c) ** 2
    full = post_mean(full_prior + Lz.sum(0))
    got = hier_means(Lz, s_c, 1e-9)
    assert np.allclose(got, full, atol=1e-6, rtol=0)
    # s_c -> 0: independent zones with prior N(0, s_z^2).
    s_z = 0.3
    zone_prior = -0.5 * (BG / s_z) ** 2
    expected = [post_mean(zone_prior + Lz[z]) for z in range(3)]
    got = hier_means(Lz, 1e-9, s_z)
    assert np.allclose(got, expected, atol=1e-6, rtol=0)


def test_known_decision_factor_with_equal_zone_factors_returns_a():
    rng = np.random.default_rng(6)
    a = 1.37
    Z = 3
    m = 9
    pbar = rng.uniform(0.0, 2.0, m)
    pbar[3] = 0.0                         # a zero-pickup customer
    zone = rng.integers(0, Z, m)
    zone_factor = np.full(Z, a)
    for k in range(m - 1):
        got = weighted_factor(pbar[k + 1:], zone_factor[zone[k + 1:]])
        assert abs(got - a) < 1e-12
    # all remaining planned pickups zero: unweighted mean
    got = weighted_factor(np.zeros(3), np.array([1.0, 2.0, 3.0]))
    assert abs(got - 2.0) < 1e-12


def _pickups(a, pbar, rng):
    shape = 1.0 / CV ** 2
    return a * pbar * rng.gamma(shape, 1.0, len(pbar)) / shape


def test_mle_and_hier_recover_constant_factor_with_many_observations():
    rng = np.random.default_rng(7)
    a_true = 1.3
    Z = 2
    n = 400
    Lz = np.zeros((Z, len(BG)))
    Sp = np.zeros(Z)
    Sb = np.zeros(Z)
    for z in range(Z):
        pbar = rng.uniform(0.5, 2.0, n)
        p = _pickups(a_true, pbar, rng)
        for pi, bi in zip(p, pbar):
            Lz[z] += loglik(pi, bi)
        Sp[z] = p.sum()
        Sb[z] = pbar.sum()
    mle = Sp / Sb
    assert np.all(np.abs(mle / a_true - 1) < 0.05)
    hier = np.array(hier_means(Lz, 0.2, 0.2))
    assert np.all(np.abs(hier / a_true - 1) < 0.05)


def test_history_estimate_recovers_hyperparameters():
    rng = np.random.default_rng(8)
    Z = 3
    per_zone = 8
    zc = np.repeat(np.arange(Z), per_zone)
    sk = np.ones(len(zc), dtype=int)             # one stop index, no drift
    pb = rng.uniform(0.5, 2.0, len(zc))
    scn = Scenario("synthetic", 0.2, 0.2)
    sc_hat, sz_hat = history_estimate(zc, sk, pb, Z, scn, H=2000,
                                      rng=np.random.default_rng(9))
    assert abs(sc_hat - 0.2) < 0.06
    assert abs(sz_hat - 0.2) < 0.06


def test_gauss_degenerate_is_delta_at_zero_grid_point():
    g = gauss(BG, 1e-9)
    assert g.sum() == 1.0
    assert BG[int(np.argmax(g))] == BG[int(np.argmin(np.abs(BG)))]


def _toy_fleet(n_routes=6, m=4, seed=11):
    """Synthetic plan with K = n_routes routes of m customers each."""
    rng = np.random.default_rng(seed)
    n = 1 + n_routes * m
    X = rng.uniform(0, 1, (n, 2))
    D = np.rint(np.hypot(*(X[:, None, :] - X[None, :, :]).transpose(2, 0, 1))
                * 1e4).astype(np.int64)
    dbar = np.zeros(n)
    pbar = np.zeros(n)
    dbar[1:] = rng.uniform(40, 120, n - 1)
    pbar[1:] = rng.uniform(40, 120, n - 1)
    routes = [list(range(1 + r * m, 1 + (r + 1) * m)) for r in range(n_routes)]
    Q = max(dbar[r].sum() for r in routes) + 150.0
    return FleetPlan(D, Q, 10000, dbar, pbar, routes), D


def _same_partition(a, b):
    """True when two label vectors induce the same partition."""
    same_a = a[:, None] == a[None, :]
    same_b = b[:, None] == b[None, :]
    return np.array_equal(same_a, same_b)


def test_mixed_day_uses_estimator_zones_and_is_reproducible():
    fl, D = _toy_fleet()
    scn = SCENARIOS["mixed"]
    assert scn.zones_true == "same"
    st = build_structure(fl, D, alt=(scn.zones_true == "alt"))
    assert st.Zt == st.Z
    assert all(np.array_equal(a, b) for a, b in zip(st.ztrue, st.zone))
    assert np.array_equal(st.zct, st.zc)
    d1 = draw_day(fl, st, scn, np.random.default_rng(21))
    d2 = draw_day(fl, st, scn, np.random.default_rng(21))
    assert np.array_equal(d1.B, d2.B)
    for r in range(st.K):
        assert np.array_equal(d1.af[r], d2.af[r])
        assert np.array_equal(d1.p[r], d2.p[r])
        assert np.array_equal(d1.d[r], d2.d[r])
        assert np.array_equal(d1.dec_known[r], d2.dec_known[r])


def test_alt_partition_has_one_more_zone_and_differs():
    fl, D = _toy_fleet()
    st = build_structure(fl, D, alt=True)
    assert st.Zt == st.Z + 1
    assert st.zct.shape == st.zc.shape
    assert set(st.zct.tolist()) <= set(range(st.Zt))
    assert not _same_partition(st.zc, st.zct)
    assert [len(z) for z in st.ztrue] == [rt.m for rt in fl.routes]
    day = draw_day(fl, st, SCENARIOS["wrongzone"], np.random.default_rng(22))
    assert day.B.shape == (fl.m_max, st.Zt)
    for r, rt in enumerate(fl.routes):
        assert day.af[r].shape == (rt.m,)
        assert np.all(np.isfinite(day.p[r])) and np.all(day.p[r] >= 0)

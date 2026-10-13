"""Tests for scripts/make_region_instances.py (no network, no raw data)."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

_WDRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_WDRO / "scripts"))

import make_region_instances as m  # noqa: E402
import dethloff_runner  # noqa: E402


# ---------------------------------------------------------------- names
@pytest.mark.parametrize("raw, expected", [
    ("Hà Nội\t", "HANOI"),
    ("Bắc Kạn", "BACKAN"),
    ("Điện Biên", "DIENBIEN"),
    ("Bà Rịa–Vũng Tàu", "BARIAVUNGTAU"),
    ("Shenzhenshi", "SHENZHENSHI"),
])
def test_slug(raw, expected):
    assert m.slug(raw) == expected


def test_instance_names_collision_suffix():
    areas = [m.Area("Bắc Kạn", None), m.Area("Bac Kan", None), m.Area("Hà Nội\t", None)]
    assert m.instance_names("VNN", areas) == ["VNN-BACKAN", "VNN-BACKAN-2", "VNN-HANOI"]


# ---------------------------------------------------------------- geometry
def test_weighted_medoid_hand_case():
    # three points on the equator, the heavy one is at lon 0.02
    lat = np.zeros(3)
    lon = np.array([0.0, 0.01, 0.02])
    pop = np.array([1.0, 1.0, 10.0])
    assert m.weighted_medoid(lat, lon, pop) == 2


def test_project_unproject_roundtrip():
    lat = np.array([21.0, 21.2])
    lon = np.array([105.8, 105.9])
    xy = m.project(lat, lon, 21.1, 105.85)
    la, lo = m.unproject(xy, 21.1, 105.85)
    np.testing.assert_allclose(la, lat, atol=1e-9)
    np.testing.assert_allclose(lo, lon, atol=1e-9)


# ---------------------------------------------------------------- k-means / split
def _blob(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.normal(0.0, 5.0, size=(n, 2))
    w = rng.uniform(5.0, 50.0, size=n)
    return xy, w


def test_kmeans_deterministic_for_fixed_seed():
    xy, w = _blob()
    a = m._kmeans_pass(xy, w, 20, seed=12345)
    b = m._kmeans_pass(xy, w, 20, seed=12345)
    assert [c.pop for c in a] == [c.pop for c in b]
    for ca, cb in zip(a, b):
        np.testing.assert_array_equal(ca.members, cb.members)


def test_kmeans_population_conserved_after_drop():
    xy, w = _blob()
    raw = m._kmeans_pass(xy, w, 25, seed=7)
    assert sum(c.pop for c in raw) == pytest.approx(w.sum())
    kept = [c for c in raw if c.pop >= m.MIN_CLUSTER_POP]
    dropped = sum(c.pop for c in raw if c.pop < m.MIN_CLUSTER_POP)
    assert sum(c.pop for c in kept) + dropped == pytest.approx(w.sum())


def test_split_leaves_no_cluster_above_split():
    # uniform dense blob: one split pass gives balanced parts
    rng = np.random.default_rng(3)
    xy = rng.uniform(-10.0, 10.0, size=(20000, 2))
    w = np.full(20000, 10.0)              # total population 2e5
    ppp = 1000.0
    split = m.SPLIT_FACTOR * ppp
    coarse = m._kmeans_pass(xy, w, 2, seed=1)            # clusters far above split
    assert max(c.pop for c in coarse) > split
    fine = m._split_oversize(coarse, xy, w, "blob", ppp, split)
    assert sum(c.pop for c in fine) == pytest.approx(w.sum())
    assert max(c.pop for c in fine) <= split


def test_split_area_gives_subareas_each_within_nmax():
    rng = np.random.default_rng(11)
    n = 2000
    lat = 21.0 + rng.uniform(-0.3, 0.3, n)
    lon = 105.8 + rng.uniform(-0.3, 0.3, n)
    pop = np.full(n, 200.0)               # P = 400000 -> 800 points at ppp=500 (> N_MAX)
    plan = m.plan_area(lat, lon, pop, "Test Area", 500.0)
    assert len(plan.pop) > m.N_MAX
    assert plan.pop_small == pytest.approx(0.0)
    parts = m.split_area(lat, lon, pop, "Test Area", 500.0, len(plan.pop))
    assert len(parts) == 2                # ceil(900 / 600)
    assert [name for name, _ in parts] == ["Test Area-1", "Test Area-2"]
    for name, sub in parts:
        assert len(sub.pop) <= m.N_MAX
        assert sub.ppp == 500.0           # ppp is never rescaled
    total = sum(sub.pop_area for _, sub in parts)
    assert total == pytest.approx(plan.pop_area)
    assert sum(sub.pop.sum() + sub.pop_small for _, sub in parts) == pytest.approx(plan.pop_area)


def test_plan_area_points_within_nmax_are_not_split():
    rng = np.random.default_rng(5)
    lat = 21.0 + rng.uniform(-0.1, 0.1, 500)
    lon = 105.8 + rng.uniform(-0.1, 0.1, 500)
    pop = np.full(500, 100.0)             # P = 50000 -> 50 points at ppp=1000
    plan = m.plan_area(lat, lon, pop, "Small", 1000.0)
    assert len(plan.pop) <= m.N_MAX
    assert plan.ppp == 1000.0


# ---------------------------------------------------------------- OSRM blocks
@pytest.mark.parametrize("n", [1, 99, 100, 101, 250])
def test_osrm_block_plan_covers_each_ordered_pair_once(n):
    blocks = m.osrm_block_plan(n)
    seen = Counter()
    for b in blocks:
        assert len(b.coords) <= 200
        assert len(b.src) <= 100 and len(b.dst) <= 100
        for s in b.src:
            for d in b.dst:
                seen[(s, d)] += 1
    assert len(seen) == n * n
    assert set(seen.values()) == {1}


def _parse_url(url):
    base, _, query = url.partition("?")
    coords = []
    for tok in base[len(m.OSRM_URL):].split(";"):
        lon, lat = tok.split(",")
        coords.append((float(lat), float(lon)))
    params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
    srcs = [int(x) for x in params["sources"].split(";")] if "sources" in params else list(range(len(coords)))
    dsts = [int(x) for x in params["destinations"].split(";")] if "destinations" in params else list(range(len(coords)))
    return coords, srcs, dsts


def _fake_index(lon):
    return int(round((lon - 105.0) / 0.001))


NULL_PAIR = (2, 5)
SNAP_BAD = 7


def _fake_osrm(url):
    coords, srcs, dsts = _parse_url(url)
    gidx = [_fake_index(lon) for _, lon in coords]

    def f(a, b):
        if a == b:
            return 0.0
        if (a, b) == NULL_PAIR:
            return None
        return 1000.0 * (a + 1) + 7.0 * b

    def snap(k):
        return 6000.0 if k == SNAP_BAD else 10.0 * k

    return {
        "code": "Ok",
        "distances": [[f(gidx[s], gidx[d]) for d in dsts] for s in srcs],
        "sources": [{"distance": snap(gidx[s])} for s in srcs],
        "destinations": [{"distance": snap(gidx[d])} for d in dsts],
    }


@pytest.mark.parametrize("n", [3, 150])
def test_build_matrix_from_fake_responses(monkeypatch, n):
    monkeypatch.setattr(m, "_osrm_get", _fake_osrm)
    coords = np.array([[21.0, 105.0 + 0.001 * k] for k in range(n)])
    M, snap, n_blocks = m.build_matrix(coords)
    assert n_blocks == len(m.osrm_block_plan(n))
    for a in range(n):
        for b in range(n):
            if a == b:
                assert M[a, b] == 0.0
            elif (a, b) == NULL_PAIR:
                assert np.isnan(M[a, b])
            else:
                assert M[a, b] == pytest.approx(1000.0 * (a + 1) + 7.0 * b)
    if n > SNAP_BAD:
        assert snap[SNAP_BAD] == 6000.0
    assert snap[0] == 0.0


def test_block_url_formats():
    coords = [(21.0, 105.0 + 0.001 * k) for k in range(150)]
    diag = m.osrm_block_plan(150)[0]
    url = m.block_url(coords, diag)
    assert "sources=" not in url and url.count(";") == 99
    off = m.osrm_block_plan(150)[1]          # (chunk0, chunk1)
    url = m.block_url(coords, off)
    assert "sources=0;" in url and "destinations=100;" in url


def test_select_nodes_snap_null_and_depot_fallback():
    n = 5                                  # node 0 depot, 1..4 customers
    M = np.full((n, n), 1000.0)
    np.fill_diagonal(M, 0.0)
    M[0, 2] = np.nan                       # null to depot -> node 2 dropped
    M[3, 4] = M[4, 3] = np.nan             # customer-customer null
    snap = np.array([10.0, 6000.0, 10.0, 10.0, 10.0])   # node 1 fails the snap filter
    coords = np.array([[21.0, 105.0 + 0.01 * k] for k in range(n)])
    pops = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    sel = m.select_nodes(M, snap, coords, pops)
    assert sel["dep"] == 0 and not sel["replaced_depot"]
    assert sel["dropped_snap"] == [1]
    assert sel["cust"] == [4]
    assert sorted(sel["dropped_null"]) == [2, 3]


def test_select_nodes_replaces_depot_when_its_snap_fails():
    n = 4
    M = np.full((n, n), 500.0)
    np.fill_diagonal(M, 0.0)
    snap = np.array([9000.0, 10.0, 10.0, 10.0])          # depot copy unsnappable
    coords = np.array([[21.0, 105.0], [21.0, 105.1], [21.0, 105.2], [21.0, 105.3]])
    pops = np.array([0.0, 1.0, 50.0, 1.0])
    sel = m.select_nodes(M, snap, coords, pops)
    assert sel["replaced_depot"]
    assert sel["dep"] == 2                 # heavy point is the medoid
    assert sel["dep"] in sel["cust"]


# ---------------------------------------------------------------- demand
VNN = m.REGION_CFG["VNN"]
GD = m.REGION_CFG["GD"]


def test_demand_rule_vnn_and_gd():
    dem = m.make_demands("TEST-X", [1000.0], VNN)
    assert dem[0, 0] == pytest.approx(87.4)                  # 1000 * 0.0874 * 1.0
    assert 43.7 - 0.1 <= dem[0, 1] <= 131.1 + 0.1            # * U(0.5, 1.5)
    dem = m.make_demands("TEST-Y", [1000.0], GD)
    assert dem[0, 0] == pytest.approx(339.6)                 # 1000 * 0.3396 * 1.0
    assert 0.5 * 913 - 0.1 <= dem[0, 1] <= 1.5 * 913 + 0.1   # 1000 * 0.913 * U(0.5, 1.5)
    assert np.allclose(dem, np.round(dem, 1))


def test_demand_floor():
    dem = m.make_demands("TEST-F", [0.5], VNN)
    assert dem[0, 0] == pytest.approx(1.0)
    assert dem[0, 1] == pytest.approx(1.0)


def test_demand_cap_assertion_half_q():
    # VNN Q/2 = 7500 kg; 100000 people -> 8740 kg delivery
    with pytest.raises(RuntimeError):
        m.make_demands("TEST-BIG", [100000.0], VNN)
    # GD Q/2 = 15000 kg; 10000 people -> pickup up to 13695 kg is allowed
    m.make_demands("TEST-OK", [10000.0], GD)


# ---------------------------------------------------------------- round trip
def test_round_trip_through_parse_dethloff(tmp_path):
    n = 4
    D_m = np.array([[0, 1234, 2000, 900],
                    [1234, 0, 700, 1500],
                    [2000, 700, 0, 400],
                    [900, 1500, 400, 0]], dtype=float)
    D = np.round(D_m * m.SCALE).astype(np.int64)
    dem = m.make_demands("TEST-RT", [2000.0, 5000.0, 9000.0], VNN)
    path = tmp_path / "TEST-RT.vrpspd"
    m.write_vrpspd(path, "TEST-RT", D, dem, VNN["Q"], "round trip")
    D2, dem2, Q2, n2, div = dethloff_runner.parse_dethloff(path)
    assert n2 == n
    assert div == 10000
    np.testing.assert_array_equal(D2, D)
    assert Q2 == VNN["Q"]
    np.testing.assert_allclose(dem2[1:], dem, atol=1e-9)
    assert dem2[0].tolist() == [0.0, 0.0]
    # 0.1 m units / 10^4 gives km
    assert D2[0, 1] / div == pytest.approx(1.234)

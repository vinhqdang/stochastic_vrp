"""Tests for ev/fleet.py: policy library, fleet-day simulation, triggers.
Deterministic (seeded). Uses one real plan instance and small synthetic
fleets. Library tests use G=800 on synthetic fleets for speed."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.costs import fit_lsm_actions, simulate_actions  # noqa: E402
from ev.vi import RouteValue  # noqa: E402
from ev.fleet import (FleetPlan, PolicyLibrary, sample_day,  # noqa: E402
                      simulate_fleet_day, estimate_a, quantize_a, make_drift,
                      _sample_pd, A_GRID, N_FIT, G_LIB, CV,
                      Never, EveryK, ValueInnovation, CostBased, Oracle)

INST = Path(__file__).resolve().parent.parent / "data" / "Dethloff" / "CON3-0.vrpspd"


def _synthetic_fleet(m_per_route=5, n_routes=2, Q_slack=150.0, seed=3):
    rng = np.random.default_rng(seed)
    n = 1 + n_routes * m_per_route
    X = rng.uniform(0, 1, (n, 2))
    D = np.rint(np.hypot(*(X[:, None, :] - X[None, :, :]).transpose(2, 0, 1))
                * 1e4).astype(np.int64)
    dbar = np.zeros(n)
    pbar = np.zeros(n)
    dbar[1:] = rng.uniform(40, 120, n - 1)
    pbar[1:] = rng.uniform(40, 120, n - 1)
    routes = [list(range(1 + r * m_per_route, 1 + (r + 1) * m_per_route))
              for r in range(n_routes)]
    Q = max(dbar[r].sum() for r in routes) + Q_slack
    return FleetPlan(D, Q, 10000, dbar, pbar, routes)


def test_grid_and_quantisation():
    assert len(A_GRID) == 16
    assert A_GRID[0] == 0.5 and A_GRID[-1] == 2.0
    assert quantize_a(0.74) == 0.7
    assert quantize_a(0.76) == 0.8
    assert quantize_a(0.1) == 0.5
    assert quantize_a(9.0) == 2.0


def test_library_a1_reproduces_direct_day0_fit():
    """The a = 1.0 library entry is the day-0 policy: it equals a direct
    fit on the same seeded planning-law sample, exactly."""
    fleet = FleetPlan.from_instance(INST, "Det")
    lib = PolicyLibrary(fleet, seed=100)
    for r, rt in enumerate(fleet.routes[:2]):
        seed = lib.seed_for(r, 1.0)
        p, d = _sample_pd(rt.pbar, rt.dbar, N_FIT, np.random.default_rng(seed))
        models = fit_lsm_actions(p - d, rt.B, rt.H, rt.E, rt.R)
        rv = RouteValue(rt.pbar, rt.dbar, rt.B, rt.H, rt.E, rt.R, models,
                        cv=CV, G=G_LIB)
        entry = lib.get(r, 1.0)
        assert entry.a == 1.0
        assert entry.rv.expected_cost() == rv.expected_cost()
        assert set(entry.rv.models) == set(models)


def test_never_on_null_day_equals_sum_of_solo_routes():
    fleet = _synthetic_fleet()
    lib = PolicyLibrary(fleet, seed=7, G=800)
    day = sample_day(fleet, "null", np.random.default_rng(11))
    res = simulate_fleet_day(lib, day, Never(), 0.0)
    assert res["n_replans"] == 0
    solo = 0.0
    for r, rt in enumerate(fleet.routes):
        g = np.asarray(day[0][r], dtype=float)[None, :]
        stats = simulate_actions(g, rt.B, rt.H, rt.E, rt.R, lib.get(r, 1.0).rv.models)
        solo += float(stats["costs"][0])
    assert res["recourse_cost"] == pytest.approx(solo, rel=1e-9, abs=1e-9)
    assert res["cost"] == pytest.approx(solo, rel=1e-9, abs=1e-9)


def test_everyk2_replans_only_at_stops_2_and_4():
    """One route of 6 stops with prices so large that no action or breach
    is ever optimal: the route stays alive, so the only replans are the
    scheduled ones that have remaining stops (stop 6 is the last)."""
    fleet = _synthetic_fleet(m_per_route=6, n_routes=1, Q_slack=1e7)
    rt = fleet.routes[0]
    big = np.full(rt.m, 1e9)
    rt.H, rt.E, rt.R = big.copy(), big.copy(), big.copy()
    lib = PolicyLibrary(fleet, seed=5, G=800)
    day = sample_day(fleet, "null", np.random.default_rng(6))
    res = simulate_fleet_day(lib, day, EveryK(2), 0.0)
    assert res["replan_stops"] == [2, 4]
    assert res["n_replans"] == 2


def test_value_innovation_no_replans_on_light_drift():
    fleet = _synthetic_fleet()
    lib = PolicyLibrary(fleet, seed=21, G=800)
    light = make_drift("light", fleet)
    trig = ValueInnovation(alpha=0.05)
    n_days = 200
    days_with_replan = 0
    for d in range(n_days):
        day = sample_day(fleet, light, np.random.default_rng([31, d]))
        res = simulate_fleet_day(lib, day, trig, 0.0)
        days_with_replan += int(res["n_replans"] > 0)
    assert days_with_replan <= 0.05 * n_days + 3


def test_oracle_never_replans_on_null():
    fleet = _synthetic_fleet()
    lib = PolicyLibrary(fleet, seed=9, G=800)
    day = sample_day(fleet, "null", np.random.default_rng(2))
    res = simulate_fleet_day(lib, day, Oracle(make_drift("null", fleet)), 0.0)
    assert res["n_replans"] == 0


def test_costbased_no_replan_when_quantised_scale_unchanged():
    fleet = _synthetic_fleet()
    lib = PolicyLibrary(fleet, seed=9, G=800)
    trig = CostBased(0.0)
    extra = {"fleet": fleet, "lib": lib, "W": np.zeros(len(fleet.routes)),
             "remaining": [0, 1], "a_q": 1.0, "a_cur": 1.0}
    assert trig.observe(1, [], extra) is False


def test_estimate_a_floor_cap_and_empty():
    assert estimate_a([], []) == 1.0
    assert estimate_a([100.0], [100.0]) == pytest.approx(1.0)
    assert estimate_a([1.0], [100.0]) == pytest.approx(0.3)
    assert estimate_a([1000.0], [100.0]) == pytest.approx(3.0)
    assert estimate_a([130.0, 0.0], [100.0, 0.0]) == pytest.approx(1.3)

"""Tests for ev/fleet.py: suffix refits, fleet-day simulation, triggers.
Deterministic (seeded). Uses one real plan route and small synthetic fleets."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.costs import simulate_actions  # noqa: E402
from ev.fleet import (FleetPlan, fit_fleet_day0, refit_suffix,  # noqa: E402
                      sample_day, simulate_fleet_day, estimate_a,
                      Never, EveryK, ValueInnovation, make_drift)

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


@pytest.fixture(scope="module")
def real_fleet():
    return FleetPlan.from_instance(INST, "Det")


def test_suffix_refit_k0_reproduces_full_route_cost(real_fleet):
    """Refit after stop 0 at load 0 with a_hat = 1 (pseudo-stop path) must
    give the same expected cost as the day-0 fit, within 1%."""
    rt = real_fleet.routes[1]  # m = 12, expected cost ~ 10
    pol0 = fit_fleet_day0(real_fleet, seed=100)[1]
    ref = pol0.rv.expected_cost()
    refit = refit_suffix(rt, 0, 0.0, 1.0, seed=[5, 0, 1])
    got = refit.rv.expected_cost()
    assert ref > 0
    assert abs(got / ref - 1.0) < 0.01


def test_never_on_null_day_equals_sum_of_solo_routes():
    fleet = _synthetic_fleet()
    pol = fit_fleet_day0(fleet, seed=7, G=800)
    rng = np.random.default_rng(11)
    day = sample_day(fleet, "null", rng)
    res = simulate_fleet_day(fleet, pol, day, Never(), 0.0,
                             np.random.default_rng(12))
    assert res["n_replans"] == 0
    solo = 0.0
    for r, rt in enumerate(fleet.routes):
        g = np.asarray(day[0][r], dtype=float)[None, :]
        stats = simulate_actions(g, rt.B, rt.H, rt.E, rt.R, pol[r].rv.models)
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
    pol = fit_fleet_day0(fleet, seed=5, G=800)
    day = sample_day(fleet, "null", np.random.default_rng(6))
    res = simulate_fleet_day(fleet, pol, day, EveryK(2), 0.0,
                             np.random.default_rng(8))
    assert res["replan_stops"] == [2, 4]
    assert res["n_replans"] == 2


def test_value_innovation_no_replans_on_light_drift():
    fleet = _synthetic_fleet()
    pol = fit_fleet_day0(fleet, seed=21, G=800)
    light = make_drift("light", fleet)
    trig = ValueInnovation(alpha=0.05)
    n_days = 200
    days_with_replan = 0
    for d in range(n_days):
        day = sample_day(fleet, light, np.random.default_rng([31, d]))
        res = simulate_fleet_day(fleet, pol, day, trig, 0.0,
                                 np.random.default_rng([32, d]))
        days_with_replan += int(res["n_replans"] > 0)
    assert days_with_replan <= 0.05 * n_days + 3


def test_estimate_a_floor_cap_and_empty():
    assert estimate_a([], []) == 1.0
    assert estimate_a([100.0], [100.0]) == pytest.approx(1.0)
    assert estimate_a([1.0], [100.0]) == pytest.approx(0.3)
    assert estimate_a([1000.0], [100.0]) == pytest.approx(3.0)
    assert estimate_a([130.0, 0.0], [100.0, 0.0]) == pytest.approx(1.3)

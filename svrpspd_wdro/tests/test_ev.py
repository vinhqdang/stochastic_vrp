"""Tests for ev/: martingale validity, Ville false-alarm control,
predictable-tilt safety, detection power, and simulator sanity."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ev.world import DayParams, DriftSpec, simulate_route_day
from ev.eprocess import (MasterEProcess, gaussian_step, poisson_step,
                         bernoulli_step, run_day)
from ev.baselines import CusumMonitor, PeriodicMonitor


def _params(m=20, seed_B=50.0):
    rng = np.random.default_rng(7)
    return DayParams(
        tau=rng.uniform(0.1, 0.4, m),
        mu_g=rng.normal(0.0, 2.0, m),
        sig_g=rng.uniform(1.0, 3.0, m),
        B=seed_B)


# ── unit bets are conditionally mean-one under H0 ───────────────────────────

def test_gaussian_step_mean_one():
    rng = np.random.default_rng(0)
    z = rng.normal(size=200_000)
    for s in (0.25, 0.6, 1.0):
        e = [gaussian_step(zi, s) for zi in z[:50_000]]
        assert abs(np.mean(e) - 1.0) < 0.02


def test_poisson_step_mean_one():
    rng = np.random.default_rng(1)
    lam = 0.15
    n = rng.poisson(lam, 200_000)
    e = np.array([poisson_step(ni, lam, 0.8) for ni in n[:50_000]])
    assert abs(e.mean() - 1.0) < 0.02


def test_bernoulli_step_mean_one():
    p0 = 0.002
    # exact: p0*e(1) + (1-p0)*e(0) = 1 by construction
    e1 = bernoulli_step(1, p0)
    e0 = bernoulli_step(0, p0)
    assert abs(p0 * e1 + (1 - p0) * e0 - 1.0) < 1e-12
    # single event alone crosses the Ville threshold at the same alpha
    assert e1 >= 1.0 / 0.05


# ── master process: E[E_T] = 1 under H0 even with predictable tilts ─────────

def test_master_mean_one_under_null():
    # E[E_T] = 1 exactly; use SHORT days so the product's heavy tail
    # does not swamp the Monte-Carlo mean, and many replications.
    p = _params(m=3)
    null = DriftSpec(kind="none")
    vals = []
    for seed in range(4000):
        rng = np.random.default_rng(1000 + seed)
        events, _ = simulate_route_day(p, null, rng)
        mon = MasterEProcess(alpha=0.05, use_sensitivity=True)
        for ev in events:
            mon.update(ev)
        vals.append(mon.E)
    assert 0.85 < np.mean(vals) < 1.2
    assert np.median(vals) < 1.05           # typical path decays


def test_ville_false_alarm_rate():
    p = _params()
    null = DriftSpec(kind="none")
    alpha = 0.10
    fired = 0
    N = 500
    for seed in range(N):
        rng = np.random.default_rng(2000 + seed)
        events, _ = simulate_route_day(p, null, rng)
        mon = MasterEProcess(alpha=alpha, use_sensitivity=True)
        if run_day(mon, events)["fired"]:
            fired += 1
    # Ville guarantees <= alpha; allow MC slack (3 sigma of binomial)
    bound = alpha + 3 * np.sqrt(alpha * (1 - alpha) / N)
    assert fired / N <= bound


# ── power: strong drift is detected, and usually after the change-point ─────

@pytest.mark.parametrize("kind,mag", [("traffic", 2.0), ("demand", 1.5),
                                      ("accident", 8.0), ("breakdown", 200.0)])
def test_detects_strong_drift(kind, mag):
    p = _params(m=40)
    drift = DriftSpec(kind=kind, t_star=9.0, magnitude=mag)
    hits, valid_hits = 0, 0
    N = 60
    for seed in range(N):
        rng = np.random.default_rng(3000 + seed)
        events, _ = simulate_route_day(p, drift, rng)
        mon = MasterEProcess(alpha=0.05)
        out = run_day(mon, events)
        if out["fired"]:
            hits += 1
            if out["alarm_clock"] >= 9.0:
                valid_hits += 1
    assert hits / N > 0.5, f"{kind}: detection rate {hits/N}"
    assert valid_hits >= 0.9 * hits          # alarms land after t*


def test_attribution_points_to_drifted_channel():
    p = _params(m=40)
    drift = DriftSpec(kind="traffic", t_star=8.5, magnitude=2.5)
    rng = np.random.default_rng(42)
    events, _ = simulate_route_day(p, drift, rng)
    mon = MasterEProcess(alpha=0.05)
    run_day(mon, events)
    att = mon.attribution()
    assert att["travel"] == max(att.values())


# ── baselines conform to the same interface ─────────────────────────────────

def test_baselines_run():
    p = _params()
    rng = np.random.default_rng(5)
    events, _ = simulate_route_day(p, DriftSpec(kind="none"), rng)
    for mon in (CusumMonitor(h=10.0), PeriodicMonitor(period=30)):
        out = run_day(mon, events)
        assert set(out) >= {"fired", "alarm_idx", "alarm_clock"}


# ── simulator sanity ─────────────────────────────────────────────────────────

def test_world_event_stream_shape():
    p = _params(m=15)
    rng = np.random.default_rng(9)
    events, meta = simulate_route_day(p, DriftSpec(kind="none"), rng)
    assert len(events) == 15 * 5             # 5 channels fire per stop
    assert meta["clock_end"] > p.t0
    chans = {e["channel"] for e in events}
    assert chans == {"travel", "demand", "dwell", "accident", "breakdown"}
    # ctx is predictable: W_prev of stop k equals sum of increments < k
    W = 0.0
    for e in events:
        if e["channel"] == "demand":
            assert abs(e["ctx"]["W_prev"] - W) < 1e-9
            W += e["z"] * p.sig_g[e["ctx"]["k"] - 1] + p.mu_g[e["ctx"]["k"] - 1]


def test_null_day_never_flags_drifted():
    p = _params()
    rng = np.random.default_rng(11)
    events, _ = simulate_route_day(p, DriftSpec(kind="none"), rng)
    assert not any(e["ctx"]["drifted"] for e in events)

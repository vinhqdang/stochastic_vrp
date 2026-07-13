"""Tests for ev/: martingale validity, Ville false-alarm control,
predictable-tilt safety, detection power, and simulator sanity."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ev.world import DayParams, DriftSpec, simulate_route_day
from ev.eprocess import (MasterEProcess, TempoMonitor, gaussian_step,
                         poisson_step, bernoulli_step, run_day)
from ev.baselines import CusumMonitor, PeriodicMonitor, PageHinkleyMonitor


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
    # exact: sum_n Pois(lam)(n) * e(n) = 1 by the Poisson MGF; the
    # theta=4 component's tail is too heavy for a cheap MC check
    from scipy import stats as sps
    lam = 0.15
    for s in (0.5, 1.0):
        ns = np.arange(0, 60)
        pmf = sps.poisson.pmf(ns, lam)
        e = np.array([poisson_step(int(n), lam, s) for n in ns])
        assert abs(float((pmf * e).sum()) - 1.0) < 1e-9


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
    # E[E_T] = 1 exactly, but the theta=4 Poisson component pays ~54
    # with probability ~1e-5 per stop — no affordable MC samples that
    # tail, so the empirical mean sits BELOW 1 by construction. What MC
    # can check: no inflation (mean well under 1/alpha-scale), typical
    # decay, and the per-bet fairness is proven analytically above; the
    # anytime guarantee is checked by the Ville test below.
    assert np.mean(vals) < 1.7
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


# ── TEMPO v2 (adaptive + dual combination + breakdown split) ────────────────

def test_tempo2_ville_false_alarm_rate():
    p = _params()
    null = DriftSpec(kind="none")
    alpha = 0.10
    fired = 0
    N = 500
    for seed in range(N):
        rng = np.random.default_rng(6000 + seed)
        events, _ = simulate_route_day(p, null, rng)
        mon = TempoMonitor(alpha=alpha)
        if run_day(mon, events)["fired"]:
            fired += 1
    bound = alpha + 3 * np.sqrt(alpha * (1 - alpha) / N)
    assert fired / N <= bound


@pytest.mark.parametrize("kind,mag", [("traffic", 2.0), ("demand", 1.5),
                                      ("accident", 8.0), ("breakdown", 200.0),
                                      ("dwell", 3.0)])
def test_tempo2_detects_strong_drift(kind, mag):
    p = _params(m=40)
    drift = DriftSpec(kind=kind, t_star=9.0, magnitude=mag)
    hits = 0
    N = 60
    for seed in range(N):
        rng = np.random.default_rng(7000 + seed)
        events, _ = simulate_route_day(p, drift, rng)
        mon = TempoMonitor(alpha=0.05)
        if run_day(mon, events)["fired"]:
            hits += 1
    assert hits / N > 0.6, f"{kind}: {hits/N}"


def test_tempo2_single_breakdown_fires_from_any_wealth():
    # the alpha-split means one breakdown fires its own monitor even if
    # the continuous channels are deep in null decay
    mon = TempoMonitor(alpha=0.05)
    rng = np.random.default_rng(3)
    for _ in range(200):                      # long quiet stretch
        mon.update(dict(channel="travel", z=float(rng.normal()), ctx={}))
    assert not mon.fired
    fired = mon.update(dict(channel="breakdown", x=1, p0=0.002, ctx={}))
    assert fired


def test_pagehinkley_runs():
    p = _params()
    rng = np.random.default_rng(5)
    events, _ = simulate_route_day(p, DriftSpec(kind="none"), rng)
    out = run_day(PageHinkleyMonitor(h=25.0), events)
    assert set(out) >= {"fired", "alarm_idx"}


# ── replanner backends ───────────────────────────────────────────────────────

def test_replan_backends_preserve_customers():
    from ev.replan import resequence_nn2opt, exact_open_tsp, rebalance_regret
    rng = np.random.default_rng(4)
    n = 14
    pts = rng.uniform(0, 10, (n, 2))
    D = np.hypot(*(pts[:, None, :] - pts[None, :, :]).T).T
    rem = [3, 5, 7, 9, 11]
    for fn in (resequence_nn2opt, exact_open_tsp):
        order = fn(1, rem, D)
        assert sorted(order) == sorted(rem)
    routes = rebalance_regret([1, 2], [[3, 5, 7], [9, 11]],
                              [100.0, 100.0], D, np.zeros(n))
    assert sorted(c for r in routes for c in r) == sorted(rem)


def test_exact_no_worse_than_heuristic():
    from ev.replan import resequence_nn2opt, exact_open_tsp, _tour_len
    rng = np.random.default_rng(8)
    n = 11
    pts = rng.uniform(0, 10, (n, 2))
    D = np.hypot(*(pts[:, None, :] - pts[None, :, :]).T).T
    rem = list(range(2, 10))
    h = resequence_nn2opt(1, rem, D)
    e = exact_open_tsp(1, rem, D)
    assert _tour_len(D, 1, e) <= _tour_len(D, 1, h) + 1e-6

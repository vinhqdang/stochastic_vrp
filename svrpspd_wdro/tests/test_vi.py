"""Tests for ev/vi.py: exact route value, value innovations, and the
value-innovation e-process monitor. All randomness is seeded."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.costs import fit_lsm_actions, simulate_actions  # noqa: E402
from ev.vi import RouteValue, ValueInnovationMonitor, run_route_day  # noqa: E402

M = 12
PBAR = np.linspace(4.0, 10.0, M)
DBAR = np.linspace(9.0, 3.0, M)
B = 18.0
H = np.linspace(40.0, 25.0, M)
E = np.linspace(90.0, 50.0, M)
R = np.linspace(20.0, 8.0, M)
CV = 0.30


def _sample_g(rng, N, pbar, dbar, cv=CV, p_scale=1.0):
    """Independent gamma pickups minus deliveries, shape 1/cv^2, mean-matched."""
    def draw(mean):
        out = np.zeros((N, len(mean)))
        for i, mu in enumerate(mean):
            if mu > 0:
                out[:, i] = rng.gamma(1.0 / cv ** 2, mu * cv ** 2, N)
        return out
    return draw(np.asarray(pbar) * p_scale) - draw(dbar)


@pytest.fixture(scope="module")
def fitted():
    g_train = _sample_g(np.random.default_rng(1), 1000, PBAR, DBAR)
    models = fit_lsm_actions(g_train, B, H, E, R)
    rv = RouteValue(PBAR, DBAR, B, H, E, R, models, cv=CV, G=1500)
    return models, rv


def _alarm_fraction(rv, g_rows, alpha=0.05, eps=0.0):
    mon = ValueInnovationMonitor(alpha=alpha)
    n_alarm = 0
    for row in g_rows:
        mon.reset()
        out = run_route_day(rv, row, monitor=mon, eps=eps)
        n_alarm += out["alarm_stop"] is not None
    return n_alarm / len(g_rows)


def _bound(alpha=0.05, n=2000):
    return alpha + 2.0 * np.sqrt(alpha * (1 - alpha) / n)


def test_expected_cost_matches_simulation(fitted):
    models, rv = fitted
    g_test = _sample_g(np.random.default_rng(2), 200_000, PBAR, DBAR)
    st = simulate_actions(g_test, B, H, E, R, models)
    sim = st["mean_cost"]
    se = st["costs"].std(ddof=1) / np.sqrt(len(g_test))
    exact = rv.expected_cost()
    tol = max(0.02 * abs(sim), 3.0 * se)
    assert abs(exact - sim) <= tol, (exact, sim, se)


def test_innovations_have_mean_zero_under_planning_law(fitted):
    _, rv = fitted
    g_rows = _sample_g(np.random.default_rng(3), 20_000, PBAR, DBAR)
    vals = []
    for row in g_rows:
        vals.extend(run_route_day(rv, row)["innovations"])
    vals = np.asarray(vals)
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    assert abs(vals.mean()) <= 3.0 * se, (vals.mean(), se)


def test_monitor_false_alarm_under_planning_law(fitted):
    _, rv = fitted
    g_rows = _sample_g(np.random.default_rng(4), 2000, PBAR, DBAR)
    frac = _alarm_fraction(rv, g_rows, alpha=0.05, eps=0.0)
    print(f"planning-law alarm fraction: {frac:.4f}")
    assert frac <= _bound()


def test_harmless_drift_lighter_pickups_no_excess_alarms(fitted):
    _, rv = fitted
    g_rows = _sample_g(np.random.default_rng(5), 2000, PBAR, DBAR, p_scale=0.7)
    frac = _alarm_fraction(rv, g_rows, alpha=0.05, eps=0.0)
    print(f"lighter-pickup alarm fraction: {frac:.4f}")
    assert frac <= _bound()


def test_harmful_drift_heavier_pickups_alarms(fitted):
    """Fleet-pooled monitor (8 routes, events interleaved by stop):
    a 30% pickup increase is detected on almost every day. A single
    12-stop route carries too few events for power; the monitor is
    meant to pool a fleet."""
    from ev.vi import ValueInnovationMonitor
    models, rv = fitted
    K, days = 8, 300
    rng = np.random.default_rng(6)
    alarms = 0
    for _ in range(days):
        mon = ValueInnovationMonitor(alpha=0.05)
        g = _sample_g(rng, K, PBAR, DBAR, p_scale=1.3)
        W = np.zeros(K)
        alive = np.ones(K, bool)
        fired = False
        for k in range(1, M + 1):
            for r in range(K):
                if not alive[r]:
                    continue
                I, L = rv.innovation(k, W[r], g[r, k - 1])
                fired |= mon.update(I, L)
                W[r] += g[r, k - 1]
                if W[r] > B:
                    alive[r] = False
                    continue
                if k < M and k in models:
                    iso, F = models[k]
                    ch = iso.predict([W[r]])[0]
                    vh, vr = H[k - 1], R[k - 1] + F
                    if vh < ch and vh <= vr:
                        alive[r] = False
                    elif vr < ch:
                        W[r] = 0.0
        alarms += fired
    assert alarms / days >= 0.8


def test_innovation_lower_bound_random_inputs(fitted):
    _, rv = fitted
    rng = np.random.default_rng(7)
    for _ in range(5000):
        k = int(rng.integers(1, M + 1))
        w = rng.uniform(rv.W_lo, B)
        gmin, gmax = rv.g_support(k)
        g = rng.uniform(gmin, gmax)
        I, L = rv.innovation(k, w, g)
        assert L >= -1e-6, (k, w, g, L)
        assert I >= -L - 1e-9, (k, w, g, I, L)
    # Planning-law demands at random loads.
    for _ in range(500):
        k = int(rng.integers(1, M + 1))
        w = rng.uniform(rv.W_lo, B)
        g = _sample_g(rng, 1, PBAR, DBAR)[0, k - 1]
        I, L = rv.innovation(k, w, g)
        assert I >= -L - 1e-9

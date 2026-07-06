"""Tests for OTR-2.0 (peak-aware labels + optimal-stopping trigger)."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.otr_endpoint import fit_otr, simulate_fast, tune_tau_fast, tau_myopic
from core.otr2 import (
    _overflow_step,
    fit_lsm,
    otr_route_v2,
    calibrate_B_empirical_peak,
    simulate_v2,
    _simulate_costs_v2,
    validate,
    fit_otr_peak,
    oracle_costs,
    simulate_oracle,
)

M = 8
N_HIST = 4000
OMEGA_F = 1.0
CFAIL = 5.0


def _make_data(n, m, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(-0.05, 0.5, (n, m))


def _make_collect_then_deliver(n, m, seed=0):
    """First half pickups, second half the exact same amounts delivered back.

    Endpoint W_m == 0 for EVERY route; the peak is the sum of pickups and
    varies across routes. This is the structure that breaks v1's endpoint
    label (all labels zero) while peak overflow is materially nonzero.
    """
    rng = np.random.default_rng(seed)
    half = m // 2
    pickups = rng.gamma(4.0, 0.25, (n, half))       # mean 1 per pickup stop
    g = np.concatenate([pickups, -pickups[:, ::-1]], axis=1)
    return g


@pytest.fixture(scope="module")
def g_hist():
    return _make_data(N_HIST, M)


@pytest.fixture(scope="module")
def B(g_hist):
    return calibrate_B_empirical_peak(g_hist, alpha=0.10)


@pytest.fixture(scope="module")
def cont_models(g_hist, B):
    return fit_lsm(g_hist, B, OMEGA_F, CFAIL)


# ─────────────────────────────────────────────
# _overflow_step
# ─────────────────────────────────────────────

class TestOverflowStep:
    def test_basic(self):
        cum = np.array([[0.5, 1.5, 0.2],    # overflows at step 2
                        [0.2, 0.3, 0.4],    # never
                        [2.0, 0.1, 0.1]])   # step 1
        ostep = _overflow_step(cum, B=1.0)
        assert list(ostep) == [2, 4, 1]

    def test_never_is_m_plus_1(self):
        cum = np.zeros((5, 3))
        assert (_overflow_step(cum, B=1.0) == 4).all()

    def test_boundary_not_overflow(self):
        """W == B exactly is NOT an overflow (strict inequality)."""
        cum = np.full((1, 3), 1.0)
        assert _overflow_step(cum, B=1.0)[0] == 4


# ─────────────────────────────────────────────
# calibrate_B_empirical_peak
# ─────────────────────────────────────────────

class TestCalibratePeak:
    def test_peak_overflow_rate_matches_alpha(self):
        g = _make_data(5000, M, seed=1)
        B = calibrate_B_empirical_peak(g, alpha=0.10)
        peak = np.cumsum(g, axis=1).max(axis=1)
        assert abs(float((peak > B).mean()) - 0.10) < 0.02

    def test_collect_then_deliver_nondegenerate(self):
        """On collect-then-deliver data the endpoint quantile is useless
        (all endpoints 0) but the peak quantile is meaningful."""
        g = _make_collect_then_deliver(5000, M, seed=2)
        B_peak = calibrate_B_empirical_peak(g, alpha=0.10)
        endpoint_q = float(np.quantile(g.sum(axis=1), 0.90))   # v1 approach
        assert abs(endpoint_q) < 1e-9          # v1's B is degenerate (~0)
        assert B_peak > 1.0                    # v2's B is a real capacity level
        peak = np.cumsum(g, axis=1).max(axis=1)
        assert abs(float((peak > B_peak).mean()) - 0.10) < 0.02


# ─────────────────────────────────────────────
# fit_lsm
# ─────────────────────────────────────────────

class TestFitLsm:
    def test_keys(self, cont_models):
        assert set(cont_models.keys()) == set(range(1, M))

    def test_monotone(self, cont_models):
        w = np.linspace(-3.0, 3.0, 200)
        for mdl in cont_models.values():
            preds = mdl.predict(w)
            assert np.all(np.diff(preds) >= -1e-12)

    def test_costs_bounded(self, cont_models):
        """Continuation cost lives in [0, Cfail]."""
        w = np.linspace(-5.0, 5.0, 100)
        for mdl in cont_models.values():
            preds = mdl.predict(w)
            assert preds.min() >= -1e-9
            assert preds.max() <= CFAIL + 1e-9

    def test_no_overflow_predicts_zero(self):
        """If no training path ever overflows, continuing is free."""
        g = np.full((500, 5), -0.1)
        mods = fit_lsm(g, B=100.0, omegaF=OMEGA_F, Cfail=CFAIL)
        for mdl in mods.values():
            assert mdl.predict(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_all_overflow_late_predicts_high(self):
        """Deterministic ramp that overflows at the last step. At k=m-1 the
        continuation cost is Cfail (> omegaF), triggering handoff there. At
        earlier steps the cost of continuing optimally is exactly omegaF
        (hand off later), so v2 correctly waits — option value at work."""
        g = np.ones((500, 5)) * 1.0 + np.random.default_rng(3).normal(0, 0.01, (500, 5))
        mods = fit_lsm(g, B=4.5, omegaF=OMEGA_F, Cfail=CFAIL)   # overflow at k=5
        chat_last = mods[4].predict(np.array([4.0]))[0]
        assert chat_last > OMEGA_F                      # doomed if it continues
        chat1 = mods[1].predict(np.array([1.0]))[0]
        assert chat1 == pytest.approx(OMEGA_F)          # can still act later

    def test_small_alive_set_no_crash(self):
        """Degenerate: nearly every path overflows at step 1."""
        g = np.ones((50, 4)) * 2.0
        g[0] = -1.0   # a single survivor
        mods = fit_lsm(g, B=1.0, omegaF=OMEGA_F, Cfail=CFAIL)
        assert set(mods.keys()) == {1, 2, 3}
        for mdl in mods.values():
            assert np.isfinite(mdl.predict(np.array([0.0]))[0])


# ─────────────────────────────────────────────
# otr_route_v2
# ─────────────────────────────────────────────

class TestOtrRouteV2:
    def _obs(self, g_vec):
        return lambda k: (0.0, g_vec[k - 1])

    def test_complete(self, cont_models, B):
        g = np.full(M, -0.5)
        action, cost, at = otr_route_v2(self._obs(g), M, B, OMEGA_F, CFAIL, cont_models)
        assert (action, cost, at) == ("COMPLETE", 0.0, M)

    def test_emergency(self, cont_models):
        g = np.full(M, 1.0)
        action, cost, at = otr_route_v2(self._obs(g), M, 0.0, OMEGA_F, CFAIL, cont_models)
        assert (action, cost, at) == ("EMERGENCY", CFAIL, 1)

    def test_handoff_on_doomed_route(self):
        """Deterministic ramp: v2 must hand off before the physical overflow."""
        g_hist = np.ones((2000, M)) + np.random.default_rng(4).normal(0, 0.05, (2000, M))
        Bv = M - 1.5                       # overflow at the last step
        mods = fit_lsm(g_hist, Bv, OMEGA_F, CFAIL)
        g = np.ones(M)
        action, cost, at = otr_route_v2(self._obs(g), M, Bv, OMEGA_F, CFAIL, mods)
        assert action == "HANDOFF"
        assert cost == OMEGA_F
        assert at < M

    def test_observe_once_per_customer(self, cont_models, B):
        calls = {}

        def obs(k):
            calls[k] = calls.get(k, 0) + 1
            return 0.0, -0.2

        otr_route_v2(obs, M, B, OMEGA_F, CFAIL, cont_models)
        assert all(c == 1 for c in calls.values())

    def test_matches_vectorized(self, cont_models, B, g_hist):
        """Scalar loop and vectorized batch must agree scenario-by-scenario."""
        g = g_hist[:300]
        vec = _simulate_costs_v2(g, B, OMEGA_F, CFAIL, cont_models)
        for s in range(g.shape[0]):
            _, cost, _ = otr_route_v2(self._obs(g[s]), M, B, OMEGA_F, CFAIL, cont_models)
            assert cost == pytest.approx(vec[s])


# ─────────────────────────────────────────────
# simulate_v2
# ─────────────────────────────────────────────

class TestSimulateV2:
    def test_rates_sum_to_one(self, cont_models, B):
        g = _make_data(500, M, seed=7)
        stats = simulate_v2(g, B, OMEGA_F, CFAIL, cont_models)
        total = stats["handoff_rate"] + stats["fail_rate"] + stats["complete_rate"]
        assert total == pytest.approx(1.0)

    def test_cost_identity(self, cont_models, B):
        g = _make_data(500, M, seed=8)
        stats = simulate_v2(g, B, OMEGA_F, CFAIL, cont_models)
        expected = stats["handoff_rate"] * OMEGA_F + stats["fail_rate"] * CFAIL
        assert stats["mean_cost"] == pytest.approx(expected)

    def test_beats_reactive(self, g_hist, B, cont_models):
        """v2 must undercut the pure reactive policy on iid test data."""
        g_test = _make_data(20000, M, seed=9)
        stats = simulate_v2(g_test, B, OMEGA_F, CFAIL, cont_models)
        peak = np.cumsum(g_test, axis=1).max(axis=1)
        reactive = CFAIL * float((peak > B).mean())
        assert stats["mean_cost"] < reactive


# ─────────────────────────────────────────────
# The headline regression: collect-then-deliver breaks v1, not v2
# ─────────────────────────────────────────────

class TestCollectThenDeliver:
    def test_v1_zero_saving_v2_large_saving(self):
        g_tr = _make_collect_then_deliver(6000, M, seed=10)
        g_te = _make_collect_then_deliver(12000, M, seed=11)
        B = calibrate_B_empirical_peak(g_tr, alpha=0.10)

        # v1: endpoint labels are all zero -> never triggers -> reactive cost
        v1_models = fit_otr(g_tr, B)
        tau = tune_tau_fast(g_tr, B, v1_models, OMEGA_F, CFAIL)
        v1 = simulate_fast(g_te, B, tau, OMEGA_F, CFAIL, v1_models)
        none = simulate_fast(g_te, B, 1.0, OMEGA_F, CFAIL, v1_models)
        assert v1["handoff_rate"] == pytest.approx(0.0, abs=1e-6)
        assert v1["mean_cost"] == pytest.approx(none["mean_cost"], rel=1e-6)

        # v2: peak-aware LSM triggers and saves materially
        cm = fit_lsm(g_tr, B, OMEGA_F, CFAIL)
        v2 = simulate_v2(g_te, B, OMEGA_F, CFAIL, cm)
        saving = (none["mean_cost"] - v2["mean_cost"]) / none["mean_cost"]
        assert v2["handoff_rate"] > 0.0
        assert saving > 0.30    # spec reports ~74%; require a solid margin

    def test_fallback_also_saves(self):
        g_tr = _make_collect_then_deliver(400, M, seed=12)   # small-N regime
        g_te = _make_collect_then_deliver(8000, M, seed=13)
        B = calibrate_B_empirical_peak(g_tr, alpha=0.10)
        mods = fit_otr_peak(g_tr, B)
        tau = tune_tau_fast(g_tr, B, mods, OMEGA_F, CFAIL)
        fb = simulate_fast(g_te, B, tau, OMEGA_F, CFAIL, mods)
        none = simulate_fast(g_te, B, 1.0, OMEGA_F, CFAIL, mods)
        assert fb["mean_cost"] < none["mean_cost"]


# ─────────────────────────────────────────────
# validate
# ─────────────────────────────────────────────

class TestValidate:
    def test_iid_data(self, g_hist, B, cont_models):
        g_val = _make_data(10000, M, seed=14)
        rep = validate(g_val, B, OMEGA_F, CFAIL, cont_models)
        assert rep["peak_overflow_rate"] > 0.005
        assert rep["policy_cost"] <= rep["reactive_cost"]
        assert isinstance(rep["deploy_ok"], bool)

    def test_collect_then_deliver_low_corr(self):
        """Endpoint-peak correlation is undefined/low when endpoints collapse."""
        g_tr = _make_collect_then_deliver(4000, M, seed=15)
        g_val = _make_collect_then_deliver(4000, M, seed=16)
        B = calibrate_B_empirical_peak(g_tr, alpha=0.10)
        cm = fit_lsm(g_tr, B, OMEGA_F, CFAIL)
        rep = validate(g_val, B, OMEGA_F, CFAIL, cm)
        # endpoints are constant 0 -> corr is nan (flagged degenerate)
        assert np.isnan(rep["corr_endpoint_peak"]) or rep["corr_endpoint_peak"] < 0.5
        assert rep["deploy_ok"]

    def test_v2_dominates_v1_tuned_on_ctd(self):
        """v2 must beat even the tuned v1 policy on collect-then-deliver."""
        g_tr = _make_collect_then_deliver(6000, M, seed=17)
        g_te = _make_collect_then_deliver(12000, M, seed=18)
        B = calibrate_B_empirical_peak(g_tr, alpha=0.10)
        v1_models = fit_otr(g_tr, B)
        tau = tune_tau_fast(g_tr, B, v1_models, OMEGA_F, CFAIL)
        v1 = simulate_fast(g_te, B, tau, OMEGA_F, CFAIL, v1_models)
        cm = fit_lsm(g_tr, B, OMEGA_F, CFAIL)
        v2 = simulate_v2(g_te, B, OMEGA_F, CFAIL, cm)
        assert v2["mean_cost"] < v1["mean_cost"]


# ─────────────────────────────────────────────
# Clairvoyant oracle
# ─────────────────────────────────────────────

class TestOracle:
    def test_costs(self):
        g = np.array([[0.2, 0.2, 0.2],     # never overflows (B=1): 0
                      [0.5, 0.8, -0.5],    # overflows at step 2: omegaF
                      [2.0, 0.0, 0.0]])    # overflows at step 1: Cfail
        costs = oracle_costs(g, B=1.0, omegaF=1.0, Cfail=5.0)
        assert list(costs) == [0.0, 1.0, 5.0]

    def test_lower_bounds_every_policy(self, g_hist, B, cont_models):
        """No implementable policy may beat the oracle."""
        g_te = _make_data(20000, M, seed=21)
        orc = simulate_oracle(g_te, B, OMEGA_F, CFAIL)
        v2 = simulate_v2(g_te, B, OMEGA_F, CFAIL, cont_models)
        v1_models = fit_otr(g_hist, B)
        tau = tune_tau_fast(g_hist, B, v1_models, OMEGA_F, CFAIL)
        v1 = simulate_fast(g_te, B, tau, OMEGA_F, CFAIL, v1_models)
        none = simulate_fast(g_te, B, 1.0, OMEGA_F, CFAIL, v1_models)
        for stats in (v2, v1, none):
            assert orc["mean_cost"] <= stats["mean_cost"] + 1e-12


# ─────────────────────────────────────────────
# fit_otr_peak (fallback)
# ─────────────────────────────────────────────

class TestFitOtrPeak:
    def test_keys_and_range(self):
        g = _make_data(1000, M, seed=19)
        B = calibrate_B_empirical_peak(g, alpha=0.10)
        mods = fit_otr_peak(g, B)
        assert set(mods.keys()) == set(range(1, M))
        w = np.linspace(-3, 3, 50)
        for mdl in mods.values():
            p = mdl.predict(w)
            assert p.min() >= -1e-9 and p.max() <= 1.0 + 1e-9
            assert np.all(np.diff(p) >= -1e-12)

    def test_m1_route(self):
        g = _make_data(100, 1, seed=20)
        assert fit_otr_peak(g, B=0.5) == {}
        assert fit_lsm(g, 0.5, OMEGA_F, CFAIL) == {}

"""Tests for the OTR (Online Threshold Reassignment) algorithm."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.otr_endpoint import (
    fit_otr,
    otr_route,
    calibrate_B,
    calibrate_B_empirical,
    tau_myopic,
    tune_tau,
    calibration_rmse,
    simulate,
    fit_gaussian_params,
    gaussian_p,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

RNG = np.random.default_rng(0)
M = 8
N_HIST = 2000


def _make_data(n, m, rng=None, seed=0):
    if rng is None:
        rng = np.random.default_rng(seed)
    return rng.normal(-0.05, 0.5, (n, m))


@pytest.fixture(scope="module")
def g_hist():
    return _make_data(N_HIST, M)


@pytest.fixture(scope="module")
def B(g_hist):
    return calibrate_B_empirical(g_hist, alpha=0.10)


@pytest.fixture(scope="module")
def models(g_hist, B):
    return fit_otr(g_hist, B)


# ─────────────────────────────────────────────
# fit_otr
# ─────────────────────────────────────────────

class TestFitOtr:
    def test_keys(self, models):
        assert set(models.keys()) == set(range(1, M))

    def test_predictions_in_01(self, models, g_hist, B):
        cum = np.cumsum(g_hist, axis=1)
        for k, iso in models.items():
            preds = iso.predict(cum[:, k - 1])
            assert preds.min() >= 0.0 - 1e-9
            assert preds.max() <= 1.0 + 1e-9

    def test_monotone(self, models):
        """Isotonic fit must be non-decreasing."""
        test_w = np.linspace(-3.0, 3.0, 200)
        for iso in models.values():
            preds = iso.predict(test_w)
            assert np.all(np.diff(preds) >= -1e-12), "model not monotone"

    def test_all_overflow_constant_one(self):
        """When B is very small, all routes overflow — model predicts ~1."""
        g = np.ones((500, 5)) * 0.1
        B_small = -100.0
        mods = fit_otr(g, B_small)
        w_high = np.array([10.0])
        for iso in mods.values():
            assert iso.predict(w_high)[0] > 0.5

    def test_no_overflow_constant_zero(self):
        """When B is very large, no routes overflow — model predicts ~0."""
        g = np.ones((500, 5)) * 0.1
        B_large = 1e6
        mods = fit_otr(g, B_large)
        w_low = np.array([-100.0])
        for iso in mods.values():
            assert iso.predict(w_low)[0] < 0.5


# ─────────────────────────────────────────────
# otr_route
# ─────────────────────────────────────────────

class TestOtrRoute:
    def _obs_from_g(self, g_vec):
        return lambda k: (0.0, g_vec[k - 1])

    def test_complete_no_overflow(self, models, B):
        """A route where W << B should complete cleanly."""
        g = np.full(M, -0.5)  # strong negative net increments
        action, cost, at = otr_route(self._obs_from_g(g), M, B, tau=0.5,
                                     omegaF=1.0, Cfail=5.0, models=models)
        assert action == "COMPLETE"
        assert cost == 0.0
        assert at == M

    def test_emergency_immediate(self, models):
        """Route that immediately overflows regardless of threshold."""
        B_tight = 0.0
        g = np.full(M, 1.0)  # every customer adds 1, so W_1 = 1 > B=0
        action, cost, at = otr_route(self._obs_from_g(g), M, B_tight, tau=0.9,
                                     omegaF=1.0, Cfail=5.0, models=models)
        assert action == "EMERGENCY"
        assert cost == 5.0
        assert at == 1

    def test_handoff_triggered(self, models, B):
        """tau=0 means any nonzero predicted probability triggers handoff."""
        g = np.full(M, 1.0)  # high overflow risk route
        action, cost, at = otr_route(self._obs_from_g(g), M, B, tau=0.0,
                                     omegaF=1.0, Cfail=5.0, models=models)
        # Either HANDOFF at some k < M, or EMERGENCY if Wk > B before model fires
        assert action in ("HANDOFF", "EMERGENCY")

    def test_tau_one_never_handoff(self, models, B):
        """tau=1 never triggers proactively — falls through to COMPLETE or EMERGENCY."""
        g = np.full(M, -0.1)  # mild negative increments — safe route
        action, cost, at = otr_route(self._obs_from_g(g), M, B, tau=1.0,
                                     omegaF=1.0, Cfail=5.0, models=models)
        assert action in ("COMPLETE", "EMERGENCY")
        assert action != "HANDOFF"

    def test_stopped_at_range(self, models, B):
        """stopped_at must always be in [1, m]."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            g = rng.normal(0, 0.5, M)
            _, _, at = otr_route(self._obs_from_g(g), M, B, tau=0.3,
                                 omegaF=1.0, Cfail=5.0, models=models)
            assert 1 <= at <= M

    def test_observe_called_exactly_once_per_customer(self, models, B):
        """observe(k) must be called exactly once for each visited customer."""
        call_counts = {}

        def counting_obs(k):
            call_counts[k] = call_counts.get(k, 0) + 1
            return 0.0, -0.2   # safe net increment

        otr_route(counting_obs, M, B, tau=0.9, omegaF=1.0, Cfail=5.0, models=models)
        for k, cnt in call_counts.items():
            assert cnt == 1, f"observe({k}) called {cnt} times"


# ─────────────────────────────────────────────
# calibrate_B
# ─────────────────────────────────────────────

class TestCalibrateB:
    def test_exact(self):
        assert calibrate_B(Q=10.0, L0=3.0) == pytest.approx(7.0)

    def test_empirical_overflow_rate(self):
        rng = np.random.default_rng(1)
        g = rng.normal(0, 1, (5000, 5))
        B = calibrate_B_empirical(g, alpha=0.10)
        W = g.sum(axis=1)
        empirical_rate = float((W > B).mean())
        assert abs(empirical_rate - 0.10) < 0.02

    def test_empirical_alpha_zero(self):
        g = np.ones((100, 3))
        B = calibrate_B_empirical(g, alpha=0.0)
        assert B == pytest.approx(g.sum(axis=1).max())


# ─────────────────────────────────────────────
# tau_myopic
# ─────────────────────────────────────────────

class TestTauMyopic:
    def test_basic(self):
        assert tau_myopic(1.0, 5.0) == pytest.approx(0.2)

    def test_in_01(self):
        for omegaF, Cfail in [(0.5, 2.0), (1.0, 10.0), (2.0, 3.0)]:
            tau = tau_myopic(omegaF, Cfail)
            assert 0.0 < tau < 1.0


# ─────────────────────────────────────────────
# tune_tau
# ─────────────────────────────────────────────

class TestTuneTau:
    def test_returns_from_grid(self, g_hist, B, models):
        grid = np.array([0.1, 0.3, 0.5, 0.7])
        tau = tune_tau(g_hist[:200], B, models, omegaF=1.0, Cfail=5.0, tau_grid=grid)
        assert tau in grid

    def test_cost_not_worse_than_tau1(self, g_hist, B, models):
        """Tuned tau should not be worse (higher cost) than a passive tau=1 policy."""
        tau_tuned = tune_tau(g_hist[:500], B, models, omegaF=1.0, Cfail=5.0)
        stats_tuned = simulate(g_hist[500:600], B, tau_tuned, 1.0, 5.0, models)
        stats_passive = simulate(g_hist[500:600], B, 1.0, 1.0, 5.0, models)
        # Tuned tau should not have a dramatically higher cost
        assert stats_tuned["mean_cost"] <= stats_passive["mean_cost"] * 1.5


# ─────────────────────────────────────────────
# calibration_rmse
# ─────────────────────────────────────────────

class TestCalibrationRmse:
    def test_returns_float(self, g_hist, B, models):
        rmse = calibration_rmse(g_hist, B, models)
        assert isinstance(rmse, float)
        assert rmse >= 0.0 or np.isnan(rmse)

    def test_self_calibration_low(self, g_hist, B, models):
        """Models calibrated on g_hist should have low RMSE on g_hist itself."""
        rmse = calibration_rmse(g_hist, B, models)
        if not np.isnan(rmse):
            assert rmse < 0.15


# ─────────────────────────────────────────────
# simulate
# ─────────────────────────────────────────────

class TestSimulate:
    def test_rates_sum_to_one(self, models, B):
        g = _make_data(300, M, seed=7)
        stats = simulate(g, B, tau=0.3, omegaF=1.0, Cfail=5.0, models=models)
        total = stats["handoff_rate"] + stats["fail_rate"] + stats["complete_rate"]
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_cost_consistent(self, models, B):
        """mean_cost must equal handoff_rate * omegaF + fail_rate * Cfail."""
        omegaF, Cfail = 1.0, 5.0
        g = _make_data(500, M, seed=8)
        stats = simulate(g, B, tau=0.3, omegaF=omegaF, Cfail=Cfail, models=models)
        expected = stats["handoff_rate"] * omegaF + stats["fail_rate"] * Cfail
        assert stats["mean_cost"] == pytest.approx(expected, rel=1e-6)

    def test_tau_tradeoff(self, models, B):
        """Lower tau -> more handoffs, fewer emergencies."""
        g = _make_data(500, M, seed=9)
        low = simulate(g, B, tau=0.05, omegaF=1.0, Cfail=5.0, models=models)
        high = simulate(g, B, tau=0.90, omegaF=1.0, Cfail=5.0, models=models)
        assert low["handoff_rate"] >= high["handoff_rate"]
        assert low["fail_rate"] <= high["fail_rate"]


# ─────────────────────────────────────────────
# Gaussian fallback
# ─────────────────────────────────────────────

class TestGaussianFallback:
    def test_params_shapes(self):
        g = _make_data(100, M)
        mu, sigma, rho = fit_gaussian_params(g)
        assert isinstance(mu, float)
        assert sigma > 0
        assert 0.0 < rho < 1.0

    def test_gaussian_p_in_01(self):
        g = _make_data(200, M)
        mu, sigma, rho = fit_gaussian_params(g)
        B_val = 1.0
        for k in range(1, M):
            for Wk in [-2.0, 0.0, 1.0, 3.0]:
                p = gaussian_p(k, Wk, M, mu, sigma, rho, B_val)
                assert 0.0 <= p <= 1.0, f"gaussian_p out of [0,1] at k={k}, Wk={Wk}"

    def test_gaussian_p_monotone_in_Wk(self):
        """Higher Wk should give higher overflow probability (all else equal)."""
        g = _make_data(300, M)
        mu, sigma, rho = fit_gaussian_params(g)
        B_val = 0.5
        k = 3
        Wk_vals = np.linspace(-1.5, 1.5, 20)
        probs = [gaussian_p(k, w, M, mu, sigma, rho, B_val) for w in Wk_vals]
        assert all(probs[i] <= probs[i + 1] + 1e-9 for i in range(len(probs) - 1))

    def test_single_customer_route(self):
        """Edge case: m=1 route has no online model needed. Gaussian fallback with k<m=1 never called."""
        g = _make_data(100, 1)
        mods = fit_otr(g, B=0.5)
        assert mods == {}   # no keys for m=1

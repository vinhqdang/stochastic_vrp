"""Tests for the published rule-based recourse baselines (pi1/pi2/pi3)."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.published_policies import pi_thresholds, simulate_pi, tune_pi, _simulate_costs_pi
from core.otr2 import calibrate_B_empirical_peak

M = 10
OMEGA_F, CFAIL = 1.0, 5.0
H = np.full(M, OMEGA_F)
E = np.full(M, CFAIL)


def _make_data(n, m, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(-0.05, 0.5, (n, m))


class TestThresholds:
    def test_pi1_shape_and_value(self):
        g_mean = np.zeros(M)
        thr = pi_thresholds("pi1", B=2.0, g_mean=g_mean, coef=0.25)
        assert thr.shape == (M - 1,)
        assert np.allclose(thr, 1.5)          # B - 0.25*B

    def test_coef_zero_never_triggers(self):
        g = _make_data(2000, M, seed=1)
        B = calibrate_B_empirical_peak(g, alpha=0.10)
        thr = pi_thresholds("pi1", B, g.mean(0), 0.0)
        stats = simulate_pi(g, B, H, E, thr)
        assert stats["handoff_rate"] == 0.0    # reduces to reactive

    def test_pi3_monotone_ref(self):
        """pi3's reference shrinks toward route end for positive-mean data,
        so thresholds rise (rule gets less conservative late)."""
        g_mean = np.full(M, 0.5)
        thr = pi_thresholds("pi3", B=5.0, g_mean=g_mean, coef=1.0)
        assert np.all(np.diff(thr) > 0)


class TestSimulate:
    def test_rates_sum_to_one(self):
        g = _make_data(1000, M, seed=2)
        B = calibrate_B_empirical_peak(g, alpha=0.10)
        thr = pi_thresholds("pi1", B, g.mean(0), 0.3)
        s = simulate_pi(g, B, H, E, thr)
        assert s["handoff_rate"] + s["fail_rate"] + s["complete_rate"] == pytest.approx(1.0)

    def test_cost_identity(self):
        g = _make_data(1000, M, seed=3)
        B = calibrate_B_empirical_peak(g, alpha=0.10)
        thr = pi_thresholds("pi2", B, g.mean(0), 1.0)
        s = simulate_pi(g, B, H, E, thr)
        expected = s["handoff_rate"] * OMEGA_F + s["fail_rate"] * CFAIL
        assert s["mean_cost"] == pytest.approx(expected)


class TestTune:
    def test_tuned_not_worse_than_reactive(self):
        g_tr = _make_data(4000, M, seed=4)
        g_te = _make_data(8000, M, seed=5)
        B = calibrate_B_empirical_peak(g_tr, alpha=0.10)
        for kind in ("pi1", "pi2", "pi3"):
            c = tune_pi(kind, g_tr, B, H, E)
            thr = pi_thresholds(kind, B, g_tr.mean(0), c)
            tuned = simulate_pi(g_te, B, H, E, thr)["mean_cost"]
            reactive = simulate_pi(
                g_te, B, H, E, pi_thresholds(kind, B, g_tr.mean(0), 0.0))["mean_cost"]
            assert tuned <= reactive * 1.05, kind

    def test_state_dependent_prices_respected(self):
        """With per-stop schedules the sim must charge H[k]/E[k] at stop k."""
        g = np.full((1, 4), 1.0)               # deterministic ramp
        B = 2.5                                 # overflow at k=3
        Hs = np.array([10.0, 1.0, 10.0, 10.0])
        Es = np.full(4, 50.0)
        # threshold that triggers exactly after stop 2 (W_2=2 > 1.9)
        thr = np.array([np.inf, 1.9, np.inf])
        costs = _simulate_costs_pi(g, B, Hs, Es, thr)
        assert costs[0] == pytest.approx(1.0)   # paid H[1], the cheap stop

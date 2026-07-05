"""Tests for the plug-in DP benchmark (core/dp_exec.py)."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.dp_exec import fit_dp, _fill_gaps, _BinnedModel
from core.otr2 import (
    calibrate_B_empirical_peak,
    fit_lsm,
    simulate_v2,
    _simulate_costs_v2,
    oracle_costs,
)

M = 10
OMEGA_F, CFAIL = 1.0, 5.0


def _flat(m):
    return np.full(m, OMEGA_F), np.full(m, CFAIL)


def _make_data(n, m, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(-0.05, 0.5, (n, m))


class TestFillGaps:
    def test_interior_and_edges(self):
        v = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
        out = _fill_gaps(v)
        assert list(out) == [1.0, 1.0, 1.0, 3.0, 3.0]

    def test_no_gaps_unchanged(self):
        v = np.array([1.0, 2.0])
        assert list(_fill_gaps(v)) == [1.0, 2.0]


class TestBinnedModel:
    def test_predict_bins(self):
        mdl = _BinnedModel(np.array([0.0, 1.0]), np.array([10.0, 20.0, 30.0]))
        assert list(mdl.predict([-5.0, 0.5, 99.0])) == [10.0, 20.0, 30.0]


class TestFitDp:
    def test_keys_and_interface(self):
        g = _make_data(3000, M)
        B = calibrate_B_empirical_peak(g, alpha=0.10)
        H, E = _flat(M)
        dp = fit_dp(g, B, H, E)
        assert set(dp.keys()) == set(range(1, M))
        for mdl in dp.values():
            p = mdl.predict(np.array([0.0, 1.0]))
            assert np.all(np.isfinite(p))
            assert p.min() >= -1e-9 and p.max() <= CFAIL + 1e-9

    def test_no_overflow_predicts_zero(self):
        g = np.full((500, 5), -0.1)
        H, E = _flat(5)
        dp = fit_dp(g, 100.0, H, E)
        for mdl in dp.values():
            assert mdl.predict(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_plugs_into_v2_simulator(self):
        """dp models must work inside the v2 vectorized simulator."""
        g = _make_data(3000, M, seed=1)
        B = calibrate_B_empirical_peak(g, alpha=0.10)
        H, E = _flat(M)
        dp = fit_dp(g, B, H, E)
        g_te = _make_data(5000, M, seed=2)
        stats = simulate_v2(g_te, B, OMEGA_F, CFAIL, dp)
        total = stats["handoff_rate"] + stats["fail_rate"] + stats["complete_rate"]
        assert total == pytest.approx(1.0)
        # must beat reactive on iid data
        peak = np.cumsum(g_te, axis=1).max(axis=1)
        reactive = CFAIL * float((peak > B).mean())
        assert stats["mean_cost"] < reactive

    def test_large_sample_dp_near_lsm(self):
        """With a big sample both estimators approximate the same DP:
        their test costs must be close, and both must respect the
        clairvoyant lower bound."""
        g_tr = _make_data(60_000, M, seed=3)
        g_te = _make_data(30_000, M, seed=4)
        B = calibrate_B_empirical_peak(g_tr, alpha=0.10)
        H, E = _flat(M)
        dp = fit_dp(g_tr, B, H, E)
        cm = fit_lsm(g_tr, B, OMEGA_F, CFAIL)
        c_dp = simulate_v2(g_te, B, OMEGA_F, CFAIL, dp)["mean_cost"]
        c_v2 = simulate_v2(g_te, B, OMEGA_F, CFAIL, cm)["mean_cost"]
        orc = float(oracle_costs(g_te, B, OMEGA_F, CFAIL).mean())
        assert c_dp >= orc - 1e-9 and c_v2 >= orc - 1e-9
        assert abs(c_dp - c_v2) / max(c_v2, 1e-9) < 0.10

    def test_small_sample_dp_noisier_than_lsm(self):
        """At small N the histogram DP should not materially beat the
        isotonic LSM (shape constraint = variance reduction)."""
        g_te = _make_data(30_000, M, seed=6)
        diffs = []
        for seed in range(5):
            g_tr = _make_data(400, M, seed=100 + seed)
            B = calibrate_B_empirical_peak(g_tr, alpha=0.10)
            H, E = _flat(M)
            dp = fit_dp(g_tr, B, H, E)
            cm = fit_lsm(g_tr, B, OMEGA_F, CFAIL)
            c_dp = simulate_v2(g_te, B, OMEGA_F, CFAIL, dp)["mean_cost"]
            c_v2 = simulate_v2(g_te, B, OMEGA_F, CFAIL, cm)["mean_cost"]
            diffs.append(c_dp - c_v2)
        assert float(np.mean(diffs)) > -0.02

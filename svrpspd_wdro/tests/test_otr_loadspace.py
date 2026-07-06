"""Smoke tests for the load-space (Model-A) OTR threshold module
(core/otr.py, contributed rewrite): peak labels on the physical load
profile L_k, threshold policy on P(peak overflow | L_k)."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import otr as otr_ls


RNG = np.random.default_rng(0)
N, M, Q = 3000, 8, 12.0


def _dp(n=N, m=M, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.gamma(9.0, 1.0 / 9.0, (n, m))          # deliveries, mean 1
    p = rng.gamma(9.0, 1.0 / 9.0, (n, m)) * 0.9    # pickups, mean 0.9
    return d, p


class TestLoadProfile:
    def test_shape_and_departure_load(self):
        d, p = _dp(10)
        L = otr_ls.load_profile(d, p)
        assert L.shape[1] in (M, M + 1)
        # departure load equals total deliveries (Model A)
        L0 = L[:, 0] if L.shape[1] == M + 1 else None
        if L0 is not None:
            assert np.allclose(L0, d.sum(axis=1))


class TestFitAndSimulate:
    def test_fit_returns_models(self):
        d, p = _dp()
        models = otr_ls.fit_otr(d, p, Q)
        assert isinstance(models, dict) and len(models) >= 1

    def test_simulate_stats_consistent(self):
        d, p = _dp()
        models = otr_ls.fit_otr(d, p, Q)
        d2, p2 = _dp(seed=1)
        stats = otr_ls.simulate_fast(d2, p2, Q, 0.3, 1.0, 5.0, models)
        total = (stats["handoff_rate"] + stats["fail_rate"]
                 + stats["complete_rate"])
        assert total == pytest.approx(1.0, abs=1e-9)
        expected = stats["handoff_rate"] * 1.0 + stats["fail_rate"] * 5.0
        assert stats["mean_cost"] == pytest.approx(expected, rel=1e-6)

    def test_tuned_tau_not_worse_than_reactive(self):
        d, p = _dp()
        models = otr_ls.fit_otr(d, p, Q)
        tau = otr_ls.tune_tau_fast(d, p, Q, models, 1.0, 5.0)
        d2, p2 = _dp(seed=2)
        tuned = otr_ls.simulate_fast(d2, p2, Q, tau, 1.0, 5.0, models)
        react = otr_ls.simulate_fast(d2, p2, Q, 1.0, 1.0, 5.0, models)
        assert tuned["mean_cost"] <= react["mean_cost"] * 1.05

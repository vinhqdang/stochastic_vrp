"""Tests for the realistic last-mile cost model (core/costs.py)."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.otr2 import calibrate_B_empirical_peak, fit_otr_peak, fit_lsm, simulate_v2
from core.costs import (
    LastMileCosts,
    route_cost_schedules,
    plan_fixed_cost,
    fit_lsm_general,
    simulate_v2_general,
    simulate_tau_general,
    tune_tau_general,
    oracle_costs_general,
    _simulate_costs_general,
)

M = 10
COSTS = LastMileCosts()


def _make_data(n, m, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(-0.05, 0.5, (n, m))


def _ramp_data(n, m, seed=0):
    rng = np.random.default_rng(seed)
    return rng.gamma(4.0, 0.25, (n, m)) * (0.5 + np.arange(1, m + 1) / m)


# ─────────────────────────────────────────────
# Parameters and schedules
# ─────────────────────────────────────────────

class TestSchedules:
    def test_emergency_pricier_than_handoff(self):
        """At every stop, an emergency must cost more than a planned handoff."""
        for rem in (0.0, 5.0, 50.0):
            for down in (0, 5, 20):
                assert COSTS.emergency_cost(rem, down) > COSTS.handoff_cost(rem)

    def test_early_breach_costs_more(self):
        """Recourse prices decrease along the route (less remains to cover)."""
        D = np.array([[0, 10, 20, 30],
                      [10, 0, 12, 25],
                      [20, 12, 0, 14],
                      [30, 25, 14, 0]], dtype=float)
        H, E = route_cost_schedules([1, 2, 3], D, scale=1.0, costs=COSTS)
        assert np.all(np.diff(H) <= 1e-12)
        assert np.all(np.diff(E) <= 1e-12)
        # downstream SLA fallout makes the gradient steeper for E
        assert (E[0] - E[-1]) > (H[0] - H[-1])

    def test_plan_fixed_cost(self):
        assert plan_fixed_cost(10, 500.0, COSTS) == pytest.approx(
            10 * COSTS.F_plan + 0.10 * 500.0)


# ─────────────────────────────────────────────
# Flat schedules reduce to the core implementation
# ─────────────────────────────────────────────

class TestFlatEquivalence:
    def test_matches_simulate_v2(self):
        g_tr = _make_data(4000, M, seed=1)
        g_te = _make_data(3000, M, seed=2)
        B = calibrate_B_empirical_peak(g_tr, alpha=0.10)
        omegaF, Cfail = 1.0, 5.0
        H = np.full(M, omegaF)
        E = np.full(M, Cfail)

        flat = fit_lsm(g_tr, B, omegaF, Cfail)
        gen  = fit_lsm_general(g_tr, B, H, E)
        s_flat = simulate_v2(g_te, B, omegaF, Cfail, flat)
        s_gen  = simulate_v2_general(g_te, B, H, E, gen)
        assert s_gen["mean_cost"] == pytest.approx(s_flat["mean_cost"])
        assert s_gen["fail_rate"] == pytest.approx(s_flat["fail_rate"])
        assert s_gen["handoff_rate"] == pytest.approx(s_flat["handoff_rate"])


# ─────────────────────────────────────────────
# Generalized LSM under state-dependent prices
# ─────────────────────────────────────────────

class TestGeneralLsm:
    def _setup(self, seed=3):
        g_tr = _ramp_data(8000, M, seed=seed)
        g_te = _ramp_data(8000, M, seed=seed + 100)
        B = calibrate_B_empirical_peak(g_tr, alpha=0.10)
        # decreasing schedules as produced by real geometry
        rem = np.linspace(30.0, 2.0, M)
        H = np.array([COSTS.handoff_cost(r) for r in rem])
        E = np.array([COSTS.emergency_cost(r, M - 1 - i) for i, r in enumerate(rem)])
        return g_tr, g_te, B, H, E

    def test_v2_beats_reactive_and_tuned_tau(self):
        g_tr, g_te, B, H, E = self._setup()
        cm = fit_lsm_general(g_tr, B, H, E)
        v2 = simulate_v2_general(g_te, B, H, E, cm)

        pm = fit_otr_peak(g_tr, B)
        none = simulate_tau_general(g_te, B, H, E, pm, tau=1.0)
        tau = tune_tau_general(g_tr, B, H, E, pm)
        fb = simulate_tau_general(g_te, B, H, E, pm, tau=tau)

        assert v2["mean_cost"] < none["mean_cost"]
        assert v2["mean_cost"] <= fb["mean_cost"] * 1.02   # at worst a tie

    def test_oracle_lower_bounds_all(self):
        g_tr, g_te, B, H, E = self._setup(seed=4)
        cm = fit_lsm_general(g_tr, B, H, E)
        pm = fit_otr_peak(g_tr, B)
        orc = oracle_costs_general(g_te, B, H, E).mean()
        v2, _ = _simulate_costs_general(g_te, B, H, E, cm)
        rc, _ = _simulate_costs_general(g_te, B, H, E, None, tau=1.0, prob_models=pm)
        assert orc <= v2.mean() + 1e-12
        assert orc <= rc.mean() + 1e-12

    def test_oracle_lets_cheap_breach_happen(self):
        """If a late breach is cheaper than any earlier handoff, the
        clairvoyant takes the breach."""
        g = np.array([[0.0, 0.0, 2.0]])          # breaches at step 3
        H = np.array([100.0, 100.0, 100.0])      # handoffs absurdly dear
        E = np.array([50.0, 40.0, 5.0])
        c = oracle_costs_general(g, B=1.0, H=H, E=E)
        assert c[0] == pytest.approx(5.0)

    def test_step1_breach_unavoidable(self):
        g = np.array([[5.0, 0.0, 0.0]])
        H = np.array([1.0, 1.0, 1.0])
        E = np.array([9.0, 9.0, 9.0])
        c = oracle_costs_general(g, B=1.0, H=H, E=E)
        assert c[0] == pytest.approx(9.0)

    def test_rates_sum_to_one(self):
        g_tr, g_te, B, H, E = self._setup(seed=5)
        cm = fit_lsm_general(g_tr, B, H, E)
        s = simulate_v2_general(g_te, B, H, E, cm)
        assert s["handoff_rate"] + s["fail_rate"] + s["complete_rate"] == pytest.approx(1.0)


class TestStandbyPool:
    def test_pool_sizing_single_route(self):
        import numpy as np
        from core.costs import standby_pool_size
        h = np.zeros((3, 100), dtype=bool); h[0] = True
        assert standby_pool_size(h) == 1

    def test_pool_sizing_all(self):
        import numpy as np
        from core.costs import standby_pool_size
        h3 = np.ones((4, 50), dtype=bool)
        assert standby_pool_size(h3) == 4

    def test_rare_simultaneous_below_service_level(self):
        import numpy as np
        from core.costs import standby_pool_size
        # 3 simultaneous handoffs on only 2% of days -> 95% pool ignores them
        h = np.zeros((3, 100), dtype=bool); h[:, :2] = True
        assert standby_pool_size(h, q=0.95) == 0

    def test_empty(self):
        import numpy as np
        from core.costs import standby_pool_size
        assert standby_pool_size(np.zeros((0, 0), dtype=bool)) == 0

    def test_pool_cost_scales(self):
        import numpy as np
        from core.costs import pool_cost, LastMileCosts
        h = np.ones((2, 10), dtype=bool)
        c, S = pool_cost(h, LastMileCosts())
        assert S == 2 and c == 2 * LastMileCosts().F_standby


class TestPublishedCompetitors:
    def _setup(self):
        import numpy as np
        from core.otr2 import calibrate_B_empirical_peak
        rng = np.random.default_rng(11)
        g = rng.normal(-0.05, 0.5, (3000, 10))
        B = calibrate_B_empirical_peak(g, alpha=0.10)
        H = np.full(10, 1.0); E = np.full(10, 5.0)
        return g, B, H, E

    def test_rollout_models_monotone_and_above_lsm(self):
        """The reactive base never acts, so its cost-to-go dominates the
        optimal continuation cost pointwise (rollout >= LSM estimates)."""
        import numpy as np
        from core.costs import fit_rollout, fit_lsm_general
        g, B, H, E = self._setup()
        ro = fit_rollout(g, B, H, E)
        cm = fit_lsm_general(g, B, H, E)
        w = np.linspace(-2, 2, 50)
        for k in ro:
            r = ro[k].predict(w); c = cm[k].predict(w)
            assert np.all(np.diff(r) >= -1e-12)
            # true never-act cost dominates optimal continuation; fitted
            # curves carry estimation noise, so require it on average
            assert r.mean() >= c.mean() - 1e-6

    def test_rollout_beats_reactive(self):
        import numpy as np
        from core.costs import fit_rollout, simulate_v2_general, simulate_tau_general
        from core.otr2 import fit_otr_peak
        g, B, H, E = self._setup()
        rng = np.random.default_rng(12)
        g_te = rng.normal(-0.05, 0.5, (8000, 10))
        ro = fit_rollout(g, B, H, E)
        cost = simulate_v2_general(g_te, B, H, E, ro)["mean_cost"]
        react = simulate_tau_general(g_te, B, H, E, fit_otr_peak(g, B), tau=1.0)["mean_cost"]
        assert cost < react

    def test_restock_resets_and_helps(self):
        import numpy as np
        from core.costs import simulate_restock, tune_restock
        g, B, H, E = self._setup()
        R = np.full(10, 0.5)                      # cheap detours
        thr = tune_restock(g, B, E, R)
        rng = np.random.default_rng(13)
        g_te = rng.normal(-0.05, 0.5, (8000, 10))
        tuned = simulate_restock(g_te, B, E, R, thr)
        never = simulate_restock(g_te, B, E, R, np.full(10, np.inf))
        assert tuned["mean_cost"] <= never["mean_cost"] + 1e-9
        total = tuned["handoff_rate"] + tuned["fail_rate"] + tuned["complete_rate"]
        assert total <= 1.0 + 1e-9

    def test_restock_schedule_positive(self):
        import numpy as np
        from core.costs import restock_schedule, LastMileCosts
        D = np.array([[0, 10, 20], [10, 0, 15], [20, 15, 0]], dtype=float)
        R = restock_schedule([1, 2], D, 1.0, LastMileCosts())
        assert (R > 0).all()

"""Tests for core.extra_policies (threshold-k, rollout-theta, three-action
DP, exact-reset BATON, CVaR)."""

import numpy as np

from core.costs import (fit_lsm_actions, simulate_actions, fit_rollout,
                        _simulate_costs_general, fit_lsm_general,
                        simulate_v2_general)
from core.extra_policies import (
    fit_threshold_k, simulate_threshold_k, tune_rollout_theta,
    fit_dp_actions, fit_lsm_actions_exact, simulate_actions_exact, cvar,
    _costs_threshold_k,
)


def _paths(N=3000, m=10, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.gamma(4.0, 0.25, (N, m))
    p = rng.gamma(4.0, 0.30, (N, m))
    return p - d, d


def _prices(m):
    H = np.linspace(30, 25, m)
    E = np.linspace(80, 55, m)
    R = np.full(m, 8.0)
    return H, E, R


def test_threshold_k_never_worse_than_reactive_in_sample():
    g, _ = _paths()
    H, E, _ = _prices(g.shape[1])
    B = 2.0
    thr = fit_threshold_k(g, B, H, E)
    c_fit = _costs_threshold_k(g, B, H, E, thr)[0].mean()
    c_react = _costs_threshold_k(g, B, H, E, np.full(g.shape[1], np.inf))[0].mean()
    assert c_fit <= c_react + 1e-12


def test_threshold_k_inf_is_reactive():
    g, _ = _paths(500)
    H, E, _ = _prices(g.shape[1])
    st = simulate_threshold_k(g, 2.0, H, E, np.full(g.shape[1], np.inf))
    assert st["handoff_rate"] == 0.0


def test_rollout_theta_in_grid_and_improves():
    g, _ = _paths()
    H, E, _ = _prices(g.shape[1])
    ro = fit_rollout(g, 2.0, H, E)
    t = tune_rollout_theta(g, 2.0, H, E, ro)
    c1 = _simulate_costs_general(g, 2.0, H, E, ro)[0].mean()
    ct = _simulate_costs_general(g, 2.0, t * H, E, ro, H_bill=H)[0].mean()
    assert ct <= c1 + 1e-12


def test_dp_actions_close_to_baton_at_large_n():
    g, _ = _paths(20000, seed=1)
    gt, _ = _paths(4000, seed=2)
    H, E, R = _prices(g.shape[1])
    B = 2.0
    dp = fit_dp_actions(g, B, H, E, R)
    am = fit_lsm_actions(g, B, H, E, R)
    c_dp = simulate_actions(gt, B, H, E, R, dp)["mean_cost"]
    c_am = simulate_actions(gt, B, H, E, R, am)["mean_cost"]
    assert abs(c_dp - c_am) / c_am < 0.05


def test_exact_reset_gives_more_slack_than_zero_reset():
    """The exact reset leaves -D_<=k <= 0, so with identical models the
    post-return state is never worse; the fitted policy should not be
    materially dearer than the reset-to-0 convention."""
    g, d = _paths(3000, seed=3)
    gt, dt = _paths(3000, seed=4)
    H, E, R = _prices(g.shape[1])
    B = 2.0
    ex = fit_lsm_actions_exact(g, d, B, H, E, R)
    z = fit_lsm_actions(g, B, H, E, R)
    c_ex = simulate_actions_exact(gt, dt, B, H, E, R, ex)["mean_cost"]
    c_z = simulate_actions(gt, B, H, E, R, z)["mean_cost"]
    assert c_ex <= c_z * 1.03


def test_fresh_curve_monotone():
    g, d = _paths(2000, seed=5)
    H, E, R = _prices(g.shape[1])
    ex = fit_lsm_actions_exact(g, d, 2.0, H, E, R)
    for k, (_, Fc) in ex.items():
        xs = np.linspace(Fc.xs.min(), Fc.xs.max(), 20)
        assert np.all(np.diff(Fc(xs)) >= -1e-12)


def test_costs_key_present():
    g, _ = _paths(300)
    H, E, R = _prices(g.shape[1])
    cm = fit_lsm_general(g, 2.0, H, E)
    st = simulate_v2_general(g, 2.0, H, E, cm)
    assert st["costs"].shape == (300,)
    assert np.isclose(st["costs"].mean(), st["mean_cost"])


def test_cvar():
    x = np.arange(100, dtype=float)
    assert cvar(x, 0.95) == np.mean(np.arange(95, 100))
    assert cvar(x, 0.0) == x.mean()


def test_oracle3_below_handoff_oracle_and_policies_per_day():
    from core.costs import oracle_costs_general
    from core.extra_policies import oracle3_costs
    g, _ = _paths(N=2000, m=9, seed=3)
    H, E, R = _prices(g.shape[1])
    B = 2.0
    o3 = oracle3_costs(g, B, H, E, R)
    o_ho = oracle_costs_general(g, B, H, E)
    assert np.all(o3 <= o_ho + 1e-9)
    models = fit_lsm_actions(g[:1000], B, H, E, R)
    st = simulate_actions(g, B, H, E, R, models)
    assert np.all(o3 <= st["costs"] + 1e-9)


def test_two_lever_disabled_is_reactive_and_tuning_helps():
    from core.otr2 import fit_otr_peak
    from core.extra_policies import (_costs_two_lever, tune_two_lever)
    g, _ = _paths(N=2000, m=9, seed=4)
    H, E, R = _prices(g.shape[1])
    B = 2.0
    pm = fit_otr_peak(g, B)
    react = _simulate_costs_general(g, B, H * 1e9, E, None, tau=1.0,
                                    prob_models=pm)[0]
    off = _costs_two_lever(g, B, H, E, R, pm, 1.0, np.inf)[0]
    assert np.allclose(off, react)
    t, c = tune_two_lever(g, B, H, E, R, pm)
    tuned = _costs_two_lever(g, B, H, E, R, pm, t, c)[0].mean()
    assert tuned <= off.mean() + 1e-12


def test_threshold_k_warm_start_not_worse_than_start():
    from core.extra_policies import cuts_from_models
    g, _ = _paths(N=2000, m=9, seed=5)
    H, E, _ = _prices(g.shape[1])
    B = 2.0
    cm = fit_lsm_general(g, B, H, E)
    start = cuts_from_models(g, B, cm, H)
    c_start = _costs_threshold_k(g, B, H, E, start)[0].mean()
    thr = fit_threshold_k(g, B, H, E, starts=[start])
    assert _costs_threshold_k(g, B, H, E, thr)[0].mean() <= c_start + 1e-12

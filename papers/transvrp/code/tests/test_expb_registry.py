"""B arm registry (EXPERIMENT_B.md 10 M20): the arm tuple, the policy map, the split and
predictor classes of 3.0 and 2, and dispatch of fit_arm to each fit function."""
import numpy as np
import pytest

from transvrp.expa.paths import Paths, RouteSpec
from transvrp.expb.alp import ALPZConfig
from transvrp.expb.dos import DOSConfig
from transvrp.expb.dqn import DQNConfig
from transvrp.expb.ppo import PPOConfig
from transvrp.expb.registry import ARM_POLICY_B, ARM_SPECS, B_ARMS, fit_arm
from transvrp.expb.rlsm import RLSMConfig
from transvrp.expb.thr import THRConfig

M = 3
N = 40


def make_spec():
    return RouteSpec(inst="TOY", route_id=0, route=(1, 2, 3), m=M, B=2.0,
                     H=np.array([0.3, 0.3, 0.3]), E=np.array([1.0, 0.8, 0.6]), E0=1.0)


def make_paths(seed=3):
    rng = np.random.default_rng(seed)
    return Paths(p=rng.uniform(0.5, 1.5, (N, M)), d=rng.uniform(0.5, 1.5, (N, M)))


SMALL = {
    "thr": THRConfig(n_grid=101),
    "alpz": ALPZConfig(n_w=31),
    "rlsm": RLSMConfig(K=6),
    "dos": DOSConfig(s_min=3, s_max=6, width=6),
    "dqn": DQNConfig(device="cpu", passes=1, n_env=8, buffer=1000, batch=16, learn_start=8,
                     target_every_tr=32, ckpt_per_pass=2),
    "ppo": PPOConfig(passes=1, ckpt_per_pass=2),
}


def test_b_arms_and_policy_map_are_the_six_named_arms():
    assert B_ARMS == ("thr", "alpz", "rlsm", "dos", "dqn", "ppo")
    assert ARM_POLICY_B == {a: a for a in B_ARMS}
    assert tuple(ARM_SPECS) == B_ARMS


def test_split_and_predictor_kind_follow_section_3():
    split = {a: ARM_SPECS[a].split for a in B_ARMS}
    assert split == {"thr": False, "alpz": False, "rlsm": True, "dos": True,
                     "dqn": True, "ppo": True}
    kind = {a: ARM_SPECS[a].predictor for a in B_ARMS}
    assert kind == {"thr": "cost", "alpz": "cost", "rlsm": "cost", "dos": "surrogate",
                    "dqn": "cost", "ppo": "surrogate"}


def test_each_arm_is_bound_to_its_own_config_class():
    expected = {"thr": THRConfig, "alpz": ALPZConfig, "rlsm": RLSMConfig, "dos": DOSConfig,
                "dqn": DQNConfig, "ppo": PPOConfig}
    assert {a: ARM_SPECS[a].config_cls for a in B_ARMS} == expected


@pytest.mark.parametrize("arm", B_ARMS)
def test_fit_arm_dispatches_and_returns_the_four_values(arm):
    spec = make_spec()
    models, stats, diag, budget = fit_arm(arm, spec, make_paths(), N, 0, SMALL[arm])
    assert set(models) == {1, 2}
    assert all(len(v) >= 1 for v in models.values())
    assert stats.mu_p.shape == (M,)
    assert isinstance(diag, list) and isinstance(budget, dict)
    assert budget["unique_paths"] == N
    assert budget["device"] == "cpu" or arm == "dqn"


def test_alpz_budget_reports_the_lp_size_and_thr_budget_the_grid():
    spec = make_spec()
    _, _, _, b_alpz = fit_arm("alpz", spec, make_paths(), N, 0, SMALL["alpz"])
    _, _, _, b_thr = fit_arm("thr", spec, make_paths(), N, 0, SMALL["thr"])
    assert b_alpz["lp_rows"] > 0 and b_alpz["lp_cols"] > 0 and b_alpz["lp_nnz"] > 0
    assert b_thr["n_grid"] == 101
    assert b_thr["path_stop_reads"] == N * M


def test_fit_arm_uses_the_default_config_when_none_is_given():
    spec = make_spec()
    models, _, _, _ = fit_arm("thr", spec, make_paths(), N, 0, None)
    assert set(models) == {1, 2}


def test_fit_arm_refuses_an_unknown_arm():
    with pytest.raises(ValueError):
        fit_arm("full", make_spec(), make_paths(), N, 0, None)


def test_fit_arm_refuses_a_config_of_the_wrong_class():
    with pytest.raises(TypeError):
        fit_arm("ppo", make_spec(), make_paths(), N, 0, SMALL["thr"])

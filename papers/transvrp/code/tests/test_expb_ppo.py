"""PPO after Kadyrov et al. (EXPERIMENT_B.md 3.6, 10 M19): discounted returns, the clipped
objective, step rewards in E0 units, the action mask at stage m, the 5% start, the
no-leakage and GRU-carry properties of the deployed predictor, cached versus recomputed
predictions, one episode per update, pass-end checkpoints, determinism and learning on
toy routes whose optimal policy is known in closed form."""
import numpy as np
import pytest
import torch

from transvrp.expa.features import RouteCache, RouteStats, stage_batch
from transvrp.expa.model import NetConfig
from transvrp.expa.paths import Paths, RouteSpec
from transvrp.expa.simulate import simulate_policy
from transvrp.expb.padded import PaddedTables, SetNetMasked
from transvrp.expb.ppo import (CONTINUE, HANDOFF, GRU_HIDDEN, POOL_DIM, PPOConfig, PPONet,
                               PolicyPredictor, allowed_mask, discounted_returns, fit_ppo,
                               mask_logits, ppo_clip_loss, rollout_episode, step_reward,
                               train_ppo_record)


def make_spec(B, H, E, m):
    return RouteSpec(inst="TOY", route_id=0, route=tuple(range(1, m + 1)), m=m,
                     B=float(B), H=np.asarray(H, float), E=np.asarray(E, float),
                     E0=float(E[0]))


def make_paths(N, m, seed=1, lo=0.5, hi=2.0):
    rng = np.random.default_rng(seed)
    return Paths(p=rng.uniform(lo, hi, (N, m)), d=rng.uniform(lo, hi, (N, m)))


def tiny_cfg(**kw) -> PPOConfig:
    base = dict(lr=1e-3, passes=1, epochs=4, ckpt_per_pass=2)
    base.update(kw)
    return PPOConfig(**base)


def randomize_policy(net: PPONet, seed: int = 0) -> None:
    """Non-zero policy head, so that the hidden state is visible in the output."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        net.policy.weight.copy_(torch.randn(net.policy.weight.shape, generator=gen))
        net.policy.bias.copy_(torch.randn(net.policy.bias.shape, generator=gen))


def breach_toy(N, seed=0):
    """Two stops, one decision (k = 1). B = 1, H = (0.5, 0.5), E = (1, 1), g_2 = 0.3.
    Breach at stop 1 iff g_1 > 1 (no decision); breach at stop 2 iff g_1 > 0.7. The
    optimal decision is handoff (0.5) iff g_1 > 0.7 (continuing then costs E[1] = 1), and
    continue (0) otherwise."""
    rng = np.random.default_rng(seed)
    spec = make_spec(1.0, [0.5, 0.5], [1.0, 1.0], 2)
    d = rng.uniform(0.5, 1.0, (N, 2))
    g1 = rng.uniform(0.0, 1.2, N)
    p = np.column_stack([d[:, 0] + g1, d[:, 1] + 0.3])
    return spec, Paths(p=p, d=d), g1


def expected_toy_cost(g1, B=1.0):
    """Optimal cost per row of the breach toy (E0 = 1)."""
    cost = np.where(g1 > B, 1.0, 0.0)
    return np.where((g1 > 0.7) & (g1 <= B), 0.5, cost)


# ------------------------------------------------------------- pure functions

def test_discounted_returns_match_hand_computation_for_gamma_half_and_one():
    r = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(discounted_returns(r, 0.5), [3.0, 4.0, 4.0])
    np.testing.assert_allclose(discounted_returns(r, 1.0), [7.0, 6.0, 4.0])


def test_discounted_returns_of_empty_and_single_reward():
    assert discounted_returns(np.zeros(0), 0.5).size == 0
    np.testing.assert_allclose(discounted_returns(np.array([-0.3]), 0.5), [-0.3])


def test_clipped_loss_matches_hand_computation():
    # ratios 1.2, 0.5, 1.0, 1.5 with advantages 1, -1, 2, 1 and eps = 0.2:
    # terms 1.2, min(-0.5, -0.8) = -0.8, 2.0, min(1.5, 1.2) = 1.2; mean 0.9, loss -0.9
    logp_old = torch.log(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    logp_new = torch.log(torch.tensor([0.6, 0.25, 0.5, 0.75]))
    adv = torch.tensor([1.0, -1.0, 2.0, 1.0])
    loss = ppo_clip_loss(logp_new, logp_old, adv, 0.2)
    assert float(loss) == pytest.approx(-0.9, abs=1e-6)


def test_clipped_loss_has_no_gradient_where_the_clip_binds():
    logp_old = torch.log(torch.tensor([0.5]))
    logp_new = torch.log(torch.tensor([0.75])).requires_grad_(True)     # ratio 1.5
    loss = ppo_clip_loss(logp_new, logp_old, torch.tensor([1.0]), 0.2)  # clipped at 1.2
    loss.backward()
    assert float(logp_new.grad) == pytest.approx(0.0, abs=1e-12)


def test_clipped_loss_gradient_raises_the_probability_of_a_positive_advantage():
    logp_old = torch.log(torch.tensor([0.5]))
    logp_new = torch.log(torch.tensor([0.55])).requires_grad_(True)     # ratio 1.1, unclipped
    loss = ppo_clip_loss(logp_new, logp_old, torch.tensor([1.0]), 0.2)
    loss.backward()
    assert float(logp_new.grad) < 0.0          # descent increases log-probability


def test_step_reward_is_scaled_by_E0_and_follows_the_breach_convention():
    spec = make_spec(1.0, [0.6, 0.6, 0.6], [2.0, 4.0, 8.0], 3)   # E0 = 2
    W = np.array([0.2, 0.5, 0.9])
    # handoff at k = 1 costs H[0] = 0.6 -> reward -0.3, episode ends
    assert step_reward(spec, W, 1, HANDOFF) == (pytest.approx(-0.3), True)
    # continue at k = 1: stop 2 does not breach -> 0, the episode goes on
    assert step_reward(spec, W, 1, CONTINUE) == (0.0, False)
    # continue at k = 2 with stop 3 not breaching: completion -> 0, ends
    assert step_reward(spec, W, 2, CONTINUE) == (0.0, True)
    # continue at k = 1 when stop 2 breaches: charged E[1] = 4 -> -2, ends
    W_breach = np.array([0.2, 1.5, 1.6])
    assert step_reward(spec, W_breach, 1, CONTINUE) == (pytest.approx(-2.0), True)


def test_allowed_mask_forbids_handoff_at_the_last_stop_only():
    mask = allowed_mask(np.array([1, 2, 3]), m=3)
    np.testing.assert_array_equal(mask, [[True, True], [True, True], [True, False]])


def test_masked_logits_give_zero_handoff_probability_at_stage_m():
    logits = torch.tensor([[0.2, 5.0], [0.2, 5.0]])
    masked = mask_logits(logits, torch.tensor([1, 2]), m=2)
    probs = torch.softmax(masked, dim=1)
    assert float(probs[0, 1]) == pytest.approx(torch.softmax(logits, 1)[0, 1].item())
    assert float(probs[1, 1]) == 0.0
    assert float(probs[1, 0]) == pytest.approx(1.0)


# ------------------------------------------------------------- network and start

def test_network_has_the_specified_shapes():
    net = PPONet()
    assert isinstance(net.trunk, SetNetMasked)
    assert net.trunk.cfg == NetConfig(d=64, layers=2, heads=4, ff=128)
    assert net.gru.input_size == POOL_DIM == 128
    assert net.gru.hidden_size == GRU_HIDDEN == 64
    assert tuple(net.policy.weight.shape) == (2, 64)
    assert isinstance(net.value[0], torch.nn.Linear) and tuple(net.value[0].weight.shape) == (64, 64)
    assert isinstance(net.value[1], torch.nn.ReLU)
    assert tuple(net.value[2].weight.shape) == (1, 64)


def test_initial_handoff_probability_is_five_percent_at_every_state():
    spec = make_spec(3.0, [0.4] * 5, np.linspace(1.0, 0.5, 5), 5)
    paths = make_paths(40, 5, seed=2)
    stats = RouteStats.fit(paths)
    cache = RouteCache(spec, paths, stats)
    pred = PolicyPredictor(PPONet(init_handoff_p=0.05), spec.E0)
    for k in range(1, 5):
        pi = pred.handoff_prob(cache, k, np.arange(40))
        np.testing.assert_allclose(pi, 0.05, atol=1e-6)


def test_deployed_predictor_is_the_surrogate_of_the_handoff_probability():
    spec = make_spec(3.0, [0.4] * 4, np.linspace(1.0, 0.5, 4), 4)
    paths = make_paths(30, 4, seed=5)
    cache = RouteCache(spec, paths, RouteStats.fit(paths))
    net = PPONet()
    randomize_policy(net, seed=3)
    pred = PolicyPredictor(net, spec.E0)
    idx = np.arange(30)
    pi = pred.handoff_prob(cache, 2, idx)
    np.testing.assert_allclose(pred.predict(cache, 2, idx),
                               spec.H[1] + spec.E0 * (pi - 0.5))


def test_predictor_has_no_future_leakage():
    spec = make_spec(3.0, [0.4] * 4, np.linspace(1.0, 0.5, 4), 4)
    paths = make_paths(30, 4, seed=7)
    stats = RouteStats.fit(paths)
    other = Paths(p=paths.p.copy(), d=paths.d.copy())
    other.p[:, 2:] = np.random.default_rng(99).uniform(0.5, 2.0, (30, 2))
    other.d[:, 2:] = np.random.default_rng(98).uniform(0.5, 2.0, (30, 2))
    net = PPONet()
    randomize_policy(net, seed=4)
    a = PolicyPredictor(net, spec.E0).handoff_prob(RouteCache(spec, paths, stats), 2,
                                                   np.arange(30))
    b = PolicyPredictor(net, spec.E0).handoff_prob(RouteCache(spec, other, stats), 2,
                                                   np.arange(30))
    np.testing.assert_array_equal(a, b)


def test_gru_state_is_carried_across_decision_stages():
    spec = make_spec(3.0, [0.4] * 4, np.linspace(1.0, 0.5, 4), 4)
    paths = make_paths(20, 4, seed=8)
    cache = RouteCache(spec, paths, RouteStats.fit(paths))
    net = PPONet()
    randomize_policy(net, seed=6)
    idx = np.arange(20)
    pi = PolicyPredictor(net, spec.E0).handoff_prob(cache, 3, idx)
    h = torch.zeros(idx.size, GRU_HIDDEN)
    with torch.no_grad():
        for j in range(1, 4):
            g, t = stage_batch(cache, j, idx, "full")
            pooled = net.encode(torch.from_numpy(g), torch.from_numpy(t),
                                torch.zeros(idx.size, 4 - j, dtype=torch.bool))
            h = net.step(pooled, h)
        p_ref = torch.softmax(net.policy(h), dim=1)[:, 1].numpy().astype(float)
    np.testing.assert_allclose(pi, p_ref, atol=1e-6)


def test_cached_and_recomputed_predictions_agree_along_a_shrinking_path_set():
    spec = make_spec(2.0, [0.4] * 6, np.linspace(1.0, 0.5, 6), 6)
    paths = make_paths(60, 6, seed=9)
    cache = RouteCache(spec, paths, RouteStats.fit(paths))
    net = PPONet()
    randomize_policy(net, seed=10)
    warm = PolicyPredictor(net, spec.E0)
    rows = np.arange(60)
    for k in range(1, 6):
        rows = rows[::2] if k % 2 == 0 else rows                  # shrink like simulate_policy
        cold = PolicyPredictor(net, spec.E0)                       # no cache: recompute
        np.testing.assert_allclose(warm.predict(cache, k, rows), cold.predict(cache, k, rows),
                                   rtol=0, atol=1e-6 * spec.E0)


def test_out_of_order_call_recomputes_from_stage_one():
    spec = make_spec(2.0, [0.4] * 4, np.linspace(1.0, 0.5, 4), 4)
    paths = make_paths(25, 4, seed=11)
    cache = RouteCache(spec, paths, RouteStats.fit(paths))
    net = PPONet()
    randomize_policy(net, seed=12)
    pred = PolicyPredictor(net, spec.E0)
    idx = np.arange(25)
    pred.predict(cache, 2, idx)
    again = pred.predict(cache, 3, idx)                 # cache at 2 exists, call 3 is valid
    fresh = PolicyPredictor(net, spec.E0).predict(cache, 3, idx)
    np.testing.assert_allclose(again, fresh, rtol=0, atol=1e-6 * spec.E0)
    jumped = pred.predict(cache, 1, idx)                # going back must not reuse the cache
    np.testing.assert_allclose(jumped, PolicyPredictor(net, spec.E0).predict(cache, 1, idx),
                               rtol=0, atol=1e-6 * spec.E0)


# ------------------------------------------------------------- rollouts

def _table(spec, paths, N):
    cache = RouteCache(spec, paths, RouteStats.fit(paths))
    return cache, PaddedTables(cache, np.arange(N))


def test_rollout_reaches_late_stages_under_five_percent_exploration():
    m = 16
    spec = make_spec(1e9, [0.4] * m, np.linspace(1.0, 0.5, m), m)     # no breach possible
    paths = make_paths(50, m, seed=13)
    cache, table = _table(spec, paths, 50)
    net = PPONet(init_handoff_p=0.05)
    rng = np.random.default_rng(0)
    reached = 0
    n_roll = 1500
    with torch.no_grad():
        for i in range(n_roll):
            ep = rollout_episode(net, table, i % 50, cache.W[i % 50], spec, rng)
            reached += int(ep.ks.size and ep.ks[-1] >= 15)
    assert reached / n_roll == pytest.approx(0.95 ** 14, abs=0.05)


def test_rollout_never_takes_a_decision_at_stage_m():
    m = 4
    spec = make_spec(1e9, [0.4] * m, np.linspace(1.0, 0.5, m), m)
    paths = make_paths(10, m, seed=14)
    cache, table = _table(spec, paths, 10)
    net = PPONet()
    randomize_policy(net, seed=15)
    rng = np.random.default_rng(1)
    with torch.no_grad():
        for i in range(200):
            ep = rollout_episode(net, table, i % 10, cache.W[i % 10], spec, rng)
            assert ep.ks.size == 0 or ep.ks.max() <= m - 1
            assert np.all(ep.actions[ep.ks == m - 1] >= 0) if ep.ks.size else True


def test_rollout_records_rewards_consistent_with_the_step_function():
    spec, paths, g1 = breach_toy(40, seed=4)
    cache, table = _table(spec, paths, 40)
    net = PPONet()
    rng = np.random.default_rng(2)
    with torch.no_grad():
        for r in range(40):
            ep = rollout_episode(net, table, r, cache.W[r], spec, rng)
            if ep.ks.size == 0:
                assert cache.W[r, 0] > spec.B
                continue
            last_action = int(ep.actions[-1])
            reward, done = step_reward(spec, cache.W[r], int(ep.ks[-1]), last_action)
            assert done and ep.rewards[-1] == pytest.approx(reward)


# ------------------------------------------------------------- training

def test_one_episode_per_update_and_four_epochs_per_episode():
    spec, paths, g1 = breach_toy(40, seed=5)
    cfg = tiny_cfg(passes=2, epochs=4)
    _, _, _, budget = fit_ppo(spec, paths, N=40, rep=0, cfg=cfg)
    decisions = int(np.sum(g1[:32] <= spec.B))       # fitting rows with a decision at k = 1
    assert budget["updates"] == 2 * decisions
    assert budget["gradient_steps"] == 4 * budget["updates"]


def test_episodes_without_decisions_make_no_update():
    spec = make_spec(0.1, [0.4, 0.4], [0.3, 0.3], 2)
    paths = Paths(p=np.full((30, 2), 1.0), d=np.zeros((30, 2)))  # stop 1 always breaches
    rec = train_ppo_record(spec, paths, N=30, rep=0, cfg=tiny_cfg(passes=1))
    assert rec["updates"] == 0 and rec["gradient_steps"] == 0
    assert len({c["sha"] for c in rec["checkpoints"]}) == 1      # parameters never moved


def test_pass_end_checkpoints_equal_the_shorter_runs():
    spec, paths, _ = breach_toy(60, seed=6)
    one = train_ppo_record(spec, paths, N=60, rep=0, cfg=tiny_cfg(passes=1))
    three = train_ppo_record(spec, paths, N=60, rep=0, cfg=tiny_cfg(passes=3))

    def pass_end_sha(rec, p):
        return [c["sha"] for c in rec["checkpoints"] if c["pass_end"] and c["pass"] == p]

    assert pass_end_sha(one, 1) == pass_end_sha(three, 1)
    assert len(pass_end_sha(three, 3)) == 1


def test_fit_is_deterministic_for_a_fixed_seed_key():
    spec, paths, _ = breach_toy(50, seed=7)
    a = train_ppo_record(spec, paths, N=50, rep=0, cfg=tiny_cfg(passes=2))
    b = train_ppo_record(spec, paths, N=50, rep=0, cfg=tiny_cfg(passes=2))
    assert [c["sha"] for c in a["checkpoints"]] == [c["sha"] for c in b["checkpoints"]]
    assert a["chosen_index"] == b["chosen_index"]


def test_checkpoint_choice_is_the_lowest_validation_cost_with_ties_to_the_earlier():
    spec, paths, _ = breach_toy(80, seed=8)
    rec = train_ppo_record(spec, paths, N=80, rep=0, cfg=tiny_cfg(passes=2))
    costs = [c["val_cost"] for c in rec["checkpoints"]]
    assert rec["chosen_index"] == int(np.argmin(costs))
    assert len(rec["pass_best"]) == 2
    assert rec["pass_best"][-1]["index"] == rec["chosen_index"]


def test_policy_learns_the_optimal_threshold_on_the_one_stage_toy():
    spec, paths, g1 = breach_toy(200, seed=3)
    cfg = tiny_cfg(lr=3e-4, passes=3, gamma=1.0, ckpt_per_pass=4)   # 3e-3 saturates
    models, stats, _, _ = fit_ppo(spec, paths, N=200, rep=0, cfg=cfg)
    fit_paths = Paths(p=paths.p[:160], d=paths.d[:160])
    cache = RouteCache(spec, fit_paths, stats)
    pi = models[1][0].handoff_prob(cache, 1, np.arange(160))
    g = g1[:160]
    decision = g <= spec.B
    clear = decision & (np.abs(g - 0.7) > 0.05)
    agree = (pi > 0.5) == (g > 0.7)
    assert agree[clear].mean() >= 0.95


def test_trained_policy_cost_is_close_to_the_optimum_on_the_toy():
    spec, paths, g1 = breach_toy(200, seed=3)
    cfg = tiny_cfg(lr=3e-4, passes=3, gamma=1.0, ckpt_per_pass=4)
    models, stats, _, _ = fit_ppo(spec, paths, N=200, rep=0, cfg=cfg)
    fit_paths = Paths(p=paths.p[:160], d=paths.d[:160])
    cache = RouteCache(spec, fit_paths, stats)
    costs, _ = simulate_policy(cache, spec.B, spec.H, spec.E, models)
    assert float(costs.mean()) <= float(expected_toy_cost(g1[:160]).mean()) + 0.02


def test_budget_reports_visits_and_environment_steps():
    spec, paths, _ = breach_toy(60, seed=9)
    models, stats, diag, budget = fit_ppo(spec, paths, N=60, rep=0, cfg=tiny_cfg(passes=2))
    for key in ("unique_paths", "path_stop_reads", "val_reads", "updates", "gradient_steps",
                "n_params", "feature_seconds", "train_seconds", "val_seconds", "device",
                "visits_per_stage"):
        assert key in budget
    assert budget["device"] == "cpu"
    assert sum(budget["visits_per_stage"]) == budget["path_stop_reads"]
    assert budget["visits_per_stage"][0] > 0
    assert set(models) == {1}
    assert any(d.get("kind") == "summary" for d in diag)


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        PPOConfig(init_handoff_p=1.0)
    with pytest.raises(ValueError):
        PPOConfig(passes=0)
    with pytest.raises(ValueError):
        PPOConfig(epochs=0)
    with pytest.raises(ValueError):
        PPOConfig(clip=0.0)


def test_fit_refuses_a_path_count_different_from_N():
    spec, paths, _ = breach_toy(30, seed=10)
    with pytest.raises(ValueError):
        fit_ppo(spec, paths, N=40, rep=0, cfg=tiny_cfg())


def test_masked_network_is_the_spec_default_when_not_given_a_config():
    assert PPOConfig() == PPOConfig(lr=1e-4, gamma=0.5, passes=3, epochs=4, clip=0.2,
                                    max_grad=1.0, vf_coef=0.5, ent_coef=0.0,
                                    init_handoff_p=0.05, ckpt_per_pass=8)

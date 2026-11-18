"""DQN after Dastpak et al. (EXPERIMENT_B.md 3.5, 10 M18): continue-forced behaviour, exact
stage counts, the double-Q target, the update and target schedules, learning on tiny
routes with known Q*, determinism, pass-end checkpoints and the GPU path."""
import numpy as np
import pytest
import torch

from transvrp.expa.features import RouteCache, RouteStats, stage_batch
from transvrp.expa.paths import Paths, RouteSpec
from transvrp.expb import dqn
from transvrp.expb.dqn import (DQNConfig, QPredictor, ReplayBuffer, always_continue,
                               collect_transitions, double_q_target, fit_dqn,
                               resolve_device, stage_counts, train_dqn_record)
from transvrp.expb.padded import PaddedTables, masked_net


def make_spec(B, H, E, m):
    return RouteSpec(inst="TOY", route_id=0, route=tuple(range(1, m + 1)), m=m,
                     B=float(B), H=np.asarray(H, float), E=np.asarray(E, float),
                     E0=float(E[0]))


def make_paths(N, m, seed=1, lo=0.5, hi=2.0):
    rng = np.random.default_rng(seed)
    return Paths(p=rng.uniform(lo, hi, (N, m)), d=rng.uniform(lo, hi, (N, m)))


def make_setup(N=120, m=4, seed=1, B=4.0):
    spec = make_spec(B, [0.6] * m, np.linspace(1.0, 0.5, m), m)
    paths = make_paths(N, m, seed)
    stats = RouteStats.fit(paths)
    return spec, paths, RouteCache(spec, paths, stats), stats


def tiny_cfg(**kw) -> DQNConfig:
    base = dict(lr=1e-3, passes=1, n_env=16, buffer=5000, batch=64, per_update=32,
                learn_start=1, target_every_tr=256, ckpt_per_pass=2, device="cpu")
    base.update(kw)
    return DQNConfig(**base)


def reference_transitions(W, B, E, m, rows):
    """Per-row simulation of the continue-forced behaviour, written from simulate_policy:
    returns sorted tuples (row, k, c, done) for the stored transitions."""
    out = []
    for r in rows:
        for k in range(1, m):
            if W[r, k - 1] > B:               # emergency at stop k: no decision
                break
            breach_next = W[r, k] > B         # continue: stop k+1
            if breach_next:
                out.append((int(r), k, float(E[k]), True))
                break
            if k + 1 == m:
                out.append((int(r), k, 0.0, True))
                break
            out.append((int(r), k, 0.0, False))
    return sorted(out)


# ------------------------------------------------------------- Q network and predictor

def test_predictor_returns_cost_units_in_zero_E0_and_has_the_right_shape():
    spec, paths, cache, stats = make_setup(N=30, m=4)
    net = masked_net()
    pred = QPredictor(net, spec.E0)
    out = pred.predict(cache, 2, np.arange(30))
    assert out.shape == (30,)
    assert np.all(out > 0) and np.all(out < spec.E0)


def test_predictor_matches_the_network_on_stage_batch_features():
    spec, paths, cache, stats = make_setup(N=25, m=4)
    net = masked_net()
    net.eval()
    pred = QPredictor(net, spec.E0)
    idx = np.arange(25)
    glob, tok = stage_batch(cache, 1, idx, "full")
    with torch.no_grad():
        ref = net(torch.from_numpy(glob), torch.from_numpy(tok),
                  torch.zeros(25, spec.m - 1, dtype=torch.bool)).numpy().astype(float)
    assert np.allclose(pred.predict(cache, 1, idx), spec.E0 * ref, atol=1e-6)


def test_q_prediction_does_not_use_outcomes_after_the_stage():
    spec, paths, cache, stats = make_setup(N=40, m=5, seed=3)
    k = 2
    p2 = Paths(p=paths.p.copy(), d=paths.d.copy())
    p2.p[:, k:] = make_paths(40, 5, seed=77).p[:, k:]
    p2.d[:, k:] = make_paths(40, 5, seed=78).d[:, k:]
    net = masked_net()
    pred = QPredictor(net, spec.E0)
    idx = np.arange(40)
    a = pred.predict(cache, k, idx)
    b = pred.predict(RouteCache(spec, p2, stats), k, idx)
    assert np.array_equal(a, b)


# ------------------------------------------------------------- transitions and counts

def test_stage_counts_are_the_alive_pairs_of_the_continue_forced_path():
    spec, paths, cache, stats = make_setup(N=200, m=5, seed=4, B=3.0)
    rows = np.arange(200)
    counts = stage_counts(cache.W[rows], spec.B, spec.m)
    alive_direct = [(~(np.any(cache.W[:, :k] > spec.B, axis=1))).sum() for k in range(1, 5)]
    assert counts.tolist() == [0] + [int(a) for a in alive_direct]


def test_transitions_match_the_simulated_continue_path_row_by_row():
    spec, paths, cache, stats = make_setup(N=80, m=5, seed=6, B=2.5)
    rows = np.arange(80)
    pos, ks, cs, dn, _ = collect_transitions(cache.W[rows], np.arange(80), spec.B, spec.E,
                                             spec.m, 16, always_continue)
    got = sorted((int(a), int(b), float(c), bool(d)) for a, b, c, d in zip(pos, ks, cs, dn))
    assert got == reference_transitions(cache.W, spec.B, spec.E, spec.m, rows)


def test_handoff_behaviour_stores_no_transition_and_stops_the_path():
    spec, paths, cache, stats = make_setup(N=60, m=5, seed=21, B=4.0)
    rows = np.arange(60)
    hand_at_one = lambda pos, k: np.full(np.asarray(pos).shape[0], k > 1)
    pos, ks, cs, dn, _ = collect_transitions(cache.W[rows], rows, spec.B, spec.E, spec.m,
                                             16, hand_at_one)
    assert not np.any(ks == 1)
    assert pos.size == 0 or np.all(ks >= 2)


def test_always_continue_visits_every_stage_with_alive_rows():
    spec, paths, cache, stats = make_setup(N=300, m=5, seed=8, B=3.5)
    rec = train_dqn_record(spec, paths, 300, 0, tiny_cfg(passes=1), keep_states=False)
    counts = rec["transitions_per_stage"]
    alive = stage_counts(cache.W[np.arange(240)], spec.B, spec.m)
    assert all(counts[k] == alive[k] for k in range(1, spec.m))
    assert min(counts[1:]) > 0


def test_behaviour_policy_is_always_continue():
    cont = always_continue(np.arange(7), 2)
    assert cont.dtype == bool and cont.all() and cont.shape == (7,)


def test_only_fitting_rows_are_environment_rows():
    spec, paths, cache, stats = make_setup(N=100, m=4, seed=9, B=2.0)
    rec = train_dqn_record(spec, paths, 100, 0, tiny_cfg(passes=1), keep_states=True)
    fit_set = set(range(80))
    assert set(rec["env_rows"].tolist()) <= fit_set
    assert len(set(rec["env_rows"].tolist())) > 0


def test_env_steps_are_the_stop_visits_summed_over_passes():
    spec, paths, cache, stats = make_setup(N=100, m=4, seed=10, B=2.5)
    rec1 = train_dqn_record(spec, paths, 100, 0, tiny_cfg(passes=1), keep_states=False)
    rec2 = train_dqn_record(spec, paths, 100, 0, tiny_cfg(passes=2), keep_states=False)
    assert rec2["env_steps"] == 2 * rec1["env_steps"]
    assert rec2["transitions_per_stage"] == [2 * c for c in rec1["transitions_per_stage"]]


# ------------------------------------------------------------- targets and schedules

def test_double_q_target_on_a_two_state_toy():
    c = torch.tensor([0.0, 0.0, 0.2, 0.0])
    done = torch.tensor([0.0, 0.0, 1.0, 0.0])
    q_on = torch.tensor([0.7, 0.3, 0.9, 0.3])     # online: 0.7 > H, 0.3 < H
    q_tg = torch.tensor([0.1, 0.45, 0.9, 0.4])    # target values
    h = torch.tensor([0.5, 0.5, 0.5, 0.5])
    y = double_q_target(c, done, q_on, q_tg, h)
    # row 0: online prefers handoff -> H = 0.5 (the target value is not used)
    # row 1: online prefers continue -> target value 0.45
    # row 2: terminal -> only the cost 0.2
    # row 3: online prefers continue -> target value 0.4
    assert torch.allclose(y, torch.tensor([0.5, 0.45, 0.2, 0.4]))


def test_handoff_value_is_the_known_terminal_cost_not_a_learned_value():
    c = torch.zeros(1)
    done = torch.zeros(1)
    y = double_q_target(c, done, torch.tensor([9.0]), torch.tensor([9.0]), torch.tensor([0.3]))
    assert torch.allclose(y, torch.tensor([0.3]))


def test_update_count_is_one_per_per_update_transitions():
    spec, paths, cache, stats = make_setup(N=150, m=4, seed=11, B=2.5)
    rec = train_dqn_record(spec, paths, 150, 0, tiny_cfg(passes=2, per_update=32),
                           keep_states=False)
    total = sum(rec["transitions_per_stage"])
    assert rec["updates"] == total // 32


def test_no_update_before_learn_start_and_the_buffer_is_capped():
    spec, paths, cache, stats = make_setup(N=150, m=4, seed=12, B=2.5)
    rec = train_dqn_record(spec, paths, 150, 0, tiny_cfg(passes=1, learn_start=10 ** 9),
                           keep_states=False)
    assert rec["updates"] == 0
    buf = ReplayBuffer(5)
    buf.add(np.arange(4), np.ones(4, int), np.zeros(4), np.zeros(4, bool))
    buf.add(np.arange(4), np.ones(4, int), np.zeros(4), np.zeros(4, bool))
    assert buf.size == 5


def test_target_copies_happen_every_target_period_in_transitions():
    spec, paths, cache, stats = make_setup(N=150, m=4, seed=13, B=2.5)
    rec = train_dqn_record(spec, paths, 150, 0, tiny_cfg(passes=2, target_every_tr=200),
                           keep_states=False)
    total = sum(rec["transitions_per_stage"])
    assert rec["target_copies"] == list(range(200, total + 1, 200))


# ------------------------------------------------------------- learning and determinism

def one_stage_setup(N=200):
    """m = 2: one decision (k = 1). Stop 2 always breaches (B = 0.5, g_2 = 1), so the
    continuation cost is E[1] = 0.6 on every state: Q*(s, continue) = 0.6 exactly."""
    spec = make_spec(0.5, [0.3, 0.7], [1.0, 0.6], 2)
    rng = np.random.default_rng(5)
    p = np.zeros((N, 2))
    d = np.zeros((N, 2))
    p[:, 0] = rng.uniform(0.9, 1.1, N)
    d[:, 0] = p[:, 0]
    p[:, 1] = 1.0
    paths = Paths(p=p, d=d)
    return spec, paths, RouteStats.fit(paths)


def test_one_stage_route_converges_to_the_breach_cost_within_two_percent():
    spec, paths, stats = one_stage_setup()
    cfg = tiny_cfg(lr=3e-3, passes=40, learn_start=32, target_every_tr=64, batch=128,
                   ckpt_per_pass=1)
    rec = train_dqn_record(spec, paths, 200, 0, cfg, keep_states=True)
    net = masked_net()
    net.load_state_dict(rec["last_state"])
    cache = RouteCache(spec, paths, stats)
    q = QPredictor(net, spec.E0).predict(cache, 1, np.arange(160, 200))
    assert np.all(np.abs(q - 0.6) / 0.6 < 0.02)


def test_fit_dqn_is_deterministic_on_cpu():
    spec, paths, cache, stats = make_setup(N=100, m=4, seed=14, B=2.5)
    cfg = tiny_cfg(passes=2)
    m1, _, d1, b1 = fit_dqn(spec, paths, 100, 0, cfg)
    m2, _, d2, b2 = fit_dqn(spec, paths, 100, 0, cfg)
    shas1 = [x["sha"] for x in d1 if x["kind"] == "ckpt"]
    shas2 = [x["sha"] for x in d2 if x["kind"] == "ckpt"]
    assert shas1 == shas2
    q1 = m1[1][0].predict(cache, 1, np.arange(20))
    q2 = m2[1][0].predict(cache, 1, np.arange(20))
    assert np.array_equal(q1, q2)


def test_pass_end_checkpoints_equal_the_shorter_runs():
    spec, paths, cache, stats = make_setup(N=100, m=4, seed=15, B=2.5)
    rec2 = train_dqn_record(spec, paths, 100, 0, tiny_cfg(passes=2), keep_states=False)
    rec1 = train_dqn_record(spec, paths, 100, 0, tiny_cfg(passes=1), keep_states=False)
    end1_of_two = [c for c in rec2["checkpoints"] if c["pass_end"] and c["pass"] == 1][0]
    end_of_one = [c for c in rec1["checkpoints"] if c["pass_end"]][-1]
    assert end1_of_two["sha"] == end_of_one["sha"]
    assert end1_of_two["update"] == end_of_one["update"]


def test_pass_end_checkpoints_are_saved_for_every_pass():
    spec, paths, cache, stats = make_setup(N=100, m=4, seed=16, B=2.5)
    rec = train_dqn_record(spec, paths, 100, 0, tiny_cfg(passes=3), keep_states=False)
    ends = [c["pass"] for c in rec["checkpoints"] if c["pass_end"]]
    assert ends == [1, 2, 3]
    assert rec["checkpoints"][0]["update"] == 0


def test_kept_checkpoint_has_the_lowest_validation_cost_and_earliest_on_ties():
    spec, paths, cache, stats = make_setup(N=100, m=4, seed=17, B=2.5)
    rec = train_dqn_record(spec, paths, 100, 0, tiny_cfg(passes=2), keep_states=False)
    costs = [c["val_cost"] for c in rec["checkpoints"]]
    best = min(range(len(costs)), key=lambda i: (costs[i], i))
    assert rec["chosen_index"] == best


def test_budget_reports_device_and_counts():
    spec, paths, cache, stats = make_setup(N=100, m=4, seed=18, B=2.5)
    models, st, diag, budget = fit_dqn(spec, paths, 100, 0, tiny_cfg(passes=1))
    assert budget["device"] == "cpu"
    assert budget["unique_paths"] == 100
    assert len(budget["transitions_per_stage"]) == spec.m - 1
    assert budget["n_params"] > 0
    assert set(models) == {1, 2, 3}


def test_resolve_device_follows_availability():
    assert resolve_device("cpu").type == "cpu"
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_training_smoke_keeps_a_cpu_model():
    spec, paths, cache, stats = make_setup(N=100, m=4, seed=19, B=2.5)
    models, _, diag, budget = fit_dqn(spec, paths, 100, 0, tiny_cfg(passes=1, device="cuda"))
    assert budget["device"] == "cuda"
    q = models[1][0].predict(cache, 1, np.arange(10))
    assert np.all(np.isfinite(q))
    assert next(models[1][0].net.parameters()).device.type == "cpu"

"""DOS: deep optimal stopping (EXPERIMENT_B.md 3.4, 10 M16). Shapes, batch norm, the
decision surrogate, cold/warm starts, and no leakage of validation rows into training."""
import copy
import math

import numpy as np
import pytest
import torch

from transvrp.expa.features import MLP_DIM, RouteCache, RouteStats, mlp_batch
from transvrp.expa.paths import Paths, RouteSpec
from transvrp.expa.train import ConstantPredictor
from transvrp.expa.backward import split_80_20
from transvrp.expb import fidelity as fd
from transvrp.expb.dos import (DOSConfig, DOSNet, DOSPredictor, dos_fit_fn, dos_loss,
                               fit_dos, init_net, stage_steps)


def make_spec(B, H, E, m):
    return RouteSpec(inst="TOY", route_id=0, route=tuple(range(1, m + 1)), m=m,
                     B=float(B), H=np.asarray(H, float), E=np.asarray(E, float),
                     E0=float(E[0]))


def make_cache(N=200, m=4, seed=1, H=0.6):
    spec = make_spec(4.0, [H] * m, [1.0] * m, m)
    rng = np.random.default_rng(seed)
    paths = Paths(p=rng.uniform(0.5, 2.0, (N, m)), d=rng.uniform(0.5, 2.0, (N, m)))
    return spec, paths, RouteCache(spec, paths, RouteStats.fit(paths))


def constant_pi_net(pi: float) -> DOSNet:
    net = DOSNet()
    with torch.no_grad():
        net.out.weight.zero_()
        net.out.bias.fill_(math.log(pi / (1.0 - pi)))
    net.eval()
    return net


def fast_cfg(**kw) -> DOSConfig:
    base = dict(batch=256, s_min=20, s_max=40, n_ckpt=4)
    base.update(kw)
    return DOSConfig(**base)


# ---------------------------------------------------------------- network

def test_network_shapes_range_and_architecture():
    net = DOSNet()
    x = torch.randn(7, MLP_DIM)
    out = net(x)
    assert out.shape == (7,)
    assert torch.all(out > 0) and torch.all(out < 1)
    linears = [mod for mod in net.modules() if isinstance(mod, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(15, 55), (55, 55), (55, 1)]
    assert sum(isinstance(mod, torch.nn.BatchNorm1d) for mod in net.modules()) == 2


def test_batch_norm_statistics_move_only_in_training_mode():
    net = DOSNet()
    bn = [mod for mod in net.modules() if isinstance(mod, torch.nn.BatchNorm1d)][0]
    before = bn.running_mean.clone()
    net.eval()
    with torch.no_grad():
        net(torch.randn(64, MLP_DIM) + 3.0)
    assert torch.equal(bn.running_mean, before)
    net.train()
    with torch.no_grad():
        net(torch.randn(64, MLP_DIM) + 3.0)
    assert not torch.equal(bn.running_mean, before)


def test_eval_output_does_not_depend_on_the_other_rows_in_the_batch():
    net = DOSNet()
    net.train()
    with torch.no_grad():
        net(torch.randn(128, MLP_DIM))                   # set running statistics
    net.eval()
    x = torch.randn(5, MLP_DIM)
    alone = net(x[:1])
    together = net(x)[:1]
    assert torch.allclose(alone, together, atol=1e-6)


def test_loss_is_the_stop_or_continue_cost_on_the_E0_scale():
    F = torch.tensor([0.2, 0.9], dtype=torch.float64)
    u = torch.tensor([0.5, 0.1], dtype=torch.float64)
    loss = dos_loss(F, u, 0.3)
    assert float(loss) == pytest.approx(np.mean([0.2 * 0.3 + 0.8 * 0.5,
                                                 0.9 * 0.3 + 0.1 * 0.1]))


def test_stage_steps_follow_the_formula_and_its_floor_and_cap():
    cfg = DOSConfig()
    assert stage_steps(10_000, cfg) == 50                # ceil(20*10000/8192) = 25 -> floor
    assert stage_steps(100_000, cfg) == 245              # ceil(244.14) = 245
    assert stage_steps(1_000_000, cfg) == 400            # capped


# ---------------------------------------------------------------- surrogate

def test_surrogate_is_H_plus_E0_times_pi_minus_half():
    spec, paths, cache = make_cache(N=60, m=3, H=0.4)
    hi = DOSPredictor(constant_pi_net(0.7), spec.E0)
    lo = DOSPredictor(constant_pi_net(0.3), spec.E0)
    idx = np.arange(5)
    assert np.allclose(hi.predict(cache, 1, idx), 0.4 + 0.2)
    assert np.allclose(lo.predict(cache, 1, idx), 0.4 - 0.2)
    assert np.allclose(hi.pi(cache, 1, idx), 0.7, atol=1e-6)


def test_fold_mean_of_surrogates_decides_at_one_half():
    spec, paths, cache = make_cache(N=60, m=3, H=0.4)
    idx = np.arange(4)
    for pa, pb, stops in [(0.7, 0.4, True), (0.6, 0.3, False)]:
        a = DOSPredictor(constant_pi_net(pa), spec.E0).predict(cache, 1, idx)
        b = DOSPredictor(constant_pi_net(pb), spec.E0).predict(cache, 1, idx)
        mean_surrogate = 0.5 * (a + b)
        assert np.all((mean_surrogate > 0.4) == stops)


def test_empty_index_gives_an_empty_prediction():
    spec, paths, cache = make_cache(N=40, m=3)
    out = DOSPredictor(constant_pi_net(0.5), spec.E0).predict(cache, 1, np.zeros(0, np.int64))
    assert out.shape == (0,)


# ---------------------------------------------------------------- starts and fallbacks

def test_warm_start_copies_the_previous_stage_network():
    net = DOSNet()
    warm = DOSPredictor(net, 1.0)
    fresh = init_net(warm, seed=5)
    for a, b in zip(fresh.state_dict().values(), net.state_dict().values()):
        assert torch.equal(a, b)


def test_cold_start_when_the_warm_predictor_is_constant():
    a = init_net(ConstantPredictor(0.3), seed=5)
    b = init_net(None, seed=5)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(),
                                                 b.state_dict().values()))
    warm_net = init_net(DOSPredictor(DOSNet(), 1.0), seed=6)
    assert not torch.equal(next(warm_net.parameters()), next(a.parameters()))


def test_fit_reports_warm_or_cold_start_and_warm_is_used():
    spec, paths, cache = make_cache(N=200, m=3, seed=3)
    fit_idx = np.arange(0, 120, 2)
    y = np.random.default_rng(4).uniform(0.0, 1.0, fit_idx.size)
    empty = np.zeros(0, np.int64)
    diag = []
    fn = dos_fit_fn("ws|0|200|0", fast_cfg(), diag)
    fn(1, cache, fit_idx, y, empty, np.zeros(0), ConstantPredictor(0.3))
    warm_pred = DOSPredictor(DOSNet(), spec.E0)
    fn(1, cache, fit_idx, y, empty, np.zeros(0), warm_pred)
    assert [d["action"] for d in diag] == ["cold", "warm"]


def test_too_few_rows_give_the_surrogate_of_the_constant_decision():
    spec, paths, cache = make_cache(N=100, m=3, seed=7, H=0.4)
    idx = np.arange(0, 10, 2)                             # 5 rows, mean target above H
    y = np.full(idx.size, 0.9)
    pred = dos_fit_fn("few|0|100|0", fast_cfg(), [])(
        1, cache, idx, y, np.zeros(0, np.int64), np.zeros(0), None)
    assert isinstance(pred, ConstantPredictor)
    assert pred.value == pytest.approx(0.4 + 0.5 * spec.E0)   # stop is cheaper: pi = 1
    y_low = np.full(idx.size, 0.1)
    pred_low = dos_fit_fn("few|0|100|0", fast_cfg(), [])(
        1, cache, idx, y_low, np.zeros(0, np.int64), np.zeros(0), None)
    assert pred_low.value == pytest.approx(0.4 - 0.5 * spec.E0)   # continue: pi = 0


def test_too_few_rows_carry_a_warm_network():
    spec, paths, cache = make_cache(N=100, m=3, seed=8)
    warm = DOSPredictor(DOSNet(), spec.E0)
    pred = dos_fit_fn("carry|0|100|0", fast_cfg(), [])(
        1, cache, np.arange(0, 10, 2), np.full(5, 0.5), np.zeros(0, np.int64),
        np.zeros(0), warm)
    assert pred is warm


def test_exactly_min_rows_is_trained():
    spec, paths, cache = make_cache(N=100, m=3, seed=9)
    diag = []
    pred = dos_fit_fn("min|0|100|0", fast_cfg(), diag)(
        1, cache, np.arange(0, 40, 2), np.random.default_rng(2).uniform(0, 1, 20),
        np.zeros(0, np.int64), np.zeros(0), None)
    assert isinstance(pred, DOSPredictor)
    assert diag[-1]["action"] == "cold"


def test_targets_outside_zero_E0_are_refused():
    spec, paths, cache = make_cache(N=100, m=3, seed=10)
    with pytest.raises(ValueError):
        dos_fit_fn("bad|0|100|0", fast_cfg(), [])(
            1, cache, np.arange(0, 60, 2), np.full(30, 2.0), np.zeros(0, np.int64),
            np.zeros(0), None)


# ---------------------------------------------------------------- no leakage

def test_training_trajectory_does_not_read_validation_rows():
    spec, paths, cache = make_cache(N=300, m=3, seed=11)
    fit_idx = np.arange(0, 160, 2)
    y = np.random.default_rng(12).uniform(0.0, 1.0, fit_idx.size)
    val_a = np.arange(200, 240)
    val_b = np.arange(260, 300)
    y_va = np.random.default_rng(13).uniform(0.0, 1.0, 40)
    y_vb = np.random.default_rng(14).uniform(0.0, 1.0, 40)
    da, db = [], []
    pa = dos_fit_fn("leak|0|300|0", fast_cfg(), da)(1, cache, fit_idx, y, val_a, y_va, None)
    pb = dos_fit_fn("leak|0|300|0", fast_cfg(), db)(1, cache, fit_idx, y, val_b, y_vb, None)
    assert da[-1]["train_loss_per_ckpt"] == db[-1]["train_loss_per_ckpt"]
    assert da[-1]["val_loss_per_ckpt"] != db[-1]["val_loss_per_ckpt"]
    assert isinstance(pa, DOSPredictor) and isinstance(pb, DOSPredictor)


def _earliest_argmin(hist):
    return min(hist, key=lambda item: (item[1], item[0]))[0]


def test_checkpoint_is_the_earliest_minimum_of_the_validation_loss_not_the_training_loss():
    spec, paths, cache = make_cache(N=300, m=3, seed=15)
    fit_idx = np.arange(0, 160, 2)
    y = np.random.default_rng(16).uniform(0.0, 1.0, fit_idx.size)
    val = np.arange(200, 240)
    disagreed = False
    for r in range(8):
        diag = []
        y_val = np.random.default_rng(17 + r).uniform(0, 1, 40)
        dos_fit_fn(f"sel|0|300|{r}", fast_cfg(), diag)(
            1, cache, fit_idx, y, val, y_val, None)
        entry = diag[-1]
        val_best = _earliest_argmin(entry["val_loss_per_ckpt"])
        train_best = _earliest_argmin(entry["train_loss_per_ckpt"])
        assert entry["chosen_ckpt"] == val_best
        disagreed = disagreed or val_best != train_best
    assert disagreed                                    # the test can tell the rules apart


def test_features_for_stage_k_do_not_read_later_stops():
    spec, paths, cache = make_cache(N=120, m=4, seed=18)
    later = Paths(p=paths.p.copy(), d=paths.d.copy())
    later.p[:, 2:] += 5.0
    cache2 = RouteCache(spec, later, cache.stats)
    idx = np.arange(120)
    assert np.array_equal(mlp_batch(cache, 1, idx), mlp_batch(cache2, 1, idx))


# ---------------------------------------------------------------- end to end

def test_fit_dos_gives_two_predictors_per_epoch_and_a_budget():
    spec, paths, _ = make_cache(N=200, m=4, seed=19)
    models, stats, diag, budget = fit_dos(spec, paths, 200, 0, fast_cfg())
    assert sorted(models) == [1, 2, 3]
    assert all(len(v) == 2 and all(isinstance(p, DOSPredictor) for p in v)
               for v in models.values())
    assert isinstance(stats, RouteStats)
    assert budget["unique_paths"] == 200 and budget["path_stop_reads"] == 800
    assert budget["n_params"] > 0
    assert any(d.get("kind") == "ls" for d in diag)


def test_fit_dos_is_reproducible_under_its_seed():
    spec, paths, _ = make_cache(N=160, m=3, seed=20)
    m1, _, _, _ = fit_dos(spec, paths, 160, 0, fast_cfg())
    m2, _, _, _ = fit_dos(spec, paths, 160, 0, fast_cfg())
    cache = RouteCache(spec, paths, RouteStats.fit(Paths(p=paths.p[:128], d=paths.d[:128])))
    idx = np.arange(160)
    for k in (1, 2):
        for a, b in zip(m1[k], m2[k]):
            assert np.array_equal(a.predict(cache, k, idx), b.predict(cache, k, idx))


def test_fit_dos_refuses_a_wrong_path_count():
    spec, paths, _ = make_cache(N=50, m=3, seed=21)
    with pytest.raises(ValueError, match="expected 60"):
        fit_dos(spec, paths, 60, 0, fast_cfg())


# ---------------------------------------------------------------- width (M21 fix 1)

def test_init_net_honours_the_configured_width():
    # REVIEW_B / M21: init_net used to build DOSNet() (width 55) whatever cfg.width was,
    # so n_params in the budget was wrong for every other width.
    net = init_net(None, seed=3, width=20)
    assert net.out.in_features == 20
    assert sum(p.numel() for p in net.parameters()) == sum(
        p.numel() for p in DOSNet(width=20).parameters())


def test_fit_fn_builds_networks_of_the_configured_width():
    spec, paths, cache = make_cache(N=200, m=4)
    cfg = DOSConfig(width=20, min_rows=5, s_min=2, s_max=3, n_ckpt=1)
    fit = dos_fit_fn("w|0|200|0|dos", cfg, diag=[])
    folds, val = split_80_20(200)
    idx = folds[0]                       # one fold per call (_fold_label)
    y = np.full(200, 0.5)
    pred = fit(1, cache, idx, y[idx], val, y[val], None)
    assert isinstance(pred, DOSPredictor)
    assert pred.net.out.in_features == 20


# ---------------------------------------------------------------- never-stop record (M22 B-DOS)

def _model_digest(models, spec, paths) -> str:
    """SHA-256 over every fitted state_dict and the surrogate values on all rows (fixed seed)."""
    import hashlib
    h = hashlib.sha256()
    cache = RouteCache(spec, paths, RouteStats.fit(paths))
    idx = np.arange(paths.p.shape[0])
    for k in sorted(models):
        for pred in models[k]:
            for name, tensor in pred.net.state_dict().items():
                h.update(name.encode())
                h.update(tensor.detach().cpu().numpy().tobytes())
            h.update(np.ascontiguousarray(pred.predict(cache, k, idx)).tobytes())
    return h.hexdigest()


# Digest of fit_dos(make_cache(N=160, m=3, seed=22), rep 0, fast_cfg()) under the code that
# did not yet record never_val (M16 fix). Any change to what DOS learns changes it.
REFERENCE_DIGEST = "4b4ae7164ebec2d3ef3da89df20ed3e512b79f6689d1e9e1a4053250d03ccca0"


def _E0_two_cache(N=200, m=3, seed=23):
    spec = make_spec(4.0, [0.6] * m, [2.0] * m, m)
    rng = np.random.default_rng(seed)
    paths = Paths(p=rng.uniform(0.5, 2.0, (N, m)), d=rng.uniform(0.5, 2.0, (N, m)))
    return spec, paths, RouteCache(spec, paths, RouteStats.fit(paths))


def test_never_val_is_the_loss_of_the_constant_continue_decision_on_val_rows():
    spec, paths, cache = _E0_two_cache()
    fit_idx = np.arange(0, 120, 2)
    val_idx = np.arange(120, 160)
    y_fit = np.random.default_rng(31).uniform(0.0, 2.0, fit_idx.size)
    y_val = np.random.default_rng(32).uniform(0.0, 2.0, val_idx.size)
    diag = []
    dos_fit_fn("never|0|200|0", fast_cfg(), diag)(1, cache, fit_idx, y_fit, val_idx, y_val, None)
    continue_everywhere = torch.zeros(val_idx.size)          # F = 0: continue on every row
    expected = float(dos_loss(continue_everywhere,
                              torch.from_numpy(y_val / spec.E0), spec.H[0] / spec.E0))
    assert diag[-1]["never_val"] == pytest.approx(expected, rel=1e-6, abs=1e-7)
    assert diag[-1]["never_val"] == pytest.approx(float(y_val.mean()) / spec.E0, rel=1e-6)


def test_never_val_is_recorded_for_every_stage_including_constant_and_carry_branches():
    spec, paths, cache = _E0_two_cache()
    y_val = np.full(40, 1.0)
    val_idx = np.arange(120, 160)
    diag = []
    fn = dos_fit_fn("every|0|200|0", fast_cfg(min_rows=20), diag)
    fn(1, cache, np.arange(0, 120, 2), np.full(60, 0.5), val_idx, y_val, None)   # trained
    fn(2, cache, np.arange(0, 10, 2), np.full(5, 0.5), val_idx, y_val, None)     # constant
    fn(2, cache, np.arange(0, 10, 2), np.full(5, 0.5), val_idx, y_val, DOSPredictor(DOSNet(), 2.0))
    assert [d["action"] for d in diag] == ["cold", "constant", "carry"]
    assert all(d["never_val"] == pytest.approx(0.5) for d in diag)


def test_never_val_is_none_without_validation_rows():
    spec, paths, cache = _E0_two_cache()
    diag = []
    dos_fit_fn("noval|0|200|0", fast_cfg(), diag)(
        1, cache, np.arange(0, 120, 2), np.full(60, 1.0), np.zeros(0, np.int64),
        np.zeros(0), None)
    assert "never_val" in diag[-1] and diag[-1]["never_val"] is None
    assert diag[-1]["n_val"] == 0


def test_fit_dos_stage_entries_all_carry_never_val_and_match_the_gate_key():
    spec, paths, _ = make_cache(N=160, m=3, seed=22)
    _, _, diag, _ = fit_dos(spec, paths, 160, 0, fast_cfg())
    stages = [d for d in diag if d.get("kind") == "stage"]
    assert stages and all("never_val" in d for d in stages)
    assert all(isinstance(d["never_val"], float) for d in stages if d["n_val"] > 0)


def test_fitted_models_are_bitwise_unchanged_on_a_fixed_seed():
    spec, paths, _ = make_cache(N=160, m=3, seed=22)
    models, _, _, _ = fit_dos(spec, paths, 160, 0, fast_cfg())
    assert _model_digest(models, spec, paths) == REFERENCE_DIGEST


# ---------------------------------------------------------------- held stages (B-DOS objective)

def test_constant_stage_objective_is_the_loss_of_the_deployed_constant_decision():
    spec, paths, cache = _E0_two_cache()
    val_idx = np.arange(120, 160)
    y_val = np.random.default_rng(41).uniform(0.0, 2.0, val_idx.size)
    stop_diag, cont_diag = [], []
    # Hk = 0.6 < mean(y_fit) = 1.5: the constant decision is stop (pi = 1)
    dos_fit_fn("cs|0|200|0", fast_cfg(min_rows=20), stop_diag)(
        1, cache, np.arange(0, 10, 2), np.full(5, 1.5), val_idx, y_val, None)
    # mean(y_fit) = 0.1 < Hk: continue (pi = 0)
    dos_fit_fn("cc|0|200|0", fast_cfg(min_rows=20), cont_diag)(
        1, cache, np.arange(0, 10, 2), np.full(5, 0.1), val_idx, y_val, None)
    assert fd._dos_objective(stop_diag[-1]) == pytest.approx(spec.H[0] / spec.E0, rel=1e-6)
    assert fd._dos_objective(cont_diag[-1]) == pytest.approx(
        float(y_val.mean()) / spec.E0, rel=1e-6)
    assert stop_diag[-1]["chosen_ckpt"] == 0


def test_carry_stage_objective_is_the_loss_of_the_carried_network_on_val_rows():
    spec, paths, cache = _E0_two_cache()
    val_idx = np.arange(120, 160)
    y_val = np.random.default_rng(42).uniform(0.0, 2.0, val_idx.size)
    # carried network that always continues (pi = 0): its objective is the never-stop value
    never_warm = DOSPredictor(constant_pi_net(1e-9), spec.E0)
    diag = []
    dos_fit_fn("kc|0|200|0", fast_cfg(min_rows=20), diag)(
        1, cache, np.arange(0, 10, 2), np.full(5, 0.5), val_idx, y_val, never_warm)
    assert diag[-1]["action"] == "carry"
    assert fd._dos_objective(diag[-1]) == pytest.approx(float(y_val.mean()) / spec.E0, abs=1e-6)
    # a generic carried network: the loss of its own surrogate F on the validation rows
    warm = DOSPredictor(DOSNet(), spec.E0)
    F = DOSPredictor(copy.deepcopy(warm.net), spec.E0).pi(cache, 1, val_idx)
    u = y_val / spec.E0
    expected = float(np.mean(F * (spec.H[0] / spec.E0) + (1.0 - F) * u))
    diag2 = []
    dos_fit_fn("kc2|0|200|0", fast_cfg(min_rows=20), diag2)(
        1, cache, np.arange(0, 10, 2), np.full(5, 0.5), val_idx, y_val, warm)
    assert fd._dos_objective(diag2[-1]) == pytest.approx(expected, rel=1e-5, abs=1e-7)


def test_carry_objective_leaves_the_carried_network_untouched():
    spec, paths, cache = _E0_two_cache()
    warm = DOSPredictor(DOSNet(), spec.E0)
    before = {k: v.clone() for k, v in warm.net.state_dict().items()}
    training_flag = warm.net.training
    y_val = np.random.default_rng(43).uniform(0.0, 2.0, 40)
    pred = dos_fit_fn("untouched|0|200|0", fast_cfg(min_rows=20), [])(
        1, cache, np.arange(0, 10, 2), np.full(5, 0.5), np.arange(120, 160), y_val, warm)
    assert pred is warm
    assert warm.net.training == training_flag
    after = warm.net.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_held_stage_without_validation_rows_has_no_objective():
    spec, paths, cache = _E0_two_cache()
    for warm in (None, DOSPredictor(DOSNet(), spec.E0)):
        diag = []
        dos_fit_fn("nov|0|200|0", fast_cfg(min_rows=20), diag)(
            1, cache, np.arange(0, 10, 2), np.full(5, 0.5), np.zeros(0, np.int64),
            np.zeros(0), warm)
        assert diag[-1]["n_val"] == 0
        assert fd._dos_objective(diag[-1]) is None


def test_smoke_fit_with_every_stage_held_records_an_objective_for_each_stage_with_val_rows():
    spec, paths, _ = make_cache(N=600, m=3, seed=44)
    cfg = fast_cfg(min_rows=10 ** 9)                      # every stage is a constant stage
    _, _, diag, _ = fit_dos(spec, paths, 600, 0, cfg)
    stages = [d for d in diag if d.get("kind") == "stage"]
    assert stages and all(d["action"] == "constant" for d in stages)
    big = [d for d in stages if int(d["n_val"]) >= 100]
    assert big                                            # the 100-row regime is exercised
    assert all(fd._dos_objective(d) is not None for d in big)

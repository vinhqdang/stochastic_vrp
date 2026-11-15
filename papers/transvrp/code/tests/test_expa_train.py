"""M5: neural stage training (EXPERIMENT_A.md section 9, M5).

Tests use CON3-0 route 1 (m = 12), N = 200 training paths and s_max = 10 so
that each fit takes seconds.
"""
import functools

import numpy as np
import pytest
import torch

import transvrp.expa.train as train_mod
from transvrp.expa.features import RouteCache
from transvrp.expa.paths import route_pool, route_specs
from transvrp.expa.train import (ConstantPredictor, NetPredictor, StageConfig,
                                 fit_neural, neural_fit_fn, stage_steps)

N = 200
ROUTE = 1


@functools.lru_cache(maxsize=None)
def _train_pool(rep):
    return route_pool("CON3-0", "train", rep=rep, N=N)[ROUTE]


@functools.lru_cache(maxsize=None)
def _dev_pool():
    return route_pool("CON3-0", "dev")[ROUTE]


@functools.lru_cache(maxsize=None)
def _spec():
    return {s.route_id: s for s in route_specs("CON3-0")}[ROUTE]


def _cfg():
    return StageConfig(s_max=10)


@functools.lru_cache(maxsize=None)
def _fit(arm, rep):
    return fit_neural(_spec(), _train_pool(rep), arm, rep, N, _cfg())


def _dev_cache(stats):
    return RouteCache(_spec(), _dev_pool(), stats)


def test_stage_steps_formula():
    cfg = StageConfig()
    assert stage_steps(10, False, cfg) == 50          # floor s_min
    assert stage_steps(10, True, cfg) == 200          # cold floor
    assert stage_steps(10**5, False, cfg) == 400      # cap
    assert stage_steps(10, False, StageConfig(s_max=10)) == 10
    assert stage_steps(2000, False, cfg) == 79        # ceil(20 * 2000 / 512)


def test_same_key_gives_identical_predictions():
    models_a, stats_a, _ = _fit("full", 0)
    models_b, stats_b, _ = fit_neural(_spec(), _train_pool(0), "full", 0, N, _cfg())
    cache_a = _dev_cache(stats_a)
    cache_b = _dev_cache(stats_b)
    idx = np.arange(100)
    for k in sorted(models_a):
        for pa, pb in zip(models_a[k], models_b[k]):
            x = pa.predict(cache_a, k, idx)
            y = pb.predict(cache_b, k, idx)
            assert np.array_equal(x, y)


def test_different_replicate_gives_different_predictions():
    models0, stats0, _ = _fit("full", 0)
    models1, stats1, _ = _fit("full", 1)
    idx = np.arange(100)
    k = max(models0)
    x = models0[k][0].predict(_dev_cache(stats0), k, idx)
    y = models1[k][0].predict(_dev_cache(stats1), k, idx)
    assert not np.array_equal(x, y)


def test_cold_stage_training_reduces_mse():
    diag = []
    _, _, diag = _fit("full", 0)
    cold = [d for d in diag if d.get("kind") == "stage" and d["k"] == _spec().m - 1]
    assert cold, "the cold stage must train on this route"
    for d in cold:
        if d["action"] == "cold":
            assert d["train_mse_last"] < d["train_mse_first"]


def test_arm_does_not_enter_seed(monkeypatch):
    """FULL and SCALAR at the same key start from identical weights at k=m-1."""
    captured = {"full": None, "scalar": None}
    orig = train_mod._train_stage
    current = {"arm": None}

    def spy(net, xs, y, xs_val, y_val, **kw):
        arm = current["arm"]
        if captured[arm] is None:      # first call = epoch m-1, fold A
            captured[arm] = {key: v.detach().clone()
                             for key, v in net.state_dict().items()}
        return orig(net, xs, y, xs_val, y_val, **kw)

    monkeypatch.setattr(train_mod, "_train_stage", spy)
    for arm in ("full", "scalar"):
        current["arm"] = arm
        fit_neural(_spec(), _train_pool(0), arm, 0, N, _cfg())
    assert captured["full"] is not None and captured["scalar"] is not None
    assert set(captured["full"]) == set(captured["scalar"])
    for key in captured["full"]:
        assert torch.equal(captured["full"][key], captured["scalar"][key])


def test_few_rows_branches_are_constant_or_carry():
    spec = _spec()
    pool = _train_pool(0)
    cache = RouteCache(spec, pool, _fit("full", 0)[1])
    cfg = _cfg()
    diag = []
    fit = neural_fit_fn("full", "t|1|200|0", cfg, diag)
    m = spec.m
    few = np.array([0, 2, 4, 6, 8], dtype=np.int64)     # 5 even rows < min_rows
    y = np.full(few.size, 3.0)
    pred = fit(m - 1, cache, few, y, np.zeros(0, dtype=np.int64), np.zeros(0), None)
    assert isinstance(pred, ConstantPredictor)
    assert pred.value == pytest.approx(3.0)
    empty = fit(m - 1, cache, np.zeros(0, dtype=np.int64), np.zeros(0),
                np.zeros(0, dtype=np.int64), np.zeros(0), None)
    assert isinstance(empty, ConstantPredictor)
    assert empty.value == pytest.approx(spec.E[m - 1])
    carried = fit(m - 2, cache, few, y, np.zeros(0, dtype=np.int64), np.zeros(0), pred)
    assert carried is pred
    assert [d["action"] for d in diag] == ["constant", "constant", "carry"]


def test_targets_above_E0_are_rejected():
    spec = _spec()
    cache = RouteCache(spec, _train_pool(0), _fit("full", 0)[1])
    fit = neural_fit_fn("full", "t|1|200|0", _cfg(), [])
    rows = np.arange(0, 60, 2, dtype=np.int64)
    bad = np.full(rows.size, spec.E0 * 2.0)
    with pytest.raises(ValueError):
        fit(spec.m - 1, cache, rows, bad, np.zeros(0, dtype=np.int64), np.zeros(0), None)


def test_mlp_arm_trains_and_predicts():
    models, stats, diag = fit_neural(_spec(), _train_pool(0), "mlp", 0, N, _cfg())
    k = _spec().m - 1
    cache = _dev_cache(stats)
    pred = models[k][0].predict(cache, k, np.arange(50))
    assert isinstance(models[k][0], NetPredictor)
    assert pred.shape == (50,)
    assert np.all(pred >= 0) and np.all(pred <= _spec().E0 + 1e-6)

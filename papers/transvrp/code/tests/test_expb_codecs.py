"""Payload codecs for the B arms (EXPERIMENT_B.md 10 M20): every arm's predictor survives
pack -> pickle -> unpack with np.array_equal predictions on 1,000 dev rows; shared DQN and
PPO networks are stored once per job; the library versions are recorded (gate item 8)."""
import pickle

import numpy as np
import pytest
import torch

from transvrp.expa.features import RouteCache, RouteStats
from transvrp.expa.paths import Paths, RouteSpec
from transvrp.expa.train import ConstantPredictor
from transvrp.expb import codecs
from transvrp.expb.alp import ALPZConfig
from transvrp.expb.dos import DOSConfig, DOSPredictor
from transvrp.expb.dqn import DQNConfig, QPredictor
from transvrp.expb.gatecheck import library_versions
from transvrp.expb.grid import GridPredictor
from transvrp.expb.ppo import PPOConfig, PolicyPredictor
from transvrp.expb.registry import fit_arm
from transvrp.expb.rlsm import RLSMConfig, RLSMPredictor
from transvrp.expb.thr import THRConfig

M = 4
N_FIT = 60
N_DEV = 1000


def make_spec():
    H = np.array([0.4, 0.4, 0.4, 0.4])
    E = np.linspace(1.0, 0.5, M)
    return RouteSpec(inst="TOY", route_id=0, route=tuple(range(1, M + 1)), m=M,
                     B=3.0, H=H, E=E, E0=float(E[0]))


def make_paths(N, seed):
    rng = np.random.default_rng(seed)
    return Paths(p=rng.uniform(0.5, 2.0, (N, M)), d=rng.uniform(0.5, 2.0, (N, M)))


TINY = {
    "thr": THRConfig(n_grid=201),
    "alpz": ALPZConfig(n_w=41),
    "rlsm": RLSMConfig(K=10),
    "dos": DOSConfig(s_min=5, s_max=10, width=8),
    "dqn": DQNConfig(device="cpu", passes=1, n_env=8, buffer=2000, batch=32, learn_start=10,
                     target_every_tr=64, ckpt_per_pass=2),
    "ppo": PPOConfig(passes=1, ckpt_per_pass=2),
}


@pytest.fixture(scope="module")
def fitted():
    """Every B arm fitted once on a tiny route; returns {arm: (models, stats, dev_cache)}."""
    spec = make_spec()
    paths = make_paths(N_FIT, seed=1)
    dev = make_paths(N_DEV, seed=2)
    out = {}
    for arm in ("thr", "alpz", "rlsm", "dos", "dqn", "ppo"):
        models, stats, _, _ = fit_arm(arm, spec, paths, N_FIT, 0, TINY[arm])
        out[arm] = (models, spec, RouteCache(spec, dev, stats))
    return out


def predictions(models, cache):
    idx = np.arange(cache.W.shape[0])
    return {k: [p.predict(cache, k, idx) for p in preds] for k, preds in models.items()}


@pytest.mark.parametrize("arm", ["thr", "alpz", "rlsm", "dos", "dqn", "ppo"])
def test_round_trip_predictions_are_identical_for_every_arm(fitted, arm):
    models, spec, cache = fitted[arm]
    blob = codecs.pack_models(models, arm, spec.E0)
    back = codecs.unpack_models(blob)
    assert set(back) == set(models)
    before, after = predictions(models, cache), predictions(back, cache)
    for k in before:
        assert len(after[k]) == len(before[k])
        for a, b in zip(before[k], after[k]):
            assert np.array_equal(a, b)


@pytest.mark.parametrize("arm", ["thr", "rlsm", "dos", "dqn", "ppo"])
def test_round_trip_survives_pickling(fitted, arm):
    models, spec, cache = fitted[arm]
    blob = pickle.loads(pickle.dumps(codecs.pack_models(models, arm, spec.E0)))
    back = codecs.unpack_models(blob)
    before, after = predictions(models, cache), predictions(back, cache)
    for k in before:
        assert np.array_equal(before[k][0], after[k][0])


def test_predictor_types_are_restored(fitted):
    for arm, cls in (("thr", GridPredictor), ("rlsm", RLSMPredictor), ("dos", DOSPredictor),
                     ("dqn", QPredictor), ("ppo", PolicyPredictor)):
        models, spec, _ = fitted[arm]
        back = codecs.unpack_models(codecs.pack_models(models, arm, spec.E0))
        pred = back[1][0]
        assert isinstance(pred, cls)


def test_constant_fallback_predictors_round_trip():
    models = {1: [ConstantPredictor(0.25)], 2: [ConstantPredictor(-0.1), ConstantPredictor(0.3)]}
    back = codecs.unpack_models(codecs.pack_models(models, "rlsm", 1.0))
    assert isinstance(back[1][0], ConstantPredictor) and back[1][0].value == 0.25
    assert [p.value for p in back[2]] == [-0.1, 0.3]


@pytest.mark.parametrize("arm", ["dqn", "ppo"])
def test_shared_network_is_stored_once_per_job(fitted, arm):
    models, spec, _ = fitted[arm]
    blob = codecs.pack_models(models, arm, spec.E0)
    assert len(blob["nets"]) == 1
    back = codecs.unpack_models(blob)
    assert all(back[k][0].net is back[1][0].net for k in back)


def test_blob_records_the_library_versions_of_the_running_process(fitted):
    models, spec, _ = fitted["thr"]
    blob = codecs.pack_models(models, "thr", spec.E0)
    assert blob["arm"] == "thr" and blob["E0"] == pytest.approx(spec.E0)
    assert blob["libraries"] == library_versions()
    assert set(blob["libraries"]) == {"python", "numpy", "scipy", "torch"}
    assert blob["libraries"]["torch"] == torch.__version__


def test_pack_refuses_an_arm_outside_the_b_family(fitted):
    models, spec, _ = fitted["thr"]
    with pytest.raises(ValueError):
        codecs.pack_models(models, "full", spec.E0)


def test_pack_refuses_an_unknown_predictor_type(fitted):
    with pytest.raises(TypeError):
        codecs.pack_models({1: [object()]}, "thr", 1.0)


def test_unpack_refuses_an_unknown_arm_or_kind(fitted):
    models, spec, _ = fitted["thr"]
    blob = codecs.pack_models(models, "thr", spec.E0)
    with pytest.raises(ValueError):
        codecs.unpack_models({**blob, "arm": "full"})
    bad = {**blob, "states": {1: [{"kind": "mystery"}]}}
    with pytest.raises(ValueError):
        codecs.unpack_models(bad)


def test_unpacked_network_carries_the_trained_weights_not_a_fresh_init(fitted):
    models, spec, cache = fitted["dqn"]
    back = codecs.unpack_models(codecs.pack_models(models, "dqn", spec.E0))
    orig = models[1][0].net.state_dict()
    new = back[1][0].net.state_dict()
    assert set(orig) == set(new)
    for name in orig:
        assert torch.equal(orig[name], new[name])

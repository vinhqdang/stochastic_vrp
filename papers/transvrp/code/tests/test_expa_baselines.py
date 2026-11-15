"""M7: classical baseline ladder (EXPERIMENT_A.md section 9, M7)."""
import functools

import numpy as np
import pytest

from transvrp.baton.otr21 import simulate_rich
from transvrp.data import route_problem
from transvrp.expa.baselines import classical_ladder, eval_classical
from transvrp.expa.features import RouteStats
from transvrp.expa.paths import route_specs


@functools.lru_cache(maxsize=None)
def _specs():
    return {s.route_id: s for s in route_specs("CON3-0")}


@functools.lru_cache(maxsize=None)
def _problems():
    return list(route_problem("CON3-0", n_train=500, n_test=2000))


def test_con3_sums_match_original_engine():
    specs = _specs()
    tot = {"baton_eq": 0.0, "oracle": 0.0}
    for p in _problems():
        spec = specs[p["route_id"]]
        models = classical_ladder(spec, p["g_train"], ("baton_eq", "oracle"))
        res = eval_classical(spec, models, p["g_test"])
        for lab in tot:
            tot[lab] += res[lab]["mean_cost"]
    assert round(tot["baton_eq"], 3) == 83.073
    assert round(tot["oracle"], 3) == 75.37


def test_all_classical_labels_on_one_route():
    spec = _specs()[0]
    p = next(q for q in _problems() if q["route_id"] == 0)
    g_tr, g_te = p["g_train"], p["g_test"]
    stats = RouteStats(mu_p=g_tr.mean(axis=0), mu_d=np.zeros(spec.m),
                       sd_p=g_tr.std(axis=0, ddof=1), sd_d=np.zeros(spec.m))
    labels = ("none", "tuned_tau", "baton_eq", "dp", "otr21", "oracle")
    models = classical_ladder(spec, g_tr, labels, stats=stats)
    assert set(models) == set(labels)
    assert models["oracle"] is None
    res = eval_classical(spec, models, g_te)
    assert set(res) == set(labels)
    oracle = res["oracle"]["costs"]
    for lab in labels:
        r = res[lab]
        assert set(r) == {"mean_cost", "handoff_rate", "fail_rate", "costs"}
        assert r["costs"].shape == (g_te.shape[0],)
        assert r["mean_cost"] == pytest.approx(float(r["costs"].mean()), rel=0, abs=1e-12)
        assert 0.0 <= r["handoff_rate"] <= 1.0 and 0.0 <= r["fail_rate"] <= 1.0
        # every classical policy has the same action space, so no path beats
        # the clairvoyant one
        assert np.all(r["costs"] >= oracle - 1e-12)
    # 'none' never hands off
    assert res["none"]["handoff_rate"] == 0.0


def test_otr21_adapter_matches_simulate_rich():
    spec = _specs()[0]
    p = next(q for q in _problems() if q["route_id"] == 0)
    g_tr, g_te = p["g_train"], p["g_test"]
    stats = RouteStats(mu_p=g_tr.mean(axis=0), mu_d=np.zeros(spec.m),
                       sd_p=g_tr.std(axis=0, ddof=1), sd_d=np.zeros(spec.m))
    models = classical_ladder(spec, g_tr, ("otr21",), stats=stats)
    res = eval_classical(spec, models, g_te)["otr21"]
    mu = models["otr21"].mu
    sig = models["otr21"].sig
    ref, act = simulate_rich(g_te, mu, sig, spec.B, spec.H, spec.E,
                             models["otr21"].models, return_actions=True,
                             H_bill=spec.H)
    assert res["mean_cost"] == pytest.approx(ref["mean_cost"], rel=0, abs=1e-12)
    assert res["handoff_rate"] == ref["handoff_rate"]
    assert res["fail_rate"] == ref["fail_rate"]


def test_otr21_requires_stats():
    spec = _specs()[0]
    p = next(q for q in _problems() if q["route_id"] == 0)
    with pytest.raises(ValueError):
        classical_ladder(spec, p["g_train"], ("otr21",))


def test_million_labels_refuse_small_fit_sets():
    spec = _specs()[0]
    p = next(q for q in _problems() if q["route_id"] == 0)
    for lab in ("baton_1e6", "dp_1e6"):
        with pytest.raises(ValueError):
            classical_ladder(spec, p["g_train"], (lab,))


def test_unknown_label_is_refused():
    spec = _specs()[0]
    p = next(q for q in _problems() if q["route_id"] == 0)
    with pytest.raises(ValueError):
        classical_ladder(spec, p["g_train"], ("nn_full",))

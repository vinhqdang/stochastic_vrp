"""M11: gradient-boosted continuation values (EXPERIMENT_A.md section 9, M11)."""
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from transvrp.expa import sweep
from transvrp.expa.baselines import classical_ladder, eval_classical
from transvrp.expa.boost import BOOST_PARAMS, BoostPredictor, boost_fit_fn, fit_boost
from transvrp.expa.paths import route_pool, route_specs
from transvrp.expa.simulate import simulate_policy
from transvrp.expa.features import RouteCache


def test_hyperparameters_are_otr21s_without_monotone_constraint():
    assert BOOST_PARAMS["max_iter"] == 120
    assert BOOST_PARAMS["max_depth"] == 3
    assert BOOST_PARAMS["learning_rate"] == 0.1
    assert BOOST_PARAMS["l2_regularization"] == 1.0
    assert BOOST_PARAMS["min_samples_leaf"] == 20
    assert BOOST_PARAMS["random_state"] == 0
    model = HistGradientBoostingRegressor(**BOOST_PARAMS)
    assert model.get_params()["monotonic_cst"] is None


@pytest.mark.slow
def test_con3_0_dev_cost_between_oracle_and_none_on_every_route():
    N = 2000
    pool = route_pool("CON3-0", "train", 0, N)
    dev = route_pool("CON3-0", "dev")
    rows = []
    for spec in route_specs("CON3-0"):
        if spec.route_id not in pool:
            continue
        models, stats, _ = fit_boost(spec, pool[spec.route_id], N, 0)
        cache = RouteCache(spec, dev[spec.route_id], stats)
        costs, _ = simulate_policy(cache, spec.B, spec.H, spec.E, models, H_bill=spec.H)
        none_m = classical_ladder(spec, pool[spec.route_id].g, ("none",))
        ref = eval_classical(spec, {**none_m, "oracle": None}, dev[spec.route_id].g)
        oracle = ref["oracle"]["mean_cost"]
        none = ref["none"]["mean_cost"]
        rows.append((spec.route_id, oracle, float(costs.mean()), none))
    assert len(rows) == 4
    for rid, oracle, boost, none in rows:
        assert oracle - 1e-12 <= boost <= none + 1e-12, (rid, oracle, boost, none)


def test_fallback_predictor_for_few_rows():
    spec = route_specs("CON3-0")[0]
    pool = route_pool("CON3-0", "train", 0, 100)[spec.route_id]
    from transvrp.expa.features import RouteStats
    stats = RouteStats.fit(pool)
    cache = RouteCache(spec, pool, stats)
    fit = boost_fit_fn()
    pred = fit(spec.m - 1, cache, np.arange(5), np.full(5, 1.0),
               np.zeros(0, dtype=np.int64), np.zeros(0), None)
    out = pred.predict(cache, spec.m - 1, np.arange(7))
    assert np.allclose(out, 1.0)


def test_boost_arm_round_trip_through_fit_and_eval(tmp_path):
    rid = next(r for r in route_specs("CON3-0") if r.route_id == 0).route_id
    job = sweep.Job("neural", "CON3-0", rid, 100, 0, arm="boost")
    ctx = sweep.Ctx(tmp_path, "boost", sweep.StageConfig(), "test")
    assert sweep.run_fit_job(job, ctx) == "fit"
    model_path = ctx.models_dir / f"{job.key}.pkl"
    assert sweep.run_eval_job(model_path, "dev", ctx) == "eval"
    rows = pd.read_csv(ctx.rows_dir / f"{job.key}.dev.csv")
    assert list(rows["policy"]) == ["boost"]
    with np.load(ctx.rows_dir / f"{job.key}.dev.npz") as z:
        stored = z["boost"]
    with open(model_path, "rb") as fh:
        payload = pickle.load(fh)
    spec = next(s for s in route_specs(job.inst) if s.route_id == job.route_id)
    pool = route_pool(job.inst, "dev")[job.route_id]
    cache = RouteCache(spec, pool, payload["stats"])
    preds = sweep._unpack(payload["models"])
    assert all(isinstance(p, BoostPredictor) or type(p).__name__ == "ConstantPredictor"
               for lst in preds.values() for p in lst)
    costs, _ = simulate_policy(cache, spec.B, spec.H, spec.E, preds, H_bill=spec.H)
    assert np.array_equal(stored, costs)

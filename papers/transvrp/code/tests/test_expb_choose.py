"""Section 6.1 selection rule (M8): lowest score, ties within 0.10 points go to fewer
passes, then fewer fit seconds, then the lower index; crashes are excluded; the chosen
configurations are read from a committed tuning record, never typed."""
import json

import numpy as np
import pandas as pd
import pytest

from transvrp.expb import choose, stages

ROUTES = [("CON3-1", 0), ("CON3-1", 1), ("SCA3-2", 3)]


def _rows(arm, costs, seconds=10.0, base=1.0, routes=ROUTES, curve=False):
    """Dev rows of one tuning config. ``costs``: {pass: cost} (curve) or a float."""
    out = []
    for inst, rid in routes:
        out.append(dict(inst=inst, route_id=rid, rep=0, policy="baton_eq", mean_cost=base,
                        fit_seconds=1.0))
        items = costs.items() if curve else [(None, costs)]
        for p, c in items:
            pol = f"{arm}@p{p}" if curve else arm
            out.append(dict(inst=inst, route_id=rid, rep=0, policy=pol, mean_cost=c,
                            fit_seconds=seconds))
    return pd.DataFrame(out)


def test_score_is_mean_percent_gap_to_baton_eq():
    rec = choose.choose_config("rlsm", {0: _rows("rlsm", 1.02)}, ROUTES)
    assert rec["config"] == 0
    assert rec["scores"]["c0@p1"] == pytest.approx(2.0)


def test_lowest_score_wins_outside_the_tie_window():
    tune = {0: _rows("rlsm", 1.03), 1: _rows("rlsm", 1.01), 2: _rows("rlsm", 1.02)}
    assert choose.choose_config("rlsm", tune, ROUTES)["config"] == 1


def test_tie_goes_to_fewer_passes_then_fewer_seconds_then_lower_index():
    # best is 1.00 points (c0@p2, c0@p3, c1@p3); c0@p1 is 1.05: tied, and has fewer passes
    tune = {0: _rows("dqn", {1: 1.0105, 2: 1.0100, 3: 1.0100}, curve=True),
            1: _rows("dqn", {1: 1.05, 2: 1.02, 3: 1.0100}, curve=True)}
    rec = choose.choose_config("dqn", tune, ROUTES, curve=True)
    assert (rec["config"], rec["passes"]) == (0, 1)
    tune = {0: _rows("rlsm", 1.0100, seconds=50.0), 1: _rows("rlsm", 1.0105, seconds=20.0)}
    assert choose.choose_config("rlsm", tune, ROUTES)["config"] == 1
    tune = {3: _rows("rlsm", 1.0100), 2: _rows("rlsm", 1.0100)}
    assert choose.choose_config("rlsm", tune, ROUTES)["config"] == 2


def test_fewer_passes_beat_fewer_seconds_inside_the_tie_window():
    slow_short = _rows("dqn", {1: 1.0100, 2: 1.0300, 3: 1.0500}, seconds=90.0, curve=True)
    fast_long = _rows("dqn", {1: 1.0900, 2: 1.0200, 3: 1.0100}, seconds=5.0, curve=True)
    rec = choose.choose_config("dqn", {0: slow_short, 1: fast_long}, ROUTES, curve=True)
    assert (rec["config"], rec["passes"]) == (0, 1)


def test_tie_window_is_a_tenth_of_a_point():
    inside = {0: _rows("rlsm", 1.0100, seconds=50.0), 1: _rows("rlsm", 1.0109, seconds=20.0)}
    outside = {0: _rows("rlsm", 1.0100, seconds=50.0), 1: _rows("rlsm", 1.0111, seconds=20.0)}
    assert choose.choose_config("rlsm", inside, ROUTES)["config"] == 1
    assert choose.choose_config("rlsm", outside, ROUTES)["config"] == 0


def test_non_finite_cost_and_missing_routes_exclude_a_config():
    tune = {0: _rows("rlsm", np.nan), 1: _rows("rlsm", 1.05, routes=ROUTES[:2]),
            2: _rows("rlsm", 1.04)}
    rec = choose.choose_config("rlsm", tune, ROUTES)
    assert rec["config"] == 2
    assert sorted(rec["excluded"]) == [0, 1]


def test_every_config_excluded_is_an_error():
    with pytest.raises(ValueError, match="no admissible"):
        choose.choose_config("rlsm", {0: _rows("rlsm", np.inf)}, ROUTES)


def test_duplicate_rows_are_an_error():
    df = _rows("rlsm", 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        choose.choose_config("rlsm", {0: pd.concat([df, df])}, ROUTES)


def test_curve_arm_requires_all_three_passes():
    tune = {0: _rows("ppo", {1: 1.02, 2: 1.01}, curve=True)}
    with pytest.raises(ValueError, match="no admissible"):
        choose.choose_config("ppo", tune, ROUTES, curve=True)


# ---------------------------------------------------------------- the committed record

def _record():
    tune = {i: _rows("rlsm", 1.05 - 0.01 * i) for i in range(len(stages.TUNE_GRID["rlsm"]))}
    return {"rlsm": choose.choose_config("rlsm", tune, ROUTES)}


def test_record_round_trip_gives_the_grid_pairs(tmp_path):
    path = tmp_path / "tuning_record.json"
    choose.write_record(path, _record(), stages.TUNE_GRID)
    chosen = choose.load_chosen(path, stages.TUNE_GRID)
    assert chosen["rlsm"] == stages.TUNE_GRID["rlsm"][-1]


def test_record_with_hand_edited_pairs_or_choice_is_refused(tmp_path):
    path = tmp_path / "tuning_record.json"
    choose.write_record(path, _record(), stages.TUNE_GRID)
    good = json.loads(path.read_text())
    bad = json.loads(json.dumps(good))
    bad["arms"]["rlsm"]["pairs"] = [["K", 7], ["ridge", 1e-3]]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="grid"):
        choose.load_chosen(path, stages.TUNE_GRID)
    bad = json.loads(json.dumps(good))
    bad["arms"]["rlsm"]["config"] = 0
    bad["arms"]["rlsm"]["pairs"] = [list(p) for p in stages.TUNE_GRID["rlsm"][0]]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="does not follow"):
        choose.load_chosen(path, stages.TUNE_GRID)


def test_record_keeps_the_seconds_that_decided_a_tie(tmp_path):
    tune = {0: _rows("rlsm", 1.0100, seconds=50.0), 1: _rows("rlsm", 1.0105, seconds=20.0)}
    rec = {"rlsm": choose.choose_config("rlsm", tune, ROUTES)}
    assert rec["rlsm"]["config"] == 1
    path = tmp_path / "tuning_record.json"
    choose.write_record(path, rec, stages.TUNE_GRID)
    assert choose.load_chosen(path, stages.TUNE_GRID)["rlsm"] == stages.TUNE_GRID["rlsm"][1]


def test_missing_record_gives_no_chosen_configuration(tmp_path):
    assert choose.load_chosen(tmp_path / "none.json", stages.TUNE_GRID) == {}


def test_record_passes_enter_the_config_pairs_for_curve_arms(tmp_path):
    tune = {i: _rows("dqn", {1: 1.04, 2: 1.03, 3: 1.02 - 0.005 * i}, curve=True)
            for i in range(len(stages.TUNE_GRID["dqn"]))}
    rec = {"dqn": choose.choose_config("dqn", tune, ROUTES, curve=True)}
    path = tmp_path / "tuning_record.json"
    choose.write_record(path, rec, stages.TUNE_GRID)
    pairs = choose.load_chosen(path, stages.TUNE_GRID)["dqn"]
    assert dict(pairs)["passes"] == rec["dqn"]["passes"]
    assert pairs[:-1] == stages.TUNE_GRID["dqn"][rec["dqn"]["config"]]


def test_source_hash_mismatch_is_refused(tmp_path):
    csv = tmp_path / "b_tune_rlsm_c0.csv"
    _rows("rlsm", 1.02).to_csv(csv, index=False)
    rec = {"rlsm": choose.choose_config("rlsm", {0: _rows("rlsm", 1.02)}, ROUTES,
                                        sources={0: csv})}
    path = tmp_path / "tuning_record.json"
    choose.write_record(path, rec, stages.TUNE_GRID)
    choose.verify_sources(path, tmp_path)
    _rows("rlsm", 1.5).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="differs"):
        choose.verify_sources(path, tmp_path)


def test_chosen_arm_cfg_is_generated_from_the_record_not_typed(tmp_path):
    assert stages.CHOSEN_ARM_CFG == stages.chosen_arm_cfg(stages.TUNING_RECORD)
    path = tmp_path / "tuning_record.json"
    choose.write_record(path, _record(), stages.TUNE_GRID)
    cfg = stages.chosen_arm_cfg(path)
    assert cfg["rlsm"] == stages.TUNE_GRID["rlsm"][-1]
    assert cfg["thr"] == () and cfg["alpz"] == ()
    assert "dqn" not in cfg

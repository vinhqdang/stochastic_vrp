"""Ordered cuts of section 9.2 (revision 3.3) as pre-committed configuration."""
import pandas as pd
import pytest

from transvrp.expa import sweep
from transvrp.expb import stages
from transvrp.expb.registry import B_ARMS

ELIG = pd.read_csv(sweep.CODE_DIR / "results" / "expA" / "eligibility_dev.csv")
CHOSEN = {"thr": (), "alpz": (), "rlsm": (("K", 20), ("ridge", 1e-6)),
          "dos": (("lr", 1e-3),),
          "dqn": (("lr", 1e-4), ("target_every_tr", 8_000), ("passes", 3)),
          "ppo": (("lr", 1e-4), ("gamma", 0.5), ("passes", 2))}


def test_no_cut_is_the_default_and_matches_the_registered_stages():
    assert stages.ACTIVE_CUTS == stages.Cuts()
    assert stages.b_stage_defs() == stages.b_stage_defs(stages.Cuts())
    assert "b_dos" in stages.b_stage_defs()
    assert stages.dropped_labels(stages.Cuts()) == []


def test_cut_values_are_validated():
    for bad in (dict(reps=(0,)), dict(reps=(1, 2)), dict(p_main=2), dict(tune_routes="subset3")):
        with pytest.raises(ValueError):
            stages.Cuts(**bad)


def test_cut_one_drops_dos_everywhere_and_labels_it():
    cuts = stages.Cuts(drop_dos=True)
    defs = stages.b_stage_defs(cuts)
    assert not [n for n in defs if "dos" in n]
    for name in ("b_smoke", "b_pilot"):
        assert "dos" not in [a for a, _ in defs[name]["neural"]]
        assert "dos" not in [a for a, _ in defs[name]["arm_cfg"]]
    assert stages.dropped_labels(cuts) == ["dos:cut"]
    assert "b_thr" in defs and "b_dqn" in defs


def test_cut_two_tunes_on_the_six_routes_3_medium_3_long():
    rows = sweep._route_rows("subset6", ELIG)
    sub = ELIG[ELIG["in_1e6_subset"].astype(bool)]
    got = ELIG.set_index(["inst", "route_id"]).loc[rows]
    assert list(got["stratum"]) == ["medium"] * 3 + ["long"] * 3
    for stratum in ("medium", "long"):
        s = sub[sub["stratum"] == stratum].sort_values(["inst", "route_id"]).head(3)
        want = [(r.inst, r.route_id) for r in s.itertuples()]
        assert [r for r, st in zip(rows, got["stratum"]) if st == stratum] == want
    defs = stages.b_stage_defs(stages.Cuts(tune_routes="subset6"))
    assert all(d["routes"] == "subset6" for n, d in defs.items() if n.startswith("b_tune_"))
    assert defs["b_thr"]["routes"] == "eligible"


def test_cut_three_uses_two_replicates_for_every_main_stage_but_not_the_rest():
    defs = stages.b_stage_defs(stages.Cuts(reps=(0, 1)), chosen=CHOSEN)
    for name in ("b_ladder", "b_full", *(f"b_{a}" for a in B_ARMS)):
        assert defs[name]["reps"] == (0, 1), name
    assert defs["b_thr_ind"]["reps"] == (0,)
    assert defs["b_tune_dqn_c0"]["reps"] == (0,)
    assert stages.b_stage_defs(chosen=CHOSEN)["b_full"]["reps"] == (0, 1, 2)


def test_cut_four_fixes_one_pass_for_dqn_and_ppo_only():
    defs = stages.b_stage_defs(stages.Cuts(p_main=1), chosen=CHOSEN)
    for arm in ("dqn", "ppo"):
        assert dict(defs[f"b_{arm}"]["arm_cfg"][0][1])["passes"] == 1
    assert dict(defs["b_rlsm"]["arm_cfg"][0][1]) == dict(CHOSEN["rlsm"])
    free = stages.b_stage_defs(chosen=CHOSEN)
    assert dict(free["b_ppo"]["arm_cfg"][0][1])["passes"] == 2


def test_cut_record_is_json_able_and_lists_active_cuts():
    rec = stages.cut_record(stages.Cuts(drop_dos=True, p_main=1))
    assert rec == {"drop_dos": True, "tune_routes": "subset", "reps": [0, 1, 2], "p_main": 1}
    assert stages.cut_record(stages.Cuts())["drop_dos"] is False

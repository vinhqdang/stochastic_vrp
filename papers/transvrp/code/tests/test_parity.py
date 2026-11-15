"""Golden numbers produced by the ORIGINAL engine (svrpspd_wdro, commit 37888fa)
on CON3-0, plan 'Det', n_train=500, n_test=2000. The copied code must reproduce
them exactly, so any later edit to baton/ or data.py that changes behaviour fails."""
import numpy as np
from transvrp.benchmark import instance_table

GOLD = {"baton": 83.073, "oracle": 75.37}


def test_baton_and_oracle_match_original_engine():
    rows = instance_table("CON3-0")
    assert len(rows) == 4
    for k, v in GOLD.items():
        assert round(float(np.sum([r[k] for r in rows])), 3) == v


def test_ladder_is_ordered():
    rows = instance_table("CON3-0")
    t = {k: sum(r[k] for r in rows) for k in ("none", "baton", "oracle")}
    assert t["oracle"] <= t["baton"] <= t["none"]

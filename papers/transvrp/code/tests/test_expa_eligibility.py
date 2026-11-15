"""M8: route eligibility on the dev stream (EXPERIMENT_A.md section 9, M8)."""
import numpy as np
import pandas as pd
import pytest

from transvrp.expa.eligibility import (COLUMNS, build_eligibility,
                                       eligible_routes, stratum_of)

ALL_INSTS = [f"{p}-{j}" for p in ("CON3", "SCA3") for j in range(10)]


@pytest.fixture(scope="module")
def con3_0():
    return build_eligibility(["CON3-0"])


def test_stratum_boundaries():
    assert stratum_of(3) == "tiny"
    assert stratum_of(4) == "short" and stratum_of(9) == "short"
    assert stratum_of(10) == "medium" and stratum_of(14) == "medium"
    assert stratum_of(15) == "long"


def test_columns_and_one_row_per_route(con3_0):
    assert list(con3_0.columns) == COLUMNS
    assert con3_0.shape[0] == 4
    assert con3_0["route_id"].is_unique


def test_rule_applied_correctly(con3_0):
    expected = (con3_0["m"] >= 4) & (con3_0["dev_breach_rate"] >= 0.02)
    assert (con3_0["eligible"].astype(bool) == expected).all()
    assert con3_0["eligible"].all()          # the four CON3-0 routes qualify
    assert (con3_0["dev_breach_rate"].between(0.0, 1.0)).all()


def test_subset_only_contains_eligible_routes(con3_0):
    sub = con3_0[con3_0["in_1e6_subset"]]
    assert sub["eligible"].all()


def test_deterministic_and_idempotent_on_tmp_path(tmp_path):
    out = tmp_path / "elig.csv"
    first = eligible_routes(["CON3-0"], out)
    assert out.exists()
    second = eligible_routes(["CON3-0"], out)       # file exists: compare only
    pd.testing.assert_frame_equal(first, second)
    third = eligible_routes(["CON3-0"], out, overwrite=True)
    pd.testing.assert_frame_equal(first, third)


def test_existing_file_that_disagrees_is_refused(tmp_path):
    out = tmp_path / "elig.csv"
    df = eligible_routes(["CON3-0"], out)
    bad = df.copy()
    bad.loc[0, "eligible"] = not bad.loc[0, "eligible"]
    bad.to_csv(out, index=False)
    with pytest.raises(ValueError):
        eligible_routes(["CON3-0"], out)


@pytest.mark.slow
def test_all_twenty_instances_match_the_spec(tmp_path):
    df = eligible_routes(ALL_INSTS, tmp_path / "all.csv")
    assert len(df) == 80
    assert int(df["eligible"].sum()) == 62
    per_inst = df.groupby("inst")["eligible"].sum()
    assert sorted(per_inst.value_counts().to_dict().items()) == [(3, 18), (4, 2)]
    counts = df[df["eligible"]].groupby("stratum").size().to_dict()
    assert counts == {"short": 2, "medium": 28, "long": 32}
    sub = df[df["in_1e6_subset"]]
    assert len(sub) == 12
    assert (sub["stratum"] == "medium").sum() == 6
    assert (sub["stratum"] == "long").sum() == 6

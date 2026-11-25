"""EXPERIMENT C analysis (EXPERIMENT_C.md 1.5-1.9, 5, 7; MC5) on synthetic stage tables:
family C (Holm over 11, the classes with delta and delta_Q1), the share-of-gain bootstrap,
the summed-cost estimand with its exclusion floor, the gap share, and the refusals of
``load_tables``. Nothing here reads test_c data or the committed results."""
import numpy as np
import pandas as pd
import pytest

from transvrp.expa import analysis as A
from transvrp.expb import analysis_b as B
from transvrp.expc import analysis_c as AC
from transvrp.expc import gates as CG

FAM_SHA = "f" * 64
N = AC.N_PATHS
ROUTES = AC.N_ROUTES
INSTANCES = AC.N_INSTANCES
REPS = (0, 1)


def _stage_factor_table():
    """(stage, policy, factor on the route baseline) for every stage the analysis uses."""
    rows = []
    for t in ("w0", "w3", "w6", "w9", "ws"):
        rows.append((f"c_ladder_{t}", "none", 1.20))
        rows.append((f"c_ladder_{t}", "baton_eq", 1.00))
        rows.append((f"c_ladder_{t}", "oracle", 0.90))
        rows.append((f"c_thr_{t}", "thr", 1.01))
        rows.append((f"c_cdp_{t}", "cdp", 0.995))
    rows += [("c_cdptrue_w6", "cdp", 0.996), ("c_dos_w6", "dos", 1.02),
             ("c_full_w6", "nn_full", 0.97), ("c_ppob_w6", "ppo", 0.99),
             ("c_ppolong_w6", "ppo", 0.965)]
    boost = {"F0": 1.06, "F0n": 1.05, "F1": 1.03, "F2": 1.02, "F3": 1.01}
    for feat, factor in boost.items():
        rows.append((f"c_boost_{feat}_w6", "boost", factor))
        rows.append((f"c_rlsm_{feat}_w6", "rlsm", factor + 0.005))
    for feat in ("F0", "F3"):
        for t in ("w0", "w3", "w9", "ws"):
            rows.append((f"c_boost_{feat}_{t}", "boost", boost[feat]))
            rows.append((f"c_rlsm_{feat}_{t}", "rlsm", boost[feat] + 0.005))
    return rows


def synthetic(*, eval_set="test_c", family_sha=FAM_SHA, n_eval=20000, eval_commit="c" * 40,
              only=None, extra=None):
    """Stage tables for 62 routes over 20 instances, reps 0-1: cost = (10 + 0.1 k) * factor
    with a small replicate effect."""
    rows = []
    for stage, policy, factor in _stage_factor_table():
        if only is not None and stage not in only:
            continue
        for k in range(ROUTES):
            inst = f"I{k % INSTANCES:02d}"
            for rep in REPS:
                rows.append(dict(
                    inst=inst, route_id=k, N=N, rep=rep, policy=policy, stage=stage,
                    mean_cost=(10.0 + 0.1 * k) * factor * (1.0 + 0.001 * rep),
                    gap_closed=np.nan, eval_set=eval_set, n_eval=n_eval,
                    fit_commit="a" * 40, eval_commit=eval_commit,
                    family_sha256=family_sha, stratum="medium", world="W6",
                    feature_set=""))
    df = pd.DataFrame(rows)
    if extra is not None:
        df = pd.concat([df, extra], ignore_index=True)
    return df


# ---------------------------------------------------------------- Holm and the margins

def test_holm_over_the_eleven_members_matches_the_hand_computation():
    p = [0.01 * (i + 1) for i in range(11)]
    expected = [0.11, 0.2, 0.27, 0.32, 0.35, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36]
    got = A.holm(p)
    assert np.allclose(got, expected, atol=1e-12)
    assert len(got) == 11 and AC.HOLM_SIZE == 11 == len(AC.FAMILY)


def test_family_c_has_the_eleven_contrasts_of_section_1_9_in_order():
    ids = [c[0] for c in AC.FAMILY]
    assert ids == ["C1-B", "C1-R", "C2-B", "C2-R", "C3-B", "C3-R", "C4-B", "C4-R",
                   "C5", "C6", "C7"]
    margins = [c[5] for c in AC.FAMILY]
    assert margins[:8] == [AC.CG.DELTA_Q1] * 8 and margins[8:] == [AC.CG.DELTA] * 3
    assert AC.CG.DELTA_Q1 == 0.25 and AC.CG.DELTA == 0.7


def test_holm_interval_over_eleven_uses_the_one_minus_quantile_of_section_1_3():
    d = np.linspace(-1.0, 1.0, 20)
    lo, hi = B.holm_ci(d, 11)
    ref = A._ci(d, 1.0 - 0.025 / 11)
    assert (lo, hi) == pytest.approx(ref)


@pytest.mark.parametrize("mean, expected", [
    (0.25, "better"),          # the boundary is inside: mean >= delta_Q1
    (0.2499, "equivalent"),    # just below: CI90 inside (-0.25, 0.25)
    (0.20, "equivalent"),
    (-0.25, "worse"),
])
def test_delta_q1_classification_boundaries(mean, expected):
    d = np.full(20, mean)
    assert A.classify(d, AC.CG.DELTA_Q1, p_adj=0.001) == expected


def test_the_same_mean_is_not_better_under_the_family_c5_margin():
    d = np.full(20, 0.25)
    assert A.classify(d, AC.CG.DELTA, p_adj=0.001) == "equivalent"


def test_a_contrast_whose_holm_p_is_not_significant_is_not_better():
    d = np.full(20, 0.3)
    assert A.classify(d, AC.CG.DELTA_Q1, p_adj=0.2) != "better"


def test_family_table_on_synthetic_tables_has_eleven_rows_with_holm_above_raw():
    df = synthetic()
    table = AC.family_table(df)
    assert len(table) == 11
    assert list(table["id"]) == [c[0] for c in AC.FAMILY]
    assert (table["p_holm"] >= table["p_raw"] - 1e-12).all()
    assert np.allclose(table["p_holm"], A.holm(list(table["p_raw"])))
    assert (table["p_holm"] > table["p_raw"]).any()
    assert set(table["classification"]) <= {"better", "worse", "equivalent", "inconclusive"}


def test_family_table_refuses_a_contrast_with_missing_rows():
    df = synthetic(only=["c_ladder_w6", "c_cdp_w6"])
    with pytest.raises(AC.AnalysisRefused):
        AC.family_table(df)


# ---------------------------------------------------------------- Q1 share of gain

def test_share_bootstrap_is_reproducible_under_the_registered_seed():
    rng = np.random.default_rng(3)
    num = rng.normal(0.5, 0.1, size=20)
    den = rng.normal(1.0, 0.1, size=20)
    first = AC.share_bootstrap(num, den)
    second = AC.share_bootstrap(num, den)
    assert first["ci95"] == second["ci95"]
    assert first["n_draws"] == 10_000 and first["seed"] == AC.SHARE_SEED == 20261125
    assert first["share"] == pytest.approx(num.mean() / den.mean())


def test_share_bootstrap_matches_a_manual_resampling():
    num = np.arange(1.0, 21.0)
    den = np.full(20, 2.0)
    got = AC.share_bootstrap(num, den, n_draws=500, seed=7)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, 20, size=(500, 20))
    draws = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    assert got["ci95"] == pytest.approx((lo, hi))


def test_share_bootstrap_changes_with_the_seed():
    rng = np.random.default_rng(4)
    num = rng.normal(0.5, 0.2, size=20)
    den = rng.normal(1.0, 0.2, size=20)
    assert AC.share_bootstrap(num, den, seed=1)["ci95"] != AC.share_bootstrap(num, den, seed=2)["ci95"]


def test_share_bootstrap_refuses_unpaired_or_empty_vectors():
    with pytest.raises(ValueError):
        AC.share_bootstrap(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        AC.share_bootstrap(np.array([]), np.array([]))


def test_share_table_gives_one_for_the_rich_set_and_a_share_for_each_feature_set():
    table = AC.share_table(synthetic())
    assert len(table) == 8
    f3 = table[table["feature_set"] == "F3"]
    assert np.allclose(f3["share"], 1.0)
    f1 = table[(table["learner"] == "boost") & (table["feature_set"] == "F1")]
    assert 0.0 < float(f1["share"].iloc[0]) < 1.0


# ---------------------------------------------------------------- Q2 summed costs

def _two_stage_frame(costs_a, costs_b, inst_route):
    rows = []
    for (inst, rid), (ca, cb) in zip(inst_route, zip(costs_a, costs_b)):
        for rep in REPS:
            rows.append(dict(inst=inst, route_id=rid, rep=rep, N=N, policy="x", stage="a",
                             mean_cost=ca))
            rows.append(dict(inst=inst, route_id=rid, rep=rep, N=N, policy="y", stage="b",
                             mean_cost=cb))
    return pd.DataFrame(rows)


def test_summed_cost_estimand_is_the_ratio_of_sums_over_the_instance():
    df = _two_stage_frame([9.0, 9.0], [10.0, 10.0], [("I00", 0), ("I00", 1)])
    e0 = {("I00", 0): 10.0, ("I00", 1): 10.0}
    res = AC.sigma_instances(df, ("a", "x"), ("b", "y"), e0)
    assert res["values"] == pytest.approx([100.0 * (20.0 - 18.0) / 20.0])
    assert res["excluded"] == 0 and res["instances"] == ["I00"]


def test_summed_cost_estimand_excludes_an_instance_below_the_floor_and_counts_it():
    df = pd.concat([
        _two_stage_frame([9.0, 9.0], [10.0, 10.0], [("I00", 0), ("I00", 1)]),
        _two_stage_frame([1e-7, 1e-7], [1e-6, 1e-6], [("I01", 2), ("I01", 3)]),
    ], ignore_index=True)
    e0 = {("I00", 0): 10.0, ("I00", 1): 10.0, ("I01", 2): 10.0, ("I01", 3): 10.0}
    res = AC.sigma_instances(df, ("a", "x"), ("b", "y"), e0)
    assert res["excluded"] == 1
    assert res["instances"] == ["I00"]
    assert res["values"] == pytest.approx([10.0])


def test_summed_cost_estimand_refuses_mismatched_route_coverage():
    a_rows = _two_stage_frame([9.0], [9.0], [("I00", 0)])
    b_rows = _two_stage_frame([9.0, 10.0], [9.0, 10.0], [("I00", 0), ("I00", 5)])
    df = pd.concat([a_rows[a_rows["policy"] == "x"],
                    b_rows[b_rows["policy"] == "y"]], ignore_index=True)
    with pytest.raises(AC.AnalysisRefused, match="same routes"):
        AC.sigma_instances(df, ("a", "x"), ("b", "y"), {("I00", 0): 1.0, ("I00", 5): 1.0})


def test_gap_share_is_the_fraction_of_the_baton_to_oracle_gap_closed_by_full():
    df = synthetic(only=["c_ladder_w6", "c_full_w6"])
    cb = AC.route_means(df, "c_ladder_w6", "baton_eq")
    cf = AC.route_means(df, "c_full_w6", "nn_full")
    co = AC.route_means(df, "c_ladder_w6", "oracle")
    expected_per_instance = []
    for inst in sorted({k[0] for k in cb.index}):
        keys = [k for k in cb.index if k[0] == inst]
        expected_per_instance.append(float((cb.loc[keys] - cf.loc[keys]).sum()) /
                                     float((cb.loc[keys] - co.loc[keys]).sum()))
    table = AC.gap_table(df)
    assert table["n"].iloc[0] == INSTANCES
    assert table["mean_share"].iloc[0] == pytest.approx(np.mean(expected_per_instance))


def test_gap_share_excludes_an_instance_without_a_positive_gap():
    df = synthetic(only=["c_ladder_w6", "c_full_w6"])
    oracle_row = (df["stage"] == "c_ladder_w6") & (df["policy"] == "oracle") & (df["inst"] == "I00")
    df.loc[oracle_row, "mean_cost"] = 1e6          # the oracle costs more than BATON here
    table = AC.gap_table(df)
    assert table["excluded"].iloc[0] == 1


# ---------------------------------------------------------------- refusals on the inputs

def _write_tables(tmp_path, df, name="c_thr_w6"):
    path = tmp_path / f"{name}.test_c.csv"
    df.to_csv(path, index=False)
    return path


def test_load_accepts_a_clean_test_c_table(tmp_path):
    df = synthetic(only=["c_thr_w6"])
    path = _write_tables(tmp_path, df.drop(columns=["stage"]))
    out = AC.load_tables([path], eval_set="test_c", family_sha256=FAM_SHA)
    assert set(out["stage"]) == {"c_thr_w6"}


def test_load_refuses_rows_not_from_test_c(tmp_path):
    df = synthetic(only=["c_thr_w6"], eval_set="dev")
    path = _write_tables(tmp_path, df.drop(columns=["stage"]))
    with pytest.raises(AC.AnalysisRefused, match="eval_set"):
        AC.load_tables([path], eval_set="test_c", family_sha256=FAM_SHA)


def test_load_refuses_a_family_hash_that_is_not_the_pinned_one(tmp_path):
    df = synthetic(only=["c_thr_w6"], family_sha="0" * 64)
    path = _write_tables(tmp_path, df.drop(columns=["stage"]))
    with pytest.raises(AC.AnalysisRefused, match="family_sha256"):
        AC.load_tables([path], eval_set="test_c", family_sha256=FAM_SHA)


def test_load_refuses_a_test_c_table_without_the_pinned_hash(tmp_path):
    df = synthetic(only=["c_thr_w6"])
    path = _write_tables(tmp_path, df.drop(columns=["stage"]))
    with pytest.raises(AC.AnalysisRefused, match="family hash"):
        AC.load_tables([path], eval_set="test_c", family_sha256=None)


def test_load_refuses_the_wrong_day_count(tmp_path):
    df = synthetic(only=["c_thr_w6"], n_eval=5000)
    path = _write_tables(tmp_path, df.drop(columns=["stage"]))
    with pytest.raises(AC.AnalysisRefused, match="day count"):
        AC.load_tables([path], eval_set="test_c", family_sha256=FAM_SHA)


def test_load_refuses_mixed_eval_commits(tmp_path):
    a = synthetic(only=["c_thr_w6"], eval_commit="a" * 40)
    b = synthetic(only=["c_cdp_w6"], eval_commit="b" * 40)
    pa = _write_tables(tmp_path, a.drop(columns=["stage"]), "c_thr_w6")
    pb = _write_tables(tmp_path, b.drop(columns=["stage"]), "c_cdp_w6")
    with pytest.raises(AC.AnalysisRefused, match="eval commits"):
        AC.load_tables([pa, pb], eval_set="test_c", family_sha256=FAM_SHA)


def test_a_file_that_is_not_a_stage_table_is_refused(tmp_path):
    df = synthetic(only=["c_thr_w6"])
    path = tmp_path / "c_thr_w6.csv"
    df.drop(columns=["stage"]).to_csv(path, index=False)
    with pytest.raises(AC.AnalysisRefused, match="not a test_c"):
        AC.load_tables([path], eval_set="test_c", family_sha256=FAM_SHA)


def test_check_complete_names_the_missing_stages():
    with pytest.raises(AC.AnalysisRefused, match="missing"):
        AC.check_complete(synthetic(only=["c_thr_w6"]))


def test_required_stages_cover_the_spec_table():
    names = AC.required_stages()
    for stage in ("c_ladder_ws", "c_cdptrue_w6", "c_ppob_w6", "c_full_w6", "c_ppolong_w6",
                  "c_dos_w6", "c_boost_F0_ws", "c_rlsm_F3_w9", "c_boost_F0n_w6",
                  "c_cdp_w0"):
        assert stage in names, stage


def test_ppo_table_refuses_when_the_subset_is_not_twelve_routes():
    with pytest.raises(AC.AnalysisRefused, match="subset"):
        AC.ppo_table(synthetic(only=["c_full_w6"]))


def test_summary_is_labelled_dev_and_names_the_family_line_for_test_c():
    tables = {"family": AC.family_table(synthetic())}
    dev = AC.render_summary(tables, eval_set="dev", label="DEV: no family line")
    assert dev.startswith("# EXPERIMENT C analysis (DEV, pilot: no family check)")
    test = AC.render_summary(tables, eval_set="test_c", label="family line X",
                             eval_commit="abc")
    assert "(test_c)" in test and "family line X" in test and "DEV" not in test


def test_family_hash_is_the_sha256_of_the_line():
    line = "TEST_C FAMILY CLOSED: family=[C5]"
    import hashlib
    assert AC.family_hash(line) == hashlib.sha256(line.encode("utf-8")).hexdigest()


def test_cdp_table_has_one_row_per_world_plus_cdp_true():
    table = AC.cdp_table(synthetic())
    assert len(table) == 6
    assert list(table["comparison"]).count("CDP vs THR") == 5
    assert "CDP-true vs CDP" in set(table["comparison"])


# ------------------------------------------------------------ breach table (MC5 issue 5)

def _write_day_file(rows_dir, inst, rid, N, rep, arrays):
    from transvrp.expa import sweep
    job = sweep.Job(kind="ladder", inst=inst, route_id=rid, N=N, rep=rep)
    rows_dir.mkdir(parents=True, exist_ok=True)
    np.savez(rows_dir / sweep.rows_file_name(job.key, "test_c", "npz"), **arrays)


def test_breach_rates_come_from_the_none_policy_when_there_is_no_breach_array(tmp_path):
    rows = tmp_path / "c_ladder_w6" / "rows"
    _write_day_file(rows, "CON3-0", 0, 100000, 0, {"none": np.array([0.0, 0.5, 0.0, 1.0])})
    _write_day_file(rows, "CON3-0", 0, 100000, 1, {"none": np.zeros(4)})
    _write_day_file(rows, "CON3-0", 1, 100000, 0, {"none": np.zeros(4)})
    _write_day_file(rows, "CON3-0", 1, 100000, 1, {"none": np.zeros(4)})
    t = AC.breach_table({"w6": rows}, [("CON3-0", 0), ("CON3-0", 1)], 100000)
    row = t.iloc[0]
    # route 0: replicate breach rates 0.5 and 0 -> 0.25; route 1: 0; world mean 0.125
    assert row["breach_rate"] == pytest.approx(0.125)
    assert row["routes_no_breach"] == 1 and row["routes"] == 2


def test_an_explicit_breach_array_takes_precedence_over_the_none_costs(tmp_path):
    rows = tmp_path / "c_ladder_w6" / "rows"
    _write_day_file(rows, "CON3-0", 0, 100000, 0,
                    {"none": np.array([1.0, 1.0]), "breach": np.array([0.0, 1.0])})
    t = AC.breach_table({"w6": rows}, [("CON3-0", 0)], 100000)
    assert t.iloc[0]["breach_rate"] == pytest.approx(0.5)


def test_breach_table_refuses_a_day_file_with_neither_breach_nor_none(tmp_path):
    rows = tmp_path / "c_ladder_w6" / "rows"
    _write_day_file(rows, "CON3-0", 0, 100000, 0, {"baton_eq": np.zeros(3)})
    with pytest.raises(AC.AnalysisRefused):
        AC.breach_table({"w6": rows}, [("CON3-0", 0)], 100000)


# ---------------------------------------------------------------- pinned lists (MC5 item 3)

PINNED_LINE = ("TEST_C FAMILY CLOSED: family=[{fam}] descriptive=[{desc}] dropped=[{drop}] "
               "references=[baton,ladder,ppo_b] fidelity_sha256=" + "a" * 64 +
               " manifests_sha256=" + "b" * 64 + " code_sha256=" + "c" * 64 +
               " analysis_sha256=" + "d" * 64)
PINNED = dict(fam="C1-B,C1-R,C2-B,C2-R,C3-B,C3-R,C4-B,C4-R,C7", desc="C5",
              drop="C6,boost_f1:p1")
PINNED_FAMILY = ["C1-B", "C1-R", "C2-B", "C2-R", "C3-B", "C3-R", "C4-B", "C4-R", "C7"]


def _pinned_lists():
    return AC.family_lists(PINNED_LINE.format(**PINNED))


def test_pinned_family_line_is_parsed_into_its_four_lists():
    assert _pinned_lists() == {
        "family": PINNED_FAMILY, "descriptive": ["C5"], "dropped": ["C6", "boost_f1:p1"],
        "references": ["baton", "ladder", "ppo_b"]}


def test_family_line_that_does_not_match_the_stream_pattern_is_refused():
    with pytest.raises(AC.AnalysisRefused, match="malformed"):
        AC.family_lists("TEST_C FAMILY CLOSED: family=[C1-B]")


def test_holm_runs_over_the_pinned_family_only_and_records_its_size():
    table = AC.family_table(synthetic(), lists=_pinned_lists())
    fam = table[table["role"] == "family"]
    assert list(fam["id"]) == PINNED_FAMILY
    assert (fam["holm_k"] == 9).all()
    assert np.allclose(fam["p_holm"], A.holm(list(fam["p_raw"])))


def test_descriptive_contrast_has_no_holm_p_and_no_win_or_tie_class():
    table = AC.family_table(synthetic(), lists=_pinned_lists())
    row = table[table["id"] == "C5"].iloc[0]
    assert row["role"] == "descriptive" and row["classification"] == "descriptive"
    assert np.isnan(row["p_holm"]) and np.isnan(row["holm_lo"]) and np.isnan(row["holm_hi"])
    assert np.isnan(row["holm_k"]) and np.isfinite(row["p_raw"])


def test_dropped_contrast_is_omitted_and_noted():
    table = AC.family_table(synthetic(), lists=_pinned_lists())
    assert "C6" not in set(table["id"])
    assert list(table["id"]) == PINNED_FAMILY[:8] + ["C5", "C7"]
    notes = AC.omission_notes(_pinned_lists())
    assert any("C6" in note for note in notes)
    assert any("boost_f1:p1" in note and "gate items" in note for note in notes)


def test_contrast_named_by_no_pinned_list_is_refused():
    lists = dict(_pinned_lists(), descriptive=[], dropped=[])
    with pytest.raises(AC.AnalysisRefused, match="not named"):
        AC.contrast_roles(lists)


def test_unknown_contrast_in_the_pinned_lists_is_refused():
    lists = dict(_pinned_lists(), family=PINNED_FAMILY + ["C9"])
    with pytest.raises(AC.AnalysisRefused, match="unknown"):
        AC.family_table(synthetic(), lists=lists)


def test_contrast_with_two_roles_is_refused():
    lists = dict(_pinned_lists(), descriptive=["C5", "C7"])
    with pytest.raises(AC.AnalysisRefused, match="overlap"):
        AC.contrast_roles(lists)


def test_default_lists_are_all_eleven_contrasts_in_the_family():
    table = AC.family_table(synthetic())
    assert (table["role"] == "family").all() and (table["holm_k"] == 11).all()


# ---------------------------------------------------------------- exact-zero route (item 7)

def test_w0_route_with_exactly_zero_summed_cost_is_excluded_not_divided_by_zero():
    """A W0 instance whose routes cost exactly zero on the baseline sum to zero and fall
    under the 1e-3 E0 floor: excluded and counted, never a NaN or a ZeroDivisionError."""
    df = pd.concat([
        _two_stage_frame([9.0, 9.0], [10.0, 10.0], [("I00", 0), ("I00", 1)]),
        _two_stage_frame([0.0, 0.0], [0.0, 0.0], [("I01", 2), ("I01", 3)]),
    ], ignore_index=True)
    e0 = {("I00", 0): 10.0, ("I00", 1): 10.0, ("I01", 2): 10.0, ("I01", 3): 10.0}
    res = AC.sigma_instances(df, ("a", "x"), ("b", "y"), e0)
    assert res["excluded"] == 1 and res["instances"] == ["I00"]
    assert np.isfinite(res["values"]).all()

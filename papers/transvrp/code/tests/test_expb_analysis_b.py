"""M23: EXPERIMENT B test_b analysis (expb/analysis_b.py; EXPERIMENT_B.md 1, 5.3, 8).

All decision tests use constructed deterministic tables with known answers. No test
data is read. Synthetic routes: 20 instances x 2 routes (40 routes), replicates 0-2;
the independence world: 10 instances, 12 routes, rep 0.
"""
import hashlib
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from transvrp.expa import analysis as A
from transvrp.expa.paths import TEST_B_N
from transvrp.expb import analysis_b as B
from transvrp.expb import registry

L = np.linspace(-1, 1, 20)
INST = [f"I{i:02d}" for i in range(20)]
ROUTES = [(i, r) for i in INST for r in (0, 1)]
NR = len(ROUTES)              # 40 synthetic routes (production: 62)
REPS = (0, 1, 2)
FAMILY_LINE = ("TEST_B FAMILY CLOSED: family=[thr,alpz,rlsm,dos,dqn,ppo] descriptive=[] "
               "dropped=[] references=[full,ladder] independence=[] "
               "fidelity_sha256=" + "1" * 64 + " manifests_sha256=" + "2" * 64
               + " code_sha256=" + "3" * 64 + " analysis_sha256=" + "4" * 64)
SHA = hashlib.sha256(FAMILY_LINE.encode("utf-8")).hexdigest()
FIT = "a" * 40
EVAL = "b" * 40
ARMS6 = ("thr", "alpz", "rlsm", "dos", "dqn", "ppo")
LISTS6 = dict(family=list(ARMS6), descriptive=[], dropped=[],
              references=["full", "ladder"], independence=[])


def _rows(stage, policy, cost, routes, reps, **over):
    base = dict(N=100_000, n_eval=TEST_B_N, eval_set="test_b", fit_commit=FIT,
                eval_commit=EVAL, family_sha256=SHA, gap_closed=0.5, fit_seconds=1.0)
    base.update(over)
    return [dict(base, stage=stage, inst=i, route_id=r, rep=rep, policy=policy,
                 mean_cost=cost[i]) for i, r in routes for rep in reps]


def _cost_from_delta(d_by_inst):
    """Arm cost with instance delta d_i against FULL at 100: 100*(C_b - 100)/C_b = d_i."""
    return {i: 100.0 / (1.0 - d / 100.0) for i, d in d_by_inst.items()}


def _main_frame(deltas, baseline=None, routes=ROUTES, **over):
    """FULL at 100 on every route; each arm's delta vs FULL is deltas[arm][instance];
    ladder: baton_eq at the baseline deltas (default 0), none 120, oracle 80."""
    insts = sorted({i for i, _ in routes})
    frames = [_rows("b_full", "nn_full", {i: 100.0 for i in insts}, routes, REPS, **over)]
    for arm, d in deltas.items():
        cost = _cost_from_delta(dict(zip(insts, d)))
        frames.append(_rows(f"b_{arm}", arm, cost, routes, REPS, **over))
    base = {i: 0.0 for i in insts} if baseline is None else dict(zip(insts, baseline))
    lad_base = _cost_from_delta(base)
    frames.append(_rows("b_ladder", "baton_eq", lad_base, routes, REPS, **over))
    frames.append(_rows("b_ladder", "none", {i: 120.0 for i in insts}, routes, REPS, **over))
    frames.append(_rows("b_ladder", "oracle", {i: 80.0 for i in insts}, routes, REPS, **over))
    return pd.DataFrame(frames_flat(frames))


def frames_flat(frames):
    return [row for block in frames for row in block]


def _tie(overrides=None):
    """Every B arm TIE-like (0.1 L) unless overridden: a complete six-arm table."""
    deltas = {arm: 0.1 * L for arm in ARMS6}
    deltas.update(overrides or {})
    return deltas


def _ind_frame(d10):
    """Independence world: 10 instances, 12 routes, rep 0 (section 3.8)."""
    insts = [f"I{i:02d}" for i in range(10)]
    routes = [("I00", 0), ("I00", 1), ("I01", 0), ("I01", 1)] + [(i, 0) for i in insts[2:]]
    d = dict(zip(insts, d10))
    rows = _rows("b_full_ind", "nn_full", {i: 100.0 for i in insts}, routes, (0,))
    rows += _rows("b_thr_ind", "thr", _cost_from_delta(d), routes, (0,))
    rows += _rows("b_alpz_ind", "alpz", _cost_from_delta(d), routes, (0,))
    rows += _rows("b_ladder_ind", "baton_eq", _cost_from_delta({i: 0.0 for i in insts}),
                  routes, (0,))
    rows += _rows("b_ladder_ind", "none", {i: 120.0 for i in insts}, routes, (0,))
    rows += _rows("b_ladder_ind", "oracle", {i: 80.0 for i in insts}, routes, (0,))
    return rows


def _csv_dir(tmp_path, df):
    paths = []
    for stage, grp in df.groupby("stage"):
        path = tmp_path / f"{stage}.test_b.csv"
        grp.drop(columns=["stage"]).to_csv(path, index=False)
        paths.append(path)
    return paths


# ---------------------------------------------------------------- constants and hash

def test_arm_list_matches_registry():
    assert B.B_ARMS == registry.B_ARMS


def test_family_hash_is_sha256_of_the_line():
    assert B.family_hash(FAMILY_LINE) == SHA
    assert B.family_hash(FAMILY_LINE + " ") != SHA


# ---------------------------------------------------------------- Holm-level CI

def test_holm_ci_size_one_is_ci95():
    d = 2 + 0.1 * L
    assert B.holm_ci(d, 1) == pytest.approx(A._ci(d, 0.975), abs=1e-12)


def test_holm_ci_size_six_uses_one_minus_0_025_over_6():
    d = 2 + 0.1 * L
    assert B.holm_ci(d, 6) == pytest.approx(A._ci(d, 1 - 0.025 / 6), abs=1e-12)
    lo1, hi1 = B.holm_ci(d, 1)
    lo6, hi6 = B.holm_ci(d, 6)
    assert lo6 < lo1 and hi6 > hi1


def test_holm_ci_rejects_empty_family():
    with pytest.raises(ValueError):
        B.holm_ci(L, 0)


# ---------------------------------------------------------------- branches (1.3)

def _p(d):
    return A._wilcoxon_p(np.asarray(d, dtype=float))


def test_go_no_go_holm_reproduces_a_vectors_with_raw_p():
    assert B.go_no_go_holm(2 + 0.1 * L, _p(2 + 0.1 * L)) == "WIN"
    assert B.go_no_go_holm(-2 + 0.1 * L, _p(-2 + 0.1 * L)) == "LOSE"
    assert B.go_no_go_holm(0.1 * L, _p(0.1 * L)) == "TIE_STRONG"
    assert B.go_no_go_holm(5 * L, _p(5 * L)) == "TIE_WEAK"


def test_delta_boundary_is_included_in_win_and_just_below_is_not():
    d = 0.7 + 0.05 * L
    assert float(d.mean()) == pytest.approx(0.7, abs=1e-9)
    assert B.go_no_go_holm(d, _p(d)) == "WIN"
    assert B.go_no_go_holm(0.6 + 0.05 * L, _p(0.6 + 0.05 * L)) == "TIE_STRONG"


def test_tie_strong_is_decided_by_ci90_not_ci95():
    # s = 0.24: CI90 half-width 0.387 s gives upper 0.693 < 0.7, CI95 half-width 0.468 s
    # gives upper 0.712 > 0.7; the rule is CI90 inside (-delta, delta)
    d = 0.6 + 0.24 * L / L.std(ddof=1)
    assert A._ci(d, 0.95)[1] < 0.7 < A._ci(d, 0.975)[1]
    assert B.go_no_go_holm(d, 1.0) == "TIE_STRONG"


def test_win_needs_ci95_lower_bound_positive_not_ci90():
    # mean 0.7 with CI90 lower > 0 but CI95 lower < 0: significance given, not a WIN
    d = 0.7 + 1.6 * L / L.std(ddof=1)
    assert A._ci(d, 0.95)[0] > 0 > A._ci(d, 0.975)[0]
    assert float(d.mean()) == pytest.approx(0.7, abs=1e-9)
    assert B.go_no_go_holm(d, 0.001) == "TIE_WEAK"


def test_significant_small_effect_is_tie_strong():
    d = 0.3 + 0.05 * L
    assert _p(d) < B.HOLM_FWER
    assert B.go_no_go_holm(d, _p(d)) == "TIE_STRONG"


def test_win_count_thirteen_is_win_and_twelve_is_tie_weak():
    d13 = np.r_[1.5 + 0.01 * np.arange(13), -0.1 - 0.01 * np.arange(7)]
    d12 = np.r_[1.5 + 0.01 * np.arange(12), -0.1 - 0.01 * np.arange(8)]
    assert int((d13 > 0).sum()) == 13 and int((d12 > 0).sum()) == 12
    assert B.go_no_go_holm(d13, _p(d13)) == "WIN"
    assert B.go_no_go_holm(d12, _p(d12)) == "TIE_WEAK"


def test_lose_mirror_needs_thirteen_negative_instances():
    d13 = -np.r_[1.5 + 0.01 * np.arange(13), -0.1 - 0.01 * np.arange(7)]
    d12 = -np.r_[1.5 + 0.01 * np.arange(12), -0.1 - 0.01 * np.arange(8)]
    assert B.go_no_go_holm(d13, _p(d13)) == "LOSE"
    assert B.go_no_go_holm(d12, _p(d12)) == "TIE_WEAK"


def test_holm_p_not_raw_p_decides_the_win():
    d = 2 + 0.1 * L
    assert B.go_no_go_holm(d, 0.06) == "TIE_WEAK"
    assert B.go_no_go_holm(d, _p(d)) == "WIN"


def test_holm_values_on_six_known_p_values():
    raw = [0.001, 0.01, 0.02, 0.04, 0.2, 0.5]
    assert A.holm(raw) == pytest.approx([0.006, 0.05, 0.08, 0.12, 0.4, 0.5], abs=1e-12)


W_DELTAS = 0.8 + 2.0 * L           # 14 positive instances, raw p in (0.05/6, 0.05)


def test_window_vector_is_win_alone_and_demoted_by_holm_in_six():
    p_raw = _p(W_DELTAS)
    assert int((W_DELTAS > 0).sum()) == 14
    assert 0.05 / 6 < p_raw < 0.05
    assert B.go_no_go_holm(W_DELTAS, p_raw) == "WIN"
    adj = A.holm([p_raw, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert adj[0] == pytest.approx(6 * p_raw, abs=1e-12)
    assert B.go_no_go_holm(W_DELTAS, adj[0]) == "TIE_WEAK"


# ---------------------------------------------------------------- family table

def test_family_table_known_answers_and_branches():
    deltas = _tie({"thr": 2 + 0.1 * L})
    df = _main_frame(deltas)
    fam = B.family_table(df[df.stage.isin(["b_full", "b_thr", "b_alpz", "b_rlsm",
                                           "b_dos", "b_dqn", "b_ppo"])],
                         ARMS6, n_routes=NR)
    row = fam.set_index("arm")
    assert row.loc["thr", "mean"] == pytest.approx(2.0, abs=1e-9)
    assert row.loc["thr", "median"] == pytest.approx(2.0, abs=1e-9)
    assert row.loc["thr", "n_better"] == 20
    assert row.loc["thr", "branch"] == "WIN"
    assert row.loc["alpz", "branch"] == "TIE_STRONG"
    assert list(fam["arm"]) == list(ARMS6)
    raw = list(fam["p_raw"])
    assert list(fam["p_holm"]) == pytest.approx(A.holm(raw), abs=1e-12)
    assert (row["holm_lo"] < row["ci95_lo"]).all()


def test_family_table_applies_holm_over_the_six_arms():
    df = _main_frame(_tie({"thr": W_DELTAS}))
    six = B.family_table(df, ARMS6, n_routes=NR).set_index("arm")
    assert six.loc["thr", "p_holm"] == pytest.approx(6 * six.loc["thr", "p_raw"], abs=1e-12)
    assert six.loc["thr", "branch"] == "TIE_WEAK"
    one = B.family_table(df, ["thr"], n_routes=NR).set_index("arm")
    assert one.loc["thr", "p_holm"] == pytest.approx(one.loc["thr", "p_raw"], abs=1e-12)
    assert one.loc["thr", "branch"] == "WIN"


def test_twelve_wins_are_not_a_win_in_the_family_table():
    d12 = np.r_[1.5 + 0.01 * np.arange(12), -0.1 - 0.01 * np.arange(8)]
    df = _main_frame(_tie({"thr": d12}))
    fam = B.family_table(df, ["thr"], n_routes=NR).set_index("arm")
    assert fam.loc["thr", "n_better"] == 12
    assert fam.loc["thr", "branch"] == "TIE_WEAK"


def test_family_table_refuses_instance_coverage_below_twenty():
    df = _main_frame(_tie({"thr": 2 + 0.1 * L}))
    short = df[~((df["stage"] == "b_thr") & (df["inst"] == "I19"))]
    with pytest.raises(B.AnalysisRefused):
        B.family_table(short, ["thr"], n_routes=NR)


def test_empty_family_gives_an_empty_table():
    df = _main_frame(_tie())
    out = B.family_table(df, [], n_routes=NR)
    assert out.empty and list(out.columns) == list(B.FAMILY_COLUMNS)


def test_outside_table_lists_each_arm_with_reason_and_no_branch():
    df = _main_frame(_tie({"dos": 2 + 0.1 * L}))
    out = B.outside_table(df, {"dos": "descriptive", "ppo": "dropped=ppo:B-PPO"},
                          n_routes=NR)
    assert list(out["arm"]) == ["dos", "ppo"]
    assert "branch" not in out.columns
    assert out.set_index("arm").loc["dos", "reason"] == "descriptive"
    assert out.set_index("arm").loc["dos", "mean"] == pytest.approx(2.0, abs=1e-9)
    assert bool(out.set_index("arm").loc["ppo", "ci90_inside_delta"]) is True


# ---------------------------------------------------------------- independence and re-eval

def test_independence_table_is_descriptive_with_ten_instances():
    d10 = 0.5 + 0.1 * np.linspace(-1, 1, 10)
    ind = pd.DataFrame(_ind_frame(d10))
    out = B.independence_table(ind)
    assert list(out["comparison"]) == ["FULL vs none", "FULL vs baton_eq", "FULL vs oracle",
                                       "FULL vs THR", "FULL vs ALP-Z"]
    assert (out["n"] == 10).all()
    thr = out.set_index("comparison").loc["FULL vs THR"]
    assert thr["mean"] == pytest.approx(0.5, abs=1e-9)
    assert "branch" not in out.columns


def test_re_eval_is_a_go_no_go_unadjusted_on_baton_eq():
    d = 2 + 0.1 * L
    df = _main_frame(_tie(), baseline=d)
    res = B.re_eval(df, n_routes=NR)
    direct = A.go_no_go(d)
    assert res["branch"] == direct["branch"] == "WIN"
    assert res["mean"] == pytest.approx(direct["mean"], abs=1e-9)
    assert res["p_greater"] == pytest.approx(direct["p_greater"], abs=1e-12)


# ---------------------------------------------------------------- membership and refusals

def _mixed_lists():
    return dict(family=["thr", "alpz", "rlsm", "dqn"], descriptive=["dos"],
                dropped=["ppo:B-bug"], references=["full", "ladder"], independence=[])


def test_build_report_covers_every_listed_arm_and_keeps_dropped_ones():
    df = _main_frame(_tie({"thr": 2 + 0.1 * L}))
    lists = _mixed_lists()
    rep = B.build_report(df, lists, n_routes=NR)
    assert list(rep["family"]["arm"]) == ["thr", "alpz", "rlsm", "dqn"]
    assert list(rep["outside"]["arm"]) == ["dos", "ppo"]
    assert rep["outside"].set_index("arm").loc["ppo", "reason"] == "dropped=ppo:B-bug"


def test_arm_not_in_any_list_is_refused():
    df = _main_frame(_tie())
    lists = _mixed_lists()
    lists["family"] = ["thr", "alpz", "rlsm"]
    with pytest.raises(B.AnalysisRefused, match="dqn"):
        B.build_report(df, lists, n_routes=NR)


def test_family_arm_with_no_rows_is_refused():
    df = _main_frame(_tie())
    lists = dict(LISTS6, family=list(ARMS6) + ["extra"])
    with pytest.raises(B.AnalysisRefused, match="no test_b rows"):
        B.build_report(df, lists, n_routes=NR)


def test_stage_writing_another_arms_policy_is_refused():
    df = _main_frame(_tie())
    df.loc[df["stage"] == "b_thr", "policy"] = "dos"
    with pytest.raises(B.AnalysisRefused, match="outside its arm"):
        B.build_report(df, LISTS6, n_routes=NR)


def test_listed_independence_stage_must_be_present():
    df = _main_frame(_tie())
    lists = dict(LISTS6, independence=["b_thr_ind"])
    with pytest.raises(B.AnalysisRefused, match="absent"):
        B.build_report(df, lists, n_routes=NR)


def test_arm_both_family_and_dropped_is_refused():
    df = _main_frame(_tie())
    lists = dict(LISTS6, dropped=["thr:B-bug"])
    with pytest.raises(B.AnalysisRefused):
        B.build_report(df, lists, n_routes=NR)


def test_load_accepts_a_clean_table_and_adds_stage(tmp_path):
    df = _main_frame(_tie())
    paths = _csv_dir(tmp_path, df)
    loaded = B.load_stage_tables(paths, family_sha256=SHA)
    assert set(loaded["stage"]) == set(df["stage"])
    assert len(loaded) == len(df)


def test_load_refuses_family_hash_mismatch(tmp_path):
    paths = _csv_dir(tmp_path, _main_frame(_tie()))
    with pytest.raises(B.AnalysisRefused, match="family_sha256"):
        B.load_stage_tables(paths, family_sha256="0" * 64)


def test_load_refuses_rows_not_from_test_b(tmp_path):
    df = _main_frame(_tie())
    df.loc[df["stage"] == "b_alpz", "eval_set"] = "dev"
    with pytest.raises(B.AnalysisRefused, match="test_b"):
        B.load_stage_tables(_csv_dir(tmp_path, df), family_sha256=SHA)


def test_load_refuses_wrong_day_count(tmp_path):
    df = _main_frame(_tie(), n_eval=TEST_B_N - 1)
    with pytest.raises(B.AnalysisRefused, match="day count"):
        B.load_stage_tables(_csv_dir(tmp_path, df), family_sha256=SHA)


def test_load_refuses_mixed_eval_commits(tmp_path):
    df = _main_frame(_tie())
    df.loc[df["stage"] == "b_dos", "eval_commit"] = "c" * 40
    with pytest.raises(B.AnalysisRefused, match="eval commits"):
        B.load_stage_tables(_csv_dir(tmp_path, df), family_sha256=SHA)


def test_load_refuses_fit_commit_not_frozen_for_the_stage(tmp_path):
    paths = _csv_dir(tmp_path, _main_frame(_tie()))
    frozen = {s: FIT for s in ["b_full", "b_thr"]}
    frozen["b_thr"] = "d" * 40
    with pytest.raises(B.AnalysisRefused, match="FROZEN"):
        B.load_stage_tables(paths, family_sha256=SHA, frozen=frozen)


def test_load_refuses_a_file_that_is_not_a_test_b_table(tmp_path):
    df = _main_frame(_tie())
    path = tmp_path / "b_thr.csv"
    df[df["stage"] == "b_thr"].drop(columns=["stage"]).to_csv(path, index=False)
    with pytest.raises(B.AnalysisRefused, match="test_b stage table"):
        B.load_stage_tables([path], family_sha256=SHA)


def test_load_refuses_a_stage_given_twice(tmp_path):
    paths = _csv_dir(tmp_path, _main_frame(_tie()))
    with pytest.raises(B.AnalysisRefused, match="twice"):
        B.load_stage_tables(paths + paths, family_sha256=SHA)


def test_cli_rejects_other_streams_at_argument_parsing():
    with pytest.raises(SystemExit):
        B.main(["--merged", "x.test_b.csv", "--eval", "dev", "--out", "y"])


# ---------------------------------------------------------------- analysis pin

def test_analysis_pin_accepts_pinned_or_last_patch():
    h1, h2 = "1" * 64, "2" * 64
    B.check_analysis_pin(h1, h1, [])
    B.check_analysis_pin(h2, h1, [h2])
    B.check_analysis_pin(h2, h1, [h1, h2])


def test_analysis_pin_refuses_unpatched_change_and_superseded_patch():
    h1, h2, h3 = "1" * 64, "2" * 64, "3" * 64
    with pytest.raises(B.AnalysisRefused):
        B.check_analysis_pin(h2, h1, [])
    with pytest.raises(B.AnalysisRefused):
        B.check_analysis_pin(h2, h1, [h2, h3])     # h2 is superseded by the last patch h3


def test_patch_lines_are_parsed_in_order():
    line = ("TEST_B ANALYSIS PATCH: analysis_sha256=" + "5" * 64
            + " date=2026-11-19 reason=fix a typo in the report")
    assert B.patch_hashes(["x", line]) == ["5" * 64]
    with pytest.raises(B.AnalysisRefused):
        B.patch_hashes(["TEST_B ANALYSIS PATCH: broken"])


# ---------------------------------------------------------------- outputs and compute

def test_outputs_state_all_arms_reported_and_never_call_write_summary(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("analysis_b must not call expa.analysis.write_summary")
    monkeypatch.setattr(A, "write_summary", boom)
    df = _main_frame(_tie({"thr": 2 + 0.1 * L}))
    rep = B.build_report(df, LISTS6, n_routes=NR)
    out = B.write_outputs(rep, tmp_path, family_line=FAMILY_LINE, eval_commit=EVAL)
    text = out.read_text(encoding="utf-8")
    assert "All family arms are reported below regardless of outcome" in text
    for arm in ARMS6:
        assert f"**{arm}**" in text
    assert "RE-EVAL" in text and "not a replication" in text
    assert "Independence world" in text
    assert (tmp_path / "family.csv").exists()
    assert len(pd.read_csv(tmp_path / "family.csv")) == 6


def test_claim_templates_never_say_equal():
    rows = pd.DataFrame([
        dict(arm="thr", mean=2.0, ci95_lo=1.0, ci95_hi=3.0, holm_lo=0.9, holm_hi=3.1,
             ci90_lo=1.2, ci90_hi=2.8, n_better=20, n=20, p_holm=0.001, branch="WIN"),
        dict(arm="dos", mean=0.1, ci95_lo=-0.1, ci95_hi=0.3, holm_lo=-0.2, holm_hi=0.4,
             ci90_lo=-0.05, ci90_hi=0.25, n_better=11, n=20, p_holm=0.001,
             branch="TIE_STRONG"),
        dict(arm="ppo", mean=0.1, ci95_lo=-0.9, ci95_hi=1.1, holm_lo=-1.0, holm_hi=1.2,
             ci90_lo=-0.8, ci90_hi=1.0, n_better=12, n=20, p_holm=0.9, branch="TIE_WEAK"),
    ])
    texts = [B.arm_claim(r) for _, r in rows.iterrows()]
    assert "lowers recourse cost by 2.00%" in texts[0]
    assert "equivalent within" in texts[1] and "significantly" in texts[1]
    assert "No difference" in texts[2]
    assert all("FULL = " not in t for t in texts)


def test_compute_table_reads_payloads_and_summarises(tmp_path):
    models = tmp_path / "b_thr"
    models.mkdir()
    for rep, secs in [(0, 10.0), (1, 30.0), (2, 20.0)]:
        payload = {"job": SimpleNamespace(arm="thr", rep=rep), "fit_seconds": secs,
                   "budget": {"path_stop_reads": 100 * (rep + 1), "device": "cpu",
                              "peak_rss_mb": 500.0 + rep, "n_params": 7}}
        with open(models / f"thr__I00__r{rep}__N100000__rep{rep}.pkl", "wb") as fh:
            pickle.dump(payload, fh)
    table = B.compute_table([models])
    assert len(table) == 3 and set(table["stage"]) == {"b_thr"}
    summ = B.compute_summary(table).set_index("arm")
    assert summ.loc["thr", "fit_median_s"] == pytest.approx(20.0)
    assert summ.loc["thr", "fit_max_s"] == pytest.approx(30.0)
    assert summ.loc["thr", "path_stop_reads_median"] == pytest.approx(200.0)
    assert summ.loc["thr", "devices"] == "cpu"

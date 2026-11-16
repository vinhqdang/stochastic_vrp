"""Pilot pass criteria P1-P6 and stop rules 2, 3, 5, 6, 7 (EXPERIMENT_A.md section 7.3
and section 10) on constructed tables. Each criterion has a PASS case and a FAIL case,
and the numbers are checked against hand computations in the comments."""
import numpy as np
import pandas as pd
import pytest

from transvrp.expa import pilot

ROUTES = [("I0", 0), ("I0", 1), ("I1", 0)]
BASE = {("I0", 0): 10.0, ("I0", 1): 12.0, ("I1", 0): 8.0}
REP_SIGN = [1.0, -1.0, 0.0]        # BATON replicate pattern: SD over reps = j * base


def _table(scalar=1.0, j=0.0, Ns=(1000, 10000)):
    """Pilot-like dev table. none = 1.2 b, oracle = 0.8 b, BATON-eq = b (1 + j s_rep),
    nn_full = BATON-eq, nn_scalar = scalar x BATON-eq (per row)."""
    rows = []
    for (inst, rid) in ROUTES:
        b = BASE[(inst, rid)]
        for N in Ns:
            for rep in range(3):
                base = b * (1 + j * REP_SIGN[rep])
                vals = {"none": 1.2 * b, "oracle": 0.8 * b, "baton_eq": base,
                        "nn_full": base, "nn_scalar": scalar * base}
                for pol, c in vals.items():
                    rows.append(dict(inst=inst, route_id=rid, m=8, stratum="short", N=N,
                                     rep=rep, policy=pol, mean_cost=c, handoff_rate=0.1,
                                     fail_rate=0.0, gap_closed=np.nan, n_eval=5000,
                                     eval_set="dev", fit_seconds=60.0, fit_commit="x",
                                     eval_commit="x"))
    return pd.DataFrame(rows)


def _set(df, pol, inst, rid, N, rep, value):
    m = ((df.policy == pol) & (df.inst == inst) & (df.route_id == rid)
         & (df.N == N) & (df.rep == rep))
    assert m.sum() == 1
    df.loc[m, "mean_cost"] = value
    return df


def _diag(rows):
    base = dict(inst="I0", route_id=0, N=10000, rep=0, arm="full", entry="stage", k=1,
                fold="A", action="warm", chosen_ckpt=0, steps=50, select_on="val",
                n_fit=100, n_val=20, n_applied=None, disagree_rate=None)
    out = []
    for r in rows:
        d = dict(base)
        d.update(r)
        out.append(d)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- P1, P2

def test_p1_reports_the_supplied_flag():
    assert pilot.p1_status(True)["status"] == "PASS"
    assert pilot.p1_status(False)["status"] == "FAIL"
    assert pilot.p1_status(None)["status"] == "NOT RUN"


def test_p2_bit_identical_rerun_passes_and_one_ulp_fails():
    df = _table()
    assert pilot.p2_status(df, df.copy())["status"] == "PASS"
    bad = df.copy()
    bad.loc[0, "mean_cost"] = np.nextafter(bad.loc[0, "mean_cost"], np.inf)
    res = pilot.p2_status(df, bad)
    assert res["status"] == "FAIL" and res["max_abs_diff"] > 0
    assert pilot.p2_status(df, None)["status"] == "NOT RUN"


# ---------------------------------------------------------------- P3 and P3'

def test_p3_passes_when_scalar_is_equal_to_baton():
    res = pilot.p3_status(_table(scalar=1.0, j=0.0))
    assert res["status"] == "PASS"
    assert res["ratio"] == pytest.approx(1.0)
    assert res["threshold"] == pytest.approx(1.02)       # no replicate noise: 2% floor


def test_p3_fails_when_scalar_is_5_percent_dearer_without_noise():
    res = pilot.p3_status(_table(scalar=1.05, j=0.0))
    assert res["status"] == "FAIL"
    assert res["ratio"] == pytest.approx(1.05)


def test_p3_tolerance_grows_with_replicate_noise():
    # SD over reps is j * base per route; relative pooled SD = j sqrt(mean b^2) / mean b
    j = 0.02
    b = np.array(list(BASE.values()))
    expected_sd_rel = j * np.sqrt(np.mean(b ** 2)) / np.mean(b)
    res = pilot.p3_status(_table(scalar=1.05, j=j))
    assert res["sd_rel_baton"] == pytest.approx(expected_sd_rel)
    assert res["tolerance"] == pytest.approx(3 * expected_sd_rel)   # 3 x SD > 2%
    assert res["status"] == "PASS"                    # 1.05 <= 1 + 3 x 0.0200 ~ 1.0614
    res_fail = pilot.p3_status(_table(scalar=1.07, j=j))
    assert res_fail["status"] == "FAIL"


def test_p3_prime_flags_a_scalar_win_over_one_percent_without_failing():
    res = pilot.p3_status(_table(scalar=0.97, j=0.0))
    assert res["status"] == "PASS"
    assert res["flag"] == "FLAG" and res["flag_id"] == "P3'"
    assert pilot.p3_status(_table(scalar=0.995, j=0.0))["flag"] == "PASS"


def test_p3_not_run_without_both_policies():
    df = _table()
    df = df[df.policy != "nn_scalar"]
    assert pilot.p3_status(df)["status"] == "NOT RUN"


# ---------------------------------------------------------------- P4

def test_p4_passes_on_a_consistent_table_and_counts_triples():
    res = pilot.p4_status(_table())
    assert res["status"] == "PASS"
    assert res["checked"] == 3 * 2 * 3 * 3       # routes x N x reps x (full, scalar, baton)


def test_p4_fails_when_a_policy_beats_the_oracle():
    df = _set(_table(), "oracle", "I0", 0, 1000, 0, 1e6)   # oracle far above every policy
    res = pilot.p4_status(df)
    assert res["status"] == "FAIL"
    assert res["lower_violations"] == 3                    # full, scalar, baton on that cell


def test_p4_fails_when_a_policy_exceeds_1_005_times_none():
    df = _set(_table(), "nn_full", "I1", 0, 10000, 1, 2 * 1.2 * 8.0)
    res = pilot.p4_status(df)
    assert res["status"] == "FAIL" and res["upper_violations"] == 1


# ---------------------------------------------------------------- P5, projection

def test_projection_hand_value_with_all_cells_at_one_minute():
    minutes = {(arm, N): 1.0 for arm, N, _ in pilot.SECTION_72_PLAN}
    res = pilot.projection_hours(minutes)
    # 930 + 372 + 186 + 24 = 1512 job-minutes; 1512 / 4 + 3 h = 381 h
    assert res["job_minutes"] == 1512
    assert res["hours"] == pytest.approx(381.0)
    assert pilot.platform_decision(res["hours"] / 24) .startswith("STOP")


def test_projection_reports_missing_cells():
    res = pilot.projection_hours({("full", 100): 1.0})
    assert res["hours"] is None and "scalar@1000000" in res["missing"]


@pytest.mark.parametrize("days, decision", [(3.0, "CPU"), (4.0, "CPU"),
                                             (5.0, "GPU"), (8.0, "STOP")])
def test_platform_rule_thresholds(days, decision):
    assert pilot.platform_decision(days).startswith(decision)


def test_p5_pass_needs_timing_and_recorded_seconds():
    df = _table()
    assert pilot.p5_status(df, None)["status"] == "FAIL"
    timing = dict(arm="full", n_rows=10 ** 6, total_seconds=600.0, m=17)
    res = pilot.p5_status(df, timing)
    assert res["status"] == "PASS"
    assert res["minutes_per_job"][("full", 10 ** 6)] == pytest.approx(10.0)
    df_bad = df.copy()
    df_bad.loc[0, "fit_seconds"] = np.nan
    assert pilot.p5_status(df_bad, timing)["status"] == "FAIL"


# ---------------------------------------------------------------- P6 and stop rule 7

def _p6_rows(chosen, N=10000, extra=()):
    rows = [dict(N=N, chosen_ckpt=c, rep=i) for i, c in enumerate(chosen)]
    rows += list(extra)
    return _diag(rows)


def test_p6_warm_share_below_half_passes():
    diag = _p6_rows([0, 0, 4, 4, 4])                           # 2/5 = 0.4
    res = pilot.p6_status(diag)
    assert res["status"] == "PASS" and res["warm_share"] == pytest.approx(0.4)


def test_p6_warm_share_of_half_or_more_fails():
    diag = _p6_rows([0, 0, 0, 4, 4])                           # 3/5 = 0.6
    assert pilot.p6_status(diag)["status"] == "FAIL"
    assert pilot.p6_status(_p6_rows([0, 4]))["status"] == "FAIL"   # exactly 1/2 is not < 1/2


def test_p6_ignores_carry_constant_and_other_sizes_and_reports_disagreement():
    extra = [dict(action="carry", chosen_ckpt=0), dict(action="constant", chosen_ckpt=0),
             dict(N=1000, chosen_ckpt=0),
             dict(entry="ls", action=None, disagree_rate=0.1),
             dict(entry="ls", action=None, disagree_rate=0.3)]
    diag = _p6_rows([4, 4, 4, 4, 0], extra=extra)
    res = pilot.p6_status(diag)
    assert res["n_stages"] == 5
    assert res["warm_share"] == pytest.approx(0.2)
    assert res["disagree_rate"] == pytest.approx(0.2)


def _rule7_diag(warm_counts):
    """One job per entry of warm_counts (number of warm-start stages out of 4)."""
    rows = []
    for j, w in enumerate(warm_counts):
        for s in range(4):
            rows.append(dict(route_id=j, chosen_ckpt=0 if s < w else 4))
    return _diag(rows)


def test_stop_rule_7_one_job_of_four_is_not_enough():
    # shares: 0.25 of jobs have > 50% warm stages: rule needs > 25%
    res = pilot.stop_rule_7(_rule7_diag([4, 2, 0, 1]))
    assert res["job_share"] == pytest.approx(0.25)
    assert res["triggered"] is False


def test_stop_rule_7_two_jobs_of_four_triggers():
    res = pilot.stop_rule_7(_rule7_diag([4, 3, 0, 1]))
    assert res["job_share"] == pytest.approx(0.5)
    assert res["triggered"] is True


def test_stop_rule_7_ignores_small_N():
    diag = _rule7_diag([4, 4, 4])
    diag["N"] = 1000
    assert pilot.stop_rule_7(diag)["triggered"] is None


# ---------------------------------------------------------------- stop rules 2, 3, 5, 6

def test_stop_rule_2_any_policy_below_oracle():
    assert pilot.stop_rule_2(_table())["triggered"] is False
    res = pilot.stop_rule_2(_set(_table(), "baton_eq", "I0", 1, 10000, 2, 1.0))
    assert res["triggered"] is True and res["violations"] == 1


def _s7_table(scalar_gain_pct):
    """20 instances, one route each, dp_big at 10^6 and nn_scalar at 10^5, three reps.
    Per-instance gain = scalar_gain_pct + 0.1 L_i (L = linspace(-1, 1, 20))."""
    L = np.linspace(-1, 1, 20)
    rows = []
    for i in range(20):
        inst = f"I{i:02d}"
        for rep in range(3):
            rows.append(dict(inst=inst, route_id=0, N=10 ** 6, rep=rep, policy="dp_big",
                             mean_cost=10.0))
            rows.append(dict(inst=inst, route_id=0, N=100000, rep=rep, policy="nn_scalar",
                             mean_cost=10.0 * (1 - (scalar_gain_pct + 0.1 * L[i]) / 100)))
    return pd.DataFrame(rows)


def test_stop_rule_3_not_applicable_without_the_big_rows():
    assert pilot.stop_rule_3(_table())["triggered"] is None


def test_stop_rule_3_triggers_when_scalar_is_significantly_better():
    res = pilot.stop_rule_3(_s7_table(1.0))
    assert res["classification"] == "better" and res["triggered"] is True


def test_stop_rule_3_does_not_trigger_when_scalar_is_worse():
    res = pilot.stop_rule_3(_s7_table(-1.0))
    assert res["classification"] == "worse" and res["triggered"] is False


def _elig(n_inst, n_routes):
    rows = []
    for r in range(n_routes):
        rows.append(dict(inst=f"I{r % n_inst:02d}", route_id=r, eligible=True))
    return pd.DataFrame(rows)


def test_stop_rule_5_thresholds_are_40_routes_and_15_instances():
    assert pilot.stop_rule_5(_elig(20, 40))["triggered"] is False
    assert pilot.stop_rule_5(_elig(20, 39))["triggered"] is True
    assert pilot.stop_rule_5(_elig(14, 40))["triggered"] is True
    assert pilot.stop_rule_5(_elig(15, 40))["triggered"] is False


def test_stop_rule_6_uses_the_seven_day_ceiling():
    assert pilot.stop_rule_6({"projection": {"hours": None}})["triggered"] is None
    assert pilot.stop_rule_6({"projection": {"hours": 6 * 24}})["triggered"] is False
    assert pilot.stop_rule_6({"projection": {"hours": 381.0}})["triggered"] is True


# ---------------------------------------------------------------- driver

def test_check_pilot_and_render_on_constructed_tables():
    df = _table()
    diag = _p6_rows([0, 4, 4, 4, 4])
    res = pilot.check_pilot(df, diag, tests_green=True, rerun=df.copy(),
                            timing=dict(arm="full", n_rows=10 ** 6, total_seconds=60.0, m=17))
    assert res["criteria"]["P1"]["status"] == "PASS"
    assert res["criteria"]["P3"]["status"] == "PASS"
    assert res["stop_rules"]["rule2"]["triggered"] is False
    assert res["criteria"]["P5"]["status"] == "PASS"
    assert res["ok"] is True                          # every criterion PASS
    text = pilot.render_markdown(res)
    assert "| P3 | PASS |" in text and "Stop rules" in text
    no_p1 = pilot.check_pilot(df, diag)
    assert no_p1["criteria"]["P1"]["status"] == "NOT RUN"
    assert no_p1["ok"] is False                       # a NOT RUN criterion is not a pass


def test_p4_upper_bound_is_inclusive_at_1_005_times_none():
    df = _set(_table(), "nn_full", "I1", 0, 10000, 1, 1.005 * 1.2 * 8.0)
    assert pilot.p4_status(df)["status"] == "PASS"          # equal to the bound: allowed
    df2 = _set(_table(), "nn_full", "I1", 0, 10000, 1, 1.006 * 1.2 * 8.0)
    assert pilot.p4_status(df2)["status"] == "FAIL"         # 0.6% above none: not allowed

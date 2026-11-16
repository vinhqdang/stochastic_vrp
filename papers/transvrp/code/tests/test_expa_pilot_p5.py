"""Pilot P5 projection from the cells the pilot measures (EXPERIMENT_A.md section 7.2 and
7.3, stop rule 6), the platform rule, the report sections (P5 table, timing headline,
section 4.2 diagnostic) and the monotonicity driver on a real tiny SCALAR fit.

The pilot-shaped tables measure exactly what the real pilot measures: FULL and SCALAR at
N = 10^3 and 10^4 on the four CON3-0 routes (m = 9, 12, 12, 17), FULL at 10^5 on route 3
only, and the 10^6 timing record. Fit minutes are built so that, after the section 7.2
scaling to the mean eligible route, the measured cells are 3.4, 7.8 and 27 minutes.
"""
import numpy as np
import pandas as pd
import pytest

from transvrp.expa import pilot, sweep
from transvrp.expa.train import StageConfig

ROUTE_M = {0: 9, 1: 12, 2: 12, 3: 17}
SCALAR_RATIO = 0.8                       # SCALAR fits take 0.8 x FULL in the synthetic pilot
FULL_MIN = {1000: 3.4, 10000: 7.8}       # per average eligible route, minutes
ROUTE3_1E5_MIN = 27.0                    # route 3 only, rep 0
STOP_MIN_17 = 4.0                        # stop updates at 10^6, per average route


def _sec(min_at_117, m):
    """Fit seconds of a route with m stops whose scaled cost is min_at_117 minutes."""
    return 60.0 * min_at_117 * pilot.route_work(m) / pilot.W_PLAN_MEAN


def _pilot_df(scale=1.0):
    rows = []

    def add(rid, m, rep, N, pol, fs):
        rows.append(dict(inst="CON3-0", route_id=rid, m=m, stratum="short", N=N, rep=rep,
                         policy=pol, mean_cost=10.0 + rid, fit_seconds=fs * scale))

    for rid, m in ROUTE_M.items():
        for rep in range(3):
            for N, minutes in FULL_MIN.items():
                add(rid, m, rep, N, "nn_full", _sec(minutes, m))
                add(rid, m, rep, N, "nn_scalar", _sec(SCALAR_RATIO * minutes, m))
    add(3, 17, 0, 100000, "nn_full", _sec(ROUTE3_1E5_MIN, 17))
    return pd.DataFrame(rows)


def _timing(scale=1.0):
    return dict(stage="pilot", inst="CON3-0", route_id=3, m=17, arm="full", rep=0,
                model_N=100000, n_rows=10 ** 6,
                generate_seconds=600.0 * scale, cache_seconds=30.0 * scale,
                stop_seconds=_sec(STOP_MIN_17, 17) * scale,
                total_seconds=(600.0 + 30.0 + _sec(STOP_MIN_17, 17)) * scale,
                process_peak_rss_mb=2048.0, traced_generate_peak_mb=900.0,
                traced_cache_stop_peak_mb=1500.0)


def _project(scale=1.0):
    return pilot.p5_status(_pilot_df(scale), _timing(scale))


# ---------------------------------------------------------------- projection from measured cells

def test_projection_is_a_number_from_the_five_measured_cells():
    res = _project()
    proj = res["projection"]
    assert res["status"] == "PASS"
    assert proj["missing"] == [] and proj["hours"] is not None
    assert np.isfinite(proj["hours"]) and proj["hours"] > 0
    assert set(proj["measured"]) == {"full@10000", "full@100000"}
    assert set(proj["inputs"]) == {"full@1000", "full@10000", "full@100000",
                                   "scalar@1000", "scalar@10000"}


def test_measured_cells_are_scaled_to_the_mean_eligible_route():
    df = _pilot_df()
    inputs = pilot._measured_plan_minutes(df)
    assert inputs[("full", 1000)] == pytest.approx(3.4)
    assert inputs[("scalar", 1000)] == pytest.approx(0.8 * 3.4)
    cells = pilot.plan_cells(df, _timing())
    assert cells[("full", 10000)]["minutes"] == pytest.approx(7.8)
    assert cells[("full", 100000)]["minutes"] == pytest.approx(27.0)
    assert cells[("full", 100000)]["source"] == "measured"


def test_extrapolated_cells_follow_the_stated_rule():
    cells = pilot.plan_cells(_pilot_df(), _timing())
    # FULL: least-squares line through the three measured FULL cells
    N = np.array([1000.0, 10000.0, 100000.0])
    y = np.array([3.4, 7.8, 27.0])
    slope, intercept = np.polyfit(N, y, 1)
    assert cells[("full", 2000)]["minutes"] == pytest.approx(intercept + slope * 2000)
    assert cells[("full", 2000)]["source"].startswith("model")
    assert cells[("full", 100)]["minutes"] == pytest.approx(intercept + slope * 100)
    # SCALAR: ratio 0.8 (measured at 10^3 and 10^4) times FULL
    assert cells[("scalar", 100000)]["minutes"] == pytest.approx(0.8 * 27.0)
    assert cells[("scalar", 2000)]["minutes"] == pytest.approx(0.8 * (intercept + slope * 2000))
    # MLP: the section 7.2 prior 0.2 x FULL
    assert cells[("mlp", 100000)]["minutes"] == pytest.approx(0.2 * 27.0)
    assert cells[("mlp", 100000)]["source"].startswith("0.2")
    # 10^6: FULL fit line at 10^6 plus the stop updates of the timing record (4 min
    # per average route, which the scaling leaves unchanged)
    assert cells[("full", 10 ** 6)]["minutes"] == pytest.approx(
        intercept + slope * 10 ** 6 + STOP_MIN_17, rel=1e-9)


# Section 7.2 plan written out here, so that the hand values below do not depend on
# pilot.py: (arm, N) -> route-replicate jobs. 186 = 62 routes x 3 replicates.
HAND_PLAN = {("full", 100): 186, ("full", 500): 186, ("full", 2000): 186,
             ("full", 10000): 186, ("full", 100000): 186,
             ("scalar", 2000): 186, ("scalar", 100000): 186,
             ("mlp", 100000): 186,
             ("full", 10 ** 6): 12, ("scalar", 10 ** 6): 12}


def _hand_hours(minutes):
    """Hours by hand: job minutes = sum of (minutes per job) x (jobs); divide by 60
    for hours, by 4 workers, then add 3 h of data generation."""
    job_minutes = sum(minutes[cell] * n for cell, n in HAND_PLAN.items())
    return job_minutes / 60.0 / 4 + 3.0


def test_projection_hours_convert_job_minutes_to_hours_before_the_workers_term():
    cells = pilot.plan_cells(_pilot_df(), _timing())
    minutes = {k: c["minutes"] for k, c in cells.items()}
    job_min = sum(minutes[cell] * n for cell, n in HAND_PLAN.items())
    proj = _project()["projection"]
    assert proj["job_minutes"] == pytest.approx(job_min)
    assert proj["hours"] == pytest.approx(job_min / 60.0 / 4 + 3.0)
    assert proj["hours"] == pytest.approx(_hand_hours(minutes))


# Real pilot (results/expA/pilot.csv and its timing JSON): minutes per job by cell, as
# reported. Hand value: 14,353.26 job-minutes / 60 / 4 + 3 = 62.81 h = 2.62 days.
REAL_MINUTES = {("full", 100): 2.86, ("full", 500): 2.92, ("full", 2000): 3.14,
                ("full", 10000): 5.42, ("full", 100000): 17.61,
                ("scalar", 2000): 3.09, ("scalar", 100000): 17.31,
                ("mlp", 100000): 3.52,
                ("full", 10 ** 6): 166.34, ("scalar", 10 ** 6): 163.78}


def test_real_pilot_minutes_give_about_63_hours_and_the_cpu_platform():
    job_min = sum(REAL_MINUTES[cell] * n for cell, n in HAND_PLAN.items())
    assert job_min == pytest.approx(14353.26)
    hand_hours = job_min / 60.0 / 4 + 3.0
    assert hand_hours == pytest.approx(62.8053, abs=1e-4)
    res = pilot.projection_hours(REAL_MINUTES)
    assert res["hours"] == pytest.approx(hand_hours, rel=0.01)
    assert res["job_minutes"] == pytest.approx(job_min)
    days = res["hours"] / 24.0
    assert days == pytest.approx(hand_hours / 24.0, rel=0.01)
    assert pilot.platform_decision(days) == "CPU"
    assert pilot.stop_rule_6({"projection": res})["triggered"] is False


def test_real_pilot_report_prints_hours_days_and_decision_with_their_units():
    res = pilot.projection_hours(REAL_MINUTES)
    cells = {f"{arm}@{N}": dict(minutes=v, source="measured")
             for (arm, N), v in REAL_MINUTES.items()}
    proj = dict(res, cells=cells, days=res["hours"] / 24.0)
    report = dict(criteria=dict(P3=dict(status="NOT RUN"),
                                P5=dict(status="PASS", projection=proj, decision="CPU")),
                  stop_rules={}, monotonicity=None, timing=None)
    text = pilot.render_markdown(report)
    assert "Projection: 62.8 h = 2.62 days -> CPU." in text
    assert "Hours = job minutes / 60 / 4 workers" in text


def test_timing_without_stop_seconds_falls_back_to_total_and_is_labelled():
    df = _pilot_df()
    timing = _timing()
    del timing["stop_seconds"]
    cells = pilot.plan_cells(df, timing)
    assert cells[("full", 10 ** 6)]["minutes"] is not None


def test_p5_is_not_computable_without_a_second_full_cell_or_the_timing_record():
    df = _pilot_df()
    # one measured FULL cell (10^5) is not enough for the line a + b N
    only_one = df[~((df.policy == "nn_full") & (df.N.isin([1000, 10000])))]
    no_fit = pilot.p5_status(only_one, _timing())
    assert no_fit["status"] == "FAIL" and no_fit["projection"]["hours"] is None
    assert pilot.stop_rule_6(no_fit)["triggered"] is None
    assert pilot.p5_status(df, None)["status"] == "FAIL"


# ---------------------------------------------------------------- platform rule and stop rule 6

def _scale_for(target_days):
    """Every minute of the projection is linear in the scale s, so by hand
    hours(s) = s (H0 - 3) + 3, where H0 is the hand value at scale 1 (``_hand_hours``)."""
    cells = pilot.plan_cells(_pilot_df(), _timing())
    h0 = _hand_hours({k: c["minutes"] for k, c in cells.items()})
    return (target_days * 24.0 - 3.0) / (h0 - 3.0)


def _project_at_days(target_days):
    """P5 at the scale that gives ``target_days`` by hand, checked against the hand
    hours of the same scaled cells before any platform decision is read off it."""
    s = _scale_for(target_days)
    res = _project(s)
    cells = pilot.plan_cells(_pilot_df(s), _timing(s))
    assert res["projection"]["hours"] == pytest.approx(
        _hand_hours({k: c["minutes"] for k, c in cells.items()}))
    assert res["projection"]["days"] == pytest.approx(target_days)
    return res


def test_platform_rule_fires_on_a_projection_above_four_days():
    res = _project_at_days(5.0)
    assert res["decision"].startswith("GPU")
    assert pilot.stop_rule_6(res)["triggered"] is False


def test_platform_rule_keeps_a_projection_of_three_days_on_cpu():
    res = _project_at_days(3.0)
    assert res["decision"] == "CPU"
    assert pilot.stop_rule_6(res)["triggered"] is False


def test_stop_rule_6_fires_above_seven_days_after_the_platform_rule():
    res = _project_at_days(8.0)
    assert res["decision"].startswith("STOP")
    assert pilot.stop_rule_6(res)["triggered"] is True


# ---------------------------------------------------------------- report

def _empty_diag():
    return pd.DataFrame(columns=["inst", "route_id", "N", "rep", "arm", "entry", "k", "fold",
                                 "action", "chosen_ckpt", "steps", "select_on", "n_fit",
                                 "n_val", "n_applied", "disagree_rate"])


def test_report_has_the_p5_table_the_timing_headline_and_the_monotonicity_diagnostic():
    mono = dict(stage="pilot", arm="scalar", N=10000, jobs=3, tol=0.01, n_curves=12,
                n_steps=400, n_steps_bad=7, curves_bad=2, frac_curves=2 / 12,
                frac_steps=7 / 400)
    res = pilot.check_pilot(_pilot_df(), _empty_diag(), timing=_timing(), mono=mono)
    text = pilot.render_markdown(res)
    assert "## P5 projection" in text and "| full@2000 |" in text and "model: a + b N" in text
    assert "Peak memory, headline (process RSS high-water mark" in text
    assert "2048 MB" in text and "tracemalloc" in text
    assert "Monotonicity diagnostic (section 4.2" in text and "16.7% of curves" in text


def test_report_without_monotonicity_or_timing_still_renders():
    res = pilot.check_pilot(_pilot_df(), _empty_diag())
    text = pilot.render_markdown(res)
    assert "## Monotonicity" not in text and "## Timing" not in text
    assert "## Stop rules" in text


# ---------------------------------------------------------------- monotonicity driver

@pytest.fixture(scope="module")
def tiny_scalar(tmp_path_factory):
    d = tmp_path_factory.mktemp("mono")
    ctx = sweep.Ctx(results_dir=d, stage="pilot", cfg=StageConfig(s_max=10), commit="m" * 40)
    job = sweep.Job("neural", sweep.TIMING_INST, sweep.TIMING_ROUTE, 100, 0, arm="scalar")
    assert sweep.run_fit_job(job, ctx) == "fit"
    return d


def test_monotonicity_driver_reads_real_scalar_models(tiny_scalar):
    mono = pilot.monotonicity_for_stage(tiny_scalar, "pilot", N=100, arm="scalar")
    assert mono is not None
    assert mono["jobs"] == 1 and mono["n_curves"] > 0 and mono["n_steps"] > 0
    assert 0.0 <= mono["frac_curves"] <= 1.0 and 0.0 <= mono["frac_steps"] <= 1.0
    assert mono["tol"] == 0.01


def test_monotonicity_driver_returns_none_without_models(tiny_scalar):
    assert pilot.monotonicity_for_stage(tiny_scalar, "pilot", N=5000, arm="scalar") is None

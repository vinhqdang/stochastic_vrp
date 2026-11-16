"""Filtered re-runs (``sweep --only``) and the P2 rerun check through the pilot CLI
(EXPERIMENT_A.md section 7.3, P2), on the smoke stage in tmp_path via subprocess.

The smoke stage's one route is the smallest-m eligible route, CON3-0 route 0 (m = 9),
so ``arm=full,inst=CON3-0,route=0,N=100,rep=0`` is the smoke version of the pilot
cell ``arm=full,inst=CON3-0,route=0,N=10000,rep=0``. The rerun goes into its own
results directory, so no fitted model or row file is reused from the main run.
"""
import ast
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from transvrp.expa import sweep

ONLY = "arm=full,inst=CON3-0,route=0,N=100,rep=0"
JOB_KEY = "full__CON3-0__r0__N100__rep0"


def _module(mod, *args):
    return subprocess.run([sys.executable, "-m", mod, *args], cwd=sweep.CODE_DIR,
                          capture_output=True, text=True, timeout=1800)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("rerun")
    main, rerun, chunks = base / "main", base / "rerun", base / "chunks"
    sw = "transvrp.expa.sweep"
    for args in (["fit", "--stage", "smoke", "--results-dir", str(main)],
                 ["eval", "--stage", "smoke", "--eval", "dev", "--results-dir", str(main)],
                 ["merge", "--stage", "smoke", "--results-dir", str(main)],
                 ["fit", "--stage", "smoke", "--only", ONLY, "--chunk-dir", str(chunks),
                  "--results-dir", str(rerun)],
                 ["eval", "--stage", "smoke", "--eval", "dev", "--only", ONLY,
                  "--results-dir", str(rerun)],
                 ["merge", "--stage", "smoke", "--only", ONLY, "--results-dir", str(rerun)]):
        done = _module(sw, *args)
        assert done.returncode == 0, done.stderr[-2000:]
    return dict(main=main, rerun=rerun, rerun_csv=rerun / "smoke.csv")


def test_only_filter_runs_exactly_one_job_and_merges_in_the_p2_layout(runs):
    models = sorted(p.stem for p in (runs["rerun"] / "models" / "smoke").glob("*.pkl"))
    assert models == [JOB_KEY]
    rows = sorted(p.name for p in (runs["rerun"] / "smoke" / "rows").glob("*.dev.csv"))
    assert rows == [f"{JOB_KEY}.dev.csv"]
    df = pd.read_csv(runs["rerun_csv"])
    assert list(df.columns) == sweep.ROW_COLUMNS
    assert len(df) == 1 and df["policy"].tolist() == ["nn_full"]
    assert (df[["inst", "route_id", "N", "rep"]].iloc[0].tolist()
            == ["CON3-0", 0, 100, 0])


def test_pilot_cli_reports_p2_pass_with_exact_rows_of_the_main_table(runs):
    done = _module("transvrp.expa.pilot", "--results-dir", str(runs["main"]), "--stage",
                   "smoke", "--rerun", str(runs["rerun_csv"]))
    assert done.returncode == 0, done.stderr[-2000:]
    statuses = ast.literal_eval(done.stdout.strip().splitlines()[-1])
    assert statuses["P2"] == "PASS"
    report = (runs["main"] / "smoke_report.md").read_text()
    assert "| P2 | PASS |" in report
    main = pd.read_csv(runs["main"] / "smoke.csv")
    rerun = pd.read_csv(runs["rerun_csv"])
    key = ["inst", "route_id", "N", "rep", "policy"]
    both = rerun.merge(main, on=key, suffixes=("_r", "_m"))
    assert len(both) == len(rerun) == 1
    for col in ("mean_cost", "handoff_rate", "fail_rate"):
        assert both[f"{col}_r"].iloc[0] == both[f"{col}_m"].iloc[0]


def test_pilot_cli_reports_p2_fail_on_a_one_ulp_change(runs, tmp_path):
    bad = pd.read_csv(runs["rerun_csv"])
    before = float(bad.loc[0, "mean_cost"])
    bad["mean_cost"] = bad["mean_cost"].astype(float)
    bad.loc[0, "mean_cost"] = np.nextafter(before, np.inf)
    assert bad.loc[0, "mean_cost"] != before           # the perturbation survived the cast
    bad_csv = tmp_path / "bad_rerun.csv"
    bad.to_csv(bad_csv, index=False)
    assert pd.read_csv(bad_csv).loc[0, "mean_cost"] == np.nextafter(before, np.inf)
    done = _module("transvrp.expa.pilot", "--results-dir", str(runs["main"]), "--stage",
                   "smoke", "--rerun", str(bad_csv))
    assert done.returncode == 0, done.stderr[-2000:]
    assert ast.literal_eval(done.stdout.strip().splitlines()[-1])["P2"] == "FAIL"


def test_filter_matching_no_planned_job_is_refused_before_any_fit(tmp_path):
    rc = sweep.main(["fit", "--stage", "smoke", "--only", "arm=full,N=999",
                     "--results-dir", str(tmp_path)])
    assert rc == 2
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("bad", ["bogus=1", "N=ten", "arm", "arm=full,,rep=0"])
def test_malformed_only_filters_are_refused(tmp_path, bad):
    rc = sweep.main(["merge", "--stage", "smoke", "--only", bad,
                     "--results-dir", str(tmp_path)])
    assert rc == 2


def test_parse_only_accepts_the_route_alias_and_types_the_values():
    assert sweep.parse_only("arm=full,inst=CON3-0,route=0,N=10000,rep=0") == dict(
        arm="full", inst="CON3-0", route_id=0, N=10000, rep=0)
    assert sweep.parse_only(None) is None and sweep.parse_only("") is None


@pytest.mark.parametrize("job", [sweep.Job("neural", "CON3-0", 0, 100, 0, arm="full"),
                                 sweep.Job("neural", "CON3-0", 3, 10 ** 5, 2, arm="boost"),
                                 sweep.Job("ladder", "CON3-0", 1, 1000, 1),
                                 sweep.Job("big", "CON3-0", 3, 10 ** 6, 0)])
def test_job_fields_round_trip_through_the_job_key(job):
    assert sweep.fields_from_key(job.key) == sweep.job_fields(job)

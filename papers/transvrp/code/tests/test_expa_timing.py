"""Pilot timing stage (sweep.run_timing) and the diagnostics table, on a tiny model.

The real pass is 10^6 paths with the 10^5 nets (EXPERIMENT_A.md section 7.3) and is
not run here. The test uses N = 100 nets and 2,000 paths, and checks the record, the
bit identity with and without the chunk cache, and the diagnostics table."""
import json

import numpy as np
import pytest

from transvrp.expa import pilot, sweep
from transvrp.expa.train import StageConfig


@pytest.fixture(scope="module")
def tiny_pilot(tmp_path_factory):
    d = tmp_path_factory.mktemp("timing")
    ctx = sweep.Ctx(results_dir=d, stage="pilot", cfg=StageConfig(s_max=10), commit="t" * 40)
    job = sweep.Job("neural", sweep.TIMING_INST, sweep.TIMING_ROUTE, 100, 0, arm="full")
    assert sweep.run_fit_job(job, ctx) == "fit"
    return d


def test_timing_record_and_bit_identity_with_chunk_cache(tiny_pilot, tmp_path):
    plain = sweep.run_timing(tiny_pilot, stage="pilot", n_rows=2000, model_N=100,
                             arm="full", rep=0)
    cached = sweep.run_timing(tiny_pilot, stage="pilot", n_rows=2000, model_N=100,
                              arm="full", rep=0, chunk_dir=tmp_path / "chunks")
    assert plain["mean_cost"] == cached["mean_cost"]          # bit identical generation
    assert plain["handoff_rate"] == cached["handoff_rate"]
    assert plain["chunk_cache"] is False and cached["chunk_cache"] is True
    for key in ("generate_seconds", "cache_seconds", "stop_seconds", "total_seconds",
                "traced_generate_peak_mb", "traced_cache_stop_peak_mb",
                "process_peak_rss_mb"):
        assert np.isfinite(plain[key]) and plain[key] >= 0
    assert plain["total_seconds"] > 0 and plain["traced_cache_stop_peak_mb"] > 0
    out = tiny_pilot / "pilot" / "timing_full_N2000.json"
    assert json.loads(out.read_text())["n_rows"] == 2000


def test_timing_needs_the_fitted_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        sweep.run_timing(tmp_path, stage="pilot", n_rows=100, model_N=100, arm="full")


def test_timing_cli_subcommand_runs(tiny_pilot):
    assert sweep.main(["timing", "--stage", "pilot", "--results-dir", str(tiny_pilot),
                       "--n-rows", "1000", "--model-n", "100"]) == 0
    assert (tiny_pilot / "pilot" / "timing_full_N1000.json").exists()


def test_diag_table_reads_stage_and_fold_entries(tiny_pilot):
    diag = pilot.diag_table(tiny_pilot, "pilot")
    assert set(diag["N"]) == {100}
    assert set(diag["arm"]) == {"full"}
    assert set(diag["entry"]) == {"stage", "ls"}
    st = diag[diag["entry"] == "stage"]
    assert set(st["action"]) <= {"warm", "cold", "carry", "constant"}
    assert st["chosen_ckpt"].dropna().ge(0).all()
    ls = diag[diag["entry"] == "ls"]
    assert ls["disagree_rate"].between(0.0, 1.0).all()
    assert {"A", "B", "val"} >= set(ls["fold"].dropna())


def test_p6_runs_on_real_pickles(tiny_pilot):
    """P6 on the real tiny model: the N = 100 cell has trained stages, so the
    criterion gives a verdict (PASS or FAIL), not NOT RUN."""
    diag = pilot.diag_table(tiny_pilot, "pilot")
    res = pilot.p6_status(diag, N=100)
    assert res["status"] in ("PASS", "FAIL")
    assert 0.0 <= res["warm_share"] <= 1.0 and res["n_stages"] > 0

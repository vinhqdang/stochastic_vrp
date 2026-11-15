"""M9: fit / eval / merge runner and the test-stream gate (EXPERIMENT_A.md section 9, M9).

The smoke stage (one route, N = 100, s_max = 10, dev stream only) runs once per
module through the CLI entry point, into pytest's tmp_path. No test evaluates
the test stream.
"""
import numpy as np
import pandas as pd
import pytest

from transvrp.expa import sweep
from transvrp.expa.eligibility import build_eligibility

SMOKE_POLICIES = {"none", "tuned_tau", "baton_eq", "baton_big", "dp_big",
                  "oracle", "nn_full", "nn_scalar", "nn_mlp"}
GOOD_HASH = "a" * 40
HEX_COMMIT = "b" * 40


@pytest.fixture(scope="module")
def smoke_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("smoke")
    assert sweep.main(["fit", "--stage", "smoke", "--results-dir", str(d)]) == 0
    assert sweep.main(["eval", "--stage", "smoke", "--eval", "dev",
                       "--results-dir", str(d)]) == 0
    assert sweep.main(["merge", "--stage", "smoke", "--results-dir", str(d)]) == 0
    return d


def test_smoke_yields_exactly_the_nine_policies(smoke_dir):
    df = pd.read_csv(smoke_dir / "smoke.csv")
    assert set(df["policy"]) == SMOKE_POLICIES
    assert list(df.columns) == sweep.ROW_COLUMNS
    assert df["eval_set"].unique().tolist() == ["dev"]
    assert df["n_eval"].unique().tolist() == [5000]
    assert df["fit_commit"].notna().all() and df["eval_commit"].notna().all()
    assert df.groupby(["inst", "route_id", "N", "rep", "policy"]).size().max() == 1


def test_smoke_rows_and_npz_are_written_per_job(smoke_dir):
    rows = sorted((smoke_dir / "smoke" / "rows").glob("*.dev.csv"))
    npz = sorted((smoke_dir / "smoke" / "rows").glob("*.dev.npz"))
    assert len(rows) == 5 and len(npz) == 5
    with np.load(npz[0]) as z:
        assert set(z.files) <= SMOKE_POLICIES
        assert all(z[k].shape == (5000,) for k in z.files)


def test_smoke_costs_respect_the_oracle(smoke_dir):
    df = pd.read_csv(smoke_dir / "smoke.csv")
    oracle = df.loc[df.policy == "oracle", "mean_cost"].iloc[0]
    assert (df["mean_cost"] >= oracle - 1e-12).all()


def test_eval_rerun_does_not_refit(smoke_dir):
    models = sorted((smoke_dir / "models" / "smoke").glob("*.pkl"))
    before = {p.name: p.stat().st_mtime_ns for p in models}
    rows_dir = smoke_dir / "smoke" / "rows"
    csvs = sorted(rows_dir.glob("*.dev.csv"))
    first = {c.name: c.read_bytes() for c in csvs}
    for c in csvs:
        c.unlink()
    assert sweep.main(["eval", "--stage", "smoke", "--eval", "dev",
                       "--results-dir", str(smoke_dir)]) == 0
    after = {p.name: p.stat().st_mtime_ns for p in models}
    assert before == after
    again = {c.name: c.read_bytes() for c in sorted(rows_dir.glob("*.dev.csv"))}
    assert again == first


def test_fit_is_resumable(smoke_dir):
    info = sweep.run_fit("smoke", build_eligibility(["CON3-0"]), smoke_dir, jobs=1)
    assert info == {"planned": 5, "fitted": 0, "skipped": 5}


def test_plan_of_main_stage_has_expected_cells():
    elig = build_eligibility(["CON3-0"])
    jobs = sweep.plan_stage("main", elig)
    kinds = {}
    for j in jobs:
        tag = (j.kind, j.arm, j.N)
        kinds[tag] = kinds.get(tag, 0) + 1
    # 4 eligible routes on CON3-0, 3 reps
    assert kinds[("neural", "full", 100000)] == 12
    assert kinds[("neural", "mlp", 100000)] == 12
    assert kinds[("neural", "scalar", 2000)] == 12
    assert kinds[("big", "", 1000000)] == 12
    assert kinds[("ladder", "", 100000)] == 12
    ladder_1e5 = [j for j in jobs if j.kind == "ladder" and j.N == 100000][0]
    assert "otr21" in ladder_1e5.labels


def test_pilot_and_big_stages_are_dev_only_or_subset():
    elig = build_eligibility(["CON3-0"])
    assert all(j.N in (1000, 10000) or (j.N == 100000 and j.route_id == 3)
               for j in sweep.plan_stage("pilot", elig) if j.kind == "neural")
    assert not sweep.stage_spec("pilot").test_allowed
    assert not sweep.stage_spec("smoke").test_allowed
    assert sweep.stage_spec("big").test_allowed


def test_eval_test_without_frozen_commit_exits_nonzero(tmp_path):
    rc = sweep.main(["eval", "--stage", "main", "--eval", "test",
                     "--results-dir", str(tmp_path)])
    assert rc != 0
    assert not (tmp_path / "models").exists()


def test_smoke_test_evaluation_is_always_refused(tmp_path):
    rc = sweep.main(["eval", "--stage", "smoke", "--eval", "test",
                     "--frozen-commit", GOOD_HASH, "--results-dir", str(tmp_path)])
    assert rc != 0


def _log(tmp_path, text):
    p = tmp_path / "RESEARCH_LOG.md"
    p.write_text(text)
    return p


def test_gate_refusals(tmp_path):
    log = _log(tmp_path, f"frozen {GOOD_HASH}\n")
    ok = dict(stage="main", eval_set="test", research_log=log, commit=GOOD_HASH)
    with pytest.raises(sweep.GateError):
        sweep.check_test_gate(frozen_commit=None, **ok)
    with pytest.raises(sweep.GateError):
        sweep.check_test_gate(frozen_commit="abc123", **ok)
    with pytest.raises(sweep.GateError):
        sweep.check_test_gate(frozen_commit=HEX_COMMIT, **{**ok, "research_log": log})
    with pytest.raises(sweep.GateError):
        sweep.check_test_gate(frozen_commit=GOOD_HASH,
                              **{**ok, "research_log": tmp_path / "missing.md"})
    with pytest.raises(sweep.GateError):
        sweep.check_test_gate(frozen_commit=GOOD_HASH, **{**ok, "commit": HEX_COMMIT})
    with pytest.raises(sweep.GateError):
        sweep.check_test_gate(frozen_commit=GOOD_HASH, **{**ok, "commit": "unknown"})
    with pytest.raises(sweep.GateError):
        sweep.check_test_gate(frozen_commit=GOOD_HASH,
                              **{**ok, "stage": "pilot"})


def test_gate_passes_only_with_hash_in_log_and_matching_commit(tmp_path):
    log = _log(tmp_path, f"frozen {GOOD_HASH}\n")
    sweep.check_test_gate("main", "test", GOOD_HASH, log, GOOD_HASH)
    sweep.check_test_gate("main", "dev", None, log, "unknown")   # dev is never gated


def test_current_commit_reads_repository_files():
    commit = sweep.current_commit()
    assert commit == "unknown" or len(commit) == 40


def test_gap_closed_rule():
    base = dict(inst="x", route_id=0, N=10, rep=0, eval_set="dev")
    rows = [dict(base, policy="none", mean_cost=10.0),
            dict(base, policy="oracle", mean_cost=6.0),
            dict(base, policy="pi", mean_cost=8.0)]
    out = sweep.add_gap_closed(pd.DataFrame(rows))
    gc = out.set_index("policy")["gap_closed"]
    assert gc["none"] == pytest.approx(0.0)
    assert gc["oracle"] == pytest.approx(1.0)
    assert gc["pi"] == pytest.approx(0.5)
    # C_none - C_oracle below 1% of C_none: undefined
    tight = pd.DataFrame([dict(base, policy="none", mean_cost=10.0),
                          dict(base, policy="oracle", mean_cost=9.95),
                          dict(base, policy="pi", mean_cost=9.9)])
    assert sweep.add_gap_closed(tight)["gap_closed"].isna().all()


def test_thread_pool_plumbing_with_two_workers():
    assert sweep._run_all(abs, [-1, -2, 3], 2) == [1, 2, 3]

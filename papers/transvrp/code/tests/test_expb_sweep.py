"""EXPERIMENT B sweep integration (EXPERIMENT_B.md 10 M21): stage definitions, B fit and eval
payloads, the independence world, the reuse and test_b rows, and the per-pass curves.
A's sweep tests stay in test_expa_sweep.py."""
import hashlib
import numpy as np
import pandas as pd
import pytest

from transvrp.expa import sweep
from transvrp.expa.eligibility import build_eligibility
from transvrp.expa.features import RouteCache, RouteStats
from transvrp.expa.paths import Paths, RouteSpec
from transvrp.expb import codecs, stages
from transvrp.expb.curves import fit_pass_curves
from transvrp.expb.dqn import DQNConfig
from transvrp.expb.gatecheck import library_versions
from transvrp.expb.ppo import PPOConfig
from transvrp.expb.registry import ARM_POLICY_B, B_ARMS
from transvrp.expb.thr import THRConfig, permute_columns

ELIG_DEV = sweep.CODE_DIR / "results" / "expA" / "eligibility_dev.csv"
DEV_STAGES = ("b_smoke", "b_pilot") + tuple(n for n in stages.b_stage_defs()
                                            if n.startswith("b_tune_"))

# Pre-edit snapshot (M21 requirement): plan_stage job keys of A's and boost's stages at
# HEAD 10e6d2f, before any sweep.py edit: (count, sha256 of the newline-joined keys).
PLAN_SNAPSHOT = {
    "main": (2604, "0df7b7a8fbea613af2e764891288ca55cca32a39ddf2fb54c84da37d765aa757"),
    "big": (36, "158c93c497c3494945e6a90a08301682f2a8e434b8df4ecaff00756de06d8b82"),
    "boost": (186, "9361dc74ff4013362f717ca14e748b9ade5c8722fb51a4cc85e948648c808471"),
    "smoke": (5, "b19d5491e6402de327d91a33a17c03abae517f4d3e8c2d3356e7ba6a40aeeaa2"),
    "pilot": (73, "b1ba756f504c837bd4a781c95d823a2d455226ebd272b9f99828398936cbfa8f"),
}


def _digest(keys) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


@pytest.fixture
def clean_tree(monkeypatch):
    """Fit refuses a dirty tree; the tests run while the code is being edited."""
    monkeypatch.setattr(sweep, "tree_is_clean", lambda: True)


def _toy_spec(m: int = 4) -> RouteSpec:
    return RouteSpec(inst="TOY", route_id=0, route=tuple(range(1, m + 1)), m=m, B=4.0,
                     H=np.full(m, 0.6), E=np.linspace(1.0, 0.5, m), E0=1.0)


def _toy_paths(N: int, m: int = 4, seed: int = 2) -> Paths:
    rng = np.random.default_rng(seed)
    return Paths(p=rng.uniform(0.5, 2.0, (N, m)), d=rng.uniform(0.5, 2.0, (N, m)))


# ---------------------------------------------------------------- stage names and registry

def test_unknown_stage_names_are_refused(tmp_path):
    with pytest.raises(ValueError):
        sweep.stage_spec("b_nope")
    with pytest.raises(ValueError):
        sweep.run_fit("b_nope", pd.DataFrame(), results_dir=tmp_path)
    with pytest.raises(ValueError):
        sweep.run_eval("b_nope", "dev", results_dir=tmp_path)


def test_cli_refuses_an_unknown_stage():
    with pytest.raises(SystemExit):
        sweep.main(["fit", "--stage", "b_nope"])


def test_every_b_stage_definition_is_registered_and_b_prefixed():
    defs = stages.b_stage_defs()
    assert set(defs) <= set(sweep.STAGE_SPECS)
    assert all(name.startswith("b_") for name in defs)
    for name in ("b_ladder_ind", "b_thr_ind", "b_alpz_ind", "b_full_ind",
                 "b_full_reuse", "b_ladder_reuse"):
        assert name in sweep.STAGE_SPECS


def test_dev_stages_are_never_test_allowed_and_test_b_stages_are_gated_to_test_b():
    for name in DEV_STAGES:
        assert sweep.stage_spec(name).test_allowed is False, name
    for name in stages.b_stage_defs():
        if name not in DEV_STAGES:
            spec = sweep.stage_spec(name)
            assert spec.test_stream == "test_b", name
            assert spec.test_allowed is True, name


def test_tune_stage_per_config_names_cover_the_grid():
    names = {f"b_tune_{a}_c{i}" for a in B_ARMS for i in range(len(stages.TUNE_GRID[a]))}
    assert names <= set(sweep.STAGE_SPECS)
    assert len(stages.TUNE_GRID["rlsm"]) == 6
    assert len(stages.TUNE_GRID["dos"]) == 2
    assert len(stages.TUNE_GRID["dqn"]) == 4
    assert len(stages.TUNE_GRID["ppo"]) == 4


def test_arm_policies_of_b_are_registered_with_policy_names():
    assert sweep.ARM_POLICY["dqn"] == "dqn"
    assert set(ARM_POLICY_B) == set(B_ARMS)
    assert sweep.ARM_POLICY["full"] == "nn_full"


# ---------------------------------------------------------------- A and boost unchanged

def test_a_and_boost_job_lists_are_unchanged_since_before_m21():
    elig = pd.read_csv(ELIG_DEV)
    for stage in ("main", "big", "boost"):
        keys = [j.key for j in sweep.plan_stage(stage, elig)]
        assert (len(keys), _digest(keys)) == PLAN_SNAPSHOT[stage], stage


def test_a_smoke_and_pilot_job_lists_are_unchanged_since_before_m21():
    for stage in ("smoke", "pilot"):
        elig = sweep.stage_eligibility(stage, None)
        keys = [j.key for j in sweep.plan_stage(stage, elig)]
        assert (len(keys), _digest(keys)) == PLAN_SNAPSHOT[stage], stage


def test_b_smoke_job_list_covers_ladder_full_and_every_b_arm():
    elig = sweep.stage_eligibility("b_smoke", None)
    jobs = sweep.plan_stage("b_smoke", elig)
    assert len(jobs) == 1 + 1 + len(B_ARMS)              # ladder, full, six B arms (1 route)
    assert {j.arm for j in jobs if j.kind == "neural"} == {"full", *B_ARMS}


# ---------------------------------------------------------------- independence world (3.8)

def test_independent_world_permutation_is_deterministic_and_keeps_marginals():
    paths = _toy_paths(200)
    spec = sweep.stage_spec("b_full_ind")
    a = sweep.stage_paths(spec, paths, "INST", 3, "dev")
    b = sweep.stage_paths(spec, paths, "INST", 3, "dev")
    assert np.array_equal(a.p, b.p) and np.array_equal(a.d, b.d)
    for j in range(paths.p.shape[1]):
        assert np.array_equal(np.sort(a.p[:, j]), np.sort(paths.p[:, j]))
    # p and d move together, so the increments g = p - d are the permuted increments
    assert np.array_equal(a.g, permute_columns(paths.g, "INST|3|dev"))
    other = sweep.stage_paths(spec, paths, "INST", 3, "test_b")
    assert not np.array_equal(a.p, other.p)


def test_world_is_identity_for_a_non_independent_stage():
    paths = _toy_paths(50)
    out = sweep.stage_paths(sweep.stage_spec("b_full"), paths, "INST", 3, "dev")
    assert out is paths


# ---------------------------------------------------------------- reuse, gate refusals

def test_reuse_stage_has_no_fit(tmp_path):
    # cut (i) is struck (revision 3.3): the stage refuses before anything else
    with pytest.raises(ValueError, match="disabled"):
        sweep.run_fit("b_full_reuse", pd.DataFrame(), results_dir=tmp_path)


def test_reuse_stage_reads_a_main_models_directory(tmp_path):
    ctx = sweep.Ctx(tmp_path / "expB", "b_full_reuse", sweep.StageConfig(), "x")
    assert ctx.models_dir == tmp_path / "expA" / "models" / "main"


def test_test_b_is_refused_on_dev_stages_and_on_the_wrong_stream(tmp_path):
    with pytest.raises(sweep.GateError):
        sweep.run_eval("b_smoke", "test_b", results_dir=tmp_path, frozen_commit="0" * 40)
    with pytest.raises(sweep.GateError):
        sweep.run_eval("b_smoke", "test", results_dir=tmp_path, frozen_commit="0" * 40)
    with pytest.raises(sweep.GateError):
        sweep.run_eval("b_full", "test", results_dir=tmp_path, frozen_commit="0" * 40)


def test_test_b_without_a_verified_gate_is_refused_before_any_data(tmp_path):
    with pytest.raises(sweep.GateError):
        sweep.run_eval("b_ladder", "test_b", results_dir=tmp_path, frozen_commit=None)


# ---------------------------------------------------------------- arm configuration lookup

def test_arm_config_lookup_builds_the_configuration_class():
    smoke = sweep.stage_spec("b_smoke")
    assert sweep.arm_config(smoke, "dqn").passes == 1
    assert sweep.arm_config(sweep.stage_spec("b_thr"), "thr") == THRConfig()
    with pytest.raises(ValueError, match="no configuration"):
        sweep.arm_config(sweep.stage_spec("b_rlsm"), "rlsm")


# ---------------------------------------------------------------- per-pass curves (6.3)

def test_pass_curves_give_one_predictor_set_per_pass_and_match_shorter_runs():
    spec, paths = _toy_spec(), _toy_paths(120)
    stats = RouteStats.fit(paths)
    cache = RouteCache(spec, paths, stats)
    cfg2 = DQNConfig(lr=1e-3, passes=2, n_env=16, buffer=5000, batch=64, learn_start=1,
                     target_every_tr=256, ckpt_per_pass=2, device="cpu")
    cfg1 = DQNConfig(lr=1e-3, passes=1, n_env=16, buffer=5000, batch=64, learn_start=1,
                     target_every_tr=256, ckpt_per_pass=2, device="cpu")
    curves, _, _, budget = fit_pass_curves("dqn", spec, paths, 120, 0, cfg2)
    single, _, _, _ = fit_pass_curves("dqn", spec, paths, 120, 0, cfg1)
    assert sorted(curves) == [1, 2]
    assert budget["passes"] == 2
    idx = np.arange(40)
    for k in range(1, spec.m):
        a = curves[1][k][0].predict(cache, k, idx)
        b = single[1][k][0].predict(cache, k, idx)
        assert np.array_equal(a, b)


def test_ppo_pass_curves_are_wired_too():
    spec, paths = _toy_spec(), _toy_paths(96)
    cfg = PPOConfig(passes=2, epochs=1, ckpt_per_pass=2)
    curves, _, _, _ = fit_pass_curves("ppo", spec, paths, 96, 0, cfg)
    assert sorted(curves) == [1, 2]
    assert all(len(curves[p]) == spec.m - 1 for p in curves)


def test_pass_curves_refuse_non_curve_arms():
    with pytest.raises(ValueError):
        fit_pass_curves("thr", _toy_spec(), _toy_paths(20), 20, 0, THRConfig())


# ---------------------------------------------------------------- test_b rows (5.3)

def test_test_b_rows_carry_family_model_and_library_columns(tmp_path):
    for col in ("family_sha256", "model_sha256", "python", "numpy", "scipy", "torch"):
        assert col in sweep.TEST_B_COLUMNS
    assert sweep.TEST_B_COLUMNS[:len(sweep.ROW_COLUMNS)] == sweep.ROW_COLUMNS
    rows_dir = tmp_path / "b_ladder" / "rows"
    rows_dir.mkdir(parents=True)
    row = {c: 0 for c in sweep.TEST_B_COLUMNS}
    row.update(inst="CON3-0", route_id=3, m=17, stratum="medium", N=100000, rep=0,
               policy="none", eval_set="test_b", fit_commit="a" * 40, eval_commit="b" * 40,
               family_sha256="c" * 64, model_sha256="d" * 64, gap_closed=np.nan)
    pd.DataFrame([row], columns=sweep.TEST_B_COLUMNS).to_csv(
        rows_dir / "ladder__CON3-0__r3__N100000__rep0.test_b.csv", index=False)
    df = sweep.merge_stage("b_ladder", "test_b", tmp_path)
    assert list(df.columns) == sweep.TEST_B_COLUMNS
    assert df.loc[0, "family_sha256"] == "c" * 64


# ---------------------------------------------------------------- b_smoke end to end

def test_b_smoke_fit_eval_merge_gives_exactly_the_ten_policies(tmp_path, clean_tree):
    elig = sweep.stage_eligibility("b_smoke", None)
    chunks = tmp_path / "chunks"
    info = sweep.run_fit("b_smoke", elig, results_dir=tmp_path, jobs=1, chunk_dir=chunks)
    assert info["fitted"] == info["planned"] == 1 + 1 + len(B_ARMS)
    models = sorted((tmp_path / "models" / "b_smoke").glob("*.pkl"))
    for path in models:
        payload = sweep.load_payload(path)
        assert payload["libraries"] == library_versions()
        assert payload["fit_tree_clean"] is True
        if path.name.split("__")[0] in B_ARMS:          # A arms (full) keep A's payload
            assert payload["budget"]["unique_paths"] == 2000
    before = {p.name: p.stat().st_mtime_ns for p in models}
    sweep.run_eval("b_smoke", "dev", results_dir=tmp_path, jobs=1)
    df = sweep.merge_stage("b_smoke", "dev", tmp_path)
    expected = {"none", "baton_eq", "oracle", "nn_full", *B_ARMS}
    assert set(df["policy"]) == expected
    # eval reruns never write models
    for row_file in (tmp_path / "b_smoke" / "rows").glob("*.dev.csv"):
        row_file.unlink()
    sweep.run_eval("b_smoke", "dev", results_dir=tmp_path, jobs=1)
    after = {p.name: p.stat().st_mtime_ns for p in models}
    assert after == before


def test_b_smoke_dqn_and_ppo_passes_and_budget_are_recorded(tmp_path, clean_tree):
    elig = sweep.stage_eligibility("b_smoke", None)
    jobs = [j for j in sweep.plan_stage("b_smoke", elig) if j.arm in ("dqn", "ppo")]
    ctx = sweep.Ctx(tmp_path, "b_smoke", sweep.StageConfig(), "x" * 40, tmp_path / "chunks")
    for job in jobs:
        sweep.run_fit_job(job, ctx)
    for job in jobs:
        payload = sweep.load_payload(ctx.models_dir / f"{job.key}.pkl")
        assert payload["budget"]["path_stop_reads"] > 0
        assert payload["budget"]["unique_paths"] == job.N


def test_b_fit_payload_budget_records_peak_rss(monkeypatch):
    monkeypatch.setattr(sweep, "process_peak_rss_mb", lambda: 1234.5)
    spec = _toy_spec()
    for stage in ("b_thr", "b_tune_ppo_c0"):
        sspec = sweep.stage_spec(stage)
        job = sweep.Job("neural", "TOY", 0, 96, 0, arm="thr" if stage == "b_thr" else "ppo")
        if sspec.pass_curve:
            sspec = sweep.StageSpec(reps=sspec.reps, neural=sspec.neural, routes=sspec.routes,
                                    test_stream=sspec.test_stream, pass_curve=1,
                                    arm_cfg=(("ppo", (("passes", 1), ("epochs", 1),
                                                      ("ckpt_per_pass", 2))),))
        payload = sweep._fit_b_payload(job, sspec, spec, _toy_paths(96))
        assert payload["budget"]["peak_rss_mb"] == 1234.5, stage


def test_tuning_stage_refuses_a_configuration_with_another_pass_count():
    # 6.3: the tuning curves are 3-point; a 2-pass configuration must not be fitted as one
    spec = _toy_spec()
    sspec = sweep.stage_spec("b_tune_ppo_c0")
    job = sweep.Job("neural", "TOY", 0, 96, 0, arm="ppo")
    wrong = sweep.StageSpec(reps=sspec.reps, neural=sspec.neural, routes=sspec.routes,
                            test_stream=sspec.test_stream, pass_curve=3,
                            arm_cfg=(("ppo", (("passes", 2),)),))
    with pytest.raises(ValueError, match="trains 3 passes"):
        sweep._fit_b_payload(job, wrong, spec, _toy_paths(96))


def test_curve_results_name_one_policy_per_pass():
    spec, paths = _toy_spec(), _toy_paths(96)
    cfg = PPOConfig(passes=2, epochs=1, ckpt_per_pass=2)
    curves, stats, _, _ = fit_pass_curves("ppo", spec, paths, 96, 0, cfg)
    payload = {"stats": stats, "pass_models": {p: codecs.pack_models(m, "ppo", spec.E0)
                                               for p, m in curves.items()}}
    job = sweep.Job("neural", "TOY", 0, 96, 0, arm="ppo")
    res = sweep._curve_results(job, payload, spec, paths)
    assert sorted(res) == ["ppo@p1", "ppo@p2"]
    assert all({"mean_cost", "handoff_rate", "fail_rate"} <= set(r) for r in res.values())


def test_dev_stream_refusal_names_the_declared_stream():
    with pytest.raises(sweep.GateError, match="evaluates on 'test_b', not on 'test'"):
        sweep.check_stream_for_stage("b_full", "test")
    with pytest.raises(sweep.GateError, match="dev-only"):
        sweep.check_stream_for_stage("b_smoke", "test_b")


def test_b_dev_stages_use_the_in_memory_con3_eligibility():
    elig = sweep.stage_eligibility("b_smoke", None)
    assert set(elig["inst"]) == {sweep.SMOKE_INSTANCE}
    assert len(sweep.stage_eligibility("b_pilot", None)) == len(
        sweep.stage_eligibility("pilot", None))


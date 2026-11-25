"""EXPERIMENT C stage definitions and sweep integration (EXPERIMENT_C.md 4, 4.3, 8, 9 MC4).

The stage table is checked against section 4, job counts against the eligibility table,
and five dev smoke runs (fit, eval, merge) go through the real sweep code with small N and
small grids. The smoke tests spy on ``route_pool`` and on the feature-set fitters, so a
wrong world or a dropped feature set fails them (mutation checks, see the report)."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from transvrp.expa import sweep
from transvrp.expb.stages import CHOSEN_ARM_CFG, SMOKE_CFG
from transvrp.expc import stages as cstages

ELIG = sweep.CODE_DIR / "results" / "expA" / "eligibility_dev.csv"
ROUTE_COUNT = {"eligible": 62, "subset": 12, "pilot": 4}

NAMES = (
    [f"c_{k}_{w}" for k in ("ladder", "thr", "cdp") for w in ("w0", "w3", "w6", "w9", "ws")]
    + [f"c_{k}_{fs}_{w}" for k in ("boost", "rlsm") for w in ("w0", "w3", "w9", "ws")
       for fs in ("F0", "F3")]
    + [f"c_{k}_{fs}_w6" for k in ("boost", "rlsm") for fs in ("F0", "F0n", "F1", "F2", "F3")]
    + ["c_cdptrue_w6", "c_dos_w6", "c_ppolong_w6", "c_full_w6", "c_ppob_w6"]
)
EVAL_ONLY = ("c_full_w6", "c_ppob_w6")
PILOT_EXCLUDED = EVAL_ONLY + ("c_ppolong_w6",)
DEFS = cstages.c_stage_defs()


@pytest.fixture
def clean_tree(monkeypatch):
    """Fit refuses a dirty tree; these tests run while the code is being edited."""
    monkeypatch.setattr(sweep, "tree_is_clean", lambda: True)


@pytest.fixture(scope="module")
def elig():
    return pd.read_csv(ELIG)


# ---------------------------------------------------------------- stage table (section 4)

def test_stage_names_match_the_spec_table():
    assert sorted(DEFS) == sorted(NAMES) and len(DEFS) == len(NAMES) == 46
    for name in NAMES:
        assert name in sweep.STAGE_SPECS, name


def test_world_suffixes_are_lowercase_and_ws_is_spelled_ws():
    assert all(not n.endswith(("W0", "W3", "W6", "W9", "WS")) for n in NAMES)
    assert {DEFS[n]["world"] for n in NAMES if n.endswith("_ws")} == {"WS"}


def test_each_world_is_the_one_named_by_its_stage():
    for name, d in DEFS.items():
        if "_w" in name.rsplit("_", 1)[-1] or name.endswith("_ws"):
            suffix = name.rsplit("_", 1)[-1]
            assert d["world"] == {"w0": "W0", "w3": "W3", "w6": "W6", "w9": "W9",
                                  "ws": "WS"}[suffix], name


def test_feature_sets_follow_section_1_3_and_4():
    assert DEFS["c_boost_F0_w6"]["feature_set"] == "F0"
    assert DEFS["c_boost_F0n_w6"]["feature_set"] == "F0n"
    assert DEFS["c_rlsm_F2_w6"]["feature_set"] == "F2"
    for w in ("w0", "w3", "w9", "ws"):
        assert f"c_boost_F1_{w}" not in DEFS and f"c_rlsm_F0n_{w}" not in DEFS
        assert f"c_boost_F0_{w}" in DEFS and f"c_boost_F3_{w}" in DEFS
    for name in ("c_ladder_w6", "c_thr_w6", "c_cdp_w6", "c_dos_w6", "c_full_w6"):
        assert DEFS[name]["feature_set"] is None


def test_routes_and_reps_follow_section_4():
    for name, d in DEFS.items():
        assert d["N"] == 100_000, name
        if name in ("c_ppolong_w6", "c_ppob_w6"):
            assert d["routes"] == "subset" and d["reps"] == (0,), name
        else:
            assert d["routes"] == "eligible" and d["reps"] == (0, 1), name


def test_eval_only_stages_reuse_b_models_and_no_other_stage_does():
    assert DEFS["c_full_w6"]["reuse_stage"] == "b_full"
    assert DEFS["c_ppob_w6"]["reuse_stage"] == "b_ppo"
    for name, d in DEFS.items():
        assert d["eval_only"] == (name in EVAL_ONLY), name
        if name not in EVAL_ONLY:
            assert d["reuse_stage"] is None, name
    for name in EVAL_ONLY:
        spec = sweep.stage_spec(name)
        assert spec.reuse_models == f"expB/models/{DEFS[name]['reuse_stage']}"
        assert spec.eval_only is True


def test_arm_configurations_are_b_chosen_and_the_stage_specific_overrides():
    assert CHOSEN_ARM_CFG["rlsm"] == (("K", 100), ("ridge", 1e-6))
    assert dict(DEFS["c_rlsm_F0_w9"]["arm_cfg"][0][1]) == {"K": 100, "ridge": 1e-6}
    assert dict(DEFS["c_cdptrue_w6"]["arm_cfg"][0][1]) == {"rho_override": 0.6}
    assert dict(DEFS["c_cdp_w3"]["arm_cfg"][0][1]) == {}
    ppo = dict(DEFS["c_ppolong_w6"]["arm_cfg"][0][1])
    assert ppo["passes"] == 9
    assert ppo["lr"] == dict(CHOSEN_ARM_CFG["ppo"])["lr"]
    assert ppo["gamma"] == dict(CHOSEN_ARM_CFG["ppo"])["gamma"]


def test_dos_config_comes_only_from_the_committed_record():
    rec = sweep.CODE_DIR / "results" / "expC" / "dos_choice.json"
    assert rec.exists()
    assert DEFS["c_dos_w6"]["arm_cfg"] == (("dos", (("lr", 1e-3),)),)
    empty = cstages.c_stage_defs(dos_pairs=())
    assert empty["c_dos_w6"]["arm_cfg"] == ()
    spec = dataclasses.replace(sweep.stage_spec("c_dos_w6"), arm_cfg=())
    with pytest.raises(ValueError, match="dos_choice"):
        sweep.arm_config(spec, "dos")


def test_pilot_stages_are_the_fitted_stages_on_con3_0_rep_0():
    pilot = cstages.c_pilot_defs(DEFS)
    assert len(pilot) == len(NAMES) - len(PILOT_EXCLUDED)
    for name, d in pilot.items():
        assert name.startswith("c_pilot_")
        assert d["routes"] == "pilot" and d["reps"] == (0,) and d["dev_only"] is True
        assert "c_" + name[len("c_pilot_"):] in DEFS
        assert name not in {"c_pilot_" + n[2:] for n in PILOT_EXCLUDED}
        assert sweep.stage_spec(name).test_allowed is False
    for name in pilot:
        assert sweep.stage_spec(name).family == "C"


def test_stage_kwargs_name_only_stage_spec_fields():
    fields = {f.name for f in dataclasses.fields(sweep.StageSpec)}
    for d in DEFS.values():
        assert set(cstages.to_stage_kwargs(d)) <= fields


# ---------------------------------------------------------------- job counts (planning)

@pytest.mark.parametrize("name", [n for n in NAMES if n not in EVAL_ONLY])
def test_every_fitted_stage_plans_routes_times_reps_jobs(name, elig):
    spec = sweep.stage_spec(name)
    routes = ROUTE_COUNT[spec.routes]
    jobs = sweep.plan_stage(name, elig)
    assert len(jobs) == routes * len(spec.reps), name
    assert len({j.key for j in jobs}) == len(jobs), name


@pytest.mark.parametrize("name", [f"c_pilot_{n[2:]}" for n in NAMES
                                  if n not in PILOT_EXCLUDED])
def test_every_pilot_stage_plans_the_con3_0_routes(name, elig):
    jobs = sweep.plan_stage(name, elig)
    assert len(jobs) == 4 and {j.inst for j in jobs} == {sweep.SMOKE_INSTANCE}


def test_eligibility_counts_match_the_spec(elig):
    assert int(elig["eligible"].astype(bool).sum()) == 62
    assert int(elig["in_1e6_subset"].astype(bool).sum()) == 12


def test_default_results_dir_of_c_stages_is_expc():
    assert sweep.default_results_dir("c_ladder_w0") == sweep.DEFAULT_RESULTS_C
    assert sweep.default_results_dir("c_pilot_ladder_w6") == sweep.DEFAULT_RESULTS_C
    assert sweep.DEFAULT_RESULTS_C.name == "expC"


def test_row_columns_add_world_and_feature_set_only_to_c_stages():
    assert sweep.row_columns("c_boost_F1_w6", "dev")[-2:] == ["world", "feature_set"]
    assert sweep.row_columns("c_ladder_w0", "test_c")[-2:] == ["world", "feature_set"]
    assert sweep.row_columns("c_ladder_w0", "test_c")[:len(sweep.TEST_B_COLUMNS)] == \
        sweep.TEST_B_COLUMNS
    assert sweep.row_columns("b_ladder", "dev") == sweep.ROW_COLUMNS


# ---------------------------------------------------------------- smoke fit, eval, merge

def _smoke_stage(tmp_path, monkeypatch, base, *, arm_cfg=None, ladder=False):
    """Registers a dev smoke copy of a C stage (one CON3-0 route, N = 2000, rep 0)."""
    src = sweep.stage_spec(base)
    name = "c_smoke_" + base[len("c_"):]
    neural = tuple((a, 2000) for a, _ in src.neural)
    spec = dataclasses.replace(
        src, reps=(0,), routes="smoke", neural=neural,
        ladder_N=(2000,) if ladder else (), test_allowed=False,
        arm_cfg=src.arm_cfg if arm_cfg is None else arm_cfg)
    monkeypatch.setitem(sweep.STAGE_SPECS, name, spec)
    return name, spec


def _route_pool_spy(monkeypatch):
    calls = []
    real = sweep.route_pool

    def spy(inst, purpose, rep=0, N=None, **kw):
        calls.append((purpose, kw.get("world", "default")))
        return real(inst, purpose, rep, N, **kw)

    monkeypatch.setattr(sweep, "route_pool", spy)
    return calls


def _assert_worlds(calls, world):
    """Every train and dev pool requested by the stage (fit and eval-side rebuilds) is in
    ``world``; a fit on the wrong world cannot hide behind a correct eval-side request."""
    assert {w for purpose, w in calls if purpose == "train"} == {world}
    assert {w for purpose, w in calls if purpose == "dev"} == {world}


def _fit_eval_merge(tmp_path, name):
    elig = sweep.stage_eligibility("b_smoke", None)
    results = tmp_path / "expC"
    info = sweep.run_fit(name, elig, results_dir=results, jobs=1,
                         chunk_dir=results / "chunks")
    assert info["fitted"] == info["planned"] == 1
    sweep.run_eval(name, "dev", results_dir=results, jobs=1)
    return sweep.merge_stage(name, "dev", results), results


def _assert_rows(df, world, feature_set, policies):
    assert set(df["world"]) == {world}
    assert set(df["feature_set"]) == {feature_set}
    assert set(df["policy"]) == set(policies)
    assert set(df["eval_set"]) == {"dev"}


def test_smoke_c_ladder_w0_fits_evaluates_and_merges_in_world_w0(tmp_path, monkeypatch,
                                                                clean_tree):
    calls = _route_pool_spy(monkeypatch)
    name, _ = _smoke_stage(tmp_path, monkeypatch, "c_ladder_w0", ladder=True)
    df, _ = _fit_eval_merge(tmp_path, name)
    _assert_rows(df, "W0", "none", {"none", "baton_eq", "oracle"})
    _assert_worlds(calls, "W0")


def test_smoke_c_boost_f1_w6_uses_the_feature_set_and_world(tmp_path, monkeypatch, clean_tree):
    seen = []
    real = cstages.fit_boost_set

    def spy(spec, paths, N, rep, set_name):
        seen.append(set_name)
        return real(spec, paths, N, rep, set_name)

    monkeypatch.setattr(cstages, "fit_boost_set", spy)
    calls = _route_pool_spy(monkeypatch)
    name, _ = _smoke_stage(tmp_path, monkeypatch, "c_boost_F1_w6")
    df, _ = _fit_eval_merge(tmp_path, name)
    assert seen == ["F1"]
    _assert_rows(df, "W6", "F1", {"boost"})
    _assert_worlds(calls, "W6")


def test_smoke_c_rlsm_f0_w9_uses_the_set_fitter_and_world(tmp_path, monkeypatch, clean_tree):
    seen = []
    real = cstages.fit_rlsm_set

    def spy(spec, paths, N, rep, cfg, set_name):
        seen.append((set_name, cfg.K, cfg.ridge))
        return real(spec, paths, N, rep, cfg, set_name)

    monkeypatch.setattr(cstages, "fit_rlsm_set", spy)
    calls = _route_pool_spy(monkeypatch)
    name, _ = _smoke_stage(tmp_path, monkeypatch, "c_rlsm_F0_w9")
    df, _ = _fit_eval_merge(tmp_path, name)
    assert seen == [("F0", 100, 1e-6)]
    _assert_rows(df, "W9", "F0", {"rlsm"})
    _assert_worlds(calls, "W9")


def test_smoke_c_cdp_w3_with_a_small_grid_fits_and_rebuilds_tables(tmp_path, monkeypatch,
                                                                  clean_tree):
    small = (("cdp", (("n_nodes", 5), ("n_eps", 16), ("n_grid", 51))),)
    calls = _route_pool_spy(monkeypatch)
    name, _ = _smoke_stage(tmp_path, monkeypatch, "c_cdp_w3", arm_cfg=small)
    df, results = _fit_eval_merge(tmp_path, name)
    _assert_rows(df, "W3", "none", {"cdp"})
    _assert_worlds(calls, "W3")
    payload = sweep.load_payload(next((results / "models" / name).glob("*.pkl")))
    assert payload["models"]["params"]["n_nodes"] == 5
    assert payload["models"]["params"]["n_grid"] == 51
    assert np.isfinite(df["mean_cost"]).all()


def test_smoke_c_dos_w6_fits_with_the_smoke_schedule(tmp_path, monkeypatch, clean_tree):
    name, _ = _smoke_stage(tmp_path, monkeypatch, "c_dos_w6",
                           arm_cfg=(("dos", SMOKE_CFG["dos"]),))
    df, _ = _fit_eval_merge(tmp_path, name)
    _assert_rows(df, "W6", "none", {"dos"})


def test_c_merged_rows_carry_world_and_feature_set_columns(tmp_path, monkeypatch, clean_tree):
    name, _ = _smoke_stage(tmp_path, monkeypatch, "c_boost_F3_w9")
    df, results = _fit_eval_merge(tmp_path, name)
    assert list(df.columns) == sweep.row_columns(name, "dev")
    assert (results / f"{name}.csv").exists()


# ---------------------------------------------------------------- eval-only stages (4.3)

def test_eval_only_stage_refuses_fit(tmp_path, elig):
    with pytest.raises(ValueError, match="has no fit"):
        sweep.run_fit("c_full_w6", elig, results_dir=tmp_path / "expC", jobs=1)
    with pytest.raises(ValueError, match="has no fit"):
        sweep.run_fit("c_ppob_w6", elig, results_dir=tmp_path / "expC", jobs=1)


def _reuse_tree(tmp_path, elig, stage, tamper=True):
    """A fake B tree: the planned model files of ``stage`` (contents differ from B's), with a
    manifest that lists them under B's fit commit and their real names."""
    planned = sorted(sweep.plan_stage(stage, elig), key=lambda j: j.key)
    expc = tmp_path / "expC"
    expa = tmp_path / "expA"
    expb = tmp_path / "expB"
    expa.mkdir(parents=True, exist_ok=True)
    (expa / "eligibility_dev.csv").write_bytes(ELIG.read_bytes())
    models = expb / "models" / sweep.stage_spec(stage).reuse_models.rsplit("/", 1)[-1]
    models.mkdir(parents=True)
    lines = []
    for job in planned:
        path = models / f"{job.key}.pkl"
        path.write_bytes(b"not the B model " + job.key.encode())
        lines.append(f"{'0' * 64} {path.name} {cstages.REUSE_FIT_COMMIT} True\n")
    manifest = expb / sweep.stage_spec(stage).reuse_models.rsplit("/", 1)[-1] / "models.sha256"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("".join(lines), encoding="utf-8")
    return expc, planned


def test_eval_only_stage_refuses_a_model_whose_hash_does_not_match_the_manifest(tmp_path, elig,
                                                                               monkeypatch):
    expc, planned = _reuse_tree(tmp_path, elig, "c_full_w6")
    monkeypatch.setattr(sweep, "current_commit", lambda: "0" * 40)
    with pytest.raises(ValueError, match="does not match B's manifest"):
        sweep.run_eval("c_full_w6", "dev", results_dir=expc, jobs=1)
    assert not list((expc / "c_full_w6" / "rows").glob("*.csv"))   # nothing evaluated


def test_eval_only_stage_refuses_a_model_missing_from_the_manifest(tmp_path, elig, monkeypatch):
    expc, planned = _reuse_tree(tmp_path, elig, "c_full_w6")
    manifest = tmp_path / "expB" / "b_full" / "models.sha256"
    manifest.write_text("", encoding="utf-8")
    monkeypatch.setattr(sweep, "current_commit", lambda: "0" * 40)
    with pytest.raises(ValueError, match="not in B's manifest"):
        sweep.run_eval("c_full_w6", "dev", results_dir=expc, jobs=1)


def test_reused_models_must_be_b_frozen_fit_commit_on_a_clean_tree(tmp_path, elig):
    expc, planned = _reuse_tree(tmp_path, elig, "c_full_w6")
    manifest = tmp_path / "expB" / "b_full" / "models.sha256"
    manifest.write_text("".join(
        f"{'0' * 64} {j.key}.pkl {'1' * 40} True\n" for j in planned), encoding="utf-8")
    model_files = [expc.parent / "expB" / "models" / "b_full" / f"{j.key}.pkl" for j in planned]
    with pytest.raises(ValueError, match="fit commit"):
        cstages.check_reuse_models(sweep.stage_spec("c_full_w6"), expc, model_files)


def test_reuse_test_c_needs_the_b_commit_and_a_verified_stage(tmp_path):
    with pytest.raises(ValueError, match="frozen commit"):
        cstages.check_reuse_gate("c_full_w6", "test_c", "0" * 40)
    try:
        from transvrp.expc import gates
    except ImportError:
        gates = None
    if getattr(gates, "verify_reuse_stage", None) is None:
        with pytest.raises(NotImplementedError, match="verify_reuse_stage"):
            cstages.check_reuse_gate("c_full_w6", "test_c", cstages.REUSE_FIT_COMMIT)


def test_reused_b_manifest_matches_the_b_fit_commit_and_real_files(elig):
    """The committed B manifest for the reused stages lists the fit commit bf3edab on clean
    trees (the value 4.3 names), and one real model file hashes to its entry."""
    import hashlib
    manifest = sweep.CODE_DIR / "results" / "expB" / "b_full" / "models.sha256"
    lines = [ln.split() for ln in manifest.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 124
    assert {ln[2] for ln in lines} == {cstages.REUSE_FIT_COMMIT}
    assert {ln[3] for ln in lines} == {"True"}
    sha, name = lines[0][0], lines[0][1]
    path = sweep.CODE_DIR / "results" / "expB" / "models" / "b_full" / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


def test_row_file_names_match_every_glob_that_reads_them():
    import fnmatch
    job = sweep.Job(kind="neural", inst="CON3-0", route_id=3, N=100000, rep=1, arm="cdp")
    npz = sweep.rows_file_name(job.key, "test_c", "npz")
    assert fnmatch.fnmatch(npz, sweep.day_file_glob("CON3-0", 3, 100000, 1, "test_c"))
    assert fnmatch.fnmatch(npz, sweep.day_file_glob("CON3-0", 3, 100000, None, "test_c"))
    assert not fnmatch.fnmatch(npz, sweep.day_file_glob("CON3-0", 13, 100000, 1, "test_c"))
    assert not fnmatch.fnmatch(npz, sweep.day_file_glob("CON3-0", 3, 100000, 0, "test_c"))
    assert not fnmatch.fnmatch(npz, sweep.day_file_glob("CON3-0", 3, 100000, 1, "dev"))
    csv = sweep.rows_file_name(job.key, "dev", "csv")
    assert csv.endswith(sweep.rows_suffix("dev", "csv"))
    assert sweep.rows_suffix("dev", "csv") == ".dev.csv"
    assert npz == f"{job.key}.test_c.npz"

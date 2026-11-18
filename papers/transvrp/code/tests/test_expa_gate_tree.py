"""Gate hardening (EXPERIMENT_A.md section 5): the clean-tree check against a real
temporary git repository, the GateToken that private loaders require for test-purpose
data, and the refusal of a model fitted on a dirty tree. Nothing here reads the test
stream: every refusal happens before any data is generated."""
import pickle
import shutil
import subprocess
import subprocess as sp

import pytest

from transvrp.expa import paths, sweep

GOOD_HASH = "a" * 40
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=test",
                    "-c", "user.email=test@example.invalid", *args],
                   check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A committed repository whose code directory holds one tracked module and one
    tracked data file. ``code`` plays the role of papers/transvrp/code."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    r = tmp_path / "repo"
    (r / "code" / "transvrp").mkdir(parents=True)
    _git(r, "init", "-q")
    (r / "code" / "transvrp" / "mod.py").write_text("x = 1\n")
    (r / "code" / "data.txt").write_text("a\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "init")
    return r


@needs_git
def test_clean_repository_passes(repo):
    assert paths.tree_is_clean(repo / "code") is True


@needs_git
def test_untracked_data_files_are_allowed(repo):
    # results and cache files are untracked but are not source code
    (repo / "code" / "results").mkdir()
    (repo / "code" / "results" / "model.pkl").write_bytes(b"\x00")
    (repo / "code" / "notes.txt").write_text("hand-written notes\n")
    assert paths.tree_is_clean(repo / "code") is True


@needs_git
def test_tracked_edit_is_dirty(repo):
    (repo / "code" / "transvrp" / "mod.py").write_text("x = 2\n")
    assert paths.tree_is_clean(repo / "code") is False


@needs_git
def test_staged_edit_is_dirty(repo):
    (repo / "code" / "data.txt").write_text("b\n")
    _git(repo, "add", "code/data.txt")
    assert paths.tree_is_clean(repo / "code") is False


@needs_git
def test_untracked_py_file_is_dirty(repo):
    (repo / "code" / "new_module.py").write_text("y = 3\n")
    assert paths.tree_is_clean(repo / "code") is False


@needs_git
def test_untracked_py_file_in_a_new_subdirectory_is_dirty(repo):
    # porcelain without --untracked-files=all would list only "?? code/newpkg/"
    (repo / "code" / "newpkg").mkdir()
    (repo / "code" / "newpkg" / "hidden.py").write_text("z = 4\n")
    assert paths.tree_is_clean(repo / "code") is False


def test_unparseable_git_output_fails_closed(monkeypatch, tmp_path):
    fake = sp.CompletedProcess(args=["git"], returncode=0, stdout="not porcelain\n", stderr="")
    monkeypatch.setattr(paths.subprocess, "run", lambda *a, **k: fake)
    assert paths.tree_is_clean(tmp_path) is False
    fake_ok = sp.CompletedProcess(args=["git"], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(paths.subprocess, "run", lambda *a, **k: fake_ok)
    assert paths.tree_is_clean(tmp_path) is True


def test_non_zero_git_exit_fails_closed(monkeypatch, tmp_path):
    fake = sp.CompletedProcess(args=["git"], returncode=128, stdout="", stderr="fatal")
    monkeypatch.setattr(paths.subprocess, "run", lambda *a, **k: fake)
    assert paths.tree_is_clean(tmp_path) is False


def test_forged_gate_token_is_refused():
    with pytest.raises(TypeError):
        paths.GateToken(GOOD_HASH)
    with pytest.raises(TypeError):
        paths.GateToken(GOOD_HASH, object())


@pytest.fixture
def gate(tmp_path, monkeypatch):
    log = tmp_path / "RESEARCH_LOG.md"
    log.write_text(f"FROZEN COMMIT: {GOOD_HASH}\n")
    monkeypatch.setattr(paths, "RESEARCH_LOG", log)
    monkeypatch.setattr(paths, "current_commit", lambda *a, **k: GOOD_HASH)
    monkeypatch.setattr(paths, "tree_is_clean", lambda *a, **k: True)
    return log


def test_verify_returns_a_token_for_the_frozen_commit(gate):
    token = paths.verify_test_gate(GOOD_HASH)
    assert isinstance(token, paths.GateToken) and token.commit == GOOD_HASH


def test_private_chunk_path_refuses_test_without_a_token(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no test data may be generated without a gate token")
    monkeypatch.setattr(paths, "gen_scenarios", boom)
    with pytest.raises(paths.TestStreamGateError):
        paths._chunk_arrays("CON3-0", "test", 0, 0, None)
    with pytest.raises(paths.TestStreamGateError):
        paths._chunk_arrays("CON3-0", "test", 0, 0, tmp_path, None)
    with pytest.raises(paths.TestStreamGateError):
        paths._generate_chunk("CON3-0", "test", 0, 0)
    with pytest.raises(paths.TestStreamGateError):
        paths._stream("CON3-0", "test", 0, 0)
    assert not any(tmp_path.iterdir())


def test_private_path_accepts_test_with_a_verified_token(gate):
    token = paths.verify_test_gate(GOOD_HASH)
    seed, n = paths._stream("CON3-0", "test", 0, 0, token)
    assert n == paths.TEST_N
    assert seed == paths.stable_seed("CON3-0") + 99_991


def test_dev_and_train_streams_need_no_token():
    assert paths._stream("CON3-0", "dev", 0, 0)[1] == paths.DEV_N
    assert paths._stream("CON3-0", "train", 1, 2)[1] == paths.CHUNK


def _model(tmp_path, fit_commit, fit_tree_clean):
    job = sweep.Job("neural", "CON3-0", 0, 100, 0, arm="full")
    payload = {"job": job, "kind": "neural", "models": {}, "stats": None, "diag": [],
               "fit_seconds": 0.0, "fit_commit": fit_commit,
               "fit_tree_clean": fit_tree_clean, "m": 3, "B": 1.0}
    path = tmp_path / "full__CON3-0__r0__N100__rep0.pkl"
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    return path


def test_model_fitted_on_a_dirty_tree_is_refused_for_test(gate, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("refusal must come before any data is loaded")
    monkeypatch.setattr(sweep, "route_pool", boom)
    ctx = sweep.Ctx(tmp_path, "main", sweep.StageConfig(), GOOD_HASH)
    model = _model(tmp_path, GOOD_HASH, fit_tree_clean=False)
    with pytest.raises(sweep.GateError, match="uncommitted"):
        sweep.run_eval_job(model, "test", ctx, frozen=GOOD_HASH)


def test_model_without_the_clean_flag_is_refused_for_test(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "route_pool", lambda *a, **k: 1 / 0)
    ctx = sweep.Ctx(tmp_path, "main", sweep.StageConfig(), GOOD_HASH)
    path = tmp_path / "old.pkl"
    with open(path, "wb") as fh:
        pickle.dump(dict(job=sweep.Job("neural", "CON3-0", 0, 100, 0, arm="full"),
                         kind="neural", models={}, stats=None, diag=[], fit_seconds=0.0,
                         fit_commit=GOOD_HASH), fh)
    with pytest.raises(sweep.GateError):
        sweep.run_eval_job(path, "test", ctx, frozen=GOOD_HASH)


def test_fit_records_the_tree_state_at_fit_time(tmp_path, monkeypatch):
    from transvrp.expa.train import StageConfig
    monkeypatch.setattr(sweep, "tree_is_clean", lambda *a, **k: False)
    ctx = sweep.Ctx(tmp_path, "pilot", StageConfig(s_max=10), "c" * 40)
    job = sweep.Job("ladder", "CON3-0", 3, 100, 0, labels=("baton_eq",))
    assert sweep.run_fit_job(job, ctx) == "fit"
    with open(tmp_path / "models" / "pilot" / f"{job.key}.pkl", "rb") as fh:
        payload = pickle.load(fh)
    assert payload["fit_commit"] == "c" * 40 and payload["fit_tree_clean"] is False

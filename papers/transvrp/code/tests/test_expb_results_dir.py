"""C1 (results directory), C2 (reuse stages disabled), M3 (one GPU process).

The end-to-end test runs the real ``b_smoke`` fit and manifest commands in a temporary
git repository with no explicit ``--results-dir`` and then runs the gate's own model and
library checks over what the commands wrote (nothing in the gate is mocked)."""
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import torch

from transvrp.expa import sweep
from transvrp.expb import gatecheck as gc
from transvrp.expb import stages

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=test",
                           "-c", "user.email=test@example.invalid", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def test_default_results_dir_of_b_stages_is_where_the_gate_reads():
    expected = Path(gc.REPO_ROOT) / gc.RESULTS_REL
    for stage in ("b_smoke", "b_pilot", "b_thr", "b_tune_dqn_c0", "b_full_ind"):
        assert sweep.default_results_dir(stage).resolve() == expected.resolve(), stage
    assert sweep.default_results_dir("main") == sweep.DEFAULT_RESULTS
    assert sweep.default_results_dir("pilot") == sweep.DEFAULT_RESULTS


def test_cli_without_results_dir_resolves_per_stage(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(sweep, "run_fit", lambda stage, elig, rd, *a, **k: seen.update(fit=rd) or {})
    monkeypatch.setattr(sweep, "write_manifest",
                        lambda stage, repo=None, results_dir=None: seen.update(man=results_dir)
                        or tmp_path / "m")
    assert sweep.main(["fit", "--stage", "b_smoke"]) == 0
    assert sweep.main(["manifest", "--stage", "b_smoke"]) == 0
    assert Path(seen["fit"]).resolve() == (Path(gc.REPO_ROOT) / gc.RESULTS_REL).resolve()
    assert Path(seen["man"]).resolve() == Path(seen["fit"]).resolve()
    assert sweep.main(["manifest", "--stage", "b_smoke", "--results-dir", str(tmp_path)]) == 0
    assert seen["man"] == tmp_path


@needs_git
def test_b_smoke_fit_manifest_and_gate_model_checks_agree_in_a_tmp_repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "core.autocrlf", "false")
    (root / ".gitignore").write_text("papers/transvrp/code/results/expB/models/\n"
                                     "papers/transvrp/code/results/expB/chunks/\n")
    (root / "README").write_text("x\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "F")
    fit_commit = _git(root, "rev-parse", "HEAD")
    results = root / gc.RESULTS_REL
    monkeypatch.setattr(gc, "REPO_ROOT", root)
    monkeypatch.setattr(sweep, "DEFAULT_RESULTS_B", results)
    monkeypatch.setattr(sweep, "tree_is_clean", lambda: True)
    monkeypatch.setattr(sweep, "current_commit", lambda: fit_commit)
    real_a = sweep.DEFAULT_RESULTS / "models" / "b_smoke"
    assert not real_a.exists()
    assert sweep.main(["fit", "--stage", "b_smoke"]) == 0
    assert sweep.main(["manifest", "--stage", "b_smoke"]) == 0
    assert not real_a.exists()
    assert gc.model_dir("b_smoke", root).is_dir()
    assert (root / gc.manifest_rel("b_smoke")).is_file()
    payload = sweep.load_payload(sorted(gc.model_dir("b_smoke", root).glob("*.pkl"))[0])
    assert payload["cuts"] == stages.cut_record(stages.ACTIVE_CUTS)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "M")
    head = _git(root, "rev-parse", "HEAD")
    planned = gc.planned_job_count("b_smoke", root)
    gc.check_models(root, head, "b_smoke", fit_commit, planned)
    gc.check_libraries(root, "b_smoke")


def test_write_manifest_honours_an_explicit_results_dir(tmp_path):
    import pickle
    models = tmp_path / "models" / "b_x"
    models.mkdir(parents=True)
    with open(models / "thr__A__r0__N1__rep0.pkl", "wb") as fh:
        pickle.dump({"fit_commit": "a" * 40, "fit_tree_clean": True}, fh)
    out = gc.write_manifest("b_x", results_dir=tmp_path)
    assert out == tmp_path / "b_x" / "models.sha256"
    assert out.read_text().split()[1:] == ["thr__A__r0__N1__rep0.pkl", "a" * 40, "True"]


# ---------------------------------------------------------------- C2: reuse stages

REUSE = ("b_full_reuse", "b_ladder_reuse")


def test_reuse_stages_are_marked_disabled_and_refuse_every_entry_point(tmp_path):
    for stage in REUSE:
        assert sweep.stage_spec(stage).disabled
        with pytest.raises(ValueError, match="disabled"):
            sweep.run_fit(stage, pd.DataFrame(), results_dir=tmp_path)
        with pytest.raises(ValueError, match="disabled"):
            sweep.run_eval(stage, "dev", results_dir=tmp_path)
        with pytest.raises(ValueError, match="disabled"):
            sweep.merge_stage(stage, "dev", tmp_path)
        with pytest.raises(sweep.GateError, match="disabled"):
            sweep.run_eval(stage, "test_b", results_dir=tmp_path, frozen_commit="0" * 40)
        assert sweep.main(["manifest", "--stage", stage, "--results-dir", str(tmp_path)]) == 2
        with pytest.raises(gc.GateRefused, match="disabled"):
            gc.verify_stage(stage, "0" * 40)
    assert not [s for s, sp in sweep.STAGE_SPECS.items() if sp.disabled and s not in REUSE]


def test_stage_level_gate_refuses_a_disabled_stage_without_the_commit_checks(monkeypatch):
    monkeypatch.setattr(sweep, "verify_stream_gate", lambda *a, **k: None)
    for stage in REUSE:
        with pytest.raises(sweep.GateError, match="disabled"):
            sweep.check_test_gate(stage, "test_b", "0" * 40)


# ---------------------------------------------------------------- M3: one GPU process

def test_gpu_semantics_is_one_process():
    assert sweep.MAX_GPU_JOBS == 1


def test_cuda_stages_are_capped_and_cpu_stages_are_not(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert sweep.effective_jobs("b_dqn", 12) == 1
    assert sweep.effective_jobs("b_tune_dqn_c0", 12) == 1
    assert sweep.effective_jobs("b_dqn", 1) == 1
    for stage in ("b_ppo", "b_thr", "b_rlsm", "b_full", "b_smoke", "b_tune_ppo_c1"):
        assert sweep.effective_jobs(stage, 12) == 12, stage
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert sweep.effective_jobs("b_dqn", 12) == 12      # CPU fallback keeps the workers


def test_run_fit_hands_the_capped_job_count_to_the_pool(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sweep, "tree_is_clean", lambda: True)
    seen = []
    monkeypatch.setattr(sweep, "_run_all", lambda fn, items, jobs: seen.append(jobs) or [])
    elig = pd.read_csv(sweep.CODE_DIR / "results" / "expA" / "eligibility_dev.csv")
    sweep.run_fit("b_dqn", elig, results_dir=tmp_path, jobs=12, commit="x" * 40)
    sweep.run_fit("b_ppo", elig, results_dir=tmp_path, jobs=12, commit="x" * 40)
    assert seen == [1, 12]

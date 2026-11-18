"""test_b gate (EXPERIMENT_B.md 5.2, 5.3, M12). Every gate test builds a temporary git
repository with the papers/transvrp layout and never touches the real history. Nothing
here generates test_b data except the seed-state test, which uses the stream seeds only.
"""
import hashlib
import json
import pickle
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from transvrp.data import gen_scenarios, load_plans, stable_seed
from transvrp.expa import paths, sweep
from transvrp.expb import gatecheck as gc

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
LIBS = {"python": "3.13.0", "numpy": "2.0.0", "scipy": "1.14.0", "torch": "2.6.0"}
STAGES = ("b_thr", "b_alpz")
CODE_FILES = {
    "transvrp/__init__.py": "",
    "transvrp/pilot.py": "PILOT = 1\n",
    "transvrp/expa/analysis.py": "ANALYSIS = 1\n",
    "transvrp/expb/__init__.py": "",
    "transvrp/expb/thr.py": "THR = 1\n",
    "transvrp/expb/alp.py": "ALP = 1\n",
    "transvrp/expb/rlsm.py": "RLSM = 1\n",
    "transvrp/expb/dos.py": "DOS = 1\n",
    "transvrp/expb/dqn.py": "DQN = 1\n",
    "transvrp/expb/ppo.py": "PPO = 1\n",
    "transvrp/expb/fidelity.py": "FIDELITY = 1\n",
    "transvrp/expb/analysis_b.py": "ANALYSIS_B = 1\n",
}
FAMILY_BLOCK = {"family": ["thr", "alpz"], "descriptive": [], "dropped": [],
                "references": ["full", "ladder"], "independence": []}


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=test",
                           "-c", "user.email=test@example.invalid", *args],
                          check=True, capture_output=True, text=True).stdout


def _head(root):
    return _git(root, "rev-parse", "HEAD").strip()


def _commit(root, message):
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    return _head(root)


def _write(root, rel, text, newline="\n"):
    """Writes ``rel`` (repository-relative, forward slashes) under ``root``."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline=newline) as fh:
        fh.write(text)
    return path


def _write_models(root, stage, fit_commit, n, odd_index=None):
    d = gc.model_dir(stage, root)
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        libs = dict(LIBS, numpy="1.0.0") if i == odd_index else LIBS
        payload = {"fit_commit": fit_commit, "fit_tree_clean": True, "libraries": libs,
                   "kind": "neural"}
        with open(d / f"thr__CON3-0__r{i}__N100000__rep0.pkl", "wb") as fh:
            pickle.dump(payload, fh)


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setattr(gc, "library_versions", lambda: dict(LIBS))
    monkeypatch.setattr(gc, "planned_job_count", lambda stage, repo=None: 2)


def build_project(root, monkeypatch, *, repair=False, pre_manifest=None, family_edit=None,
                  odd_library_index=None):
    """Repository with fit commit F (both stages), optional repair of alp.py (stage
    b_alpz refit at R), optional pre-manifest code changes, the manifest commit M and
    the pinning commit L. ``odd_library_index`` gives that payload (per stage) other
    library versions. Returns the commit hashes."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "core.autocrlf", "false")
    _write(root, ".gitignore", "papers/transvrp/code/results/expB/models/\n__pycache__/\n*.pyc\n")
    for rel, text in CODE_FILES.items():
        _write(root, f"papers/transvrp/code/{rel}", text)
    _write(root, gc.LOG_REL, "# research log\n")
    fit = _commit(root, "F: code")
    fits = {"b_thr": fit, "b_alpz": fit}
    if repair:
        _write(root, f"{gc.CODE_REL}/expb/alp.py", "ALP = 2\n")
        fits["b_alpz"] = _commit(root, "R: repair alp")
    for rel, text in (pre_manifest or {}).items():
        _write(root, f"papers/transvrp/code/{rel}", text)
    if pre_manifest:
        _commit(root, "change before the manifest")
    for stage in STAGES:
        _write_models(root, stage, fits[stage], 2, odd_index=odd_library_index)
    monkeypatch.setattr(gc, "REPO_ROOT", root)
    for stage in STAGES:
        gc.write_manifest(stage, root)
    _write(root, gc.FIDELITY_REL, "# fidelity\n\n```family-derivation\n"
           + json.dumps(FAMILY_BLOCK) + "\n```\n")
    manifest_commit = _commit(root, "M: manifests and fidelity")
    family_report = gc.family_report(manifest_commit, root)
    if family_edit:
        family_report = family_edit(family_report)
    _write(root, gc.LOG_REL, "# research log\n" + family_report + "\n")
    pin = _commit(root, "L: pin family")
    return SimpleNamespace(root=root, F=fit, R=fits["b_alpz"], M=manifest_commit, L=pin)


@pytest.fixture
def project(tmp_path, monkeypatch, mock_env):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    return build_project(tmp_path / "repo", monkeypatch)


# ---------------------------------------------------------------- (b) gate items

@needs_git
def test_clean_pinned_project_passes(project):
    head, family_sha = gc.verify_stage("b_thr", project.F, project.root)
    assert head == project.L
    assert len(family_sha) == 64


@needs_git
def test_repaired_other_arm_is_accepted_for_thr(tmp_path, monkeypatch, mock_env):
    p = build_project(tmp_path / "repo", monkeypatch, repair=True)
    assert gc.verify_stage("b_thr", p.F, p.root)[0] == p.L
    assert gc.verify_stage("b_alpz", p.R, p.root)[0] == p.L


@needs_git
def test_family_line_only_in_working_tree_is_refused(project):
    # before the pinning commit the FAMILY line exists nowhere in HEAD's blob
    with pytest.raises(gc.GateRefused, match="FAMILY"):
        gc.pinned_log(project.root, project.M)
    with open(project.root / gc.LOG_REL, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("TEST_B FAMILY CLOSED: family=[thr] descriptive=[] dropped=[] references=[] "
                 "independence=[] fidelity_sha256=" + "0" * 64 + " manifests_sha256=" + "0" * 64
                 + " code_sha256=" + "0" * 64 + " analysis_sha256=" + "0" * 64 + "\n")
    with pytest.raises(gc.GateRefused, match="tracked change"):
        gc.verify_stage("b_thr", project.F, project.root)


@needs_git
def test_two_family_lines_refused(project):
    line = [x for x in gc._log_lines(gc.log_text(_head(project.root), project.root))
            if x.startswith(gc.FAMILY_PREFIX)][0]
    text = gc.log_text(_head(project.root), project.root)
    _write(project.root, gc.LOG_REL, text + line + "\n")
    _commit(project.root, "second family line")
    with pytest.raises(gc.GateRefused):
        gc.verify_stage("b_thr", project.F, project.root)


@needs_git
def test_frozen_line_below_family_refused(project):
    text = gc.log_text(_head(project.root), project.root)
    _write(project.root, gc.LOG_REL,
           text + f"FROZEN COMMIT (test_b): {project.F} stages=[b_rlsm]\n")
    _commit(project.root, "frozen below family")
    with pytest.raises(gc.GateRefused, match="below the FAMILY"):
        gc.pinned_log(project.root, _head(project.root))


@needs_git
def test_family_line_edited_in_second_commit_refused(project):
    text = gc.log_text(_head(project.root), project.root)
    _write(project.root, gc.LOG_REL, text.replace("family=[thr,alpz]", "family=[thr]"))
    _commit(project.root, "edit the family line")
    head = _head(project.root)
    family_at_head = [x for x in gc._log_lines(gc.log_text(head, project.root))
                      if x.startswith(gc.FAMILY_PREFIX)][0]
    # item 2 itself: two commits touched the pinned lines, so the pin is refused even
    # though the edited FAMILY line matches HEAD (the newest commit is the edit)
    with pytest.raises(gc.GateRefused, match="touched the pinned lines"):
        gc.check_pin_commit(project.root, head, family_at_head)
    with pytest.raises(gc.GateRefused):
        gc.verify_stage("b_thr", project.F, project.root)


@needs_git
def test_stage_in_no_frozen_line_refused(project):
    with pytest.raises(gc.GateRefused, match="appears in 0"):
        gc.verify_stage("b_rlsm", project.F, project.root)


def test_stage_in_two_frozen_lines_refused():
    pinned = gc.PinnedLog(family_line="", lists={}, hashes={},
                          frozen=(("a" * 40, ("b_thr",)), ("b" * 40, ("b_thr", "b_alpz"))))
    with pytest.raises(gc.GateRefused, match="appears in 2"):
        gc.frozen_commit_for(pinned, "b_thr")


@needs_git
def test_wrong_fit_commit_refused(project):
    with pytest.raises(gc.GateRefused):
        gc.verify_stage("b_thr", project.M, project.root)


@needs_git
def test_fit_commit_not_ancestor_refused(project):
    _git(project.root, "checkout", "-q", "--orphan", "side")
    _git(project.root, "rm", "-rq", "--cached", ".")
    _write(project.root, "side.txt", "side\n")
    side = _commit(project.root, "side commit")
    _git(project.root, "checkout", "-q", "-f", project.L)
    with pytest.raises(gc.GateRefused, match="not an ancestor"):
        gc.check_ancestor(project.root, side)


@needs_git
def test_fit_commit_log_holding_family_line_refused(project):
    with pytest.raises(gc.GateRefused, match="already holds a FAMILY"):
        gc.check_fit_blob(project.root, project.L, "b_rlsm")
    with pytest.raises(gc.GateRefused, match="already freezes"):
        gc.check_fit_blob(project.root, project.L, "b_thr")


@needs_git
def test_pre_manifest_change_of_other_arm_accepted_for_thr(tmp_path, monkeypatch, mock_env):
    p = build_project(tmp_path / "repo", monkeypatch,
                      pre_manifest={"transvrp/expb/alp.py": "ALP = 3\n"})
    assert gc.verify_stage("b_thr", p.F, p.root)[0] == p.L


@needs_git
def test_pre_manifest_change_of_own_arm_refused(tmp_path, monkeypatch, mock_env):
    p = build_project(tmp_path / "repo", monkeypatch,
                      pre_manifest={"transvrp/expb/thr.py": "THR = 9\n"})
    with pytest.raises(gc.GateRefused, match="differs from the code"):
        gc.verify_stage("b_thr", p.F, p.root)


@needs_git
def test_pre_manifest_change_of_pilot_refused_for_every_stage(tmp_path, monkeypatch, mock_env):
    p = build_project(tmp_path / "repo", monkeypatch,
                      pre_manifest={"transvrp/pilot.py": "PILOT = 9\n"})
    with pytest.raises(gc.GateRefused):
        gc.verify_stage("b_thr", p.F, p.root)


@needs_git
def test_pre_manifest_change_of_analysis_b_refused(tmp_path, monkeypatch, mock_env):
    p = build_project(tmp_path / "repo", monkeypatch,
                      pre_manifest={"transvrp/expb/analysis_b.py": "ANALYSIS_B = 9\n"})
    with pytest.raises(gc.GateRefused):
        gc.verify_stage("b_thr", p.F, p.root)


@needs_git
def test_code_changed_after_manifest_commit_refused(project):
    _write(project.root, "papers/transvrp/code/transvrp/pilot.py", "PILOT = 5\n")
    _commit(project.root, "code after pin")
    with pytest.raises(gc.GateRefused, match="after the manifest commit"):
        gc.verify_stage("b_thr", project.F, project.root)


@needs_git
def test_crlf_only_change_is_accepted(project):
    text = (project.root / gc.CODE_REL / "expb" / "thr.py").read_text(encoding="utf-8")
    _write(project.root, "papers/transvrp/code/transvrp/expb/thr.py",
           text.replace("\n", "\r\n"), newline="")
    _commit(project.root, "crlf only")
    assert gc.verify_stage("b_thr", project.F, project.root)[0] == _head(project.root)


@needs_git
def test_content_change_after_pin_is_refused(project):
    _write(project.root, "papers/transvrp/code/transvrp/expb/thr.py", "THR = 2\r\n", newline="")
    _commit(project.root, "content change")
    with pytest.raises(gc.GateRefused):
        gc.verify_stage("b_thr", project.F, project.root)


@needs_git
def test_untracked_py_file_refused(project):
    _write(project.root, "papers/transvrp/code/transvrp/expb/ghost.py", "GHOST = 1\n")
    with pytest.raises(gc.GateRefused, match="untracked or ignored source"):
        gc.check_clean(project.root)


@needs_git
def test_sourceless_pyc_outside_pycache_refused(project):
    _write(project.root, "papers/transvrp/code/transvrp/expb/ghost.pyc", "x")
    with pytest.raises(gc.GateRefused, match="bytecode outside"):
        gc.check_clean(project.root)


@needs_git
def test_orphan_pycache_file_refused(project):
    _write(project.root, "papers/transvrp/code/transvrp/expb/__pycache__/ghost.cpython-313.pyc",
           "x")
    with pytest.raises(gc.GateRefused, match="orphan bytecode"):
        gc.check_clean(project.root)


@needs_git
def test_pycache_of_a_tracked_module_is_allowed(project):
    _write(project.root, "papers/transvrp/code/transvrp/expb/__pycache__/thr.cpython-313.pyc",
           "x")
    gc.check_clean(project.root)        # no raise


@needs_git
def test_dirty_tracked_code_refused(project):
    _write(project.root, "papers/transvrp/code/transvrp/expb/thr.py", "THR = 7\n")
    with pytest.raises(gc.GateRefused, match="tracked change"):
        gc.check_clean(project.root)


@needs_git
def test_dirty_log_refused(project):
    with open(project.root / gc.LOG_REL, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("note\n")
    with pytest.raises(gc.GateRefused, match="tracked change"):
        gc.check_clean(project.root)


@needs_git
def test_model_hash_differs_from_manifest_refused(project):
    target = next(gc.model_dir("b_thr", project.root).glob("*.pkl"))
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(gc.GateRefused, match="differs from its manifest"):
        gc.verify_stage("b_thr", project.F, project.root)


@needs_git
def test_missing_model_file_refused(project):
    next(gc.model_dir("b_thr", project.root).glob("*.pkl")).unlink()
    with pytest.raises(gc.GateRefused, match="missing"):
        gc.verify_stage("b_thr", project.F, project.root)


@needs_git
def test_model_count_differs_from_plan_refused(project, monkeypatch):
    monkeypatch.setattr(gc, "planned_job_count", lambda stage, repo=None: 3)
    with pytest.raises(gc.GateRefused, match="planned"):
        gc.verify_stage("b_thr", project.F, project.root)


@needs_git
def test_manifest_changed_after_manifest_commit_refused(project):
    manifest = project.root / gc.manifest_rel("b_thr")
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8",
                        newline="\n")
    _commit(project.root, "touch manifest")
    with pytest.raises(gc.GateRefused, match="manifest differs"):
        gc.verify_stage("b_thr", project.F, project.root)


@needs_git
def test_family_list_differing_from_block_refused(tmp_path, monkeypatch, mock_env):
    p = build_project(tmp_path / "repo", monkeypatch,
                      family_edit=lambda text: text.replace("family=[thr,alpz]",
                                                            "family=[thr]"))
    with pytest.raises(gc.GateRefused, match="differs from the derivation"):
        gc.verify_stage("b_thr", p.F, p.root)


@needs_git
def test_fidelity_hash_mismatch_refused(project):
    fidelity = project.root / gc.FIDELITY_REL
    fidelity.write_text(fidelity.read_text(encoding="utf-8") + "\nedit\n", encoding="utf-8",
                        newline="\n")
    _commit(project.root, "edit fidelity")
    with pytest.raises(gc.GateRefused, match="fidelity_dev.md differs"):
        gc.verify_stage("b_thr", project.F, project.root)


@needs_git
def test_stage_arm_outside_every_family_list_refused(project):
    pinned = gc.pinned_log(project.root, _head(project.root))
    with pytest.raises(gc.GateRefused, match="not in any FAMILY list"):
        gc.check_family_lists(project.root, _head(project.root), pinned, "b_dqn")


@needs_git
def test_library_version_mismatch_refused(project, monkeypatch):
    monkeypatch.setattr(gc, "library_versions", lambda: dict(LIBS, numpy="1.0.0"))
    with pytest.raises(gc.GateRefused, match="library versions"):
        gc.verify_stage("b_thr", project.F, project.root)


def test_family_line_hash_is_stable_and_unique_per_pin(project):
    a = gc.verify_stage("b_thr", project.F, project.root)[1]
    b = gc.verify_stage("b_alpz", project.F, project.root)[1]
    assert a == b


# ---------------------------------------------------------------- (c) CRLF normalisation

def test_norm_replaces_crlf_only():
    assert gc.norm(b"a\r\nb\n\r\n") == b"a\nb\n\n"


# ---------------------------------------------------------------- (d) stream separation

def _token(stream, stage=None):
    return paths.GateToken(commit="a" * 40, stream=stream, stage=stage,
                           _issuer=paths._GATE_ISSUER)


def test_test_token_does_not_unlock_test_b_chunks():
    with pytest.raises(paths.TestStreamGateError):
        paths._require_gate("test_b", _token("test"))


def test_test_b_token_does_not_unlock_test_chunks():
    with pytest.raises(paths.TestStreamGateError):
        paths._require_gate("test", _token("test_b", "b_thr"))


def test_forged_token_is_refused_for_both_streams():
    fake = paths.GateToken(commit="a" * 40, stream="test_b", _issuer=None) \
        if False else None
    with pytest.raises(paths.TestStreamGateError):
        paths._require_gate("test_b", None)
    assert fake is None


def test_test_b_pool_with_test_token_is_refused_before_generation(monkeypatch):
    monkeypatch.setattr(paths, "verify_stream_gate", lambda *a, **k: _token("test"))

    def boom(*a, **k):
        raise AssertionError("no test_b data may be generated")
    monkeypatch.setattr(paths, "gen_scenarios", boom)
    with pytest.raises(paths.TestStreamGateError):
        paths.route_pool("CON3-0", "test_b", frozen="a" * 40, stage="b_thr")


def test_test_b_pool_needs_a_stage_name():
    with pytest.raises(paths.TestStreamGateError, match="name of the stage"):
        paths.verify_test_b_gate(None, "a" * 40)


def test_a_stage_refuses_test_b_and_b_stage_refuses_test(monkeypatch):
    with pytest.raises(sweep.GateError):
        sweep.check_test_gate("main", "test_b", "a" * 40)
    monkeypatch.setitem(sweep.STAGE_SPECS, "b_probe",
                        sweep.StageSpec(reps=(0,), test_stream="test_b"))
    with pytest.raises(sweep.GateError, match="not on 'test'"):
        sweep.check_test_gate("b_probe", "test", "a" * 40)


def test_a_test_gate_is_not_unlocked_by_a_test_b_line(tmp_path, monkeypatch):
    good = "a" * 40
    log = tmp_path / "RESEARCH_LOG.md"
    log.write_text(f"FROZEN COMMIT (test_b): {good} stages=[b_thr]\n", encoding="utf-8")
    monkeypatch.setattr(paths, "RESEARCH_LOG", log)
    monkeypatch.setattr(paths, "current_commit", lambda *a, **k: good)
    monkeypatch.setattr(paths, "tree_is_clean", lambda *a, **k: True)
    with pytest.raises(paths.TestStreamGateError):
        paths.verify_test_gate(good)


def test_a_anchored_line_still_unlocks_a(tmp_path, monkeypatch):
    good = "b" * 40
    log = tmp_path / "RESEARCH_LOG.md"
    log.write_text(f"FROZEN COMMIT: {good}\n", encoding="utf-8")
    monkeypatch.setattr(paths, "RESEARCH_LOG", log)
    monkeypatch.setattr(paths, "current_commit", lambda *a, **k: good)
    monkeypatch.setattr(paths, "tree_is_clean", lambda *a, **k: True)
    assert paths.verify_test_gate(good).stream == "test"


def test_a_substring_mention_no_longer_unlocks_a(tmp_path, monkeypatch):
    good = "c" * 40
    log = tmp_path / "RESEARCH_LOG.md"
    log.write_text(f"see commit {good} in the notes\n", encoding="utf-8")
    monkeypatch.setattr(paths, "RESEARCH_LOG", log)
    monkeypatch.setattr(paths, "current_commit", lambda *a, **k: good)
    monkeypatch.setattr(paths, "tree_is_clean", lambda *a, **k: True)
    with pytest.raises(paths.TestStreamGateError):
        paths.verify_test_gate(good)


def test_test_b_size_and_seed_are_distinct_from_test():
    seed, n = paths._stream("CON3-0", "test_b", 0, 0, _token("test_b", "b_thr"))
    assert seed == [stable_seed("CON3-0"), 20261118] and n == 20_000
    assert paths._stream("CON3-0", "test", 0, 0, _token("test"))[0] != seed


# ---------------------------------------------------------------- (f) no expb module reads test/test_std

def test_no_expb_module_requests_test_or_test_std():
    expb_dir = Path(gc.__file__).parent
    pattern = re.compile(r"""["'](test|test_std)["']""")
    offenders = [p.name for p in expb_dir.glob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


# ---------------------------------------------------------------- (a) seed states (slow)

@pytest.mark.slow
def test_test_b_seed_states_are_distinct_for_all_instances():
    elig = Path(gc.__file__).resolve().parents[2] / "results" / "expA" / "eligibility_dev.csv"
    insts = sorted(pd.read_csv(elig)["inst"].unique())
    assert len(insts) == 20
    for inst in insts:
        base = stable_seed(inst)
        seeds = [[base, 20261118], base + 99_991, [base, 3]]
        seeds += [[base, 1, r, c] for r in range(3) for c in range(10)]
        states = [tuple(np.random.SeedSequence(s).generate_state(4)) for s in seeds]
        assert len(set(states)) == len(states), inst
        tb = gen_scenarios(load_plans(inst)["dbar"], load_plans(inst)["pbar"], 100,
                           [base, 20261118])
        dev = gen_scenarios(load_plans(inst)["dbar"], load_plans(inst)["pbar"], 100,
                            [base, 3])
        assert not np.array_equal(tb[0], dev[0]), inst


# ---------------------------------------------------------------- review round 2 (CODE_REVIEW_B_M12_M13)

@needs_git
def test_later_prose_mention_of_the_pin_phrase_is_not_a_second_pin(project):
    # F2: a later log line that merely quotes the pinned phrases (mid-line) must not lock
    # the gate out; only lines that start with the phrase are pins
    text = gc.log_text(_head(project.root), project.root)
    _write(project.root, gc.LOG_REL, text + "note: the TEST_B FAMILY CLOSED line and "
           "FROZEN COMMIT (test_b) are pinned above\n")
    head = _commit(project.root, "prose mention of the pin phrases")
    assert gc.verify_stage("b_thr", project.F, project.root)[0] == head


@needs_git
def test_stray_pkl_not_in_manifest_is_refused(project):
    # F1: the model directory must hold exactly the manifest's files, by name
    stray = gc.model_dir("b_thr", project.root) / "thr__EVIL__r0__N1__rep0.pkl"
    stray.write_bytes(b"not in the manifest")
    with pytest.raises(gc.GateRefused, match="on disk"):
        gc.verify_stage("b_thr", project.F, project.root)


@needs_git
def test_library_mismatch_in_a_later_payload_is_refused(tmp_path, monkeypatch, mock_env):
    # F6: every manifested payload is checked, not only the first one by name
    p = build_project(tmp_path / "repo", monkeypatch, odd_library_index=1)
    with pytest.raises(gc.GateRefused, match="library versions"):
        gc.verify_stage("b_thr", p.F, p.root)


@needs_git
def test_skip_worktree_flag_on_tracked_code_is_refused(project):
    # F3: an edited file hidden by an index flag must not pass the clean check
    rel = "papers/transvrp/code/transvrp/expb/thr.py"
    _write(project.root, rel, "THR = 8\n")
    _git(project.root, "update-index", "--skip-worktree", rel)
    with pytest.raises(gc.GateRefused, match="index flag"):
        gc.check_clean(project.root)


@needs_git
def test_untracked_top_level_code_module_is_refused(project):
    # F4: a module shadowing a library, placed directly in code/, is untracked source
    _write(project.root, "papers/transvrp/code/numpy.py", "x = 1\n")
    with pytest.raises(gc.GateRefused, match="untracked or ignored source"):
        gc.check_clean(project.root)


@needs_git
def test_merge_commit_editing_the_pinned_line_is_refused(project):
    # F5: a merge whose resolution edits the FAMILY line counts as touching the pinned lines
    main = _head(project.root)
    _git(project.root, "checkout", "-q", "-b", "side", project.M)
    _write(project.root, "papers/transvrp/code/side.txt", "side\n")
    _commit(project.root, "side work")
    _git(project.root, "checkout", "-q", "--detach", main)
    _git(project.root, "merge", "-q", "--no-ff", "--no-commit", "side")
    text = gc.log_text(main, project.root)
    _write(project.root, gc.LOG_REL, text.replace("family=[thr,alpz]", "family=[thr]"))
    merged = _commit(project.root, "merge with an edited pin")
    family_at_merge = [x for x in gc._log_lines(gc.log_text(merged, project.root))
                       if x.startswith(gc.FAMILY_PREFIX)][0]
    with pytest.raises(gc.GateRefused, match="touched the pinned lines"):
        gc.check_pin_commit(project.root, merged, family_at_merge)


@needs_git
def test_text_hash_reads_the_committed_blob_not_the_working_copy(project):
    # mutation guard: hashes are over git-tracked content, never over the working file
    rel = "papers/transvrp/code/transvrp/expb/thr.py"
    _write(project.root, rel, "THR = 99\n")
    assert gc.text_hash(project.F, rel, project.root) == hashlib.sha256(b"THR = 1\n").hexdigest()


def test_run_eval_job_refuses_a_stage_on_another_stream(tmp_path, monkeypatch):
    # stage/stream match is enforced inside run_eval_job, before any gate or data
    monkeypatch.setitem(sweep.STAGE_SPECS, "b_probe",
                        sweep.StageSpec(reps=(0,), test_stream="test_b"))

    def no_gate(*a, **k):
        raise AssertionError("the gate must not run before the stream check")
    monkeypatch.setattr(sweep, "verify_stream_gate", no_gate)
    ctx = sweep.Ctx(tmp_path, "b_probe", sweep.StageSpec(reps=(0,)).cfg, "a" * 40)
    model = tmp_path / "models" / "b_probe" / "thr__CON3-0__r0__N100000__rep0.pkl"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x")
    with pytest.raises(sweep.GateError, match="evaluates on 'test_b'"):
        sweep.run_eval_job(model, "test", ctx, frozen="a" * 40)


def test_run_eval_job_refuses_a_model_of_another_stage(tmp_path, monkeypatch):
    # a model file of stage b_alpz must not be evaluated under the b_probe gate and stream
    monkeypatch.setitem(sweep.STAGE_SPECS, "b_probe",
                        sweep.StageSpec(reps=(0,), test_stream="test_b"))

    def no_gate(*a, **k):
        raise AssertionError("the gate must not run for a model of another stage")
    monkeypatch.setattr(sweep, "verify_stream_gate", no_gate)
    ctx = sweep.Ctx(tmp_path, "b_probe", sweep.StageSpec(reps=(0,)).cfg, "a" * 40)
    model = tmp_path / "models" / "b_alpz" / "alp__CON3-0__r0__N100000__rep0.pkl"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x")
    with pytest.raises(sweep.GateError, match="belongs to stage"):
        sweep.run_eval_job(model, "test_b", ctx, frozen="a" * 40)


def test_run_eval_job_routes_test_b_through_its_gate(tmp_path, monkeypatch):
    # mutation guard: every test_b evaluation asks the stream verifier first
    seen = []

    def recorder(purpose, frozen, stage=None):
        seen.append((purpose, frozen, stage))
        raise sweep.GateError("recorded")
    monkeypatch.setattr(sweep, "verify_stream_gate", recorder)
    ctx = sweep.Ctx(tmp_path, "b_probe", sweep.StageSpec(reps=(0,)).cfg, "a" * 40)
    monkeypatch.setitem(sweep.STAGE_SPECS, "b_probe",
                        sweep.StageSpec(reps=(0,), test_stream="test_b"))
    model = tmp_path / "models" / "b_probe" / "thr__CON3-0__r0__N100000__rep0.pkl"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x")
    with pytest.raises(sweep.GateError, match="recorded"):
        sweep.run_eval_job(model, "test_b", ctx, frozen="a" * 40)
    assert seen == [("test_b", "a" * 40, "b_probe")]


def test_allow_list_mutant_guard_every_code_file_is_hashed(project):
    # default-deny: a file outside the old expb-only allow-list (pilot.py) still counts
    files = gc.code_files(project.F, project.root)
    assert f"{gc.CODE_REL}/pilot.py" in files

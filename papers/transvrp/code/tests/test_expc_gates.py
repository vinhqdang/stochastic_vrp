"""test_c gate (EXPERIMENT_C.md 4.3, 5, 6; MC5). The stream checks run in temporary git
repositories with the papers/transvrp layout (as tests/test_expb_gate.py does). The dev
gates and the family derivation run on synthetic inputs. The W6 identity and the
prediction identity of the reused models (4.3 a-b) read the committed plans and, for the
prediction test, B's gitignored model payloads; they skip when those are absent.
"""
import dataclasses
import hashlib
import json
import pickle
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from transvrp.expa import sweep
from transvrp.expb import fidelity as F
from transvrp.expb import gatecheck as gc
from transvrp.expc import gates as CG
from transvrp.expc import identity_check as IC

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
LIBS = {"python": "3.13.0", "numpy": "2.0.0", "scipy": "1.14.0", "torch": "2.6.0"}
PLANNED = 2
STAGE_THR = "c_thr_w6"
STAGE_BOOST = "c_boost_F0_w6"
STAGE_REUSE = "c_full_w6"
CODE_FILES = {
    "transvrp/__init__.py": "",
    "transvrp/pilot.py": "PILOT = 1\n",
    "transvrp/expa/analysis.py": "ANALYSIS = 1\n",
    "transvrp/expb/__init__.py": "",
    "transvrp/expb/thr.py": "THR = 1\n",
    "transvrp/expb/rlsm.py": "RLSM = 1\n",
    "transvrp/expb/dos.py": "DOS = 1\n",
    "transvrp/expb/ppo.py": "PPO = 1\n",
    "transvrp/expb/fidelity.py": "FIDELITY = 1\n",
    "transvrp/expb/analysis_b.py": "ANALYSIS_B = 1\n",
    "transvrp/expc/__init__.py": "",
    "transvrp/expc/boost_sets.py": "BOOST = 1\n",
    "transvrp/expc/rlsm_sets.py": "RLSM_SETS = 1\n",
    "transvrp/expc/cdp.py": "CDP = 1\n",
    "transvrp/expc/analysis_c.py": "ANALYSIS_C = 1\n",
    "transvrp/expc/gates.py": "GATES = 1\n",
}
C_BLOCK = {"family": ["C5"], "descriptive": [], "dropped": [],
           "references": ["baton", "ladder", "ppo_b"]}


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
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline=newline) as fh:
        fh.write(text)
    return path


def _write_models(directory, stage, fit_commit, n):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        payload = {"fit_commit": fit_commit, "fit_tree_clean": True, "libraries": LIBS}
        with open(directory / f"x__CON3-0__r{i}__N100000__rep0.pkl", "wb") as fh:
            pickle.dump(payload, fh)


def _c_stream(reuse_fit: str) -> gc.Stream:
    """C's stream with the test planned count and the reused fit commit of the temp repo."""
    reuse = {STAGE_REUSE: (f"{gc.RESULTS_REL}/b_full/models.sha256",
                           f"{gc.RESULTS_REL}/models/b_full", reuse_fit)}
    return dataclasses.replace(CG.C_STREAM, reuse=reuse, planned=lambda s, r: PLANNED)


def _derivation():
    arms = {"thr": {"stages": [STAGE_THR], "passed": True},
            "boost_f0": {"stages": [STAGE_BOOST], "passed": True},
            "full": {"stages": [STAGE_REUSE], "passed": True}}
    return {**C_BLOCK, "arms": arms,
            "stages": {STAGE_THR: {"arm": "thr", "world": "W6", "passed": True, "gates": {}}}}


GITIGNORE = ("papers/transvrp/code/results/expC/models/\n"
             "papers/transvrp/code/results/expB/models/\n__pycache__/\n*.pyc\n")


def _build(root, monkeypatch, *, repair_cdp=False, repair_pilot=False, line_family=None):
    """Temporary C repository. Commit F holds the code; with ``repair_cdp`` a repair of
    another owner's file follows F (commit R) and precedes the manifest commit. The two C
    stages are fitted at F; B's b_full models and manifest (reused by c_full_w6) are
    written at F as well. M holds the manifests and the fidelity file; L pins the family
    line (``line_family`` replaces the family list of the pinned line, to test refusals)."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "core.autocrlf", "false")
    _write(root, ".gitignore", GITIGNORE)
    for rel, text in CODE_FILES.items():
        _write(root, f"papers/transvrp/code/{rel}", text)
    _write(root, gc.LOG_REL, "# research log\n")
    fit = _commit(root, "F: code")
    if repair_cdp:
        _write(root, "papers/transvrp/code/transvrp/expc/cdp.py", "CDP = 2\n")
        _commit(root, "R: repair of another arm")
    if repair_pilot:
        _write(root, "papers/transvrp/code/transvrp/pilot.py", "PILOT = 2\n")
        _commit(root, "R: repair of a module no owner excludes")
    stream = _c_stream(fit)
    monkeypatch.setattr(gc, "REPO_ROOT", root)
    monkeypatch.setattr(gc, "library_versions", lambda: dict(LIBS))
    for stage in (STAGE_THR, STAGE_BOOST):
        _write_models(root / stream.results_rel / "models" / stage, stage, fit, PLANNED)
        gc.write_manifest(stage, root, results_dir=root / stream.results_rel, stream=stream)
    _write_models(root / gc.RESULTS_REL / "models" / "b_full", "b_full", fit, PLANNED)
    gc.write_manifest("b_full", root, results_dir=root / gc.RESULTS_REL, stream=gc.B_STREAM)
    block = json.dumps(_derivation(), sort_keys=True)
    _write(root, stream.fidelity_rel,
           "# fidelity\n\n```family-derivation\n" + block + "\n```\n")
    manifest_commit = _commit(root, "M: manifests and fidelity")
    report = gc.family_report(manifest_commit, root, stream)
    if line_family is not None:
        report = report.replace("family=[C5]", f"family=[{line_family}]")
    _write(root, gc.LOG_REL, "# research log\n" + report + "\n")
    pin = _commit(root, "L: pin family")
    return SimpleNamespace(root=root, F=fit, M=manifest_commit, L=pin, stream=stream)


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setattr(gc, "library_versions", lambda: dict(LIBS))


@pytest.fixture
def project(tmp_path, monkeypatch, mock_env):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    return _build(tmp_path / "repo", monkeypatch)


# ---------------------------------------------------------------- stream checks (git)

@needs_git
def test_clean_pinned_c_project_passes_for_a_fitted_stage(project):
    head, family_sha = gc.verify_stage(STAGE_THR, project.F, project.root, stream=project.stream)
    assert head == project.L
    assert len(family_sha) == 64


@needs_git
def test_reused_stage_passes_at_its_fit_commit(project):
    head, _ = gc.verify_stage(STAGE_REUSE, project.F, project.root, stream=project.stream)
    assert head == project.L


@needs_git
def test_c_family_line_has_the_pinned_test_c_shape(project):
    text = (project.root / gc.LOG_REL).read_text(encoding="utf-8")
    family = [x for x in text.splitlines() if x.startswith("TEST_C FAMILY CLOSED")]
    assert len(family) == 1
    assert "independence=" not in family[0]
    assert CG.C_STREAM.family_re.fullmatch(family[0]) is not None
    assert f"FROZEN COMMIT (test_c): {project.F} stages=[" in text


@needs_git
def test_a_test_b_label_on_the_frozen_line_refuses_the_test_c_stage(project):
    """The test_c frozen pattern does not match a FROZEN (test_b) line, so the stage is in
    no test_c FROZEN line and the gate refuses."""
    text = (project.root / gc.LOG_REL).read_text(encoding="utf-8")
    _write(project.root, gc.LOG_REL, text.replace("FROZEN COMMIT (test_c)", "FROZEN COMMIT (test_b)"))
    _commit(project.root, "tamper: test_b label")
    with pytest.raises(gc.GateRefused):
        gc.verify_stage(STAGE_THR, project.F, project.root, stream=project.stream)


@needs_git
def test_a_test_b_style_family_line_is_not_read_as_the_test_c_line(project):
    text = (project.root / gc.LOG_REL).read_text(encoding="utf-8")
    family = [x for x in text.splitlines() if x.startswith("TEST_C FAMILY CLOSED")][0]
    b_style = family.replace("TEST_C", "TEST_B").replace(
        "references=[baton,ladder,ppo_b]", "references=[baton,ladder,ppo_b] independence=[]")
    _write(project.root, gc.LOG_REL, text.replace(family, b_style))
    _commit(project.root, "tamper: b-style line")
    with pytest.raises(gc.GateRefused, match="FAMILY"):
        gc.verify_stage(STAGE_THR, project.F, project.root, stream=project.stream)


@needs_git
def test_malformed_test_c_family_line_is_refused(project):
    text = (project.root / gc.LOG_REL).read_text(encoding="utf-8")
    family = [x for x in text.splitlines() if x.startswith("TEST_C FAMILY CLOSED")][0]
    _write(project.root, gc.LOG_REL, text.replace(family, family.replace("code_sha256=", "x=")))
    _commit(project.root, "tamper: malformed")
    with pytest.raises(gc.GateRefused, match="malformed"):
        gc.verify_stage(STAGE_THR, project.F, project.root, stream=project.stream)


@needs_git
def test_family_line_differing_from_the_derivation_block_is_refused(tmp_path, monkeypatch, mock_env):
    project = _build(tmp_path / "repo", monkeypatch, line_family="")
    with pytest.raises(gc.GateRefused, match="differs from the derivation"):
        gc.verify_stage(STAGE_THR, project.F, project.root, stream=project.stream)


@needs_git
def test_cli_family_line_for_test_c_prints_the_pinned_line(project, capsys):
    assert gc.main(["family-line", "--stream", "test_c", "--commit", project.M]) == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("TEST_C FAMILY CLOSED: family=[C5] descriptive=[]")
    assert CG.C_STREAM.family_re.fullmatch(last) is not None


@needs_git
def test_unlisted_arm_is_refused(project):
    fake = dataclasses.replace(project.stream, core_of=lambda s: "nowhere")
    with pytest.raises(gc.GateRefused, match="not in any FAMILY list"):
        gc.verify_stage(STAGE_THR, project.F, project.root, stream=fake)


@needs_git
def test_reused_stage_is_refused_at_any_other_fit_commit(project):
    manifest, models, _ = project.stream.reuse[STAGE_REUSE]
    other = dataclasses.replace(project.stream, reuse={STAGE_REUSE: (manifest, models, project.M)})
    with pytest.raises(gc.GateRefused, match="accepted only at"):
        gc.verify_stage(STAGE_REUSE, project.F, project.root, stream=other)


@needs_git
def test_stage_frozen_at_another_commit_is_refused(project):
    with pytest.raises(gc.GateRefused, match="frozen at another commit"):
        gc.verify_stage(STAGE_THR, project.M, project.root, stream=project.stream)


@needs_git
def test_repair_of_another_owner_is_accepted_for_the_thr_stage(tmp_path, monkeypatch, mock_env):
    """EXCL (B 5.3): the cdp file may differ between F and the evaluation commit, because it
    belongs to another owner."""
    project = _build(tmp_path / "repo", monkeypatch, repair_cdp=True)
    head, _ = gc.verify_stage(STAGE_THR, project.F, project.root, stream=project.stream)
    assert head == project.L


@needs_git
def test_a_repair_of_the_stages_own_file_is_refused(tmp_path, monkeypatch, mock_env):
    project = _build(tmp_path / "repo", monkeypatch)
    _write(project.root, "papers/transvrp/code/transvrp/expb/thr.py", "THR = 2\n")
    _commit(project.root, "repair of the thr file itself")
    with pytest.raises(gc.GateRefused):
        gc.verify_stage(STAGE_THR, project.F, project.root, stream=project.stream)


def test_excluded_files_are_the_other_owners_files():
    assert CG.c_owner_of("c_thr_w6") == "thr"
    excl_thr = gc.excluded_for_stage("c_thr_w6", CG.C_STREAM)
    assert f"{gc.CODE_REL}/expb/thr.py" not in excl_thr
    assert f"{gc.CODE_REL}/expc/cdp.py" in excl_thr
    excl_cdp = gc.excluded_for_stage("c_cdp_w6", CG.C_STREAM)
    assert f"{gc.CODE_REL}/expc/cdp.py" not in excl_cdp
    assert f"{gc.CODE_REL}/expb/thr.py" in excl_cdp


def test_c_arm_names_follow_the_family_table():
    assert CG.c_core_of("c_boost_F0n_w6") == "boost_f0n"
    assert CG.c_core_of("c_rlsm_F3_ws") == "rlsm_f3"
    assert CG.c_core_of("c_cdptrue_w6") == "cdp_true"
    assert CG.c_core_of("c_ppolong_w6") == "ppo_long"
    assert CG.c_core_of("c_ppob_w6") == "ppo_b"
    assert CG.c_core_of("c_ladder_w0") == "ladder"
    with pytest.raises(ValueError):
        CG.c_core_of("c_unknown_w6")


def test_resolve_stream_names_and_refuses_others():
    assert gc.resolve_stream("test_b") is gc.B_STREAM
    assert gc.resolve_stream("test_c") is CG.C_STREAM
    with pytest.raises(ValueError):
        gc.resolve_stream("test_d")


def test_c_stream_lines_are_exact():
    line = ("TEST_C FAMILY CLOSED: family=[C5] descriptive=[C7] dropped=[boost_f1:p1] "
            "references=[baton,ladder,ppo_b] fidelity_sha256=" + "a" * 64 +
            " manifests_sha256=" + "b" * 64 + " code_sha256=" + "c" * 64 +
            " analysis_sha256=" + "d" * 64)
    m = CG.C_STREAM.family_re.fullmatch(line)
    assert m is not None
    assert m.group(1) == "C5" and m.group(3) == "boost_f1:p1"
    frozen = f"FROZEN COMMIT (test_c): {'e' * 40} stages=[c_thr_w6,c_boost_F0_w6]"
    assert CG.C_STREAM.frozen_re.fullmatch(frozen) is not None
    assert CG.C_STREAM.frozen_re.fullmatch(frozen.replace("test_c", "test_b")) is None


def test_b_default_stream_is_unchanged_in_its_constants():
    assert gc.B_STREAM.family_keys == gc.FAMILY_KEYS
    assert gc.B_STREAM.family_re is gc.FAMILY_LINE
    assert gc.B_STREAM.frozen_re is gc.FROZEN_LINE
    assert gc.B_STREAM.patch_re is gc.PATCH_LINE
    assert gc.manifest_rel("b_thr") == f"{gc.RESULTS_REL}/b_thr/models.sha256"
    assert gc.excluded_for_stage("b_thr") == [f"{gc.CODE_REL}/expb/{f}" for f in
                                             ("alp.py", "rlsm.py", "dos.py", "dqn.py", "ppo.py")]


def test_verify_reuse_stage_refuses_a_fitted_stage():
    with pytest.raises(gc.GateRefused, match="not a reused stage"):
        CG.verify_reuse_stage(STAGE_THR)


def test_reused_stages_point_at_b_manifests_and_models():
    manifest, models, fit = CG.C_STREAM.reuse[STAGE_REUSE]
    assert manifest.endswith("results/expB/b_full/models.sha256")
    assert models.endswith("results/expB/models/b_full")
    assert fit == CG.REUSE_FIT_COMMIT
    assert CG.REUSE_FIT_COMMIT == "bf3edabd9857644c28ac4d301ea82063950715ff"
    assert CG.REUSED == {"c_full_w6": "b_full", "c_ppob_w6": "b_ppo"}


# ---------------------------------------------------------------- dev gates (synthetic)

def _rows(stage, policy, costs, rep=0, N=100000):
    """Keyed rows: costs maps (inst, route) -> mean cost."""
    return pd.DataFrame([dict(stage=stage, inst=i, route_id=r, N=N, rep=rep, policy=policy,
                              mean_cost=c, eval_set="dev") for (i, r), c in costs.items()])


def test_g0_passes_when_every_planned_job_is_fitted_and_above_oracle():
    keys = [("I0", 1, 0), ("I0", 2, 0)]
    arm = _rows("s", "boost", {("I0", 1): 10.0, ("I0", 2): 11.0})
    oracle = _rows("l", "oracle", {("I0", 1): 9.0, ("I0", 2): 10.5})
    gate = CG.gate_g0(arm, oracle, keys)
    assert gate.passed


def test_g0_refuses_a_missing_job_or_a_cost_below_the_oracle():
    keys = [("I0", 1, 0), ("I0", 2, 0)]
    arm = _rows("s", "boost", {("I0", 1): 10.0})
    oracle = _rows("l", "oracle", {("I0", 1): 9.0, ("I0", 2): 10.5})
    assert not CG.gate_g0(arm, oracle, keys).passed
    arm2 = _rows("s", "boost", {("I0", 1): 8.0, ("I0", 2): 11.0})
    assert not CG.gate_g0(arm2, oracle, keys).passed


def test_g0_refuses_an_unplanned_extra_job():
    """A fitted arm with more route-reps than planned is not the planned stage."""
    keys = [("I0", 1, 0), ("I0", 2, 0)]
    arm = _rows("s", "boost", {("I0", 1): 10.0, ("I0", 2): 11.0, ("I0", 3): 12.0})
    oracle = _rows("l", "oracle", {("I0", 1): 9.0, ("I0", 2): 10.5, ("I0", 3): 11.0})
    assert not CG.gate_g0(arm, oracle, keys).passed


def test_cdp0_gate_passes_for_identical_costs_and_fails_for_a_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(10.0, 1.0, size=2000)
    thr = {("I0", 1): [base, base + 1e-9]}
    cdp_same = {("I0", 1): [base, base]}
    cdp_shift = {("I0", 1): [base + 2.0, base + 2.0]}
    e0 = {("I0", 1): 10.0}
    assert CG.gate_cdp0(cdp_same, thr, e0).passed
    shifted = CG.gate_cdp0(cdp_shift, thr, e0)
    assert not shifted.passed and shifted.value == 0.0


def test_cdp0_share_threshold_is_95_percent():
    rng = np.random.default_rng(2)
    base = rng.normal(10.0, 1.0, size=2000)
    thr = {(f"I{k}", k): [base] for k in range(20)}
    cdp = {key: [base] for key in thr}
    cdp[("I0", 0)] = [base + 5.0]        # one route of twenty fails: share 0.95, passes
    assert CG.gate_cdp0(cdp, thr, {k: 10.0 for k in thr}).passed
    cdp[("I1", 1)] = [base + 5.0]        # two of twenty: share 0.90, refused
    assert not CG.gate_cdp0(cdp, thr, {k: 10.0 for k in thr}).passed


def test_cdp_node_gate_rebuilds_tables_through_the_eval_function_and_fails_closed(monkeypatch):
    blob = {"arm": "cdp", "cdp_model": object(), "params": {}}
    payloads = [{"job": SimpleNamespace(inst="I0", route_id=k, rep=0), "models": blob}
                for k in range(20)]
    seen = []

    def fake_tables(b, spec, train):
        seen.append(train)
        return "TABLES"

    monkeypatch.setattr(CG, "cdp_tables_of", fake_tables)
    monkeypatch.setattr(CG.CDP, "node_check",
                        lambda spec, model, tables, n, seed: {"ok": tables == "TABLES"})
    gate = CG.gate_cdp_node(payloads, lambda i, r: None, lambda job: ("train", job.route_id))
    assert gate.passed and gate.value == 1.0
    assert seen == [("train", k) for k in range(20)]     # rebuilt from train_of(job), per route
    monkeypatch.setattr(CG.CDP, "node_check", lambda spec, model, tables, n, seed: {"ok": False})
    assert not CG.gate_cdp_node(payloads, lambda i, r: None, lambda job: None).passed
    not_cdp = [{"job": SimpleNamespace(inst="I0", route_id=0, rep=0),
                "models": {"arm": "boost"}}]
    assert not CG.gate_cdp_node(not_cdp, lambda i, r: None, lambda job: None).passed


def test_w6_identity_gate_is_true_on_the_committed_plans():
    """G-W6 (6): the world-aware W6 generator reproduces A's original call bit for bit."""
    hashes = CG.w6_identity("CON3-0")
    assert hashes["train_a"] == hashes["train_c"]
    assert hashes["dev_a"] == hashes["dev_c"]
    assert CG.gate_w6("CON3-0").passed


# ---------------------------------------------------------------- family derivation

def _gate(gid, passed, value=1.0):
    return F.Gate(gid, passed, value, "t", "d")


def _w6_reports(fail=None):
    """Reports for every W6 stage of the family, all passing; ``fail`` maps a stage to a
    gate id that fails there."""
    fail = fail or {}
    spec = {
        "c_boost_F0_w6": "boost", "c_boost_F0n_w6": "boost", "c_boost_F1_w6": "boost",
        "c_boost_F2_w6": "boost", "c_boost_F3_w6": "boost",
        "c_rlsm_F0_w6": "rlsm", "c_rlsm_F0n_w6": "rlsm", "c_rlsm_F1_w6": "rlsm",
        "c_rlsm_F2_w6": "rlsm", "c_rlsm_F3_w6": "rlsm",
        "c_cdp_w6": "cdp", "c_dos_w6": "dos", "c_full_w6": "full",
    }
    reports = {}
    for stage, kind in spec.items():
        gates = [_gate("g0", True), _gate("p1", True)]
        if stage in fail:
            gates = [_gate(g, g != fail[stage]) for g in ("g0", fail[stage])]
        reports[stage] = CG.make_report(stage, "W6", kind, gates)
    return reports


def test_all_gates_passing_puts_all_eleven_contrasts_in_the_family():
    lists = CG.derive_c_family(_w6_reports())
    assert lists["family"] == [c.cid for c in CG.CONTRASTS]
    assert len(lists["family"]) == 11
    assert lists["descriptive"] == [] and lists["dropped"] == []


def test_failing_boost_f1_drops_the_contrasts_that_contain_it_only():
    lists = CG.derive_c_family(_w6_reports({"c_boost_F1_w6": "p1"}))
    assert lists["dropped"] == ["boost_f1:p1"]
    assert set(lists["descriptive"]) == {"C1-B", "C2-B", "C3-B"}
    assert "C1-R" in lists["family"] and "C4-B" in lists["family"]


def test_failing_cdp_moves_c5_c6_and_keeps_c7_in_the_family():
    lists = CG.derive_c_family(_w6_reports({"c_cdp_w6": "g_cdp_node"}))
    assert set(lists["descriptive"]) == {"C5", "C6"}
    assert "C7" in lists["family"]
    assert lists["dropped"] == ["cdp:g_cdp_node"]


def test_failing_full_p1_drops_c6_and_c7():
    lists = CG.derive_c_family(_w6_reports({"c_full_w6": "p1"}))
    assert set(lists["descriptive"]) == {"C6", "C7"}


def test_a_stage_without_a_report_is_refused():
    reports = _w6_reports()
    del reports["c_dos_w6"]
    with pytest.raises(ValueError, match="no gate report"):
        CG.derive_c_family(reports)


def test_rendered_report_carries_a_parseable_derivation_block():
    reports = _w6_reports({"c_dos_w6": "b_dos"})
    lists = CG.derive_c_family(reports)
    text = CG.render_c_report(reports, lists)
    block = text.split("```family-derivation\n", 1)[1].split("\n```", 1)[0]
    parsed = json.loads(block)
    assert parsed["family"] == lists["family"]
    assert parsed["dropped"] == ["dos:b_dos"]
    assert set(CG.C_FAMILY_KEYS) <= set(parsed)


# ---------------------------------------------------------------- prediction identity (4.3 b)

def _identity_record(project, **overrides):
    """A record that shows the identity, bound to the temporary repository: B's fit commit
    is F, C's commit is the pinned L (clean tree, same code), three routes, two instances
    plus one, both reused stages."""
    root = project.root
    routes = [{"inst": "CON3-0", "route_id": 0, "rep": 0},
              {"inst": "CON3-5", "route_id": 1, "rep": 0},
              {"inst": "SCA3-0", "route_id": 2, "rep": 0}]
    per_route = [{"inst": r["inst"], "route_id": r["route_id"], "rep": 0, "identical": True,
                  "max_abs_diff": 0.0, "keys": {"k1_0": 0.0, "k2_0": 0.0}} for r in routes]
    rec = {
        "rule": "EXPERIMENT_C 4.3(b)", "tolerance": 0, "identical": True,
        "commit_b": project.F, "commit_c": project.L, "tree_clean": True,
        "code_sha256_b": gc.tree_hash(project.F, gc.code_files(project.F, root), root),
        "code_sha256_c": gc.tree_hash(project.L, gc.code_files(project.L, root), root),
        "routes": routes,
        "stages": {s: {"max_abs_diff": 0.0, "per_route": json.loads(json.dumps(per_route))}
                   for s in CG.IDENTITY_STAGES},
    }
    rec.update(overrides)
    return rec


def _check(project, rec):
    return CG.check_identity_record(rec, project.root, project.F)


@needs_git
def test_identity_record_with_zero_differences_is_accepted(project):
    assert _check(project, _identity_record(project))["identical"] is True


@needs_git
def test_missing_identity_record_refuses_the_reuse_gate(project):
    with pytest.raises(gc.GateRefused, match="missing"):
        CG.require_identity(project.root, project.F)


@needs_git
def test_reused_stage_gate_refuses_without_the_identity_record(project, monkeypatch):
    monkeypatch.setattr(gc, "verify_stage", lambda *a, **k: (project.L, "e" * 64))
    with pytest.raises(gc.GateRefused, match="missing"):
        CG.verify_test_c_stage(STAGE_REUSE, CG.REUSE_FIT_COMMIT, project.root)


@needs_git
def test_fitted_stage_gate_does_not_read_the_identity_record(project, monkeypatch):
    monkeypatch.setattr(gc, "verify_stage", lambda *a, **k: (project.L, "e" * 64))
    assert CG.verify_test_c_stage(STAGE_THR, project.F, project.root) == (project.L, "e" * 64)


@needs_git
def test_identity_record_read_from_the_working_tree_is_checked(project):
    path = project.root / CG.IDENTITY_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_identity_record(project)), encoding="utf-8")
    assert CG.require_identity(project.root, project.F)["identical"] is True


@needs_git
def test_nonzero_difference_on_one_key_refuses_the_record(project):
    rec = _identity_record(project)
    rec["stages"]["b_ppo"]["per_route"][1]["keys"]["k1_0"] = 1e-16
    with pytest.raises(gc.GateRefused, match="nonzero or missing"):
        _check(project, rec)


@needs_git
def test_per_key_difference_is_recomputed_when_the_summary_says_zero(project):
    rec = _identity_record(project)
    rec["stages"]["b_full"]["per_route"][0]["keys"]["k2_0"] = 3e-9
    rec["stages"]["b_full"]["max_abs_diff"] = 0.0
    with pytest.raises(gc.GateRefused, match="nonzero or missing"):
        _check(project, rec)


@needs_git
def test_summary_difference_refuses_a_record_whose_keys_are_zero(project):
    rec = _identity_record(project)
    rec["stages"]["b_full"]["max_abs_diff"] = 0.5
    with pytest.raises(gc.GateRefused, match="summary"):
        _check(project, rec)


@needs_git
def test_a_route_without_keys_refuses_the_record(project):
    rec = _identity_record(project)
    rec["stages"]["b_ppo"]["per_route"][2]["keys"] = {}
    with pytest.raises(gc.GateRefused, match="nonzero or missing"):
        _check(project, rec)


@needs_git
def test_fewer_than_three_instances_refuses_the_record(project):
    rec = _identity_record(project)
    for item in rec["routes"]:
        item["inst"] = "CON3-0"
    with pytest.raises(gc.GateRefused, match="three instances"):
        _check(project, rec)


@needs_git
def test_record_bound_to_another_fit_commit_is_refused(project):
    with pytest.raises(gc.GateRefused, match="commit_b"):
        CG.check_identity_record(_identity_record(project), project.root, project.M)


@needs_git
def test_record_measured_on_uncommitted_code_is_refused(project):
    with pytest.raises(gc.GateRefused, match="uncommitted"):
        _check(project, _identity_record(project, tree_clean=False))


@needs_git
def test_record_whose_code_hash_is_not_the_commit_code_is_refused(project):
    with pytest.raises(gc.GateRefused, match="code_sha256_c"):
        _check(project, _identity_record(project, code_sha256_c="0" * 64))


def test_identity_routes_pick_one_route_for_each_of_three_instances(tmp_path):
    for stage, prefix in (("b_full", "full"), ("b_ppo", "ppo")):
        for inst in ("CON3-0", "SCA3-0", "CON3-5"):
            d = tmp_path / stage
            d.mkdir(exist_ok=True)
            (d / f"{prefix}__{inst}__r3__N100000__rep0.pkl").write_bytes(b"x")
    assert CG.identity_routes(tmp_path) == [("CON3-0", 3), ("CON3-5", 3), ("SCA3-0", 3)]


def test_identity_predict_reads_the_payload_root_given_by_flag(tmp_path, monkeypatch):
    seen = {}

    def fake(stage_dir, inst, route_id, rep=0):
        seen["dir"] = Path(stage_dir)
        return {"k1_0": np.zeros(2)}

    monkeypatch.setattr(IC, "predict_stage_route", fake)
    out = tmp_path / "out.npz"
    assert IC.main(["predict", "--stage", "b_full", "--inst", "CON3-0", "--route", "0",
                    "--code", str(tmp_path / "elsewhere"), "--payloads", str(tmp_path / "pl"),
                    "--out", str(out)]) == 0
    assert seen["dir"] == tmp_path / "pl" / "b_full"
    assert out.exists()


@needs_git
def test_reused_stage_is_refused_after_a_code_edit_following_the_family_line(project):
    _write(project.root, "papers/transvrp/code/transvrp/pilot.py", "PILOT = 5\n")
    _commit(project.root, "code after the pin")
    with pytest.raises(gc.GateRefused, match="after the manifest commit"):
        gc.verify_stage(STAGE_REUSE, project.F, project.root, stream=project.stream)


@needs_git
def test_reused_stage_accepts_a_repair_between_its_fit_and_the_manifest(tmp_path, monkeypatch,
                                                                       mock_env):
    """4.3: the features code may change after the fit as long as W6 predictions do not
    change (checked by the identity record). The fit-versus-HEAD code comparison is skipped
    for a reused stage; the pinned code hash still holds."""
    project = _build(tmp_path / "repo", monkeypatch, repair_pilot=True)
    head, _ = gc.verify_stage(STAGE_REUSE, project.F, project.root, stream=project.stream)
    assert head == project.L


@needs_git
def test_reused_stage_is_refused_when_library_versions_differ(project, monkeypatch):
    monkeypatch.setattr(gc, "library_versions", lambda: dict(LIBS, numpy="1.0.0"))
    with pytest.raises(gc.GateRefused, match="library versions"):
        gc.verify_stage(STAGE_REUSE, project.F, project.root, stream=project.stream)


def test_identity_compare_detects_a_single_changed_bit(tmp_path):
    arr = np.linspace(0.0, 1.0, 50)
    a = IC.save({"k1_0": arr}, tmp_path / "a.npz")
    changed = arr.copy()
    changed[7] = np.nextafter(changed[7], 2.0)
    b = IC.save({"k1_0": changed}, tmp_path / "b.npz")
    same, detail = IC.compare(a, b)
    assert not same and detail["k1_0"] > 0.0
    assert IC.compare(a, a)[0]


def test_identity_compare_refuses_different_keys(tmp_path):
    a = IC.save({"k1_0": np.zeros(3)}, tmp_path / "a.npz")
    b = IC.save({"k2_0": np.zeros(3)}, tmp_path / "b.npz")
    same, _ = IC.compare(a, b)
    assert not same


def test_gate_stage_names_and_plan_are_derived_from_the_spec_table():
    import re
    from transvrp.expc.stages import WORLD_SUFFIX, c_stage_defs
    defs = c_stage_defs()
    assert CG.WORLD_TAGS == ("w0", "w3", "w6", "w9", "ws")
    assert all(re.search(r"_(w0|w3|w6|w9|ws)$", name) for name in defs)
    for name in defs:
        CG.c_core_of(name)                                  # every spec stage is parseable
    fitted = {n for n, d in defs.items()
              if not d["eval_only"] and d["arms"] and not n.startswith("c_ladder_")}
    plan = CG.fitted_stage_plan()
    assert {row[0] for row in plan} == fitted
    for stage, tag, kind, policy in plan:
        assert tag == WORLD_SUFFIX[defs[stage]["world"]]
        assert kind == defs[stage]["arms"][0]
        assert policy == kind                               # every arm's row name is its arm


def test_day_costs_reads_the_file_the_sweep_writes_for_a_c_job(tmp_path):
    from transvrp.expa import sweep
    job = sweep.Job(kind="neural", inst="CON3-0", route_id=3, N=100000, rep=1, arm="cdp")
    np.savez(tmp_path / sweep.rows_file_name(job.key, "dev", "npz"), cdp=np.arange(4.0))
    got = CG.day_costs(tmp_path, "CON3-0", 3, 100000, 1, "cdp", "dev")
    assert np.array_equal(got, np.arange(4.0))
    with pytest.raises(ValueError):
        CG.day_costs(tmp_path, "CON3-0", 3, 100000, 0, "cdp", "dev")


def test_b_payloads_carry_the_library_record_that_the_reuse_gate_compares():
    from transvrp.expa import sweep
    from transvrp.expb import gatecheck as GB
    files = sorted((GB.REPO_ROOT / GB.RESULTS_REL / "models" / "b_full").glob("*.pkl"))[:1]
    if not files:
        pytest.skip("B's models are not present in this checkout")
    lib = sweep.load_payload(files[0]).get("libraries")
    assert lib is not None and set(lib) >= {"python", "numpy", "scipy", "torch"}

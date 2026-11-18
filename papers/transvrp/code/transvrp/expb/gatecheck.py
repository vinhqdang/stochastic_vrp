"""The test_b gate (EXPERIMENT_B.md section 5.3, module M12).

``verify_stage(stage, fit_commit)`` decides whether a B stage may read the test_b
stream. It reads the pinned log from committed blobs, compares every code and model
file with its content hash (CRLF-normalised, from git-tracked content), and returns
``(HEAD, sha256 of the FAMILY line)`` to ``paths.verify_test_b_gate``. Each check is a
separate function so that tests can exercise it alone. Every refusal raises
``GateRefused``, a ``TestStreamGateError``: the gate fails closed.

``write_manifest`` and ``family_report`` are the producer side (``sweep manifest`` and
the ``family-line`` step of section 5.3); they never change the verdict logic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy

from ..expa.paths import TestStreamGateError

REPO_ROOT = Path(__file__).resolve().parents[5]     # .../stochastic_vrp
PROJECT_REL = "papers/transvrp"
LOG_REL = f"{PROJECT_REL}/RESEARCH_LOG.md"
CODE_REL = f"{PROJECT_REL}/code/transvrp"
RESULTS_REL = f"{PROJECT_REL}/code/results/expB"
ELIGIBILITY_REL = f"{PROJECT_REL}/code/results/expA/eligibility_dev.csv"
FIDELITY_REL = f"{RESULTS_REL}/fidelity_dev.md"
ANALYSIS_RELS = (f"{CODE_REL}/expa/analysis.py", f"{CODE_REL}/expb/analysis_b.py",
                 f"{CODE_REL}/expb/fidelity.py")
OWN_FILES = {"thr": "thr.py", "alpz": "alp.py", "rlsm": "rlsm.py",
             "dos": "dos.py", "dqn": "dqn.py", "ppo": "ppo.py"}
# files a *_reuse stage (cut (i), section 9.2) compares instead of the whole package
REUSE_PREFIXES = ("data.py", "baton/", "expa/features.py", "expa/model.py",
                  "expa/backward.py", "expa/simulate.py", "expa/train.py",
                  "expa/baselines.py")
FAMILY_KEYS = ("family", "descriptive", "dropped", "references", "independence")
HASH_KEYS = ("fidelity_sha256", "manifests_sha256", "code_sha256", "analysis_sha256")
SOURCE_SUFFIXES = (".py", ".pyd", ".so", ".pth")
BYTECODE_RE = re.compile(r"^(?P<dir>.*)/__pycache__/(?P<mod>[^./]+)\.[^/]*\.pyc$")
DERIVATION_RE = re.compile(r"^```family-derivation\r?\n(.*?)\r?\n```", re.M | re.S)

FAMILY_PREFIX = "TEST_B FAMILY CLOSED"
FAMILY_LINE = re.compile(
    r"TEST_B FAMILY CLOSED: family=\[([a-z0-9_,]*)\] descriptive=\[([a-z0-9_,]*)\] "
    r"dropped=\[([a-z0-9_:,]*)\] references=\[([a-z0-9_,]*)\] independence=\[([a-z0-9_,]*)\] "
    r"fidelity_sha256=([0-9a-f]{64}) manifests_sha256=([0-9a-f]{64}) "
    r"code_sha256=([0-9a-f]{64}) analysis_sha256=([0-9a-f]{64})")
FROZEN_PREFIX = "FROZEN COMMIT (test_b)"
FROZEN_LINE = re.compile(
    r"FROZEN COMMIT \(test_b\): ([0-9a-f]{40}) stages=\[([a-z0-9_]+(?:,[a-z0-9_]+)*)\]")
PATCH_PREFIX = "TEST_B ANALYSIS PATCH"
PATCH_LINE = re.compile(
    r"TEST_B ANALYSIS PATCH: analysis_sha256=([0-9a-f]{64}) date=(\d{4}-\d{2}-\d{2}) "
    r"reason=(.+)")
# only lines that START with a pinned phrase are pins (EXPERIMENT_B.md 5.3, item 2)
PIN_PATTERN = "^(TEST_B FAMILY CLOSED|FROZEN COMMIT [(]test_b[)])"
INDEX_FLAG_TAG = "H"          # git ls-files -v tag of a plain tracked file
CODE_TOP_GLOB = f":(glob){PROJECT_REL}/code/*.py"   # modules directly in code/ (item 1)
MODEL_LINE = re.compile(r"([0-9a-f]{64}) (\S+\.pkl) ([0-9a-f]{40}) (True|False)")


class GateRefused(TestStreamGateError):
    """A test_b gate condition failed; the message names the condition."""


@dataclass(frozen=True)
class PinnedLog:
    family_line: str
    lists: dict          # FAMILY_KEYS -> list[str]
    hashes: dict         # HASH_KEYS -> hex digest
    frozen: tuple        # ((fit commit, (stage, ...)), ...)


# ---------------------------------------------------------------- git helpers

def _root(repo: Path | None) -> Path:
    return Path(REPO_ROOT if repo is None else repo)


def _git(root: Path, *args: str, check: bool = True) -> bytes:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                          timeout=600, check=False)
    if check and proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise GateRefused(f"git {args[0]} failed: {detail}")
    return proc.stdout


def norm(data: bytes) -> bytes:
    """CRLF to LF. The repository is checked out with CRLF on Windows, so every hash
    is taken over normalised git-tracked content, never over working-tree bytes."""
    return data.replace(b"\r\n", b"\n")


def text_hash(commit: str, path: str, repo: Path | None = None) -> str:
    return hashlib.sha256(norm(_git(_root(repo), "show", f"{commit}:{path}"))).hexdigest()


def tracked_under(commit: str, prefix: str, repo: Path | None = None) -> list[str]:
    out = _git(_root(repo), "ls-tree", "-r", "--name-only", commit, "--", prefix)
    return sorted(line for line in out.decode("utf-8").splitlines() if line)


def tree_hash(commit: str, files: list[str], repo: Path | None = None) -> str:
    """sha256 of the sorted lines "<path> <text hash>\\n" over the given files."""
    lines = sorted(f"{p} {text_hash(commit, p, repo)}\n" for p in files)
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def code_files(commit: str, repo: Path | None = None, exclude=()) -> list[str]:
    """CODE minus ``exclude``: every tracked file under the package (default-deny)."""
    skip = set(exclude)
    return [p for p in tracked_under(commit, CODE_REL, repo) if p not in skip]


def _log_lines(text: str) -> list[str]:
    return [line.rstrip("\r") for line in text.splitlines()]


def log_text(commit: str, repo: Path | None = None) -> str:
    """The committed log; a commit without the file has no lines."""
    return norm(_git(_root(repo), "show", f"{commit}:{LOG_REL}", check=False)).decode("utf-8")


# ---------------------------------------------------------------- stages

def core_of_stage(stage: str) -> str:
    """b_thr -> thr, b_alpz_ind -> alpz, b_tune_dqn_c1 -> dqn, b_full_reuse -> full."""
    core = stage[2:] if stage.startswith("b_") else stage
    if core.startswith("tune_"):
        core = core[len("tune_"):].split("_c")[0]
    for suffix in ("_reuse", "_ind"):
        if core.endswith(suffix):
            core = core[: -len(suffix)]
    return core


def excluded_for_stage(stage: str) -> list[str]:
    """EXCL(stage): the OWN files of every B arm other than the stage's own arm."""
    arm = core_of_stage(stage)
    return [f"{CODE_REL}/expb/{OWN_FILES[a]}" for a in OWN_FILES if a != arm]


def manifest_rel(stage: str) -> str:
    return f"{RESULTS_REL}/{stage}/models.sha256"


def model_dir(stage: str, repo: Path | None = None) -> Path:
    return _root(repo) / RESULTS_REL / "models" / stage


# ---------------------------------------------------------------- checks

def check_clean(root: Path) -> None:
    """Gate item 1: no tracked change, no untracked or ignored source or extension
    module, no stray bytecode, no orphan __pycache__ file, under CODE and the log."""
    tracked = set(_git(root, "ls-files", "--", CODE_REL).decode("utf-8").splitlines())
    pathspec = [CODE_REL, LOG_REL, CODE_TOP_GLOB]
    check_index_flags(root, pathspec)
    out = _git(root, "status", "--porcelain", "--untracked-files=all", "--ignored",
               "--", *pathspec).decode("utf-8")
    for line in _log_lines(out):
        if not line:
            continue
        status, path = line[:2], line[3:].strip('"')
        if status not in ("??", "!!"):
            raise GateRefused(f"tracked change in {path}: the test_b gate needs a clean tree")
        problem = _stray_problem(path, tracked)
        if problem:
            raise GateRefused(f"{path}: {problem}")


def check_index_flags(root: Path, pathspec: list[str]) -> None:
    """Refuses skip-worktree and assume-unchanged flags: they hide working-tree edits
    from git status (review F3). Every entry must carry the plain tag H."""
    for line in _log_lines(_git(root, "ls-files", "-v", "--", *pathspec).decode("utf-8")):
        if line and line[0] != INDEX_FLAG_TAG:
            raise GateRefused(f"index flag {line[0]!r} on {line[2:]}: hidden edits are refused")


def _stray_problem(path: str, tracked: set) -> str | None:
    if path.endswith(SOURCE_SUFFIXES):
        return "untracked or ignored source or extension module"
    if not path.endswith((".pyc", ".pyo")):
        return None
    match = BYTECODE_RE.match(path)
    if match is None:
        return "bytecode outside a __pycache__ directory"
    if f"{match['dir']}/{match['mod']}.py" not in tracked:
        return "orphan bytecode whose source is not tracked"
    return None


def pinned_log(root: Path, head: str) -> PinnedLog:
    """Gate item 2 (lines and their order at HEAD, one pinning commit) and the parse."""
    lines = _log_lines(log_text(head, root))
    fam = [i for i, line in enumerate(lines) if line.startswith(FAMILY_PREFIX)]
    if len(fam) != 1:
        raise GateRefused(f"HEAD's log holds {len(fam)} FAMILY lines; exactly one is required")
    family_line = lines[fam[0]]
    match = FAMILY_LINE.fullmatch(family_line)
    if match is None:
        raise GateRefused("the FAMILY line is malformed")
    frozen = _parse_frozen(lines, fam[0])
    check_patches_below(lines, fam[0])
    check_pin_commit(root, head, family_line)
    lists = {k: [x for x in match.group(i + 1).split(",") if x]
             for i, k in enumerate(FAMILY_KEYS)}
    hashes = dict(zip(HASH_KEYS, match.groups()[5:]))
    return PinnedLog(family_line=family_line, lists=lists, hashes=hashes, frozen=frozen)


def _parse_frozen(lines: list[str], family_index: int) -> tuple:
    frozen = []
    for i, line in enumerate(lines):
        if not line.startswith(FROZEN_PREFIX):
            continue
        match = FROZEN_LINE.fullmatch(line)
        if match is None:
            raise GateRefused("a FROZEN (test_b) line is malformed")
        if i > family_index:
            raise GateRefused("a FROZEN (test_b) line appears below the FAMILY line")
        frozen.append((match.group(1), tuple(match.group(2).split(","))))
    return tuple(frozen)


def check_patches_below(lines: list[str], family_index: int) -> None:
    for i, line in enumerate(lines):
        if line.startswith(PATCH_PREFIX):
            if PATCH_LINE.fullmatch(line) is None:
                raise GateRefused("an ANALYSIS PATCH line is malformed")
            if i < family_index:
                raise GateRefused("an ANALYSIS PATCH line appears above the FAMILY line")


def check_pin_commit(root: Path, head: str, family_line: str) -> None:
    """Exactly one commit may ever have added (or removed) a pinned line, and its
    FAMILY line must equal HEAD's byte for byte."""
    # -m: a merge is diffed against each parent, so a merge resolution that edits a
    # pinned line is seen (review F5); each hash is kept once
    out = _git(root, "log", "-m", "--format=%H", "-G", PIN_PATTERN, "--", LOG_REL)
    commits = list(dict.fromkeys(out.decode("utf-8").split()))
    if len(commits) != 1:
        raise GateRefused(f"{len(commits)} commits touched the pinned lines; exactly one "
                          "(the pinning commit) is allowed")
    pinned = [x for x in _log_lines(log_text(commits[0], root))
              if x.startswith(FAMILY_PREFIX)]
    if pinned != [family_line]:
        raise GateRefused("the FAMILY line at HEAD differs from the one pinned in its commit")


def frozen_commit_for(pinned: PinnedLog, stage: str) -> str:
    """Gate item 3: the stage appears in exactly one FROZEN line; return its hash."""
    hits = [commit for commit, stages in pinned.frozen if stage in stages]
    if len(hits) != 1:
        raise GateRefused(f"stage {stage!r} appears in {len(hits)} FROZEN (test_b) lines; "
                          "exactly one is required")
    return hits[0]


def check_ancestor(root: Path, fit_commit: str) -> None:
    """Gate item 4, first half: the fit commit is an ancestor of HEAD."""
    proc = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor",
                           fit_commit, "HEAD"], capture_output=True, check=False)
    if proc.returncode != 0:
        raise GateRefused(f"fit commit {fit_commit} is not an ancestor of HEAD")


def check_fit_blob(root: Path, fit_commit: str, stage: str) -> None:
    """Gate item 4, second half: the fit commit's own log holds no pinned line for it."""
    for line in _log_lines(log_text(fit_commit, root)):
        if line.startswith(FAMILY_PREFIX):
            raise GateRefused("the fit commit's log already holds a FAMILY line")
        match = FROZEN_LINE.fullmatch(line) if line.startswith(FROZEN_PREFIX) else None
        if match and stage in match.group(2).split(","):
            raise GateRefused(f"the fit commit's log already freezes stage {stage!r}")


def check_code(root: Path, head: str, fit_commit: str, stage: str,
               pinned: PinnedLog) -> None:
    """Gate item 5: CODE is unchanged since the manifest commit, and the files the
    stage evaluates equal the files it was fitted with (EXCL(stage) removed)."""
    if tree_hash(head, code_files(head, root)) != pinned.hashes["code_sha256"]:
        raise GateRefused("the code changed after the manifest commit (code_sha256 differs)")
    excl = excluded_for_stage(stage)
    head_files = code_files(head, root, excl)
    fit_files = code_files(fit_commit, root, excl)
    if stage.endswith("_reuse"):
        head_files = _reuse_scope(head_files)
        fit_files = _reuse_scope(fit_files)
    if tree_hash(fit_commit, fit_files, root) != tree_hash(head, head_files, root):
        raise GateRefused(f"the code evaluated for {stage!r} differs from the code it was "
                          "fitted with")


def _reuse_scope(files: list[str]) -> list[str]:
    def rel(path: str) -> str:
        return path[len(CODE_REL) + 1:]
    return [p for p in files if rel(p).startswith(REUSE_PREFIXES)]


def check_manifests(root: Path, head: str, pinned: PinnedLog) -> None:
    """Gate item 6, first half: the manifests at HEAD hash to manifests_sha256."""
    stages = sorted({stage for _, group in pinned.frozen for stage in group})
    digest = _manifests_digest(head, stages, root)
    if digest != pinned.hashes["manifests_sha256"]:
        raise GateRefused("a model manifest differs from the one pinned at the manifest commit")


def _manifests_digest(commit: str, stages: list[str], root: Path | None) -> str:
    lines = sorted(f"{s} {text_hash(commit, manifest_rel(s), root)}\n" for s in stages)
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def check_models(root: Path, head: str, stage: str, fit_commit: str,
                 planned: int) -> None:
    """Gate item 6, second half: every model file of the stage exists, hashes to its
    manifest line, was fitted at ``fit_commit`` on a clean tree; the count is planned."""
    text = _git(root, "show", f"{head}:{manifest_rel(stage)}").decode("utf-8")
    entries = [MODEL_LINE.fullmatch(line.rstrip("\r")) for line in text.splitlines() if line]
    if any(e is None for e in entries):
        raise GateRefused(f"the manifest of {stage!r} has a malformed line")
    if len(entries) != planned:
        raise GateRefused(f"{len(entries)} model files pinned for {stage!r}; {planned} planned")
    for entry in entries:
        digest, name, commit, clean = entry.groups()
        path = model_dir(stage, root) / name
        if not path.exists():
            raise GateRefused(f"model file {name} of {stage!r} is missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise GateRefused(f"model file {name} of {stage!r} differs from its manifest line")
        if commit != fit_commit or clean != "True":
            raise GateRefused(f"model {name} was not fitted at {fit_commit} on a clean tree")
    on_disk = {p.name for p in model_dir(stage, root).glob("*.pkl")}
    pinned = {entry.group(2) for entry in entries}
    if on_disk != pinned:
        raise GateRefused(f"model files on disk for {stage!r} differ from the manifest "
                          f"(extra: {sorted(on_disk - pinned)}, missing: {sorted(pinned - on_disk)})")


def check_family_lists(root: Path, head: str, pinned: PinnedLog, stage: str) -> None:
    """Gate item 7: fidelity_dev.md is the pinned file, and the FAMILY lists follow from
    its derivation block, and the stage's arm is accounted for."""
    if text_hash(head, FIDELITY_REL, root) != pinned.hashes["fidelity_sha256"]:
        raise GateRefused("fidelity_dev.md differs from the file pinned in the FAMILY line")
    block = read_derivation(head, root)
    for key in FAMILY_KEYS:
        if set(block[key]) != set(pinned.lists[key]):
            raise GateRefused(f"FAMILY list {key!r} differs from the derivation block")
    dropped_arms = {item.split(":")[0] for item in pinned.lists["dropped"]}
    allowed = (set(pinned.lists["family"]) | set(pinned.lists["descriptive"])
               | dropped_arms | set(pinned.lists["references"]))
    if stage not in pinned.lists["independence"] and core_of_stage(stage) not in allowed:
        raise GateRefused(f"arm of stage {stage!r} is not in any FAMILY list")


def read_derivation(commit: str, root: Path | None = None) -> dict:
    text = norm(_git(_root(root), "show", f"{commit}:{FIDELITY_REL}")).decode("utf-8")
    match = DERIVATION_RE.search(text)
    if match is None:
        raise GateRefused("fidelity_dev.md has no family-derivation block")
    try:
        block = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise GateRefused(f"family-derivation block is not valid JSON: {exc}") from exc
    if not all(isinstance(block.get(k), list) for k in FAMILY_KEYS):
        raise GateRefused("family-derivation block lacks one of its five lists")
    return block


def library_versions() -> dict:
    import torch
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__}


def check_libraries(root: Path, stage: str) -> None:
    """Gate item 8: the running library versions equal those recorded in every payload
    of the stage (all of them, not a sample; check_models has made the names exact).
    TODO(M21): fit payloads must record "libraries" (sweep.run_fit_job writes none yet),
    so until M21 lands this check refuses every test_b stage (fail closed)."""
    from ..expa import sweep     # noqa: PLC0415 (payload reader; sweep imports torch)
    files = sorted(model_dir(stage, root).glob("*.pkl"))
    if not files:
        raise GateRefused(f"no model payload to read library versions from for {stage!r}")
    expected = library_versions()
    for path in files:
        recorded = sweep.load_payload(path).get("libraries")
        if recorded != expected:
            raise GateRefused(f"library versions of {path.name} differ from the running "
                              f"process: {recorded}")


def planned_job_count(stage: str, root: Path | None = None) -> int:
    """Planned fit jobs of a stage (sweep's plan on the committed eligibility table)."""
    from ..expa import sweep     # noqa: PLC0415
    elig = sweep.stage_eligibility(stage, _root(root) / ELIGIBILITY_REL)
    return len(sweep.plan_stage(stage, elig))


# ---------------------------------------------------------------- verdict

def verify_stage(stage: str, fit_commit: str, repo: Path | None = None) -> tuple[str, str]:
    """Runs gate items 1 to 8. Returns (HEAD, sha256 of the FAMILY line)."""
    root = _root(repo)
    try:
        return _verify(root, stage, fit_commit)
    except ValueError as exc:
        raise GateRefused(str(exc)) from exc


def _verify(root: Path, stage: str, fit_commit: str) -> tuple[str, str]:
    check_clean(root)
    head = _git(root, "rev-parse", "HEAD").decode("utf-8").strip()
    pinned = pinned_log(root, head)
    if frozen_commit_for(pinned, stage) != fit_commit:
        raise GateRefused(f"stage {stage!r} is frozen at another commit, not {fit_commit}")
    check_ancestor(root, fit_commit)
    check_fit_blob(root, fit_commit, stage)
    check_code(root, head, fit_commit, stage, pinned)
    planned = planned_job_count(stage, root)
    check_manifests(root, head, pinned)
    check_models(root, head, stage, fit_commit, planned)
    check_family_lists(root, head, pinned, stage)
    check_libraries(root, stage)
    return head, hashlib.sha256(pinned.family_line.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- producer side

def write_manifest(stage: str, repo: Path | None = None) -> Path:
    """Writes models.sha256 for the stage's model files (sorted lines
    "<sha256> <key>.pkl <fit_commit> <fit_tree_clean>")."""
    from ..expa import sweep     # noqa: PLC0415
    root = _root(repo)
    lines = []
    for path in sorted(model_dir(stage, root).glob("*.pkl")):
        payload = sweep.load_payload(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest} {path.name} {payload['fit_commit']} "
                     f"{payload['fit_tree_clean']}\n")
    out = root / manifest_rel(stage)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(sorted(lines)), encoding="utf-8", newline="\n")
    return out


def family_report(commit: str, repo: Path | None = None) -> str:
    """The FROZEN lines (one per fit commit) and the FAMILY line, computed at ``commit``
    (the manifest commit M). Lists are copied from the derivation block."""
    root = _root(repo)
    block = read_derivation(commit, root)
    stages = sorted(p.split("/")[-2] for p in tracked_under(commit, RESULTS_REL, root)
                    if p.endswith("/models.sha256") and p.count("/") == RESULTS_REL.count("/") + 2)
    by_commit: dict[str, list[str]] = {}
    for stage in stages:
        fits = {m.group(3) for line in _log_lines(_manifest_text(commit, stage, root))
                if (m := MODEL_LINE.fullmatch(line))}
        if len(fits) != 1:
            raise GateRefused(f"stage {stage!r} has model files from {len(fits)} commits")
        by_commit.setdefault(fits.pop(), []).append(stage)
    frozen = [f"FROZEN COMMIT (test_b): {c} stages=[{','.join(sorted(s))}]"
              for c, s in sorted(by_commit.items())]
    hashes = {
        "fidelity_sha256": text_hash(commit, FIDELITY_REL, root),
        "manifests_sha256": _manifests_digest(commit, stages, root),
        "code_sha256": tree_hash(commit, code_files(commit, root), root),
        "analysis_sha256": tree_hash(commit, list(ANALYSIS_RELS), root),
    }
    fam = " ".join([
        "TEST_B FAMILY CLOSED:",
        *(f"{k}=[{','.join(block[k])}]" for k in FAMILY_KEYS),
        *(f"{k}={hashes[k]}" for k in HASH_KEYS),
    ])
    return "\n".join([*frozen, fam])


def _manifest_text(commit: str, stage: str, root: Path) -> str:
    return norm(_git(root, "show", f"{commit}:{manifest_rel(stage)}")).decode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m transvrp.expb.gatecheck")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mp = sub.add_parser("manifest", help="write models.sha256 for one stage")
    mp.add_argument("--stage", required=True)
    fp = sub.add_parser("family-line", help="print FROZEN lines and the FAMILY line at a commit")
    fp.add_argument("--commit", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "manifest":
            print(write_manifest(args.stage))
        else:
            print(family_report(args.commit))
    except (GateRefused, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

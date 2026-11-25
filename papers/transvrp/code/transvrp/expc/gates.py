"""EXPERIMENT C gates (EXPERIMENT_C.md sections 4.3, 5, 6; module MC5).

Three parts:

* ``C_STREAM``: the test_c stream for ``expb.gatecheck`` (the eight checks of B 5.3, with
  C's patterns, owner files and family keys). ``verify_test_c_stage`` runs them.
* ``verify_reuse_stage``: section 4.3 for c_full_w6 and c_ppob_w6, which reuse B's frozen
  models at fit commit bf3edab. Their gate accepts model files that hash to B's manifest,
  matching library versions and the family checks; the prediction identity (4.3 b) is a
  test (tests/test_expc_gates.py) and ``expc.identity_check`` for the bf3edab comparison.
* the dev gates of section 6 and the family derivation of section 1.9, written to
  ``results/expC/fidelity_c_dev.md`` with a ``family-derivation`` block (CLI below).

Arm names: a stage's arm is ``c_core_of(stage)`` (``boost_f1``, ``rlsm_f3``, ``cdp``,
``cdp_true``, ``thr``, ``dos``, ``ppo_long``, ``ppo_b``, ``full``, ``ladder``). Its owner
(EXCL, B 5.3) is the family name. Day arrays are read from ``results/expC/<stage>/rows``
and the policy key of each arm is its family name (``boost``, ``rlsm``, ``cdp``, ``thr``,
``dos``, ``ppo``, ``nn_full``).

Run: ``python -m transvrp.expc.gates --out results/expC/fidelity_c_dev.md`` from the
``papers/transvrp/code`` directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import gen_scenarios, load_plans, stable_seed
from ..expa import paths as P
from ..expa import sweep
from ..expb import fidelity as F
from ..expb import gatecheck as G
from . import cdp as CDP
from . import identity_check as IC
from .stages import WORLD_SUFFIX, c_stage_defs, cdp_tables_of, cdp_train_paths

DELTA = 0.7                  # section 1.6: contrasts against BATON, CDP, DOS
DELTA_Q1 = 0.25              # section 1.6: contrasts between feature sets
CDP0_SHARE = 0.95            # G-CDP0 and G-CDP-node: share of routes
NODE_SHARE = 0.95
NODE_PATHS = 20_000          # G-CDP-node simulated paths per route (section 6)
W6_TAG = "w6"
G_W6_INSTANCE = "CON3-0"     # section 6: the instance of G-W6

REUSE_FIT_COMMIT = "bf3edabd9857644c28ac4d301ea82063950715ff"   # B's fit commit F (4.3)
C_RESULTS_REL = f"{G.PROJECT_REL}/code/results/expC"
C_FIDELITY_REL = f"{C_RESULTS_REL}/fidelity_c_dev.md"
C_ANALYSIS_RELS = tuple(f"{G.CODE_REL}/{p}" for p in (
    "expa/analysis.py", "expb/analysis_b.py", "expb/fidelity.py",
    "expc/analysis_c.py", "expc/gates.py"))
C_FAMILY_KEYS = ("family", "descriptive", "dropped", "references")
REFERENCES = ("baton", "ladder", "ppo_b")
# stages that reuse B's frozen models: stage -> B stage whose manifest and models they use
REUSED = {"c_full_w6": "b_full", "c_ppob_w6": "b_ppo"}
WORLD_TAGS = tuple(WORLD_SUFFIX.values())               # ("w0", "w3", "w6", "w9", "ws")
WORLD_OF_TAG = {tag: world for world, tag in WORLD_SUFFIX.items()}
# arms with a P1 bound in W6 (section 6): every learned and DP arm
P1_KINDS = ("boost", "rlsm", "thr", "cdp", "dos", "ppo", "full")

C_FAMILY_LINE = re.compile(
    r"TEST_C FAMILY CLOSED: family=\[([A-Za-z0-9_,:-]*)\] descriptive=\[([A-Za-z0-9_,:-]*)\] "
    r"dropped=\[([A-Za-z0-9_:,-]*)\] references=\[([A-Za-z0-9_,:-]*)\] "
    r"fidelity_sha256=([0-9a-f]{64}) manifests_sha256=([0-9a-f]{64}) "
    r"code_sha256=([0-9a-f]{64}) analysis_sha256=([0-9a-f]{64})")
C_FROZEN_LINE = re.compile(
    r"FROZEN COMMIT \(test_c\): ([0-9a-f]{40}) stages=\[([A-Za-z0-9_]+(?:,[A-Za-z0-9_]+)*)\]")
C_PATCH_LINE = re.compile(
    r"TEST_C ANALYSIS PATCH: analysis_sha256=([0-9a-f]{64}) date=(\d{4}-\d{2}-\d{2}) "
    r"reason=(.+)")
C_PIN = "^(TEST_C FAMILY CLOSED|FROZEN COMMIT [(]test_c[)])"
C_OWN = {
    "thr": (f"{G.CODE_REL}/expb/thr.py",),
    "boost": (f"{G.CODE_REL}/expc/boost_sets.py",),
    "rlsm": (f"{G.CODE_REL}/expb/rlsm.py", f"{G.CODE_REL}/expc/rlsm_sets.py"),
    "cdp": (f"{G.CODE_REL}/expc/cdp.py",),
    "dos": (f"{G.CODE_REL}/expb/dos.py",),
    "ppo": (f"{G.CODE_REL}/expb/ppo.py",),
}


# ---------------------------------------------------------------- stage names

def _base(stage: str) -> str:
    body = stage[2:] if stage.startswith("c_") else stage
    return re.sub(r"_(" + "|".join(WORLD_TAGS) + r")$", "", body)


def c_core_of(stage: str) -> str:
    """Arm name of a test_c stage: c_boost_F1_w6 -> boost_f1, c_cdptrue_w6 -> cdp_true."""
    body = _base(stage)
    named = {"ladder": "ladder", "thr": "thr", "cdp": "cdp", "cdptrue": "cdp_true",
             "dos": "dos", "ppolong": "ppo_long", "ppob": "ppo_b", "full": "full"}
    if body in named:
        return named[body]
    match = re.fullmatch(r"(boost|rlsm)_F(0n|0|1|2|3)", body)
    if match:
        return f"{match.group(1)}_f{match.group(2)}"
    raise ValueError(f"not a test_c stage: {stage!r}")


def c_owner_of(stage: str) -> str:
    """Owner key of ``C_OWN`` (EXCL, B 5.3): the family of the arm."""
    return c_core_of(stage).split("_")[0]


def c_planned_jobs(stage: str, elig_path: Path | None = None) -> list:
    """Planned fit jobs of a stage (sweep's plan on the committed eligibility table).
    A reused stage plans the B stage whose models it reuses."""
    name = REUSED.get(stage, stage)
    path = Path(elig_path) if elig_path is not None else G.REPO_ROOT / G.ELIGIBILITY_REL
    return sweep.plan_stage(name, sweep.stage_eligibility(name, path))


def c_planned(stage: str, root: Path) -> int:
    return len(c_planned_jobs(stage, Path(root) / G.ELIGIBILITY_REL))


def _reuse_entry(b_stage: str) -> tuple[str, str, str]:
    return (f"{G.RESULTS_REL}/{b_stage}/models.sha256",
            f"{G.RESULTS_REL}/models/{b_stage}", REUSE_FIT_COMMIT)


C_STREAM = G.Stream(
    name="test_c", family_prefix="TEST_C FAMILY CLOSED", family_re=C_FAMILY_LINE,
    frozen_prefix="FROZEN COMMIT (test_c)", frozen_re=C_FROZEN_LINE,
    patch_prefix="TEST_C ANALYSIS PATCH", patch_re=C_PATCH_LINE, pin_pattern=C_PIN,
    family_keys=C_FAMILY_KEYS, results_rel=C_RESULTS_REL, fidelity_rel=C_FIDELITY_REL,
    analysis_rels=C_ANALYSIS_RELS, own=C_OWN, core_of=c_core_of, owner_of=c_owner_of,
    reuse={stage: _reuse_entry(b) for stage, b in REUSED.items()}, planned=c_planned)


def verify_test_c_stage(stage: str, fit_commit: str, repo: Path | None = None) -> tuple[str, str]:
    """Gate items 1 to 8 of B 5.3 for a test_c stage. A reused stage (4.3) also needs the
    bf3edab identity record: missing, nonzero or unbound records refuse the stage."""
    head, sha = G.verify_stage(stage, fit_commit, repo, stream=C_STREAM)
    if stage in REUSED:
        require_identity(G._root(repo), fit_commit)
    return head, sha


def verify_reuse_stage(stage: str, repo: Path | None = None) -> tuple[str, str]:
    """Section 4.3 for c_full_w6 and c_ppob_w6: fit commit bf3edab, B's manifests, the
    library versions of the payloads, and the family checks. Raises GateRefused otherwise.

    Library versions come from the payloads' own libraries record (checked by
    expb.gatecheck.check_libraries). B's payloads carry it (every results/expB model
    pickle has python, numpy, scipy and torch; test_b rows repeat it in their library
    columns), so the CSV columns are not needed and results/expB is not read for them.
    """
    if stage not in REUSED:
        raise G.GateRefused(f"{stage!r} is not a reused stage (reused: {sorted(REUSED)})")
    return verify_test_c_stage(stage, REUSE_FIT_COMMIT, repo)


# ---------------------------------------------------------------- day arrays and helpers

def day_costs(rows_dir: Path, inst: str, route_id: int, N: int, rep: int, policy: str,
              eval_set: str = "dev") -> np.ndarray:
    """Per-day costs of one route-rep and policy. The rows directory of a stage holds one
    file per job; the glob selects the single file of this route-rep."""
    matches = sorted(Path(rows_dir).glob(sweep.day_file_glob(inst, route_id, N, rep, eval_set)))
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} day files for {inst} r{route_id} rep {rep} in {rows_dir}")
    with np.load(matches[0]) as z:
        if policy not in z.files:
            raise ValueError(f"{matches[0].name} holds no policy {policy!r}")
        return np.asarray(z[policy], dtype=float)


def _mean_over_reps(arrays: list) -> np.ndarray:
    return np.mean([np.asarray(a, dtype=float) for a in arrays], axis=0)


def _sha(*arrays) -> str:
    digest = hashlib.sha256()
    for arr in arrays:
        digest.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------- dev gates

def gate_g0(arm_df: pd.DataFrame, oracle_df: pd.DataFrame, planned_keys) -> F.Gate:
    """G0 (section 6): every planned job fitted, every dev cost finite, C >= C_oracle."""
    arm = F._keyed(arm_df)
    oracle = F._keyed(oracle_df)
    planned = {tuple(k) for k in planned_keys}
    fitted = bool(planned) and set(arm) == planned
    finite = all(math.isfinite(v) for v in arm.values())
    no_oracle = sorted(set(arm) - set(oracle))
    worse = [k for k in arm if k in oracle
             and arm[k] < oracle[k] - F.EXACT_TOL * max(1.0, abs(oracle[k]))]
    passed = fitted and finite and not no_oracle and not worse
    detail = (f"fitted {len(set(arm) & planned)}/{len(planned)} planned; "
              f"{len(worse)} below oracle; {len(no_oracle)} without oracle row")
    return F.Gate("g0", passed, None, "all planned jobs fitted; finite; C >= C_oracle", detail)


def w6_identity(inst: str = G_W6_INSTANCE) -> dict:
    """G-W6: the W6 train rep 0 chunk 0 and the dev stream of ``inst``, generated by the
    world-aware code with W6 and by A's original call (no world argument), hash equal."""
    sol = load_plans(inst)
    dbar, pbar = sol["dbar"], sol["pbar"]
    base = stable_seed(inst)
    a_train = gen_scenarios(dbar, pbar, P.CHUNK, [base, 1, 0, 0])
    a_dev = gen_scenarios(dbar, pbar, P.DEV_N, [base, 3])
    c_train = P._chunk_arrays(inst, "train", 0, 0, None, None, "W6")
    c_dev = P._chunk_arrays(inst, "dev", 0, 0, None, None, "W6")
    return {"train_a": _sha(*a_train), "train_c": _sha(*c_train),
            "dev_a": _sha(*a_dev), "dev_c": _sha(*c_dev)}


def gate_w6(inst: str = G_W6_INSTANCE) -> F.Gate:
    hashes = w6_identity(inst)
    passed = hashes["train_a"] == hashes["train_c"] and hashes["dev_a"] == hashes["dev_c"]
    return F.Gate("g_w6", passed, None, f"W6 train rep0 chunk0 and dev of {inst} equal A's",
                  f"train {hashes['train_c'][:12]} vs {hashes['train_a'][:12]}; "
                  f"dev {hashes['dev_c'][:12]} vs {hashes['dev_a'][:12]}")


def gate_cdp0(cdp_days: dict, thr_days: dict, e0: dict, share_min: float = CDP0_SHARE) -> F.Gate:
    """G-CDP0 (W0): per route, the replicate-averaged CDP and THR day costs satisfy
    |mean diff| <= 4 SE + 1e-3 E0 (CDP.g_cdp0_within). Keys: (inst, route_id) -> list of
    per-replicate day arrays."""
    routes = sorted(set(cdp_days) & set(thr_days))
    if not routes:
        return F.Gate("g_cdp0", False, None, "CDP vs THR per route", "no common routes")
    ok = [CDP.g_cdp0_within(_mean_over_reps(cdp_days[k]), _mean_over_reps(thr_days[k]), e0[k])
          for k in routes]
    share = float(np.mean(ok))
    return F.Gate("g_cdp0", share >= share_min, share,
                  f"CDP vs THR within 4 SE + 1e-3 E0 on >= {share_min:.0%} of routes",
                  f"{sum(ok)}/{len(ok)} routes")


def gate_cdp_node(payloads: list, spec_of, train_of, share_min: float = NODE_SHARE,
                  n: int = NODE_PATHS) -> F.Gate:
    """G-CDP-node: for every route (rep 0), CDP.node_check at the node (0.5, -0.5) within
    its tolerance on >= share_min of routes. Payloads keep no tables: the tables are rebuilt
    from the stored model and ``train_of(job)`` (the fit's training paths) by
    ``stages.cdp_tables_of``, the function the eval uses. A payload that is not a CDP
    payload fails the gate (fail closed)."""
    results = []
    for p in payloads:
        job = p["job"]
        blob = p.get("models")
        if not isinstance(blob, dict) or blob.get("arm") != "cdp" or "cdp_model" not in blob:
            return F.Gate("g_cdp_node", False, None, "CDP node check",
                          "payloads are not CDP payloads (no cdp_model)")
        spec = spec_of(job.inst, job.route_id)
        tables = cdp_tables_of(blob, spec, train_of(job))
        res = CDP.node_check(spec, blob["cdp_model"], tables, n=n, seed=0)
        results.append(bool(res["ok"]))
    if not results:
        return F.Gate("g_cdp_node", False, None, "CDP node check", "no rep-0 payloads")
    share = float(np.mean(results))
    return F.Gate("g_cdp_node", share >= share_min, share,
                  f"node check within tolerance on >= {share_min:.0%} of routes",
                  f"{sum(results)}/{len(results)} routes")


def gate_from_fidelity(b_stage: str, arm: str, gate_id: str, dev_df: pd.DataFrame,
                       models_dir, rows_dir) -> F.Gate:
    """G-DOS / G-PPO: B's own gate (fidelity.gate_report), one gate extracted by id."""
    report = F.gate_report(b_stage, arm, dev_df, models_dir, rows_dir)
    match = next(g for g in report["gates"] if g["id"] == gate_id)
    return F.Gate(gate_id, bool(match["passed"]), match["value"], match["threshold"],
                  match["detail"])


def gate_p1(jobs: list, rows_dir, policy: str) -> F.Gate:
    """P1 (B 7), by B's implementation (fidelity._gate_p1) on the W6 dev stream."""
    ctx = F._Ctx(stage="p1", arm=policy, jobs=jobs, source=F.DevSource(),
                 rows_dir=Path(rows_dir), independent=False, g2_keys=frozenset(),
                 thr_dir=None, info={}, dos_tol=F.DOS_TOL, passes=None)
    return F._gate_p1(ctx)


# ---------------------------------------------------------------- reports and family

@dataclass(frozen=True)
class Contrast:
    cid: str
    a_stage: str
    a_policy: str
    b_stage: str
    b_policy: str
    margin: float


CONTRASTS = (
    Contrast("C1-B", "c_boost_F1_w6", "boost", "c_boost_F0_w6", "boost", DELTA_Q1),
    Contrast("C1-R", "c_rlsm_F1_w6", "rlsm", "c_rlsm_F0_w6", "rlsm", DELTA_Q1),
    Contrast("C2-B", "c_boost_F1_w6", "boost", "c_boost_F0n_w6", "boost", DELTA_Q1),
    Contrast("C2-R", "c_rlsm_F1_w6", "rlsm", "c_rlsm_F0n_w6", "rlsm", DELTA_Q1),
    Contrast("C3-B", "c_boost_F2_w6", "boost", "c_boost_F1_w6", "boost", DELTA_Q1),
    Contrast("C3-R", "c_rlsm_F2_w6", "rlsm", "c_rlsm_F1_w6", "rlsm", DELTA_Q1),
    Contrast("C4-B", "c_boost_F3_w6", "boost", "c_boost_F2_w6", "boost", DELTA_Q1),
    Contrast("C4-R", "c_rlsm_F3_w6", "rlsm", "c_rlsm_F2_w6", "rlsm", DELTA_Q1),
    Contrast("C5", "c_cdp_w6", "cdp", "c_ladder_w6", "baton_eq", DELTA),
    Contrast("C6", "c_full_w6", "nn_full", "c_cdp_w6", "cdp", DELTA),
    Contrast("C7", "c_full_w6", "nn_full", "c_dos_w6", "dos", DELTA),
)


def derive_c_family(reports: dict, contrasts=CONTRASTS) -> dict:
    """Section 1.9: a contrast leaves the family (to descriptive) when any of its W6 stages
    fails a gate; ``dropped`` lists every failing ``arm:gate`` of every report."""
    family, descriptive = [], []
    for c in contrasts:
        for stage in (c.a_stage, c.b_stage):
            if stage not in reports and stage not in ("c_ladder_w6",):
                raise ValueError(f"no gate report for {stage} (contrast {c.cid})")
        stages = {c.a_stage, c.b_stage} & set(reports)
        if any(not reports[s]["passed"] for s in stages):
            descriptive.append(c.cid)
        else:
            family.append(c.cid)
    dropped = sorted({item for r in reports.values() for item in r["dropped"]})
    arms = {}
    for stage, r in sorted(reports.items()):
        entry = arms.setdefault(r["arm"], {"stages": [], "passed": True})
        entry["stages"].append(stage)
        entry["passed"] = entry["passed"] and bool(r["passed"])
    return {"family": family, "descriptive": descriptive, "dropped": dropped,
            "references": list(REFERENCES), "arms": arms,
            "stages": {s: {"arm": r["arm"], "world": r["world"], "passed": r["passed"],
                           "gates": {g["id"]: {"passed": g["passed"], "value": g["value"]}
                                     for g in r["gates"]}}
                       for s, r in sorted(reports.items())}}


def make_report(stage: str, world: str, kind: str, gates: list[F.Gate]) -> dict:
    """One stage's report. ``dropped`` lists ``arm:gate`` for each failing gate."""
    arm = c_core_of(stage)
    return {"stage": stage, "arm": arm, "world": world, "kind": kind,
            "gates": [g.as_dict() for g in gates],
            "passed": all(g.passed for g in gates),
            "dropped": [f"{arm}:{g.id}" for g in gates if not g.passed]}


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.6g}"


def render_c_report(reports: dict, lists: dict, extra: dict | None = None) -> str:
    """fidelity_c_dev.md: the gate table, the family lists, and the ``family-derivation``
    block that ``gatecheck`` parses back (EXPERIMENT_C 5)."""
    lines = ["# Fidelity on dev for EXPERIMENT C (EXPERIMENT_C.md section 6)", "",
             "Dev rows only (eval_set dev). Generated by `python -m transvrp.expc.gates`.",
             "", "| stage | arm | gate | verdict | value | threshold | detail |",
             "|---|---|---|---|---|---|---|"]
    for stage in sorted(reports):
        r = reports[stage]
        for g in r["gates"]:
            verdict = "pass" if g["passed"] else "FAIL"
            lines.append(f"| {stage} | {r['arm']} | {g['id']} | {verdict} | {_fmt(g['value'])} | "
                         f"{str(g['threshold']).replace('|', '/')} | "
                         f"{str(g['detail']).replace('|', '/')} |")
    lines += ["", "## Family derivation", "",
              f"family={lists['family']}; descriptive={lists['descriptive']}; "
              f"dropped={lists['dropped']}; references={lists['references']}", ""]
    for name, value in sorted((extra or {}).items()):
        lines.append(f"- {name}: {json.dumps(value, sort_keys=True)}")
    block = json.dumps(lists, sort_keys=True, indent=1)
    lines += ["", "```family-derivation", block, "```", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- 4.3 (b): bf3edab identity

IDENTITY_REL = f"{C_RESULTS_REL}/identity_bf3edab.json"
IDENTITY_STAGES = tuple(REUSED.values())         # ("b_full", "b_ppo")
IDENTITY_MIN_ROUTES = 3
IDENTITY_MIN_INSTANCES = 3
IDENTITY_ROUTE_CANDIDATES = (("CON3-0", 0), ("CON3-5", 0), ("SCA3-0", 0), ("SCA3-5", 0))


@contextmanager
def _worktree(root: Path, commit: str, scratch: Path):
    """A detached git worktree of ``commit`` outside the repository tree, removed on exit."""
    path = Path(scratch) / f"wt_{commit[:12]}"
    G._git(root, "worktree", "add", "--detach", str(path), commit)
    try:
        yield path
    finally:
        G._git(root, "worktree", "remove", "--force", str(path), check=False)


def _predict_in_worktree(wt: Path, script: Path, stage: str, inst: str, route_id: int,
                         payloads: Path, out: Path) -> Path:
    """Runs the identity script with the worktree's code tree on the given payloads."""
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415
    code = wt / G.PROJECT_REL / "code"
    env = dict(os.environ, PYTHONPATH=str(code))
    subprocess.run([sys.executable, str(script), "predict", "--stage", stage, "--inst", inst,
                    "--route", str(route_id), "--code", str(code), "--payloads", str(payloads),
                    "--out", str(out)], check=True, env=env, cwd=str(code))
    return Path(out)


def identity_routes(payload_root: Path) -> list[tuple[str, int]]:
    """The first rep-0 route of each candidate instance that has a payload for both stages."""
    chosen = []
    for inst, _ in IDENTITY_ROUTE_CANDIDATES:
        ids = []
        for stage in IDENTITY_STAGES:
            names = [p.name for p in (Path(payload_root) / stage).glob(f"*__{inst}__r*__rep0.pkl")]
            ids.append({int(n.split("__r")[1].split("__")[0]) for n in names})
        common = sorted(set.intersection(*ids)) if ids else []
        if common:
            chosen.append((inst, common[0]))
    return chosen


def _max_diff(diffs: dict) -> float:
    if "keys" in diffs:
        raise G.GateRefused(f"prediction keys differ: {diffs['keys']}")
    return max(diffs.values(), default=0.0)


def write_identity_record(out: Path | None = None, scratch: Path | None = None,
                          repo: Path | None = None) -> dict:
    """EXPERIMENT_C 4.3 (b): predictions of B's FULL and PPO payloads on dev routes at
    bf3edab (a worktree) and at the current tree, compared exactly. Writes the record: per
    stage and route the max abs diff per key and the identity flag, both commits, the code
    hashes, and whether the tree was clean. Run by hand; the gate reads the file."""
    import tempfile  # noqa: PLC0415
    root = G._root(repo)
    scratch = Path(scratch or tempfile.mkdtemp(prefix="identity_"))
    scratch.mkdir(parents=True, exist_ok=True)
    out = Path(out) if out is not None else root / IDENTITY_REL
    head = G._git(root, "rev-parse", "HEAD").decode("utf-8").strip()
    payload_root = root / G.RESULTS_REL / "models"
    routes = identity_routes(payload_root)
    if len({inst for inst, _ in routes}) < IDENTITY_MIN_INSTANCES:
        raise G.GateRefused(f"payloads cover {len(routes)} candidate routes; too few instances")
    script = scratch / "identity_check_at_b.py"
    script.write_bytes(Path(IC.__file__).read_bytes())
    stages = {}
    with _worktree(root, REUSE_FIT_COMMIT, scratch) as wt:
        for stage in IDENTITY_STAGES:
            per_route = []
            for inst, rid in routes:
                print(f"identity {stage} {inst} r{rid}: predicting at bf3edab", flush=True)
                a = _predict_in_worktree(wt, script, stage, inst, rid, payload_root,
                                         scratch / f"a_{stage}_{inst}_{rid}.npz")
                print(f"identity {stage} {inst} r{rid}: predicting at the current tree",
                      flush=True)
                b = IC.save(IC.predict_stage_route(payload_root / stage, inst, rid),
                            scratch / f"b_{stage}_{inst}_{rid}.npz")
                same, diffs = IC.compare(a, b)
                per_route.append({"inst": inst, "route_id": rid, "rep": 0,
                                  "identical": bool(same), "max_abs_diff": _max_diff(diffs),
                                  "keys": diffs})
            stages[stage] = {"max_abs_diff": max(r["max_abs_diff"] for r in per_route),
                             "per_route": per_route}
    dirty = bool(G._git(root, "status", "--porcelain", "--", G.CODE_REL).strip())
    record = {
        "rule": "EXPERIMENT_C 4.3(b)", "tolerance": 0, "split": "dev", "n_rows": IC.N_ROWS,
        "identical": all(r["identical"] for s in stages.values() for r in s["per_route"]),
        "commit_b": REUSE_FIT_COMMIT, "commit_c": head, "tree_clean": not dirty,
        "code_sha256_b": G.tree_hash(REUSE_FIT_COMMIT, G.code_files(REUSE_FIT_COMMIT, root), root),
        "code_sha256_c": G.tree_hash(head, G.code_files(head, root), root),
        "routes": [{"inst": i, "route_id": r, "rep": 0} for i, r in routes],
        "stages": stages,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8",
                   newline="\n")
    print(out, flush=True)
    return record


def check_identity_record(rec: dict, root: Path, fit_commit: str) -> dict:
    """Refuses unless the record shows the bf3edab identity: exact equality on every key of
    every route (recomputed from the per-key values, not the summary), at least three
    routes from three instances, both reused stages, bound to B's fit commit, and to a
    clean C commit that is an ancestor of HEAD, whose code hash it states."""
    def refuse(why: str):
        raise G.GateRefused(f"identity record {IDENTITY_REL}: {why}")

    if rec.get("commit_b") != fit_commit:
        refuse(f"commit_b is not {fit_commit}")
    if rec.get("identical") is not True:
        refuse("the record is not marked identical")
    stages = rec.get("stages", {})
    if set(stages) != set(IDENTITY_STAGES):
        refuse(f"stages {sorted(stages)} are not {sorted(IDENTITY_STAGES)}")
    routes = rec.get("routes", [])
    if len(routes) < IDENTITY_MIN_ROUTES or len({r["inst"] for r in routes}) < IDENTITY_MIN_INSTANCES:
        refuse("fewer than three routes from three instances")
    for stage in IDENTITY_STAGES:
        per_route = stages[stage].get("per_route", [])
        if len(per_route) != len(routes):
            refuse(f"{stage} has {len(per_route)} route entries, not {len(routes)}")
        for item in per_route:
            diffs = list(item.get("keys", {}).values())
            if not diffs or any(d != 0.0 for d in diffs):
                refuse(f"{stage} has a nonzero or missing prediction difference")
        if any(item.get("identical") is not True for item in per_route):
            refuse(f"{stage} has a route that is not identical")
        if stages[stage].get("max_abs_diff") != 0.0:
            refuse(f"{stage} summary max_abs_diff is nonzero")
    if rec.get("tree_clean") is not True:
        refuse("it was measured on uncommitted code; rerun after the commit")
    commit_c = rec.get("commit_c", "")
    G.check_ancestor(root, commit_c)
    if rec.get("code_sha256_c") != G.tree_hash(commit_c, G.code_files(commit_c, root), root):
        refuse("code_sha256_c differs from the code of commit_c")
    if rec.get("code_sha256_b") != G.tree_hash(fit_commit, G.code_files(fit_commit, root), root):
        refuse("code_sha256_b differs from the code of the B fit commit")
    return rec


def require_identity(root: Path, fit_commit: str) -> dict:
    """Reads the identity record of the working tree; refuses when it is missing."""
    path = Path(root) / IDENTITY_REL
    if not path.exists():
        raise G.GateRefused(f"identity record {IDENTITY_REL} is missing (EXPERIMENT_C 4.3 b)")
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise G.GateRefused(f"identity record is not valid JSON: {exc}") from exc
    return check_identity_record(rec, root, fit_commit)


# ---------------------------------------------------------------- CLI: build the dev report

def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"inst": str})


def fitted_stage_plan() -> list[tuple[str, str, str, str]]:
    """(stage, world tag, kind, policy) of every fitted C stage, read from
    ``stages.c_stage_defs()`` (the spec table): the eval-only stages (reused models) and
    the ladder are excluded. The kind is the arm; the policy is the arm's row name."""
    rows = []
    for name, d in c_stage_defs().items():
        if d["eval_only"] or not d["arms"] or name in REUSED or name.startswith("c_ladder_"):
            continue
        arm = d["arms"][0]
        rows.append((name, WORLD_SUFFIX[d["world"]], arm, sweep.ARM_POLICY[arm]))
    return rows


def cdp0_inputs(results: Path, dev_cdp: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Per-route, per-replicate day arrays of CDP and THR in W0 (dev), and E0 per route."""
    n = int(dev_cdp["N"].iloc[0])
    cdp_dir, thr_dir = results / "c_cdp_w0" / "rows", results / "c_thr_w0" / "rows"
    cdp_days, thr_days, e0 = {}, {}, {}
    for (inst, route_id), grp in dev_cdp.groupby(["inst", "route_id"]):
        key = (str(inst), int(route_id))
        reps = sorted(int(r) for r in grp["rep"].unique())
        cdp_days[key] = [day_costs(cdp_dir, key[0], key[1], n, r, "cdp") for r in reps]
        thr_days[key] = [day_costs(thr_dir, key[0], key[1], n, r, "thr") for r in reps]
        e0[key] = float(sweep._spec_of(key[0], key[1]).E0)
    return cdp_days, thr_days, e0


def required_tables(stage: str, tag: str, results: Path) -> list[Path]:
    """Files a stage's dev gates read: its merged rows, the ladder rows of its world, its
    models, and (for the W0 CDP gate) THR's W0 merged rows. Missing ones make the stage
    ``missing`` in the report, not an error."""
    paths = [results / f"{stage}.csv", results / f"c_ladder_{tag}.csv",
             results / "models" / stage]
    if stage == "c_cdp_w0":
        paths.append(results / "c_thr_w0.csv")
    return paths


def build_reports(results: Path, expb: Path, elig_path: Path | None = None) -> tuple[dict, dict]:
    """Every dev gate of section 6 from the committed dev tables. Returns (reports,
    extra): extra holds the W6 identity hashes and ``missing``, the fitted stages whose
    inputs are absent (not gated; a partial run is reported, never a crash)."""
    reports: dict = {}
    extra = {"g_w6": w6_identity(), "missing": []}
    for stage, tag, kind, policy in fitted_stage_plan():
        if not all(p.exists() for p in required_tables(stage, tag, results)):
            extra["missing"].append(stage)
            continue
        dev = _read(results / f"{stage}.csv")
        ladder = _read(results / f"c_ladder_{tag}.csv")
        models = results / "models" / stage
        rows = results / stage / "rows"
        jobs = F.load_jobs(models)
        planned = [F._key(j) for j in c_planned_jobs(stage, elig_path)]
        arm_df = dev[dev["policy"] == policy]
        oracle_df = ladder[ladder["policy"] == "oracle"]
        gates = [gate_g0(arm_df, oracle_df, planned)]
        if tag == "w6" and kind in P1_KINDS:
            gates.append(gate_p1(jobs, rows, policy))
        if kind == "cdp":
            node_jobs = [p for p in jobs if p["job"].rep == 0]
            world = WORLD_OF_TAG[tag]
            gates.append(gate_cdp_node(node_jobs, lambda i, r: sweep._spec_of(i, r),
                                       lambda job: cdp_train_paths(job, world)))
        if stage == "c_cdp_w0":
            gates.append(gate_cdp0(*cdp0_inputs(results, dev[dev["policy"] == "cdp"])))
        both = pd.concat([dev, ladder], ignore_index=True)
        if stage == "c_dos_w6":
            gates.append(gate_from_fidelity("b_dos", "dos", "b_dos", both, models, rows))
        if stage == "c_ppolong_w6":
            gates.append(gate_from_fidelity("b_ppo", "ppo", "b_ppo", both, models, rows))
        reports[stage] = make_report(stage, WORLD_OF_TAG[tag], kind, gates)
    reports.update(_reused_reports(expb))
    extra["missing"] = sorted(extra["missing"])
    return reports, extra


def _reused_reports(expb: Path) -> dict:
    """c_full_w6: P1 of FULL on B's dev rows and models (the same models, 4.3). c_ppob_w6
    is eval-only and carries no C gate (its B-PPO gate is B's, in B's family)."""
    jobs = F.load_jobs(expb / "models" / "b_full")
    gates = [gate_p1(jobs, expb / "b_full" / "rows", "nn_full")]
    return {"c_full_w6": make_report("c_full_w6", "W6", "full", gates)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m transvrp.expc.gates")
    ap.add_argument("--results", default=str(G.REPO_ROOT / C_RESULTS_REL))
    ap.add_argument("--expb", default=str(G.REPO_ROOT / G.RESULTS_REL))
    ap.add_argument("--out", default=None)
    ap.add_argument("--identity", action="store_true",
                    help="write the 4.3 (b) bf3edab identity record and stop")
    ap.add_argument("--identity-out", default=None)
    ap.add_argument("--scratch", default=None)
    args = ap.parse_args(argv)
    if args.identity:
        write_identity_record(Path(args.identity_out) if args.identity_out else None,
                              Path(args.scratch) if args.scratch else None)
        return 0
    results, expb = Path(args.results), Path(args.expb)
    reports, extra = build_reports(results, expb)
    decidable = [c for c in CONTRASTS
                 if all(s in reports or s == "c_ladder_w6" for s in (c.a_stage, c.b_stage))]
    lists = derive_c_family(reports, contrasts=tuple(decidable))
    extra["undecided_contrasts"] = [c.cid for c in CONTRASTS if c not in decidable]
    out = Path(args.out) if args.out else results / "fidelity_c_dev.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_c_report(reports, lists, extra=extra), encoding="utf-8",
                   newline="\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

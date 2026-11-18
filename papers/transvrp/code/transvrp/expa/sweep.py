"""Fit / eval / merge runner for EXPERIMENT A (module M9).

EXPERIMENT_A.md sections 5, 7.2, 7.3 and M9.

    python -m transvrp.expa.sweep fit   --stage {smoke,pilot,main,big,boost} [--eligibility CSV] [--jobs N]
    python -m transvrp.expa.sweep eval  --stage ... --eval {dev,test} [--frozen-commit HASH] [--jobs N]
    python -m transvrp.expa.sweep merge --stage ... [--eval dev|test]

Layout under ``--results-dir`` (default ``code/results/expA``):

* ``models/<stage>/<key>.pkl``: one fitted object per job, written by fit.
  Neural nets are stored as state dicts. Each file records the fit commit.
* ``<stage>/rows/<key>.<eval_set>.csv`` and ``.npz``: one job's evaluation,
  written by eval. Only one writer per file.
* ``<stage>.csv`` (dev) or ``<stage>.test.csv`` (test): concatenation written by
  merge, with ``gap_closed`` filled in.

Fit and eval are resumable: an existing model file or rows file is skipped.
Eval never refits and never writes models, so model mtimes are unchanged.

Test-stream gate (section 5). The gate lives in the data layer
(``paths.verify_test_gate``), so ``route_pool(inst, "test", frozen=...)`` and
``run_eval_job(..., frozen=...)`` refuse without it. ``--eval test`` also needs a
stage that allows test evaluation (smoke and pilot do not). A verified hash must
be a full 40-character hex commit recorded in RESEARCH_LOG.md, equal to the
running checkout's commit, with a clean code tree (no tracked change and no
untracked .py file); every model loaded for test must carry that hash as its fit
commit and must have been fitted on a clean tree (``fit_tree_clean``, recorded at
fit time). The checks run before any data is generated. Private loaders take
test-purpose data only with the ``GateToken`` that ``verify_test_gate`` returns.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from functools import partial  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from .baselines import classical_ladder, eval_classical  # noqa: E402
from .boost import BoostPredictor, fit_boost  # noqa: E402
from .eligibility import build_eligibility, stratum_of  # noqa: E402
from .features import RouteCache, RouteStats  # noqa: E402
from .model import MLPNet, SetNet  # noqa: E402
from .paths import (TEST_STREAMS, Paths, RouteSpec, TestStreamGateError,  # noqa: E402
                    current_commit, route_pool, route_specs, tree_is_clean,
                    verify_stream_gate, verify_test_gate)
from .measure import Stopwatch, process_peak_rss_mb, traced_peak  # noqa: E402
from .simulate import simulate_policy, summarize  # noqa: E402
from .train import ConstantPredictor, NetPredictor, StageConfig, fit_neural  # noqa: E402
from ..expb import codecs  # noqa: E402
from ..expb.curves import CURVE_ARMS, fit_pass_curves  # noqa: E402
from ..expb.gatecheck import library_versions, write_manifest  # noqa: E402
from ..expb.registry import ARM_POLICY_B, ARM_SPECS, B_ARMS, fit_arm  # noqa: E402
from ..expb.stages import b_stage_defs  # noqa: E402
from ..expb.thr import permute_columns  # noqa: E402

EXPA_DIR = Path(__file__).resolve().parent
CODE_DIR = EXPA_DIR.parents[1]
DEFAULT_RESULTS = CODE_DIR / "results" / "expA"
SMOKE_INSTANCE = "CON3-0"
OTR21_MIN_N = 10_000
LADDER_LABELS = ("none", "tuned_tau", "baton_eq", "oracle")
BIG_LABELS = ("baton_eq", "dp", "none", "oracle")
BIG_RENAME = {"baton_eq": "baton_big", "dp": "dp_big"}
ARM_POLICY = {"full": "nn_full", "scalar": "nn_scalar", "mlp": "nn_mlp",
              "boost": "boost"}
ARM_POLICY.update(ARM_POLICY_B)
EVAL_SETS = ("dev", "test", "test_b")
TIMING_INST, TIMING_ROUTE = "CON3-0", 3   # section 7.3: the one full-size timing route
ROW_COLUMNS = ["inst", "route_id", "m", "stratum", "N", "rep", "policy",
               "mean_cost", "handoff_rate", "fail_rate", "gap_closed", "n_eval",
               "eval_set", "fit_seconds", "fit_commit", "eval_commit"]
KEY_COLUMNS = ["inst", "route_id", "N", "rep", "policy", "eval_set"]
# EXPERIMENT_B 5.3: every test_b row also carries the family hash, the model hash and the
# library versions of its fit (dev and A rows keep ROW_COLUMNS).
TEST_B_COLUMNS = ROW_COLUMNS + ["family_sha256", "model_sha256", "python", "numpy",
                                "scipy", "torch"]
SMOKE_LIKE = ("smoke", "pilot", "b_smoke", "b_pilot")   # in-memory CON3-0 eligibility


GateError = TestStreamGateError     # refusals of the test-stream gate


@dataclass(frozen=True)
class Job:
    kind: str                 # "ladder" | "big" | "neural"
    inst: str
    route_id: int
    N: int
    rep: int
    arm: str = ""             # neural arm: full | scalar | mlp
    labels: tuple = ()        # classical fit labels (ladder and big jobs)
    otr21: bool = False

    @property
    def key(self) -> str:
        tag = self.arm if self.kind == "neural" else self.kind
        return f"{tag}__{self.inst}__r{self.route_id}__N{self.N}__rep{self.rep}"


@dataclass(frozen=True)
class StageSpec:
    reps: tuple
    ladder_N: tuple = ()
    ladder_labels: tuple = LADDER_LABELS
    ladder_otr21: bool = False
    big: bool = False
    big_N: int = 0
    neural: tuple = ()        # (arm, N) pairs, every route and rep
    timing: tuple = ()        # (arm, N) pairs, route 3 of CON3-0 only, rep 0
    routes: str = "eligible"  # eligible | subset | smoke
    cfg: StageConfig = field(default_factory=StageConfig)
    test_allowed: bool = True
    test_stream: str = "test"     # the one gated stream this stage may evaluate on
    arm_cfg: tuple = ()           # EXPERIMENT_B (M21): ((arm, ((field, value), ...)), ...)
    independent: bool = False     # EXPERIMENT_B 3.8: stop columns permuted
    pass_curve: int = 0           # B tuning: passes of one DQN/PPO run, each scored (6.3)
    reuse_models: str = ""        # B cut (i): A's model directory, relative to results_dir's parent


STAGE_SPECS: dict[str, StageSpec] = {
    "smoke": StageSpec(reps=(0,), ladder_N=(100,), big=True, big_N=2000,
                       neural=(("full", 100), ("scalar", 100), ("mlp", 100)),
                       routes="smoke", cfg=StageConfig(s_max=10),
                       test_allowed=False),
    "pilot": StageSpec(reps=(0, 1, 2), ladder_N=(1000, 10000),
                       ladder_labels=("none", "baton_eq", "oracle"),
                       neural=(("full", 1000), ("scalar", 1000),
                               ("full", 10000), ("scalar", 10000)),
                       timing=(("full", 100000),), routes="eligible",
                       test_allowed=False),
    "main": StageSpec(reps=(0, 1, 2), ladder_N=(100, 500, 2000, 10000, 100000),
                      ladder_otr21=True, big=True, big_N=10 ** 6,
                      neural=(("full", 100), ("full", 500), ("full", 2000),
                              ("full", 10000), ("full", 100000),
                              ("scalar", 2000), ("scalar", 100000),
                              ("mlp", 100000)),
                      routes="eligible"),
    "big": StageSpec(reps=(0,), big=True, big_N=10 ** 6,
                     neural=(("full", 10 ** 6), ("scalar", 10 ** 6)),
                     routes="subset"),
    # M11: run only after a WIN (section 1.5). Not in the M9 list; added so that
    # the boosted arm goes through the same fit, eval and merge path.
    "boost": StageSpec(reps=(0, 1, 2), neural=(("boost", 100000),),
                       routes="eligible"),
}
STAGE_SPECS.update({name: StageSpec(**kw) for name, kw in b_stage_defs().items()})


def stage_spec(stage: str) -> StageSpec:
    if stage not in STAGE_SPECS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {tuple(STAGE_SPECS)}")
    return STAGE_SPECS[stage]


@dataclass(frozen=True)
class Ctx:
    results_dir: Path
    stage: str
    cfg: StageConfig
    commit: str
    chunk_dir: Path | None = None   # on-disk chunk cache; default <results_dir>/chunks

    @property
    def models_dir(self) -> Path:
        reuse = STAGE_SPECS[self.stage].reuse_models if self.stage in STAGE_SPECS else ""
        if reuse:
            return self.results_dir.parent / reuse
        return self.results_dir / "models" / self.stage

    @property
    def chunks_dir(self) -> Path:
        return chunk_cache_dir(self)

    @property
    def rows_dir(self) -> Path:
        return self.results_dir / self.stage / "rows"


# ---------------------------------------------------------------- planning

def _route_rows(routes: str, elig: pd.DataFrame) -> list[tuple[str, int]]:
    if routes == "smoke":
        el = elig[elig["eligible"].astype(bool)].sort_values(["m", "route_id"])
        return [(str(el.iloc[0]["inst"]), int(el.iloc[0]["route_id"]))] if len(el) else []
    if routes == "eligible":
        el = elig[elig["eligible"].astype(bool)]
    elif routes == "subset":
        el = elig[elig["in_1e6_subset"].astype(bool)]
    else:
        raise ValueError(f"unknown route selection {routes!r}")
    el = el.sort_values(["inst", "route_id"])
    return [(str(r.inst), int(r.route_id)) for r in el.itertuples()]


FILTER_FIELDS = ("kind", "inst", "route_id", "N", "rep", "arm")
FILTER_ALIASES = {"route": "route_id"}
_INT_FIELDS = ("route_id", "N", "rep")


def parse_only(text: str | None) -> dict | None:
    """``--only`` filter: ``arm=full,inst=CON3-0,route=0,N=10000,rep=0`` becomes an
    exact-match dict on Job fields (``route`` is an alias of ``route_id``). None when
    no filter is given. Raises ValueError on unknown fields or malformed items."""
    if text is None or not text.strip():
        return None
    out: dict = {}
    for item in text.split(","):
        if "=" not in item:
            raise ValueError(f"--only item {item!r} is not field=value")
        name, value = (part.strip() for part in item.split("=", 1))
        name = FILTER_ALIASES.get(name, name)
        if name not in FILTER_FIELDS:
            raise ValueError(f"--only field {name!r} is not one of {FILTER_FIELDS}")
        if name in _INT_FIELDS:
            try:
                value = int(value)
            except ValueError as exc:
                raise ValueError(f"--only {name} must be an integer, got {value!r}") from exc
        out[name] = value
    return out


def job_fields(job: Job) -> dict:
    return dict(kind=job.kind, inst=job.inst, route_id=job.route_id, N=job.N,
                rep=job.rep, arm=job.arm)


def fields_from_key(key: str) -> dict:
    """Job fields recovered from a job key ``<arm|ladder|big>__<inst>__r<route>__N<N>__rep<rep>``
    (the file stem of models and row files). Inverse of ``Job.key``."""
    m = re.fullmatch(r"(?P<tag>[A-Za-z0-9]+)__(?P<inst>.+?)__r(?P<route>\d+)__N(?P<N>\d+)"
                     r"__rep(?P<rep>\d+)", key)
    if m is None:
        raise ValueError(f"not a job key: {key!r}")
    tag = m["tag"]
    kind, arm = ("neural", tag) if tag not in ("ladder", "big") else (tag, "")
    return dict(kind=kind, inst=m["inst"], route_id=int(m["route"]), N=int(m["N"]),
                rep=int(m["rep"]), arm=arm)


def _selected(fields: dict, only: dict | None) -> bool:
    return only is None or all(fields[k] == v for k, v in only.items())


def plan_stage(stage: str, elig: pd.DataFrame, only: dict | None = None) -> list[Job]:
    """All fit jobs of a stage, in a deterministic order (instance, then rep). With
    ``only`` (see ``parse_only``) just the jobs whose fields match exactly."""
    spec = stage_spec(stage)
    jobs: list[Job] = []
    for inst, rid in _route_rows(spec.routes, elig):
        for rep in spec.reps:
            for N in spec.ladder_N:
                otr = spec.ladder_otr21 and N >= OTR21_MIN_N
                labels = spec.ladder_labels + (("otr21",) if otr else ())
                jobs.append(Job("ladder", inst, rid, N, rep, labels=labels,
                                otr21=otr))
            if spec.big:
                jobs.append(Job("big", inst, rid, spec.big_N, rep, labels=BIG_LABELS))
            for arm, N in spec.neural:
                jobs.append(Job("neural", inst, rid, N, rep, arm=arm))
        if inst == SMOKE_INSTANCE and rid == 3:
            for arm, N in spec.timing:
                jobs.append(Job("neural", inst, rid, N, 0, arm=arm))
    jobs.sort(key=lambda j: (j.inst, j.rep, j.route_id, j.N, j.kind, j.arm))
    return [j for j in jobs if _selected(job_fields(j), only)]


def stage_eligibility(stage: str, path: Path | None) -> pd.DataFrame:
    """Smoke and pilot use CON3-0 computed on dev in memory. Other stages read the
    eligibility file written by M8."""
    if stage in SMOKE_LIKE:
        return build_eligibility([SMOKE_INSTANCE])
    if path is None:
        raise ValueError(f"stage {stage!r} needs --eligibility (the M8 table)")
    return pd.read_csv(path)


# ---------------------------------------------------------------- gate

def check_stream_for_stage(stage: str, eval_set: str) -> None:
    """Shared by ``check_test_gate`` and ``run_eval_job``: a gated stream is read only by
    the stage that declares it (EXPERIMENT_B.md 5.2). Dev is never gated."""
    if eval_set not in TEST_STREAMS:
        return
    spec = stage_spec(stage)
    if eval_set != spec.test_stream:
        raise GateError(f"stage {stage!r} evaluates on {spec.test_stream!r}, "
                        f"not on {eval_set!r}")
    if not spec.test_allowed:
        raise GateError(f"stage {stage!r} is dev-only; test evaluation is refused")


def model_owner_stage(model_path: Path) -> str | None:
    """The stage that owns a model file stored as ``models/<stage>/<key>.pkl`` (the layout
    ``run_fit_job`` writes); None for a file outside that layout."""
    return model_path.parent.name if model_path.parent.parent.name == "models" else None


def check_test_gate(stage: str, eval_set: str, frozen_commit: str | None) -> None:
    """Stage-level part of the gate (section 5). Raises GateError unless test
    evaluation is allowed; the commit checks are ``paths.verify_test_gate``."""
    if eval_set not in EVAL_SETS:
        raise GateError(f"eval set must be one of {EVAL_SETS}")
    if eval_set not in TEST_STREAMS:
        return
    check_stream_for_stage(stage, eval_set)
    verify_stream_gate(eval_set, frozen_commit, stage)


# ---------------------------------------------------------------- packing

def _pack(models: dict, arm: str, E0: float) -> dict:
    """Neural predictors -> plain state dicts (section 5: neural state dicts per fold and k)."""
    if arm in B_ARMS:
        return codecs.pack_models(models, arm, E0)
    states: dict[int, list] = {}
    for k, preds in models.items():
        packed = []
        for p in preds:
            if isinstance(p, NetPredictor):
                packed.append({"kind": "net",
                               "state": {n: t.detach().cpu().clone()
                                         for n, t in p.net.state_dict().items()}})
            elif isinstance(p, ConstantPredictor):
                packed.append({"kind": "const", "value": p.value})
            elif isinstance(p, BoostPredictor):
                packed.append({"kind": "boost", "model": p.model})
            else:
                raise TypeError(f"cannot pack predictor {type(p).__name__}")
        states[k] = packed
    return {"arm": arm, "E0": float(E0), "states": states}


def _unpack(blob: dict) -> dict:
    if blob["arm"] in B_ARMS:
        return codecs.unpack_models(blob)
    out: dict[int, list] = {}
    for k, items in blob["states"].items():
        preds = []
        for it in items:
            if it["kind"] == "boost":
                preds.append(BoostPredictor(it["model"]))
            elif it["kind"] == "net":
                net = MLPNet() if blob["arm"] == "mlp" else SetNet()
                net.load_state_dict(it["state"])
                preds.append(NetPredictor(blob["arm"], net, blob["E0"]))
            else:
                preds.append(ConstantPredictor(it["value"]))
        out[k] = preds
    return out


def _atomic_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


_MAIN_MODULES = ("__main__", "__mp_main__")


class _PayloadUnpickler(pickle.Unpickler):
    """Resolves classes this module defines that were pickled under a main-module name.

    Before the CLI was routed through this module, a run as ``python -m
    transvrp.expa.sweep`` (or a spawn worker of it) recorded ``Job`` as
    ``__main__.Job`` / ``__mp_main__.Job``. Only names that are classes defined here
    are remapped; everything else is resolved as usual.
    """

    def find_class(self, module, name):
        obj = globals().get(name)
        if module in _MAIN_MODULES and isinstance(obj, type):
            module = __name__
        return super().find_class(module, name)


def load_payload(path: Path):
    """Read one fitted-model pickle (``models/<stage>/<key>.pkl``) and return its payload."""
    with open(path, "rb") as fh:
        return _PayloadUnpickler(fh).load()


def chunk_cache_dir(ctx: Ctx) -> Path:
    """On-disk chunk cache (section 5, M1). Keys include the instance, purpose,
    replicate, chunk and seed, so one cache serves every route of an instance
    and every stage. The default is ``<results_dir>/chunks``; ``--chunk-dir``
    places it elsewhere. Chunk files are large: keep them out of version control."""
    return Path(ctx.chunk_dir) if ctx.chunk_dir is not None else Path(ctx.results_dir) / "chunks"


def _spec_of(inst: str, route_id: int) -> RouteSpec:
    for s in route_specs(inst):
        if s.route_id == route_id:
            return s
    raise ValueError(f"{inst} has no route {route_id}")


def _policy(kind: str, label: str) -> str:
    return BIG_RENAME.get(label, label) if kind == "big" else label


def arm_config(sspec: StageSpec, arm: str):
    """Configuration object of one B arm in a stage (EXPERIMENT_B 6.1). A main stage whose
    arm has no chosen configuration yet has no entry and refuses."""
    entries = [pairs for a, pairs in sspec.arm_cfg if a == arm]
    if len(entries) != 1:
        raise ValueError(f"stage has no configuration for arm {arm!r} "
                         "(a main stage needs the chosen configuration first)")
    return ARM_SPECS[arm].config_cls(**dict(entries[0]))


def stage_paths(sspec: StageSpec, paths: Paths, inst: str, route_id: int, purpose: str) -> Paths:
    """Independence world (EXPERIMENT_B 3.8): every stop column of the pool is permuted by
    its own seeded permutation (``permute_columns``, the same permutation for p and d, so
    the increments g = p - d move together and column marginals are kept). Identity unless
    the stage is independent. Applied to training and evaluation pools alike."""
    if not sspec.independent:
        return paths
    key = f"{inst}|{route_id}|{purpose}"
    return Paths(p=permute_columns(paths.p, key), d=permute_columns(paths.d, key))


def _fit_b_payload(job: Job, sspec: StageSpec, spec: RouteSpec, paths: Paths) -> dict:
    """Payload of one B arm fit. A tuning stage (``pass_curve``) trains one run and keeps a
    packed predictor set per pass; any other B stage packs the single fitted set."""
    cfg = arm_config(sspec, job.arm)
    if sspec.pass_curve:
        if job.arm not in CURVE_ARMS:
            raise ValueError(f"pass curves exist for {CURVE_ARMS}, not for {job.arm!r}")
        if cfg.passes != sspec.pass_curve:
            raise ValueError(f"the tuning stage trains {sspec.pass_curve} passes, the "
                             f"configuration has {cfg.passes}")
        curves, stats, diag, budget = fit_pass_curves(job.arm, spec, paths, job.N, job.rep, cfg)
        return {"kind": "neural", "arm": job.arm, "stats": stats, "diag": diag,
                "budget": budget,
                "pass_models": {p: codecs.pack_models(m, job.arm, spec.E0)
                                for p, m in curves.items()}}
    models, stats, diag, budget = fit_arm(job.arm, spec, paths, job.N, job.rep, cfg)
    return {"kind": "neural", "models": codecs.pack_models(models, job.arm, spec.E0),
            "stats": stats, "diag": diag, "budget": budget}


# ---------------------------------------------------------------- fit

def run_fit_job(job: Job, ctx: Ctx) -> str:
    path = ctx.models_dir / f"{job.key}.pkl"
    if path.exists():
        return "skip"
    sspec = stage_spec(ctx.stage)
    torch.set_num_threads(1)
    spec = _spec_of(job.inst, job.route_id)
    paths = route_pool(job.inst, "train", job.rep, job.N, cache_dir=chunk_cache_dir(ctx),
                       routes=[job.route_id])[job.route_id]
    paths = stage_paths(sspec, paths, job.inst, job.route_id, "train")
    t0 = time.perf_counter()
    if job.kind in ("ladder", "big"):
        stats = RouteStats.fit(paths)
        models = classical_ladder(spec, paths.g, job.labels, stats=stats)
        payload = {"kind": job.kind, "models": models, "stats": stats, "diag": []}
    elif job.kind == "neural" and job.arm in B_ARMS:
        payload = _fit_b_payload(job, sspec, spec, paths)
    elif job.kind == "neural" and job.arm == "boost":
        models, stats, diag = fit_boost(spec, paths, job.N, job.rep)
        payload = {"kind": "neural", "models": _pack(models, job.arm, spec.E0),
                   "stats": stats, "diag": diag}
    elif job.kind == "neural":
        models, stats, diag = fit_neural(spec, paths, job.arm, job.rep, job.N, ctx.cfg)
        payload = {"kind": "neural", "models": _pack(models, job.arm, spec.E0),
                   "stats": stats, "diag": diag}
    else:
        raise ValueError(f"unknown job kind {job.kind!r}")
    if sspec.test_stream == "test_b":
        # EXPERIMENT_B 5.3 item 8: the gate compares these with the running process
        payload["libraries"] = library_versions()
    # the commit alone does not prove the code that was fitted: record whether the
    # tree was clean when the fit ran (run_eval_job refuses test use otherwise)
    payload.update(job=job, fit_seconds=time.perf_counter() - t0,
                   fit_commit=ctx.commit, fit_tree_clean=bool(tree_is_clean()),
                   m=spec.m, B=spec.B)
    _atomic_pickle(payload, path)
    return "fit"


# ---------------------------------------------------------------- eval

def _rows_for(payload: dict, results: dict, spec: RouteSpec, eval_set: str,
              n_eval: int, eval_commit: str, extra: dict | None = None
              ) -> tuple[list[dict], dict]:
    job: Job = payload["job"]
    rows, arrays = [], {}
    for pol, res in results.items():
        rows.append(dict(inst=job.inst, route_id=job.route_id, m=spec.m,
                         stratum=stratum_of(spec.m), N=job.N, rep=job.rep,
                         policy=pol, mean_cost=res["mean_cost"],
                         handoff_rate=res["handoff_rate"],
                         fail_rate=res["fail_rate"], gap_closed=np.nan,
                         n_eval=n_eval, eval_set=eval_set,
                         fit_seconds=payload["fit_seconds"],
                         fit_commit=payload["fit_commit"],
                         eval_commit=eval_commit, **(extra or {})))
        arrays[pol] = np.asarray(res["costs"], dtype=np.float64)
    return rows, arrays


def _test_b_columns(payload: dict, gate, model_path: Path) -> dict:
    """The per-row columns of a test_b evaluation (EXPERIMENT_B 5.3)."""
    libs = payload.get("libraries") or {}
    out = {"family_sha256": gate.family_sha256,
           "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    out.update({name: libs.get(name) for name in ("python", "numpy", "scipy", "torch")})
    return out


def _curve_results(job: Job, payload: dict, spec: RouteSpec, pool: Paths) -> dict:
    """One policy per pass of a tuning run, named ``<arm>@p<pass>`` (EXPERIMENT_B 6.3)."""
    out = {}
    for p in sorted(payload["pass_models"]):
        cache = RouteCache(spec, pool, payload["stats"])
        costs, act = simulate_policy(cache, spec.B, spec.H, spec.E,
                                     _unpack(payload["pass_models"][p]), H_bill=spec.H)
        out[f"{ARM_POLICY[job.arm]}@p{p}"] = summarize(costs, act)
    return out


def run_eval_job(model_path: Path, eval_set: str, ctx: Ctx,
                 frozen: str | None = None) -> str:
    job_key = model_path.stem
    sspec = stage_spec(ctx.stage)
    gate = None
    if eval_set in TEST_STREAMS:
        check_stream_for_stage(ctx.stage, eval_set)
        owner = model_owner_stage(model_path)
        expected = sspec.reuse_models.rsplit("/", 1)[-1] if sspec.reuse_models else ctx.stage
        if owner is not None and owner != expected:
            raise GateError(f"model {job_key} belongs to stage {owner!r}, not {expected!r}; "
                            "it is never evaluated under another stage's gate or stream")
        gate = verify_stream_gate(eval_set, frozen, ctx.stage)
    gated = gate is not None
    rows_csv = ctx.rows_dir / f"{job_key}.{eval_set}.csv"
    if rows_csv.exists():
        return "skip"
    payload = load_payload(model_path)
    if gated and payload.get("fit_commit") != frozen:
        raise GateError(f"model {job_key} was fitted at {payload.get('fit_commit')}, "
                        f"not at the frozen commit")
    if gated and payload.get("fit_tree_clean") is not True:
        raise GateError(f"model {job_key} was fitted on a tree with uncommitted or untracked "
                        f".py changes (or before the flag existed); refusing test evaluation")
    job: Job = payload["job"]
    spec = _spec_of(job.inst, job.route_id)
    pool = route_pool(job.inst, eval_set, 0, frozen=frozen,
                      routes=[job.route_id], stage=ctx.stage)[job.route_id]
    pool = stage_paths(sspec, pool, job.inst, job.route_id, eval_set)
    n_eval = int(pool.p.shape[0])
    if payload["kind"] in ("ladder", "big"):
        res = eval_classical(spec, payload["models"], pool.g)
        results = {_policy(job.kind, lab): r for lab, r in res.items()}
        if sspec.reuse_models:       # cut (i): only the chosen ladder labels of section 3.7
            results = {pol: r for pol, r in results.items() if pol in sspec.ladder_labels}
    elif "pass_models" in payload:
        results = _curve_results(job, payload, spec, pool)
    else:
        preds = _unpack(payload["models"])
        cache = RouteCache(spec, pool, payload["stats"])
        costs, act = simulate_policy(cache, spec.B, spec.H, spec.E, preds,
                                     H_bill=spec.H)
        results = {ARM_POLICY[job.arm]: summarize(costs, act)}
    extra = _test_b_columns(payload, gate, model_path) if eval_set == "test_b" else None
    rows, arrays = _rows_for(payload, results, spec, eval_set, n_eval, ctx.commit, extra)
    ctx.rows_dir.mkdir(parents=True, exist_ok=True)
    npz_path = ctx.rows_dir / f"{job_key}.{eval_set}.npz"
    tmp_npz = npz_path.with_name(npz_path.name + ".tmp.npz")
    np.savez(tmp_npz, **arrays)
    os.replace(tmp_npz, npz_path)
    tmp_csv = rows_csv.with_name(rows_csv.name + ".tmp")
    columns = TEST_B_COLUMNS if eval_set == "test_b" else ROW_COLUMNS
    pd.DataFrame(rows, columns=columns).to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, rows_csv)
    return "eval"


# ---------------------------------------------------------------- parallel map

def _init_worker() -> None:
    torch.set_num_threads(1)


def _run_all(fn, items: list, jobs: int) -> list:
    if jobs <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    ctx_mp = get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx_mp,
                             initializer=_init_worker) as ex:
        return list(ex.map(fn, items, chunksize=1))


def run_fit(stage: str, elig: pd.DataFrame, results_dir: Path = DEFAULT_RESULTS,
            jobs: int = 1, commit: str | None = None,
            chunk_dir: Path | None = None, only: dict | None = None) -> dict:
    spec = stage_spec(stage)
    if spec.reuse_models:
        raise ValueError(f"stage {stage!r} reads A's model files ({spec.reuse_models}); "
                         "it has no fit")
    if spec.test_stream == "test_b" and not tree_is_clean():
        raise GateError(f"stage {stage!r} is fitted only on a clean code tree; "
                        "commit or remove uncommitted .py changes first")
    ctx = Ctx(Path(results_dir), stage, spec.cfg,
              current_commit() if commit is None else commit, chunk_dir)
    planned = plan_stage(stage, elig, only)
    if only is not None and not planned:
        raise ValueError(f"--only {only} matches no planned job of stage {stage!r}")
    done = _run_all(partial(run_fit_job, ctx=ctx), planned, jobs)
    return {"planned": len(planned), "fitted": done.count("fit"),
            "skipped": done.count("skip")}


def planned_keys(stage: str, results_dir: Path) -> set[str]:
    """Keys of a reuse stage's planned jobs, on A's eligibility table (cut (i))."""
    elig = pd.read_csv(Path(results_dir).parent / "expA" / "eligibility_dev.csv")
    return {j.key for j in plan_stage(stage, elig)}


def run_eval(stage: str, eval_set: str, results_dir: Path = DEFAULT_RESULTS,
             frozen_commit: str | None = None, jobs: int = 1,
             only: dict | None = None) -> dict:
    spec = stage_spec(stage)
    check_test_gate(stage, eval_set, frozen_commit)
    ctx = Ctx(Path(results_dir), stage, spec.cfg, current_commit())
    model_files = [p for p in sorted(ctx.models_dir.glob("*.pkl"))
                   if _selected(fields_from_key(p.stem), only)]
    if spec.reuse_models:
        keys = planned_keys(stage, results_dir)
        model_files = [p for p in model_files if p.stem in keys]
    if not model_files:
        raise FileNotFoundError(f"no fitted models for stage {stage!r} in {ctx.models_dir}")
    done = _run_all(partial(_eval_one, eval_set=eval_set, ctx=ctx, frozen=frozen_commit),
                    model_files, jobs)
    return {"models": len(model_files), "evaluated": done.count("eval"),
            "skipped": done.count("skip")}


def _eval_one(model_path: Path, eval_set: str, ctx: Ctx, frozen: str | None = None) -> str:
    return run_eval_job(model_path, eval_set, ctx, frozen=frozen)


# ---------------------------------------------------------------- merge

def add_gap_closed(df: pd.DataFrame) -> pd.DataFrame:
    """GC = (C_none - C_pi) / (C_none - C_oracle), NaN when C_none - C_oracle < 0.01 C_none
    (section 6). Uses the none and oracle rows with the same key."""
    key = ["inst", "route_id", "N", "rep", "eval_set"]
    base = df[df["policy"].isin(["none", "oracle"])]
    piv = base.pivot_table(index=key, columns="policy", values="mean_cost", aggfunc="first")
    out = df.copy()
    out["gap_closed"] = np.nan
    if piv.empty or not {"none", "oracle"} <= set(piv.columns):
        return out
    piv = piv[["none", "oracle"]].rename(columns={"none": "c_none", "oracle": "c_or"})
    merged = out.merge(piv, left_on=key, right_index=True, how="left")
    denom = merged["c_none"] - merged["c_or"]
    ok = denom >= 0.01 * merged["c_none"]
    gc = (merged["c_none"] - merged["mean_cost"]) / denom.where(ok)
    out["gap_closed"] = gc.to_numpy(dtype=float)
    return out


def merge_stage(stage: str, eval_set: str = "dev",
                results_dir: Path = DEFAULT_RESULTS, only: dict | None = None) -> pd.DataFrame:
    """Concatenates the row files of a stage into ``<results_dir>/<stage>.csv`` (dev) or
    ``<stage>.test.csv``. With ``only`` (see ``parse_only``) just the files of the jobs
    that match; the layout is the same, so a filtered merge can be passed to pilot --rerun."""
    rows_dir = Path(results_dir) / stage / "rows"
    suffix = f".{eval_set}.csv"
    files = sorted(f for f in rows_dir.glob(f"*{suffix}")
                   if _selected(fields_from_key(f.name[:-len(suffix)]), only))
    if not files:
        raise FileNotFoundError(f"no {eval_set} rows for stage {stage!r} in {rows_dir}"
                                + (f" matching --only {only}" if only else ""))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dup = df.duplicated(subset=KEY_COLUMNS, keep=False)
    if dup.any():
        raise ValueError(f"duplicate rows for {int(dup.sum())} keys in stage {stage!r}")
    columns = TEST_B_COLUMNS if eval_set == "test_b" else ROW_COLUMNS
    df = add_gap_closed(df)[columns]
    name = f"{stage}.csv" if eval_set == "dev" else f"{stage}.{eval_set}.csv"
    out = Path(results_dir) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------- timing (section 7.3)

def run_timing(results_dir: Path = DEFAULT_RESULTS, *, stage: str = "pilot",
               n_rows: int = 10 ** 6, model_N: int = 10 ** 5, arm: str = "full",
               rep: int = 0, inst: str = TIMING_INST, route_id: int = TIMING_ROUTE,
               chunk_dir: Path | None = None) -> dict:
    """Pilot 10^6 timing pass (section 7.3): generate the training pool of one route
    at ``n_rows`` paths, then apply the stop updates of the fitted nets from the
    ``model_N`` cell (the pilot's full-size cell by default). Data generation is
    timed without the chunk cache (``chunk_dir`` None), so it measures generation.

    Writes ``<results_dir>/<stage>/timing_<arm>_N<n_rows>.json`` and returns the
    record. Peak memory is the traced peak of the generation step and of the
    cache-plus-stop step, and the process high-water mark at the end.
    """
    results_dir = Path(results_dir)
    key = Job("neural", inst, route_id, model_N, rep, arm=arm).key
    model_path = results_dir / "models" / stage / f"{key}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"timing needs the fitted model {model_path}")
    payload = load_payload(model_path)
    spec = _spec_of(inst, route_id)
    stats = payload["stats"]
    preds = _unpack(payload["models"])
    sw = Stopwatch()
    with sw.lap("generate"), traced_peak() as gen_mem:
        pool = route_pool(inst, "train", rep, int(n_rows), cache_dir=chunk_dir,
                          routes=[route_id])[route_id]
    # cache and stop updates share one traced block: the stage batches built by the
    # stop updates are the largest allocation of this step
    with traced_peak() as stop_mem:
        with sw.lap("cache"):
            cache = RouteCache(spec, pool, stats)
            del pool
        with sw.lap("stop_updates"):
            costs, act = simulate_policy(cache, spec.B, spec.H, spec.E, preds,
                                         H_bill=spec.H)
    rec = dict(stage=stage, inst=inst, route_id=route_id, m=spec.m, arm=arm,
               rep=rep, model_N=int(model_N), n_rows=int(n_rows),
               generate_seconds=sw.seconds["generate"],
               cache_seconds=sw.seconds["cache"],
               stop_seconds=sw.seconds["stop_updates"],
               total_seconds=sum(sw.seconds.values()),
               mean_cost=float(costs.mean()),
               handoff_rate=float((act == 1).mean()),
               fail_rate=float((act == 2).mean()),
               # headline memory figure: process high-water mark (RSS, whole run)
               process_peak_rss_mb=process_peak_rss_mb(),
               # secondary: tracemalloc peaks (NumPy buffers only), one per step
               traced_generate_peak_mb=gen_mem["peak_mb"],
               traced_cache_stop_peak_mb=stop_mem["peak_mb"],
               traced_cache_stop_retained_mb=stop_mem["retained_mb"],
               chunk_cache=chunk_dir is not None)
    out = results_dir / stage / f"timing_{arm}_N{int(n_rows)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, out)
    return rec


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m transvrp.expa.sweep")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "eval", "merge"):
        p = sub.add_parser(name)
        p.add_argument("--stage", required=True, choices=tuple(STAGE_SPECS))
        p.add_argument("--jobs", type=int, default=1)
        p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
        p.add_argument("--only", default=None, metavar="FIELD=VALUE[,...]",
                       help="restrict to jobs matching these Job fields exactly, e.g. "
                            "arm=full,inst=CON3-0,route=0,N=10000,rep=0 (fields: "
                            + ", ".join(FILTER_FIELDS) + ")")
        if name == "fit":
            p.add_argument("--eligibility", type=Path, default=None)
            p.add_argument("--chunk-dir", type=Path, default=None,
                           help="chunk cache directory (default <results-dir>/chunks)")
        if name == "eval":
            p.add_argument("--eval", dest="eval_set", required=True, choices=EVAL_SETS)
            p.add_argument("--frozen-commit", default=None)
        if name == "merge":
            p.add_argument("--eval", dest="eval_set", default="dev", choices=EVAL_SETS)
    mp = sub.add_parser("manifest", help="model manifest of a B stage (EXPERIMENT_B 5.3)")
    mp.add_argument("--stage", required=True, choices=tuple(STAGE_SPECS))
    tp = sub.add_parser("timing", help="pilot 10^6 timing pass (section 7.3)")
    tp.add_argument("--stage", default="pilot", choices=tuple(STAGE_SPECS))
    tp.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    tp.add_argument("--n-rows", type=int, default=10 ** 6)
    tp.add_argument("--model-n", type=int, default=10 ** 5)
    tp.add_argument("--arm", default="full", choices=("full", "scalar", "mlp"))
    tp.add_argument("--chunk-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        only = parse_only(getattr(args, "only", None))
        if args.cmd == "timing":
            info = run_timing(args.results_dir, stage=args.stage, n_rows=args.n_rows,
                              model_N=args.model_n, arm=args.arm,
                              chunk_dir=args.chunk_dir)
            print({k: info[k] for k in ("generate_seconds", "stop_seconds",
                                        "process_peak_rss_mb",
                                        "traced_cache_stop_peak_mb")})
            return 0
        if args.cmd == "manifest":
            info = {"manifest": str(write_manifest(args.stage))}
        elif args.cmd == "fit":
            elig = stage_eligibility(args.stage, args.eligibility)
            info = run_fit(args.stage, elig, args.results_dir, args.jobs,
                           chunk_dir=args.chunk_dir, only=only)
        elif args.cmd == "eval":
            info = run_eval(args.stage, args.eval_set, args.results_dir,
                            args.frozen_commit, args.jobs, only=only)
        else:
            df = merge_stage(args.stage, args.eval_set, args.results_dir, only=only)
            info = {"rows": len(df), "policies": sorted(df["policy"].unique())}
    except GateError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(info)
    return 0


if __name__ == "__main__":
    # Run through the importable module so that classes such as Job pickle as
    # transvrp.expa.sweep.Job rather than __main__.Job.
    from transvrp.expa import sweep as _sweep
    raise SystemExit(_sweep.main())

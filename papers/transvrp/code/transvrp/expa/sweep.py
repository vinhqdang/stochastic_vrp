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
running checkout's commit, with no uncommitted changes under the code directory;
every model loaded for test must carry that hash as its fit commit. The checks
run before any data is generated.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
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
from .paths import (RouteSpec, TestStreamGateError, current_commit,  # noqa: E402
                    route_pool, route_specs, verify_test_gate)
from .simulate import simulate_policy, summarize  # noqa: E402
from .train import ConstantPredictor, NetPredictor, StageConfig, fit_neural  # noqa: E402

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
EVAL_SETS = ("dev", "test")
ROW_COLUMNS = ["inst", "route_id", "m", "stratum", "N", "rep", "policy",
               "mean_cost", "handoff_rate", "fail_rate", "gap_closed", "n_eval",
               "eval_set", "fit_seconds", "fit_commit", "eval_commit"]
KEY_COLUMNS = ["inst", "route_id", "N", "rep", "policy", "eval_set"]


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

    @property
    def models_dir(self) -> Path:
        return self.results_dir / "models" / self.stage

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


def plan_stage(stage: str, elig: pd.DataFrame) -> list[Job]:
    """All fit jobs of a stage, in a deterministic order (instance, then rep)."""
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
    return jobs


def stage_eligibility(stage: str, path: Path | None) -> pd.DataFrame:
    """Smoke and pilot use CON3-0 computed on dev in memory. Other stages read the
    eligibility file written by M8."""
    if stage in ("smoke", "pilot"):
        return build_eligibility([SMOKE_INSTANCE])
    if path is None:
        raise ValueError(f"stage {stage!r} needs --eligibility (the M8 table)")
    return pd.read_csv(path)


# ---------------------------------------------------------------- gate

def check_test_gate(stage: str, eval_set: str, frozen_commit: str | None) -> None:
    """Stage-level part of the gate (section 5). Raises GateError unless test
    evaluation is allowed; the commit checks are ``paths.verify_test_gate``."""
    if eval_set not in EVAL_SETS:
        raise GateError(f"eval set must be one of {EVAL_SETS}")
    if eval_set != "test":
        return
    if not stage_spec(stage).test_allowed:
        raise GateError(f"stage {stage!r} is dev-only; test evaluation is refused")
    verify_test_gate(frozen_commit)


# ---------------------------------------------------------------- packing

def _pack(models: dict, arm: str, E0: float) -> dict:
    """Neural predictors -> plain state dicts (section 5: neural state dicts per fold and k)."""
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


def _spec_of(inst: str, route_id: int) -> RouteSpec:
    for s in route_specs(inst):
        if s.route_id == route_id:
            return s
    raise ValueError(f"{inst} has no route {route_id}")


def _policy(kind: str, label: str) -> str:
    return BIG_RENAME.get(label, label) if kind == "big" else label


# ---------------------------------------------------------------- fit

def run_fit_job(job: Job, ctx: Ctx) -> str:
    path = ctx.models_dir / f"{job.key}.pkl"
    if path.exists():
        return "skip"
    torch.set_num_threads(1)
    spec = _spec_of(job.inst, job.route_id)
    paths = route_pool(job.inst, "train", job.rep, job.N)[job.route_id]
    t0 = time.perf_counter()
    if job.kind in ("ladder", "big"):
        stats = RouteStats.fit(paths)
        models = classical_ladder(spec, paths.g, job.labels, stats=stats)
        payload = {"kind": job.kind, "models": models, "stats": stats, "diag": []}
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
    payload.update(job=job, fit_seconds=time.perf_counter() - t0,
                   fit_commit=ctx.commit, m=spec.m, B=spec.B)
    _atomic_pickle(payload, path)
    return "fit"


# ---------------------------------------------------------------- eval

def _rows_for(payload: dict, results: dict, spec: RouteSpec, eval_set: str,
              n_eval: int, eval_commit: str) -> tuple[list[dict], dict]:
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
                         eval_commit=eval_commit))
        arrays[pol] = np.asarray(res["costs"], dtype=np.float64)
    return rows, arrays


def run_eval_job(model_path: Path, eval_set: str, ctx: Ctx,
                 frozen: str | None = None) -> str:
    job_key = model_path.stem
    if eval_set == "test":
        verify_test_gate(frozen)
    rows_csv = ctx.rows_dir / f"{job_key}.{eval_set}.csv"
    if rows_csv.exists():
        return "skip"
    with open(model_path, "rb") as fh:
        payload = pickle.load(fh)
    if eval_set == "test" and payload.get("fit_commit") != frozen:
        raise GateError(f"model {job_key} was fitted at {payload.get('fit_commit')}, "
                        f"not at the frozen commit")
    job: Job = payload["job"]
    spec = _spec_of(job.inst, job.route_id)
    pool = route_pool(job.inst, eval_set, 0, frozen=frozen)[job.route_id]
    n_eval = int(pool.p.shape[0])
    if payload["kind"] in ("ladder", "big"):
        res = eval_classical(spec, payload["models"], pool.g)
        results = {_policy(job.kind, lab): r for lab, r in res.items()}
    else:
        preds = _unpack(payload["models"])
        cache = RouteCache(spec, pool, payload["stats"])
        costs, act = simulate_policy(cache, spec.B, spec.H, spec.E, preds,
                                     H_bill=spec.H)
        results = {ARM_POLICY[job.arm]: summarize(costs, act)}
    rows, arrays = _rows_for(payload, results, spec, eval_set, n_eval, ctx.commit)
    ctx.rows_dir.mkdir(parents=True, exist_ok=True)
    npz_path = ctx.rows_dir / f"{job_key}.{eval_set}.npz"
    tmp_npz = npz_path.with_name(npz_path.name + ".tmp.npz")
    np.savez(tmp_npz, **arrays)
    os.replace(tmp_npz, npz_path)
    tmp_csv = rows_csv.with_name(rows_csv.name + ".tmp")
    pd.DataFrame(rows, columns=ROW_COLUMNS).to_csv(tmp_csv, index=False)
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
            jobs: int = 1, commit: str | None = None) -> dict:
    spec = stage_spec(stage)
    ctx = Ctx(Path(results_dir), stage, spec.cfg,
              current_commit() if commit is None else commit)
    planned = plan_stage(stage, elig)
    done = _run_all(partial(run_fit_job, ctx=ctx), planned, jobs)
    return {"planned": len(planned), "fitted": done.count("fit"),
            "skipped": done.count("skip")}


def run_eval(stage: str, eval_set: str, results_dir: Path = DEFAULT_RESULTS,
             frozen_commit: str | None = None, jobs: int = 1) -> dict:
    spec = stage_spec(stage)
    check_test_gate(stage, eval_set, frozen_commit)
    ctx = Ctx(Path(results_dir), stage, spec.cfg, current_commit())
    model_files = sorted(ctx.models_dir.glob("*.pkl"))
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
                results_dir: Path = DEFAULT_RESULTS) -> pd.DataFrame:
    rows_dir = Path(results_dir) / stage / "rows"
    files = sorted(rows_dir.glob(f"*.{eval_set}.csv"))
    if not files:
        raise FileNotFoundError(f"no {eval_set} rows for stage {stage!r} in {rows_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dup = df.duplicated(subset=KEY_COLUMNS, keep=False)
    if dup.any():
        raise ValueError(f"duplicate rows for {int(dup.sum())} keys in stage {stage!r}")
    df = add_gap_closed(df)[ROW_COLUMNS]
    name = f"{stage}.csv" if eval_set == "dev" else f"{stage}.{eval_set}.csv"
    out = Path(results_dir) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m transvrp.expa.sweep")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "eval", "merge"):
        p = sub.add_parser(name)
        p.add_argument("--stage", required=True, choices=tuple(STAGE_SPECS))
        p.add_argument("--jobs", type=int, default=1)
        p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
        if name == "fit":
            p.add_argument("--eligibility", type=Path, default=None)
        if name == "eval":
            p.add_argument("--eval", dest="eval_set", required=True, choices=EVAL_SETS)
            p.add_argument("--frozen-commit", default=None)
        if name == "merge":
            p.add_argument("--eval", dest="eval_set", default="dev", choices=EVAL_SETS)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "fit":
            elig = stage_eligibility(args.stage, args.eligibility)
            info = run_fit(args.stage, elig, args.results_dir, args.jobs)
        elif args.cmd == "eval":
            info = run_eval(args.stage, args.eval_set, args.results_dir,
                            args.frozen_commit, args.jobs)
        else:
            df = merge_stage(args.stage, args.eval_set, args.results_dir)
            info = {"rows": len(df), "policies": sorted(df["policy"].unique())}
    except GateError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(info)
    return 0


if __name__ == "__main__":
    sys.exit(main())

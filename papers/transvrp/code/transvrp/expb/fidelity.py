"""Dev fidelity gates and the family derivation (EXPERIMENT_B.md section 7, M22; the
family-derivation block of section 5.3).

``gate_report(stage, arm, dev_df, models_dir, rows_dir)`` computes the section-7 gates of
one arm on dev for its main-fit stage: B0 (all arms), the arm's bug gate, and P1. Every
gate is a function of payloads (models, stats, diag, budget, job), dev rows, per-day dev
costs and the route pools that a ``DevSource`` supplies. The real source reads the dev and
train streams through ``expa.paths``; tests inject constructed pools. Nothing here reads
test or test_b data.

``derive_family`` turns the six arm reports into the five FAMILY lists (sections 1.3, 5.3);
``render_report`` writes ``fidelity_dev.md`` whose ``family-derivation`` JSON block
``gatecheck`` parses back. The CLI (``python -m transvrp.expb.fidelity``) runs all arms.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..expa import sweep
from ..expa.backward import split_80_20
from ..expa.features import RouteCache
from ..expa.paths import Paths, RouteSpec, route_pool
from ..expa.simulate import simulate_policy
from .alp import (ALPZConfig, build_lp, continuation_table, initial_value_dp,
                  lattice_dp, lattice_pmf, make_lattice, solve_lp)
from .codecs import unpack_models
from .dqn import stage_counts
from .gatecheck import core_of_stage
from .grid import lin_interp, tie_tolerance
from .ppo import GRU_HIDDEN, HANDOFF, PolicyPredictor, mask_logits
from .registry import B_ARMS
from .thr import THRConfig, independent_dp, permute_columns, thresholds

T1_TOL = 1e-6          # T1 tie window and LP identity tolerance, in units of E0
EXACT_TOL = 1e-9       # relative tolerance of exact identities (monotone, C >= C_oracle)
T4_SE = 4.0            # B-THR: per-day standard errors of the route statistic (see thr_route_within)
T4_SHARE = 0.99        # B-THR: share of routes within T4_SE
T5_SHARE = 0.999       # B-ALPZ (d): share of visited states with equal decisions
T6_SE = 2.0            # P1: standard errors above the best trivial policy
T7_SHARE = {"rlsm": 0.90, "dos": 0.95}   # bug-detector shares of stages
T9_POINTS = 0.5        # budget-limited label, points of the tuning score
T9_PMAX = 3            # passes of every DQN and PPO run
T15_SHARE = 0.99       # "coincides with THR" wording, descriptive only
MIN_VAL_ROWS = 100     # stages with fewer alive validation rows are not counted
RLSM_RESIDUAL = 1e-8   # B-RLSM: relative ridge normal-equation residual
DOS_TOL = 0.02         # B-DOS: relative slack, tau * |always stop - never stop| (revision 3.4)
PPO_ROWS = 1000        # B-PPO: dev rows per job for the cached-versus-recomputed check
CUT_SUFFIX = ":cut"      # dropped entry of a compute-cut arm (9.2 (ii)); no space allowed
REFERENCES = ("full", "ladder")   # section 3.7 reference stages (not gated, not family)
INDEPENDENCE_STAGES = ("b_ladder_ind", "b_thr_ind", "b_alpz_ind", "b_full_ind")  # 3.8

GATE_OF_ARM = {"thr": "b_thr", "alpz": "b_alpz", "rlsm": "b_rlsm", "dos": "b_dos",
               "dqn": "b_dqn", "ppo": "b_ppo"}
FULL_POLICY = "nn_full"        # sweep.ARM_POLICY["full"]
ALLOWED_PURPOSES = ("dev", "train")
RESULTS_DIR = sweep.CODE_DIR / "results" / "expB"


class DevSource:
    """Route data for the gates. The default reads the real dev and train streams (the
    only purposes allowed here) and the route specifications of ``expa.sweep``."""

    def spec(self, inst: str, route_id: int) -> RouteSpec:
        return sweep._spec_of(inst, route_id)

    def pool(self, inst: str, route_id: int, purpose: str, rep: int = 0,
             N: int | None = None) -> Paths:
        if purpose not in ALLOWED_PURPOSES:
            raise ValueError(f"fidelity reads {ALLOWED_PURPOSES} only, not {purpose!r}")
        return route_pool(inst, purpose, rep, N, routes=[route_id])[route_id]


@dataclass(frozen=True)
class Gate:
    id: str
    passed: bool
    value: float | None
    threshold: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "passed": bool(self.passed), "value": _finite(self.value),
                "threshold": self.threshold, "detail": self.detail}


@dataclass
class _Ctx:
    stage: str
    arm: str
    jobs: list                     # payload dicts, sorted by job key
    source: DevSource
    rows_dir: Path
    independent: bool              # 3.8 stages: stop columns permuted in train and dev
    g2_keys: frozenset | None      # route-reps of the uniform-LP identity (default rep 0)
    thr_dir: Path | None = None    # THR models, for the descriptive ALP-Z vs THR wording
    info: dict = field(default_factory=dict)
    dos_tol: float = DOS_TOL       # B-DOS relative slack tau; a parameter, not a constant
    passes: int | None = None      # B-DQN: passes of the main fit (None: any of 1..T9_PMAX)


# ---------------------------------------------------------------- small helpers

def _finite(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _key(job) -> tuple:
    return (str(job.inst), int(job.route_id), int(job.rep))


def _keyed(df: pd.DataFrame) -> dict:
    """(inst, route, rep) -> mean_cost of the rows; duplicates are an error."""
    out: dict = {}
    for row in df.itertuples(index=False):
        key = (str(row.inst), int(row.route_id), int(row.rep))
        if key in out:
            raise ValueError(f"duplicate dev row for {key}")
        out[key] = float(row.mean_cost)
    return out


def _se(costs: np.ndarray) -> float:
    n = costs.size
    return float(costs.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0


def _share(count: int, total: int) -> float:
    return count / total if total else float("nan")


def _meets(share: float, threshold: float) -> bool:
    """Share gates (B-THR, B-ALPZ, B-RLSM, B-DOS): share at least the threshold."""
    return share >= threshold


def _within_band(mean: float, dp: float, se: float, width: float | None = None) -> bool:
    """B-THR (a): |simulated - DP| within ``width`` (default T4_SE) per-day standard errors
    (boundary in)."""
    return abs(mean - dp) <= (T4_SE if width is None else width) * se


def thr_route_within(day_costs, dps, width: float | None = None) -> bool:
    """B-THR (a) statistic of ONE route. ``day_costs`` holds the per-day dev cost series of
    each replicate (the same permuted dev days: the dev stream does not depend on the
    replicate), ``dps`` the replicates' own-model DP values. The statistic compares the
    replicate mean of the simulated costs with the replicate mean of the DP values, with
    the SE of the replicate-averaged per-day series.

    Why per route and not per job (review M6/M5): the replicates of a route share their
    dev days, so their errors are almost perfectly correlated and fail together; counting
    them as three independent tests inflated the false-drop probability. Averaging over
    replicates before taking the SE accounts for the sharing exactly (no independence
    assumed), and the share is then taken over routes, which are independent (distinct
    seeds and days). With the share rule at 99% over 62 routes no failure is allowed, so the
    band must make P(some of 62 exact routes outside) <= 1%: 1 - (1 - 2 Phi(-4))^62 = 0.4%
    at 4 SE, against 15% at 3 SE."""
    mean_days = np.mean([np.asarray(c, dtype=float) for c in day_costs], axis=0)
    return _within_band(float(mean_days.mean()), float(np.mean(dps)), _se(mean_days), width)


def _monotone(c: np.ndarray, E0: float) -> bool:
    """B-THR (b): every C_k nondecreasing along the grid, up to EXACT_TOL * E0."""
    return bool(np.all(np.diff(c, axis=1) >= -EXACT_TOL * E0)) if c.shape[1] > 1 else True


def thr_dp_value(g1: np.ndarray, w_grid: np.ndarray, c1: np.ndarray, H1: float,
                 E_first: float, B: float) -> float:
    """DP value of THR at the initial state (3.1): mean over first-stop increments of
    1{g1 > B} E[0] + 1{g1 <= B} min(H[0], C_1(g1)), C_1 read by linear interpolation."""
    g1 = np.asarray(g1, dtype=float)
    w0, h = float(w_grid[0]), float(w_grid[1] - w_grid[0])
    v1 = np.minimum(H1, lin_interp(g1, w0, h, c1))
    return float(np.mean(np.where(g1 > B, E_first, v1)))


def visited_states(cache: RouteCache, B: float, H: np.ndarray, models: dict) -> list:
    """(k, rows, loads W_k, handoff) for every decision the deployed policy evaluates, in
    the loop of ``simulate_policy``: a breach at stop k ends a path before its decision."""
    W = cache.W
    N, m = W.shape
    stopped = np.zeros(N, dtype=bool)
    out = []
    for k in range(1, m):
        stopped |= (~stopped) & (W[:, k - 1] > B)
        rows = np.flatnonzero(~stopped)
        if rows.size == 0:
            continue
        chat = np.mean([p.predict(cache, k, rows) for p in models.get(k, [])], axis=0)
        trig = chat > H[k - 1]
        out.append((k, rows, W[rows, k - 1], trig))
        stopped[rows[trig]] = True
    return out


def _pred(payload: dict) -> tuple[dict, object]:
    """The unpacked predictors of a payload, and the predictor of epoch 1 (shared by k)."""
    models = unpack_models(payload["models"])
    if 1 not in models:
        raise ValueError("a fitted model holds no decision epoch 1 (route with m < 2)")
    return models, models[1][0]


def _day_costs(rows_dir: Path, job, arm: str) -> np.ndarray:
    with np.load(Path(rows_dir) / f"{job.key}.dev.npz") as z:
        return np.asarray(z[arm], dtype=float)


def _perm(paths: Paths, inst: str, route_id: int, purpose: str) -> Paths:
    key = f"{inst}|{route_id}|{purpose}"
    return Paths(p=permute_columns(paths.p, key), d=permute_columns(paths.d, key))


def _dev(ctx: _Ctx, job, always: bool = False) -> Paths:
    paths = ctx.source.pool(job.inst, job.route_id, "dev")
    return _perm(paths, job.inst, job.route_id, "dev") if (always or ctx.independent) else paths


def _train(ctx: _Ctx, job) -> Paths:
    paths = ctx.source.pool(job.inst, job.route_id, "train", rep=job.rep, N=job.N)
    return _perm(paths, job.inst, job.route_id, "train") if ctx.independent else paths


def _cache(spec: RouteSpec, paths: Paths, payload: dict) -> RouteCache:
    return RouteCache(spec, paths, payload["stats"])


# ---------------------------------------------------------------- B0 and P1 (all arms)

def require_dev(dev_df: pd.DataFrame, rows_dir: Path) -> None:
    """Fidelity reads dev only: every row is a dev row, and the rows directory holds no
    test or test_b file."""
    if "eval_set" not in dev_df.columns:
        raise ValueError("dev rows carry no eval_set column; refusing to run")
    sets = sorted(set(dev_df["eval_set"].astype(str)))
    if sets != ["dev"]:
        raise ValueError(f"fidelity reads dev rows only; found eval_set {sets}")
    leaked = sorted(p.name for p in Path(rows_dir).glob("*") if ".test" in p.name)
    if leaked:
        raise ValueError(f"rows directory holds non-dev files {leaked}; refusing to run")


def load_jobs(models_dir: Path) -> list:
    """The payloads of one stage's models, in job-key order (sweep.load_payload)."""
    paths = sorted(Path(models_dir).glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"no model payloads in {models_dir}")
    payloads = [sweep.load_payload(p) for p in paths]
    payloads.sort(key=lambda p: p["job"].key)
    index = [_key(p["job"]) for p in payloads]
    if len(set(index)) != len(index):
        raise ValueError(f"several payloads for one route-rep in {models_dir}")
    return payloads


def _b0(ctx: _Ctx, arm_df: pd.DataFrame, oracle_df: pd.DataFrame) -> Gate:
    arm = _keyed(arm_df)
    oracle = _keyed(oracle_df)
    index = {_key(p["job"]): p for p in ctx.jobs}
    covered = len(set(arm) & set(index))
    fitted = bool(arm) and set(arm) == set(index) and all(p["models"]["states"]
                                                           for p in ctx.jobs)
    finite = all(math.isfinite(v) for v in arm.values()) and all(
        np.all(np.isfinite(_day_costs(ctx.rows_dir, p["job"], ctx.arm))) for p in ctx.jobs)
    no_oracle = sorted(set(arm) - set(oracle))
    worse = [k for k in arm if k in oracle
             and arm[k] < oracle[k] - EXACT_TOL * max(1.0, abs(oracle[k]))]
    passed = fitted and finite and not no_oracle and not worse
    detail = (f"fitted {covered}/{len(arm)} route-reps; {len(worse)} below oracle; "
              f"{len(no_oracle)} without oracle row")
    return Gate("b0", passed, _share(covered, len(arm)),
                "100% fitted, finite, C >= C_oracle", detail)


def _gate_p1(ctx: _Ctx) -> Gate:
    """P1: pooled dev cost (sum over routes, mean over reps) <= min(C_none, C_h1) + 2 SE.
    The competitor is the smaller pooled mean. SE: per route, the paired per-day difference
    to that competitor, each rep's standard error divided by the number of reps and summed
    in quadrature over reps; route SEs summed in quadrature over routes."""
    rows = []
    for p in ctx.jobs:
        job = p["job"]
        spec = ctx.source.spec(job.inst, job.route_id)
        cache = _cache(spec, _dev(ctx, job), p)
        none_c = simulate_policy(cache, spec.B, spec.H, spec.E, {}, H_bill=spec.H)[0]
        h1_c = np.where(cache.W[:, 0] > spec.B, spec.E[0], spec.H[0])
        rows.append((_key(job), job.rep, _day_costs(ctx.rows_dir, job, ctx.arm), none_c, h1_c))
    n_reps = len({r[1] for r in rows})
    if n_reps == 0:
        return Gate("p1", False, None, "pooled <= min(C_none, C_h1) + 2 SE", "no jobs")
    pooled = {name: sum(float(np.mean(r[i])) for r in rows) / n_reps
              for name, i in (("arm", 2), ("none", 3), ("h1", 4))}
    comp = "none" if pooled["none"] <= pooled["h1"] else "h1"
    route_var: dict = {}
    for key, _rep, arm_c, none_c, h1_c in rows:
        comp_c = none_c if comp == "none" else h1_c
        route_var[key[:2]] = route_var.get(key[:2], 0.0) + (_se(arm_c - comp_c) / n_reps) ** 2
    pool_se = math.sqrt(sum(route_var.values()))
    bound = pooled[comp]
    passed = pooled["arm"] <= bound + T6_SE * pool_se
    detail = (f"pooled arm {pooled['arm']:.6g}, competitor {comp} {bound:.6g}, "
              f"SE {pool_se:.3g}")
    return Gate("p1", passed, pooled["arm"] - bound,
                "pooled <= min(C_none, C_h1) + 2 SE", detail)


# ---------------------------------------------------------------- arm bug gates

def _refinement_report(ctx: _Ctx) -> dict:
    """4,001-point THR refinement on the G2 route-reps (descriptive; not part of any gate)."""
    items = [thr_grid_refinement(ctx.source.spec(p["job"].inst, p["job"].route_id),
                                 _train(ctx, p["job"]).g)
             for p in ctx.jobs if _key(p["job"]) in ctx.g2_keys]
    return {"route_reps": len(items),
            "max_rel_diff": max((i["max_rel_diff"] for i in items), default=0.0),
            "inf_mismatch": sum(i["inf_mismatch"] for i in items), "n_fine": THR_FINE_GRID}


def _gate_thr(ctx: _Ctx) -> Gate:
    mono, dp_by_key = True, {}
    by_route: dict = {}
    for p in ctx.jobs:
        job = p["job"]
        spec = ctx.source.spec(job.inst, job.route_id)
        models, pred = _pred(p)
        mono = mono and _monotone(pred.c, spec.E0)
        dp = thr_dp_value(_train(ctx, job).g[:, 0], pred.w_grid, pred.c[0], spec.H[0],
                          spec.E[0], spec.B)
        costs = simulate_policy(_cache(spec, _dev(ctx, job, always=True), p), spec.B,
                                spec.H, spec.E, models, H_bill=spec.H)[0]
        by_route.setdefault((str(job.inst), int(job.route_id)), []).append((costs, dp))
        dp_by_key[_key(job)] = dp
    total = len(by_route)
    within = sum(thr_route_within([c for c, _ in reps], [d for _, d in reps])
                 for reps in by_route.values())
    share = _share(within, total)
    ok = mono and _meets(share, T4_SHARE)
    ctx.info["thr_dp"] = dp_by_key
    ctx.info["thr_refinement"] = _refinement_report(ctx)
    return Gate("b_thr", ok, share,
                f"routes (replicate mean) within {T4_SE:g} SE on >= {T4_SHARE:.0%}; "
                "C_k nondecreasing",
                f"{within}/{total} routes within band; monotone={mono}")


THR_FINE_GRID = 4001   # T3: refinement reported on the G2 route-reps, never gated


def thr_grid_refinement(spec: RouteSpec, g: np.ndarray, n_fine: int = THR_FINE_GRID,
                        n_base: int | None = None) -> dict:
    """Report item (T3, 7(c)): the THR thresholds on an ``n_fine``-point grid against the
    base grid (default 2,001), as max over stops of |W*_fine - W*_base| / E0. Stops whose
    threshold is infinite on exactly one of the grids are counted in ``inf_mismatch``."""
    n_base = THRConfig().n_grid if n_base is None else n_base
    tie = tie_tolerance(spec.E0)
    thr = {}
    for n in (n_base, n_fine):
        w_grid, C, _ = independent_dp(spec, g, n)
        thr[n] = thresholds(w_grid, C, spec.H, tie)
    diffs = [abs(thr[n_fine][k] - thr[n_base][k]) / float(spec.E0) for k in thr[n_base]
             if math.isfinite(thr[n_base][k]) and math.isfinite(thr[n_fine][k])]
    mismatch = sum(math.isfinite(thr[n_base][k]) != math.isfinite(thr[n_fine][k])
                   for k in thr[n_base])
    return {"n_base": int(n_base), "n_fine": int(n_fine),
            "max_rel_diff": max(diffs) if diffs else 0.0, "inf_mismatch": int(mismatch)}


def _gate_alpz(ctx: _Ctx) -> Gate:
    cfg = ALPZConfig()
    status_ok = obj_ok = g2_ok = True
    agree = total = exc = thr_agree = thr_total = 0
    tol = T1_TOL
    for p in ctx.jobs:
        job = p["job"]
        spec = ctx.source.spec(job.inst, job.route_id)
        lps = [d["lp_status"] for d in p["diag"] if "lp_status" in d]
        status_ok = status_ok and bool(lps) and all(s == 0 for s in lps)
        tr = _train(ctx, job)
        lat = make_lattice(spec, tr.g, cfg.n_w)
        pmfs = lattice_pmf(tr.g, lat.h)
        V = lattice_dp(spec, pmfs, lat)
        init = initial_value_dp(spec, pmfs, lat)
        obj_ok = obj_ok and abs(p["budget"]["lp_objective"] - init) <= tol * spec.E0
        if _key(job) in ctx.g2_keys:
            g2_ok = g2_ok and _uniform_identity(spec, pmfs, lat, V, cfg.n_w, tol)
        models, _ = _pred(p)
        cache = _cache(spec, _dev(ctx, job), p)
        a, t, x, thr_pair = _decision_agreement(spec, pmfs, lat, V, cache, models,
                                                ctx.thr_dir, job, tol)
        agree, total, exc = agree + a, total + t, exc + x
        thr_agree, thr_total = thr_agree + thr_pair[0], thr_total + thr_pair[1]
    share = _share(agree, total)
    ok = status_ok and obj_ok and g2_ok and _meets(share, T5_SHARE)
    ctx.info["alpz"] = {"decision_share": _finite(share), "exceptions": exc,
                        "g2_route_reps": len(ctx.g2_keys)}
    if thr_total:
        t15 = _share(thr_agree, thr_total)
        ctx.info["alpz"].update(thr_share=_finite(t15), coincides=bool(t15 > T15_SHARE))
    return Gate("b_alpz", ok, share,
                f"LP optimal; LP = DP; uniform LP = DP on G2; decisions >= {T5_SHARE:.1%}",
                f"status={status_ok} objective={obj_ok} uniform={g2_ok} "
                f"decisions {agree}/{total} (exceptions {exc})")


def _uniform_identity(spec: RouteSpec, pmfs, lat, V, n_w: int, tol: float) -> bool:
    """B-ALPZ (c): the uniform-weight LP returns the lattice DP at every lattice point."""
    c, A, b, info = build_lp(spec, pmfs, lat, ALPZConfig(n_w=n_w, objective="uniform"))
    x, status, _ = solve_lp(c, A, b, info)
    f = x.reshape(spec.m - 1, lat.n_w)
    return status == 0 and float(np.max(np.abs(f - V))) <= tol * spec.E0


def _decision_agreement(spec, pmfs, lat, V, cache, models, thr_dir, job, tol):
    """B-ALPZ (d): deployed decisions against the lattice-DP decisions on the visited
    states (near-tie states with |L_DP - H| <= tol are exceptions, counted as agreeing).
    Returns (agree, total, exceptions, (THR agree, THR total)) where the last pair is the
    descriptive THR comparison on the same states, empty without THR models."""
    states = visited_states(cache, spec.B, spec.H, models)
    if not states:
        return 0, 0, 0, (0, 0)
    ks = np.concatenate([np.full(r.size, k) for k, r, _, _ in states])
    loads = np.concatenate([w for _, _, w, _ in states])
    deployed = np.concatenate([t for _, _, _, t in states])
    L = continuation_table(spec, pmfs, lat, V, loads)
    Ldp = L[ks - 1, np.arange(loads.size)]
    Hk = spec.H[ks - 1]
    exc = np.abs(Ldp - Hk) <= tol * spec.E0
    dp_dec = Ldp >= Hk - tol * spec.E0
    agree = int(np.sum((dp_dec == deployed) | exc))
    thr_pair = (0, 0)
    if thr_dir is not None:
        thr_pair = _thr_decisions(thr_dir, job, cache, states, spec)
    return agree, int(loads.size), int(exc.sum()), thr_pair


def _thr_decisions(thr_dir, job, cache, states, spec) -> tuple[int, int]:
    """Descriptive (T15): THR's deployed rule on the states the ALP-Z policy visits.
    Returns (agree with ALP-Z's decisions, total states)."""
    _, thr_pred = _pred(_thr_payload(thr_dir, job))
    agree = total = 0
    for k, rows, _loads, trig in states:
        thr_trig = thr_pred.predict(cache, k, rows) > spec.H[k - 1]
        agree += int(np.sum(thr_trig == trig))
        total += rows.size
    return agree, total


def _thr_payload(thr_dir, job) -> dict:
    for path in sorted(Path(thr_dir).glob("*.pkl")):
        payload = sweep.load_payload(path)
        if _key(payload["job"]) == _key(job):
            return payload
    raise FileNotFoundError(f"no THR payload for route-rep {_key(job)} in {thr_dir}")


def _gate_rlsm(ctx: _Ctx) -> Gate:
    resid_ok, rows_total, passing, missing = True, 0, 0, 0
    for p in ctx.jobs:
        for d in p["diag"]:
            if d.get("kind") != "stage":
                continue
            if d.get("action") == "fit":
                resid_ok = resid_ok and float(d["ne_residual_rel"]) <= RLSM_RESIDUAL
            if int(d.get("n_val", 0)) < MIN_VAL_ROWS:
                continue
            rows_total += 1
            if d.get("action") == "constant":
                passing += 1                  # constant predictor: MSE equal to its own
            elif d.get("val_mse") is None:
                missing += 1
            else:
                passing += d["val_mse"] <= d["const_val_mse"]
    share = _share(passing, rows_total)
    ok = resid_ok and rows_total > 0 and _meets(share, T7_SHARE["rlsm"])
    return Gate("b_rlsm", ok, share,
                f"ridge residual <= 1e-8; MSE <= const on >= {T7_SHARE['rlsm']:.0%} of "
                f"stages with >= {MIN_VAL_ROWS} val rows",
                f"{passing}/{rows_total} stages; residual ok={resid_ok}; missing {missing}")


def _gate_dos(ctx: _Ctx) -> Gate:
    passing, total, missing = 0, 0, 0
    for p in ctx.jobs:
        spec = ctx.source.spec(p["job"].inst, p["job"].route_id)
        for d in p["diag"]:
            if d.get("kind") != "stage" or int(d.get("n_val", 0)) < MIN_VAL_ROWS:
                continue
            total += 1
            obj = _dos_objective(d)
            if obj is None or d.get("never_val") is None:
                missing += 1
                continue
            always = spec.H[d["k"] - 1] / spec.E0      # F = 1 everywhere: loss = H / E0
            never = float(d["never_val"])
            passing += obj <= min(always, never) + ctx.dos_tol * abs(always - never)
    share = _share(passing, total)
    ok = total > 0 and missing == 0 and _meets(share, T7_SHARE["dos"])
    return Gate("b_dos", ok, share,
                f"validation objective <= min(always, never) + {ctx.dos_tol:g} |always - never| on >= "
                f"{T7_SHARE['dos']:.0%} of stages with >= {MIN_VAL_ROWS} val rows",
                f"{passing}/{total} stages; {missing} without recorded never/objective")


def _dos_objective(d: dict) -> float | None:
    """Validation objective of the checkpoint kept for a stage (normalised by E0)."""
    chosen = d.get("chosen_ckpt")
    for step, value in d.get("val_loss_per_ckpt") or []:
        if step == chosen:
            return float(value)
    return None


def _gate_dqn(ctx: _Ctx) -> Gate:
    counts_ok, finite_ok, n = True, True, 0
    for p in ctx.jobs:
        job = p["job"]
        spec = ctx.source.spec(job.inst, job.route_id)
        folds, _ = split_80_20(job.N)
        fit = np.sort(np.concatenate(folds))
        W = np.cumsum(_train(ctx, job).g[fit], axis=1)
        counts = stage_counts(W, spec.B, spec.m)[1:]
        trans = np.asarray(p["budget"]["transitions_per_stage"], dtype=np.int64)
        allowed = range(1, T9_PMAX + 1) if ctx.passes is None else (ctx.passes,)
        counts_ok = counts_ok and any(np.array_equal(trans, counts * P) for P in allowed)
        finite_ok = finite_ok and all(np.all(np.isfinite(np.asarray(v, dtype=float)))
                                      for net in p["models"]["nets"]
                                      for v in net["state"].values())
        n += 1
    return Gate("b_dqn", counts_ok and finite_ok, None,
                f"stored transitions = alive fitting pairs x passes (1..{T9_PMAX}); "
                "checkpoint finite",
                f"transitions ok={counts_ok}; finite={finite_ok}; jobs {n}")


def _ppo_masks_last_stage(net, m: int) -> bool:
    """B-PPO: the deployed head, after the stage mask, hands off with probability exactly 0
    at k = m (the head is run on a zero hidden state; the mask must not depend on it)."""
    with torch.no_grad():
        logits, _ = net.heads(torch.zeros(3, GRU_HIDDEN))
        probs = torch.softmax(mask_logits(logits, np.full(3, m), m), dim=1)
    return bool(torch.all(probs[:, HANDOFF] == 0.0))


def _gate_ppo(ctx: _Ctx) -> Gate:
    zero_ok, same_ok, finite_ok = True, True, True
    for p in ctx.jobs:
        job = p["job"]
        spec = ctx.source.spec(job.inst, job.route_id)
        models, pred = _pred(p)
        zero_ok = (zero_ok and set(models) == set(range(1, spec.m))
                   and _ppo_masks_last_stage(pred.net, spec.m))
        cache = _cache(spec, _dev(ctx, job), p)
        rows = np.arange(min(PPO_ROWS, cache.W.shape[0]))
        cached = [pred.predict(cache, k, rows) for k in range(1, spec.m)]
        fresh = [PolicyPredictor(pred.net, spec.E0).predict(cache, k, rows)
                 for k in range(1, spec.m)]
        gap = max(float(np.max(np.abs(a - b))) for a, b in zip(cached, fresh))
        same_ok = same_ok and gap <= EXACT_TOL * spec.E0
        finite_ok = finite_ok and all(np.all(np.isfinite(a)) for a in cached)
    return Gate("b_ppo", zero_ok and same_ok and finite_ok, None,
                "no predictor at k = m; cached == recomputed; finite",
                f"zero-at-m={zero_ok}; cached==recomputed={same_ok}; finite={finite_ok}")


ARM_GATE = {"thr": _gate_thr, "alpz": _gate_alpz, "rlsm": _gate_rlsm, "dos": _gate_dos,
            "dqn": _gate_dqn, "ppo": _gate_ppo}


# ---------------------------------------------------------------- reports

def gate_report(stage: str, arm: str, dev_df: pd.DataFrame, models_dir, rows_dir, *,
                source: DevSource | None = None, g2_keys=None, thr_models_dir=None,
                dos_tol: float = DOS_TOL, passes: int | None = None) -> dict:
    """Section-7 gates of one arm on dev, for its main-fit stage. ``dev_df`` holds the
    stage's arm rows and the ladder rows (none, oracle, baton_eq) of the same route-reps.
    ``passed`` is the family condition (1.3); ``dropped`` lists the failing gates as
    ``arm:gate``, ``reasons`` gives their details. ``g2_keys`` (default: rep 0 of every
    route) selects the route-reps of the uniform-LP identity, B-ALPZ (c)."""
    if arm not in B_ARMS:
        raise ValueError(f"arm must be one of {B_ARMS}, got {arm!r}")
    if core_of_stage(stage) != arm:
        raise ValueError(f"stage {stage!r} does not hold arm {arm!r}")
    require_dev(dev_df, rows_dir)
    jobs = load_jobs(models_dir)
    if g2_keys is None:
        g2_keys = [_key(p["job"]) for p in jobs if p["job"].rep == 0]
    ctx = _Ctx(stage=stage, arm=arm, jobs=jobs, source=source or DevSource(),
               rows_dir=Path(rows_dir), independent=stage.endswith("_ind"),
               g2_keys=frozenset(g2_keys),
               thr_dir=None if thr_models_dir is None else Path(thr_models_dir),
               dos_tol=float(dos_tol), passes=passes)
    arm_df = dev_df[dev_df["policy"] == arm]
    oracle_df = dev_df[dev_df["policy"] == "oracle"]
    gates = [_b0(ctx, arm_df, oracle_df), ARM_GATE[arm](ctx), _gate_p1(ctx)]
    return {"stage": stage, "arm": arm, "gates": [g.as_dict() for g in gates],
            "passed": all(g.passed for g in gates),
            "dropped": [f"{arm}:{g.id}" for g in gates if not g.passed],
            "reasons": {f"{arm}:{g.id}": g.detail for g in gates if not g.passed},
            "info": ctx.info}


def tuning_scores(tune_df: pd.DataFrame, arm: str, passes=(1, 2, 3)) -> dict:
    """Section 6.1 score per pass of one tuning run: mean over routes of
    100 (C - C_baton_eq) / C_baton_eq on the same rows."""
    base = _keyed(tune_df[tune_df["policy"] == "baton_eq"])
    out = {}
    for p in passes:
        cur = _keyed(tune_df[tune_df["policy"] == f"{arm}@p{p}"])
        keys = sorted(set(cur) & set(base))
        if not keys:
            raise ValueError(f"no tuning rows for {arm}@p{p}")
        out[p] = float(np.mean([100.0 * (cur[k] - base[k]) / base[k] for k in keys]))
    return out


def budget_label(scores: dict, *, p_main: int | None = None, p_star: int | None = None) -> dict:
    """Section 6.3: budget-limited if score(2) - score(3) > 0.5 points, or if the cut
    forces the main fit to p_main < p_star with score(p_main) - score(p_star) > 0.5."""
    still = scores[2] - scores[3] > T9_POINTS
    cut = (p_main is not None and p_star is not None and p_main < p_star
           and scores[p_main] - scores[p_star] > T9_POINTS)
    return {"budget_limited": bool(still or cut), "gain_2_3": float(scores[2] - scores[3]),
            "cut": bool(cut)}


def independence_items(ind_df: pd.DataFrame) -> dict:
    """Section 3.8 descriptive item: FULL against THR in the independent world."""
    thr = _keyed(ind_df[ind_df["policy"] == "thr"])
    full = _keyed(ind_df[ind_df["policy"] == FULL_POLICY])
    keys = sorted(set(thr) & set(full))
    if not keys:
        raise ValueError("independence rows hold no thr and nn_full pair")
    c_thr = float(np.mean([thr[k] for k in keys]))
    c_full = float(np.mean([full[k] for k in keys]))
    return {"full_vs_thr_pct": 100.0 * (c_full - c_thr) / c_thr, "routes": len(keys)}


def cut_arms(lists: dict) -> list:
    """Arms removed by a compute cut (section 9.2 (ii)): ``arm:cut`` entries of ``dropped``."""
    return [item.split(":")[0] for item in lists["dropped"] if item.endswith(CUT_SUFFIX)]


def derive_family(reports: dict, independence=INDEPENDENCE_STAGES,
                  references=REFERENCES, cut=()) -> dict:
    """The five FAMILY lists (1.3, 5.3). family = arms passing every bug gate and P1;
    descriptive = arms outside the family that were evaluated; dropped = failing gates as
    ``arm:gate``. An arm in ``cut`` (a compute cut, 9.2 (ii)) has no report: it is not
    fitted, not evaluated, and appears only as ``arm:cut`` in ``dropped``."""
    cut = tuple(cut)
    unknown = [a for a in cut if a not in B_ARMS]
    if unknown:
        raise ValueError(f"cut arms {unknown} are not B arms")
    reported = [a for a in cut if a in reports]
    if reported:
        raise ValueError(f"cut arms {reported} have a gate report; a cut arm is not fitted")
    missing = [a for a in B_ARMS if a not in reports and a not in cut]
    if missing:
        raise ValueError(f"no gate report for arms {missing}")
    family = [a for a in B_ARMS if a not in cut and reports[a]["passed"]]
    dropped = []
    for a in B_ARMS:
        if a in cut:
            dropped.append(f"{a}{CUT_SUFFIX}")
        else:
            dropped += [f"{a}:{g['id']}" for g in reports[a]["gates"] if not g["passed"]]
    return {"family": family,
            "descriptive": [a for a in B_ARMS if a not in family and a not in cut],
            "dropped": dropped,
            "references": list(references),
            "independence": sorted(independence)}


def dp_block(thr_dp: dict) -> dict:
    """THR's own-model DP values keyed ``inst|route|rep`` for the derivation block (3.8)."""
    return {f"{i}|{r}|{rep}": float(v) for (i, r, rep), v in sorted(thr_dp.items())}


def _derivation_block(reports: dict, lists: dict, labels: dict | None = None,
                      thr_dp: dict | None = None) -> dict:
    arms = {}
    for a in B_ARMS:
        if a in reports:
            arms[a] = {"passed": bool(reports[a]["passed"]),
                       "gates": {g["id"]: {"passed": g["passed"], "value": g["value"]}
                                 for g in reports[a]["gates"]}}
        else:
            arms[a] = {"cut": True, "passed": False, "gates": {}}
    block = {**lists, "arms": arms}
    if labels:
        block["budget_limited"] = {a: bool(v) for a, v in sorted(labels.items())}
    if thr_dp:
        block["thr_dp"] = dict(thr_dp)
    return block


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.6g}"


def _cell(text: str) -> str:
    return str(text).replace("|", "/").replace("`", "'")


def render_report(reports: dict, lists: dict, *, extra: dict | None = None,
                  labels: dict | None = None, thr_dp: dict | None = None) -> str:
    """fidelity_dev.md: the gate table, the budget labels and descriptive items, and the
    family-derivation JSON block parsed by gatecheck (EXPERIMENT_B 5.3). Deterministic."""
    lines = ["# Fidelity on dev (EXPERIMENT_B.md section 7)", "",
             "Dev rows only (eval_set dev). Generated by `python -m transvrp.expb.fidelity`.",
             "", "| arm | gate | verdict | value | threshold | detail |",
             "|---|---|---|---|---|---|"]
    for a in B_ARMS:
        for g in reports.get(a, {"gates": []})["gates"]:
            verdict = "pass" if g["passed"] else "FAIL"
            lines.append(f"| {a} | {g['id']} | {verdict} | {_fmt(g['value'])} | "
                         f"{_cell(g['threshold'])} | {_cell(g['detail'])} |")
    lines += ["", "## Family derivation", "",
              f"family={lists['family']}; descriptive={lists['descriptive']}; "
              f"dropped={lists['dropped']}; references={lists['references']}; "
              f"independence={lists['independence']}", ""]
    for name, value in sorted((extra or {}).items()):
        lines.append(f"- {name}: {json.dumps(value, sort_keys=True)}")
    block = json.dumps(_derivation_block(reports, lists, labels, thr_dp), sort_keys=True, indent=1)
    lines += ["", "```family-derivation", block, "```", ""]
    return "\n".join(lines)


def write_report(text: str, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    return out


# ---------------------------------------------------------------- CLI

def _parse_pairs(text: str) -> dict:
    if not text:
        return {}
    return {a.strip(): int(i) for a, i in (item.split("=") for item in text.split(","))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m transvrp.expb.fidelity")
    ap.add_argument("--results", default=str(RESULTS_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--chosen", default="", help="arm=config index, e.g. dqn=2,ppo=0")
    ap.add_argument("--p-main", type=int, default=T9_PMAX,
                    help="passes of the DQN main fit (B-DQN checks exactly this count)")
    ap.add_argument("--dos-tol", type=float, default=DOS_TOL,
                    help="B-DOS relative slack tau: obj <= min + tau |always - never|")
    ap.add_argument("--cut", default="", help="comma list of arms removed by a compute cut")
    args = ap.parse_args(argv)
    results = Path(args.results)
    ladder = pd.read_csv(results / "b_ladder.csv")
    cut = tuple(a for a in args.cut.split(",") if a)
    reports = {}
    for arm in (a for a in B_ARMS if a not in cut):
        stage = GATE_OF_ARM[arm]
        dev_df = pd.concat([pd.read_csv(results / f"{stage}.csv"), ladder], ignore_index=True)
        reports[arm] = gate_report(stage, arm, dev_df, results / "models" / stage,
                                   results / stage / "rows",
                                   thr_models_dir=results / "models" / "b_thr",
                                   dos_tol=args.dos_tol,
                                   passes=args.p_main if arm == "dqn" else None)
    lists = derive_family(reports, cut=cut)
    extra, labels = {}, {}
    for arm, idx in _parse_pairs(args.chosen).items():
        tune = pd.read_csv(results / f"b_tune_{arm}_c{idx}.csv")
        extra[f"budget_{arm}"] = budget_label(tuning_scores(tune, arm), p_main=args.p_main)
        labels[arm] = extra[f"budget_{arm}"]["budget_limited"]
    thr_dp = dp_block(reports["thr"].get("info", {}).get("thr_dp", {})) if "thr" in reports else None
    out = write_report(render_report(reports, lists, extra=extra, labels=labels,
                                     thr_dp=thr_dp),
                       Path(args.out) if args.out else results / "fidelity_dev.md")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

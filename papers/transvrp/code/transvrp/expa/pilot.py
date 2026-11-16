"""Pilot pass criteria P1-P6 (EXPERIMENT_A.md section 7.3) and stop rules (section 10).

``check_pilot`` takes the pilot's merged dev table (the M9 CSV layout, see
``sweep.ROW_COLUMNS``), the diagnostics of its fitted neural models
(``diag_table``) and optional extras (a P1 flag from the test run, a rerun table
for P2, the timing record of ``sweep.run_timing``). It returns each criterion's
status (PASS, FAIL, FLAG or NOT RUN) with the numbers behind it, and the stop
rules that can be evaluated from these inputs.

Interpretations that the spec leaves open are stated where they are made:

* P3 uses pooled ratios of route-averaged costs. The "pooled between-replicate
  SD" is the root mean square over routes of the BATON replicate SDs, divided by
  the mean route cost, so it is a relative SD.
* P6 pools all trained stages of the N = 10^4 jobs.
* Stop rule 3 is the S7 classification of section 1.5 without the Holm
  adjustment, and is only computable on a stage that has the N* = 10^5 classical
  10^6 rows; the pilot does not have them.
"""
from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis

PILOT_N = (1000, 10000)
P3_GATE_N = 10000
P3_MIN_TOL = 0.02          # section 7.3 P3: max(2%, 3 x SD)
P3_SD_MULT = 3.0
P3_FLAG = 0.01             # section 7.3 P3': SCALAR cheaper by more than 1%
P4_UPPER = 1.005           # section 7.3 P4
P6_MAX_SHARE = 0.5         # section 7.3 P6
ORACLE_TOL = 1e-9
RULE7_JOB_SHARE = 0.25     # section 10, stop rule 7
RULE7_STAGE_SHARE = 0.5
RULE5_ROUTES = 40
RULE5_INSTANCES = 15
CPU_CEILING_DAYS = 4.0     # section 7.2 platform rule
HARD_CEILING_DAYS = 7.0    # section 10, stop rule 6
WORKERS = 4
DATA_GEN_HOURS = 3.0       # section 7.2 table: data generation at 10^6 and fits, fixed part

# Section 7.2 plan: (arm, N, jobs). Jobs are route-replicates (186 = 62 routes x 3).
SECTION_72_PLAN = (
    ("full", 100, 186), ("full", 500, 186), ("full", 2000, 186),
    ("full", 10000, 186), ("full", 100000, 186),
    ("scalar", 2000, 186), ("scalar", 100000, 186),
    ("mlp", 100000, 186),
    ("full", 1000000, 12), ("scalar", 1000000, 12),
)

KEY = ["inst", "route_id", "N", "rep", "policy"]


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------- diagnostics table

def diag_table(results_dir: Path, stage: str) -> pd.DataFrame:
    """One row per neural stage entry and per fold-disagreement entry of every
    neural model of a stage (``models/<stage>/*.pkl``). Columns include inst,
    route_id, N, rep, arm, entry ('stage' or 'ls'), k, fold, action, chosen_ckpt,
    steps, select_on, n_fit, n_val, disagree_rate."""
    from .sweep import Job  # noqa: F401  (unpickling needs the class)

    rows = []
    for path in sorted((Path(results_dir) / "models" / stage).glob("*.pkl")):
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("kind") != "neural":
            continue
        job = payload["job"]
        for e in payload["diag"]:
            e = dict(e)
            entry = e.pop("kind")
            for drop in ("val_mse_per_ckpt", "train_mse_first", "train_mse_last", "arm"):
                e.pop(drop, None)               # the job's arm is authoritative
            rows.append(dict(inst=job.inst, route_id=job.route_id, N=job.N,
                             rep=job.rep, arm=job.arm, entry=entry, **e))
    cols = ["inst", "route_id", "N", "rep", "arm", "entry", "k", "fold", "action",
            "chosen_ckpt", "steps", "select_on", "n_fit", "n_val", "n_applied",
            "disagree_rate"]
    return pd.DataFrame(rows).reindex(columns=cols)


def _trained(diag: pd.DataFrame) -> pd.DataFrame:
    st = diag[(diag["entry"] == "stage") & diag["action"].isin(["warm", "cold"])]
    return st


# ---------------------------------------------------------------- criteria

def p1_status(tests_green: bool | None) -> dict:
    """P1: all tests green, including M3/M6 parity and GOLD (supplied by the
    caller from the pytest run; this module cannot run it)."""
    if tests_green is None:
        return dict(id="P1", status="NOT RUN", detail="pass --tests-green after the suite passes")
    return dict(id="P1", status=_status(bool(tests_green)),
                detail=f"tests green: {bool(tests_green)}")


def p2_status(df: pd.DataFrame, rerun: pd.DataFrame | None) -> dict:
    """P2: rerunning one cell gives bit-identical dev costs."""
    if rerun is None:
        return dict(id="P2", status="NOT RUN", detail="no rerun table given")
    a = df.set_index(KEY)["mean_cost"]
    b = rerun.set_index(KEY)["mean_cost"]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return dict(id="P2", status="FAIL", detail="rerun shares no rows with the pilot table")
    same = bool(np.array_equal(a.loc[common].to_numpy(), b.loc[common].to_numpy()))
    maxdiff = float(np.max(np.abs(a.loc[common].to_numpy() - b.loc[common].to_numpy())))
    return dict(id="P2", status=_status(same), rows=int(len(common)), max_abs_diff=maxdiff,
                detail=f"{len(common)} rows compared, max |diff| = {maxdiff:.3e}")


def _pooled_scalar_vs_baton(df: pd.DataFrame, N: int) -> dict | None:
    s = df[(df["policy"] == "nn_scalar") & (df["N"] == N)]
    b = df[(df["policy"] == "baton_eq") & (df["N"] == N)]
    if s.empty or b.empty:
        return None
    ms = s.groupby(["inst", "route_id"])["mean_cost"].mean()
    mb = b.groupby(["inst", "route_id"])["mean_cost"].mean()
    keep = ms.index.intersection(mb.index)
    if len(keep) == 0:
        return None
    # pooled between-replicate SD of BATON-eq (relative to the mean route cost)
    var_r = b.groupby(["inst", "route_id"])["mean_cost"].var(ddof=1)
    var_r = var_r.loc[var_r.index.intersection(keep)]
    sd_pooled = math.sqrt(float(var_r.mean()))
    mean_b = float(mb.loc[keep].mean())
    return dict(N=N, n_routes=int(len(keep)),
                c_scalar=float(ms.loc[keep].mean()), c_baton=mean_b,
                sd_rel_baton=sd_pooled / mean_b)


def p3_status(df: pd.DataFrame) -> dict:
    """P3 (gate) and P3' (flag) at N = 10^4."""
    r = _pooled_scalar_vs_baton(df, P3_GATE_N)
    if r is None:
        return dict(id="P3", status="NOT RUN", detail="nn_scalar or baton_eq missing at N = 10^4")
    ratio = r["c_scalar"] / r["c_baton"]
    tol = max(P3_MIN_TOL, P3_SD_MULT * r["sd_rel_baton"])
    threshold = 1.0 + tol
    ok = ratio <= threshold
    gain = 1.0 - ratio
    flag = gain > P3_FLAG
    return dict(id="P3", status=_status(ok), ratio=ratio, threshold=threshold,
                tolerance=tol, sd_rel_baton=r["sd_rel_baton"], n_routes=r["n_routes"],
                c_scalar=r["c_scalar"], c_baton=r["c_baton"],
                flag_id="P3'", flag="FLAG" if flag else "PASS",
                flag_detail=f"SCALAR is {100 * gain:+.2f}% cheaper than BATON-eq at N = 10^4 "
                            f"(flag above +{100 * P3_FLAG:.0f}%: audit the M3 index logs)",
                detail=f"SCALAR / BATON-eq = {ratio:.5f}, threshold {threshold:.5f} "
                       f"(tolerance max(2%, 3 SD) = {100 * tol:.2f}%)")


def p4_status(df: pd.DataFrame) -> dict:
    """P4: on every route and cell, C_oracle <= C_pi <= 1.005 C_none for the
    neural arms and BATON-eq (pi), over every cell where none and oracle exist."""
    key = ["inst", "route_id", "N", "rep"]
    piv = df.pivot_table(index=key, columns="policy", values="mean_cost", aggfunc="first")
    if not {"none", "oracle"} <= set(piv.columns):
        return dict(id="P4", status="NOT RUN", detail="none or oracle rows missing")
    checked = 0
    lower, upper = [], []
    for pol in ("nn_full", "nn_scalar", "baton_eq"):
        if pol not in piv.columns:
            continue
        sub = piv[[pol, "none", "oracle"]].dropna()
        checked += len(sub)
        bad_lo = sub[sub[pol] < sub["oracle"] - ORACLE_TOL]
        bad_hi = sub[sub[pol] > P4_UPPER * sub["none"]]
        lower += [(*idx, pol) for idx in bad_lo.index]
        upper += [(*idx, pol) for idx in bad_hi.index]
    ok = not lower and not upper
    return dict(id="P4", status=_status(ok), checked=int(checked),
                lower_violations=len(lower), upper_violations=len(upper),
                examples=(lower + upper)[:5],
                detail=f"{checked} (route, cell, policy) triples; {len(lower)} below oracle, "
                       f"{len(upper)} above 1.005 x none")


W_PLAN_MEAN = 117.0        # section 7.2: mean eligible route, sum over stages of (m - k + 1)
MLP_TO_FULL = 0.2          # section 7.2: "MLP, 10^5 x 186 (~1/5 cost)"; a prior, not measured


def route_work(m: int) -> float:
    """Work of one route pass, sum_{k=1..m} (m - k + 1) = m(m + 1) / 2 (section 7.2)."""
    return m * (m + 1) / 2.0


def p5_status(df: pd.DataFrame, timing: dict | None) -> dict:
    """P5: seconds per stage and per job are recorded, and the platform rule is
    applied to the projection (section 7.2). The projection is always computed
    from the cells this pilot measures; cells it does not measure are extrapolated
    by the rule in ``plan_cells`` and labelled as such."""
    recorded = bool(df["fit_seconds"].notna().all() and (df["fit_seconds"] > 0).all())
    minutes = job_minutes(df, timing)
    cells = plan_cells(df, timing)
    proj = projection_hours({k: c["minutes"] for k, c in cells.items()
                             if c["minutes"] is not None})
    proj["cells"] = {f"{arm}@{N}": c for (arm, N), c in cells.items()}
    proj["measured"] = sorted(f"{arm}@{N}" for (arm, N), c in cells.items()
                              if c["source"] == "measured")
    proj["inputs"] = {f"{arm}@{N}": v for (arm, N), v in sorted(_measured_plan_minutes(df).items())}
    decision = None
    if proj["hours"] is not None:
        proj["days"] = proj["hours"] / 24.0
        decision = platform_decision(proj["days"])
    ok = recorded and timing is not None and proj["hours"] is not None
    detail = (f"fit_seconds recorded on all rows: {recorded}; "
              f"cells measured: {len(proj['measured'])}, extrapolated: "
              f"{len(cells) - len(proj['measured'])}")
    if proj["hours"] is not None:
        detail += f"; projection {proj['days']:.2f} days -> {decision}"
    else:
        detail += f"; projection not computable, missing: {', '.join(proj['missing'])}"
    if timing is None:
        detail += "; no timing record given"
    return dict(id="P5", status=_status(ok), minutes_per_job=minutes, projection=proj,
                decision=decision, detail=detail)


def job_minutes(df: pd.DataFrame, timing: dict | None) -> dict:
    """Mean fit minutes per job by (arm, N) from the table, raw (per route as
    measured); the timing record adds (arm, 10^6) with its total minutes."""
    out: dict = {}
    d = df.copy()
    d = d[d["policy"].isin(["nn_full", "nn_scalar", "nn_mlp"])]
    arm_of = {"nn_full": "full", "nn_scalar": "scalar", "nn_mlp": "mlp"}
    for (pol, N), g in d.groupby(["policy", "N"]):
        out[(arm_of[pol], int(N))] = float(g["fit_seconds"].mean()) / 60.0
    if timing is not None:
        out[(timing["arm"], int(timing["n_rows"]))] = float(timing["total_seconds"]) / 60.0
    return out


def _measured_plan_minutes(df: pd.DataFrame) -> dict:
    """(arm, N) -> minutes per plan job, from the measured fits: each job's minutes
    are divided by its route work, then the mean is taken and multiplied by
    W_PLAN_MEAN (per average eligible route)."""
    arm_of = {"nn_full": "full", "nn_scalar": "scalar", "nn_mlp": "mlp"}
    d = df[df["policy"].isin(list(arm_of))].copy()
    d["arm"] = d["policy"].map(arm_of)
    d["rate"] = d["fit_seconds"] / 60.0 / d["m"].map(lambda m: route_work(int(m)))
    g = d.groupby(["arm", "N"])["rate"].mean() * W_PLAN_MEAN
    return {(arm, int(N)): float(v) for (arm, N), v in g.items()}


def plan_cells(df: pd.DataFrame, timing: dict | None, plan=SECTION_72_PLAN) -> dict:
    """Minutes per job for every cell of the section 7.2 plan, with its source.

    Projection rule (section 7.2 gives the plan and the 186-job count but not how
    unmeasured cells are extrapolated, so this module fixes the rule):

    1. Measured cells: mean fit minutes per job, scaled to the mean eligible route
       (route work m(m+1)/2, see ``route_work``; mean 117 over the 62 routes).
    2. FULL at an unmeasured N: a line a + b N fitted by least squares to the
       measured FULL cells. Training is capped at S_k steps, so a carries the
       N-independent training cost and b the per-path cost (generation, features).
    3. SCALAR at an unmeasured N: the mean measured SCALAR / FULL ratio at the N
       where both are measured, times the FULL cell.
    4. MLP: 0.2 x FULL (section 7.2 prior, not measured).
    5. Cells at 10^6 add the stop-update minutes of the timing record, scaled the
       same way. The 10^6 data generation stays in the fixed DATA_GEN_HOURS term,
       as in the section 7.2 table.

    Returns {(arm, N): {"minutes": float or None, "source": str}}."""
    measured = _measured_plan_minutes(df)
    full_pts = {N: v for (arm, N), v in measured.items() if arm == "full"}
    line = None
    if len(full_pts) >= 2:
        slope, intercept = np.polyfit(np.array(list(full_pts), dtype=float),
                                      np.array(list(full_pts.values())), 1)
        line = (float(intercept), float(slope))
    both = [N for N in full_pts if ("scalar", N) in measured]
    ratio = (float(np.mean([measured[("scalar", N)] / full_pts[N] for N in both]))
             if both else None)
    stop_min = None
    if timing is not None:
        stop_sec = timing.get("stop_seconds", timing["total_seconds"])
        stop_min = float(stop_sec) / 60.0 / route_work(int(timing["m"])) * W_PLAN_MEAN

    def fit_minutes(arm: str, N: int):
        if (arm, N) in measured:
            return measured[(arm, N)], "measured"
        if arm == "full":
            if line is None:
                return None, "missing (needs two measured FULL cells)"
            a, b = line
            return a + b * N, "model: a + b N"
        if arm == "scalar":
            if ratio is None:
                return None, "missing (no SCALAR/FULL overlap)"
            v, _ = fit_minutes("full", N)
            return (None, "missing") if v is None else (ratio * v, "ratio x FULL")
        if arm == "mlp":
            v, _ = fit_minutes("full", N)
            return (None, "missing") if v is None else (MLP_TO_FULL * v, "0.2 x FULL (prior)")
        raise ValueError(f"unknown arm {arm!r}")

    cells: dict = {}
    for arm, N, _ in plan:
        v, src = fit_minutes(arm, N)
        if v is not None and N == 10 ** 6:
            if stop_min is None:
                v, src = None, "missing (no timing record)"
            else:
                v, src = v + stop_min, src + " + stop updates (timing)"
        cells[(arm, N)] = dict(minutes=v, source=src)
    return cells


def projection_hours(minutes: dict, plan=SECTION_72_PLAN, workers: int = WORKERS) -> dict:
    """Wall hours of the section 7.2 plan on ``workers`` processes, from minutes per
    job for every plan cell. None when a plan cell is missing from ``minutes``."""
    missing = [f"{arm}@{N}" for arm, N, _ in plan if (arm, N) not in minutes]
    if missing:
        return dict(hours=None, missing=missing)
    job_min = sum(minutes[(arm, N)] * n for arm, N, n in plan)
    return dict(hours=job_min / workers + DATA_GEN_HOURS, missing=[], job_minutes=job_min)


def platform_decision(days: float) -> str:
    """Section 7.2 platform rule and section 10 stop rule 6."""
    if days > HARD_CEILING_DAYS:
        return "STOP (stop rule 6: beyond 7 days after all cuts)"
    if days <= CPU_CEILING_DAYS:
        return "CPU"
    return "GPU (Colab) or cut 1 (S_k cap 200), logged"


def p6_status(diag: pd.DataFrame, N: int = 10000) -> dict:
    """P6: at N = 10^4 the selected checkpoint is the warm start on < 50% of the
    trained stages. Also the mean fold-disagreement rate at that N."""
    tr = _trained(diag)
    tr = tr[tr["N"] == N]
    if tr.empty:
        return dict(id="P6", status="NOT RUN", detail=f"no trained stages at N = {N}")
    share = float((tr["chosen_ckpt"] == 0).mean())
    ls = diag[(diag["entry"] == "ls") & (diag["N"] == N)]
    dis = float(ls["disagree_rate"].mean()) if len(ls) else float("nan")
    # stages that select on training MSE (no validation row alive): section 8 fallback
    no_val = int((tr["select_on"] == "train").sum())
    return dict(id="P6", status=_status(share < P6_MAX_SHARE), warm_share=share,
                n_stages=int(len(tr)), disagree_rate=dis, stages_select_on_train=no_val,
                detail=f"warm-start checkpoint on {100 * share:.1f}% of {len(tr)} trained "
                       f"stages (limit < 50%); mean fold disagreement {100 * dis:.2f}%; "
                       f"{no_val} stages selected on training MSE (no validation row alive)")


# ---------------------------------------------------------------- stop rules

MONO_N = 10000             # largest SCALAR cell of the pilot (N* = 10^5 is not in the pilot)


def monotonicity_for_stage(results_dir: Path, stage: str, N: int = MONO_N,
                           arm: str = "scalar") -> dict | None:
    """Section 4.2 diagnostic for the fitted ``arm`` models at ``N`` of a stage: the
    W-grid curves of every route and replicate (dev pool for the route minimum of
    W_k), pooled over jobs. Each curve keeps its own route's E0. Returns None when
    no such model exists."""
    from .analysis import MONO_TOL, monotonicity_curves, monotonicity_diagnostic
    from .paths import route_pool
    from .sweep import _spec_of, _unpack

    files = sorted((Path(results_dir) / "models" / stage).glob(f"{arm}__*__N{N}__rep*.pkl"))
    tot = dict(n_curves=0, n_steps=0, n_steps_bad=0, curves_bad=0)
    jobs = 0
    for path in files:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("kind") != "neural":
            continue
        job = payload["job"]
        spec = _spec_of(job.inst, job.route_id)
        g = route_pool(job.inst, "dev", routes=[job.route_id])[job.route_id].g
        curves = monotonicity_curves(_unpack(payload["models"]), spec, payload["stats"], g)
        d = monotonicity_diagnostic(curves, spec.E0)
        jobs += 1
        tot["n_curves"] += d["n_curves"]
        tot["n_steps"] += d["n_steps"]
        tot["n_steps_bad"] += d["n_steps_bad"]
        tot["curves_bad"] += int(round(d["frac_curves"] * d["n_curves"])) if d["n_curves"] else 0
    if jobs == 0:
        return None
    nc, ns = tot["n_curves"], tot["n_steps"]
    return dict(stage=stage, arm=arm, N=int(N), jobs=jobs, tol=MONO_TOL, **tot,
                frac_curves=(tot["curves_bad"] / nc) if nc else float("nan"),
                frac_steps=(tot["n_steps_bad"] / ns) if ns else float("nan"))


def stop_rule_2(df: pd.DataFrame) -> dict:
    """Any policy's mean cost below the oracle on any route."""
    key = ["inst", "route_id", "N", "rep"]
    piv = df.pivot_table(index=key, columns="policy", values="mean_cost", aggfunc="first")
    if "oracle" not in piv.columns:
        return dict(rule=2, triggered=None, detail="no oracle rows")
    bad = 0
    for pol in piv.columns:
        if pol == "oracle":
            continue
        sub = piv[[pol, "oracle"]].dropna()
        bad += int((sub[pol] < sub["oracle"] - ORACLE_TOL).sum())
    return dict(rule=2, triggered=bad > 0, violations=bad,
                detail=f"{bad} (route, cell, rep, policy) below the oracle")


def stop_rule_3(df: pd.DataFrame, N: int = 100000) -> dict:
    """SCALAR 'better' than DP-1e6 at N* on dev (section 1.5 classification of S7,
    unadjusted). Needs nn_scalar at N and the 10^6 classical rows."""
    pols = set(zip(df["policy"], df["N"]))
    if ("nn_scalar", N) not in pols or ("dp_big", 1000000) not in pols:
        return dict(rule=3, triggered=None,
                    detail="not applicable: N* rows of SCALAR and DP-1e6 are absent")
    d = analysis.instance_deltas(df, "nn_scalar", "dp_big", N, N_b=1000000).to_numpy()
    cls = analysis.classify(d)
    return dict(rule=3, triggered=cls == "better", classification=cls,
                detail=f"SCALAR vs DP-1e6 at N = {N}: {cls} (audit the M3 logs if better)")


def stop_rule_5(elig: pd.DataFrame) -> dict:
    """Fewer than 40 eligible routes, or fewer than 15 instances with one."""
    el = elig[elig["eligible"].astype(bool)]
    n_routes = int(len(el))
    n_inst = int(el["inst"].nunique())
    trig = n_routes < RULE5_ROUTES or n_inst < RULE5_INSTANCES
    return dict(rule=5, triggered=trig, eligible_routes=n_routes, instances=n_inst,
                detail=f"{n_routes} eligible routes over {n_inst} instances")


def stop_rule_6(p5: dict) -> dict:
    proj = p5["projection"]
    if proj.get("hours") is None:
        return dict(rule=6, triggered=None, detail="projection not computable yet")
    days = proj["hours"] / 24.0
    return dict(rule=6, triggered=days > HARD_CEILING_DAYS, days=days,
                detail=f"projection {days:.2f} days (ceiling 7 after cuts)")


def stop_rule_7(diag: pd.DataFrame, N_min: int = 10000) -> dict:
    """Warm-start checkpoint selected on > 50% of the trained stages of a job, for
    more than 25% of the jobs at N >= 10^4."""
    tr = _trained(diag)
    tr = tr[tr["N"] >= N_min]
    if tr.empty:
        return dict(rule=7, triggered=None, detail=f"no trained stages at N >= {N_min}")
    job_cols = ["inst", "route_id", "N", "rep", "arm"]
    per_job = tr.groupby(job_cols)["chosen_ckpt"].apply(lambda x: float((x == 0).mean()))
    share_jobs = float((per_job > RULE7_STAGE_SHARE).mean())
    return dict(rule=7, triggered=share_jobs > RULE7_JOB_SHARE, job_share=share_jobs,
                jobs=int(len(per_job)),
                detail=f"{100 * share_jobs:.1f}% of {len(per_job)} jobs at N >= {N_min} "
                       f"have warm-start on > 50% of stages (rule: > 25%)")


# ---------------------------------------------------------------- driver

def check_pilot(df: pd.DataFrame, diag: pd.DataFrame, *, tests_green: bool | None = None,
                rerun: pd.DataFrame | None = None, timing: dict | None = None,
                eligibility: pd.DataFrame | None = None,
                mono: dict | None = None) -> dict:
    """Evaluates P1-P6 and the stop rules on the pilot tables. Nothing is written.
    ``mono`` is the section 4.2 diagnostic (``monotonicity_for_stage``); it is
    reported, not a criterion."""
    p3 = p3_status(df)
    p5 = p5_status(df, timing)
    crit = dict(P1=p1_status(tests_green), P2=p2_status(df, rerun), P3=p3,
                P4=p4_status(df), P5=p5, P6=p6_status(diag))
    rules = dict(rule2=stop_rule_2(df), rule3=stop_rule_3(df), rule6=stop_rule_6(p5),
                 rule7=stop_rule_7(diag))
    if eligibility is not None:
        rules["rule5"] = stop_rule_5(eligibility)
    return dict(criteria=crit, stop_rules=rules, monotonicity=mono, timing=timing,
                ok=all(c["status"] == "PASS" for c in crit.values()))


def render_markdown(res: dict) -> str:
    lines = ["# EXPERIMENT A pilot report (section 7.3)", "",
             "| criterion | status | detail |", "|---|---|---|"]
    for name, c in res["criteria"].items():
        lines.append(f"| {name} | {c['status']} | {c.get('detail', '')} |")
    p3 = res["criteria"]["P3"]
    if p3.get("flag") is not None:
        lines.append(f"| P3' (flag) | {p3['flag']} | {p3.get('flag_detail', '')} |")
    proj = res["criteria"]["P5"].get("projection") or {}
    if proj.get("cells"):
        lines += ["", "## P5 projection (section 7.2 plan, minutes per job)", "",
                  f"Rule: see `plan_cells` in pilot.py. Scaled to the mean eligible route "
                  f"(work {W_PLAN_MEAN:.0f}). Hours = job minutes / {WORKERS} workers + "
                  f"{DATA_GEN_HOURS:g} h data generation.", "",
                  "| cell | minutes per job | source |", "|---|---|---|"]
        for name, c in proj["cells"].items():
            val = "not computable" if c["minutes"] is None else f"{c['minutes']:.2f}"
            lines.append(f"| {name} | {val} | {c['source']} |")
        if proj.get("inputs"):
            lines += ["", "Measured inputs (scaled as above, including cells outside the plan): "
                      + ", ".join(f"{k} = {v:.2f}" for k, v in proj["inputs"].items()) + "."]
        if proj.get("hours") is not None:
            lines += ["", f"Projection: {proj['hours']:.1f} h = {proj['days']:.2f} days "
                          f"-> {res['criteria']['P5'].get('decision')}."]
    tim = res.get("timing")
    if tim:
        def mb(key):
            return f"{tim[key]:.0f} MB" if key in tim else "not recorded"
        lines += ["", "## Timing (10^6 pass)", "",
                  f"- Peak memory, headline (process RSS high-water mark, whole run): "
                  f"**{mb('process_peak_rss_mb')}**.",
                  f"- Secondary (tracemalloc, NumPy buffers only): cache plus stop updates peak "
                  f"{mb('traced_cache_stop_peak_mb')}; generation peak "
                  f"{mb('traced_generate_peak_mb')}."]
        if "total_seconds" in tim:
            lines.append(f"- Seconds: generation {tim.get('generate_seconds', float('nan')):.1f}, "
                         f"cache {tim.get('cache_seconds', float('nan')):.1f}, stop updates "
                         f"{tim.get('stop_seconds', float('nan')):.1f}, total "
                         f"{tim['total_seconds']:.1f}.")
    mono = res.get("monotonicity")
    if mono:
        lines += ["", "## Monotonicity diagnostic (section 4.2, reported)", "",
                  f"{mono['arm'].upper()} at N = {mono['N']:,} over {mono['jobs']} jobs "
                  f"({mono['n_curves']} W-grid curves): "
                  f"{100 * mono['frac_curves']:.1f}% of curves have a neighbour decrease "
                  f"above {mono['tol']} E0; {100 * mono['frac_steps']:.2f}% of neighbour steps "
                  f"do. (Section 4.2 names N*; the pilot's largest SCALAR cell is used.)", ""]
    lines += ["", "## Stop rules (section 10)", "", "| rule | triggered | detail |", "|---|---|---|"]
    for name, r in res["stop_rules"].items():
        trig = {True: "YES", False: "no", None: "not applicable"}[r.get("triggered")]
        lines.append(f"| {r.get('rule', name)} | {trig} | {r.get('detail', '')} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import json

    ap = argparse.ArgumentParser(prog="python -m transvrp.expa.pilot")
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--stage", default="pilot")
    ap.add_argument("--tests-green", action="store_true", default=None)
    ap.add_argument("--rerun", type=Path, default=None, help="dev CSV of one rerun cell (P2)")
    ap.add_argument("--timing", type=Path, default=None, help="timing JSON from sweep timing")
    ap.add_argument("--eligibility", type=Path, default=None, help="M8 table (stop rule 5)")
    args = ap.parse_args(argv)
    from .sweep import merge_stage

    df = merge_stage(args.stage, "dev", args.results_dir)
    diag = diag_table(args.results_dir, args.stage)
    rerun = pd.read_csv(args.rerun) if args.rerun else None
    timing = json.loads(args.timing.read_text()) if args.timing else None
    elig = pd.read_csv(args.eligibility) if args.eligibility else None
    mono = monotonicity_for_stage(args.results_dir, args.stage)
    res = check_pilot(df, diag, tests_green=True if args.tests_green else None,
                      rerun=rerun, timing=timing, eligibility=elig, mono=mono)
    out = Path(args.results_dir) / f"{args.stage}_report.md"
    out.write_text(render_markdown(res))
    print({k: v["status"] for k, v in res["criteria"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

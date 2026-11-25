"""EXPERIMENT C analysis of the test_c stream (EXPERIMENT_C.md sections 1.5-1.9, 5, 7; MC5).

Reads one merged ``<stage>.test_c.csv`` per C stage and writes family C and the descriptive
tables of 1.9 to ``results/expC/analysis_test_c/``. It refuses unless every row is from
test_c, every row carries the family hash of HEAD's TEST_C FAMILY line, and the analysis
files are the committed ones (``gatecheck.check_analysis_clean``). With ``--eval dev`` it
runs on the dev tables for the pilot: no family check, and every output is labelled DEV.

Statistics are A's (``expa.analysis``: contrast, instance deltas, coverage, Holm, classify,
the cluster bootstrap); the Holm-level interval is B's ``analysis_b.holm_ci``.

Run from ``papers/transvrp/code``: ``python -m transvrp.expc.analysis_c --eval test_c``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..expa import analysis as A
from ..expa import paths
from ..expa import sweep
from ..expb import analysis_b as B
from ..expb import gatecheck as G
from . import gates as CG
from .stages import c_stage_defs

EVAL_SETS = ("test_c", "dev")
TEST_C_SUFFIX = ".test_c.csv"
N_PATHS = 100_000                    # every C stage at 10^5 paths (section 3)
N_INSTANCES = 20                     # unit of analysis (section 7)
N_ROUTES = 62                        # eligible routes (section 1.4 of A)
N_REPS = 2                           # reps 0-1 (section 3)
SUB_INSTANCES = 10                   # PPO-long subset: 12 routes from 10 instances (1.8)
SUB_ROUTES = 12
SUB_REPS = 1                         # rep 0 only
SHARE_DRAWS = 10_000                 # share-of-gain bootstrap (1.5)
SHARE_SEED = 20261125
SIGMA_FLOOR = 1e-3                   # exclusion floor of the summed-cost estimand (1.5)
HOLM_SIZE = 11                       # family C (1.9)
ALPHA_HOLM = B.HOLM_FWER
WORLD_TAGS = CG.WORLD_TAGS           # ("w0", "w3", "w6", "w9", "ws"), from expc.stages
STAGE_COLUMNS = ("stage",)
REQUIRED = tuple(B.REQUIRED_COLUMNS) + ("world", "feature_set")
FAMILY_COLUMNS = ("id", "role", "a", "b", "world", "margin", "n", "mean", "median",
                  "ci95_lo", "ci95_hi", "holm_lo", "holm_hi", "ci90_lo", "ci90_hi",
                  "n_better", "p_raw", "p_holm", "classification", "holm_k")
SHARE_COLUMNS = ("learner", "feature_set", "n", "share", "share_ci95_lo", "share_ci95_hi",
                 "delta_vs_f0", "ci95_lo", "ci95_hi", "p_raw")
SIGMA_COLUMNS = ("world", "contrast", "n", "included", "excluded", "mean", "ci95_lo",
                 "ci95_hi")
CDP_COLUMNS = ("world", "comparison", "n", "mean", "median", "ci95_lo", "ci95_hi",
               "n_better", "p_raw")
PPO_COLUMNS = ("comparison", "n", "mean", "median", "ci95_lo", "ci95_hi", "n_better",
               "p_raw")
GAP_COLUMNS = ("n", "excluded", "mean_share", "ci95_lo", "ci95_hi")

# (contrast id, a stage, a policy, b stage, b policy); the family of 1.9
FAMILY = tuple((c.cid, c.a_stage, c.a_policy, c.b_stage, c.b_policy, c.margin)
               for c in CG.CONTRASTS)
LEARNERS = (("boost", "boost"), ("rlsm", "rlsm"))
SHARE_FEATURES = ("F0n", "F1", "F2", "F3")


class AnalysisRefused(RuntimeError):
    """The analysis refuses: its inputs are not what the pinned family allows."""


def _refuse(message: str):
    raise AnalysisRefused(message)


def family_hash(family_line: str) -> str:
    """sha256 of the TEST_C FAMILY line, as written into every test_c row."""
    return hashlib.sha256(family_line.encode("utf-8")).hexdigest()


def required_stages() -> list[str]:
    """Every stage the tables use: the stages of the spec table (``stages.c_stage_defs``,
    reused stages included) and the stages named by the family."""
    names = set(c_stage_defs()) | {c[1] for c in FAMILY} | {c[3] for c in FAMILY}
    return sorted(names)


# ---------------------------------------------------------------- input

def stage_of_file(path, eval_set: str) -> str:
    name = Path(path).name
    suffix = TEST_C_SUFFIX if eval_set == "test_c" else ".csv"
    if not name.endswith(suffix):
        _refuse(f"{name}: not a {eval_set} stage table")
    return name[: -len(suffix)]


def load_tables(paths_, *, eval_set: str, family_sha256: str | None = None) -> pd.DataFrame:
    """One frame of the stage tables with a ``stage`` column. For test_c: refuses rows not
    from test_c, rows whose family hash differs from the pinned line, day counts other
    than 20,000, and several eval commits."""
    stages = [stage_of_file(p, eval_set) for p in paths_]
    if not stages:
        _refuse("no stage tables given")
    frames = []
    for path, stage in zip(paths_, stages):
        df = pd.read_csv(path, dtype={"family_sha256": str, "fit_commit": str,
                                      "eval_commit": str, "inst": str})
        required = REQUIRED if eval_set == "test_c" else tuple(
            c for c in REQUIRED if c != "family_sha256")
        missing = [c for c in required if c not in df.columns]
        if missing:
            _refuse(f"{stage}: columns missing: {missing}")
        if df.empty:
            _refuse(f"{stage}: no rows")
        if not (df["eval_set"].astype(str) == eval_set).all():
            _refuse(f"{stage}: rows not from eval_set {eval_set!r}")
        if eval_set == "test_c":
            if family_sha256 is None:
                _refuse("test_c tables need the family hash of HEAD's TEST_C FAMILY line")
            if not (df["family_sha256"].astype(str) == family_sha256).all():
                _refuse(f"{stage}: family_sha256 differs from the TEST_C FAMILY line")
            if not (df["n_eval"] == paths.TEST_C_N).all():
                _refuse(f"{stage}: day count differs from {paths.TEST_C_N}")
        frames.append(df.assign(stage=stage))
    out = pd.concat(frames, ignore_index=True)
    if eval_set == "test_c" and out["eval_commit"].astype(str).nunique() != 1:
        _refuse("rows from several eval commits; exactly one is allowed (5.3)")
    return out


def check_complete(df: pd.DataFrame, stages=None) -> None:
    """Every stage the tables use is present."""
    absent = sorted(set(required_stages() if stages is None else stages) - set(df["stage"]))
    if absent:
        _refuse(f"stage tables missing: {absent}")


def _pair(df: pd.DataFrame, left: tuple, right: tuple, *, rep=None, routes=None) -> pd.DataFrame:
    """Rows of two (stage, policy) pairs under the names A and B, optionally restricted to
    one replicate and a set of (inst, route_id), so that A's statistics compare exactly
    the two."""
    out = []
    for name, (stage, policy) in (("A", left), ("B", right)):
        sub = df[(df["stage"] == stage) & (df["policy"] == policy)]
        if rep is not None:
            sub = sub[sub["rep"] == rep]
        if routes is not None:
            keep = [(str(i), int(r)) in routes for i, r in zip(sub["inst"], sub["route_id"])]
            sub = sub[np.asarray(keep, dtype=bool)]
        if sub.empty:
            _refuse(f"no rows for {stage}/{policy}")
        out.append(sub.assign(policy=name))
    return pd.concat(out, ignore_index=True)


def _checked(df: pd.DataFrame, cid: str, left: tuple, right: tuple, *, delta: float,
             n_instances: int, n_routes: int, n_reps: int, rep=None, routes=None,
             p_adj: float | None = None) -> dict:
    """A's contrast after A's coverage check; a coverage failure is a refusal."""
    pair = _pair(df, left, right, rep=rep, routes=routes)
    try:
        A.check_coverage(pair, (cid, "A", N_PATHS, "B", N_PATHS), n_instances, n_routes,
                         n_reps)
    except ValueError as exc:
        _refuse(str(exc))
    return A.contrast(pair, cid, "A", N_PATHS, "B", N_PATHS, delta, p_adj=p_adj)


# ---------------------------------------------------------------- family C

def default_lists() -> dict:
    """Family lists when no pinned line exists (dev pilot): all eleven contrasts."""
    return {"family": [c[0] for c in FAMILY], "descriptive": [], "dropped": [],
            "references": list(CG.REFERENCES)}


def family_lists(family_line: str) -> dict:
    """The family, descriptive, dropped and references lists of a pinned TEST_C FAMILY line
    (parsed by the stream's own pattern, EXPERIMENT_C 5)."""
    match = CG.C_STREAM.family_re.fullmatch(family_line)
    if match is None:
        _refuse("pinned family line is malformed")

    def items(text: str) -> list:
        return [x for x in text.split(",") if x]

    return {"family": items(match.group(1)), "descriptive": items(match.group(2)),
            "dropped": items(match.group(3)), "references": items(match.group(4))}


def contrast_roles(lists: dict) -> dict:
    """Role of every contrast of FAMILY: 'family' (Holm), 'descriptive' (no Holm, no
    WIN/TIE) or 'dropped' (omitted). Dropped entries that name no contrast are gate items
    (arm:gate) and only appear in the notes."""
    known = [c[0] for c in FAMILY]
    fam, desc = list(lists["family"]), list(lists["descriptive"])
    dropped = {x for x in lists["dropped"] if x in known}
    if len(set(fam)) != len(fam) or len(set(desc)) != len(desc):
        _refuse("pinned family or descriptive list repeats a contrast")
    if (set(fam) & set(desc)) or (dropped & (set(fam) | set(desc))):
        _refuse("pinned lists overlap: a contrast has two roles")
    unknown = sorted((set(fam) | set(desc)) - set(known))
    if unknown:
        _refuse(f"pinned lists name unknown contrasts: {unknown}")
    if not fam:
        _refuse("the pinned family is empty")
    roles = {}
    for cid in known:
        if cid in fam:
            roles[cid] = "family"
        elif cid in desc:
            roles[cid] = "descriptive"
        elif cid in dropped:
            roles[cid] = "dropped"
        else:
            _refuse(f"contrast {cid} is not named by the pinned family line")
    return roles


def omission_notes(lists: dict) -> list:
    """Notes for the summary: contrasts omitted as dropped, and the gate items the pinned
    line dropped (arm:gate), which no contrast row carries."""
    known = {c[0] for c in FAMILY}
    notes = []
    omitted = [x for x in lists["dropped"] if x in known]
    items = [x for x in lists["dropped"] if x not in known]
    if omitted:
        notes.append(f"contrasts omitted as dropped by the pinned line: {omitted}")
    if items:
        notes.append(f"gate items dropped by the pinned line (arm:gate): {items}")
    return notes


def family_table(df: pd.DataFrame, *, lists: dict | None = None,
                 n_instances: int = N_INSTANCES, n_routes: int = N_ROUTES,
                 n_reps: int = N_REPS) -> pd.DataFrame:
    """Family C (1.9) from the pinned lists (default: all eleven). Family contrasts: two-
    sided exact Wilcoxon, Holm over the family members only (holm_k = their number), and
    the classes of A 1.5 with each margin (delta_Q1 or delta). Descriptive contrasts: raw
    statistics, no Holm p, classification 'descriptive'. Dropped contrasts are omitted."""
    roles = contrast_roles(default_lists() if lists is None else lists)
    found = {}
    for cid, a_stage, a_pol, b_stage, b_pol, margin in FAMILY:
        if roles[cid] == "dropped":
            continue
        found[cid] = _checked(df, cid, (a_stage, a_pol), (b_stage, b_pol), delta=margin,
                              n_instances=n_instances, n_routes=n_routes, n_reps=n_reps)
    members = [c[0] for c in FAMILY if roles[c[0]] == "family"]
    k = len(members)
    adjusted = dict(zip(members, A.holm([found[cid]["p_two_sided"] for cid in members])))
    out = []
    for cid, a_stage, a_pol, b_stage, b_pol, margin in FAMILY:
        if roles[cid] == "dropped":
            continue
        res = found[cid]
        d = res["_d"]
        base = dict(id=cid, role=roles[cid], a=a_stage, b=b_stage, world="W6",
                    margin=margin, n=res["n"], mean=res["mean"], median=res["median"],
                    ci95_lo=res["ci95"][0], ci95_hi=res["ci95"][1],
                    ci90_lo=res["ci90"][0], ci90_hi=res["ci90"][1],
                    n_better=res["n_better"], p_raw=res["p_two_sided"])
        if roles[cid] == "family":
            lo_h, hi_h = B.holm_ci(d, k)
            p_adj = adjusted[cid]
            base.update(holm_lo=lo_h, holm_hi=hi_h, p_holm=p_adj, holm_k=k,
                        classification=A.classify(d, margin, p_adj=p_adj))
        else:
            base.update(holm_lo=np.nan, holm_hi=np.nan, p_holm=np.nan, holm_k=np.nan,
                        classification="descriptive")
        out.append(base)
    return pd.DataFrame(out, columns=FAMILY_COLUMNS)


# ---------------------------------------------------------------- Q1: share of gain

def share_bootstrap(num: np.ndarray, den: np.ndarray, n_draws: int = SHARE_DRAWS,
                    seed: int = SHARE_SEED) -> dict:
    """Instance bootstrap of mean(num) / mean(den) over paired instances (1.5). The draws
    resample instances with replacement, one ``integers`` call of shape (draws, n)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    if num.size != den.size or num.size == 0:
        raise ValueError("share bootstrap needs paired, non-empty instance vectors")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n_draws, num.size))
    draws = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(share=float(num.mean() / den.mean()), ci95=(float(lo), float(hi)),
                n_draws=int(n_draws), seed=int(seed))


def _instance_deltas(df: pd.DataFrame, left: tuple, right: tuple) -> pd.Series:
    pair = _pair(df, left, right)
    return A.instance_deltas(pair, "A", "B", N_PATHS)


def share_table(df: pd.DataFrame) -> pd.DataFrame:
    """Q1 (descriptive): per learner, F0n vs F0 and share(F) = Delta(F vs F0) /
    Delta(F3 vs F0) with the instance bootstrap CI (1.5)."""
    out = []
    for learner, pol in LEARNERS:
        base = f"c_{learner}_F0_w6"
        den = _instance_deltas(df, (f"c_{learner}_F3_w6", pol), (base, pol))
        for feat in SHARE_FEATURES:
            stage = f"c_{learner}_{feat}_w6"
            num = _instance_deltas(df, (stage, pol), (base, pol))
            common = sorted(set(num.index) & set(den.index))
            x, y = num.loc[common].to_numpy(), den.loc[common].to_numpy()
            boot = share_bootstrap(x, y)
            contrast = dict(delta_vs_f0=float(x.mean()), ci95_lo=None, ci95_hi=None,
                            p_raw=None)
            if feat == "F0n":
                res = _checked(df, f"Q1_{learner}_F0n", (stage, pol), (base, pol),
                               delta=A.DELTA, n_instances=N_INSTANCES, n_routes=N_ROUTES,
                               n_reps=N_REPS)
                contrast.update(ci95_lo=res["ci95"][0], ci95_hi=res["ci95"][1],
                                p_raw=res["p_two_sided"])
            out.append(dict(learner=learner, feature_set=feat, n=len(common),
                            share=boot["share"], share_ci95_lo=boot["ci95"][0],
                            share_ci95_hi=boot["ci95"][1], **contrast))
    return pd.DataFrame(out, columns=SHARE_COLUMNS)


# ---------------------------------------------------------------- Q2: summed costs

def route_means(df: pd.DataFrame, stage: str, policy: str) -> pd.Series:
    """Route means over replicates, indexed by (inst, route_id)."""
    sub = df[(df["stage"] == stage) & (df["policy"] == policy)]
    if sub.empty:
        _refuse(f"no rows for {stage}/{policy}")
    return sub.groupby(["inst", "route_id"])["mean_cost"].mean()


def sigma_instances(df: pd.DataFrame, a: tuple, b: tuple, e0: dict) -> dict:
    """Summed-cost estimand (1.5): per instance, Delta_i = 100 (sum_r C_b - sum_r C_a) /
    sum_r C_b over the eligible routes. An instance with sum_r C_b < 1e-3 sum_r E0 is
    excluded and counted."""
    ca, cb = route_means(df, *a), route_means(df, *b)
    if not ca.index.equals(cb.index):
        _refuse(f"{a} and {b} do not cover the same routes")
    values, excluded, names = [], 0, []
    for inst in sorted(set(ca.index.get_level_values(0))):
        keys = [k for k in ca.index if k[0] == inst]
        s_a = float(ca.loc[keys].sum())
        s_b = float(cb.loc[keys].sum())
        floor = SIGMA_FLOOR * sum(float(e0[k]) for k in keys)
        if s_b < floor:
            excluded += 1
            continue
        values.append(100.0 * (s_b - s_a) / s_b)
        names.append(inst)
    return {"values": np.asarray(values, dtype=float), "instances": names,
            "excluded": excluded}


def sigma_table(df: pd.DataFrame, e0: dict) -> pd.DataFrame:
    """Q2 per world (descriptive, no trend test): summed-cost Delta-bar of BOOST-F3, RLSM-F3,
    BOOST-F0 against BATON, BATON against THR, and CDP against BATON (positive: the first
    named is cheaper)."""
    out = []
    for tag in WORLD_TAGS:
        ladder = f"c_ladder_{tag}"
        specs = [
            (f"boost_F3 vs baton", (f"c_boost_F3_{tag}", "boost"), (ladder, "baton_eq")),
            (f"rlsm_F3 vs baton", (f"c_rlsm_F3_{tag}", "rlsm"), (ladder, "baton_eq")),
            (f"boost_F0 vs baton", (f"c_boost_F0_{tag}", "boost"), (ladder, "baton_eq")),
            ("baton vs thr", (ladder, "baton_eq"), (f"c_thr_{tag}", "thr")),
            ("cdp vs baton", (f"c_cdp_{tag}", "cdp"), (ladder, "baton_eq")),
        ]
        for label, a, b in specs:
            res = sigma_instances(df, a, b, e0)
            d = res["values"]
            if d.size == 0:
                _refuse(f"{tag} {label}: every instance is below the floor")
            mean_lo, mean_hi = _ci_of_instances(d)
            out.append(dict(world=tag.upper(), contrast=label, n=d.size + res["excluded"],
                            included=d.size, excluded=res["excluded"], mean=float(d.mean()),
                            ci95_lo=mean_lo, ci95_hi=mean_hi))
    return pd.DataFrame(out, columns=SIGMA_COLUMNS)


def _ci_of_instances(values: np.ndarray) -> tuple[float, float]:
    """CI95 of the mean over instances (A's cluster bootstrap on singleton instances)."""
    if values.size < 2:
        return float("nan"), float("nan")
    boot = A.cluster_bootstrap([[v] for v in values])
    return boot["ci95"]


def day_breaches(z) -> np.ndarray:
    """Per-day breach indicators of one day-cost file. An explicit ``breach`` array is used
    when the file has one; otherwise a day breaches iff its cost under the no-recourse
    policy ``none`` is > 0 (the ladder rows carry ``none`` for every day)."""
    if "breach" in z.files:
        return np.asarray(z["breach"], dtype=float)
    if "none" in z.files:
        return (np.asarray(z["none"], dtype=float) > 0.0).astype(float)
    _refuse("day file holds neither a 'breach' array nor the 'none' policy's costs")


def breach_table(rows_dirs: dict, routes: list, N: int) -> pd.DataFrame:
    """Per world: breach rate without recourse and the number of routes with no breach.
    Per-day breaches come from ``day_breaches`` on each world's ladder rows directory
    (one file per route-rep, ``*.test_c.npz``)."""
    out = []
    for tag, rows_dir in rows_dirs.items():
        rates = []
        for inst, rid in routes:
            reps = []
            pattern = sweep.day_file_glob(inst, rid, N, None, "test_c")
            for path in sorted(Path(rows_dir).glob(pattern)):
                with np.load(path) as z:
                    reps.append(day_breaches(z).mean())
            if not reps:
                _refuse(f"no test_c day files for {inst} r{rid} in {rows_dir}")
            rates.append(float(np.mean(reps)))
        rates = np.asarray(rates)
        out.append(dict(world=tag.upper(), breach_rate=float(rates.mean()),
                        routes_no_breach=int((rates == 0).sum()), routes=rates.size))
    return pd.DataFrame(out, columns=("world", "breach_rate", "routes_no_breach", "routes"))


# ---------------------------------------------------------------- Q3 and Q4

def cdp_table(df: pd.DataFrame) -> pd.DataFrame:
    """Q3 (descriptive): CDP against THR per world (same DP engine), and CDP-true against
    CDP in W6 (sensitivity)."""
    out = []
    for tag in WORLD_TAGS:
        res = _checked(df, f"Q3_{tag}", (f"c_cdp_{tag}", "cdp"), (f"c_thr_{tag}", "thr"),
                       delta=A.DELTA, n_instances=N_INSTANCES, n_routes=N_ROUTES,
                       n_reps=N_REPS)
        out.append(_cdp_row(tag.upper(), "CDP vs THR", res))
    res = _checked(df, "Q3_true", ("c_cdptrue_w6", "cdp"), ("c_cdp_w6", "cdp"),
                   delta=A.DELTA, n_instances=N_INSTANCES, n_routes=N_ROUTES, n_reps=N_REPS)
    out.append(_cdp_row("W6", "CDP-true vs CDP", res))
    return pd.DataFrame(out, columns=CDP_COLUMNS)


def _cdp_row(world: str, label: str, res: dict) -> dict:
    return dict(world=world, comparison=label, n=res["n"], mean=res["mean"],
                median=res["median"], ci95_lo=res["ci95"][0], ci95_hi=res["ci95"][1],
                n_better=res["n_better"], p_raw=res["p_two_sided"])


def ppo_subset_routes(df: pd.DataFrame) -> set:
    sub = df[(df["stage"] == "c_ppolong_w6") & (df["policy"] == "ppo")]
    return {(str(i), int(r)) for i, r in zip(sub["inst"], sub["route_id"])}


def ppo_table(df: pd.DataFrame) -> pd.DataFrame:
    """Q4 (descriptive, n = 10 instances, 12 routes, rep 0): PPO-long against FULL and against
    B's PPO on the same routes."""
    routes = ppo_subset_routes(df)
    if len(routes) != SUB_ROUTES:
        _refuse(f"PPO-long subset holds {len(routes)} routes; {SUB_ROUTES} expected")
    cover = dict(delta=A.DELTA, n_instances=SUB_INSTANCES, n_routes=SUB_ROUTES,
                 n_reps=SUB_REPS, rep=0, routes=routes)
    out = []
    for label, left, right in (
            ("PPO-long vs FULL", ("c_ppolong_w6", "ppo"), ("c_full_w6", "nn_full")),
            ("PPO-long vs B's PPO", ("c_ppolong_w6", "ppo"), ("c_ppob_w6", "ppo"))):
        res = _checked(df, label, left, right, **cover)
        out.append(dict(comparison=label, n=res["n"], mean=res["mean"], median=res["median"],
                        ci95_lo=res["ci95"][0], ci95_hi=res["ci95"][1],
                        n_better=res["n_better"], p_raw=res["p_two_sided"]))
    return pd.DataFrame(out, columns=PPO_COLUMNS)


def gap_table(df: pd.DataFrame) -> pd.DataFrame:
    """FULL's share of the BATON-to-oracle gap in W6: per instance,
    sum_r (C_BATON - C_FULL) / sum_r (C_BATON - C_oracle) over route means; instances with a
    non-positive denominator are excluded and counted; CI95 by the instance bootstrap."""
    cb = route_means(df, "c_ladder_w6", "baton_eq")
    cf = route_means(df, "c_full_w6", "nn_full")
    co = route_means(df, "c_ladder_w6", "oracle")
    if not (cb.index.equals(cf.index) and cb.index.equals(co.index)):
        _refuse("BATON, FULL and oracle do not cover the same route-reps")
    shares, excluded = [], 0
    for inst in sorted(set(cb.index.get_level_values(0))):
        keys = [k for k in cb.index if k[0] == inst]
        den = float((cb.loc[keys] - co.loc[keys]).sum())
        if den <= 0:
            excluded += 1
            continue
        shares.append(float((cb.loc[keys] - cf.loc[keys]).sum()) / den)
    values = np.asarray(shares, dtype=float)
    lo, hi = _ci_of_instances(values)
    return pd.DataFrame([dict(n=values.size, excluded=excluded, mean_share=float(values.mean()),
                              ci95_lo=lo, ci95_hi=hi)], columns=GAP_COLUMNS)


# ---------------------------------------------------------------- output

def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_summary(tables: dict, *, eval_set: str, label: str, eval_commit: str = "n/a",
                   notes: tuple = ()) -> str:
    head = ["# EXPERIMENT C analysis" + (" (DEV, pilot: no family check)"
                                         if eval_set == "dev" else " (test_c)"), "",
            f"eval set: {eval_set}; eval commit: {eval_commit}; {label}", ""]
    head += [f"- note: {note}" for note in notes] + ([""] if notes else [])
    titles = {"family": "Family C (11 contrasts, Holm)",
              "share": "Q1 share of gain (descriptive)",
              "sigma": "Q2 summed-cost estimand per world (descriptive)",
              "breach": "Q2 breach table (no recourse)",
              "cdp": "Q3 CDP against THR and CDP-true (descriptive)",
              "ppo": "Q4 PPO-long on the 12-route subset (descriptive)",
              "gap": "FULL's share of the BATON-to-oracle gap in W6"}
    body = []
    for key, title in titles.items():
        if key in tables:
            body += [f"## {title}", "", _md(tables[key]), ""]
    return "\n".join(head + body)


def write_outputs(tables: dict, out_dir: Path, **meta) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in tables.items():
        df.to_csv(out_dir / f"{key}.csv", index=False, lineterminator="\n")
    (out_dir / "summary.md").write_text(render_summary(tables, **meta), encoding="utf-8",
                                        newline="\n")
    return out_dir


# ---------------------------------------------------------------- pin and CLI

def pinned_family(root: Path, head: str) -> str:
    """HEAD's TEST_C FAMILY line after the stream's own checks (gatecheck.pinned_log)."""
    return G.pinned_log(root, head, CG.C_STREAM).family_line


def _head(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def load_routes_and_e0(df: pd.DataFrame) -> dict:
    """E0 per (inst, route_id) from the route specifications (sweep)."""
    from ..expa import sweep     # noqa: PLC0415
    e0 = {}
    for inst, rid in {(str(i), int(r)) for i, r in zip(df["inst"], df["route_id"])}:
        e0[(inst, rid)] = float(sweep._spec_of(inst, rid).E0)
    return e0


def run(eval_set: str, tables_dir: Path, out_dir: Path | None = None, root: Path | None = None):
    """Full analysis; returns the output directory. Refuses as the module docstring says."""
    tables_dir = Path(tables_dir)
    root = G._root(root)
    if eval_set == "test_c":
        G.check_analysis_clean(stream=CG.C_STREAM, repo=root)
        head = _head(root)
        family = pinned_family(root, head)
        sha = family_hash(family)
        lists = family_lists(family)
        suffix = TEST_C_SUFFIX
    else:
        family, sha, head, suffix, lists = None, None, "n/a", ".csv", None
    wanted = set(required_stages())
    files = sorted(p for p in tables_dir.glob(f"*{suffix}")
                   if stage_of_file(p, eval_set) in wanted)
    df = load_tables(files, eval_set=eval_set, family_sha256=sha)
    if eval_set == "test_c":
        check_complete(df)
    e0 = load_routes_and_e0(df)
    tables = {"family": family_table(df, lists=lists), "share": share_table(df),
              "sigma": sigma_table(df, e0), "cdp": cdp_table(df), "ppo": ppo_table(df),
              "gap": gap_table(df)}
    notes = omission_notes(lists) if lists is not None else [
        "DEV: no pinned family line; all eleven contrasts are treated as family"]
    try:
        routes = sorted({(str(i), int(r)) for i, r in zip(df["inst"], df["route_id"])})
        rows_dirs = {tag: tables_dir / f"c_ladder_{tag}" / "rows" for tag in WORLD_TAGS}
        tables["breach"] = breach_table(rows_dirs, routes, N_PATHS)
    except AnalysisRefused as exc:
        notes.append(f"breach table not computed: {exc}")
    label = (f"family line {family}" if family else "DEV: no family line")
    default = "analysis_test_c" if eval_set == "test_c" else "analysis_dev"
    out = Path(out_dir) if out_dir else root / CG.C_RESULTS_REL / default
    write_outputs(tables, out, eval_set=eval_set, label=label, eval_commit=head,
                  notes=tuple(notes))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m transvrp.expc.analysis_c")
    ap.add_argument("--eval", choices=EVAL_SETS, default="test_c")
    ap.add_argument("--tables", default=None, help="directory of the stage tables")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = G._root(None)
    tables = Path(args.tables) if args.tables else root / CG.C_RESULTS_REL
    try:
        out = run(args.eval, tables, Path(args.out) if args.out else None)
    except (AnalysisRefused, G.GateRefused) as exc:
        print(f"refused: {exc}")
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

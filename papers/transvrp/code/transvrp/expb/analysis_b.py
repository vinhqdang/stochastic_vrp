"""EXPERIMENT B analysis of the test_b stream (EXPERIMENT_B.md sections 1, 5.3, 8; M23).

Reads one merged ``<stage>.test_b.csv`` per B stage and refuses unless the tables are
exactly what the FAMILY line pinned at HEAD allows. The statistics are A's, imported
read-only from ``expa.analysis`` (contrast, instance deltas, coverage, Holm, the CIs and
the exact Wilcoxon p). This module never calls ``expa.analysis.write_summary``: that
function serves A's streams and carries A's eval-set guard and family size.

Branches (section 1.3): WIN and LOSE use the Holm-adjusted p and the 13/20 count;
TIE_STRONG uses the unadjusted CI90 inside (-delta, delta); everything else is TIE_WEAK.
Every family arm is reported whatever its branch. Descriptive and dropped arms, the
independence world and the A-baseline re-evaluation are reported without Holm.

Run ``python -m transvrp.expb.analysis_b --merged <stage>.test_b.csv ... --eval test_b
--out DIR [--models DIR ...]`` from inside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..expa import analysis as A
from ..expa.paths import TEST_B_N
from . import gatecheck as G
from . import stages

DELTA = A.DELTA
TOL = A.TOL
WIN_MIN_COUNT = A.WIN_MIN_COUNT          # section 1.3: 13 of 20 instances
HOLM_FWER = 0.05                         # section 1.3: FWER over the family
STREAM = "test_b"
TEST_B_SUFFIX = f".{STREAM}.csv"
N_PATHS = 100_000                        # section 4.1: every arm at 10^5 paths
N_INSTANCES = 20                         # section 1.2: unit = instance
N_ROUTES = 62                            # section 1.1: eligible routes
N_REPS = 3                               # section 4.1: replicates 0-2 (no-cut default; main() follows the cut)
IND_INSTANCES = 10                       # section 3.8: 12 subset routes from 10 instances
IND_ROUTES = 12
IND_REPS = 1                             # section 3.8: rep 0 only
B_ARMS = ("thr", "alpz", "rlsm", "dos", "dqn", "ppo")   # = registry.B_ARMS (tested)
IND_CORES = ("full", "ladder", "thr", "alpz")
FULL = "nn_full"
BASELINE = "baton_eq"
LADDER_POLICIES = ("none", BASELINE, "oracle")
REQUIRED_COLUMNS = ("inst", "route_id", "N", "rep", "policy", "mean_cost", "gap_closed",
                    "eval_set", "n_eval", "fit_commit", "eval_commit", "family_sha256",
                    "stratum")
CUT_SUFFIX = ":cut"                      # dropped entry of a compute-cut arm (9.2 (ii))
SAME_COST_TOL = 1e-9                     # relative: two route-rep costs count as identical
DA_COLUMNS = ("arm", "stage", "n", "mean", "median", "ci95_lo", "ci95_hi", "ci90_lo",
              "ci90_hi", "n_better", "p_greater", "p_less", "branch")
STRATA_COLUMNS = ("arm", "stratum", "n", "mean", "ci95_lo", "ci95_hi", "n_better")
REPLICATE_COLUMNS = ("arm", "routes", "mean_sd_pct", "max_sd_pct")
BUDGET_LABEL = "budget-limited"          # section 1.3 arm-level qualifier
# independence world (section 3.8): (label, arm core, policy), each compared with FULL
IND_COMPARISONS = (("none", "ladder", "none"), (BASELINE, "ladder", BASELINE),
                   ("oracle", "ladder", "oracle"), ("THR", "thr", "thr"),
                   ("ALP-Z", "alpz", "alpz"))
FAMILY_COLUMNS = ("arm", "stage", "n", "mean", "median", "ci95_lo", "ci95_hi", "holm_lo",
                  "holm_hi", "ci90_lo", "ci90_hi", "n_better", "rank_biserial",
                  "pooled_ratio", "gc_mean_full", "p_raw", "p_holm", "branch")
OUTSIDE_COLUMNS = ("arm", "reason", "stage", "n", "mean", "median", "ci95_lo", "ci95_hi",
                   "ci90_lo", "ci90_hi", "n_better", "p_raw", "ci90_inside_delta")
COMPUTE_FIELDS = ("feature_seconds", "train_seconds", "val_seconds", "unique_paths",
                  "path_stop_reads", "val_reads", "updates", "lp_rows", "lp_cols",
                  "lp_nnz", "n_params", "peak_rss_mb", "device")
COMPUTE_COLUMNS = ("stage", "arm", "rep", "fit_seconds") + COMPUTE_FIELDS


class AnalysisRefused(ValueError):
    """An input breaks the pinned protocol (section 5.3); nothing is written."""


def _refuse(message: str):
    raise AnalysisRefused(message)


def family_hash(family_line: str) -> str:
    """sha256 of the FAMILY line, as written into every test_b row (section 5.3)."""
    return hashlib.sha256(family_line.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- input and membership

def stage_of_file(path) -> str:
    name = Path(path).name
    if not name.endswith(TEST_B_SUFFIX):
        _refuse(f"{name}: not a test_b stage table (expected <stage>{TEST_B_SUFFIX})")
    return name[: -len(TEST_B_SUFFIX)]


def expected_policies(core: str) -> frozenset:
    """The policy names a stage of arm ``core`` may write (sections 3 and 3.7)."""
    if core == "full":
        return frozenset({FULL})
    if core == "ladder":
        return frozenset(LADDER_POLICIES)
    if core in B_ARMS:
        return frozenset({core})
    _refuse(f"arm {core!r} is not a known B arm or reference")


def _read_stage(path, stage: str, sha: str, frozen: dict | None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"family_sha256": str, "fit_commit": str,
                                  "eval_commit": str, "inst": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        _refuse(f"{stage}: columns missing: {missing}")
    if df.empty:
        _refuse(f"{stage}: no rows")
    if not (df["eval_set"].astype(str) == STREAM).all():
        _refuse(f"{stage}: rows not from stream {STREAM!r}")
    if not (df["family_sha256"].astype(str) == sha).all():
        _refuse(f"{stage}: family_sha256 differs from the FAMILY line pinned at HEAD")
    if not (df["n_eval"] == TEST_B_N).all():
        _refuse(f"{stage}: day count differs from {TEST_B_N}")
    if not (df["N"] == N_PATHS).all():
        _refuse(f"{stage}: rows not at N = {N_PATHS}")
    fits = set(df["fit_commit"].astype(str))
    if len(fits) != 1:
        _refuse(f"{stage}: rows from {len(fits)} fit commits; one is required")
    fit = fits.pop()
    if frozen is not None and frozen.get(stage) != fit:
        _refuse(f"{stage}: fit commit {fit} is not the FROZEN commit of the stage")
    return df.assign(stage=stage)


def load_stage_tables(paths, *, family_sha256: str, frozen: dict | None = None) -> pd.DataFrame:
    """One frame of all stage tables, with a ``stage`` column taken from the file name.
    Refuses rows not from test_b, rows whose family hash differs from the pinned FAMILY
    line, day counts other than 20,000, several fit commits in a stage, a fit commit other
    than the stage's FROZEN commit (when ``frozen`` maps stage -> commit), and several eval
    commits across the tables (section 5.3: one evaluation commit)."""
    stages = [stage_of_file(p) for p in paths]
    if not stages:
        _refuse("no stage tables given")
    if len(set(stages)) != len(stages):
        _refuse("a stage is given twice")
    frames = [_read_stage(Path(p), s, family_sha256, frozen) for p, s in zip(paths, stages)]
    df = pd.concat(frames, ignore_index=True)
    commits = set(df["eval_commit"].astype(str))
    if len(commits) != 1:
        _refuse(f"rows from {len(commits)} eval commits; exactly one is allowed (5.3)")
    return df


def cut_arms(lists: dict) -> list:
    """Arms removed by a compute cut: ``arm:cut`` entries of the FAMILY line's ``dropped``.
    They have no fit and no rows; they are reported as cut and nothing else."""
    return [item.split(":")[0] for item in lists["dropped"] if item.endswith(CUT_SUFFIX)]


def check_membership(df: pd.DataFrame, lists: dict) -> None:
    """Every stage is an arm of the FAMILY line; every listed arm and independence stage
    has rows; each stage writes only its own policies. ``lists`` is the FAMILY line's
    lists (gatecheck.PinnedLog.lists)."""
    family, descriptive = set(lists["family"]), set(lists["descriptive"])
    cut = cut_arms(lists)
    dropped = {item.split(":")[0] for item in lists["dropped"]} - set(cut)
    references, independence = set(lists["references"]), set(lists["independence"])
    if family & dropped:
        _refuse(f"arms both in the family and dropped: {sorted(family & dropped)}")
    allowed = family | descriptive | dropped | references
    stages = set(df["stage"])
    cut_with_rows = sorted(set(cut) & {G.core_of_stage(s) for s in stages})
    if cut_with_rows:
        _refuse(f"arm(s) {cut_with_rows} are listed as cut (dropped=[arm:cut]) but have rows")
    for stage in sorted(stages):
        core = G.core_of_stage(stage)
        if stage in independence:
            if core not in IND_CORES:
                _refuse(f"independence stage {stage!r} has an unexpected arm {core!r}")
        elif core not in allowed:
            _refuse(f"arm {core!r} of stage {stage!r} is not in the family, descriptive, "
                    "dropped or reference list")
        extra = set(df.loc[df["stage"] == stage, "policy"]) - expected_policies(core)
        if extra:
            _refuse(f"stage {stage!r} writes policies {sorted(extra)} outside its arm")
    absent_ind = sorted(independence - stages)
    if absent_ind:
        _refuse(f"independence stages listed in the FAMILY line but absent: {absent_ind}")
    main_cores = [G.core_of_stage(s) for s in stages if s not in independence]
    dup = sorted({c for c in main_cores if main_cores.count(c) > 1})
    if dup:
        _refuse(f"more than one stage for arm(s) {dup}")
    absent = sorted(allowed - set(main_cores))
    if absent:
        _refuse(f"arm(s) listed in the FAMILY line with no test_b rows: {absent}")


def main_stage(df: pd.DataFrame, core: str) -> str:
    """The one stage of arm ``core`` in ``df`` (a reuse stage counts as its arm)."""
    stages = sorted({s for s in df["stage"] if G.core_of_stage(s) == core})
    if len(stages) != 1:
        _refuse(f"arm {core!r}: {len(stages)} stages in the table, exactly one is required")
    return stages[0]


# ---------------------------------------------------------------- statistics

def holm_ci(d, size: int) -> tuple[float, float]:
    """Holm-level interval (section 1.3): two-sided level 1 - 0.05/size, i.e. the
    one-sided t quantile 1 - 0.025/size. size = 1 gives CI95."""
    if size < 1:
        raise ValueError("the family size must be at least 1")
    return A._ci(np.asarray(d, dtype=float), 1.0 - HOLM_FWER / (2.0 * size))


def go_no_go_holm(d, p_holm: float, delta: float = DELTA) -> str:
    """Section 1.3 branch of one family contrast. ``d`` holds the instance deltas and
    ``p_holm`` the Holm-adjusted two-sided exact Wilcoxon p over the family."""
    d = np.asarray(d, dtype=float)
    mean = float(d.mean())
    lo95, hi95 = A._ci(d, 0.975)
    lo90, hi90 = A._ci(d, 0.95)
    significant = p_holm < HOLM_FWER
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    if (mean >= delta - TOL and mean > 0 and significant and lo95 > 0
            and n_pos >= WIN_MIN_COUNT):
        return "WIN"
    if (mean <= -(delta - TOL) and mean < 0 and significant and hi95 < 0
            and n_neg >= WIN_MIN_COUNT):
        return "LOSE"
    if lo90 > -delta and hi90 < delta:
        return "TIE_STRONG"
    return "TIE_WEAK"


def _pair(df: pd.DataFrame, left: tuple, right: tuple) -> pd.DataFrame:
    """Rows of two (stage, policy) pairs under the policy names A and B, so that the
    statistics of ``expa.analysis`` (keyed on policy) compare exactly the two."""
    a = df[(df["stage"] == left[0]) & (df["policy"] == left[1])].assign(policy="A")
    b = df[(df["stage"] == right[0]) & (df["policy"] == right[1])].assign(policy="B")
    if a.empty or b.empty:
        _refuse(f"no rows for {left} or for {right}")
    return pd.concat([a, b], ignore_index=True)


def _checked_contrast(df, cid, left, right, *, delta, n_instances, n_routes, n_reps) -> dict:
    """A's contrast of ``left`` (A) against ``right`` (B) after A's coverage check; a
    coverage failure is a refusal, never a dropped arm. Positive: A is cheaper."""
    pair = _pair(df, left, right)
    try:
        A.check_coverage(pair, (cid, "A", N_PATHS, "B", N_PATHS), n_instances, n_routes,
                         n_reps)
    except ValueError as exc:
        _refuse(str(exc))
    return A.contrast(pair, cid, "A", N_PATHS, "B", N_PATHS, delta)


def family_table(df: pd.DataFrame, family, delta: float = DELTA, *,
                 n_instances: int = N_INSTANCES, n_routes: int = N_ROUTES,
                 n_reps: int = N_REPS) -> pd.DataFrame:
    """Section 1.3: FULL against each family arm at 10^5 paths, Holm over the family."""
    full = main_stage(df, "full")
    cover = dict(delta=delta, n_instances=n_instances, n_routes=n_routes, n_reps=n_reps)
    found, raw, deltas = [], [], []
    for arm in family:
        stage = main_stage(df, arm)
        res = _checked_contrast(df, f"HB_{arm}", (full, FULL), (stage, arm), **cover)
        found.append((arm, stage, res))
        raw.append(res["p_two_sided"])
        deltas.append(res["_d"])
    if not found:
        return pd.DataFrame(columns=FAMILY_COLUMNS)
    adjusted = A.holm(raw)
    out = []
    for (arm, stage, res), p_adj, d in zip(found, adjusted, deltas):
        lo_h, hi_h = holm_ci(d, len(found))
        out.append(dict(
            arm=arm, stage=stage, n=res["n"], mean=res["mean"], median=res["median"],
            ci95_lo=res["ci95"][0], ci95_hi=res["ci95"][1], holm_lo=lo_h, holm_hi=hi_h,
            ci90_lo=res["ci90"][0], ci90_hi=res["ci90"][1], n_better=res["n_better"],
            rank_biserial=res["rank_biserial"], pooled_ratio=res["pooled_ratio"],
            gc_mean_full=res["gc_mean_a"], p_raw=res["p_two_sided"], p_holm=p_adj,
            branch=go_no_go_holm(d, p_adj, delta)))
    return pd.DataFrame(out, columns=FAMILY_COLUMNS)


def outside_table(df: pd.DataFrame, arms: dict, delta: float = DELTA, *,
                  n_instances: int = N_INSTANCES, n_routes: int = N_ROUTES,
                  n_reps: int = N_REPS) -> pd.DataFrame:
    """Descriptive and dropped arms outside F (section 1.4): FULL against each, unadjusted,
    no branch. ``arms`` maps arm -> the reason copied from the FAMILY line."""
    full = main_stage(df, "full")
    cover = dict(delta=delta, n_instances=n_instances, n_routes=n_routes, n_reps=n_reps)
    out = []
    for arm, reason in arms.items():
        stage = main_stage(df, arm)
        res = _checked_contrast(df, f"OUT_{arm}", (full, FULL), (stage, arm), **cover)
        lo90, hi90 = res["ci90"]
        out.append(dict(
            arm=arm, reason=reason, stage=stage, n=res["n"], mean=res["mean"],
            median=res["median"], ci95_lo=res["ci95"][0], ci95_hi=res["ci95"][1],
            ci90_lo=lo90, ci90_hi=hi90, n_better=res["n_better"], p_raw=res["p_two_sided"],
            ci90_inside_delta=bool(lo90 > -delta and hi90 < delta)))
    return pd.DataFrame(out, columns=OUTSIDE_COLUMNS)


def independence_table(df: pd.DataFrame, delta: float = DELTA, *,
                       n_instances: int = IND_INSTANCES, n_routes: int = IND_ROUTES,
                       n_reps: int = IND_REPS) -> pd.DataFrame:
    """Section 3.8, descriptive, no Holm: FULL in the independence world against the
    independence ladder, THR and ALP-Z (12 subset routes, rep 0)."""
    full = main_stage(df, "full")
    cover = dict(delta=delta, n_instances=n_instances, n_routes=n_routes, n_reps=n_reps)
    out = []
    for label, core, policy in IND_COMPARISONS:
        stage = main_stage(df, core)
        res = _checked_contrast(df, f"IND_{label}", (full, FULL), (stage, policy), **cover)
        out.append(dict(comparison=f"FULL vs {label}", stage=stage, n=res["n"],
                        mean=res["mean"], ci95_lo=res["ci95"][0], ci95_hi=res["ci95"][1],
                        ci90_lo=res["ci90"][0], ci90_hi=res["ci90"][1],
                        n_better=res["n_better"], p_raw=res["p_two_sided"]))
    return pd.DataFrame(out)


def _arm_stages(df: pd.DataFrame, arms) -> list:
    return [(arm, main_stage(df, arm)) for arm in arms]


def da_table(df: pd.DataFrame, arms, delta: float = DELTA, *, n_instances: int = N_INSTANCES,
             n_routes: int = N_ROUTES, n_reps: int = N_REPS) -> pd.DataFrame:
    """D_a (section 1.4): each evaluated arm against baton_eq at 10^5 on test_b, classified
    by A's unadjusted go/no-go. Outside F; no Holm; ``arms`` lists every arm with rows."""
    ladder = main_stage(df, "ladder")
    out = []
    for arm, stage in _arm_stages(df, arms):
        pair = _pair(df, (stage, arm), (ladder, BASELINE))
        try:
            A.check_coverage(pair, (f"D_{arm}", "A", N_PATHS, "B", N_PATHS), n_instances,
                             n_routes, n_reps)
        except ValueError as exc:
            _refuse(str(exc))
        d = A.instance_deltas(pair, "A", "B", N_PATHS).to_numpy()
        res = A.go_no_go(d, delta)
        out.append(dict(arm=arm, stage=stage, n=res["n"], mean=res["mean"],
                        median=res["median"], ci95_lo=res["ci95"][0], ci95_hi=res["ci95"][1],
                        ci90_lo=res["ci90"][0], ci90_hi=res["ci90"][1], n_better=res["n_pos"],
                        p_greater=res["p_greater"], p_less=res["p_less"],
                        branch=res["branch"]))
    return pd.DataFrame(out, columns=DA_COLUMNS)


def alpz_vs_thr(df: pd.DataFrame, delta: float = DELTA, *, n_instances: int = N_INSTANCES,
                n_routes: int = N_ROUTES, n_reps: int = N_REPS) -> dict | None:
    """ALP-Z against THR on test_b (section 1.4, descriptive): the paired cost difference
    (positive: ALP-Z cheaper) and the share of route-reps with identical mean cost (the rows
    hold costs, not decisions: identical per-route costs are the test_b counterpart of
    identical decisions). None if either arm has no rows (cut or dropped away)."""
    stages = {c: [s for s in set(df["stage"]) if G.core_of_stage(s) == c]
              for c in ("alpz", "thr")}
    if not stages["alpz"] or not stages["thr"]:
        return None
    a_stage, t_stage = main_stage(df, "alpz"), main_stage(df, "thr")
    res = _checked_contrast(df, "ALPZ_THR", (a_stage, "alpz"), (t_stage, "thr"), delta=delta,
                            n_instances=n_instances, n_routes=n_routes, n_reps=n_reps)
    key = ["inst", "route_id", "rep"]
    a = df[(df["stage"] == a_stage) & (df["policy"] == "alpz")].set_index(key)
    t = df[(df["stage"] == t_stage) & (df["policy"] == "thr")].set_index(key)
    both = a.index.intersection(t.index)
    ca, ct = a.loc[both, "mean_cost"], t.loc[both, "mean_cost"]
    same = (ca - ct).abs() <= SAME_COST_TOL * ct.abs().clip(lower=1.0)
    out = dict(n=res["n"], mean=res["mean"], ci95=res["ci95"], n_better=res["n_better"],
               identical_share=float(same.mean()), route_reps=int(len(both)))
    if "handoff_rate" in df.columns:
        out["handoff_rate_gap"] = float((a.loc[both, "handoff_rate"]
                                         - t.loc[both, "handoff_rate"]).abs().mean())
    return out


def strata_table(df: pd.DataFrame, arms, delta: float = DELTA) -> pd.DataFrame:
    """Section 8 item 5, descriptive: FULL against each arm by stratum (medium, long). The
    instance deltas are taken over the routes of the stratum; no coverage rule is applied."""
    full = main_stage(df, "full")
    out = []
    for arm, stage in _arm_stages(df, arms):
        pair = _pair(df, (full, FULL), (stage, arm))
        for stratum in sorted(set(pair["stratum"].astype(str))):
            sub = pair[pair["stratum"].astype(str) == stratum]
            d = A.instance_deltas(sub, "A", "B", N_PATHS).to_numpy()
            lo, hi = A._ci(d, 0.975) if d.size > 1 else (float("nan"), float("nan"))
            out.append(dict(arm=arm, stratum=stratum, n=int(d.size), mean=float(d.mean()),
                            ci95_lo=lo, ci95_hi=hi, n_better=int((d > 0).sum())))
    return pd.DataFrame(out, columns=STRATA_COLUMNS)


def replicate_sd(df: pd.DataFrame) -> pd.DataFrame:
    """Section 8 item 4: between-replicate SD of the route mean cost, per policy, in percent
    of the route's replicate mean (sample SD over the replicates; mean and maximum over the
    routes). Needs at least two replicates per route."""
    key = ["policy", "inst", "route_id"]
    grp = df.groupby(key)["mean_cost"]
    sd = 100.0 * grp.std(ddof=1) / grp.mean().abs()
    out = []
    for policy, vals in sd.groupby(level="policy", sort=True):
        out.append(dict(arm=policy, routes=int(vals.size), mean_sd_pct=float(vals.mean()),
                        max_sd_pct=float(vals.max())))
    return pd.DataFrame(out, columns=REPLICATE_COLUMNS)


def thr_over_dp(df: pd.DataFrame, dp: dict) -> dict | None:
    """Section 3.8: THR's realised test_b cost against its own-model DP value, pooled over
    route-reps, in percent of the DP value. ``dp`` maps ``inst|route|rep`` to the DP value
    (written into fidelity_dev.md by fidelity). None without DP values or THR rows."""
    stages = [s for s in set(df["stage"]) if G.core_of_stage(s) == "thr"]
    if not dp or not stages:
        return None
    rows = df[(df["stage"] == main_stage(df, "thr")) & (df["policy"] == "thr")]
    keys = [f"{r.inst}|{r.route_id}|{r.rep}" for r in rows.itertuples()]
    missing = [k for k in keys if k not in dp]
    if missing:
        _refuse(f"DP value missing for {len(missing)} THR route-reps (first: {missing[0]})")
    cost = float(rows["mean_cost"].sum())
    model = float(sum(dp[k] for k in keys))
    return dict(excess_pct=100.0 * (cost - model) / model, route_reps=len(keys))


def re_eval(df: pd.DataFrame, delta: float = DELTA, *, n_instances: int = N_INSTANCES,
            n_routes: int = N_ROUTES, n_reps: int = N_REPS) -> dict:
    """RE-EVAL (section 1.1, outside F): A's unadjusted go/no-go (one-sided 0.025 per
    direction) on FULL vs baton_eq at 10^5 on test_b. Not a replication: test days only."""
    full = main_stage(df, "full")
    ladder = main_stage(df, "ladder")
    pair = _pair(df, (full, FULL), (ladder, BASELINE))
    try:
        A.check_coverage(pair, ("RE-EVAL", "A", N_PATHS, "B", N_PATHS), n_instances,
                         n_routes, n_reps)
    except ValueError as exc:
        _refuse(str(exc))
    d = A.instance_deltas(pair, "A", "B", N_PATHS).to_numpy()
    return A.go_no_go(d, delta)


def build_report(df: pd.DataFrame, lists: dict, *, delta: float = DELTA,
                 n_instances: int = N_INSTANCES, n_routes: int = N_ROUTES,
                 n_reps: int = N_REPS, labels: dict | None = None,
                 thr_dp: dict | None = None) -> dict:
    """Membership checks, then every table. Nothing is omitted because of its result.
    ``labels`` maps arm -> budget-limited (section 6.3), ``thr_dp`` the DP values of 3.8."""
    check_membership(df, lists)
    independence = set(lists["independence"])
    main = df[~df["stage"].isin(independence)]
    ind = df[df["stage"].isin(independence)]
    cut = cut_arms(lists)
    labels = dict(labels or {})
    reasons = {arm: "descriptive" for arm in lists["descriptive"]}
    for item in lists["dropped"]:
        if not item.endswith(CUT_SUFFIX):
            reasons[item.split(":")[0]] = f"dropped={item}"
    for arm in reasons:
        if labels.get(arm):
            reasons[arm] += f"; {BUDGET_LABEL}"
    evaluated = [a for a in B_ARMS if a in set(lists["family"]) | set(reasons)]
    cover = dict(delta=delta, n_instances=n_instances, n_routes=n_routes, n_reps=n_reps)
    return dict(
        lists=lists, labels=labels, cut=cut,
        family=family_table(main, lists["family"], **cover),
        outside=outside_table(main, reasons, **cover),
        independence=independence_table(ind, delta) if independence else None,
        re_eval=re_eval(main, **cover),
        da=da_table(main, evaluated, **cover),
        alpz_thr=alpz_vs_thr(main, **cover),
        thr_dp=thr_over_dp(main, thr_dp or {}),
        strata=strata_table(main, evaluated, delta),
        rep_sd=replicate_sd(main),
    )


def compute_table(models_dirs) -> pd.DataFrame:
    """Section 4.2 accounting per fitted payload: fit seconds, budget fields, device."""
    from ..expa import sweep     # noqa: PLC0415 (payload reader; sweep imports torch)
    rows = []
    for models_dir in models_dirs:
        models_dir = Path(models_dir)
        files = sorted(models_dir.glob("*.pkl"))
        if not files:
            _refuse(f"no model payloads in {models_dir}")
        for path in files:
            payload = sweep.load_payload(path)
            budget = payload.get("budget") or {}
            row = dict(stage=models_dir.name, arm=payload["job"].arm,
                       rep=payload["job"].rep, fit_seconds=payload.get("fit_seconds"))
            row.update({k: budget.get(k) for k in COMPUTE_FIELDS})
            rows.append(row)
    return pd.DataFrame(rows, columns=COMPUTE_COLUMNS)


def payload_cut_reps(models_dirs) -> set:
    """Replicate counts of the cut (iii) record carried by the B payloads (``cuts``)."""
    from ..expa import sweep     # noqa: PLC0415
    found = set()
    for models_dir in models_dirs:
        for path in sorted(Path(models_dir).glob("*.pkl")):
            cuts = sweep.load_payload(path).get("cuts")
            if cuts is not None:
                found.add(len(cuts["reps"]))
    return found


def resolve_n_reps(requested: int | None, models_dirs) -> int:
    """Replicates per route: the cut recorded in the payloads, else ``stages.ACTIVE_CUTS``;
    an explicit ``--n-reps`` must agree with the payload record (section 9.2 (iii))."""
    recorded = payload_cut_reps(models_dirs or [])
    if len(recorded) > 1:
        _refuse(f"payloads record different cut (iii) replicate counts {sorted(recorded)}")
    if requested is None:
        return recorded.pop() if recorded else len(stages.ACTIVE_CUTS.reps)
    if recorded and requested not in recorded:
        _refuse(f"--n-reps {requested} contradicts the cut recorded in the payloads "
                f"({sorted(recorded)[0]} replicates)")
    return int(requested)


def _float_or_nan(value) -> float:
    return float("nan") if value is None or pd.isna(value) else float(value)


def compute_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per arm: payload count, median and maximum fit seconds, median path-stop reads,
    maximum peak RSS, median parameters and the devices used."""
    rows = []
    for arm, grp in table.groupby("arm", sort=True):
        devices = sorted({str(x) for x in grp["device"].dropna()})
        rows.append(dict(
            arm=arm, payloads=len(grp),
            fit_median_s=_float_or_nan(grp["fit_seconds"].median()),
            fit_max_s=_float_or_nan(grp["fit_seconds"].max()),
            path_stop_reads_median=_float_or_nan(grp["path_stop_reads"].median()),
            peak_rss_max_mb=_float_or_nan(grp["peak_rss_mb"].max()),
            n_params_median=_float_or_nan(grp["n_params"].median()),
            devices=", ".join(devices) or "-"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- rendering

def _cell(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "nan" if np.isnan(value) else f"{value:.4g}"
    return str(value)


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_"
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def _ci(lo: float, hi: float) -> str:
    return f"[{lo:+.2f}, {hi:+.2f}]"


def arm_claim(row: pd.Series, delta: float = DELTA, budget_limited: bool = False) -> str:
    """The claim template of section 1.3 for one family row. Never 'FULL = arm'. A
    budget-limited arm carries the label in the sentence (arm-level qualifier of 1.3)."""
    arm, branch = row["arm"], row["branch"]
    if budget_limited:
        arm = f"{arm} ({BUDGET_LABEL})"
    mean = float(row["mean"])
    ci95 = _ci(row["ci95_lo"], row["ci95_hi"])
    holm = _ci(row["holm_lo"], row["holm_hi"])
    if branch == "WIN":
        return (f"At 10^5 paths, FULL lowers recourse cost by {mean:.2f}% {ci95} versus "
                f"{arm} trained on the same paths ({int(row['n_better'])}/{int(row['n'])} "
                f"instances; Holm-level CI {holm}).")
    if branch == "LOSE":
        return (f"{arm} is {-mean:.2f}% cheaper than FULL {ci95} (Holm-level CI {holm}); "
                "the section 12 threat check applies.")
    if branch == "TIE_STRONG":
        text = (f"FULL and {arm} are equivalent within ±{delta:.1f}% "
                f"{_ci(row['ci90_lo'], row['ci90_hi'])} (CI90; per arm, not simultaneous)")
        if row["p_holm"] < HOLM_FWER:
            text += f", although significantly different (p_Holm = {row['p_holm']:.3g})"
        return text + "."
    return (f"No difference of ±{delta:.1f}% was established between FULL and {arm} "
            f"(CI95 {ci95}).")


def _family_display(family: pd.DataFrame) -> pd.DataFrame:
    shown = family.copy()
    for col in ("ci95_lo", "ci95_hi", "holm_lo", "holm_hi", "ci90_lo", "ci90_hi"):
        shown[col] = shown[col].map(lambda x: f"{x:+.2f}")
    shown["n_better"] = [f"{int(k)}/{int(n)}" for k, n in zip(family["n_better"], family["n"])]
    shown["ci_95"] = [f"[{a}, {b}]" for a, b in zip(shown["ci95_lo"], shown["ci95_hi"])]
    shown["holm_ci"] = [f"[{a}, {b}]" for a, b in zip(shown["holm_lo"], shown["holm_hi"])]
    shown["ci_90"] = [f"[{a}, {b}]" for a, b in zip(shown["ci90_lo"], shown["ci90_hi"])]
    return shown[["arm", "n", "mean", "median", "ci_95", "holm_ci", "ci_90", "n_better",
                  "rank_biserial", "pooled_ratio", "gc_mean_full", "p_raw", "p_holm",
                  "branch"]]


def render_summary(report: dict, *, family_line: str, eval_commit: str,
                   compute: pd.DataFrame | None = None, delta: float = DELTA) -> str:
    lists, fam, outside = report["lists"], report["family"], report["outside"]
    lines = [
        "# EXPERIMENT B: test_b analysis", "",
        f"Stream: `{STREAM}` (20,000 days per instance). Evaluation commit: `{eval_commit}`.",
        f"FAMILY line (HEAD, pinned): `{family_line}`",
        f"family_sha256: `{family_hash(family_line)}`", "",
        f"Family F = {lists['family']}; descriptive = {lists['descriptive']}; "
        f"dropped = {lists['dropped']}; references = {lists['references']}; "
        f"independence = {lists['independence']}.", "",
        "**All family arms are reported below regardless of outcome. No arm is omitted from "
        "any table because of its result.**", "",
        f"## 1. Primary family (FULL vs each arm at 10^5, Holm over |F| = {len(fam)})", "",
        _md(_family_display(fam)), "", "Claims (section 1.3 templates):", ""]
    labels = report.get("labels", {})
    lines += [f"- **{row['arm']}** [{row['branch']}]: "
              f"{arm_claim(row, delta, bool(labels.get(row['arm'])))}"
              for _, row in fam.iterrows()]
    if report.get("cut"):
        lines += ["", f"Arms removed by a compute cut (9.2 (ii)), not fitted and not "
                      f"evaluated: {report['cut']}."]
    lines += ["", "## 2. Descriptive and dropped arms (outside F; unadjusted; no branch)", "",
              _md(outside), ""]
    lines += ["## 3. Independence world (section 3.8; descriptive; no Holm; 12 routes, rep 0)",
              "", _md(report["independence"]) if report["independence"] is not None
              else "_(no independence stages listed)_", ""]
    re = report["re_eval"]
    lines += ["## 4. RE-EVAL (outside F; renamed from REP; re-samples test days only; "
              "not a replication)", "",
              f"FULL vs baton_eq at 10^5 on test_b: Δ̄ = {re['mean']:+.2f}% "
              f"{_ci(*re['ci95'])} (CI95), {re['n_pos']}/{re['n']} instances better; A's "
              f"unadjusted rule gives **{re['branch']}** (one-sided p: greater "
              f"{re['p_greater']:.3g}, less {re['p_less']:.3g}).", "",
              "## 5. D_a: each evaluated arm vs baton_eq at 10^5 (outside F; A's unadjusted "
              "rule)", "", _md(report["da"]), "",
              "## 6. ALP-Z vs THR on test_b (descriptive)", ""]
    av = report["alpz_thr"]
    lines += ["_(an arm is absent)_" if av is None else
              f"Paired difference (positive: ALP-Z cheaper) {av['mean']:+.3f}% "
              f"{_ci(*av['ci95'])}, ALP-Z better on {av['n_better']}/{av['n']} instances; "
              f"identical mean cost on {av['identical_share']:.1%} of {av['route_reps']} "
              "route-reps (rows hold costs, not decisions)"
              + (f"; mean |handoff-rate gap| {av['handoff_rate_gap']:.4f}"
                 if "handoff_rate_gap" in av else "") + ".", "",
              "## 7. THR realised cost over its own-model DP value (section 3.8)", ""]
    td = report["thr_dp"]
    lines += ["_(no DP values in the derivation block)_" if td is None else
              f"THR's realised test_b cost exceeds its own-model DP value by "
              f"{td['excess_pct']:+.2f}% (pooled over {td['route_reps']} route-reps).", "",
              "## 8. Strata (descriptive; FULL vs arm, positive: FULL cheaper)", "",
              _md(report["strata"]), "",
              "## 9. Between-replicate SD of the route mean cost (percent)", "",
              _md(report["rep_sd"]), "",
              "## 10. Wall-clock and budget", ""]
    if compute is None:
        lines += ["_(no payload directories given with --models)_", ""]
    else:
        lines += [_md(compute_summary(compute)), ""]
    return "\n".join(lines) + "\n"


def write_outputs(report: dict, out_dir, *, family_line: str, eval_commit: str,
                  compute: pd.DataFrame | None = None, delta: float = DELTA) -> Path:
    """Writes summary.md and the CSV tables into ``out_dir``. Never calls A's
    ``write_summary``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = render_summary(report, family_line=family_line, eval_commit=eval_commit,
                          compute=compute, delta=delta)
    (out / "summary.md").write_text(text, encoding="utf-8")
    report["family"].to_csv(out / "family.csv", index=False)
    report["outside"].to_csv(out / "outside.csv", index=False)
    if report["independence"] is not None:
        report["independence"].to_csv(out / "independence.csv", index=False)
    report["da"].to_csv(out / "da.csv", index=False)
    report["strata"].to_csv(out / "strata.csv", index=False)
    report["rep_sd"].to_csv(out / "replicate_sd.csv", index=False)
    if compute is not None:
        compute.to_csv(out / "compute.csv", index=False)
        compute_summary(compute).to_csv(out / "compute_summary.csv", index=False)
    return out / "summary.md"


# ---------------------------------------------------------------- analysis pin and CLI

def patch_hashes(lines) -> list[str]:
    """analysis_sha256 of every ANALYSIS PATCH line, in log order (section 5.3)."""
    out = []
    for line in lines:
        if line.startswith(G.PATCH_PREFIX):
            match = G.PATCH_LINE.fullmatch(line)
            if match is None:
                _refuse("an ANALYSIS PATCH line is malformed")
            out.append(match.group(1))
    return out


def check_analysis_pin(current: str, pinned: str, patches: list[str]) -> None:
    """tree_hash(HEAD, ANALYSIS) must equal the pinned hash or the last ANALYSIS PATCH."""
    accepted = {pinned} | ({patches[-1]} if patches else set())
    if current not in accepted:
        _refuse("tree_hash(HEAD, ANALYSIS) matches neither the pinned analysis_sha256 nor "
                "the last ANALYSIS PATCH line; add a patch line first (5.3)")


def _head(root: Path) -> str:
    proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=False, timeout=120)
    if proc.returncode != 0:
        _refuse(f"git rev-parse HEAD failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m transvrp.expb.analysis_b")
    ap.add_argument("--merged", type=Path, nargs="+", required=True,
                    help="one merged <stage>.test_b.csv per B stage")
    ap.add_argument("--eval", dest="eval_set", required=True, choices=(STREAM,))
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--models", type=Path, action="append", default=None,
                    help="a models/<stage> directory of payloads (wall-clock and budget)")
    ap.add_argument("--repo", type=Path, default=None,
                    help="repository root (default: the checkout this module lives in)")
    ap.add_argument("--n-routes", type=int, default=N_ROUTES,
                    help="eligible routes covered (62; fewer only by a pre-committed cut)")
    ap.add_argument("--n-reps", type=int, default=None,
                    help="replicates per route; default: the cut recorded in the --models "
                         "payloads, else the active cut (3; 2 under cut (iii), section 9.2)")
    args = ap.parse_args(argv)
    try:
        root = G._root(args.repo)
        head = _head(root)
        G.check_analysis_clean(head)          # working tree == hashed blobs (review M4)
        pinned = G.pinned_log(root, head)
        check_analysis_pin(G.tree_hash(head, list(G.ANALYSIS_RELS), root),
                           pinned.hashes["analysis_sha256"],
                           patch_hashes(G.log_text(head, root).splitlines()))
        frozen = {stage: commit for commit, stages in pinned.frozen for stage in stages}
        df = load_stage_tables(args.merged, family_sha256=family_hash(pinned.family_line),
                               frozen=frozen)
        block = G.read_derivation(head, root)
        n_reps = resolve_n_reps(args.n_reps, args.models)
        report = build_report(df, pinned.lists, labels=block.get("budget_limited"),
                              thr_dp=block.get("thr_dp"), n_routes=args.n_routes,
                              n_reps=n_reps)
        compute = compute_table(args.models) if args.models else None
        path = write_outputs(report, args.out, family_line=pinned.family_line,
                             eval_commit=str(df["eval_commit"].iloc[0]), compute=compute)
    except (AnalysisRefused, G.GateRefused, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

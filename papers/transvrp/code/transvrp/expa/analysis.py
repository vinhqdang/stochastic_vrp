"""Decision rules and summary tables for EXPERIMENT A (module M10).

EXPERIMENT_A.md sections 1.2, 1.4, 1.5, 6, 8 and M10.

Unit of analysis: the instance (n = 20). For a contrast "a vs b" at a stated N,

    C_x,r = mean over replicates of the route's mean test cost,
    delta_r = 100 * (C_b,r - C_a,r) / C_b,r        (positive: a is cheaper),
    delta_i = mean of delta_r over the routes of instance i.

``classify`` implements the four-way rule of section 1.5 (better, worse,
equivalent, inconclusive). ``go_no_go`` implements the primary rule of
section 1.4 (WIN, LOSE, TIE_STRONG, TIE_WEAK). The two differ in one place:
section 1.4 tests each direction one-sided at 0.025, while section 1.5 uses
the two-sided Holm-adjusted p (section 8) with threshold 0.05.

Run ``python -m transvrp.expa.analysis --merged <stage>.csv --big <stage>.csv``
to write ``summary.md`` and ``curve.csv``. Nothing is written to the repository
unless the output directory is given.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DELTA = 0.7                # section 1.3, frozen
N_INSTANCES = 20
WIN_MIN_COUNT = 13         # section 1.4
ALPHA_DIRECTION = 0.025    # section 1.4, Bonferroni over two directions
ALPHA_ATTRIBUTION = 0.05   # section 1.5 and M11
TOL = 1e-9                 # boundary tolerance for delta (M10 test: Delta = delta included)
BRANCHES = ("WIN", "LOSE", "TIE_STRONG", "TIE_WEAK")

# section 8 family (Holm, FWER 0.05). (id, policy a, N_a, policy b, N_b).
FAMILY = [
    ("S1", "nn_full", 100, "baton_eq", 100),
    ("S2", "nn_full", 500, "baton_eq", 500),
    ("S3", "nn_full", 2000, "baton_eq", 2000),
    ("S4", "nn_full", 10000, "baton_eq", 10000),
    ("S5", "nn_full", 100000, "nn_scalar", 100000),
    ("S6", "nn_full", 2000, "nn_scalar", 2000),
    ("S7", "nn_scalar", 100000, "dp_big", 1000000),
    ("S8", "nn_scalar", 2000, "baton_eq", 2000),
    ("S9", "nn_full", 100000, "nn_mlp", 100000),
    ("S10", "nn_full", 100000, "baton_big", 1000000),
    ("S11", "nn_full", 100000, "dp_big", 1000000),
]
PRIMARY = ("PRIMARY", "nn_full", 100000, "baton_eq", 100000)
M11 = ("M11", "nn_full", 100000, "boost", 100000)
CURVE_POLICIES = ("nn_full", "baton_eq")


# ---------------------------------------------------------------- statistics

def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return [float(x) for x in adj]


def _ci(d: np.ndarray, q: float) -> tuple[float, float]:
    """Mean +/- t_{n-1, q} * s / sqrt(n); q = 0.975 for CI95, 0.95 for CI90."""
    n = d.size
    half = stats.t.ppf(q, n - 1) * d.std(ddof=1) / np.sqrt(n)
    mean = float(d.mean())
    return mean - half, mean + half


def _wilcoxon_p(d: np.ndarray, alternative: str = "two-sided") -> float:
    if np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(d, alternative=alternative, method="exact").pvalue)


def _rank_biserial(d: np.ndarray) -> float:
    nz = d[d != 0]
    if nz.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


# ---------------------------------------------------------------- data

def _policy_rows(df: pd.DataFrame, policy: str, N: int) -> pd.DataFrame:
    sub = df[(df["policy"] == policy) & (df["N"] == N)]
    if sub.empty:
        raise ValueError(f"no rows for policy {policy!r} at N = {N}")
    return sub


def _paired_route_means(df: pd.DataFrame, a: str, N_a: int, b: str, N_b: int
                        ) -> pd.DataFrame:
    """Per-route mean over replicates for both policies (columns a, b)."""
    ra = _policy_rows(df, a, N_a)
    rb = _policy_rows(df, b, N_b)
    key = ["inst", "route_id", "rep"]
    ka = ra.set_index(key)["mean_cost"]
    kb = rb.set_index(key)["mean_cost"]
    if ka.index.has_duplicates or kb.index.has_duplicates:
        raise ValueError("duplicate (inst, route, rep) rows")
    common = ka.index.intersection(kb.index)
    if len(common) != len(ka) or len(common) != len(kb):
        raise ValueError(f"{a}@{N_a} and {b}@{N_b} do not cover the same routes and replicates")
    pair = pd.DataFrame({"a": ka.loc[common], "b": kb.loc[common]})
    return pair.groupby(level=["inst", "route_id"]).mean()


def instance_deltas(df: pd.DataFrame, a: str, b: str, N: int,
                    N_b: int | None = None) -> pd.Series:
    """delta_i (section 1.2) for "a vs b", indexed by instance. ``N_b`` allows the
    comparison of a policy at one size with a policy fitted at another (S7,
    S10, S11 use 10^6 paths for the classical side)."""
    N_b = N if N_b is None else N_b
    routes = _paired_route_means(df, a, N, b, N_b)
    delta_r = 100.0 * (routes["b"] - routes["a"]) / routes["b"]
    return delta_r.groupby(level="inst").mean()


def classify(d: np.ndarray, delta: float = DELTA, p_adj: float | None = None) -> str:
    """Section 1.5: better, worse, equivalent or inconclusive.

    ``p_adj`` is the Holm-adjusted two-sided p of the family. When it is None,
    the unadjusted exact two-sided Wilcoxon p is used.
    """
    d = np.asarray(d, dtype=float)
    mean = float(d.mean())
    lo95, hi95 = _ci(d, 0.975)
    lo90, hi90 = _ci(d, 0.95)
    p = _wilcoxon_p(d) if p_adj is None else float(p_adj)
    if mean >= delta - TOL and lo95 > 0 and p < ALPHA_ATTRIBUTION:
        return "better"
    if mean <= -(delta - TOL) and hi95 < 0 and p < ALPHA_ATTRIBUTION:
        return "worse"
    if lo90 > -delta and hi90 < delta:
        return "equivalent"
    return "inconclusive"


def go_no_go(d: np.ndarray, delta: float = DELTA) -> dict:
    """Primary decision of section 1.4 on the 20 instance deltas."""
    d = np.asarray(d, dtype=float)
    n = d.size
    mean = float(d.mean())
    lo95, hi95 = _ci(d, 0.975)
    lo90, hi90 = _ci(d, 0.95)
    p_greater = _wilcoxon_p(d, "greater")
    p_less = _wilcoxon_p(d, "less")
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    if (mean >= delta - TOL and lo95 > 0 and p_greater < ALPHA_DIRECTION
            and n_pos >= WIN_MIN_COUNT):
        branch = "WIN"
    elif (mean <= -(delta - TOL) and hi95 < 0 and p_less < ALPHA_DIRECTION
          and n_neg >= WIN_MIN_COUNT):
        branch = "LOSE"
    elif lo90 > -delta and hi90 < delta:
        branch = "TIE_STRONG"
    else:
        branch = "TIE_WEAK"
    if p_greater < ALPHA_DIRECTION:
        flag = "better"
    elif p_less < ALPHA_DIRECTION:
        flag = "worse"
    else:
        flag = None
    return dict(branch=branch, n=n, mean=mean, median=float(np.median(d)),
                ci95=(lo95, hi95), ci90=(lo90, hi90), sd=float(d.std(ddof=1)),
                p_greater=p_greater, p_less=p_less, n_pos=n_pos, n_neg=n_neg,
                significance=flag, delta=float(delta))


# ---------------------------------------------------------------- tables

def _gc_mean(df: pd.DataFrame, policy: str, N: int) -> float:
    sub = df[(df["policy"] == policy) & (df["N"] == N)]["gap_closed"].dropna()
    return float(sub.mean()) if len(sub) else float("nan")


def contrast(df: pd.DataFrame, cid: str, a: str, N_a: int, b: str, N_b: int,
             delta: float = DELTA, p_adj: float | None = None) -> dict | None:
    """Effect sizes and classification of one contrast (section 8, item 4).
    Returns None when the rows are missing."""
    try:
        d = instance_deltas(df, a, b, N_a, N_b=N_b).to_numpy()
        routes = _paired_route_means(df, a, N_a, b, N_b)
    except ValueError:
        return None
    lo95, hi95 = _ci(d, 0.975)
    lo90, hi90 = _ci(d, 0.95)
    return dict(id=cid, a=a, N_a=N_a, b=b, N_b=N_b, n=d.size,
                mean=float(d.mean()), median=float(np.median(d)),
                ci95=(lo95, hi95), ci90=(lo90, hi90),
                n_better=int((d > 0).sum()), rank_biserial=_rank_biserial(d),
                pooled_ratio=float(routes["a"].sum() / routes["b"].sum()),
                gc_mean_a=_gc_mean(df, a, N_a), p_two_sided=_wilcoxon_p(d),
                sd_instances=float(d.std(ddof=1)),
                classification=classify(d, delta, p_adj=p_adj), _d=d)


def curve(df: pd.DataFrame, delta: float = DELTA) -> pd.DataFrame:
    """Delta-bar (FULL vs BATON-eq) with CI95 against log10 N over the grid."""
    sub = df[df["policy"].isin(CURVE_POLICIES)]
    grid = sorted(set(sub.loc[sub["policy"] == "nn_full", "N"].astype(int)))
    rows = []
    for N in grid:
        c = contrast(df, f"curve@{N}", "nn_full", N, "baton_eq", N, delta)
        if c is None:
            continue
        rows.append(dict(N=N, log10_N=float(np.log10(N)), delta_bar=c["mean"],
                         ci95_lo=c["ci95"][0], ci95_hi=c["ci95"][1]))
    return pd.DataFrame(rows, columns=["N", "log10_N", "delta_bar", "ci95_lo", "ci95_hi"])


def crossover_n(curve_df: pd.DataFrame, delta: float = DELTA) -> int | None:
    """Smallest grid N with CI95 lower > -delta at it and at every larger grid N
    (section 8, item 5). None if no such N exists."""
    c = curve_df.sort_values("N").reset_index(drop=True)
    ok = (c["ci95_lo"] > -delta).to_numpy()
    for i in range(len(c)):
        if ok[i:].all():
            return int(c.loc[i, "N"])
    return None


# ---------------------------------------------------------------- summary

def _fmt_ci(ci) -> str:
    return f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"


def _claim_primary(res: dict, df: pd.DataFrame, delta: float) -> list[str]:
    lines = []
    b = res["branch"]
    if b == "WIN":
        lines.append(f"At 10^5 paths, FULL lowers recourse cost by {res['mean']:.2f}% "
                     f"{_fmt_ci(res['ci95'])} versus BATON at equal paths "
                     f"(n = {res['n']} instances; {res['n_pos']}/{res['n']} better).")
    elif b == "LOSE":
        lines.append(f"FULL is {-res['mean']:.2f}% {_fmt_ci(res['ci95'])} dearer than BATON.")
        s7 = contrast(df, "S7", "nn_scalar", 100000, "dp_big", 1000000, delta=delta)
        if s7 is not None and s7["classification"] == "worse":
            lines[-1] = lines[-1][:-1] + (
                f", which is attributable to a pipeline deficit of "
                f"{-s7['mean']:.2f}% (SCALAR vs DP-1e6).")
        else:
            lines[-1] = lines[-1][:-1] + (
                ", although the pipeline matches the scalar optimum, so the "
                "richer state does not pay here.")
    elif b == "TIE_STRONG":
        s = f"FULL and BATON are equivalent within ±{delta:.1f}% {_fmt_ci(res['ci90'])} (CI90)."
        if res["significance"] == "better":
            s += f" although FULL is significantly better by {res['mean']:.2f}%."
        elif res["significance"] == "worse":
            s += f" although FULL is significantly worse by {-res['mean']:.2f}%."
        lines.append(s)
    else:
        lines.append(f"No difference of ±{delta:.1f}% was established; the comparison is "
                     f"inconclusive (CI95 {_fmt_ci(res['ci95'])}).")
    return lines


def _combine(df: pd.DataFrame, big_df: pd.DataFrame | None) -> pd.DataFrame:
    """Main and big stage rows as one table. The two stages share the classical
    10^6 rows of the subset routes (rep 0); they must agree, and one copy is kept."""
    if big_df is None:
        return df
    key = ["inst", "route_id", "N", "rep", "policy"]
    both = pd.concat([df, big_df], ignore_index=True)
    dup = both[both.duplicated(subset=key, keep=False)]
    if not dup.empty:
        spread = dup.groupby(key)["mean_cost"].agg(lambda x: x.max() - x.min())
        if (spread > 1e-9).any():
            raise ValueError("main and big stages disagree on shared rows")
    return both.drop_duplicates(subset=key, keep="first").reset_index(drop=True)


def write_summary(df: pd.DataFrame, out_dir: Path, big_df: pd.DataFrame | None = None,
                  delta: float = DELTA) -> dict:
    """Writes summary.md and curve.csv into out_dir and returns the key results.

    ``df`` is the merged main-stage table. ``big_df`` (optional) is the merged big
    stage; its rows are concatenated so that the 10^6 classical rows are found.
    """
    full = _combine(df, big_df)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prim = contrast(full, *PRIMARY, delta=delta)
    fam_rows = [contrast(full, *spec, delta=delta) for spec in FAMILY]
    present = [r for r in fam_rows if r is not None]
    adj = holm([r["p_two_sided"] for r in present])
    for r, p in zip(present, adj):
        r["p_holm"] = p
        r["classification"] = classify(r["_d"], delta, p_adj=p)
    # attribution contrasts that are not in the family use their own alpha
    m11 = contrast(full, *M11, delta=delta)

    go = go_no_go(prim["_d"], delta) if prim is not None else None
    cur = curve(full, delta)
    cross = crossover_n(cur, delta) if len(cur) else None

    by_id = {r["id"]: r for r in present}
    lines = ["# EXPERIMENT A summary", "",
             f"delta = {delta:.2f}% of BATON recourse cost (section 1.3). "
             "Unit: instance (n = 20). CI95 uses t_{19,0.975}; CI90 uses t_{19,0.95}.", ""]
    if go is None:
        lines += ["## Primary (section 1.4)", "", "Not available: primary rows missing.", ""]
    else:
        lines += ["## Primary: FULL vs BATON-eq at N = 10^5 (section 1.4)", "",
                  f"Branch: **{go['branch']}**. Mean Δ̄ = {go['mean']:+.3f}%, "
                  f"median {go['median']:+.3f}%, CI95 {_fmt_ci(go['ci95'])}, "
                  f"CI90 {_fmt_ci(go['ci90'])}, s = {go['sd']:.3f}.",
                  f"One-sided exact Wilcoxon: p(FULL cheaper) = {go['p_greater']:.4g}, "
                  f"p(FULL dearer) = {go['p_less']:.4g}. Instances better: "
                  f"{go['n_pos']}/{go['n']}, worse: {go['n_neg']}/{go['n']}. "
                  f"Significance flag: {go['significance'] or 'none'}.", ""]
        lines += ["Claim (template filled from the data):", ""]
        lines += [f"> {x}" for x in _claim_primary(go, full, delta)]
        lines += [""]

    lines += ["## Secondary family (section 8, Holm over S1 to S11)", "",
              "| id | a @ N | b @ N | Δ̄ % | median | CI95 | CI90 | better | rank-biserial | pooled ratio a/b | mean GC(a) | p (two-sided) | p (Holm) | class |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in present:
        lines.append(
            f"| {r['id']} | {r['a']} @ {r['N_a']} | {r['b']} @ {r['N_b']} | "
            f"{r['mean']:+.3f} | {r['median']:+.3f} | {_fmt_ci(r['ci95'])} | "
            f"{_fmt_ci(r['ci90'])} | {r['n_better']}/{r['n']} | {r['rank_biserial']:+.2f} | "
            f"{r['pooled_ratio']:.4f} | {r['gc_mean_a']:.3f} | {r['p_two_sided']:.4g} | "
            f"{r['p_holm']:.4g} | {r['classification']} |")
    if not present:
        lines.append("| - | no family rows present | | | | | | | | | | | | |")
    lines.append("")

    lines += ["## Attribution (section 1.5)", ""]
    h2 = by_id.get("S5")
    h3 = by_id.get("S7")
    arch = by_id.get("S9")
    if h2 is not None:
        c = h2["classification"]
        txt = {"better": "the gain is information",
               "equivalent": "the richer state carries no usable information here"}.get(
                   c, "reported as such (no information claim)")
        lines.append(f"- H2 (FULL vs SCALAR @ 10^5): {c}; {txt}.")
    if h3 is not None:
        c = h3["classification"]
        txt = {"equivalent": "the pipeline reaches the (k, W) optimum",
               "worse": "pipeline deficit",
               "better": "leakage audit before any claim (stop rule 3)"}.get(c, "no claim")
        lines.append(f"- H3 (SCALAR vs DP-1e6 @ 10^5): {c}; {txt}.")
    if arch is not None:
        c = arch["classification"]
        lines.append(f"- ARCH (FULL vs MLP @ 10^5): {c}; "
                     + ('the paper may say "set encoder".' if c == "better"
                        else 'the paper says only "neural regressor".'))
    if m11 is not None:
        c = m11["classification"]
        txt = {"better": "a neural regressor is needed",
               "equivalent": "trees on the same state suffice"}.get(c, "no claim")
        lines.append(f"- M11 (FULL vs BOOST @ 10^5, own alpha 0.05): {c}; {txt}.")
    else:
        lines.append("- M11 (FULL vs BOOST @ 10^5): not run.")
    if go is not None and go["branch"] == "WIN" and h2 is not None and h2["classification"] != "better":
        lines.append("- WIN without an H2 gain: the claim becomes \"the gain comes from the "
                     "regressor/procedure within BATON's information\"; S10 and S11 must "
                     "also be better, or the WIN is called fragile.")
        for sid in ("S10", "S11"):
            r = by_id.get(sid)
            if r is not None and r["classification"] != "better":
                lines.append(f"  - {sid} is {r['classification']}: the WIN is fragile.")
    lines.append("")

    lines += ["## Data-size curve (section 8, item 5)", "",
              "| N | Δ̄ % (FULL vs BATON-eq) | CI95 |", "|---|---|---|"]
    for _, row in cur.iterrows():
        lines.append(f"| {int(row['N'])} | {row['delta_bar']:+.3f} | "
                     f"[{row['ci95_lo']:+.2f}, {row['ci95_hi']:+.2f}] |")
    lines += ["", f"Crossover N: {cross if cross is not None else 'none'}.", ""]

    lines += ["## Variance reporting (section 8, item 2)", ""]
    for pol, N in (("nn_full", 100000), ("baton_eq", 100000)):
        sub = full[(full["policy"] == pol) & (full["N"] == N)]
        if sub.empty:
            continue
        sd = sub.groupby(["inst", "route_id"])["mean_cost"].std(ddof=1).median()
        lines.append(f"- median over routes of the between-replicate SD of {pol} @ {N}: "
                     f"{sd:.4f}.")
    lines += ["- (b), the per-day paired SE, needs the per-day arrays and is not in this table.", ""]

    (out_dir / "summary.md").write_text("\n".join(lines))
    cur.to_csv(out_dir / "curve.csv", index=False)
    return dict(primary=go, family={r["id"]: {k: v for k, v in r.items() if k != "_d"}
                                    for r in present},
                m11=None if m11 is None else m11["classification"],
                crossover_n=cross, curve=cur)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m transvrp.expa.analysis")
    ap.add_argument("--merged", type=Path, required=True,
                    help="merged main-stage CSV (dev or test)")
    ap.add_argument("--big", type=Path, default=None, help="merged big-stage CSV")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    args = ap.parse_args(argv)
    df = pd.read_csv(args.merged)
    big = pd.read_csv(args.big) if args.big else None
    res = write_summary(df, args.out, big)
    print({"branch": None if res["primary"] is None else res["primary"]["branch"],
           "crossover_n": res["crossover_n"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

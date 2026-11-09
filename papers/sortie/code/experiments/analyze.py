"""Tables (markdown + LaTeX) and figures for the SORTIE report from results/*.csv.

Run from papers/sortie/code:  python experiments/analyze.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SORTIE_RESULTS", os.path.join(HERE, "..", "..", "results"))
TAB = os.path.join(RES, "tables")
FIG = os.path.join(HERE, "..", "..", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

MEAS = ["max", "range", "gmd", "mad", "var"]
MNAME = {"max": "Max", "range": "Range", "gmd": "GMD", "mad": "MAD", "var": "Var"}
out_md = []


def load(name):
    p = os.path.join(RES, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def emit(title, df, fname, floatfmt="%.2f"):
    out_md.append(f"\n### {title}\n\n" + df.to_markdown(index=False, floatfmt=".3g") + "\n")
    with open(os.path.join(TAB, fname + ".tex"), "w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: floatfmt % x, na_rep="--"))


def wpval(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    if len(d) < 6 or np.allclose(d, 0):
        return np.nan
    return wilcoxon(d).pvalue


def e1():
    df = load("e1_accuracy")
    if df is None:
        return
    rows = []
    for meas, g in df.groupby("measure"):
        rows.append(dict(measure=MNAME.get(meas, meas), plans=len(g),
                         mean_abs_rel_err_pct=100 * g["rel_err"].abs().mean(),
                         max_abs_rel_err_pct=100 * g["rel_err"].abs().max(),
                         share_abs_z_lt_3=(g["z_score"].abs() < 3).mean()))
    emit("E1 - PBOI (gamma marginals) vs arc-level Monte Carlo", pd.DataFrame(rows), "e1_accuracy", "%.3f")
    t = load("e1_timing")
    if t is not None:
        emit("E1 - evaluation time per plan", t, "e1_timing", "%.4g")


def e2():
    df = load("e2_exact")
    if df is None:
        return
    emit("E2 - heuristic vs exact enumeration (tiny instances)", df.drop(columns=[c for c in df.columns if c in ("key",)], errors="ignore"), "e2_exact_raw", "%.3f")


def paired_table(df, base, others, metric, group):
    rows = []
    for keys, g in df.groupby(group):
        piv = g.pivot_table(index="instance", columns="method", values=metric)
        if base not in piv:
            continue
        row = dict(zip(group, keys if isinstance(keys, tuple) else (keys,)))
        row["inst"] = len(piv)
        for o in others:
            if o not in piv:
                continue
            ok = piv[[base, o]].dropna()
            imp = 100 * (ok[base] - ok[o]) / ok[base].abs()
            row[f"{o}_impr_pct"] = imp.mean()
            row[f"{o}_wins"] = int((ok[o] < ok[base] - 1e-9).sum())
            row[f"{o}_p"] = wpval(ok[base], ok[o])
        rows.append(row)
    return pd.DataFrame(rows)


def e3():
    df = load("e3_main")
    if df is None:
        return
    fair = df[df.method != "cost"].copy()
    # 1) out-of-sample improvement over det (true ex-post objective, with day factor)
    t = paired_table(fair, "det", ["saa", "sortie0", "sortie1"], "oos_obj", ["measure", "B"])
    emit("E3 - out-of-sample ex-post unfairness: improvement over the deterministic fair plan (%)", t, "e3_improvement")
    t2 = paired_table(fair, "det", ["saa", "sortie0", "sortie1"], "oos_obj", ["n", "B"])
    emit("E3 - improvement over det by size (all measures pooled)", t2, "e3_by_n")
    # 2) fairness illusion: what each method believes vs truth
    rows = []
    for (meth, meas), g in fair.groupby(["method", "measure"]):
        truth = g["oos_obj_X"]
        rows.append(dict(method=meth, measure=MNAME[meas],
                         insample_vs_truth_pct=100 * ((g["obj_insample"] - truth) / truth).mean(),
                         exante_vs_truth_pct=100 * ((g["exante"] - truth) / truth).mean(),
                         pboi_vs_truth_pct=100 * ((g["pboi_pred"] - truth) / truth).mean(),
                         cost_ratio=g["cost_ratio"].mean()))
    emit("E3 - believed vs true unfairness (mean signed error %, truth = MC without day factor)",
         pd.DataFrame(rows), "e3_illusion")
    # 3) absolute levels per method (averaged), plus cost plan
    lv = df.groupby(["measure", "method"])["oos_obj"].mean().unstack()
    lv = lv.reset_index()
    emit("E3 - mean out-of-sample ex-post value", lv, "e3_levels", "%.1f")
    # 4) CVaR of range across methods when optimising range
    r = df[(df.measure == "range")]
    cv = r.groupby(["B", "method"])["CVaR90_range"].mean().unstack().reset_index()
    emit("E3 - CVaR90 of the realised range (range-optimised plans)", cv, "e3_cvar", "%.1f")
    # 5) runtime / iterations
    it = fair.groupby("method")[["iters", "n_exact_evals", "runtime"]].mean().reset_index()
    emit("E3 - search effort", it, "e3_effort", "%.0f")


def e4():
    df = load("e4_inconsistency")
    if df is None:
        return
    rows = []
    for (meth, meas), g in df.groupby(["method", "measure"]):
        rows.append(dict(method=meth, measure=MNAME[meas], runs=len(g),
                         runs_with_nontsp=int((g["nontsp_routes"] > 0).sum()),
                         mean_nontsp_routes=g["nontsp_routes"].mean(),
                         mean_excess_len=g["excess_len"].mean(),
                         insample_vs_truth_pct=100 * ((g["obj_insample"] - g["oos_obj_X"]) / g["oos_obj_X"]).mean(),
                         oos_obj=g["oos_obj"].mean()))
    emit("E4 - route inconsistency: free tours vs lower-level tours", pd.DataFrame(rows), "e4_inconsistency")


def e5():
    df = load("e5_ablation")
    if df is None:
        return
    t = paired_table(df, "sortie0_random", ["sortie0"], "oos_obj", ["measure"])
    emit("E5 - rank-price screening vs random candidate moves (improvement of price screen, %)", t, "e5_ablation")
    it = df.groupby("method")[["iters", "n_exact_evals"]].mean().reset_index()
    emit("E5 - effort", it, "e5_effort", "%.0f")


def e6():
    df = load("e6_sensitivity")
    if df is None:
        return
    t = paired_table(df, "det", ["sortie0"], "oos_obj", ["v", "measure"])
    rows = []
    for (v, meas), g in df[df.method == "det"].groupby(["v", "measure"]):
        rows.append(dict(v=v, measure=MNAME[meas],
                         det_illusion_pct=100 * ((g["exante"] - g["oos_obj_X"]) / g["oos_obj_X"]).mean()))
    t = t.merge(pd.DataFrame(rows).assign(measure=lambda d: d.measure.map({v: k for k, v in MNAME.items()})),
                on=["v", "measure"], how="left")
    emit("E6 - sensitivity to arc noise v", t, "e6_sensitivity")


def e7():
    df = load("e7_scale")
    if df is None:
        return
    t = paired_table(df, "det", ["saa", "sortie0"], "oos_obj", ["n", "measure"])
    emit("E7 - scalability (n = 50-100)", t, "e7_scale")


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = load("e3_main")
    if df is not None:
        fair = df[df.method.isin(["det", "saa", "sortie0", "sortie1"])]
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharey=False)
        for ax, meas in zip(axes, MEAS):
            g = fair[(fair.measure == meas) & (fair.B == 1.10)]
            piv = g.pivot_table(index="instance", columns="method", values="oos_obj")
            data = [100 * (piv["det"] - piv[mth]) / piv["det"] for mth in ["saa", "sortie0", "sortie1"] if mth in piv]
            ax.boxplot(data, tick_labels=[m for m in ["SAA", "SORTIE-0", "SORTIE-1"]][:len(data)])
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_title(MNAME[meas])
        axes[0].set_ylabel("ex-post improvement over det (%)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "e3_improvement.pdf"))
        fig.savefig(os.path.join(FIG, "e3_improvement.png"), dpi=150)
        plt.close(fig)
        # illusion: believed vs truth for det and sortie0
        fig, ax = plt.subplots(figsize=(4.5, 4))
        for meth, mk, lab, col in [("det", "o", "det: ex-ante belief", "#c0392b"),
                                   ("sortie0", "^", "SORTIE: PBOI belief", "#2471a3")]:
            g = fair[(fair.method == meth) & (fair.measure == "range")]
            ax.scatter(g["oos_obj_X"], g["obj_insample"], s=12, marker=mk, label=lab, color=col, alpha=0.7)
        lim = [0, fair[fair.measure == "range"]["oos_obj_X"].max() * 1.05]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("true expected range (MC)")
        ax.set_ylabel("range the planner believes")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "illusion_range.pdf"))
        fig.savefig(os.path.join(FIG, "illusion_range.png"), dpi=150)
        plt.close(fig)
    df6 = load("e6_sensitivity")
    if df6 is not None:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        for meas, col in [("range", "#c0392b"), ("gmd", "#2471a3")]:
            g = df6[(df6.method == "det") & (df6.measure == meas)]
            s = g.groupby("v").apply(lambda d: 100 * ((d["oos_obj_X"] - d["exante"]) / d["oos_obj_X"]).mean())
            ax.plot(s.index, s.values, "o-", color=col, label=f"{MNAME[meas]}: unfairness hidden by det plan")
        ax.set_xlabel("arc coefficient of variation v")
        ax.set_ylabel("% of true ex-post unfairness")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "e6_illusion_v.pdf"))
        fig.savefig(os.path.join(FIG, "e6_illusion_v.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    for fn in (e1, e2, e3, e4, e5, e6, e7):
        try:
            fn()
        except Exception as exc:  # keep going; report at the end
            print(f"{fn.__name__} failed: {exc!r}", file=sys.stderr)
    figures()
    with open(os.path.join(TAB, "ALL_TABLES.md"), "w") as f:
        f.write("# SORTIE result tables (generated by experiments/analyze.py)\n" + "".join(out_md))
    print("".join(out_md))

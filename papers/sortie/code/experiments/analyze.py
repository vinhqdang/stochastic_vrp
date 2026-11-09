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
        tt = t.groupby("method")[["time_per_eval_s", "ratio_to_pboi"]].agg(["mean", "max"])
        tt.columns = ["_".join(c) for c in tt.columns]
        emit("E1 - evaluation time per plan (mean/max over plans)", tt.reset_index(), "e1_timing", "%.4g")


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


E7_METHODS = ["saa", "sortie0", "sortie1"]


def _convergence(df):
    """Per run: in-model best value at half the time limit vs at the limit (from plan histories)."""
    import json
    rows = []
    for _, r in df.iterrows():
        pf = r.get("plan_file")
        if not isinstance(pf, str) or not os.path.exists(pf):
            alt = os.path.join(RES, "plans", "e7", os.path.basename(str(pf)))
            if not os.path.exists(alt):
                continue
            pf = alt
        h = json.load(open(pf)).get("history") or []
        if not h:
            continue
        half = float(r["time_limit"]) / 2.0
        v_half = min([v for t, v in h if t <= half], default=h[0][1])
        v_end = min(v for _, v in h)
        rows.append(dict(n=r["n"], method=r["method"], measure=r["measure"],
                         gain_pct=100 * (v_half - v_end) / abs(v_half) if v_half else 0.0))
    return pd.DataFrame(rows)


def e7():
    df = load("e7_scale")
    if df is None:
        return
    fair = df[df.method != "cost"]
    others = [o for o in E7_METHODS if o in set(fair.method)]
    t = paired_table(fair, "det", others, "oos_obj", ["n", "measure"])
    emit("E7 - scalability (n = 50-100) by size and measure", t, "e7_scale")
    t = paired_table(fair.assign(inst_meas=fair.instance + "|" + fair.measure).drop(columns="instance")
                     .rename(columns={"inst_meas": "instance"}), "det", others, "oos_obj", ["n"])
    emit("E7 - scalability pooled over measures", t, "e7_pooled")
    cv = _convergence(fair)
    if len(cv):
        g = cv.groupby(["n", "method"])["gain_pct"].agg(
            mean="mean", max="max", share_gt_half_pct=lambda x: (x > 0.5).mean()).reset_index()
        emit("E7 - convergence: in-model improvement in the second half of the run (%)", g,
             "e7_convergence", "%.3f")


def e8():
    df = load("e8_congestion")
    if df is None:
        return
    fair = df[df.method != "cost"]
    t = paired_table(fair, "det", ["saa", "sortie0", "sortie1"], "oos_obj", ["measure"])
    emit("E8 - heterogeneous (congested) arc noise: improvement over det (%)", t, "e8_congestion")
    t2 = paired_table(fair, "det", ["saa", "sortie0", "sortie1"], "oos_obj", ["n"])
    emit("E8 - improvement over det by size (measures pooled)", t2, "e8_by_n")


def e7b():
    df = load("e7b_seeds")
    if df is None or len(df) == 0:
        return
    f = df[df.method != "cost"]
    # search noise: coefficient of variation over seeds of the same (instance, measure, method)
    sd = f.groupby(["instance", "measure", "method"])["oos_obj"].agg(lambda x: 100 * x.std() / x.mean())
    emit("E7b - search noise: CV over three seeds (%)",
         sd.groupby(level=["measure", "method"]).mean().unstack().reset_index(), "e7b_noise")
    mean = f.groupby(["instance", "measure", "method"])["oos_obj"].mean().reset_index()
    mean["instance"] = mean["instance"] + "|" + mean["measure"]
    others = ["saa", "sortie0"]
    t = paired_table(mean, "det", others, "oos_obj", ["measure"])
    emit("E7b - n = 100, mean over 3 seeds: improvement over det by measure", t, "e7b_by_measure")
    mean["all"] = "all"
    t = paired_table(mean, "det", others, "oos_obj", ["all"])
    emit("E7b - n = 100, mean over 3 seeds: pooled", t, "e7b_pooled")
    t = paired_table(mean, "saa", ["sortie0"], "oos_obj", ["all"])
    emit("E7b - n = 100, mean over 3 seeds: SORTIE-0 vs SAA", t, "e7b_vs_saa")


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


# ---------------------------------------------------------------- paper tables
PT = os.path.join(TAB, "paper")
os.makedirs(PT, exist_ok=True)
MLAB = {"det": "Det", "saa": "SAA", "sortie0": "SORTIE-0", "sortie1": "SORTIE-1"}


def _rank_direction(diff):
    """Signed-rank sum of the paired differences (zeros dropped, as in the Wilcoxon test)."""
    from scipy.stats import rankdata
    d = np.asarray(diff, float)
    d = d[np.abs(d) > 1e-9]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float(r[d > 0].sum() - r[d < 0].sum())


def _cell(imp, wins, n, p, med=0.0):
    """Mean improvement %; star: Wilcoxon p<0.05 with the signed-rank sum
    favouring the method, dagger: p<0.05 favouring the baseline."""
    star = ""
    if p == p and p < 0.05:
        star = "$^\\ast$" if med > 0 else "$^\\dagger$"
    return f"{imp:+.2f}{star} ({wins}/{n})"


def _improve_rows(df, group, others, base="det"):
    rows = []
    for keys, g in df.groupby(group):
        piv = g.pivot_table(index="instance", columns="method", values="oos_obj")
        keys = keys if isinstance(keys, tuple) else (keys,)
        cells = []
        for o in others:
            ok = piv[[base, o]].dropna()
            imp = (100 * (ok[base] - ok[o]) / ok[base].abs()).mean()
            cells.append(_cell(imp, int((ok[o] < ok[base] - 1e-9).sum()), len(ok), wpval(ok[base], ok[o]),
                               _rank_direction(ok[base] - ok[o])))
        rows.append((keys, cells))
    return rows


def _write(name, header, body):
    with open(os.path.join(PT, name + ".tex"), "w") as f:
        f.write(header + "\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n")


def paper_tables():
    df = load("e1_accuracy")
    if df is not None:
        body = ""
        for meas in MEAS:
            g = df[df.measure == meas]
            body += f"{MNAME[meas]} & {len(g)} & {100*g.rel_err.abs().mean():.3f} & {100*g.rel_err.abs().max():.3f} & {(g.z_score.abs()<3).mean():.3f} \\\\\n"
        _write("tab_e1", "\\begin{tabular}{lrrrr}\\toprule\nMeasure & Plans & Mean $|$err$|$ (\\%) & Max $|$err$|$ (\\%) & Share $|z|<3$ \\\\\n", body)
    df = load("e3_main")
    if df is not None:
        fair = df[df.method != "cost"]
        others = ["saa", "sortie0", "sortie1"]
        body = ""
        for (meas, B), cells in _improve_rows(fair, ["measure", "B"], others):
            body += f"{MNAME[meas]} & {B:.2f} & " + " & ".join(cells) + " \\\\\n"
        _write("tab_e3", "\\begin{tabular}{llrrr}\\toprule\nMeasure & $B$ & SAA & SORTIE-0 & SORTIE-1 \\\\\n", body)
        body = ""
        for meas in MEAS:
            cells = []
            for meth in ["det", "saa", "sortie0", "sortie1"]:
                g = fair[(fair.method == meth) & (fair.measure == meas)]
                cells.append(f"{100*((g.obj_insample-g.oos_obj_X)/g.oos_obj_X).mean():+.2f}")
            g = fair[fair.measure == meas]
            cells.append(f"{100*((g.pboi_pred-g.oos_obj_X)/g.oos_obj_X).abs().max():.2f}")
            body += f"{MNAME[meas]} & " + " & ".join(cells) + " \\\\\n"
        _write("tab_illusion", "\\begin{tabular}{lrrrrr}\\toprule\n & \\multicolumn{4}{c}{Believed vs true (mean signed \\%)} & PBOI \\\\\nMeasure & Det & SAA & SORTIE-0 & SORTIE-1 & max $|$err$|$ \\% \\\\\n", body)
    df = load("e4_inconsistency")
    if df is not None:
        body = ""
        for meas in ["max", "range", "gmd"]:
            for meth in ["det_lower", "det_free", "pboi_lower", "pboi_free"]:
                g = df[(df.method == meth) & (df.measure == meas)]
                body += (f"{MNAME[meas]} & {meth.replace('pboi', 'SORTIE').replace('det', 'Det').replace('_', '-')} & {int((g.nontsp_routes>0).sum())}/{len(g)} & "
                         f"{g.nontsp_routes.mean():.2f} & {g.excess_len.mean():.1f} & "
                         f"{100*((g.obj_insample-g.oos_obj_X)/g.oos_obj_X).mean():+.2f} & {g.oos_obj.mean():.1f} \\\\\n")
        _write("tab_e4", "\\begin{tabular}{llrrrrr}\\toprule\nMeasure & Method & Runs$^a$ & Routes$^b$ & Excess & Belief err (\\%) & Ex-post \\\\\n", body)
    df = load("e6_sensitivity")
    if df is not None:
        body = ""
        for (v, meas), cells in _improve_rows(df, ["v", "measure"], ["sortie0"]):
            g = df[(df.method == "det") & (df.v == v) & (df.measure == meas)]
            ill = 100 * ((g.exante - g.oos_obj_X) / g.oos_obj_X).mean()
            body += f"{v:.1f} & {MNAME[meas]} & {ill:+.1f} & {cells[0]} \\\\\n"
        _write("tab_e6", "\\begin{tabular}{llrr}\\toprule\n$v$ & Measure & Det belief err (\\%) & SORTIE-0 vs Det \\\\\n", body)
    for name, csvn, others in [("tab_e7", "e7_scale", ["saa", "sortie0", "sortie1"]), ("tab_e8", "e8_congestion", ["saa", "sortie0", "sortie1"])]:
        df = load(csvn)
        if df is None or len(df) == 0:
            continue
        fair = df[df.method != "cost"]
        grp = ["n", "measure"] if name == "tab_e7" else ["measure"]
        body = ""
        for keys, cells in _improve_rows(fair, grp, others):
            lab = " & ".join(str(k) if not isinstance(k, str) else MNAME[k] for k in keys)
            body += f"{lab} & " + " & ".join(cells) + " \\\\\n"
        cols = "l" * len(grp) + "r" * len(others)
        head = " & ".join("$n$" if c == "n" else "Measure" for c in grp) + " & " + " & ".join(MLAB[o] for o in others)
        _write(name, f"\\begin{{tabular}}{{{cols}}}\\toprule\n{head} \\\\\n", body)
        if name == "tab_e7":
            pooled = fair.assign(instance=fair.instance + "|" + fair.measure)
            body = ""
            for keys, cells in _improve_rows(pooled, ["n"], others):
                body += f"{keys[0]} & " + " & ".join(cells) + " \\\\\n"
            _write("tab_e7_pooled", "\\begin{tabular}{lrrr}\\toprule\n$n$ & "
                   + " & ".join(MLAB[o] for o in others) + " \\\\\n", body)


if __name__ == "__main__":
    for fn in (e1, e2, e3, e4, e5, e6, e7, e7b, e8):
        try:
            fn()
        except Exception as exc:  # keep going; report at the end
            print(f"{fn.__name__} failed: {exc!r}", file=sys.stderr)
    figures()
    paper_tables()
    with open(os.path.join(TAB, "ALL_TABLES.md"), "w") as f:
        f.write("# SORTIE result tables (generated by experiments/analyze.py)\n" + "".join(out_md))
    print("".join(out_md))

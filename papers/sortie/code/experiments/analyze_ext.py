"""Tables and figures for the extension experiments E9-E17 of the SORTIE report.

Run from papers/sortie/code:  python experiments/analyze_ext.py

Inputs (a missing CSV is skipped with a message):
  <results>/e9_budget.csv, e10_fleet.csv, e10_seeds.csv, e11_dayfactor.csv,
  e12_misspec.csv, e13_kappa.csv, e14_moves.csv, e14_ksweep.csv, e15_saa.csv,
  e16_baselines.csv, e17_family.csv, e17_seeds.csv.
  Joins with E3 (cost plans, det and SORTIE-0 at B = 1.10) read the complete
  e3_main.csv from SORTIE_SOURCE_RESULTS (default papers/sortie/results).

Environment:
  SORTIE_RESULTS          results directory (default papers/sortie/results)
  SORTIE_SOURCE_RESULTS   directory holding the complete e3_main.csv
  SORTIE_FIGURES          figure directory (default papers/sortie/figures)

Outputs:
  <results>/tables/ext/*.tex and <results>/tables/ext/ALL_EXT.md (full tables)
  <results>/tables/paper/tab_*.tex (paper-ready booktabs tabulars)
  <figures>/fig_*.pdf and fig_*.png (200 dpi)

Improvement of method m over the base method = 100 * (base - m) / base of oos_obj,
paired by (instance, measure) (plus other keys where stated). Wilcoxon p-values and
the stars come from experiments/analyze.py (wpval, _rank_direction, _cell).
"""

import os
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, ".."))
PAPER = os.path.normpath(os.path.join(CODE, ".."))
for _p in (HERE, CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analyze as an  # noqa: E402  (reuses helpers; analyze's main runs only as __main__)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sortie.instances import DATA_PATH, load_vrp  # noqa: E402

RES = os.path.abspath(an.RES)
SRC = os.path.abspath(os.environ.get("SORTIE_SOURCE_RESULTS") or os.path.join(PAPER, "results"))
FIG = os.path.abspath(os.environ.get("SORTIE_FIGURES") or os.path.join(PAPER, "figures"))
XDIR = os.path.join(os.path.dirname(DATA_PATH), "X")
TAB_EXT = os.path.join(RES, "tables", "ext")
TAB_PAPER = os.path.join(RES, "tables", "paper")
for _d in (TAB_EXT, TAB_PAPER):
    os.makedirs(_d, exist_ok=True)

MEAS5 = ["max", "range", "gmd", "mad", "var"]
MEAS3 = ["range", "gmd", "max"]
MNAME = an.MNAME
LAB = dict(an.MLAB, detsig="Det-σ", pboi_normal="PBOI-N", cost="Cost")
LAB_TEX = dict(LAB, detsig="Det-$\\sigma$")
COL = {"det": "#c0392b", "saa": "#7f8c8d", "sortie0": "#2471a3", "sortie1": "#17a589",
       "cost": "#8e44ad", "detsig": "#d68910", "pboi_normal": "#6c3483"}
MCOL = {"max": "#0072b2", "range": "#d55e00", "gmd": "#009e73", "mad": "#cc79a7", "var": "#e69f00"}
DEPOT_COL = {"central": "#2471a3", "intermediate": "#7f8c8d", "eccentric": "#d68910",
             "unknown": "#cccccc"}
FULL_W, HALF_W = 6.5, 3.2
PAIR = ("instance", "measure")
SRC_LAB = {"e0": "Cost plan", "e3_sortie0": "SORTIE-0 plan"}
INT_COLS = {"n", "m", "K", "wins", "plans", "groups", "moves", "eligible", "N", "rank"}

OUT_MD = []
_E3 = {}


# ----------------------------------------------------------------------------
# I/O and guards
# ----------------------------------------------------------------------------

def load(name):
    """Rows of <results>/<name>.csv, or None (with a message) when missing or empty."""
    path = os.path.join(RES, f"{name}.csv")
    if not os.path.exists(path):
        print(f"[skip] {name}.csv not found in {RES}")
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        print(f"[skip] {name}.csv has no rows")
        return None
    return df


def load_e3():
    """The complete E3 table (joins only); None when no copy is available."""
    if "e3" not in _E3:
        path = os.path.join(SRC, "e3_main.csv")
        if not os.path.exists(path):
            path = os.path.join(RES, "e3_main.csv")
        _E3["e3"] = pd.read_csv(path) if os.path.exists(path) else None
        if _E3["e3"] is None:
            print("[skip] e3_main.csv not found: E3 joins skipped")
    return _E3["e3"]


def e3_b110(method):
    e3 = load_e3()
    if e3 is None:
        return None
    b = pd.to_numeric(e3["B"], errors="coerce")
    return e3[(e3["method"] == method) & np.isclose(b, 1.10)]


def cost_ref():
    """(instance, measure) -> value of the cost plan (E3 method 'cost', E_<measure>)."""
    e3 = load_e3()
    if e3 is None:
        return None
    c = e3[e3["method"] == "cost"]
    parts = [pd.DataFrame({"instance": c["instance"].values, "measure": m,
                           "cost_val": pd.to_numeric(c[f"E_{m}"]).values})
             for m in MEAS5 if f"E_{m}" in c.columns]
    if not parts:
        return None
    return pd.concat(parts).drop_duplicates(["instance", "measure"])


def guard(label, fn, *args, **kw):
    """Run one table or figure step; a failure is reported and the run goes on."""
    try:
        return fn(*args, **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] failed: {exc!r}", file=sys.stderr)
        traceback.print_exc()
        return None


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------

def _stats(df, base, others, metric="oos_obj", pair=PAIR):
    """Improvement of each method in `others` over `base`, paired on the `pair` columns."""
    if df is None or not len(df) or not {"method", metric, *pair}.issubset(df.columns):
        return {}
    d = df.dropna(subset=[metric])
    key = d[list(pair)].astype(str).agg("|".join, axis=1)
    piv = d.assign(_pk=key).pivot_table(index="_pk", columns="method", values=metric,
                                        aggfunc="mean")
    if base not in piv.columns:
        return {}
    out = {}
    for o in others:
        if o not in piv.columns:
            continue
        ok = piv[[base, o]].dropna()
        if not len(ok):
            continue
        a, b = ok[base], ok[o]
        imp = 100 * (a - b) / a.abs()
        out[o] = dict(impr=float(imp.mean()),
                      sd=float(imp.std(ddof=1)) if len(imp) > 1 else np.nan,
                      wins=int((b < a - 1e-9).sum()), n=int(len(ok)),
                      p=an.wpval(a, b), rank=an._rank_direction((a - b).values))
    return out


def cell(s):
    """'mean% (wins/n)' with the significance star of analyze._cell."""
    if not s:
        return "--"
    return an._cell(s["impr"], s["wins"], s["n"], s["p"], s["rank"])


def group_stats(df, group, base, others, metric="oos_obj"):
    """[(key tuple, stats dict)] per group; one pooled group when group is empty."""
    if df is None or not len(df):
        return []
    if not group:
        df, group = df.assign(scope="pooled"), ["scope"]
    out = []
    for keys, g in df.groupby(list(group)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        out.append((keys, _stats(g, base, others, metric)))
    return out


def impr_long(df, group, base, others):
    """Long numeric table of improvements (full tables): one row per group and comparison.

    A group without a measure key is pooled over measures and labelled 'Pooled'."""
    rows = []
    gcols = list(group)
    for keys, st in group_stats(df, group, base, others):
        rec = {}
        if "measure" not in gcols:
            rec["measure"] = "Pooled"
        for c, k in zip(gcols, keys):
            rec[c] = MNAME.get(k, k) if c == "measure" else k
        for o in others:
            s = st.get(o)
            rows.append(dict(rec, comparison=f"{LAB[o]} vs {LAB[base]}",
                             impr_pct=s["impr"] if s else np.nan,
                             wins=s["wins"] if s else np.nan,
                             n=s["n"] if s else np.nan,
                             p=s["p"] if s else np.nan))
    return pd.DataFrame(rows)


def _truth(df):
    """True value of each row's own measure without the day factor (E_<measure>_X)."""
    t = pd.Series(np.nan, index=df.index)
    for m in MEAS5:
        c = f"E_{m}_X"
        if c in df.columns:
            sel = df["measure"] == m
            t[sel] = df.loc[sel, c]
    return t


def _fmt(x, spec):
    """Number with a format spec, or '--' when missing."""
    return "--" if x is None or x != x else format(x, spec)


def _iqr(x):
    return pd.Series(x).quantile(0.25), pd.Series(x).quantile(0.75)


# ----------------------------------------------------------------------------
# output helpers
# ----------------------------------------------------------------------------

def emit(name, title, df, fmt="%.3f"):
    """Full table: <results>/tables/ext/<name>.tex and a section of ALL_EXT.md."""
    if df is None or not len(df):
        print(f"[skip] {name}: nothing to tabulate")
        return
    d = df.copy()
    for c in d.columns:
        if (c in INT_COLS or c.endswith(" wins") or c.endswith(" n")) and pd.api.types.is_float_dtype(d[c]):
            if d[c].dropna().apply(float.is_integer).all():
                d[c] = pd.array(d[c].round(), dtype="Int64")
    tex = d.rename(columns=lambda c: c.replace("σ", "sigma"))
    path = os.path.join(TAB_EXT, name + ".tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(tex.to_latex(index=False, float_format=lambda x: fmt % x, na_rep="--",
                              escape=True))
    OUT_MD.append(f"\n### {title}\n\n" + d.to_markdown(index=False, floatfmt=".3g") + "\n")
    print(f"[table] {path}")


def write_paper(name, colspec, head, rows):
    """Paper-ready booktabs tabular in <results>/tables/paper/<name>.tex."""
    if not rows:
        print(f"[skip] {name}: no rows")
        return
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", head + " \\\\", "\\midrule"]
    lines += [r + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path = os.path.join(TAB_PAPER, name + ".tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[table] {path}")


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "savefig.dpi": 200,
        "figure.dpi": 100,
    })


def save(fig, stem):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"), dpi=200, bbox_inches="tight",
                    pad_inches=0.03)
    plt.close(fig)
    print(f"[figure] {os.path.join(FIG, stem)}.pdf/.png")


# ----------------------------------------------------------------------------
# E9 budget frontier
# ----------------------------------------------------------------------------

def e9():
    df = load("e9_budget")
    if df is None:
        return
    ref = cost_ref()
    if ref is None:
        print("[e9] no E3 cost plans: normalised values are empty")
        d = df.assign(cost_val=np.nan)
    else:
        d = df.merge(ref, on=["instance", "measure"], how="left")
    d = d.assign(value_ratio=d["oos_obj"] / d["cost_val"])
    fr = (d.groupby(["measure", "B", "method"], as_index=False)
          .agg(cost_ratio=("cost_ratio", "mean"), value_ratio=("value_ratio", "mean"),
               n=("value_ratio", "count")))
    guard("tab e9", tables_e9, d, fr)
    guard("fig e9", fig_e9, fr)


def tables_e9(d, fr):
    emit("e9_frontier", "E9 budget frontier: cost ratio and ex-post value / cost plan (means)", fr)
    ext = pd.concat([impr_long(d, ["B"], "det", ["sortie0"]),
                     impr_long(d, ["B", "measure"], "det", ["sortie0"])], ignore_index=True)
    emit("e9_improve", "E9 improvement of SORTIE-0 over Det by budget B (%, wins/n, Wilcoxon p)", ext)
    rows = []
    for B in sorted(d["B"].unique()):
        sb = d[d["B"] == B]
        cr = sb[sb["method"] == "sortie0"]["cost_ratio"].mean()
        cells = [cell(_stats(sb, "det", ["sortie0"]).get("sortie0"))]
        cells += [cell(_stats(sb[sb["measure"] == m], "det", ["sortie0"]).get("sortie0"))
                  for m in MEAS3]
        rows.append(f"{B:.2f} & {cr:.3f} & " + " & ".join(cells))
    write_paper("tab_e9", "lrcccc", "$B$ & Cost ratio & Pooled & Range & GMD & Max", rows)


def fig_e9(fr):
    if fr["value_ratio"].isna().all():
        print("[fig e9] no normalised values: figure skipped")
        return
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.5), sharey=True)
    for ax, meas in zip(axes, MEAS3):
        g = fr[fr["measure"] == meas]
        for meth, mk in (("det", "o"), ("sortie0", "^")):
            h = g[g["method"] == meth].sort_values("B")
            if h.empty:
                continue
            ax.plot(h["cost_ratio"], h["value_ratio"], marker=mk, ms=4, lw=1.0,
                    color=COL[meth], label=LAB[meth])
            off = (0, 5) if meth == "det" else (0, -9)
            for _, r in h.iterrows():
                ax.annotate(f"{r['B']:.2f}", (r["cost_ratio"], r["value_ratio"]),
                            xytext=off, textcoords="offset points", fontsize=5.5,
                            color=COL[meth], ha="center")
        ax.plot([1], [1], marker="s", ms=5, ls="none", color=COL["cost"], label="Cost")
        ax.axhline(1, color="grey", lw=0.6, ls=":")
        ax.set_xlabel(r"cost ratio $\sum L / C^\ast$")
        ax.text(0.04, 0.04, MNAME[meas], transform=ax.transAxes, va="bottom", fontsize=8)
    axes[0].set_ylabel("ex-post value / cost plan value")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_e9_frontier")


# ----------------------------------------------------------------------------
# E10 fleet size
# ----------------------------------------------------------------------------

def e10():
    df = load("e10_fleet")
    if df is None:
        return
    d = df[df["method"].isin(["det", "saa", "sortie0"])].copy()
    d["illusion"] = 100 * (d["exante"] - d["oos_obj_X"]) / d["oos_obj_X"]
    guard("tab e10", tables_e10, d)
    guard("fig e10", fig_e10, d)


def _ill(d, m=None, meas=None):
    s = d[(d["method"] == "det")]
    if m is not None:
        s = s[s["m"] == m]
    if meas is not None:
        s = s[s["measure"] == meas]
    return float(s["illusion"].mean()) if len(s) else np.nan


def tables_e10(d):
    ill = (d[d["method"] == "det"].groupby(["m", "measure"], as_index=False)["illusion"].mean()
           .assign(measure=lambda x: x["measure"].map(MNAME)))
    emit("e10_illusion", "E10 Det belief error 100(ex-ante - truth)/truth (%) by fleet size", ill, "%.2f")
    ext = pd.concat([impr_long(d, ["m"], "det", ["saa", "sortie0"]),
                     impr_long(d, ["m", "measure"], "det", ["saa", "sortie0"])], ignore_index=True)
    emit("e10_improve", "E10 improvement over Det by fleet size m (%, wins/n, p)", ext)
    rows = []
    for m in sorted(d["m"].unique()):
        for scope in ["pooled"] + MEAS3:
            sub = d[d["m"] == m] if scope == "pooled" else d[(d["m"] == m) & (d["measure"] == scope)]
            st = _stats(sub, "det", ["saa", "sortie0"])
            label = "Pooled" if scope == "pooled" else MNAME[scope]
            rows.append(f"{int(m)} & {label} & {_fmt(_ill(sub), '+.1f')} & "
                        f"{cell(st.get('saa'))} & {cell(st.get('sortie0'))}")
    write_paper("tab_e10", "llrcc", "$m$ & Measure & Det belief err. (\\%) & SAA vs Det & SORTIE-0 vs Det",
                rows)


def fig_e10(d):
    _style()
    ms = sorted(int(x) for x in d["m"].unique())
    x = np.arange(len(ms))
    w = 0.26
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.5))
    for i, meas in enumerate(MEAS3):
        vals = [_ill(d, m=m, meas=meas) for m in ms]
        a.bar(x + (i - 1) * w, vals, w, color=MCOL[meas], label=MNAME[meas])
    a.axhline(0, color="grey", lw=0.6)
    a.set_xticks(x)
    a.set_xticklabels([str(m) for m in ms])
    a.set_xlabel("vehicles $m$")
    a.set_ylabel("Det belief error (%)")
    lo, hi = a.get_ylim()
    a.set_ylim(lo, hi + 0.45 * (hi - lo))
    a.legend(frameon=False, loc="upper right", ncol=3, fontsize=6.5)
    for j, meth in enumerate(["saa", "sortie0"]):
        vals, errs = [], []
        for m in ms:
            s = _stats(d[d["m"] == m], "det", [meth]).get(meth)
            vals.append(s["impr"] if s else np.nan)
            errs.append(s["sd"] / np.sqrt(s["n"]) if s and s["n"] > 1 else 0.0)
        b.bar(x + (j - 0.5) * w * 1.1, vals, w * 1.1, yerr=errs, capsize=2, color=COL[meth],
              label=LAB[meth], error_kw=dict(lw=0.6))
    b.axhline(0, color="grey", lw=0.6)
    b.set_xticks(x)
    b.set_xticklabels([str(m) for m in ms])
    b.set_xlabel("vehicles $m$")
    b.set_ylabel("ex-post improvement over Det (%)")
    b.legend(frameon=False, loc="best")
    fig.tight_layout()
    save(fig, "fig_e10")


# ----------------------------------------------------------------------------
# E11 day factor
# ----------------------------------------------------------------------------

def e11():
    df = load("e11_dayfactor")
    if df is None:
        return
    recs = []
    for m in MEAS5:
        if f"E_{m}" in df.columns and f"E_{m}_X" in df.columns:
            recs.append(pd.DataFrame({"sY": df["sY"], "quantity": MNAME[m],
                                      "ratio": df[f"E_{m}"] / df[f"E_{m}_X"]}))
    if {"CVaR90_range", "CVaR90_range_X"}.issubset(df.columns):
        recs.append(pd.DataFrame({"sY": df["sY"], "quantity": "CVaR90 range",
                                  "ratio": df["CVaR90_range"] / df["CVaR90_range_X"]}))
    if not recs:
        print("[skip] e11: no E_* columns")
        return
    long = pd.concat(recs, ignore_index=True)
    tab = (long.groupby(["sY", "quantity"])["ratio"].agg(["mean", "std", "count"])
           .reset_index().rename(columns={"mean": "ratio_mean", "std": "ratio_sd", "count": "n"}))
    emit("e11_ratio", "E11 day factor: E_meas(s_Y) / E_meas_X over all E3 plans (B = 1.10)", tab)
    guard("tab e11", tables_e11, tab)
    guard("fig e11", fig_e11, tab)


def tables_e11(tab):
    order = [MNAME[m] for m in MEAS5] + ["CVaR90 range"]
    rows = []
    for sY in sorted(tab["sY"].unique()):
        cells = []
        for q in order:
            s = tab[(tab["sY"] == sY) & (tab["quantity"] == q)]
            cells.append(f"{s['ratio_mean'].iloc[0]:.3f} $\\pm$ {s['ratio_sd'].iloc[0]:.3f}"
                         if len(s) else "--")
        rows.append(f"{sY:.2f} & " + " & ".join(cells) + f" & {np.exp(sY ** 2):.3f}")
    head = "$s_Y$ & " + " & ".join(order) + " & $e^{s_Y^2}$"
    write_paper("tab_e11", "l" + "c" * (len(order) + 1), head, rows)


def fig_e11(tab):
    _style()
    qs = [MNAME[m] for m in MEAS5] + ["CVaR90 range"]
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 4.0), sharex=True)
    for ax, q in zip(axes.flat, qs):
        s = tab[tab["quantity"] == q].sort_values("sY")
        if s.empty:
            ax.set_visible(False)
            continue
        ax.errorbar(s["sY"], s["ratio_mean"], yerr=s["ratio_sd"].fillna(0), fmt="o-", ms=3,
                    lw=1, capsize=2, color="#2471a3")
        xs = np.linspace(0, max(s["sY"].max(), 0.01), 50)
        if q == "Var":
            ax.plot(xs, np.exp(xs ** 2), "k--", lw=0.8, label=r"$e^{s_Y^2}$")
            ax.legend(frameon=False, loc="upper left")
        else:
            ax.axhline(1, color="grey", lw=0.8, ls="--")
        ax.text(0.97, 0.05, q, transform=ax.transAxes, ha="right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(r"day-factor scale $s_Y$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$E(s_Y)\,/\,E_X$")
    fig.tight_layout()
    save(fig, "fig_e11")


# ----------------------------------------------------------------------------
# E12 misspecification
# ----------------------------------------------------------------------------

FAM_LAB = {"gamma": "Gamma", "lognormal": "Lognormal", "incident": "Incident", "regional": "Regional"}
SRC12 = {"e3": "E3", "e8": "E8"}


def e12():
    df = load("e12_misspec")
    if df is None:
        return
    # The arc-noise family is the `dist` column; `family` is the benchmark family (X641).
    d = df.assign(truth=_truth(df), family=df["dist"])
    d["pboi_err"] = 100 * (d["pboi_pred"] - d["truth"]) / d["truth"]
    d["src"] = d["source"].map(SRC12).fillna(d["source"].astype(str))
    summ = (d.groupby(["src", "family"], as_index=False)
            .agg(plans=("pboi_err", "size"),
                 abs_err_pct=("pboi_err", lambda x: x.abs().mean()),
                 signed_err_pct=("pboi_err", "mean")))
    imp = impr_long(d[d["method"].isin(["det", "sortie0"])], ["src", "family"], "det", ["sortie0"])
    imp = imp.drop(columns=["measure", "comparison"])
    full = summ.merge(imp, on=["src", "family"], how="left")
    guard("tab e12", tables_e12, d, summ)
    guard("fig e12", fig_e12, d, summ)
    emit("e12_misspec", "E12 misspecification: PBOI error and SORTIE-0 vs Det by family", full)


def tables_e12(d, summ):
    rows = []
    for _, r in summ.iterrows():
        sub = d[(d["src"] == r["src"]) & (d["family"] == r["family"])]
        st = _stats(sub, "det", ["sortie0"]).get("sortie0")
        rows.append(f"{FAM_LAB.get(r['family'], r['family'])} & {r['src']} & {int(r['plans'])} & "
                    f"{r['abs_err_pct']:.2f} & {r['signed_err_pct']:+.2f} & {cell(st)}")
    write_paper("tab_e12", "llrrrc",
                "Family & Source & Plans & PBOI $|$err$|$ (\\%) & PBOI signed err. (\\%) & SORTIE-0 vs Det",
                rows)


def fig_e12(d, summ):
    _style()
    fams = [f for f in FAM_LAB if f in set(summ["family"])]
    if not fams:
        print("[fig e12] no families: figure skipped")
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.6))
    x = np.arange(len(fams))
    w = 0.38
    src_col = {"E3": "#2471a3", "E8": "#d68910"}
    for i, src in enumerate(["E3", "E8"]):
        vals = [summ[(summ.src == src) & (summ.family == f)]["abs_err_pct"] for f in fams]
        vals = [v.iloc[0] if len(v) else np.nan for v in vals]
        a.bar(x + (i - 0.5) * w, vals, w, color=src_col[src], label=f"{src} plans")
        imps, errs = [], []
        for f in fams:
            s = _stats(d[(d.src == src) & (d.family == f)], "det", ["sortie0"]).get("sortie0")
            imps.append(s["impr"] if s else np.nan)
            errs.append(s["sd"] / np.sqrt(s["n"]) if s and s["n"] > 1 else 0.0)
        b.bar(x + (i - 0.5) * w, imps, w, yerr=errs, capsize=2, color=src_col[src],
              label=f"{src} plans", error_kw=dict(lw=0.6))
    a.set_ylabel("PBOI mean $|$error$|$ (%)")
    a.set_ylim(0, np.nanmax(summ["abs_err_pct"].values) * 1.4)
    b.set_ylabel("SORTIE-0 improvement over Det (%)")
    b.axhline(0, color="grey", lw=0.6)
    for ax in (a, b):
        ax.set_xticks(x)
        ax.set_xticklabels([FAM_LAB[f] for f in fams])
        ax.legend(frameon=False, loc="upper right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_e12")


# ----------------------------------------------------------------------------
# E13 risk attitude
# ----------------------------------------------------------------------------

def e13():
    df = load("e13_kappa")
    if df is None:
        return
    base = df[df["kappa"] == 0][["instance", "measure", "oos_obj"]].rename(columns={"oos_obj": "oos0"})
    d = df.merge(base, on=["instance", "measure"], how="left")
    d = d.assign(share_pct=100 * d["n_routes_nontsp"] / d["m"], oos_rel=d["oos_obj"] / d["oos0"])
    tab = (d.groupby(["measure", "kappa"], as_index=False)
           .agg(n=("oos_rel", "count"), routes_nontsp=("n_routes_nontsp", "mean"),
                share_pct=("share_pct", "mean"), extra_len=("extra_len_vs_tsp", "mean"),
                mean_risk=("mean_risk", "mean"), oos_rel_median=("oos_rel", "median"),
                oos_rel_q25=("oos_rel", lambda x: x.quantile(0.25)),
                oos_rel_q75=("oos_rel", lambda x: x.quantile(0.75)),
                cost_ratio=("cost_ratio", "mean")))
    emit("e13_kappa", "E13 risk-consistent tours: structure and ex-post value vs kappa", tab)
    guard("tab e13", tables_e13, tab)
    guard("fig e13", fig_e13, tab)


def tables_e13(tab):
    rows = []
    for meas in ["range", "max"]:
        for k in sorted(tab["kappa"].unique()):
            s = tab[(tab["measure"] == meas) & (tab["kappa"] == k)]
            if s.empty:
                continue
            s = s.iloc[0]
            rows.append(f"{MNAME[meas]} & {k:g} & {s.share_pct:.1f} & {s.extra_len:.1f} & "
                        f"{s.mean_risk:.1f} & {s.oos_rel_median:.3f} [{s.oos_rel_q25:.3f}, "
                        f"{s.oos_rel_q75:.3f}] & {s.cost_ratio:.3f}")
    write_paper("tab_e13", "llrrrrr",
                "Measure & $\\kappa$ & Routes changed (\\%) & Extra length & Mean risk & "
                "Ex-post vs $\\kappa{=}0$ [IQR] & Cost ratio", rows)


def fig_e13(tab):
    _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.6))
    at = a.twinx()
    for meas in ["range", "max"]:
        s = tab[tab["measure"] == meas].sort_values("kappa")
        if s.empty:
            continue
        a.plot(s["kappa"], s["share_pct"], "o-", ms=3, lw=1, color=MCOL[meas],
               label=f"{MNAME[meas]}: routes changed")
        at.plot(s["kappa"], s["extra_len"], "s--", ms=3, lw=1, color=MCOL[meas], alpha=0.7,
                label=f"{MNAME[meas]}: extra length")
    a.set_xlabel(r"risk weight $\kappa$")
    a.set_ylabel("routes changed vs TSP (%)")
    at.set_ylabel("extra length vs TSP tours")
    at.spines["right"].set_visible(True)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = at.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="upper center",
             bbox_to_anchor=(0.5, -0.2), ncol=2)
    for meas in ["range", "max"]:
        s = tab[tab["measure"] == meas].sort_values("kappa")
        if s.empty:
            continue
        lo = s["oos_rel_median"] - s["oos_rel_q25"]
        hi = s["oos_rel_q75"] - s["oos_rel_median"]
        b.errorbar(s["kappa"], s["oos_rel_median"], yerr=[lo, hi], fmt="o-", ms=3, lw=1,
                   capsize=2, color=MCOL[meas], label=MNAME[meas])
    b.axhline(1, color="grey", lw=0.8, ls="--")
    b.set_xlabel(r"risk weight $\kappa$")
    b.set_ylabel(r"ex-post value / value at $\kappa=0$")
    b.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_e13")


# ----------------------------------------------------------------------------
# E14 screening diagnostics
# ----------------------------------------------------------------------------

KS_HIT = (1, 4, 8, 16, 32)


def _move_groups(mv):
    """One record per (plan source, instance, measure): the rank of the exact-best improving move."""
    recs = []
    for (src, inst, meas), g in mv.groupby(["plan_source", "instance", "measure"]):
        N = len(g)
        imp = g[g["exact"] < -1e-9]
        if imp.empty:
            recs.append(dict(plan_source=src, instance=inst, measure=meas, N=N,
                             eligible=False, rank=np.nan))
            continue
        best = imp.loc[imp["exact"].idxmin()]
        recs.append(dict(plan_source=src, instance=inst, measure=meas, N=N, eligible=True,
                         rank=int(best["pred_rank"])))
    return pd.DataFrame(recs)


def e14():
    mv = load("e14_moves")
    if mv is not None:
        guard("e14 moves", _e14_moves, mv)
    ks = load("e14_ksweep")
    if ks is not None:
        guard("e14 ksweep", _e14_ksweep, ks)


def _e14_moves(mv):
    mv = mv.copy()
    mv["plan_source"] = mv["plan_source"].astype(str)
    rec = _move_groups(mv)
    measures = [m for m in MEAS3 if m in set(mv["measure"])]
    sources = sorted(mv["plan_source"].unique())
    rows = []
    for meas in measures:
        for src in sources + ["all"]:
            m_ = mv[mv["measure"] == meas] if src == "all" else mv[(mv["measure"] == meas) & (mv["plan_source"] == src)]
            r_ = rec[rec["measure"] == meas] if src == "all" else rec[(rec["measure"] == meas) & (rec["plan_source"] == src)]
            el = r_[r_["eligible"]]
            ok = m_[np.isfinite(m_["predicted"]) & np.isfinite(m_["exact"])]
            rho = spearmanr(ok["predicted"], ok["exact"]).correlation if len(ok) >= 3 else np.nan
            rec_row = dict(measure=MNAME[meas],
                           plan_source=SRC_LAB.get(src, "All plans") if src != "all" else "All plans",
                           moves=len(m_), exact_finite=len(ok), groups=len(r_), eligible=len(el),
                           spearman_rho=rho)
            for K in KS_HIT:
                rec_row[f"top{K} hit"] = (el["rank"] < K).mean() if len(el) else np.nan
                rec_row[f"top{K} random"] = (np.minimum(K, el["N"]) / el["N"]).mean() if len(el) else np.nan
            rows.append(rec_row)
    tab = pd.DataFrame(rows)
    emit("e14_screen", "E14 screening: Spearman rho and top-K hit rate of the exact-best improving move", tab)
    guard("tab e14", tables_e14, tab)
    guard("fig e14 scatter", fig_e14_scatter, mv, rec)
    guard("fig e14 topk", fig_e14_topk, rec, measures)


def tables_e14(tab):
    rows = []
    for _, r in tab.iterrows():
        rows.append(f"{r['measure']} & {r['plan_source']} & {int(r['moves'])} & "
                    f"{int(r['eligible'])}/{int(r['groups'])} & {_fmt(r['spearman_rho'], '.2f')} & "
                    f"{_fmt(r['top8 hit'], '.2f')} ({_fmt(r['top8 random'], '.2f')})")
    write_paper("tab_e14", "llrrrc",
                "Measure & Plans & Moves & Groups with improving move & Spearman $\\rho$ & "
                "Top-8 hit (random)", rows)


def fig_e14_scatter(mv, rec):
    _style()
    ok = mv[np.isfinite(mv["predicted"]) & np.isfinite(mv["exact"])]
    if ok.empty:
        print("[fig e14] no finite moves: scatter skipped")
        return
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.5))
    for ax, meas in zip(axes, MEAS3):
        g = ok[ok["measure"] == meas]
        if g.empty:
            ax.set_visible(False)
            continue
        for src, col in (("e0", COL["cost"]), ("e3_sortie0", COL["sortie0"])):
            h = g[g["plan_source"] == src]
            if len(h):
                ax.scatter(h["predicted"], h["exact"], s=6, alpha=0.35, color=col,
                           label=SRC_LAB.get(src, src), linewidths=0)
        rho = spearmanr(g["predicted"], g["exact"]).correlation if len(g) >= 3 else np.nan
        ax.text(0.04, 0.95, f"{MNAME[meas]}\n$\\rho$ = {rho:.2f}", transform=ax.transAxes,
                va="top", fontsize=8)
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_xlabel("predicted change")
    axes[0].set_ylabel("exact change")
    axes[2].legend(frameon=False, fontsize=6.5, loc="lower right", markerscale=2)
    fig.tight_layout()
    save(fig, "fig_e14_scatter")


def fig_e14_topk(rec, measures):
    _style()
    Ks = np.arange(1, 51)
    fig, ax = plt.subplots(figsize=(HALF_W, 2.6))
    for meas in measures:
        el = rec[(rec["measure"] == meas) & rec["eligible"]]
        if el.empty:
            continue
        hit = [(el["rank"] < K).mean() for K in Ks]
        rnd = [(np.minimum(K, el["N"]) / el["N"]).mean() for K in Ks]
        ax.plot(Ks, hit, "-", lw=1.1, color=MCOL[meas], label=MNAME[meas])
        ax.plot(Ks, rnd, ":", lw=0.9, color="0.55")
    ax.plot([], [], ":", color="0.55", label="random order")
    ax.set_xlabel("K (predicted-best moves kept)")
    ax.set_ylabel("exact-best improving move in top K")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(1, 50)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_e14_topk")


def _e14_ksweep(ks):
    base = ks[ks["K"] == 8][["instance", "measure", "oos_obj"]].rename(columns={"oos_obj": "oos8"})
    d = ks.merge(base, on=["instance", "measure"], how="left")
    d = d.assign(ratio=d["oos_obj"] / d["oos8"])
    tab = (d.groupby(["measure", "K"], as_index=False)
           .agg(n=("ratio", "count"), median=("ratio", "median"),
                q25=("ratio", lambda x: x.quantile(0.25)),
                q75=("ratio", lambda x: x.quantile(0.75))))
    tab["measure"] = tab["measure"].map(MNAME)
    emit("e14k_ratio", "E14 price-screen width K: ex-post value relative to K = 8", tab)
    guard("tab e14k", tables_e14k, tab)
    guard("fig e14k", fig_e14_k, tab)


def tables_e14k(tab):
    measures = [MNAME[m] for m in ["range", "gmd"] if MNAME[m] in set(tab["measure"])]
    rows = []
    for K in sorted(tab["K"].unique()):
        cells = []
        for meas in measures:
            s = tab[(tab["measure"] == meas) & (tab["K"] == K)]
            cells.append(f"{s['median'].iloc[0]:.3f} [{s['q25'].iloc[0]:.3f}, {s['q75'].iloc[0]:.3f}]"
                         if len(s) else "--")
        rows.append(f"{int(K)} & " + " & ".join(cells))
    write_paper("tab_e14k", "r" + "c" * len(measures),
                "$K$ & " + " & ".join(f"{m} median [IQR]" for m in measures), rows)


def fig_e14_k(tab):
    _style()
    fig, ax = plt.subplots(figsize=(HALF_W, 2.6))
    for meas in ["Range", "GMD"]:
        s = tab[tab["measure"] == meas].sort_values("K")
        if s.empty:
            continue
        col = MCOL[{v: k for k, v in MNAME.items()}[meas]]
        ax.plot(s["K"], s["median"], "o-", ms=3, lw=1, color=col, label=meas)
        ax.fill_between(s["K"], s["q25"], s["q75"], color=col, alpha=0.2, lw=0)
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("price-screen width $K$")
    ax.set_ylabel(r"ex-post value / value at $K{=}8$")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_e14_k")


# ----------------------------------------------------------------------------
# E15 SAA sample size
# ----------------------------------------------------------------------------

def e15():
    df = load("e15_saa")
    s0 = e3_b110("sortie0")
    if df is None:
        return
    if s0 is None:
        print("[skip] e15: E3 SORTIE-0 rows unavailable")
        return
    saa = df[df["method"] == "saa"]
    s0 = s0[["instance", "measure", "oos_obj"]].assign(method="sortie0")
    recs = []
    for S in sorted(saa["saa_S"].dropna().unique()):
        sub = saa[saa["saa_S"] == S]
        both = pd.concat([sub, s0], ignore_index=True)
        st_all = {"pooled": _stats(both, "saa", ["sortie0"]).get("sortie0")}
        for meas in MEAS3:
            st_all[meas] = _stats(both[both["measure"] == meas], "saa", ["sortie0"]).get("sortie0")
        rt = pd.to_numeric(sub["runtime"], errors="coerce") if "runtime" in sub.columns else None
        mrg = sub[["instance", "measure", "oos_obj"]].merge(s0, on=["instance", "measure"],
                                                            suffixes=("", "_s0"))
        ratio = mrg["oos_obj"] / mrg["oos_obj_s0"] if len(mrg) else pd.Series(dtype=float)
        for key in ["pooled"] + MEAS3:
            s = st_all[key]
            r_ = ratio if key == "pooled" else ratio[mrg["measure"] == key]
            q25, q75 = _iqr(r_) if len(r_) else (np.nan, np.nan)
            if rt is None or not len(rt):
                rt_med = np.nan
            else:
                rt_med = float(rt.median())
            recs.append(dict(saa_S=int(S), measure="Pooled" if key == "pooled" else MNAME[key],
                             n=s["n"] if s else 0,
                             sortie0_vs_saa_pct=s["impr"] if s else np.nan,
                             wins=s["wins"] if s else np.nan, p=s["p"] if s else np.nan,
                             rank=s["rank"] if s else np.nan,
                             saa_over_sortie0_median=float(r_.median()) if len(r_) else np.nan,
                             q25=float(q25) if len(r_) else np.nan,
                             q75=float(q75) if len(r_) else np.nan,
                             runtime_median=rt_med))
    tab = pd.DataFrame(recs)
    emit("e15_saa", "E15 SAA sample size: SORTIE-0 improvement over SAA(S) and ratio to SORTIE-0", tab)
    guard("tab e15", tables_e15, tab)
    guard("fig e15", fig_e15, tab, saa, s0)


def tables_e15(tab):
    rows = []
    for S in sorted(tab["saa_S"].unique()):
        sub = tab[tab["saa_S"] == S].set_index("measure")
        cells = []
        for m in ["Range", "GMD", "Max", "Pooled"]:
            r = sub.loc[m] if m in sub.index else None
            cells.append("--" if r is None or r["n"] == 0 else an._cell(
                r["sortie0_vs_saa_pct"], int(r["wins"]), int(r["n"]), r["p"], r["rank"]))
        p = sub.loc["Pooled", "p"] if "Pooled" in sub.index else np.nan
        pt = "--" if p != p else f"{p:.3f}"
        rows.append(f"{int(S)} & " + " & ".join(cells) + f" & {pt}")
    write_paper("tab_e15", "rcccc" + "c",
                "$S$ & Range & GMD & Max & Pooled & $p$ (pooled)", rows)


def fig_e15(tab, saa, s0):
    _style()
    if tab.empty:
        return
    has_rt = "runtime_median" in tab.columns and tab["runtime_median"].notna().any()
    fig, axes = plt.subplots(1, 2 if has_rt else 1, figsize=(FULL_W, 2.6), squeeze=False)
    ax = axes[0, 0]
    for meas in MEAS3:
        s = tab[tab["measure"] == MNAME[meas]].sort_values("saa_S")
        if s.empty:
            continue
        ax.plot(s["saa_S"], s["saa_over_sortie0_median"], "o-", ms=3, lw=1, color=MCOL[meas],
                label=MNAME[meas])
        ax.fill_between(s["saa_S"], s["q25"], s["q75"], color=MCOL[meas], alpha=0.15, lw=0)
    ax.axhline(1, color=COL["sortie0"], lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("SAA scenarios $S$")
    ax.set_ylabel("SAA ex-post value / SORTIE-0")
    ax.legend(frameon=False)
    if has_rt:
        bx = axes[0, 1]
        s = tab[tab["measure"] == "Pooled"].sort_values("saa_S")
        bx.plot(s["saa_S"], s["runtime_median"], "o-", ms=3, lw=1, color=COL["saa"])
        bx.set_xscale("log")
        bx.set_xlabel("SAA scenarios $S$")
        bx.set_ylabel("solve runtime (s, median)")
    fig.tight_layout()
    save(fig, "fig_e15")


# ----------------------------------------------------------------------------
# E16 baselines
# ----------------------------------------------------------------------------

def e16():
    df = load("e16_baselines")
    e3d = e3_b110("det")
    e3s = e3_b110("sortie0")
    if df is None:
        return
    if e3d is None or e3s is None:
        print("[skip] e16: E3 det/sortie0 rows unavailable")
        return
    cols = ["instance", "measure", "method", "oos_obj"]
    d = pd.concat([e3d[cols], e3s[cols], df[cols]], ignore_index=True)
    d = d[d["method"].isin(["det", "detsig", "pboi_normal", "sortie0"])]
    ext = pd.concat([impr_long(d, ["measure"], "det", ["detsig", "pboi_normal", "sortie0"]),
                     impr_long(d, [], "det", ["detsig", "pboi_normal", "sortie0"]),
                     impr_long(d, ["measure"], "detsig", ["sortie0"]),
                     impr_long(d, [], "detsig", ["sortie0"])], ignore_index=True)
    emit("e16_improve", "E16 baselines: improvement over Det (B = 1.10) and SORTIE-0 vs Det-sigma (%)", ext)
    guard("tab e16", tables_e16, d)


def tables_e16(d):
    rows = []
    for scope in MEAS3 + ["pooled"]:
        sub = d if scope == "pooled" else d[d["measure"] == scope]
        st = _stats(sub, "det", ["detsig", "pboi_normal", "sortie0"])
        h2h = _stats(sub, "detsig", ["sortie0"]).get("sortie0")
        label = "Pooled" if scope == "pooled" else MNAME[scope]
        rows.append(f"{label} & {cell(st.get('detsig'))} & {cell(st.get('pboi_normal'))} & "
                    f"{cell(st.get('sortie0'))} & {cell(h2h)}")
    write_paper("tab_e16", "lcccc",
                "Measure & Det-$\\sigma$ vs Det & PBOI-N vs Det & SORTIE-0 vs Det & SORTIE-0 vs Det-$\\sigma$",
                rows)


# ----------------------------------------------------------------------------
# E17 second family
# ----------------------------------------------------------------------------

def depot_type(stem, n=50):
    """Depot position in the head instance: distance of the depot from the bounding-box centre,
    relative to the half-diagonal (central < 0.2 <= intermediate <= 0.8 < eccentric)."""
    try:
        _, coords, _, _ = load_vrp(os.path.join(XDIR, stem + ".vrp"))
        xy = np.asarray(coords, dtype=float)[: n + 1]
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        centre = (lo + hi) / 2
        half_diag = 0.5 * float(np.hypot(*(hi - lo)))
        r = float(np.hypot(*(xy[0] - centre)) / half_diag)
    except Exception as exc:  # noqa: BLE001
        print(f"[e17] depot position of {stem} unavailable: {exc!r}")
        return "unknown", np.nan
    kind = "central" if r < 0.2 else ("eccentric" if r > 0.8 else "intermediate")
    return kind, r


def e17():
    df = load("e17_family")
    if df is None:
        return
    avg = (df.groupby(["instance", "measure", "method"], as_index=False)
           .agg(oos_obj=("oos_obj", "mean"), exante=("exante", "mean"),
                oos_obj_X=("oos_obj_X", "mean"), seeds=("oos_obj", "size")))
    avg = avg.assign(illusion=100 * (avg["exante"] - avg["oos_obj_X"]) / avg["oos_obj_X"])
    seeds = load("e17_seeds")
    cstar = {} if seeds is None else dict(zip(seeds["instance"], seeds["Cstar"]))
    others = ["saa", "sortie0", "detsig"]
    rows_inst = []
    for inst, g in avg.groupby("instance"):
        stem = inst[:-len("-h50")] if inst.endswith("-h50") else inst
        kind, ratio = depot_type(stem)
        s = _stats(g, "det", ["sortie0"], pair=("measure",)).get("sortie0")
        rows_inst.append(dict(instance=inst, depot=kind, depot_ratio=ratio,
                              Cstar=cstar.get(inst, np.nan),
                              sortie0_vs_det_pct=s["impr"] if s else np.nan,
                              wins=s["wins"] if s else np.nan, n=s["n"] if s else np.nan))
    inst_tab = pd.DataFrame(rows_inst)
    ext = pd.concat([impr_long(avg, ["measure"], "det", others),
                     impr_long(avg, [], "det", others)], ignore_index=True)
    ill = (avg[avg["method"] == "det"].groupby("measure", as_index=False)["illusion"].mean()
           .assign(measure=lambda x: x["measure"].map(MNAME)))
    ill_pooled = avg[avg["method"] == "det"]["illusion"].mean()
    emit("e17_improve", "E17 second family: improvement over Det (seed-averaged, %)", ext)
    emit("e17_illusion", "E17 Det belief error 100(ex-ante - truth)/truth (%)", ill, "%.2f")
    emit("e17_inst", "E17 per-instance: depot position, C*, SORTIE-0 vs Det (pooled over measures)",
         inst_tab, "%.2f")
    guard("tab e17", tables_e17, avg, ill, ill_pooled)
    guard("tab e17 inst", tables_e17_inst, inst_tab)
    guard("fig e17", fig_e17, inst_tab)


def tables_e17(avg, ill, ill_pooled):
    rows = []
    for scope in MEAS3 + ["pooled"]:
        sub = avg if scope == "pooled" else avg[avg["measure"] == scope]
        st = _stats(sub, "det", ["saa", "sortie0", "detsig"])
        il = ill_pooled if scope == "pooled" else float(sub[sub["method"] == "det"]["illusion"].mean())
        label = "Pooled" if scope == "pooled" else MNAME[scope]
        rows.append(f"{label} & {cell(st.get('saa'))} & {cell(st.get('sortie0'))} & "
                    f"{cell(st.get('detsig'))} & {_fmt(il, '+.1f')}")
    write_paper("tab_e17", "lcccc",
                "Measure & SAA vs Det & SORTIE-0 vs Det & Det-$\\sigma$ vs Det & Det belief err. (\\%)",
                rows)


def tables_e17_inst(inst_tab):
    rows = []
    for _, r in inst_tab.iterrows():
        label = r["instance"].replace("_", "\\_")
        cs = "--" if r["Cstar"] != r["Cstar"] else f"{r['Cstar']:.0f}"
        cellv = "--" if r["sortie0_vs_det_pct"] != r["sortie0_vs_det_pct"] else (
            f"{r['sortie0_vs_det_pct']:+.2f} ({int(r['wins'])}/{int(r['n'])})")
        rows.append(f"{label} & {r['depot']} & {cs} & {cellv}")
    write_paper("tab_e17_inst", "lrrr",
                "Instance & Depot & $C^\\ast$ & SORTIE-0 vs Det, mean of measures (wins/3)", rows)


def fig_e17(inst_tab):
    _style()
    t = inst_tab.dropna(subset=["sortie0_vs_det_pct"]).sort_values("sortie0_vs_det_pct", ascending=False)
    if t.empty:
        print("[fig e17] no per-instance values: figure skipped")
        return
    fig, ax = plt.subplots(figsize=(FULL_W, 2.6))
    x = np.arange(len(t))
    cols = [DEPOT_COL.get(k, DEPOT_COL["unknown"]) for k in t["depot"]]
    ax.bar(x, t["sortie0_vs_det_pct"], color=cols, width=0.7)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([s[:-len("-h50")] if s.endswith("-h50") else s
                        for s in t["instance"]], rotation=90, fontsize=6.5)
    ax.set_ylabel("SORTIE-0 vs Det (%)")
    from matplotlib.patches import Patch
    handles = [Patch(color=DEPOT_COL[k], label=k) for k in ["central", "intermediate", "eccentric"]
               if (t["depot"] == k).any()]
    ax.legend(handles=handles, frameon=False, title="depot position", title_fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_e17")


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

EXPERIMENTS = [e9, e10, e11, e12, e13, e14, e15, e16, e17]


def main():
    for fn in EXPERIMENTS:
        guard(fn.__name__, fn)
    path = os.path.join(TAB_EXT, "ALL_EXT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# SORTIE extension tables E9-E17 (generated by experiments/analyze_ext.py)\n"
                 + "".join(OUT_MD))
    print(f"[table] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

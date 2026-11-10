"""Per-instance appendix tables (longtable) for the SORTIE manuscript.

Run from papers/sortie/code:  python experiments/appendix_tables.py
Writes results/tables/paper/app_*.tex.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SORTIE_RESULTS", os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RES, "tables", "paper")
os.makedirs(OUT, exist_ok=True)
MNAME = {"max": "Max", "range": "Range", "gmd": "GMD", "mad": "MAD", "var": "Var"}
LAB = {"cost": "Cost", "det": "Det", "saa": "SAA", "sortie0": "S-0", "sortie1": "S-1",
       "detsig": "Det-$\\sigma$", "pboi_normal": "PBOI-N"}


def _fmt(x):
    return "--" if pd.isna(x) else (f"{x:,.0f}" if abs(x) >= 100 else f"{x:.1f}")


def _longtable(name, cols, header, rows, caption, label):
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write("\\begin{small}\n\\begin{longtable}{" + cols + "}\n")
        f.write(f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n\\toprule\n{header}\\\\\n\\midrule\n\\endfirsthead\n")
        f.write(f"\\toprule\n{header}\\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n")
        for r in rows:
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\end{longtable}\n\\end{small}\n")


def e3_tables():
    d = pd.read_csv(os.path.join(RES, "e3_main.csv"))
    cost = d[d.method == "cost"].set_index("instance")
    fair = d[d.method != "cost"]
    meths = ["det", "saa", "sortie0", "sortie1"]
    for B in (1.05, 1.10):
        g = fair[abs(fair.B - B) < 1e-9]
        piv = g.pivot_table(index=["instance", "measure"], columns="method", values="oos_obj")
        cr = g.pivot_table(index=["instance", "measure"], columns="method", values="cost_ratio")
        rows = []
        insts = sorted(piv.index.get_level_values(0).unique(),
                       key=lambda s: (int(s.split("-n")[1].split("-")[0]), int(s.split("-b")[1])))
        for inst in insts:
            for meas in ["max", "range", "gmd", "mad", "var"]:
                if (inst, meas) not in piv.index:
                    continue
                v = piv.loc[(inst, meas)]
                best = min(v[m] for m in meths if m in v)
                cells = []
                for m in meths:
                    c = _fmt(v.get(m))
                    if m in v and abs(v[m] - best) < 1e-9:
                        c = "\\textbf{" + c + "}"
                    cells.append(c)
                cval = cost.loc[inst, f"E_{meas}"] if inst in cost.index else float("nan")
                imp = 100 * (v["det"] - v["sortie0"]) / v["det"]
                rows.append([inst.replace("X641-", "") if meas == "max" else "", MNAME[meas],
                             _fmt(cval)] + cells + [f"{imp:+.2f}", f"{cr.loc[(inst, meas)].get('sortie0', float('nan')):.3f}"])
        header = ("Instance & Measure & Cost & " + " & ".join(LAB[m] for m in meths)
                  + " & Gain (\\%) & $\\sum L/C^\\ast$")
        tag = "105" if B < 1.07 else "110"
        _longtable(f"app_e3_B{tag}", "llrrrrrrr", header, rows,
                   f"E3, $B={B:.2f}$: out-of-sample ex-post value of every plan (bold: best); "
                   "Cost: the cost-optimal plan; S-0/S-1: SORTIE-0/1; Gain: improvement of S-0 over Det; "
                   "last column: total length of the S-0 plan relative to $C^\\ast$.", f"tab:app_e3_{tag}")


def e7b_table():
    p = os.path.join(RES, "e7b_seeds.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[d.method != "cost"]
    agg = d.groupby(["instance", "measure", "method"])["oos_obj"].agg(["mean", "std"]).unstack()
    rows = []
    for (inst, meas), r in agg.iterrows():
        cells = [f"{_fmt(r[('mean', m)])} ({_fmt(r[('std', m)])})" for m in ["det", "saa", "sortie0"]]
        imp = 100 * (r[("mean", "det")] - r[("mean", "sortie0")]) / r[("mean", "det")]
        rows.append([inst.replace("X641-", ""), MNAME[meas]] + cells + [f"{imp:+.2f}"])
    _longtable("app_e7b", "llrrrr", "Instance & Measure & Det & SAA & S-0 & S-0 vs Det (\\%)", rows,
               "E7b, $n=100$: mean (standard deviation) over three solver seeds of the ex-post value.",
               "tab:app_e7b")


def e17_table():
    p = os.path.join(RES, "e17_family.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[d.method != "cost"]
    agg = d.groupby(["instance", "measure", "method"])["oos_obj"].mean().unstack()
    rows = []
    prev = None
    for (inst, meas), r in agg.iterrows():
        meths = [m for m in ["det", "saa", "sortie0", "detsig"] if m in r]
        best = min(r[m] for m in meths)
        cells = []
        for m in meths:
            c = _fmt(r[m])
            cells.append("\\textbf{" + c + "}" if abs(r[m] - best) < 1e-9 else c)
        imp = 100 * (r["det"] - r["sortie0"]) / r["det"]
        rows.append([inst if inst != prev else "", MNAME[meas]] + cells + [f"{imp:+.2f}"])
        prev = inst
    _longtable("app_e17", "llrrrrr", "Instance & Measure & Det & SAA & S-0 & Det-$\\sigma$ & S-0 vs Det (\\%)",
               rows, "E17: ex-post value, mean over two solver seeds (bold: best).", "tab:app_e17")


if __name__ == "__main__":
    for fn in (e3_tables, e7b_table, e17_table):
        try:
            fn()
        except Exception as exc:  # report and continue
            print(f"{fn.__name__} failed: {exc!r}")
    print("written to", OUT)

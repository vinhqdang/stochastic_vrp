"""Tables, figures and in-text numbers of the TRANSVRP manuscript, generated from the
committed result files only. Never edit tables/*, figures/* or numbers.tex by hand.

Usage (from papers/transvrp):  python make_tables.py
Inputs:  code/results/expA/main_boost.test.csv          (experiment A, test stream)
         code/results/expA/eligibility_dev.csv           (routes)
         code/results/expB/*.test_b.csv                  (experiment B, test_b stream)
         code/results/expB/analysis_test_b/*.csv         (analysis_b output)
Outputs: tables/*.tex, figures/*.pdf, numbers.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
sys.path.insert(0, str(CODE))
from transvrp.expa import analysis as A  # noqa: E402

RES_A = CODE / "results" / "expA"
RES_B = CODE / "results" / "expB"
AB = RES_B / "analysis_test_b"
TAB = HERE / "tables"
FIG = HERE / "figures"
N_STAR = 100_000
DELTA = A.DELTA

ARM_NAME = {  # display names used everywhere in the paper
    "nn_full": r"\textsc{Full}", "full": r"\textsc{Full}",
    "nn_scalar": r"\textsc{Scalar}", "nn_mlp": r"\textsc{Mlp}", "boost": r"\textsc{Boost}",
    "baton_eq": r"\textsc{Baton}", "baton_big": r"\textsc{Baton}-$10^6$",
    "dp_big": r"DP-$10^6$", "otr21": r"\textsc{Baton}-rich",
    "thr": r"\textsc{Thr}", "alpz": r"\textsc{Alp-Z}", "rlsm": r"\textsc{Rlsm}",
    "dqn": r"\textsc{Dqn}", "ppo": r"\textsc{Ppo}", "none": "never hand off",
    "oracle": "clairvoyant", "tuned_tau": "tuned threshold",
}
PLAIN = {"nn_full": "FULL", "nn_scalar": "SCALAR", "nn_mlp": "MLP", "boost": "BOOST",
         "thr": "THR", "alpz": "ALP-Z", "rlsm": "RLSM", "dqn": "DQN", "ppo": "PPO",
         "otr21": "BATON-rich", "dp_big": "DP-1e6", "baton_big": "BATON-1e6"}
STATE = {"nn_full": "rich", "nn_mlp": "rich", "boost": "rich", "rlsm": "rich",
         "dqn": "rich", "ppo": "rich", "otr21": "rich", "nn_scalar": "scalar",
         "thr": "scalar", "alpz": "scalar", "dp_big": "scalar", "baton_big": "scalar"}
BRANCH = {"WIN": "win", "TIE_STRONG": "equivalent", "TIE_WEAK": "inconclusive",
          "LOSE": "lose", "better": "better", "worse": "worse",
          "equivalent": "equivalent", "inconclusive": "inconclusive"}


# ---------------------------------------------------------------- helpers

def f2(x: float) -> str:
    out = f"{x:+.2f}"
    return "0.00" if out in ("+0.00", "-0.00") else out


def ci(lo: float, hi: float) -> str:
    return f"[{lo:+.2f}, {hi:+.2f}]"


def sec(x: float) -> str:
    return f"{x:,.0f}" if x >= 10 else f"{x:.1f}"


def pfmt(p: float) -> str:
    if p < 1e-4:
        m, e = f"{p:.1e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{p:.3f}" if p >= 1e-3 else f"{p:.4f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def table(cols: str, head: str, rows: list[str]) -> str:
    return ("\\begin{tabular}{" + cols + "}\n\\toprule\n" + head + " \\\\\n\\midrule\n"
            + "\n".join(r + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------- data

def load_a() -> pd.DataFrame:
    return pd.read_csv(RES_A / "main_boost.test.csv")


def load_b() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(RES_B.glob("b_*.test_b.csv"))
              if not p.name.startswith(("b_ladder_ind", "b_thr_ind", "b_alpz_ind",
                                        "b_full_ind"))]
    df = pd.concat(frames, ignore_index=True)
    df["policy"] = df["policy"].replace({"full": "nn_full"})
    return df


def contrasts_a(df: pd.DataFrame) -> list[dict]:
    """The pre-registered contrasts of A (section 8), recomputed with A's own code; the
    Holm adjustment over S1-S11 is taken from the committed summary (identical values)."""
    spec = [("P", "nn_full", N_STAR, "baton_eq", N_STAR),
            ("S1", "nn_full", 100, "baton_eq", 100), ("S2", "nn_full", 500, "baton_eq", 500),
            ("S3", "nn_full", 2000, "baton_eq", 2000),
            ("S4", "nn_full", 10_000, "baton_eq", 10_000),
            ("S5", "nn_full", N_STAR, "nn_scalar", N_STAR),
            ("S6", "nn_full", 2000, "nn_scalar", 2000),
            ("S7", "nn_scalar", N_STAR, "dp_big", 1_000_000),
            ("S8", "nn_scalar", 2000, "baton_eq", 2000),
            ("S9", "nn_full", N_STAR, "nn_mlp", N_STAR),
            ("S10", "nn_full", N_STAR, "baton_big", 1_000_000),
            ("S11", "nn_full", N_STAR, "dp_big", 1_000_000),
            ("M11", "nn_full", N_STAR, "boost", N_STAR)]
    out = [A.contrast(df, *s) for s in spec]
    sec = [c for c in out if c["id"].startswith("S")]
    for c, p in zip(sec, A.holm([c["p_two_sided"] for c in sec])):
        c["p_holm"] = p
        c["classification"] = A.classify(c["_d"], DELTA, p_adj=p)
    out[0]["classification"] = A.go_no_go(out[0]["_d"], DELTA)["branch"]
    out[-1]["classification"] = A.classify(out[-1]["_d"], DELTA, p_adj=out[-1]["p_two_sided"])
    return out


def vs_baton(df: pd.DataFrame, arms: list[str]) -> list[dict]:
    """Descriptive: each arm against BATON at 10^5 on the same stream (A's statistics)."""
    return [A.contrast(df, f"D_{a}", a, N_STAR, "baton_eq", N_STAR) for a in arms]


# ---------------------------------------------------------------- tables

def t_routes() -> None:
    e = pd.read_csv(RES_A / "eligibility_dev.csv")
    el = e[e["eligible"].astype(bool)] if "eligible" in e else e
    rows = []
    for s in ("short", "medium", "long"):
        sub = el[el["stratum"] == s]
        rows.append(f"{s} & {len(sub)} & {sub['m'].min()}--{sub['m'].max()} & "
                    f"{sub['inst'].nunique()}")
    rows.append(f"all & {len(el)} & {el['m'].min()}--{el['m'].max()} & {el['inst'].nunique()}")
    write(TAB / "routes.tex", table("lrrr", "stratum & routes & stops $m$ & instances", rows))


def t_expa(cs: list[dict]) -> None:
    label = {"P": "primary", "M11": "own $\\alpha$"}
    rows = []
    for c in cs:
        a = f"{ARM_NAME[c['a']]} @ {c['N_a']:,}".replace(",", "{,}")
        b = f"{ARM_NAME[c['b']]} @ {c['N_b']:,}".replace(",", "{,}")
        p = c.get("p_holm", c["p_two_sided"])
        rows.append(f"{c['id']} & {a} & {b} & {f2(c['mean'])} & {ci(*c['ci95'])} & "
                    f"{ci(*c['ci90'])} & {c['n_better']}/{c['n']} & {pfmt(p)} & "
                    f"{BRANCH[c['classification']]}")
    write(TAB / "expa.tex", table(
        "llllllrll", "id & $a$ & $b$ & $\\bar\\Delta$ (\\%) & CI95 & CI90 & $a$ better & "
        "$p$ & class", rows))


def t_expb() -> None:
    fam = pd.read_csv(AB / "family.csv")
    out = pd.read_csv(AB / "outside.csv")
    rows = []
    for r in fam.itertuples():
        rows.append(f"{ARM_NAME[r.arm]} & family & {f2(r.mean)} & {ci(r.ci95_lo, r.ci95_hi)} & "
                    f"{ci(r.holm_lo, r.holm_hi)} & {ci(r.ci90_lo, r.ci90_hi)} & "
                    f"{r.n_better}/{r.n} & {pfmt(r.p_holm)} & {BRANCH[r.branch]}")
    for r in out.itertuples():
        rows.append(f"{ARM_NAME[r.arm]} & descriptive & {f2(r.mean)} & "
                    f"{ci(r.ci95_lo, r.ci95_hi)} & -- & {ci(r.ci90_lo, r.ci90_hi)} & "
                    f"{r.n_better}/{r.n} & {pfmt(r.p_raw)} & --")
    write(TAB / "expb.tex", table(
        "llllllrll", "arm & status & $\\bar\\Delta$ (\\%) & CI95 & Holm CI & CI90 & "
        "\\textsc{Full} better & $p$ & class", rows))


def t_indep() -> None:
    ind = pd.read_csv(AB / "independence.csv")
    rows = [f"{r.comparison.replace('FULL', ARM_NAME['nn_full']).replace('baton_eq', ARM_NAME['baton_eq']).replace('THR', ARM_NAME['thr']).replace('ALP-Z', ARM_NAME['alpz'])} & "
            f"{f2(r.mean)} & {ci(r.ci95_lo, r.ci95_hi)} & {r.n_better}/{r.n} & {pfmt(r.p_raw)}"
            for r in ind.itertuples()]
    write(TAB / "indep.tex", table("lllrl", "comparison & $\\bar\\Delta$ (\\%) & CI95 & "
                                   "\\textsc{Full} better & $p$", rows))


def t_compute(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    rows = []
    a = df_a[df_a["N"] == N_STAR]
    for pol, src in (("nn_full", a), ("nn_mlp", a), ("boost", a)):
        s = src[src["policy"] == pol]["fit_seconds"]
        rows.append(f"{ARM_NAME[pol]} & A & {sec(s.median())} & {sec(s.max())}")
    for pol in ("thr", "alpz", "rlsm", "dqn", "ppo"):
        s = df_b[df_b["policy"] == pol]["fit_seconds"]
        rows.append(f"{ARM_NAME[pol]} & B & {sec(s.median())} & {sec(s.max())}")
    rows = [r.replace(",", "{,}") for r in rows]
    write(TAB / "compute.tex", table("llrr", "method & experiment & median (s) & max (s)",
                                     rows))


# ---------------------------------------------------------------- figures

def fig_curve(df: pd.DataFrame) -> pd.DataFrame:
    cur = A.curve(df, DELTA)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = cur["N"].to_numpy()
    ax.axhspan(-DELTA, DELTA, color="0.9", lw=0)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.errorbar(x, cur["delta_bar"], yerr=[cur["delta_bar"] - cur["ci95_lo"],
                cur["ci95_hi"] - cur["delta_bar"]], fmt="o-", color="#1f4e79", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("training paths per route $N$")
    ax.set_ylabel(r"$\bar\Delta$ FULL vs BATON (%)")
    fig.tight_layout()
    fig.savefig(FIG / "curve.pdf")
    plt.close(fig)
    return cur


def fig_forest(da: list[dict], db: pd.DataFrame) -> None:
    items = [(PLAIN[c["a"]] + " (A)", STATE[c["a"]], c["mean"], *c["ci95"]) for c in da]
    items += [(PLAIN[r.arm] + " (B)", STATE[r.arm], r.mean, r.ci95_lo, r.ci95_hi)
              for r in db.itertuples()]
    items.sort(key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for i, (name, st, m, lo, hi) in enumerate(items):
        col = "#1f4e79" if st == "rich" else "#c55a11"
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o" if st == "rich" else "s",
                    color=col, capsize=2)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(len(items)), [t[0] for t in items], fontsize=8)
    ax.set_xlabel(r"$\bar\Delta$ vs BATON at $10^5$ paths (%; positive = cheaper)")
    ax.plot([], [], "o", color="#1f4e79", label="rich state")
    ax.plot([], [], "s", color="#c55a11", label="load-only state")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "forest.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- numbers.tex

def macro(name: str, val: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{val}}}\n"


def numbers(cs: list[dict], cur: pd.DataFrame, da: list[dict]) -> None:
    by = {c["id"]: c for c in cs}
    fam = pd.read_csv(AB / "family.csv").set_index("arm")
    out = pd.read_csv(AB / "outside.csv").set_index("arm")
    dab = pd.read_csv(AB / "da.csv").set_index("arm")
    ind = pd.read_csv(AB / "independence.csv").set_index("comparison")
    s = "% generated by make_tables.py; do not edit\n"
    words = {"P": "Prim", "S1": "Sone", "S2": "Stwo", "S3": "Sthree", "S4": "Sfour",
             "S5": "Sfive", "S6": "Ssix", "S7": "Sseven", "S8": "Seight", "S9": "Snine",
             "S10": "Sten", "S11": "Seleven", "M11": "Boost"}
    for cid, w in words.items():
        c = by[cid]
        s += macro(f"A{w}", f"{c['mean']:+.2f}\\%")
        s += macro(f"A{w}CI", ci(*c["ci95"]))
        s += macro(f"A{w}CIn", ci(*c["ci90"]))
        s += macro(f"A{w}Better", f"{c['n_better']}/{c['n']}")
    for arm, w in (("alpz", "Alpz"), ("rlsm", "Rlsm"), ("dqn", "Dqn"), ("ppo", "Ppo")):
        r = fam.loc[arm]
        s += macro(f"B{w}", f"{r['mean']:+.2f}\\%")
        s += macro(f"B{w}CI", ci(r["ci95_lo"], r["ci95_hi"]))
        s += macro(f"B{w}CIn", ci(r["ci90_lo"], r["ci90_hi"]))
        s += macro(f"B{w}Better", f"{int(r['n_better'])}/{int(r['n'])}")
        s += macro(f"B{w}VsBaton", f"{dab.loc[arm, 'mean']:+.2f}\\%")
    r = out.loc["thr"]
    s += macro("BThr", f"{r['mean']:+.2f}\\%")
    s += macro("BThrCI", ci(r["ci95_lo"], r["ci95_hi"]))
    s += macro("BThrVsBaton", f"{dab.loc['thr', 'mean']:+.2f}\\%")
    for c in da:
        s += macro(f"Avs{PLAIN[c['a']].replace('-', '').replace('1e6', 'Big')}",
                   f2(c["mean"]) + "\\%")
    for key, w in (("FULL vs baton_eq", "IndBaton"), ("FULL vs THR", "IndThr"),
                   ("FULL vs ALP-Z", "IndAlpz"), ("FULL vs none", "IndNone")):
        s += macro(f"B{w}", f"{ind.loc[key, 'mean']:+.2f}\\%")
        s += macro(f"B{w}CI", ci(ind.loc[key, "ci95_lo"], ind.loc[key, "ci95_hi"]))
    s += macro("Crossover", f"{int(A.crossover_n(cur, DELTA)):,}".replace(",", "{,}"))
    s += macro("Margin", f"{DELTA:.1f}\\%")
    s += extra_numbers(by, da)
    write(HERE / "numbers.tex", s)


def extra_numbers(by: dict, da: list[dict]) -> str:
    """Derived quantities quoted in the prose (all from the committed result files)."""
    df_b = load_b()
    s = ""
    reval = A.contrast(df_b, "RE", "nn_full", N_STAR, "baton_eq", N_STAR)
    s += macro("BReEval", f2(reval["mean"]) + "\\%")
    s += macro("BReEvalCI", ci(*reval["ci95"]))
    gap = A.contrast(df_b, "TA", "alpz", N_STAR, "thr", N_STAR)   # positive: ALP-Z cheaper
    s += macro("ThrAlpzGap", f"{abs(gap['mean']):.2f}\\%")
    scal = next(c for c in da if c["a"] == "nn_scalar")
    s += macro("AvsSCALARCIn", ci(*scal["ci90"]))
    prim = by["P"]["mean"]
    for cid, w in (("S9", "Mlp"), ("M11", "Boost")):
        s += macro(f"Share{w}", f"{100 * by[cid]['mean'] / prim:.0f}\\%")
    fam = pd.read_csv(AB / "family.csv").set_index("arm")
    for arm, w in (("rlsm", "Rlsm"), ("dqn", "Dqn")):
        s += macro(f"Share{w}", f"{100 * fam.loc[arm, 'mean'] / reval['mean']:.0f}\\%")
    sd = pd.read_csv(AB / "replicate_sd.csv").set_index("arm")["mean_sd_pct"]
    for arm, w in (("nn_full", "Full"), ("baton_eq", "Baton"), ("rlsm", "Rlsm"),
                   ("dqn", "Dqn"), ("ppo", "Ppo")):
        s += macro(f"RepSd{w}", f"{sd[arm]:.2f}\\%")
    ppo = df_b[df_b["policy"] == "ppo"]["fit_seconds"].median()
    s += macro("PpoHours", f"{ppo / 3600:.1f}")
    full_b = df_b[df_b["policy"] == "nn_full"]["fit_seconds"].median()
    s += macro("FullMinutesB", f"{full_b / 60:.0f}")
    return s


def main() -> int:
    TAB.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df_a, df_b = load_a(), load_b()
    cs = contrasts_a(df_a)
    da = vs_baton(df_a, ["nn_full", "nn_scalar", "nn_mlp", "boost", "otr21"])
    cur = fig_curve(df_a)
    t_routes()
    t_expa(cs)
    t_expb()
    t_indep()
    t_compute(df_a, df_b)
    fig_forest(da, pd.read_csv(AB / "da.csv"))
    numbers(cs, cur, da)
    print("tables:", sorted(p.name for p in TAB.glob("*.tex")))
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

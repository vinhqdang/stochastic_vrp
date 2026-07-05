#!/usr/bin/env python3
"""
make_tables.py — regenerate every manuscript table and inline-number macro
from the result CSVs. NEVER hand-edit the files in tables/; rerun this.

Inputs  (svrpspd_wdro/results/):
    results_otr2_synthetic.csv        structural synthetic benchmark
    results_otr2_eval.csv             Dethloff, flat two-price cost model
    results_realistic_eval.csv        Dethloff, realistic last-mile costs
    results_otr2_sensitivity.csv      5-sweep sensitivity study

Outputs (paper/tables/):
    macros.tex, tab_synthetic.tex, tab_dethloff.tex,
    tab_realistic.tex, tab_sensitivity.tex

Missing inputs produce placeholder macros/tables so the draft still
builds; rerun after the corresponding experiment completes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE    = Path(__file__).resolve().parent
RESULTS = HERE.parent / "svrpspd_wdro" / "results"
OUT     = HERE / "tables"
OUT.mkdir(exist_ok=True)

PLACEHOLDER = r"\emph{(pending)}"


def _read(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    if not p.exists():
        print(f"  [warn] {name} missing — placeholders emitted")
        return None
    return pd.read_csv(p)


def _pct(x, nd=1):
    return f"{x:.{nd}f}\\%"


def _write(name: str, content: str) -> None:
    (OUT / name).write_text(content)
    print(f"  wrote tables/{name}")


# ═══════════════════════════════════════════════════════════════════════════════
macros: dict[str, str] = {}

syn  = _read("results_otr2_synthetic.csv")
deth = _read("results_otr2_eval.csv")
real = _read("results_realistic_eval.csv")
sens = _read("results_otr2_sensitivity.csv")


# ── synthetic table + macros ─────────────────────────────────────────────────
if syn is not None:
    def _ms(scen, col):
        sub = syn[syn["scenario"] == scen][col]
        return sub.mean(), sub.std()

    rows = []
    NAMES = {"collect_then_deliver": "Collect-then-deliver",
             "milk_run_regime":      "Milk run + regime switching",
             "high_cost_ratio":      r"Milk run, $C_{\mathrm{fail}}/\omega_F{=}20$"}
    for scen, disp in NAMES.items():
        cells = [disp]
        for lbl in ("v1_myo", "v1_tun", "fb_tun", "v2_lsm"):
            m, s = _ms(scen, f"{lbl}_saving")
            cells.append(f"${m:.1f} \\pm {s:.1f}$")
        mfail, _ = _ms(scen, "v2_lsm_fail")
        cells.append(f"${100 * mfail:.2f}$")
        rows.append(" & ".join(cells) + r" \\")

    tab = r"""\begin{table}[t]
\caption{Structural synthetic scenarios: execution-cost saving over the
reactive baseline (\%, mean $\pm$ s.d.\ over five seeds, $1.2\times10^4$
test routes each). The endpoint-trained v1 policy saves exactly zero on
collect-then-deliver routes because its training labels are almost surely
zero (Proposition~\ref{prop:bias}).}
\label{tab:synthetic}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
Scenario & v1 myopic & v1 tuned & Peak fallback & OTR-2.0 &
Emerg.\ \% (v2) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_synthetic.tex", tab)

    ctd  = syn[syn["scenario"] == "collect_then_deliver"]
    milk = syn[syn["scenario"] == "milk_run_regime"]
    high = syn[syn["scenario"] == "high_cost_ratio"]
    macros["vTwoCtdSaving"]  = _pct(ctd["v2_lsm_saving"].mean())
    macros["fbCtdSaving"]    = _pct(ctd["fb_tun_saving"].mean())
    macros["vTwoMilkSaving"] = _pct(milk["v2_lsm_saving"].mean())
    macros["vOneMilkSaving"] = _pct(milk["v1_tun_saving"].mean())
    macros["vTwoMilkEdge"]   = f"{milk['v2_lsm_saving'].mean() - milk['v1_tun_saving'].mean():.1f}"
    macros["vTwoHighSaving"] = _pct(high["v2_lsm_saving"].mean())
    macros["vOneHighSaving"] = _pct(high["v1_tun_saving"].mean())


# ── Dethloff flat-cost table + Wilcoxon macro ────────────────────────────────
if deth is not None:
    rows, wtexts = [], []
    for plan in ("Det", "SAA", "WDRO"):
        sub = deth[deth["Plan"] == plan]
        cells = [rf"\textsc{{{plan}}}"]
        for lbl in ("none", "v1_myo", "v1_tun", "fb_tun", "v2_lsm"):
            cells.append(f"{sub[f'{lbl}_TBC'].mean():,.0f}")
        for lbl in ("v1_tun", "fb_tun", "v2_lsm"):
            cells.append(f"{sub[f'{lbl}_saving'].mean():.1f}")
        cells.append(f"{100 * sub['v2_lsm_fail'].mean():.2f}")
        rows.append(" & ".join(cells) + r" \\")

        d = sub["v1_tun_exec"].values - sub["v2_lsm_exec"].values
        if np.allclose(d, 0):
            p = 1.0
        else:
            p = float(sps.wilcoxon(d, alternative="greater").pvalue)
        if p < 0.001:
            wtexts.append(rf"$p<0.001$ on \textsc{{{plan}}}")
        elif p < 0.05:
            wtexts.append(rf"$p={p:.3f}$ on \textsc{{{plan}}}")
        else:
            wtexts.append(rf"$p={p:.2f}$ (n.s.) on \textsc{{{plan}}}")
    macros["wilcoxonDethloff"] = ", ".join(wtexts) + \
        " (one-sided Wilcoxon signed-rank, v2 vs tuned v1, $n{=}40$ instances)"

    tab = r"""\begin{table}[t]
\caption{Dethloff benchmark, flat two-price cost model
($C_{\mathrm{fail}}/\omega_F = 5$): total budget cost (travel +
vehicles + expected execution cost; mean over 40 instances) and
execution-cost saving over the reactive policy, by planning gate.}
\label{tab:dethloff}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrccc c}
\toprule
& \multicolumn{5}{c}{Total budget cost} &
\multicolumn{3}{c}{Saving vs reactive (\%)} & Emerg.\,\% \\
\cmidrule(lr){2-6}\cmidrule(lr){7-9}
Gate & reactive & v1 myo & v1 tuned & fallback & OTR-2.0 &
v1 tuned & fallback & OTR-2.0 & (v2) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_dethloff.tex", tab)


# ── realistic-cost table + macros ────────────────────────────────────────────
if real is not None:
    rows = []
    for plan in ("Det", "SAA", "WDRO"):
        sub = real[real["Plan"] == plan]
        cells = [rf"\textsc{{{plan}}}", f"{sub['Fixed_cost'].mean():,.0f}"]
        for lbl in ("none", "v1_end", "fb_tau", "v2_lsm", "oracle"):
            cells.append(f"{sub[f'{lbl}_TBC'].mean():,.0f}")
        for lbl in ("v1_end", "fb_tau", "v2_lsm"):
            cells.append(f"{100 - sub[f'{lbl}_gap'].mean():.1f}")
        rows.append(" & ".join(cells) + r" \\")

    tab = r"""\begin{table}[t]
\caption{Dethloff benchmark under the realistic last-mile cost model of
Section~\ref{sec:costs} (state-dependent handoff and emergency prices
from real route geometry): total budget cost (mean over 40 instances)
and the share of the oracle-achievable recourse saving each policy
captures.}
\label{tab:realistic}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrr ccc}
\toprule
& & \multicolumn{5}{c}{Total budget cost} &
\multicolumn{3}{c}{Share of oracle saving (\%)} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-10}
Gate & Fixed & reactive & v1 endpoint & fallback & OTR-2.0 & oracle &
v1 endp. & fallback & OTR-2.0 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_realistic.tex", tab)

    macros["vTwoRealisticShare"] = _pct(100 - real["v2_lsm_gap"].mean())
    macros["fbRealisticShare"]   = _pct(100 - real["fb_tau_gap"].mean())
    macros["vOneRealisticShare"] = _pct(100 - real["v1_end_gap"].mean())
else:
    macros.setdefault("vTwoRealisticShare", PLACEHOLDER)
    macros.setdefault("fbRealisticShare",   PLACEHOLDER)
    macros.setdefault("vOneRealisticShare", PLACEHOLDER)
    _write("tab_realistic.tex",
           "% pending: results_realistic_eval.csv not yet available\n")


# ── sensitivity table ────────────────────────────────────────────────────────
if sens is not None:
    SWEEP_DISP = {
        "cost_ratio":  r"$C_{\mathrm{fail}}/\omega_F$",
        "n_train":     r"history $N$",
        "route_len":   r"route length $m$",
        "family":      "increment family",
        "correlation": r"common factor $\rho$",
    }
    rows = []
    for sw in ("cost_ratio", "n_train", "route_len", "family", "correlation"):
        grp = sens[sens["sweep"] == sw]
        first = True
        for x in sorted(grp["x"].unique(), key=str):
            cells = [SWEEP_DISP[sw] if first else "", str(x)]
            first = False
            for scen in ("ctd", "milk_run"):
                sub = grp[(grp["scenario"] == scen) & (grp["x"] == x)]
                for lbl in ("v1_tun", "fb_tun", "v2_lsm"):
                    cells.append(f"{sub[f'{lbl}_saving'].mean():.1f}")
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    while rows and rows[-1] == r"\addlinespace":
        rows.pop()

    tab = r"""\begin{table}[t]
\caption{Sensitivity study: execution-cost saving over the reactive
baseline (\%, mean over five seeds) as one factor varies with all others
at their defaults ($C_{\mathrm{fail}}/\omega_F{=}5$, $N{=}10^4$,
$m{=}12$, gamma increments, $\rho{=}0$).}
\label{tab:sensitivity}
\centering
\setlength{\tabcolsep}{4pt}
\begin{tabular}{ll ccc ccc}
\toprule
& & \multicolumn{3}{c}{Collect-then-deliver} &
\multicolumn{3}{c}{Milk run} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Factor & Value & v1 & fallb. & v2 & v1 & fallb. & v2 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_sensitivity.tex", tab)


# ── macros file ──────────────────────────────────────────────────────────────
for key in ("vTwoCtdSaving", "fbCtdSaving", "vTwoMilkSaving",
            "vOneMilkSaving", "vTwoMilkEdge", "vTwoHighSaving",
            "vOneHighSaving", "wilcoxonDethloff"):
    macros.setdefault(key, PLACEHOLDER)

lines = ["% generated by make_tables.py — do not hand-edit"]
for k, v in sorted(macros.items()):
    lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
_write("macros.tex", "\n".join(lines) + "\n")

print("done.")

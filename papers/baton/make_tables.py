#!/usr/bin/env python3
"""
make_tables.py — generate ALL manuscript tables + inline-number macros from
the result CSVs in ../svrpspd_wdro/results/. Never hand-edit paper/tables/*.

Inputs (all produced by svrpspd_wdro/scripts/):
    results_grand_dethloff.csv       6 gates x 13 policies x 40 instances
    results_salhinagy_eval.csv       14 instances (Det gate)
    results_city_eval.csv            19 shop-based city instances
    results_cityuniform_eval.csv     9 uniform-scatter twins
    results_costsens_*.csv           8 one-factor economic configurations
    results_otr2_synthetic.csv       structural synthetic scenarios
    results_mip_cert.csv / _gurobi   planning-layer MIP certification
    rl_results.json, rl_strong_s*.json   RL baseline (Colab T4)
    rl_bundle.npz                    routes for the RL head-to-head
    results_cityzp{25,50}_eval.csv   deliver-only sensitivity twins
    routes/*_routes.csv              route-level results (supplementary)
    r1/*.csv                         robustness, pool, budget, timing,
                                     regret experiments (run_baton_r1.py)

Outputs: tables/tab_*.tex and tables/macros.tex
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "svrpspd_wdro" / "results"
OUT = HERE / "tables"
OUT.mkdir(exist_ok=True)
PLACEHOLDER = r"\emph{(pending)}"

macros: dict[str, str] = {}


def _read(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def _pct(x, nd=1):
    return f"{x:.{nd}f}\\%"


def _write(name, content):
    (OUT / name).write_text(content)
    print(f"  wrote tables/{name}")


def _wilcox(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return sps.wilcoxon(d, alternative="greater").pvalue


def _pfmt(p):
    if p >= 0.01:
        return f"$p = {p:.2f}$"
    exp = int(np.floor(np.log10(p)))
    return rf"$p \le 10^{{{exp + 1}}}$"


GATE_DISP = {"Det": r"\textsc{Det}", "SAA": r"\textsc{SAA}",
             "WDRO": r"\textsc{WDRO}", "Gounaris": r"\textsc{Rob-G}",
             "Cui": r"\textsc{Rob-BS}", "MDRO": r"\textsc{M-DRO}"}
GATES = ["Det", "SAA", "WDRO", "Gounaris", "Cui", "MDRO"]


# ═══════════════════════════════════════════════════════════════════════════
# Table 1 — grand comparison (the centrepiece)
# ═══════════════════════════════════════════════════════════════════════════
grand = _read("results_grand_dethloff.csv")
COMPETITORS = ["pi1", "pi2", "pi3", "rollout", "ro_theta", "restock",
               "fb_tau", "thr_k", "thr2", "dp_n", "dp3_n"]
BOOT = np.random.default_rng(20260926)


def _boot_ci(d, n=10_000):
    # fresh generator per call: the same data always give the same
    # interval, whether computed for a table or for an inline macro
    d = np.asarray(d, float)
    idx = np.random.default_rng(20260926).integers(0, len(d), (n, len(d)))
    m = d[idx].mean(axis=1)
    return np.quantile(m, 0.025), np.quantile(m, 0.975)


def _rank_biserial(d):
    d = np.asarray(d, float)
    d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return 0.0
    r = sps.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def _wilcox2(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    return float(sps.wilcoxon(d).pvalue)


def _holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


IMPL = ["pi1", "pi2", "pi3", "rollout", "ro_theta", "restock", "fb_tau",
        "thr_k", "thr2", "dp_n", "dp3_n", "v2_lsm", "v2_act", "v2_cf"]


def _best_impl(d):
    """Implementable policy with the highest mean saving in frame d."""
    return max((l for l in IMPL if f"{l}_saving" in d),
               key=lambda l: d[f"{l}_saving"].mean())


if grand is not None:
    cols = [("pi3", r"$\pi_3$"), ("ro_theta", r"roll.-$\theta$"),
            ("restock", "restock"), ("fb_tau", "thr."), ("thr_k", r"thr.-$k$"),
            ("thr2", "two-lever"), ("dp3_n", r"DP$^3_N$"),
            ("v2_lsm", r"\textsc{Baton-ho}"), ("v2_act", r"\textsc{Baton}"),
            ("v2_cf", r"\textsc{Baton-cf}"),
            ("dp_xl", r"DP$_{50\mathrm{k}}$"),
            ("dp_xl3", r"DP$^3_{50\mathrm{k}}$"), ("oracle", "oracle-ho"),
            ("oracle3", r"oracle$^3$")]
    rows = []
    for g in GATES:
        s_ = grand[grand.Plan == g]
        best = _best_impl(s_)
        bv = f"{s_[f'{best}_saving'].mean():.1f}"
        cells = [GATE_DISP[g]]
        for lbl, _ in cols:
            v = s_[f"{lbl}_saving"].mean()
            cell = f"{v:.1f}"
            if lbl in IMPL and cell == bv:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        rows.append(" & ".join(cells) + r" \\")
    tab = r"""\begin{table}[t]
\caption{Expected-recourse saving over the reactive policy (\%, mean over
the 40 Dethloff instances) for each planning gate and execution policy
under the three-class fleet cost model; \emph{higher is better}; the
best implementable policy of each row is in bold. thr.\ is the
label-corrected tuned threshold, thr.-$k$ the position-dependent
threshold, roll.-$\theta$ the cost-scaled rollout, two-lever a tuned
handoff threshold combined with a tuned depot-return trigger that may
fire repeatedly, and DP$^3_N$ the three-action plug-in dynamic program
fitted on the same $10^3$ training days as \textsc{Baton}. $\pi_1$,
$\pi_2$ and the plain rollout are dominated by $\pi_3$ and
roll.-$\theta$ on every gate and are omitted; the equal-data
handoff-only program DP$_N$ is discussed in Section~\ref{sec:learning}. \textsc{Baton-cf}, the
state-conditional variant of Section~\ref{sec:cf}, was introduced during
the revision after the city results and is reported for completeness.
The last four columns are reference points, not competitors:
DP$_{50\mathrm{k}}$ and DP$^3_{50\mathrm{k}}$ are high-data plug-in
programs in the $(k, W_k)$ state for the handoff-only and the
three-action problem (fifty times the training data); oracle-ho is the
clairvoyant restricted to the handoff lever, which \textsc{Baton} may
therefore exceed; oracle$^3$ is the clairvoyant with all three actions,
a lower bound on the cost of every policy.}
\label{tab:grand}
\centering
\footnotesize
\setlength{\tabcolsep}{2.2pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{l ccccccc ccc cccc}
\toprule
& \multicolumn{7}{c}{published / tuned competitors}
& \multicolumn{3}{c}{this paper} & \multicolumn{4}{c}{reference points} \\
\cmidrule(lr){2-8}\cmidrule(lr){9-11}\cmidrule(lr){12-15}
Gate & """ + " & ".join(h for _, h in cols) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    _write("tab_grand.tex", tab)

    macros["batonGrandBest"] = _pct(
        max(grand[grand.Plan == g]["v2_act_saving"].mean() for g in GATES))
    for g in GATES:
        s_ = grand[grand.Plan == g]
        macros[f"sv{g}Baton"] = _pct(s_.v2_act_saving.mean())
        macros[f"sv{g}Ho"] = _pct(s_.v2_lsm_saving.mean())
        macros[f"sv{g}Orc"] = _pct(s_.oracle_saving.mean())
        macros[f"sv{g}Thresh"] = _pct(s_.fb_tau_saving.mean())
        macros[f"sv{g}ThrK"] = _pct(s_.thr_k_saving.mean())
        macros[f"sv{g}RoTheta"] = _pct(s_.ro_theta_saving.mean())
        macros[f"sv{g}DpThree"] = _pct(s_.dp_xl3_saving.mean())
        macros[f"sv{g}Cf"] = _pct(s_.v2_cf_saving.mean())
        macros[f"sv{g}DpN"] = _pct(s_.dp_n_saving.mean())
        macros[f"sv{g}Two"] = _pct(s_.thr2_saving.mean())
        macros[f"sv{g}DpThreeN"] = _pct(s_.dp3_n_saving.mean())
        macros[f"sv{g}OrcThree"] = _pct(s_.oracle3_saving.mean())
    ratio = [grand[grand.Plan == g].v2_act_saving.mean() /
             grand[grand.Plan == g].dp_xl3_saving.mean() for g in GATES]
    macros["ratioDpThreeLo"] = f"{100 * min(ratio):.0f}\\%"
    macros["ratioDpThreeHi"] = f"{100 * max(ratio):.0f}\\%"
    RATIOS = {f"Dethloff {g}": r for g, r in zip(GATES, ratio)}
    ro3 = [grand[grand.Plan == g].v2_act_saving.mean() /
           grand[grand.Plan == g].oracle3_saving.mean() for g in GATES]
    macros["ratioOrcThreeLo"] = f"{100 * min(ro3):.0f}\\%"
    macros["ratioOrcThreeHi"] = f"{100 * max(ro3):.0f}\\%"
    cf_top = sum(grand[grand.Plan == g].v2_cf_saving.mean() >
                 grand[grand.Plan == g].v2_act_saving.mean() for g in GATES)
    macros["nCfAbove"] = ["none", "one", "two", "three", "four", "five", "all"][cf_top]
    cfg_ = [round(grand[grand.Plan == g].v2_cf_saving.mean(), 1) -
            round(grand[grand.Plan == g].v2_act_saving.mean(), 1) for g in GATES]
    macros["cfGapLo"] = f"{min(cfg_):.1f}"
    macros["cfGapHi"] = f"{max(cfg_):.1f}"
    d_ = grand[grand.Plan == "Det"]
    macros["detActGain"] = f"{d_.v2_act_saving.mean() - d_.v2_lsm_saving.mean():.1f}"
    b_top = sum(_best_impl(grand[grand.Plan == g]) in ("v2_act", "v2_cf")
                for g in GATES)
    macros["nBatonTop"] = ["zero", "one", "two", "three", "four", "five",
                           "six"][b_top]
    two_gap = [round(grand[grand.Plan == g].v2_act_saving.mean(), 1) -
               round(grand[grand.Plan == g].thr2_saving.mean(), 1) for g in GATES]
    macros["twoGapLo"] = f"{min(two_gap):.1f}"
    macros["twoGapHi"] = f"{max(two_gap):.1f}"
    orc_beat = sum(grand[grand.Plan == g].v2_act_saving.mean() >
                   grand[grand.Plan == g].oracle_saving.mean() for g in GATES)
    macros["nOrcBeat"] = ["zero", "one", "two", "three", "four", "five",
                          "six"][orc_beat]
    # standby pool the policies would need (95% of days), pay-per-use model
    macros["poolBatonMed"] = f"{grand.v2_act_S.median():.0f}"
    macros["poolBatonMax"] = f"{grand.v2_act_S.max():.0f}"
    macros["poolHoMed"] = f"{grand.v2_lsm_S.median():.0f}"
    macros["poolHoMax"] = f"{grand.v2_lsm_S.max():.0f}"
    for g in GATES:
        s_ = grand[grand.Plan == g]
        macros[f"pool{g}Med"] = f"{s_.v2_act_S.median():.1f}".replace(".0", "")
        macros[f"pool{g}Max"] = f"{s_.v2_act_S.max():.0f}"

    # ── paired statistics, instance = experimental unit, per gate ──
    body, pvals, cells = [], [], []
    for g in GATES:
        s_ = grand[grand.Plan == g]
        best = max(COMPETITORS, key=lambda l: s_[f"{l}_saving"].mean())
        row = [g, best]
        for ref in (best, "v2_lsm", "dp_xl3"):
            d = (s_["v2_act_saving"] - s_[f"{ref}_saving"]).to_numpy()
            lo, hi = _boot_ci(d)
            wins = int((d > 1e-9).sum())
            ties = int((np.abs(d) <= 1e-9).sum())
            row.append((d.mean(), lo, hi, wins, ties, _rank_biserial(d)))
            pvals.append(_wilcox2(d))
        cells.append(row)
    adj = _holm(pvals)
    NICE = {"pi1": r"$\pi_1$", "pi2": r"$\pi_2$", "pi3": r"$\pi_3$",
            "rollout": "roll.", "ro_theta": r"roll.-$\theta$",
            "restock": "restock", "fb_tau": "thr.", "thr_k": r"thr.-$k$",
            "thr2": "two-lever", "dp_n": r"DP$_N$", "dp3_n": r"DP$^3_N$"}
    for i, row in enumerate(cells):
        out = [GATE_DISP[row[0]], NICE[row[1]]]
        for j in range(3):
            mu, lo, hi, w, t, r = row[2 + j]
            p = adj[3 * i + j]
            ptxt = (r"$<\!10^{-4}$" if p < 1e-4 else
                    r"$<\!10^{-3}$" if p < 1e-3 else f"{p:.3f}")
            out.append(f"{mu:+.1f} [{lo:+.1f}, {hi:+.1f}] & {w} & {r:+.2f} & {ptxt}")
        body.append(" & ".join(out) + r" \\")
    tab = r"""\begin{table}[t]
\caption{Paired comparison of \textsc{Baton} with, in turn, the strongest
competitor on each gate (the implementable policy other than the
\textsc{Baton} variants with the highest mean saving in
Table~\ref{tab:grand}), its handoff-only restriction, and the high-data
three-action reference program. The experimental unit is the
instance ($n = 40$ per gate; the routes of a plan share their test days
and are aggregated within the plan). $\Delta$: mean difference in saving
(percentage points; positive favours \textsc{Baton}) with a 95\%
bootstrap confidence interval; W: instances on which \textsc{Baton} is
better; $r$: matched-pairs rank-biserial effect size; $p$: two-sided
Wilcoxon signed-rank test, Holm-adjusted over the 18 comparisons.}
\label{tab:stats}
\centering
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{l l cccc cccc cccc}
\toprule
& & \multicolumn{4}{c}{vs.\ strongest competitor}
& \multicolumn{4}{c}{vs.\ \textsc{Baton-ho}}
& \multicolumn{4}{c}{vs.\ DP$^3_{50\mathrm{k}}$ (reference)} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}
Gate & competitor & $\Delta$ [95\% CI] & W & $r$ & $p$
& $\Delta$ [95\% CI] & W & $r$ & $p$ & $\Delta$ [95\% CI] & W & $r$ & $p$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    _write("tab_stats.tex", tab)
    for i, row in enumerate(cells):
        g = row[0]
        mu, lo, hi, w, t, r = row[2]
        macros[f"stat{g}Comp"] = NICE[row[1]]
        macros[f"stat{g}Delta"] = f"{mu:+.1f}"
        macros[f"stat{g}CI"] = f"[{lo:+.1f}, {hi:+.1f}]"
        macros[f"stat{g}Win"] = str(w)
        macros[f"stat{g}P"] = f"{adj[3 * i]:.2f}"
        mu, lo, hi, w, t, r = row[4]
        macros[f"stat{g}DpDelta"] = f"{mu:+.1f}"
    cons = [c[2] for c in cells if c[0] != "Det"]
    macros["statConsDeltaMin"] = f"{min(c[0] for c in cons):.1f}"
    macros["statConsDeltaMax"] = f"{max(c[0] for c in cons):.1f}"
    macros["statConsWinMin"] = f"{min(c[3] for c in cons)}"
    macros["statConsPmax"] = (r"$p_{\mathrm{Holm}} < 10^{-3}$"
                              if max(adj[3:18:3]) < 1e-3 else
                              f"$p_{{\\mathrm{{Holm}}}} \\le {max(adj[3:18:3]):.3f}$")
    dpd = [c[4][0] for c in cells]
    macros["statDpDeltaMin"] = f"{min(dpd):.1f}"
    macros["statDpDeltaMax"] = f"{max(dpd):.1f}"
    thk = [round(grand[grand.Plan == g].v2_lsm_saving.mean(), 1) -
           round(grand[grand.Plan == g].thr_k_saving.mean(), 1) for g in GATES]
    macros["thrkHoGapMin"] = f"{min(thk):.1f}"
    macros["thrkHoGapMax"] = f"{max(thk):.1f}"
    comp = [c[2] for c in cells]
    macros["statCompDeltaMin"] = f"{min(c[0] for c in comp):.1f}"
    macros["statCompDeltaMax"] = f"{max(c[0] for c in comp):.1f}"
    macros["statCompLoMin"] = f"{min(c[1] for c in comp):.1f}"
    macros["statCompWinMin"] = f"{min(c[3] for c in comp)}"
    macros["statPmax"] = (r"$p_{\mathrm{Holm}} < 10^{-4}$"
                          if max(adj[0::3]) < 1e-4 else f"$p_{{\\mathrm{{Holm}}}} \\le {max(adj[0::3]):.3f}$")

    # ── tail risk of the daily plan bill ──
    body = []
    for g in GATES:
        s_ = grand[grand.Plan == g]
        base = s_.none_cvar95.mean()
        def red(l):
            return 100 * (1 - s_[f"{l}_cvar95"].mean() / base)
        TL = ("fb_tau", "v2_lsm", "v2_act")
        rv = {l: red(l) for l in TL}
        pv = {l: 100 * s_[f"{l}_pem"].mean() for l in TL}
        bc, bp = max(rv, key=rv.get), min(pv, key=pv.get)
        c1 = [rf"\textbf{{{rv[l]:.1f}}}" if l == bc else f"{rv[l]:.1f}" for l in TL]
        c2 = [rf"\textbf{{{pv[l]:.1f}}}" if abs(pv[l] - pv[bp]) < 0.05 else f"{pv[l]:.1f}" for l in TL]
        body.append(f"{GATE_DISP[g]} & {base:.1f} & " + " & ".join(c1) +
                    f" & {100 * s_.none_pem.mean():.1f} & " + " & ".join(c2) + r" \\")
        if g == "Det":
            macros["tailDetBaton"] = _pct(rv["v2_act"])
            macros["tailDetThr"] = _pct(rv["fb_tau"])
            macros["tailDetHo"] = _pct(rv["v2_lsm"])
    tab = r"""\begin{table}[t]
\caption{Tail risk of the daily recourse bill of a plan (40 Dethloff
instances per gate, 2{,}000 test days each). CVaR$_{95}$: mean of the
worst 5\% of daily plan bills; the reactive column gives its level in
currency units, the others its reduction (\%, higher is better).
$P(\mathrm{emg})$: share of days on which at least one route of the plan
suffers an emergency (\%, lower is better). Best value of each group in
bold.}
\label{tab:tail}
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l c ccc cccc}
\toprule
& \multicolumn{4}{c}{CVaR$_{95}$ of daily plan bill} &
\multicolumn{4}{c}{$P(\mathrm{emg})$ per day (\%)} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Gate & reactive & thr. & \textsc{Baton-ho} & \textsc{Baton} &
reactive & thr. & \textsc{Baton-ho} & \textsc{Baton} \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_tail.tex", tab)
    red_all = [100 * (1 - grand[grand.Plan == g].v2_act_cvar95.mean() /
                      grand[grand.Plan == g].none_cvar95.mean()) for g in GATES]
    macros["tailBatonLo"] = _pct(min(red_all))
    macros["tailBatonHi"] = _pct(max(red_all))


# ═══════════════════════════════════════════════════════════════════════════
# Table 2 — large-scale benchmarks (Salhi–Nagy + city, shops & uniform)
# ═══════════════════════════════════════════════════════════════════════════
sn = _read("results_salhinagy_eval.csv")
city = _read("results_city_eval.csv")
cityu = _read("results_cityuniform_eval.csv")
zp25 = _read("results_cityzp25_eval.csv")
zp50 = _read("results_cityzp50_eval.csv")
if sn is not None and city is not None:
    LCOLS = ["restock", "fb_tau", "thr_k", "thr2", "dp_n", "dp3_n", "v2_lsm", "v2_act",
             "v2_cf", "dp_xl", "dp_xl3", "oracle", "oracle3"]
    LCOMP = ["restock", "fb_tau", "thr_k", "thr2", "dp_n", "dp3_n", "pi3", "ro_theta"]

    def _delta(d):
        """BATON minus the strongest non-BATON competitor of the row, per
        instance, with a 95% bootstrap interval."""
        best = max(LCOMP, key=lambda l: d[f"{l}_saving"].mean())
        dd = (d.v2_act_saving - d[f"{best}_saving"]).to_numpy()
        lo, hi = _boot_ci(dd)
        return best, dd.mean(), lo, hi

    LNICE = {"restock": "restock", "fb_tau": "thr.", "thr_k": r"thr.-$k$",
             "thr2": "two-lever", "dp3_n": r"DP$^3_N$", "pi3": r"$\pi_3$",
             "ro_theta": r"roll.-$\theta$", "dp_n": r"DP$_N$"}

    def _row(name, d):
        vals = []
        best = _best_impl(d)
        bv = f"{d[f'{best}_saving'].mean():.1f}"
        for l in LCOLS:
            v = f"{d[f'{l}_saving'].mean():.1f}"
            vals.append(rf"\textbf{{{v}}}" if (l in IMPL and v == bv) else v)
        bc, mu, lo, hi = _delta(d)
        return (f"{name} & {len(d)} & " + " & ".join(vals) +
                f" & {mu:+.1f} [{lo:+.1f}, {hi:+.1f}] ({LNICE[bc]})" + r" \\")
    LROWS = [(r"Salhi--Nagy", sn, "Salhi"), (r"City, real shops", city, "City")]
    if cityu is not None:
        LROWS.append((r"City, uniform", cityu, "CityU"))
    if zp25 is not None:
        LROWS.append((r"City, 25\% deliver-only", zp25, "ZpTwentyFive"))
    if zp50 is not None:
        LROWS.append((r"City, 50\% deliver-only", zp50, "ZpFifty"))
    csaa = _read("results_city_saa_eval.csv")
    if csaa is not None:
        # SAA plans on the city instances carry so much slack that the
        # reactive policy costs almost nothing: reported in the text only
        macros["citySaaReact"] = f"{csaa.none_rec.mean():.2f}"
        macros["citySaaK"] = f"{csaa.K_routes.mean():.1f}"
        macros["cityDetK"] = f"{city.K_routes.mean():.1f}"
        macros["cityDetReact"] = f"{city.none_rec.mean():.1f}"
        macros["citySaaPem"] = _pct(100 * csaa.none_pem.mean(), 2)
    rows = [_row(nm, d) for nm, d, _ in LROWS]
    for nm, d, key in LROWS:
        bc, mu, lo, hi = _delta(d)
        macros[f"lg{key}Delta"] = f"{mu:+.1f}"
        macros[f"lg{key}DeltaCI"] = f"[{lo:+.1f}, {hi:+.1f}]"
        macros[f"lg{key}Comp"] = LNICE[bc]
        macros[f"lg{key}Baton"] = _pct(d.v2_act_saving.mean())
        macros[f"lg{key}CompSv"] = _pct(d[f"{bc}_saving"].mean())
        macros[f"lg{key}Ratio"] = f"{100 * d.v2_act_saving.mean() / d.dp_xl3_saving.mean():.0f}\\%"
        macros[f"lg{key}RatioOrcThree"] = f"{100 * d.v2_act_saving.mean() / d.oracle3_saving.mean():.0f}\\%"
        RATIOS[nm] = d.v2_act_saving.mean() / d.dp_xl3_saving.mean()
    tab = r"""\begin{sidewaystable}
\caption{Large-scale benchmarks under the fleet cost model
(\textsc{Det}-gate plans unless stated; saving \% vs.\ reactive,
\emph{higher is better}; the best implementable policy of each row is in
bold; columns as in Table~\ref{tab:grand}, the four columns after
\textsc{Baton-cf} being reference points). Salhi--Nagy instances carry
50--199 customers; the city instances (100--400 customers) place
customers at real OSM shop locations on the road networks of Ho Chi Minh
City, Hanoi, New York, Paris and Shanghai. The uniform twins use the
same cities and demands with uniformly scattered customers; the
deliver-only twins set the pickup of a random 25\% or 50\% of the
customers to zero and are re-planned. Last column: \textsc{Baton} minus
the strongest competitor of the row (named), mean over instances in
percentage points with a 95\% paired bootstrap interval.}
\label{tab:large}
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{l r cccccc ccc cccc l}
\toprule
Benchmark & $n$ & restock & thr. & thr.-$k$ & two-lever & DP$_N$ & DP$^3_N$ &
\textsc{Baton-ho} & \textsc{Baton} & \textsc{Baton-cf} &
DP$_{50\mathrm{k}}$ & DP$^3_{50\mathrm{k}}$ & oracle-ho & oracle$^3$ &
$\Delta$ vs.\ best competitor \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{sidewaystable}
"""
    _write("tab_large.tex", tab)
    macros["svSalhiBaton"] = _pct(sn.v2_act_saving.mean())
    macros["svSalhiHo"] = _pct(sn.v2_lsm_saving.mean())
    macros["svSalhiOrc"] = _pct(sn.oracle_saving.mean())
    macros["svSalhiDpThree"] = _pct(sn.dp_xl3_saving.mean())
    macros["svCityBaton"] = _pct(city.v2_act_saving.mean())
    macros["svCityHo"] = _pct(city.v2_lsm_saving.mean())
    macros["svCityDp"] = _pct(city.dp_xl_saving.mean())
    macros["svCityCf"] = _pct(city.v2_cf_saving.mean())
    macros["svCityOrc"] = _pct(city.oracle_saving.mean())
    macros["nCity"] = str(len(city))
    macros["cityActShare"] = _pct(100 * city.act_routes.sum() / city.n_routes.sum(), 0)
    macros["salhiActShare"] = _pct(100 * sn.act_routes.sum() / sn.n_routes.sum(), 0)
    if cityu is not None:
        m = city.merge(cityu, on="Instance", suffixes=("_s", "_u"))
        macros["shopTravelSaving"] = _pct(
            (100 * (m.Travel_km_u - m.Travel_km_s) / m.Travel_km_u).mean())
    if zp50 is not None:
        macros["svZpFiftyBaton"] = _pct(zp50.v2_act_saving.mean())
        macros["svZpFiftyThr"] = _pct(zp50.fb_tau_saving.mean())
        macros["svZpTwentyFiveBaton"] = _pct(zp25.v2_act_saving.mean())


# ═══════════════════════════════════════════════════════════════════════════
# Table 3 — cost-parameter sensitivity
# ═══════════════════════════════════════════════════════════════════════════
CS_DISP = [
    ("F_emg_25",     r"cheap emergencies ($F_{\mathrm{emg}}{=}25$)"),
    ("F_emg_60",     r"dear emergencies ($F_{\mathrm{emg}}{=}60$)"),
    ("F_standby_10", r"cheap standby ($F_{\mathrm{sb}}{=}10$)"),
    ("F_standby_35", r"dear standby ($F_{\mathrm{sb}}{=}35$)"),
    ("p_late_0_5",   r"low SLA price ($p_{\mathrm{late}}{=}0.5$)"),
    ("p_late_3_0",   r"high SLA price ($p_{\mathrm{late}}{=}3$)"),
    ("s_emg_1_5",    r"mild surge ($s_{\mathrm{emg}}{=}1.5$)"),
    ("s_emg_4_0",    r"heavy surge ($s_{\mathrm{emg}}{=}4$)"),
    ("F_standby_60", r"standby above emergency ($F_{\mathrm{sb}}{=}60$)"),
    ("F_return_5",   r"depot-return fee 5"),
    ("F_return_10",  r"depot-return fee 10"),
    ("F_return_20",  r"depot-return fee 20"),
    ("F_return_30",  r"depot-return fee 30"),
]
cs_files = {Path(f).stem.replace("results_costsens_", ""): pd.read_csv(f)
            for f in glob.glob(str(RES / "results_costsens_*.csv"))}
if cs_files and grand is not None:
    ref = next(iter(cs_files.values()))
    base = grand[grand.Plan.isin(["Det", "SAA"]) &
                 grand.Instance.isin(ref.Instance)]
    rows = [("baseline", base)] + [(disp, cs_files[tag])
                                   for tag, disp in CS_DISP if tag in cs_files]
    body = []
    for disp, d in rows:
        two = f"{d.thr2_saving.mean():.1f}" if "thr2_saving" in d else "--"
        if disp.startswith("depot-return fee 5"):
            body.append(r"\midrule")
        ho = f"{d.v2_lsm_saving.mean():.1f}"
        if ho == f"{d.v2_act_saving.mean():.1f}":
            ho = rf"\textbf{{{ho}}}"
        body.append(f"{disp} & {d.restock_saving.mean():.1f} & "
                    f"{d.fb_tau_saving.mean():.1f} & {two} & "
                    f"{ho} & "
                    rf"\textbf{{{d.v2_act_saving.mean():.1f}}} & "
                    f"{d.oracle_saving.mean():.1f} \\\\")
    tab = r"""\begin{table}[t]
\caption{Sensitivity of the recourse saving (\%, \emph{higher is
better}; best non-clairvoyant value per row in bold) to the fleet-economics
parameters, one factor at a time around the defaults (12 Dethloff
instances, \textsc{Det} and \textsc{SAA} gates; the baseline row is
computed on the same 12 instances). two-lever: tuned handoff threshold
plus tuned repeatable depot-return trigger. In the
$F_{\mathrm{sb}} = 60$ row the standby rate exceeds the emergency price
at late stops, so Inequality~\eqref{eq:prices} fails there. The last
rows add a fixed fee (dwell, loading-dock time) to every depot return on
top of its detour, transfer and lateness costs.}
\label{tab:costsens}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{l cccccc}
\toprule
Configuration & restock & threshold & two-lever & \textsc{Baton-ho} &
\textsc{Baton} & oracle-ho \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    _write("tab_costsens.tex", tab)
    if "F_standby_35" in cs_files:
        d = cs_files["F_standby_35"]
        macros["dearSbThresh"] = _pct(d.fb_tau_saving.mean())
        macros["dearSbBaton"] = _pct(d.v2_act_saving.mean())
        macros["dearSbOrc"] = _pct(d.oracle_saving.mean())
    fees = [(0.0, base)] + [(float(t.split("_")[-1]), cs_files[t])
                            for t in ("F_return_5", "F_return_10", "F_return_20",
                                      "F_return_30") if t in cs_files]
    if len(fees) > 1:
        # break-even fee: the depot return stops adding more than 0.1 points
        # over the handoff-only menu (deployment selection then falls back
        # to it, so the gap never turns negative)
        gap = [(f, d.v2_act_saving.mean() - d.v2_lsm_saving.mean() - 0.1)
               for f, d in fees]
        be = None
        for (f0, g0), (f1, g1) in zip(gap, gap[1:]):
            if g0 > 0 >= g1:
                be = f0 + (f1 - f0) * g0 / (g0 - g1)
                break
        macros["feeBreakEven"] = (f"{be:.0f}" if be is not None
                                  else rf"above {gap[-1][0]:.0f}")
        macros["feeGapZero"] = f"{gap[0][1] + 0.1:.1f}"
        macros["feeGapMax"] = f"{gap[-1][1] + 0.1:.1f}"
        macros["feeMax"] = f"{gap[-1][0]:.0f}"
        macros["feeBatonMax"] = _pct(fees[-1][1].v2_act_saving.mean())
        macros["feeHoMax"] = _pct(fees[-1][1].v2_lsm_saving.mean())
    if "F_standby_60" in cs_files:
        d = cs_files["F_standby_60"]
        macros["sbSixtyBaton"] = _pct(d.v2_act_saving.mean())
        macros["sbSixtyThresh"] = _pct(d.fb_tau_saving.mean())
        macros["sbSixtyHo"] = _pct(d.v2_lsm_saving.mean())


# ═══════════════════════════════════════════════════════════════════════════
# Table 4 — RL head-to-head
# ═══════════════════════════════════════════════════════════════════════════
def _rl_table():
    import sys
    sys.path.insert(0, str(HERE.parent.parent / "svrpspd_wdro"))
    sys.path.insert(0, str(HERE.parent.parent / "svrpspd_wdro" / "scripts"))
    try:
        from core.costs import (fit_lsm_general, simulate_v2_general,
                                fit_lsm_actions, simulate_actions,
                                LastMileCosts, restock_schedule)
        from dethloff_runner import parse_dethloff
    except Exception:
        return
    bundle = RES / "rl_bundle.npz"
    first = RES / "rl_results.json"
    if not (bundle.exists() and first.exists()):
        return
    z = np.load(bundle, allow_pickle=True)
    n = int(z["n_routes"][0])
    v2, v3, tr_s = [], [], []
    for i in range(n):
        g_tr = z[f"r{i}_g_train"].astype(float)
        g_te = z[f"r{i}_g_test"].astype(float)
        H, E = z[f"r{i}_H"].astype(float), z[f"r{i}_E"].astype(float)
        R = z[f"r{i}_R"].astype(float)
        B = float(z[f"r{i}_B"][0])
        import time as _t
        t0 = _t.perf_counter()
        cm = fit_lsm_general(g_tr, B, H, E)
        am = fit_lsm_actions(g_tr, B, H, E, R)
        use = (simulate_actions(g_tr, B, H, E, R, am)["mean_cost"] <=
               simulate_v2_general(g_tr, B, H, E, cm)["mean_cost"])
        tr_s.append(_t.perf_counter() - t0)
        c_ho = simulate_v2_general(g_te, B, H, E, cm)["mean_cost"]
        v2.append(c_ho)
        v3.append(simulate_actions(g_te, B, H, E, R, am)["mean_cost"] if use else c_ho)
    v2, v3 = np.array(v2), np.array(v3)
    ref = json.load(open(first))
    if len(ref) != n:
        print("  rl results do not match the bundle yet; skipping tab_rl")
        return
    re = np.array([r["reactive_cost"] for r in ref])

    runs = [("RL, 40 epochs", first)]
    for f in sorted(RES.glob("rl_strong_s*.json")):
        runs.append((f"RL, 150 epochs, seed {f.stem[-1]}", f))
    body, best_rl = [], None
    for name, f in runs:
        rl = np.array([r["rl_cost"] for r in json.load(open(f))])
        sv = 100 * (re.sum() - rl.sum()) / re.sum()
        if best_rl is None or sv > best_rl[0]:
            best_rl = (sv, rl)
        body.append(f"{name} & {sv:.1f} \\\\")
    sv_v2 = 100 * (re.sum() - v2.sum()) / re.sum()
    sv_v3 = 100 * (re.sum() - v3.sum()) / re.sum()
    nb = int((v2 < best_rl[1] - 1e-9).sum())
    # instance-clustered comparison: routes of one instance share test days
    inst = np.array([str(z[f"r{i}_inst"][0]) for i in range(n)])
    ids = sorted(set(inst))
    d_i = np.array([100 * (best_rl[1][inst == u].sum() - v2[inst == u].sum()) /
                    max(re[inst == u].sum(), 1e-9) for u in ids])
    p = _wilcox(d_i, np.zeros_like(d_i))
    lo, hi = _boot_ci(d_i)
    macros["rlNInst"] = str(len(ids))
    macros["rlInstWin"] = f"{int((d_i > 1e-9).sum())}/{len(ids)}"
    macros["rlInstDelta"] = f"{d_i.mean():.1f}"
    macros["rlInstCI"] = f"[{lo:.1f}, {hi:.1f}]"
    tab = r"""\begin{table}[t]
\caption{Reinforcement-learning baseline (re-implementation of the policy
architecture of Iklassov et al., 2024) versus \textsc{Baton} on the
identical """ + str(n) + r""" routes and out-of-sample test days; saving \% vs.\ the
reactive policy, \emph{higher is better}. The learned policy acts on the
handoff lever only, so \textsc{Baton-ho} is the like-for-like
comparison; the full \textsc{Baton} row adds the depot return.}
\label{tab:rl}
\centering
\begin{tabular}{l c}
\toprule
Policy & saving vs.\ reactive (\%) \\
\midrule
""" + "\n".join(body) + rf"""
\midrule
\textsc{{Baton-ho}} (handoff only) & \textbf{{{sv_v2:.1f}}} \\
\textsc{{Baton}} (handoff and depot return) & \textbf{{{sv_v3:.1f}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    _write("tab_rl.tex", tab)
    macros["rlBest"] = _pct(best_rl[0])
    macros["rlBatonHo"] = _pct(sv_v2)
    macros["rlBaton"] = _pct(sv_v3)
    macros["rlWinN"] = f"{nb}/{n}"
    macros["nRL"] = str(n)
    rl_t = [json.load(open(f))[0].get("train_s") for _, f in runs]
    rl_t = [t for t in rl_t if t]
    if rl_t:
        macros["rlTrainMin"] = f"{min(rl_t) / 60:.0f}"
        macros["rlTrainMax"] = f"{max(rl_t) / 60:.0f}"
    macros["rlBatonFitS"] = f"{sum(tr_s):.1f}"
    macros["rlWinP"] = _pfmt(p)


_rl_table()


# ═══════════════════════════════════════════════════════════════════════════
# Table 5 — synthetic structural scenarios (label defect isolated)
# ═══════════════════════════════════════════════════════════════════════════
syn = _read("results_otr2_synthetic.csv")
if syn is not None:
    def _ms(scen, col):
        return syn[syn.scenario == scen][col].mean()
    sya = _read("r1/synthetic_actions.csv")
    rows = []
    for scen, disp in (("collect_then_deliver", "collect-then-deliver"),
                       ("milk_run_regime", "milk run, regime switching"),
                       ("high_cost_ratio", r"milk run, $C/\omega = 20$")):
        full = (sya[sya.scenario == scen].baton_saving.mean()
                if sya is not None else float("nan"))
        rows.append(f"{disp} & {_ms(scen, 'v1_tun_saving'):.1f} & "
                    f"{_ms(scen, 'fb_tun_saving'):.1f} & "
                    rf"{_ms(scen, 'v2_lsm_saving'):.1f} & \textbf{{{full:.1f}}} \\")
    tab = r"""\begin{table}[t]
\caption{Structural synthetic scenarios (five seeds, $1.2\times10^4$ test
routes each, flat prices $\omega_F = 1$, $C_{\mathrm{fail}}/\omega_F$
as shown or 5): saving \% over the reactive policy, \emph{higher is
better}. The synthetic routes have no geometry, so the depot return of
the full \textsc{Baton} is priced flat at half a handoff; the first
three columns use the handoff lever only. On collect-then-deliver
structure the endpoint-labelled predecessor never intervenes.}
\label{tab:synthetic}
\centering
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{l cccc}
\toprule
Scenario & endpoint + $\tau$ & peak + $\tau$ & \textsc{Baton-ho} &
\textsc{Baton} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    _write("tab_synthetic.tex", tab)
    macros["ctdVOne"] = _pct(_ms("collect_then_deliver", "v1_tun_saving"))
    macros["ctdBaton"] = _pct(_ms("collect_then_deliver", "v2_lsm_saving"))
    if sya is not None:
        macros["ctdBatonFull"] = _pct(
            sya[sya.scenario == "collect_then_deliver"].baton_saving.mean())


# ═══════════════════════════════════════════════════════════════════════════
# Revision experiments (svrpspd_wdro/scripts/run_baton_r1.py -> results/r1)
# ═══════════════════════════════════════════════════════════════════════════
def _tex_table(label, caption, colspec, header, body, size=r"\footnotesize",
               sep="3.5pt", env="table"):
    return (rf"\begin{{{env}}}" + ("[t]" if env == "table" else "") + "\n" + rf"\caption{{{caption}}}" + "\n" +
            rf"\label{{{label}}}" + "\n" + r"\centering" + "\n" + size + "\n" +
            rf"\setlength{{\tabcolsep}}{{{sep}}}" + "\n" +
            r"\begin{adjustbox}{max width=\linewidth}" + "\n" +
            rf"\begin{{tabular}}{{{colspec}}}" + "\n" + r"\toprule" + "\n" +
            header + "\n" + r"\midrule" + "\n" + "\n".join(body) + "\n" +
            r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" +
            r"\end{adjustbox}" + "\n" + rf"\end{{{env}}}" + "\n")


# ── dependence + shape ─────────────────────────────────────────────────────
dep = _read("r1/dependence.csv")
shp = _read("r1/shape.csv")
if dep is not None:
    if shp is not None:
        shp = shp.assign(rho=shp.rho.astype(str))
    CFG = [("rho0", r"$\rho = 0$", "0.0"),
           ("rho03", r"$\rho = 0.3$", "0.3"), ("rho06", r"$\rho = 0.6$ (bench.)", "0.6"),
           ("rho09", r"$\rho = 0.9$", "0.9"),
           ("dayfac", r"day factor, $\rho = 0$", "dayfac")]
    DC = ["thr_k", "thr2", "v2_lsm", "v2_act", "v2_cf", "dp_xl3", "oracle3"]
    DCOMP = [l for l in ("fb_tau", "thr_k", "thr2", "dp_n", "dp3_n", "restock",
                         "pi3", "ro_theta") if f"{l}_saving" in dep]
    DNICE = {"restock": "restock", "fb_tau": "thr.", "thr_k": r"thr.-$k$",
             "thr2": "two-lever", "dp3_n": r"DP$^3_N$", "pi3": r"$\pi_3$",
             "ro_theta": r"roll.-$\theta$", "dp_n": r"DP$_N$"}
    DC = [l for l in DC if f"{l}_saving" in dep]
    body = []
    close = []
    for tag, disp, rho in CFG:
        first = True
        for pln in ("Det", "SAA"):
            d = dep[(dep.cfg == tag) & (dep.Plan == pln)]
            if d.empty:
                continue
            best = _best_impl(d)
            bv = f"{d[f'{best}_saving'].mean():.1f}"
            vals = []
            for l in DC:
                v = f"{d[f'{l}_saving'].mean():.1f}"
                vals.append(rf"\textbf{{{v}}}" if (l in IMPL and v == bv) else v)
            bc = max(DCOMP, key=lambda l: d[f"{l}_saving"].mean())
            dd = (d.v2_act_saving - d[f"{bc}_saving"]).to_numpy()
            lo, hi = _boot_ci(dd)
            if lo <= 0:
                close.append((tag, pln, bc, dd.mean(), lo, hi))
            if first and shp is not None and (shp.rho == rho).any():
                sh = shp[shp.rho == rho]
                vr = f"{100 * sh.viol.sum() / max(sh.pairs.sum(), 1):.1f}"
                if "dip_hits_10" in sh:
                    vr += (f" & {100 * sh.dip_hits_10.sum() / max(sh.dip_tests.sum(), 1):.0f}"
                           f" & {100 * sh.dip_hits_25.sum() / max(sh.dip_tests.sum(), 1):.0f}")
                else:
                    vr += " & & "
            else:
                vr = "& &" if not first else "-- & & "
            body.append(f"{disp if first else ''} & \\textsc{{{pln}}} & {d.none_rec.mean():.1f} & " +
                        " & ".join(vals) +
                        f" & {dd.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] ({DNICE[bc]}) & {vr} \\\\")
            first = False
        nm = {"rho0": "RhoZero", "rho03": "RhoThree", "rho06": "RhoSix",
              "rho09": "RhoNine", "dayfac": "Dayfac"}[tag]
        d = dep[dep.cfg == tag]
        macros[f"dep{nm}Baton"] = _pct(d.v2_act_saving.mean())
        for pln in ("Det", "SAA"):
            dd = d[d.Plan == pln]
            k = pln.capitalize()
            macros[f"dep{nm}{k}Baton"] = _pct(dd.v2_act_saving.mean())
            macros[f"dep{nm}{k}Ho"] = _pct(dd.v2_lsm_saving.mean())
            macros[f"dep{nm}{k}Thr"] = _pct(dd.fb_tau_saving.mean())
            macros[f"dep{nm}{k}ThrK"] = _pct(dd.thr_k_saving.mean())
            macros[f"dep{nm}{k}Cf"] = _pct(dd.v2_cf_saving.mean())
            macros[f"dep{nm}{k}DpThree"] = _pct(dd.dp_xl3_saving.mean())
            macros[f"dep{nm}{k}React"] = f"{dd.none_rec.mean():.1f}"
            macros[f"dep{nm}{k}Ratio"] = f"{100 * dd.v2_act_saving.mean() / dd.dp_xl3_saving.mean():.0f}\\%"
            if dd.none_rec.mean() >= 3.0:        # rows with something to save
                RATIOS[f"dependence {tag} {pln}"] = dd.v2_act_saving.mean() / dd.dp_xl3_saving.mean()
            bc = max(DCOMP, key=lambda l: dd[f"{l}_saving"].mean())
            x = (dd.v2_act_saving - dd[f"{bc}_saving"]).to_numpy()
            lo, hi = _boot_ci(x)
            macros[f"dep{nm}{k}Delta"] = f"{x.mean():+.1f}"
            macros[f"dep{nm}{k}DeltaCI"] = f"[{lo:+.1f}, {hi:+.1f}]"
            macros[f"dep{nm}{k}Comp"] = DNICE[bc]
            macros[f"dep{nm}{k}CompSv"] = _pct(dd[f"{bc}_saving"].mean())
    macros["depNClose"] = str(len(close))
    cf_rows = sum(dep[(dep.cfg == t) & (dep.Plan == p_)].v2_cf_saving.mean() >
                  dep[(dep.cfg == t) & (dep.Plan == p_)].v2_act_saving.mean()
                  for t, _, _ in CFG for p_ in ("Det", "SAA"))
    macros["depCfAbove"] = str(cf_rows)
    if shp is not None:
        for rho, nm in (("0.0", "Zero"), ("0.3", "Three"), ("0.6", "Six"),
                        ("0.9", "Nine"), ("dayfac", "Dayfac")):
            sh = shp[shp.rho == rho]
            if sh.empty:
                continue
            macros[f"shapeViol{nm}"] = _pct(100 * sh.viol.sum() / max(sh.pairs.sum(), 1))
            if "dip_hits_10" in sh:
                macros[f"shapePowTen{nm}"] = _pct(100 * sh.dip_hits_10.sum() / max(sh.dip_tests.sum(), 1), 0)
                macros[f"shapePowTwentyFive{nm}"] = _pct(100 * sh.dip_hits_25.sum() / max(sh.dip_tests.sum(), 1), 0)
        if "dip_hits_10" in shp:
            macros["shapePowTenAll"] = _pct(100 * shp.dip_hits_10.sum() / shp.dip_tests.sum(), 0)
            macros["shapePowTwentyFiveAll"] = _pct(100 * shp.dip_hits_25.sum() / shp.dip_tests.sum(), 0)
    hdr = " & ".join({"fb_tau": "thr.", "thr_k": r"thr.-$k$", "thr2": "two-lever",
                      "v2_lsm": r"\textsc{Baton-ho}", "v2_act": r"\textsc{Baton}",
                      "v2_cf": r"\textsc{Baton-cf}", "dp_xl3": r"DP$^3_{50\mathrm{k}}$",
                      "oracle": "oracle-ho", "oracle3": r"oracle$^3$"}[l] for l in DC)
    _write("tab_dependence.tex", _tex_table(
        "tab:dependence",
        r"""Robustness to the demand dependence (40 Dethloff instances,
\textsc{Det} and \textsc{SAA} plans, saving \% vs.\ reactive; plans are
held fixed and policies are re-fitted and re-tested under each demand
law; react.: expected daily recourse cost of the reactive policy per
plan, which shows how much there is to save; best implementable policy
in bold; the last two policy columns are reference points; the tuned
global threshold and the handoff-only oracle are in the replication
files). $\rho$ is
the Gaussian-copula equicorrelation among deliveries and among pickups;
the day-factor law multiplies every demand of a day by a common
lognormal factor (s.d.\ 0.25) on top of independent marginals.
$\Delta$: \textsc{Baton} minus the strongest competitor of the row
(named; \textsc{Baton} variants excluded), mean over instances with a
95\% paired bootstrap interval. Shape test (both gates pooled): viol.\ is
the share of adjacent-bin pairs in which the unconstrained binned
estimate of $\mathbb E[\text{future cost} \mid W_k]$, fitted on
$5\times10^4$ paths per route with 25 bins, \emph{decreases}
significantly (one-sided, 2.5\% level; the rate under independence,
where monotonicity is a theorem, calibrates the test); power: share of
tests that detect a dip of 10\% or 25\% of the mean continuation cost
injected into the middle bin.""",
        "l l c " + "c" * len(DC) + " l ccc",
        r"& & & \multicolumn{" + str(len(DC)) + r"}{c}{saving \%} & & \multicolumn{3}{c}{shape test (\%)} \\ \cmidrule(lr){4-" + str(3 + len(DC)) + r"}\cmidrule(lr){" + str(5 + len(DC)) + "-" + str(7 + len(DC)) + r"}" + "\n" +
        r"Demand law & gate & react. & " + hdr + r" & $\Delta$ [95\% CI] & viol. & pow.$_{10}$ & pow.$_{25}$ \\",
        body, size=r"\footnotesize", sep="3pt", env="sidewaystable"))

# ── exact DP under independent demands ───────────────────────────────────
ex = _read("r1/exact.csv")
if ex is not None:
    body = []
    for pln in ("Det", "SAA"):
        d = ex[ex.Plan == pln]
        if d.empty:
            continue
        en, mn = d.exact_none.sum(), d.none_rec.sum()

        def esv(c):
            return 100 * (en - d[c].sum()) / en

        def msv(l):
            return 100 * (mn - d[f"{l}_rec"].sum()) / mn
        cells = [rf"\textsc{{{pln}}}", str(d.Instance.nunique()), str(len(d)),
                 f"{100 * abs(mn - en) / en:.2f}",
                 f"{esv('exact_ho'):.1f}", f"{esv('exact_3'):.1f}"]
        for l in ("fb_tau", "thr_k", "thr2", "dp3_n", "v2_lsm", "v2_act",
                  "v2_cf", "dp_xl", "dp_xl3"):
            cells.append(f"{msv(l):.1f}")
        def share(a, b):
            return f"{100 * a / b:.0f}" if b >= 1.0 else "--"
        cells += [share(msv('v2_lsm'), esv('exact_ho')),
                  share(msv('v2_act'), esv('exact_3')),
                  share(msv('dp_xl3'), esv('exact_3'))]
        body.append(" & ".join(cells) + r" \\")
        k = pln.capitalize()
        macros[f"ex{k}Ho"] = _pct(esv("exact_ho"))
        macros[f"ex{k}Three"] = _pct(esv("exact_3"))
        macros[f"ex{k}Baton"] = _pct(msv("v2_act"))
        macros[f"ex{k}BatonHo"] = _pct(msv("v2_lsm"))
        macros[f"ex{k}DpThree"] = _pct(msv("dp_xl3"))
        macros[f"ex{k}ShareHo"] = f"{100 * msv('v2_lsm') / esv('exact_ho'):.0f}\\%"
        macros[f"ex{k}Share"] = f"{100 * msv('v2_act') / esv('exact_3'):.0f}\\%"
        macros[f"ex{k}ShareDp"] = f"{100 * msv('dp_xl3') / esv('exact_3'):.0f}\\%"
        macros[f"ex{k}McErr"] = f"{100 * abs(mn - en) / en:.2f}\\%"
    _write("tab_exact.tex", _tex_table(
        "tab:exact",
        r"""Exact optimum under independent demands ($\rho = 0$; 40 Dethloff
instances, all routes of the \textsc{Det} and \textsc{SAA} plans).
The exact values solve the dynamic programs over $(k, W_k)$ by numerical
convolution of the Gamma demand densities on a grid of 1{,}500 load
cells, for the handoff-only menu and for the three-action menu with the
conservative reset. Policies are fitted on $10^3$ days and evaluated on
$2\times10^4$ test days; MC err.: relative difference between the
simulated and the exact reactive cost, a check of the grid. Savings are
\% of the reactive cost, pooled over routes; the last three columns give
\textsc{Baton-ho}'s share of the exact handoff-only saving and
\textsc{Baton}'s and DP$^3_{50\mathrm{k}}$'s share of the exact
three-action saving (--: the exact saving is below one point, so the
share is not meaningful).""",
        "l cc c cc ccccccccc ccc",
        r"& & & MC & \multicolumn{2}{c}{exact optimum} & & & & & & & & & & \multicolumn{3}{c}{share of exact (\%)} \\ \cmidrule(lr){5-6}\cmidrule(lr){16-18}" + "\n" +
        r"Gate & inst. & routes & err.\ (\%) & ho & 3-act. & thr. & thr.-$k$ & two-lever & DP$^3_N$ & \textsc{Baton-ho} & \textsc{Baton} & \textsc{Baton-cf} & DP$_{50\mathrm{k}}$ & DP$^3_{50\mathrm{k}}$ & \textsc{Baton-ho} & \textsc{Baton} & DP$^3_{50\mathrm{k}}$ \\",
        body, size=r"\scriptsize", sep="2.2pt"))

# ── day types ──────────────────────────────────────────────────────────────
dty = _read("r1/daytype.csv")
if dty is not None:
    def _sv(tr, te, l):
        d = dty[(dty.train == tr) & (dty.test == te)]
        return d[f"{l}_rec"].sum(), d["none_rec"].sum()

    def _mixsv(pairs, l):
        c = n_ = 0.0
        for (tr, te), w in pairs:
            a, b = _sv(tr, te, l)
            c += w * a
            n_ += w * b
        return 100 * (n_ - c) / n_
    STRAT = [("pooled fit (day type ignored)", ("mix", "normal"), ("mix", "promo")),
             ("day-type-specific fits", ("normal", "normal"), ("promo", "promo")),
             ("stale fit (normal days only)", ("normal", "normal"), ("normal", "promo"))]
    body = []
    for disp, pn, pp in STRAT:
        cells = [disp]
        for l in ("fb_tau", "v2_act"):
            sn_ = _mixsv([(pn, 1.0)], l)
            sp_ = _mixsv([(pp, 1.0)], l)
            sa_ = _mixsv([(pn, 0.8), (pp, 0.2)], l)
            cells += [f"{sn_:.1f}", f"{sp_:.1f}", f"{sa_:.1f}"]
        body.append(" & ".join(cells) + r" \\")
    def _unit_sv(tr, te, l):
        d = dty[(dty.train == tr) & (dty.test == te)].set_index(["Instance", "Plan"])
        return d[f"{l}_rec"], d["none_rec"]

    def _unit_mix(pairs, l):
        c = n_ = 0.0
        for (tr, te), w in pairs:
            a, b = _unit_sv(tr, te, l)
            c = c + w * a
            n_ = n_ + w * b
        return 100 * (n_ - c) / n_
    aw = _unit_mix([(("normal", "normal"), .8), (("promo", "promo"), .2)], "v2_act")
    po = _unit_mix([(("mix", "normal"), .8), (("mix", "promo"), .2)], "v2_act")
    st_ = _unit_mix([(("normal", "normal"), .8), (("normal", "promo"), .2)], "v2_act")
    for nm, x in (("AwarePooled", (aw - po).to_numpy()), ("StalePooled", (st_ - po).to_numpy())):
        lo, hi = _boot_ci(x)
        macros[f"dt{nm}Delta"] = f"{x.mean():+.1f}"
        macros[f"dt{nm}CI"] = f"[{lo:+.1f}, {hi:+.1f}]"
    if "n_train" in dty:
        macros["dtNPromo"] = f"{dty[dty.train == 'promo'].n_train.mean():.0f}"
        macros["dtNNormal"] = f"{dty[dty.train == 'normal'].n_train.mean():.0f}"
    macros["dtPromoAware"] = _pct(_mixsv([(("promo", "promo"), 1.0)], "v2_act"))
    macros["dtPromoStale"] = _pct(_mixsv([(("normal", "promo"), 1.0)], "v2_act"))
    macros["dtPromoPooled"] = _pct(_mixsv([(("mix", "promo"), 1.0)], "v2_act"))
    macros["dtAllAware"] = _pct(_mixsv([(("normal", "normal"), .8), (("promo", "promo"), .2)], "v2_act"))
    macros["dtAllPooled"] = _pct(_mixsv([(("mix", "normal"), .8), (("mix", "promo"), .2)], "v2_act"))
    _write("tab_daytype.tex", _tex_table(
        "tab:daytype",
        r"""Non-exchangeable days (40 Dethloff instances, \textsc{Det} and
\textsc{SAA} plans; saving \% vs.\ reactive on the same days, pooled
over plans as a ratio of sums, with the ``all'' column weighting normal
and promotion days 0.8/0.2). One day
in five is a promotion day with mean deliveries $\times 1.15$ and mean
pickups $\times 1.35$. Pooled: one fit on a history that mixes both day
types; day-type-specific: separate fits, the day type being known in
advance (promotions are scheduled), each fitted on the days of its type
in the same pooled history (about 800 normal and 200 promotion days), so
that the comparison isolates the day-type information from the sample
size; stale: fitted on the normal days of that history only and applied
unchanged on promotion days. The difference between day-type-specific
and pooled \textsc{Baton} over all days is reported in the text with a
paired bootstrap interval over the 80 plans.""",
        "l ccc ccc",
        r"& \multicolumn{3}{c}{tuned threshold} & \multicolumn{3}{c}{\textsc{Baton}} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}" + "\n" +
        r"Fitting strategy & normal & promotion & all & normal & promotion & all \\",
        body))

# ── fresh-start bias and the exact reset ──────────────────────────────────
fr = _read("r1/fresh.csv")
if fr is not None:
    body = []
    FAM = [("SalhiNagy", "Salhi--Nagy (Det)"), ("Dethloff", "Dethloff (SAA)"),
           ("City", "City, real shops (Det)")]
    for fam, disp in FAM:
        d = fr[fr.fam == fam]
        if d.empty:
            continue
        re_ = d.reactive.sum()

        def sv(c):
            return 100 * (re_ - d[c].sum()) / re_
        b_in = 100 * (d.bias_in / d.H_mean).mean()
        b_out = 100 * (d.bias_out / d.H_mean).mean()
        body.append(f"{disp} & {b_in:+.2f} & {b_out:+.2f} & {sv('cost_ho'):.1f} & "
                    f"{sv('cost_baton3'):.1f} & {sv('cost_baton3_Fout'):.1f} & "
                    f"{sv('cost_baton3_Fstar'):.1f} & {sv('cost_cf'):.1f} & "
                    f"{sv('cost_exact_reset'):.1f} & {sv('cost_dp3cf'):.1f} \\\\")
        key = {"SalhiNagy": "Salhi", "Dethloff": "Deth", "City": "City"}[fam]
        macros[f"fr{key}BiasIn"] = f"{b_in:+.2f}\\%"
        macros[f"fr{key}BiasOut"] = f"{b_out:+.2f}\\%"
        macros[f"fr{key}Three"] = _pct(sv("cost_baton3"))
        macros[f"fr{key}Cf"] = _pct(sv("cost_cf"))
        macros[f"fr{key}Exact"] = _pct(sv("cost_exact_reset"))
        macros[f"fr{key}Fstar"] = _pct(sv("cost_baton3_Fstar"))
        macros[f"fr{key}Ho"] = _pct(sv("cost_ho"))
        macros[f"fr{key}DpThree"] = _pct(sv("cost_dp3cf"))
        macros[f"fr{key}RsBaton"] = _pct(100 * d.rs_baton3.mean())
        macros[f"fr{key}RsExact"] = _pct(100 * d.rs_exact_reset.mean())
    _write("tab_fresh.tex", _tex_table(
        "tab:fresh",
        r"""The fresh-start value and the post-return state. Bias columns:
mean error of the estimated fresh-start value $\widehat F_k$ against the
high-data reference $F_k$ of DP$^3_{50\mathrm{k}}$, in \% of the handoff price;
in-sample is \textsc{Baton}'s own estimate, out-of-sample re-evaluates
the same fitted downstream policy on $5\times10^4$ independent paths.
Saving columns (\% vs.\ reactive, all \emph{without} deployment
selection, so that the three-action fit is exposed): the three-action
policy as fitted, with $\widehat F_k$ replaced by its out-of-sample
value, with $F_k$ replaced by the high-data reference value, with the
state-conditional fresh-start value (\textsc{Baton-cf}), and with the
exact residual-capacity reset $x_k = -D_{\le k}$. Savings in this table
pool the route costs of all plans of a benchmark (ratio of sums), whereas
Tables~\ref{tab:grand} and~\ref{tab:large} average the plan-level
savings over instances, so the two differ slightly.""",
        "l cc c ccccc c",
        r"& \multicolumn{2}{c}{bias of $\widehat F_k$ (\% of $H$)} & & \multicolumn{5}{c}{three-action \textsc{Baton}, saving \%} & \\ \cmidrule(lr){2-3}\cmidrule(lr){5-9}" + "\n" +
        r"Benchmark & in-sample & out-of-sample & \textsc{Baton-ho} & as fitted & $F$ out-of-s. & $F$ reference & \textsc{cf} & exact reset & DP$^3_{50\mathrm{k}}$ \\",
        body, size=r"\scriptsize", sep="2.6pt"))

# ── capped standby pool ────────────────────────────────────────────────────
pl = _read("r1/pool.csv")
if pl is not None and "shadow_rec" in pl:
    POOL_H = (20.0, 5.0)

    def _sstar(d, h):
        """Per-pool cost-minimising number of reserved vehicles at holding
        cost h per vehicle-day (shadow-priced policy)."""
        t = d.assign(tot=h * d.S + d.shadow_rec)
        return t.loc[t.groupby("Instance").tot.idxmin()]
    body = []
    for fam, pln, disp in (("Dethloff", "Det", r"Dethloff, \textsc{Det}"),
                           ("Dethloff", "SAA", r"Dethloff, \textsc{SAA}"),
                           ("City", "Det", r"City, \textsc{Det}"),
                           ("Metro", "Det", r"metro pools, \textsc{Det}")):
        d = pl[(pl.fam == fam) & (pl.Plan == pln)]
        if d.empty:
            continue
        d0 = d[d.S == 0].set_index("Instance")
        d1 = d[d.S == 1].set_index("Instance")
        h1 = (d0.shadow_rec - d1.shadow_rec)
        gain = (d1.naive_rec - d1.shadow_rec).to_numpy()
        lo, hi = _boot_ci(gain) if len(gain) > 1 else (gain.mean(), gain.mean())
        ss = [_sstar(d, h) for h in POOL_H]
        cells = [disp, str(d.Instance.nunique()), f"{d0.K.mean():.1f}",
                 f"{d0.ppu_rec.mean():.1f}",
                 f"{d0.naive_rec.mean():.1f}", f"{d0.shadow_rec.mean():.1f}",
                 f"{d1.naive_rec.mean():.1f}", f"{d1.shadow_rec.mean():.1f}",
                 f"{d1.ppuprice_rec.mean():.1f}", f"{d1.fallback_rec.mean():.1f}",
                 f"{gain.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]",
                 f"{h1.mean():.1f}"] + [f"{x.S.mean():.1f}" for x in ss]
        body.append(" & ".join(cells) + r" \\")
        key = {"Dethloff": "Deth", "City": "City", "Metro": "Metro"}[fam] + \
            ("Saa" if pln == "SAA" else "")
        macros[f"pool{key}Ppu"] = f"{d0.ppu_rec.mean():.1f}"
        macros[f"pool{key}NaiveZero"] = f"{d0.naive_rec.mean():.1f}"
        macros[f"pool{key}ShadowZero"] = f"{d0.shadow_rec.mean():.1f}"
        macros[f"pool{key}NaiveOne"] = f"{d1.naive_rec.mean():.1f}"
        macros[f"pool{key}ShadowOne"] = f"{d1.shadow_rec.mean():.1f}"
        macros[f"pool{key}BreakEven"] = f"{h1.mean():.1f}"
        macros[f"pool{key}Sstar"] = f"{ss[0].S.mean():.1f}"
        macros[f"pool{key}SstarLow"] = f"{ss[1].S.mean():.1f}"
        macros[f"pool{key}K"] = f"{d0.K.mean():.0f}"
        macros[f"pool{key}TotHigh"] = f"{(POOL_H[0] * ss[0].S + ss[0].shadow_rec).mean():.1f}"
        macros[f"pool{key}TotLow"] = f"{(POOL_H[1] * ss[1].S + ss[1].shadow_rec).mean():.1f}"
        macros[f"pool{key}BeatLow"] = f"{int((POOL_H[1] * ss[1].S + ss[1].shadow_rec < ss[1].ppu_rec - 1e-9).sum())}/{len(ss[1])}"
        macros[f"pool{key}GainOne"] = _pct(100 * gain.mean() / d1.naive_rec.mean())
        macros[f"pool{key}GainZero"] = _pct(
            100 * (d0.naive_rec.mean() - d0.shadow_rec.mean()) / d0.naive_rec.mean())
    m = pl[pl.fam == "Metro"]
    if not m.empty:
        # largest pool-size gain of shadow over naive pricing, metro pools
        g = m.groupby("S")[["naive_rec", "shadow_rec"]].mean()
        rel = 100 * (g.naive_rec - g.shadow_rec) / g.naive_rec
        macros["poolMetroGainMax"] = _pct(rel.max())
        macros["poolMetroGainMaxS"] = str(int(rel.idxmax()))
        lam = m.groupby("S")["lambda"].mean()
        macros["poolMetroLamSmall"] = f"{lam.loc[2]:.0f}"
        macros["poolMetroLamLarge"] = f"{lam.iloc[-1]:.1f}"
        macros["poolMetroSmax"] = str(int(m.S.max()))
    _write("tab_pool.tex", _tex_table(
        "tab:pool",
        r"""Shared, capacitated standby pool: expected daily recourse cost per
pool (currency units, excluding the holding cost of the reserved
vehicles). A pool serves the routes of one plan, or, in the metro rows,
the \textsc{Det} plans of all ten instances of a Dethloff class
(CON3, CON8, SCA3, SCA8). Pay-per-use: the default model (a handoff is
billed at its full price $H_k$ and capacity is unlimited). Reserved
pool of $S$ vehicles: a handoff costs its marginal price $H_k -
F_{\mathrm{sb}}$, requests are served first come, first served in the
order of their time along the routes, and a route whose request is
refused chooses at the refusal stop between continuing and a depot
return. Naive: every route decides at the marginal price and ignores
the cap; shadow: every route decides at $H_k - F_{\mathrm{sb}} +
\lambda$, with $\lambda$ chosen on the training days from
$\{0, 2, 5, 10, 20, 40, \infty\}$; the two fixed-price columns use
$\lambda = F_{\mathrm{sb}}$ (pay-per-use pricing) and $\lambda = \infty$
(never hand off) at $S = 1$. Gain: naive minus shadow at $S = 1$ with a
95\% bootstrap interval over pools. $h_1$: break-even holding cost of the
first reserved vehicle (cost at $S = 0$ minus cost at $S = 1$, shadow
pricing); a reservation pays when the vehicle-day costs less than $h_1$.
$S^\star$: mean cost-minimising pool size at a holding cost of 20 (the
standby day rate) and of 5 per vehicle-day.""",
        "l cc c cc cccc c c cc",
        r"& & & pay-per- & \multicolumn{2}{c}{$S = 0$} & \multicolumn{4}{c}{$S = 1$} & gain & & \multicolumn{2}{c}{$S^\star$ at $h =$} \\ \cmidrule(lr){5-6}\cmidrule(lr){7-10}\cmidrule(lr){13-14}" + "\n" +
        r"Benchmark & pools & routes & use & naive & shadow & naive & shadow & $\lambda{=}F_{\mathrm{sb}}$ & $\lambda{=}\infty$ & at $S{=}1$ [95\% CI] & $h_1$ & 20 & 5 \\",
        body, size=r"\scriptsize", sep="2.4pt"))

# ── training-data budget ───────────────────────────────────────────────────
bud = _read("r1/budget.csv")
if bud is not None:
    g = bud.groupby("N")
    cross = None
    for N, d in g:
        ok = True
        for (fam, pln), dd in d.groupby(["fam", "Plan"]):
            if dd.v2_act_saving.mean() + 1e-9 < dd.fb_tau_saving.mean():
                ok = False
        if ok and cross is None:
            cross = N
    macros["budgetCross"] = f"{cross:,}".replace(",", "{,}") if cross else "(none)"
    body = []
    for N, d in g:
        cells = [f"{N:,}".replace(",", "{,}")]
        for fam, pln in (("Dethloff", "Det"), ("Dethloff", "SAA"), ("City", "Det")):
            dd = d[(d.fam == fam) & (d.Plan == pln)]
            cells += [f"{dd.fb_tau_saving.mean():.1f}", f"{dd.dp_n_saving.mean():.1f}",
                      f"{dd.v2_act_saving.mean():.1f}"]
        body.append(" & ".join(cells) + r" \\")
    ref = []
    for fam, pln in (("Dethloff", "Det"), ("Dethloff", "SAA"), ("City", "Det")):
        dd = bud[(bud.fam == fam) & (bud.Plan == pln) & (bud.N == bud.N.max())]
        ref += ["", f"{dd.dp_xl_saving.mean():.1f}", f"{dd.dp_xl3_saving.mean():.1f}"]
    body.append(r"\midrule")
    body.append(r"ref.: DP$_{50\mathrm{k}}$ / DP$^3_{50\mathrm{k}}$ & " + " & ".join(ref) + r" \\")
    for fam, pln, key in (("Dethloff", "Det", "DethDet"), ("Dethloff", "SAA", "DethSaa"), ("City", "Det", "City")):
        dd = bud[(bud.fam == fam) & (bud.Plan == pln)]
        for N, nm in ((100, "Small"), (1000, "Mid"), (20000, "Large")):
            macros[f"bud{key}{nm}"] = _pct(dd[dd.N == N].v2_act_saving.mean())
            macros[f"bud{key}Thr{nm}"] = _pct(dd[dd.N == N].fb_tau_saving.mean())
            macros[f"bud{key}DpN{nm}"] = _pct(dd[dd.N == N].dp_n_saving.mean())
        macros[f"bud{key}Dp"] = _pct(dd[dd.N == dd.N.max()].dp_xl_saving.mean())
        macros[f"bud{key}DpThree"] = _pct(dd[dd.N == dd.N.max()].dp_xl3_saving.mean())
    _write("tab_budget.tex", _tex_table(
        "tab:budget",
        r"""Training-data budget: saving \% vs.\ reactive as a function of
the number $N$ of training days per route (same test days throughout).
thr.: label-corrected tuned threshold; DP$_N$: plug-in dynamic program
at equal data; \textsc{Baton} with deployment selection. The last row
gives the high-data reference programs (handoff-only / three-action).
The $N = 10^3$ row re-draws the training days of a smaller instance set
(Section~\ref{sec:budget}) and therefore differs from
Tables~\ref{tab:grand} and~\ref{tab:large} by up to 0.3 points.""",
        "r ccc ccc ccc",
        r"& \multicolumn{3}{c}{Dethloff, \textsc{Det}} & \multicolumn{3}{c}{Dethloff, \textsc{SAA}} & \multicolumn{3}{c}{City, \textsc{Det}} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}" + "\n" +
        r"$N$ & thr. & DP$_N$ & \textsc{Baton} & thr. & DP$_N$ & \textsc{Baton} & thr. & DP$_N$ & \textsc{Baton} \\",
        body))

# ── timing ─────────────────────────────────────────────────────────────────
tm = _read("r1/timing.csv")
if tm is not None:
    TP = [("fb_tau", "tuned threshold (peak label)"), ("thr_k", "position-dependent threshold"),
          ("pi3", r"$\pi_3$ rule (grid-tuned)"), ("rollout", "rollout"),
          ("ro_theta", r"cost-scaled rollout"), ("restock", "depot-return restocking"),
          ("thr2", "two-lever rule (joint grid)"),
          ("dp_n", r"DP$_N$ (equal data)"), ("dp3_n", r"DP$^3_N$ (equal data)"),
          ("v2_lsm", r"\textsc{Baton-ho}"),
          ("v2_act", r"\textsc{Baton} (incl.\ deployment selection)"),
          ("v2_cf", r"\textsc{Baton-cf}"),
          ("dp_xl", r"DP$_{50\mathrm{k}}$ (reference, $5\times10^4$ paths)"),
          ("dp_xl3", r"DP$^3_{50\mathrm{k}}$ (reference, $5\times10^4$ paths)")]
    body = []
    for l, disp in TP:
        c = f"{l}_fit_s"
        if c not in tm:
            continue
        data_n = r"$5\times10^4$" if l in ("dp_xl", "dp_xl3") else "$10^3$"
        body.append(f"{disp} & {data_n} & {1000 * tm[c].median():.1f} & {1000 * tm[c].quantile(0.9):.1f} \\\\")
    macros["timeBatonMs"] = f"{1000 * tm.v2_act_fit_s.median():.0f}\\,ms"
    macros["timeBatonHoMs"] = f"{1000 * tm.v2_lsm_fit_s.median():.0f}\\,ms"
    macros["timeOnlineUs"] = f"{tm.lat_interp_us.median():.1f}\\,$\\mu$s"
    macros["timeOnlineSkUs"] = f"{tm.lat_sklearn_us.median():.0f}\\,$\\mu$s"
    macros["nTimingRoutes"] = str(len(tm))
    _write("tab_timing.tex", _tex_table(
        "tab:timing",
        r"""Offline fitting time per route (single CPU core, milliseconds;
median and 90th percentile over """ + str(len(tm)) + r""" routes of ten Dethloff
\textsc{SAA} plans and four city plans). Every policy is fitted on the
same $10^3$ training days except the two reference programs. Tuning by
grid search on realised training cost is included; the
position-dependent threshold includes the fits of its two starting
points. The online decision
of every cost-comparison policy is one lookup in a monotone
piecewise-linear function and two comparisons: """ + f"{tm.lat_interp_us.median():.1f}" + r"""\,$\mu$s per decision through the
fitted breakpoints (""" + f"{tm.lat_sklearn_us.median():.0f}" + r"""\,$\mu$s through the
scikit-learn prediction call).""",
        "l c cc",
        r"Policy & training days & median & p90 \\",
        body))

# ── regret of the myopic rule ─────────────────────────────────────────────
rg = _read("r1/regret.csv")
if rg is not None:
    body = []
    for (fam, pln), d in rg.groupby(["fam", "Plan"], sort=False):
        re_ = d.reactive.sum()
        body.append(
            f"{fam}, \\textsc{{{pln}}} & {len(d)} & {d.m.mean():.1f} & "
            f"{100 * d.regret.sum() / re_:.1f} & {100 * d.bound.sum() / re_:.1f} & "
            f"{100 * d.bound_clean.sum() / re_:.1f} & "
            f"{100 * d.p_premature.mean():.1f} & {100 * d.p_premature_clean.mean():.1f} & "
            f"{100 * d.p_contained.mean():.1f} & "
            f"{100 * (d.thr_cost.sum() - d.ho_cost.sum()) / re_:.1f} \\\\")
    rg["mbin"] = pd.cut(rg.m, [0, 12, 18, 100], labels=[r"$m \le 12$", r"$13 \le m \le 18$", r"$m \ge 19$"])
    gap = {}
    for b, d in rg.groupby("mbin", observed=True):
        gap[str(b)] = 100 * (d.thr_cost.sum() - d.ho_cost.sum()) / d.reactive.sum()
        rgl = 100 * d.regret.sum() / d.reactive.sum()
        body.append(f"\\quad all, {b} & {len(d)} & {d.m.mean():.1f} & {rgl:.1f} & "
                    f"{100 * d.bound.sum() / d.reactive.sum():.1f} & "
                    f"{100 * d.bound_clean.sum() / d.reactive.sum():.1f} & "
                    f"{100 * d.p_premature.mean():.1f} & {100 * d.p_premature_clean.mean():.1f} & "
                    f"{100 * d.p_contained.mean():.1f} & {gap[str(b)]:.1f} \\\\")
    macros["regretAll"] = _pct(100 * rg.regret.sum() / rg.reactive.sum())
    macros["regretBoundAll"] = _pct(100 * rg.bound.sum() / rg.reactive.sum())
    macros["regretContained"] = _pct(100 * rg.p_contained.mean())
    macros["regretBoundHolds"] = f"{int((rg.regret <= rg.bound + 1e-9).sum())}/{len(rg)}"
    vals = list(gap.values())
    macros["thrGapShort"] = _pct(vals[0])
    macros["thrGapMid"] = _pct(vals[1])
    macros["thrGapLong"] = _pct(vals[-1])
    for b, nm in zip(list(gap.keys()), ("Short", "Mid", "Long")):
        d = rg[rg.mbin.astype(str) == b]
        macros[f"regret{nm}"] = _pct(100 * d.regret.sum() / d.reactive.sum())
    macros["regretCleanShare"] = _pct(100 * rg.bound_clean.sum() / rg.bound.sum(), 0)
    for (fam, pln), nm in ((("Dethloff", "Det"), "Det"), (("Dethloff", "SAA"), "Saa"),
                           (("City", "Det"), "City")):
        d = rg[(rg.fam == fam) & (rg.Plan == pln)]
        macros[f"regret{nm}"] = _pct(100 * d.regret.sum() / d.reactive.sum())
        macros[f"thrGap{nm}"] = _pct(100 * (d.thr_cost.sum() - d.ho_cost.sum()) / d.reactive.sum())
    per = {}
    for (fam, pln), d in rg.groupby(["fam", "Plan"], sort=False):
        per[f"{fam} {pln}"] = 100 * d.regret.sum() / d.reactive.sum()
    lo_k = min(per, key=per.get)
    macros["regretMinRow"] = lo_k.replace("Dethloff", "Dethloff,").replace(" Det", r" \textsc{Det}").replace(" SAA", r" \textsc{SAA}")
    macros["regretMin"] = _pct(per[lo_k])
    macros["regretRouteMean"] = _pct(100 * (rg.regret / rg.reactive.clip(lower=1e-9))[rg.reactive > 1e-6].mean())
    macros["regretBoundFail"] = str(int((rg.regret > rg.bound + 1e-9).sum()))
    fl = rg[rg.regret > rg.bound + 1e-9]
    macros["regretFailMax"] = (_pct(100 * ((fl.regret - fl.bound) / fl.reactive).max(), 2)
                               if len(fl) else "0\\%")
    macros["nRegretRoutes"] = str(len(rg))
    _write("tab_regret.tex", _tex_table(
        "tab:regret",
        r"""The price of over-triggering (Proposition~\ref{prop:regret}),
per route, on 2{,}000 test days. The myopic rule and the optimal rule
are both estimated on $5\times10^4$ paths (the myopic cost-to-go by the
rollout regression, the optimal one by backward induction), so that the
comparison reflects the functional form rather than estimation noise.
Regret: excess expected cost of the myopic rule; bound: right-hand side
of~\eqref{eq:regret}; clean-day term: $\mathbb E[H_{\sigma^0}
\mathbf 1\{\sigma^0 < \sigma^\star, T = \infty\}]$; all in \% of the
reactive cost. $P(\sigma^0 < \sigma^\star)$: share of days on which the
myopic rule hands off before the optimal one; containment: share of days
with $\sigma^0 \le \sigma^\star$. Last column: excess cost of the tuned
threshold over \textsc{Baton-ho}, both fitted on $10^3$ days (\% of
reactive cost). All columns pool the routes of a row (ratios of sums).
On breach days the integrand $H_{\sigma^0} - L_{\sigma^0}$ of the bound
can be negative (a handoff cheaper than the clairvoyant remainder), so
the clean-day term can exceed the bound itself.""",
        "l cc ccc ccc c",
        r"& & & & & clean-day & & $\sigma^0 < \sigma^\star$, & & thr.\ $-$ \\" + "\n" +
        r"Routes & $n$ & $\bar m$ & regret & bound & term & $P(\sigma^0 < \sigma^\star)$ & $T = \infty$ & contain. & \textsc{Baton-ho} \\",
        body, size=r"\scriptsize", sep="2.4pt"))


if "RATIOS" in globals() and RATIOS:
    lo_k = min(RATIOS, key=RATIOS.get)
    hi_k = max(RATIOS, key=RATIOS.get)
    macros["ratioAllLo"] = f"{100 * RATIOS[lo_k]:.0f}\\%"
    macros["ratioAllHi"] = f"{100 * RATIOS[hi_k]:.0f}\\%"
    print("  BATON / DP3 ratio range:", lo_k, RATIOS[lo_k], hi_k, RATIOS[hi_k])


# ═══════════════════════════════════════════════════════════════════════════
# certification macros
# ═══════════════════════════════════════════════════════════════════════════
grb = _read("results_mip_cert_gurobi.csv")
hgs = _read("results_mip_cert.csv")
if grb is not None:
    macros["certGap"] = _pct(grb.gap_alns_pct.mean(), 2)
    macros["certGapMax"] = _pct(grb.gap_alns_pct.max(), 2)
    macros["certBeatN"] = f"{int((grb.ALNS_obj < grb.MIP_UB).sum())}/40"
if hgs is not None:
    macros["certGapHighs"] = _pct(hgs.gap_alns_pct.mean(), 2)


# ═══════════════════════════════════════════════════════════════════════════
# write macros
# ═══════════════════════════════════════════════════════════════════════════
lines = ["% generated by make_tables.py — do not hand-edit"]
for k, v in sorted(macros.items()):
    lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
_write("macros.tex", "\n".join(lines) + "\n")
print("done.")

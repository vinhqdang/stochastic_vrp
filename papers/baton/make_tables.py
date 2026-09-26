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
               "fb_tau", "thr_k"]
BOOT = np.random.default_rng(20260926)


def _boot_ci(d, n=10_000):
    d = np.asarray(d, float)
    idx = BOOT.integers(0, len(d), (n, len(d)))
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


if grand is not None:
    cols = [("pi3", r"$\pi_3$"), ("rollout", "roll."),
            ("ro_theta", r"roll.-$\theta$"), ("restock", "restock"),
            ("fb_tau", "thr."), ("thr_k", r"thr.-$k$"),
            ("v2_lsm", r"\textsc{Baton-ho}"), ("v2_act", r"\textsc{Baton}"),
            ("dp_xl", r"DP$_{50\mathrm{k}}$"),
            ("dp_xl3", r"DP$^3_{50\mathrm{k}}$"), ("oracle", "oracle")]
    rows = []
    for g in GATES:
        s_ = grand[grand.Plan == g]
        cells = [GATE_DISP[g]]
        for lbl, _ in cols:
            v = s_[f"{lbl}_saving"].mean()
            cell = f"{v:.1f}"
            if lbl == "v2_act":
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        rows.append(" & ".join(cells) + r" \\")
    tab = r"""\begin{table}[t]
\caption{Expected-recourse saving over the reactive policy (\%, mean over
the 40 Dethloff instances) for each planning gate and execution policy
under the three-class fleet cost model; \emph{higher is better}.
\textsc{Baton} (bold) has the highest saving of all implementable
policies on every gate. thr.\ is the label-corrected tuned threshold,
thr.-$k$ the position-dependent threshold, roll.-$\theta$ the
cost-scaled rollout. The last three columns are reference points, not
competitors: DP$_{50\mathrm{k}}$ and DP$^3_{50\mathrm{k}}$ are
near-exact dynamic programs for the handoff-only and the three-action
problem with fifty times the training data, and the clairvoyant oracle
is restricted to the handoff lever, so \textsc{Baton} may legitimately
exceed it.}
\label{tab:grand}
\centering
\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l cccccc cc ccc}
\toprule
& \multicolumn{6}{c}{published / tuned competitors}
& \multicolumn{2}{c}{this paper} & \multicolumn{3}{c}{reference points} \\
\cmidrule(lr){2-7}\cmidrule(lr){8-9}\cmidrule(lr){10-12}
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
    ratio = [grand[grand.Plan == g].v2_act_saving.mean() /
             grand[grand.Plan == g].dp_xl3_saving.mean() for g in GATES]
    macros["ratioDpThreeLo"] = f"{100 * min(ratio):.0f}\\%"
    macros["ratioDpThreeHi"] = f"{100 * max(ratio):.0f}\\%"
    orc_beat = sum(grand[grand.Plan == g].v2_act_saving.mean() >
                   grand[grand.Plan == g].oracle_saving.mean() for g in GATES)
    macros["nOrcBeat"] = ["zero", "one", "two", "three", "four", "five",
                          "six"][orc_beat]
    # standby pool the policies would need (95% of days), pay-per-use model
    macros["poolBatonMed"] = f"{grand.v2_act_S.median():.0f}"
    macros["poolBatonMax"] = f"{grand.v2_act_S.max():.0f}"
    macros["poolHoMed"] = f"{grand.v2_lsm_S.median():.0f}"
    macros["poolHoMax"] = f"{grand.v2_lsm_S.max():.0f}"

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
            "restock": "restock", "fb_tau": "thr.", "thr_k": r"thr.-$k$"}
    for i, row in enumerate(cells):
        out = [GATE_DISP[row[0]], NICE[row[1]]]
        for j in range(3):
            mu, lo, hi, w, t, r = row[2 + j]
            p = adj[3 * i + j]
            ptxt = r"$<\!10^{-4}$" if p < 1e-4 else f"{p:.3f}"
            out.append(f"{mu:+.1f} [{lo:+.1f}, {hi:+.1f}] & {w} & {r:+.2f} & {ptxt}")
        body.append(" & ".join(out) + r" \\")
    tab = r"""\begin{table}[t]
\caption{Paired comparison of \textsc{Baton} with, in turn, the strongest
competitor on each gate (the implementable policy with the highest mean
saving in Table~\ref{tab:grand}), its handoff-only restriction, and the
near-exact three-action dynamic program. The experimental unit is the
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
\begin{adjustbox}{max width=\textwidth}
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
        body.append(f"{GATE_DISP[g]} & {base:.1f} & {red('fb_tau'):.1f} & "
                    f"{red('v2_lsm'):.1f} & \\textbf{{{red('v2_act'):.1f}}} & "
                    f"{100 * s_.none_pem.mean():.1f} & {100 * s_.fb_tau_pem.mean():.1f} & "
                    f"{100 * s_.v2_lsm_pem.mean():.1f} & \\textbf{{{100 * s_.v2_act_pem.mean():.1f}}} \\\\")
    tab = r"""\begin{table}[t]
\caption{Tail risk of the daily recourse bill of a plan (40 Dethloff
instances per gate, 2{,}000 test days each). CVaR$_{95}$: mean of the
worst 5\% of daily plan bills; the reactive column gives its level in
currency units, the others its reduction (\%, higher is better).
$P(\mathrm{emg})$: share of days on which at least one route of the plan
suffers an emergency (\%, lower is better).}
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
    LCOLS = ["restock", "fb_tau", "thr_k", "v2_lsm", "v2_act", "dp_xl",
             "dp_xl3", "oracle"]

    def _row(name, d):
        vals = []
        for l in LCOLS:
            v = f"{d[f'{l}_saving'].mean():.1f}"
            vals.append(rf"\textbf{{{v}}}" if l == "v2_act" else v)
        return f"{name} & {len(d)} & " + " & ".join(vals) + r" \\"
    rows = [_row(r"Salhi--Nagy", sn), _row(r"City, real shops", city)]
    if cityu is not None:
        rows.append(_row(r"City, uniform", cityu))
    if zp25 is not None:
        rows.append(_row(r"City, 25\% deliver-only", zp25))
    if zp50 is not None:
        rows.append(_row(r"City, 50\% deliver-only", zp50))
    tab = r"""\begin{table}[t]
\caption{Large-scale benchmarks under the fleet cost model
(Det-gate plans; saving \% vs.\ reactive, \emph{higher is better};
the proposed policy is in bold; the last three columns are reference
points). Salhi--Nagy instances carry 50--199 customers; the city
instances (100--400 customers) place customers at real OSM shop
locations on the road networks of Ho Chi Minh City, Hanoi, New York,
Paris and Shanghai. The uniform twins use the same cities and demands
with uniformly scattered customers; the deliver-only twins set the
pickup of a random 25\% or 50\% of the customers to zero and are
re-planned.}
\label{tab:large}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l r cccccccc}
\toprule
Benchmark & $n$ & restock & thr. & thr.-$k$ & \textsc{Baton-ho} &
\textsc{Baton} & DP$_{50\mathrm{k}}$ & DP$^3_{50\mathrm{k}}$ & oracle \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
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
        body.append(f"{disp} & {d.restock_saving.mean():.1f} & "
                    f"{d.fb_tau_saving.mean():.1f} & "
                    f"{d.v2_lsm_saving.mean():.1f} & "
                    rf"\textbf{{{d.v2_act_saving.mean():.1f}}} & "
                    f"{d.oracle_saving.mean():.1f} \\\\")
    tab = r"""\begin{table}[t]
\caption{Sensitivity of the recourse saving (\%, \emph{higher is
better}; best non-clairvoyant value per row in bold) to the fleet-economics
parameters, one factor at a time around the defaults (12 Dethloff
instances, \textsc{Det} and \textsc{SAA} gates; the baseline row is
computed on the same 12 instances). \textsc{Baton} re-balances its
action mix as prices move and leads in every configuration. In the last
row the standby rate exceeds the emergency price at late stops, so
Inequality~\eqref{eq:prices} fails there.}
\label{tab:costsens}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l ccccc}
\toprule
Configuration & restock & threshold & \textsc{Baton-ho} &
\textsc{Baton} & oracle (HO) \\
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
    p = _wilcox(best_rl[1], v2)
    nb = int((v2 < best_rl[1] - 1e-9).sum())
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
\begin{adjustbox}{max width=\textwidth}
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
               sep="3.5pt"):
    return (r"\begin{table}[t]" + "\n" + rf"\caption{{{caption}}}" + "\n" +
            rf"\label{{{label}}}" + "\n" + r"\centering" + "\n" + size + "\n" +
            rf"\setlength{{\tabcolsep}}{{{sep}}}" + "\n" +
            r"\begin{adjustbox}{max width=\textwidth}" + "\n" +
            rf"\begin{{tabular}}{{{colspec}}}" + "\n" + r"\toprule" + "\n" +
            header + "\n" + r"\midrule" + "\n" + "\n".join(body) + "\n" +
            r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" +
            r"\end{adjustbox}" + "\n" + r"\end{table}" + "\n")


# ── dependence + shape ─────────────────────────────────────────────────────
dep = _read("r1/dependence.csv")
shp = _read("r1/shape.csv")
if dep is not None:
    CFG = [("rho0", r"$\rho = 0$ (independent)", 0.0),
           ("rho03", r"$\rho = 0.3$", None), ("rho06", r"$\rho = 0.6$ (benchmark)", 0.6),
           ("rho09", r"$\rho = 0.9$", 0.9),
           ("dayfac", r"day factor, $\rho = 0$", None)]
    DC = ["fb_tau", "thr_k", "v2_lsm", "v2_act", "v2_cf", "dp_xl3", "oracle"]
    body = []
    for tag, disp, rho in CFG:
        first = True
        for pln in ("Det", "SAA"):
            d = dep[(dep.cfg == tag) & (dep.Plan == pln)]
            if d.empty:
                continue
            vals = []
            for l in DC:
                v = f"{d[f'{l}_saving'].mean():.1f}"
                vals.append(rf"\textbf{{{v}}}" if l == "v2_act" else v)
            if first and shp is not None and rho is not None and (shp.rho == rho).any():
                sh = shp[shp.rho == rho]
                vr = f"{100 * sh.viol.sum() / max(sh.pairs.sum(), 1):.1f}"
            else:
                vr = "" if not first else "--"
            body.append(f"{disp if first else ''} & \\textsc{{{pln}}} & {d.none_rec.mean():.1f} & " +
                        " & ".join(vals) + f" & {vr} \\\\")
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
    if shp is not None:
        for rho, nm in ((0.0, "Zero"), (0.6, "Six"), (0.9, "Nine")):
            sh = shp[shp.rho == rho]
            macros[f"shapeViol{nm}"] = _pct(100 * sh.viol.sum() / max(sh.pairs.sum(), 1))
    _write("tab_dependence.tex", _tex_table(
        "tab:dependence",
        r"""Robustness to the demand dependence (40 Dethloff instances,
\textsc{Det} and \textsc{SAA} plans, saving \% vs.\ reactive; plans are
held fixed and policies are re-fitted and re-tested under each demand
law; react.: expected daily recourse cost of the reactive policy per
plan, which shows how much there is to save). $\rho$ is the Gaussian-copula equicorrelation among deliveries and
among pickups; the day-factor law multiplies every demand of a day by a
common lognormal factor (s.d.\ 0.25) on top of independent marginals.
Last column: share of adjacent-bin pairs in which the unconstrained
binned estimate of $\mathbb E[\text{future cost} \mid W_k]$, fitted on
$5\times10^4$ paths per route with 25 bins, \emph{decreases}
significantly (one-sided, 2.5\% level; the rate under independence,
where monotonicity is a theorem, calibrates the test).""",
        "l l c ccccccc c",
        r"Demand law & gate & react. & thr. & thr.-$k$ & \textsc{Baton-ho} & \textsc{Baton} & \textsc{Baton-cf} & DP$^3_{50\mathrm{k}}$ & oracle & viol.\ (\%) \\",
        body))

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
    macros["dtPromoAware"] = _pct(_mixsv([(("promo", "promo"), 1.0)], "v2_act"))
    macros["dtPromoStale"] = _pct(_mixsv([(("normal", "promo"), 1.0)], "v2_act"))
    macros["dtPromoPooled"] = _pct(_mixsv([(("mix", "promo"), 1.0)], "v2_act"))
    macros["dtAllAware"] = _pct(_mixsv([(("normal", "normal"), .8), (("promo", "promo"), .2)], "v2_act"))
    macros["dtAllPooled"] = _pct(_mixsv([(("mix", "normal"), .8), (("mix", "promo"), .2)], "v2_act"))
    _write("tab_daytype.tex", _tex_table(
        "tab:daytype",
        r"""Non-exchangeable days (40 Dethloff instances, \textsc{Det} and
\textsc{SAA} plans; saving \% vs.\ reactive on the same days). One day
in five is a promotion day with mean deliveries $\times 1.15$ and mean
pickups $\times 1.35$. Pooled: one fit on a history that mixes both day
types; day-type-specific: separate fits, the day type being known in
advance (promotions are scheduled); stale: fitted on normal days only
and applied unchanged on promotion days.""",
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
near-exact $F_k$ of DP$^3_{50\mathrm{k}}$, in \% of the handoff price;
in-sample is \textsc{Baton}'s own estimate, out-of-sample re-evaluates
the same fitted downstream policy on $5\times10^4$ independent paths.
Saving columns (\% vs.\ reactive, all \emph{without} deployment
selection, so that the three-action fit is exposed): the three-action
policy as fitted, with $\widehat F_k$ replaced by its out-of-sample
value, with $F_k$ replaced by the near-exact value, with the
state-conditional fresh-start value (\textsc{Baton-cf}), and with the
exact residual-capacity reset $x_k = -D_{\le k}$.""",
        "l cc c ccccc c",
        r"& \multicolumn{2}{c}{bias of $\widehat F_k$ (\% of $H$)} & & \multicolumn{5}{c}{three-action \textsc{Baton}, saving \%} & \\ \cmidrule(lr){2-3}\cmidrule(lr){5-9}" + "\n" +
        r"Benchmark & in-sample & out-of-sample & \textsc{Baton-ho} & as fitted & $F$ out-of-s. & $F$ near-exact & \textsc{cf} & exact reset & DP$^3_{50\mathrm{k}}$ \\",
        body, size=r"\scriptsize", sep="2.6pt"))

# ── capped standby pool ────────────────────────────────────────────────────
pl = _read("r1/pool.csv")
if pl is not None:
    pl = pl.copy()
    pl["naive_r"] = pl.naive_total - pl.hold
    pl["shadow_r"] = pl.shadow_total - pl.hold
    body = []
    for fam, disp in (("Dethloff", "Dethloff (SAA plans)"), ("City", "City (Det plans)")):
        d = pl[pl.fam == fam]
        if d.empty:
            continue
        best = d.loc[d.groupby("Instance").shadow_total.idxmin()]
        cells = [disp, str(best.Instance.nunique()), f"{best.K.mean():.1f}",
                 f"{best.ppu_rec.mean():.1f}"]
        for S in (0, 1, 2):
            dd = d[d.S == S]
            cells += [f"{dd.naive_r.mean():.1f}", f"{dd.shadow_r.mean():.1f}"]
        cells += [f"{best.S.mean():.1f}"]
        body.append(" & ".join(cells) + r" \\")
        key = "Deth" if fam == "Dethloff" else "City"
        d0, d1 = d[d.S == 0], d[d.S == 1]
        macros[f"pool{key}Sstar"] = f"{best.S.mean():.1f}"
        macros[f"pool{key}Ppu"] = f"{best.ppu_rec.mean():.1f}"
        macros[f"pool{key}NaiveZero"] = f"{d0.naive_r.mean():.1f}"
        macros[f"pool{key}ShadowZero"] = f"{d0.shadow_r.mean():.1f}"
        macros[f"pool{key}NaiveOne"] = f"{d1.naive_r.mean():.1f}"
        macros[f"pool{key}ShadowOne"] = f"{d1.shadow_r.mean():.1f}"
        macros[f"pool{key}GainZero"] = _pct(100 * (d0.naive_r.mean() - d0.shadow_r.mean()) / d0.naive_r.mean())
        macros[f"pool{key}GainOne"] = _pct(100 * (d1.naive_r.mean() - d1.shadow_r.mean()) / d1.naive_r.mean())
        macros[f"pool{key}Hold"] = f"{best.hold.max() if best.S.max() > 0 else 20:.0f}"
    _write("tab_pool.tex", _tex_table(
        "tab:pool",
        r"""Shared, capacitated standby pool: expected daily recourse cost per
plan (currency units, excluding the holding cost of reserved vehicles).
Pay-per-use: the default model (pooled capacity, day rate billed per
handoff). Reserved pool of $S$ vehicles: each held vehicle costs
$F_{\mathrm{sb}} = 20$ per day whether or not it is used, a handoff then
costs its marginal price $H_k - F_{\mathrm{sb}}$, requests are served
first come, first served in the order of their time along the routes, and
a route whose request is refused continues under its fallback menu
(continue or return to the depot). Naive: every route decides at the
marginal price and ignores the cap. Shadow: every route decides at
$H_k - F_{\mathrm{sb}} + \lambda$, with the shadow price $\lambda$ of a
standby vehicle chosen on the training days. $S^\star$: pool size
minimising holding plus recourse for the shadow-priced policy.""",
        "l cc c cc cc cc c",
        r"& & & pay-per- & \multicolumn{2}{c}{$S = 0$} & \multicolumn{2}{c}{$S = 1$} & \multicolumn{2}{c}{$S = 2$} & \\ \cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}" + "\n" +
        r"Benchmark & plans & routes & use & naive & shadow & naive & shadow & naive & shadow & $S^\star$ \\",
        body))

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
        macros[f"bud{key}Dp"] = _pct(dd[dd.N == dd.N.max()].dp_xl_saving.mean())
        macros[f"bud{key}DpThree"] = _pct(dd[dd.N == dd.N.max()].dp_xl3_saving.mean())
    _write("tab_budget.tex", _tex_table(
        "tab:budget",
        r"""Training-data budget: saving \% vs.\ reactive as a function of
the number $N$ of training days per route (same test days throughout).
thr.: label-corrected tuned threshold; DP$_N$: plug-in dynamic program
at equal data; \textsc{Baton} with deployment selection. The last row
gives the near-exact references (handoff-only / three-action).""",
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
          ("dp_n", r"DP$_N$ (equal data)"), ("v2_lsm", r"\textsc{Baton-ho}"),
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
    macros["timeOnlineUs"] = f"{tm.lat_interp_us.median():.0f}\\,$\\mu$s"
    macros["timeOnlineSkUs"] = f"{tm.lat_sklearn_us.median():.0f}\\,$\\mu$s"
    macros["nTimingRoutes"] = str(len(tm))
    _write("tab_timing.tex", _tex_table(
        "tab:timing",
        r"""Offline fitting time per route (single CPU core, milliseconds;
median and 90th percentile over """ + str(len(tm)) + r""" routes of ten Dethloff
\textsc{SAA} plans and four city plans). Every policy is fitted on the
same $10^3$ training days except the two reference programs. Tuning by
grid search on realised training cost is included. The online decision
of every cost-comparison policy is one lookup in a monotone step function
and two comparisons: """ + f"{tm.lat_interp_us.median():.1f}" + r"""\,$\mu$s per decision through the
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
reactive cost).""",
        "l cc ccc ccc c",
        r"& & & & & clean-day & & $\sigma^0 < \sigma^\star$, & & thr.\ $-$ \\" + "\n" +
        r"Routes & $n$ & $\bar m$ & regret & bound & term & $P(\sigma^0 < \sigma^\star)$ & $T = \infty$ & contain. & \textsc{Baton-ho} \\",
        body, size=r"\scriptsize", sep="2.4pt"))


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

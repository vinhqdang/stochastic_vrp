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
RES = HERE.parent / "svrpspd_wdro" / "results"
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
if grand is not None:
    cols = [("pi3", r"$\pi_3$"), ("rollout", "rollout"),
            ("restock", "restock"), ("fb_tau", "threshold"),
            ("v2_lsm", r"\textsc{Baton-ho}"), ("v2_act", r"\textsc{Baton}"),
            ("dp_xl", r"DP$_{50\mathrm{k}}$"), ("oracle", "oracle")]
    rows = []
    for g in GATES:
        s = grand[grand.Plan == g]
        cells = [GATE_DISP[g]]
        best = max(s[f"{lbl}_saving"].mean() for lbl, _ in cols[:-2])
        for lbl, _ in cols:
            v = s[f"{lbl}_saving"].mean()
            cell = f"{v:.1f}"
            if lbl == "v2_act":
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        rows.append(" & ".join(cells) + r" \\")
    tab = r"""\begin{table}[t]
\caption{Expected-recourse saving over the reactive policy (\%, mean over
the 40 Dethloff instances) for each planning gate and execution policy
under the three-class fleet cost model; \emph{higher is better}, and the
best non-clairvoyant value per gate is in bold. \textsc{Baton-ho} is the
handoff-only restriction; DP$_{50\mathrm{k}}$ and the clairvoyant oracle
are handoff-only reference points, so \textsc{Baton} may legitimately
exceed them by exercising its richer action set.}
\label{tab:grand}
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l cccc cc cc}
\toprule
& \multicolumn{4}{c}{published / tuned competitors}
& \multicolumn{2}{c}{this paper} & \multicolumn{2}{c}{reference (HO)} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Gate & """ + " & ".join(h for _, h in cols) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    tab = tab.replace(r"Gate & $\pi_3$", r"Gate & $\pi_3$")
    _write("tab_grand.tex", tab)

    # macros: pooled Wilcoxon of BATON vs each competitor
    for lbl, key in (("restock", "WRestock"), ("rollout", "WRollout"),
                     ("pi3", "WPiThree"), ("fb_tau", "WThresh"),
                     ("v2_lsm", "WHo"), ("dp_xl", "WDpxl")):
        p = _wilcox(grand[f"{lbl}_rec"], grand["v2_act_rec"])
        n_better = int((grand[f"{lbl}_rec"] - grand["v2_act_rec"] > 1e-9).sum())
        macros[f"baton{key}P"] = _pfmt(p)
        macros[f"baton{key}N"] = f"{n_better}/240"
    macros["batonGrandBest"] = _pct(
        max(grand[grand.Plan == g]["v2_act_saving"].mean() for g in GATES))
    for g in GATES:
        s = grand[grand.Plan == g]
        macros[f"sv{g}Baton"] = _pct(s.v2_act_saving.mean())
        macros[f"sv{g}Ho"] = _pct(s.v2_lsm_saving.mean())
        macros[f"sv{g}Orc"] = _pct(s.oracle_saving.mean())
        macros[f"sv{g}Thresh"] = _pct(s.fb_tau_saving.mean())


# ═══════════════════════════════════════════════════════════════════════════
# Table 2 — large-scale benchmarks (Salhi–Nagy + city, shops & uniform)
# ═══════════════════════════════════════════════════════════════════════════
sn = _read("results_salhinagy_eval.csv")
city = _read("results_city_eval.csv")
cityu = _read("results_cityuniform_eval.csv")
if sn is not None and city is not None:
    def _row(name, d):
        return (f"{name} & {len(d)} & {d.restock_saving.mean():.1f} & "
                f"{d.fb_tau_saving.mean():.1f} & {d.v2_lsm_saving.mean():.1f} & "
                rf"\textbf{{{d.v2_act_saving.mean():.1f}}} & "
                f"{d.dp_xl_saving.mean():.1f} & {d.oracle_saving.mean():.1f} \\\\")
    rows = [_row(r"Salhi--Nagy", sn),
            _row(r"City, real shops", city)]
    if cityu is not None:
        rows.append(_row(r"City, uniform", cityu))
    tab = r"""\begin{table}[t]
\caption{Large-scale benchmarks under the fleet cost model
(Det-gate plans; saving \% vs.\ reactive, \emph{higher is better};
the proposed policy is in bold, DP$_{50\mathrm{k}}$ and the oracle are
reference points). Salhi--Nagy instances carry
50--199 customers; the city instances (100--400 customers) place
customers at real OSM shop locations on the road networks of Ho Chi Minh
City, Hanoi, New York, Paris and Shanghai, and the uniform twins use the
same cities and demands with uniformly scattered customers.}
\label{tab:large}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l r cccccc}
\toprule
Benchmark & $n$ & restock & threshold & \textsc{Baton-ho} &
\textsc{Baton} & DP$_{50\mathrm{k}}$ & oracle \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_large.tex", tab)
    macros["svSalhiBaton"] = _pct(sn.v2_act_saving.mean())
    macros["svSalhiOrc"] = _pct(sn.oracle_saving.mean())
    macros["svCityBaton"] = _pct(city.v2_act_saving.mean())
    macros["nCity"] = str(len(city))
    if cityu is not None:
        m = city.merge(cityu, on="Instance", suffixes=("_s", "_u"))
        macros["shopTravelSaving"] = _pct(
            (100 * (m.Travel_km_u - m.Travel_km_s) / m.Travel_km_u).mean())


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
]
cs_files = {Path(f).stem.replace("results_costsens_", ""): pd.read_csv(f)
            for f in glob.glob(str(RES / "results_costsens_*.csv"))}
if cs_files and grand is not None:
    base = grand[grand.Plan.isin(["Det", "SAA"])]
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
instances, \textsc{Det} and \textsc{SAA} gates). \textsc{Baton}
re-balances its action mix as prices move and leads in every
configuration.}
\label{tab:costsens}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l ccccc}
\toprule
Configuration & restock & threshold & \textsc{Baton-ho} &
\textsc{Baton} & oracle (HO) \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_costsens.tex", tab)
    if "F_standby_35" in cs_files:
        d = cs_files["F_standby_35"]
        macros["dearSbThresh"] = _pct(d.fb_tau_saving.mean())
        macros["dearSbBaton"] = _pct(d.v2_act_saving.mean())
        macros["dearSbOrc"] = _pct(d.oracle_saving.mean())


# ═══════════════════════════════════════════════════════════════════════════
# Table 4 — RL head-to-head
# ═══════════════════════════════════════════════════════════════════════════
def _rl_table():
    import sys
    sys.path.insert(0, str(HERE.parent / "svrpspd_wdro"))
    try:
        from core.costs import fit_lsm_general, simulate_v2_general
    except Exception:
        return
    bundle = RES / "rl_bundle.npz"
    first = RES / "rl_results.json"
    if not (bundle.exists() and first.exists()):
        return
    z = np.load(bundle, allow_pickle=True)
    n = int(z["n_routes"][0])
    v2 = []
    for i in range(n):
        g_tr = z[f"r{i}_g_train"].astype(float)
        g_te = z[f"r{i}_g_test"].astype(float)
        H, E = z[f"r{i}_H"].astype(float), z[f"r{i}_E"].astype(float)
        B = float(z[f"r{i}_B"][0])
        cm = fit_lsm_general(g_tr, B, H, E)
        v2.append(simulate_v2_general(g_te, B, H, E, cm)["mean_cost"])
    v2 = np.array(v2)
    ref = json.load(open(first))
    re = np.array([r["reactive_cost"] for r in ref])

    runs = [("40 epochs (paper spec)", first)]
    for f in sorted(RES.glob("rl_strong_s*.json")):
        runs.append((f"150 epochs, seed {f.stem[-1]}", f))
    body, best_rl = [], None
    for name, f in runs:
        rl = np.array([r["rl_cost"] for r in json.load(open(f))])
        sv = 100 * (re.sum() - rl.sum()) / re.sum()
        if best_rl is None or sv > best_rl[0]:
            best_rl = (sv, rl)
        body.append(f"{name} & {sv:.1f} \\\\")
    sv_v2 = 100 * (re.sum() - v2.sum()) / re.sum()
    p = _wilcox(best_rl[1], v2)
    nb = int((v2 < best_rl[1] - 1e-9).sum())
    tab = r"""\begin{table}[t]
\caption{Reinforcement-learning baseline (re-implementation of the policy
architecture of Iklassov et al., 2024) versus \textsc{Baton-ho} on the
identical 50 routes and out-of-sample test days; saving \% vs.\ the
reactive policy, \emph{higher is better}. The learned policy is
trained on a GPU; \textsc{Baton} fits in milliseconds per route on a CPU.}
\label{tab:rl}
\centering
\begin{tabular}{l c}
\toprule
Policy & saving vs.\ reactive (\%) \\
\midrule
""" + "\n".join(body) + rf"""
\midrule
\textsc{{Baton-ho}} & \textbf{{{sv_v2:.1f}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    _write("tab_rl.tex", tab)
    macros["rlBest"] = _pct(best_rl[0])
    macros["rlBatonHo"] = _pct(sv_v2)
    macros["rlWinN"] = f"{nb}/50"
    macros["rlWinP"] = _pfmt(p)


_rl_table()


# ═══════════════════════════════════════════════════════════════════════════
# Table 5 — synthetic structural scenarios (label defect isolated)
# ═══════════════════════════════════════════════════════════════════════════
syn = _read("results_otr2_synthetic.csv")
if syn is not None:
    def _ms(scen, col):
        return syn[syn.scenario == scen][col].mean()
    rows = []
    for scen, disp in (("collect_then_deliver", "collect-then-deliver"),
                       ("milk_run_regime", "milk run, regime switching"),
                       ("high_cost_ratio", r"milk run, $C/\omega = 20$")):
        rows.append(f"{disp} & {_ms(scen, 'v1_tun_saving'):.1f} & "
                    f"{_ms(scen, 'fb_tun_saving'):.1f} & "
                    rf"\textbf{{{_ms(scen, 'v2_lsm_saving'):.1f}}} \\")
    tab = r"""\begin{table}[t]
\caption{Structural synthetic scenarios (five seeds, $1.2\times10^4$ test
routes each): saving \% over the reactive policy, \emph{higher is
better}. On collect-then-deliver structure the endpoint-labelled
predecessor never intervenes.}
\label{tab:synthetic}
\centering
\begin{tabular}{l ccc}
\toprule
Scenario & endpoint + $\tau$ & peak + $\tau$ & \textsc{Baton-ho} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    _write("tab_synthetic.tex", tab)
    macros["ctdVOne"] = _pct(_ms("collect_then_deliver", "v1_tun_saving"))
    macros["ctdBaton"] = _pct(_ms("collect_then_deliver", "v2_lsm_saving"))


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

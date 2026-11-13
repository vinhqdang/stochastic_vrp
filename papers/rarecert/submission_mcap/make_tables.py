"""Generate the table inputs of rarecert_mcap.tex from the result files.

Reads svrpspd_wdro/results/rarecert/peis/{conditions.csv, bias_check.csv} and writes
tab_conditions.tex, tab_bias.tex and bias_sentence.tex next to this script.
Never edit those three files by hand.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parents[2] / "svrpspd_wdro" / "results" / "rarecert" / "peis"


def pct(x):
    return f"{100 * x:.1f}\\%"


def conditions():
    d = pd.read_csv(RES / "conditions.csv")
    rows = []
    for rho in (0.3, 0.6, 0.8):
        for reg, name in (("VNN", "Vietnam"), ("GD", "Guangdong")):
            s = d[(d.rho == rho) & (d.region == reg)]
            rows.append(
                f"{rho:g} & {name} & {len(s)} & {s.varrho.median():.2f} & {pct((s.varrho > 0.25).mean())} & "
                f"{pct(s.thm3_closed.mean())} & {pct(s.thm3.mean())} & {pct(s.thm3_massw.mean())} \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering\small
\caption{{Conditions of the theorems on all routes with at least two stops: median
end-mode ratio $\varrho$, share of routes with $\varrho>1/4$ (single shift not
efficient, Theorem~\ref{{thm:single}}), and share satisfying the covering
condition~\eqref{{eq:cover}} with components for both blocks of every stage with
$r_k>1/2$: by the closed-form cases of Corollary~\ref{{cor:closed}}, with the exact
constant $\psi$ and constant weights, and with mass weights
$\delta_m=1/\lambda_m-1/\lambda^\star$. The conditions do not depend on the coefficient
of variation.}}
\label{{tab:conditions}}
\begin{{tabular}}{{llrrrrrr}}
\toprule
$\rho$ & region & routes & $\varrho$ & $\varrho>1/4$ & Cor.~\ref{{cor:closed}} & $\psi$, constant & $\Psi$, mass \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


N_DRAWS = {"single": 1_000_000, "gl": 20_000}
ERR = {"single": "single_relerr", "gl": "gl_shift_relerr"}
NAME = {"single": "single CE shift", "gl": "Glasserman--Li"}


def bias():
    d = pd.read_csv(RES / "bias_check.csv")
    b = d[d.ratio > 0.75].copy()
    b["p_B"] = b.s_missed * b.p_ref
    rows = []
    stats = {}
    for est in ("single", "gl"):
        for g in (1e-4, 1e-6):
            s = b[b.gamma == g]
            NpB = N_DRAWS[est] * s.p_B
            under = -s[ERR[est]]
            exp = f"10^{{{int(round(np.log10(g)))}}}"
            ratio = (under / s.s_missed)[s.s_missed > 1e-2]
            rows.append(f"{NAME[est]} & ${exp}$ & {N_DRAWS[est]:.0e} & {np.median(NpB):.2g} & "
                        f"{np.median(s.s_missed):.3f} & {100 * np.median(under):.1f}\\% & "
                        f"{np.median(ratio):.2f} & {(NpB < 1).sum()} \\\\".replace("e+0", "e").replace("1e6", "$10^6$").replace("2e4", "$2\\times10^4$"))
            stats[(est, g)] = (NpB, under, s.s_missed)
    body = "\n".join(rows)
    tab = rf"""\begin{{table}}[t]
\centering\small
\caption{{Silent underestimation on the 15 balanced routes ($\varrho>0.75$) of
Table~\ref{{tab:rivals}}. $p_{{\mathcal B}}$ is the probability of the weaker end event,
estimated on the single block, and $s=p_{{\mathcal B}}/p$ its share, with $p$ the quadrature
reference; $Np_{{\mathcal B}}$ is the expected number of draws that reach that mode.
Underestimate: $-(\hat p-p)/p$. Ratio: underestimate$/s$ on the rows with $s>0.01$. All
medians over routes; last column: routes with $Np_{{\mathcal B}}<1$.}}
\label{{tab:bias}}
\begin{{tabular}}{{llrrrrrr}}
\toprule
estimator & $\gamma$ & $N$ & $Np_{{\mathcal B}}$ & $s$ & underestimate & ratio & $Np_{{\mathcal B}}<1$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    from scipy.stats import spearmanr
    NpB, under, sm = [pd.concat(x) for x in zip(*[stats[k] for k in stats])]
    m = NpB < 1
    rho_s = spearmanr(under[m], sm[m]).correlation
    med_ratio = np.median((under[m] / sm[m])[sm[m] > 1e-2])
    nd = int(m.sum())
    sent = (f"In the {nd} route--level rows where an estimator expects fewer than one draw in the "
            f"weaker mode ($Np_{{\\mathcal B}}<1$), its underestimate follows $s$: the Spearman "
            f"correlation is ${rho_s:.2f}$ and the median ratio of underestimate to $s$ is "
            f"${med_ratio:.2f}$. The single shift at $10^{{-4}}$ expects about "
            f"${np.median(stats[('single', 1e-4)][0]):.0f}$ such draws in median; it then reaches the "
            f"weaker mode, and the deficit shows partly as error and partly as variance (relative "
            f"variance $212$ in Table~\\ref{{tab:rivals}}). On the routes with $\\varrho\\le0.75$, "
            f"$s<10^{{-15}}$ and both estimators are unbiased within their sampling error.")
    return tab, sent


if __name__ == "__main__":
    (HERE / "tab_conditions.tex").write_text(conditions())
    t, s = bias()
    (HERE / "tab_bias.tex").write_text(t)
    (HERE / "bias_sentence.tex").write_text(s + "\n")
    print(s)

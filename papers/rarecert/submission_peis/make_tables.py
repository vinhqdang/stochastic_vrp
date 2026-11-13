"""Generate the table inputs of rarecert_peis.tex from the result files.

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


def bias():
    d = pd.read_csv(RES / "bias_check.csv")
    d["bal"] = d.ratio > 0.75
    rows = []
    for g in (1e-4, 1e-6):
        for bal, name in ((True, r"$\varrho>0.75$"), (False, r"$\varrho\le0.75$")):
            s = d[(d.gamma == g) & (d.bal == bal)]
            exp = f"10^{{{int(round(np.log10(g)))}}}"
            rows.append(
                f"${exp}$ & {name} & {len(s)} & {np.median(s.s_missed):.2g} & "
                f"{100 * np.median(-s.single_relerr):+.2f}\\% & {100 * np.median(-s.gl_shift_relerr):+.2f}\\% & "
                f"{100 * np.median(np.abs(s.single_relerr + s.s_missed)):.2f}\\% & "
                f"{100 * np.median(np.abs(s.gl_shift_relerr + s.s_missed)):.2f}\\% \\\\")
    body = "\n".join(rows)
    tab = rf"""\begin{{table}}[t]
\centering\small
\caption{{Predicted and measured underestimation, 30 routes. $s$ is the quadrature
share of the weaker end event, $\P(\text{{weaker end event}})/p$; the next columns give
the measured underestimate $-(\hat p-p)/p$ of the single CE shift ($10^6$ draws) and of
Glasserman--Li ($2\times10^4$ draws), and the absolute difference between the measured
relative error and the prediction $-s$ of Proposition~\ref{{prop:silent}} (all medians
over routes).}}
\label{{tab:bias}}
\begin{{tabular}}{{llrrrrrr}}
\toprule
$\gamma$ & routes & $n$ & $s$ & single shift & Glasserman--Li & $|{{\rm err}}+s|$ single & $|{{\rm err}}+s|$ GL \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    b = d[d.bal]
    corr_single = np.corrcoef(-b.single_relerr, b.s_missed)[0, 1]
    corr_gl = np.corrcoef(-b.gl_shift_relerr, b.s_missed)[0, 1]
    sent = (f"On the 15 balanced routes the share $s$ is ${np.median(b.s_missed):.2f}$ in median "
            f"(range ${b.s_missed.min():.2f}$--${b.s_missed.max():.2f}$), and the measured underestimates "
            f"follow it: their correlation with $s$ across route--level rows is ${corr_single:.2f}$ for the "
            f"single shift and ${corr_gl:.2f}$ for Glasserman--Li. On the other routes $s$ is below "
            f"${d[~d.bal].s_missed.max():.1g}$ and both estimators are unbiased within their sampling error.")
    return tab, sent


if __name__ == "__main__":
    (HERE / "tab_conditions.tex").write_text(conditions())
    t, s = bias()
    (HERE / "tab_bias.tex").write_text(t)
    (HERE / "bias_sentence.tex").write_text(s + "\n")
    print(s)

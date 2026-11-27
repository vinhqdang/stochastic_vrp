"""LaTeX tables for papers/tempo/tempo.tex from results/tempo_pool/<family>.csv.

Writes papers/tempo/tables/{gap,pairs,slope}.tex. Same statistics as
scripts/tempo_pool_tables.py (instance-clustered bootstrap, same seed).

Usage (from svrpspd_wdro/): python scripts/tempo_pool_latex.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_W))
sys.path.insert(0, str(_W / "scripts"))

from tempo_pool_tables import SEED, _matrix, boot_ci, slope_points  # noqa: E402
from ev.pooling import SCENARIOS  # noqa: E402

FAMS = [("Dethloff", "Dethloff"), ("SalhiNagy", "Salhi--Nagy"), ("City", "City")]
SCN = list(SCENARIOS)
SCN_TEX = {"city": "city", "mixed": "mixed", "zone": "zone", "drift": "drift",
           "delfactor": "delivery", "heavytail": "heavy tail", "cvmis": "noise"}
GAP_POL = ["route", "full", "zone", "hier", "hier_eb", "mle"]
PAIR_B = ["route", "full", "zone", "hier"]
OUT = _W.parent / "papers" / "tempo" / "tables"


def load(fam):
    return pd.read_csv(_W / "results" / "tempo_pool" / f"{fam}.csv",
                       keep_default_na=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    gap, pairs, slope = [], [], {}
    for fam, ftex in FAMS:
        df = load(fam)
        for i, s in enumerate(SCN):
            sub = df[df["scenario"] == s]
            m = {p: float(sub[f"cost_{p}"].mean())
                 for p in ["fixed", "known"] + GAP_POL}
            den = m["fixed"] - m["known"]
            lead = f"\\multirow{{7}}{{*}}{{{ftex}}}" if i == 0 else ""
            cells = [f"{(m['fixed'] - m[p]) / den:.2f}" for p in GAP_POL]
            best = max(range(len(GAP_POL)), key=lambda j: float(cells[j]))
            cells = [f"\\textbf{{{c}}}" if float(c) == float(cells[best]) else c
                     for c in cells]
            gap.append(" & ".join([lead, SCN_TEX[s], f"{m['fixed']:.1f}",
                                   f"{m['known']:.1f}"] + cells) + r" \\")
            row = [lead, SCN_TEX[s]]
            for b in PAIR_B:
                M = _matrix(sub, "cost_hier_eb") - _matrix(sub, f"cost_{b}")
                pt, lo, hi = boot_ci(M, rng)
                txt = f"{pt:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                txt = txt.replace("-", "$-$").replace("+", "$+$")
                if hi < 0 or lo > 0:
                    txt = f"\\textbf{{{txt}}}"
                row.append(txt)
            pairs.append(" & ".join(row) + r" \\")
            sl, used, tot = slope_points(sub)
            slope[(fam, s)] = (sl, used, tot)
        gap.append(r"\midrule")
        pairs.append(r"\midrule")
    gap[-1] = pairs[-1] = r"\bottomrule"

    (OUT / "gap.tex").write_text("\n".join([
        r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
        r" & & \multicolumn{2}{c}{Mean daily cost} & \multicolumn{6}{c}{Share of the gap closed} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-10}",
        r"Family & Scenario & Fixed & Known & Route & Full & Zone & Two-level & EB & MLE \\",
        r"\midrule"] + gap + [r"\end{tabular}", ""]))
    (OUT / "pairs.tex").write_text("\n".join([
        r"\begin{tabular}{llllll}", r"\toprule",
        r"Family & Scenario & EB $-$ route & EB $-$ full & EB $-$ zone & EB $-$ two-level \\",
        r"\midrule"] + pairs + [r"\end{tabular}", ""]))
    rows = []
    for s in SCN:
        rows.append(" & ".join([SCN_TEX[s]] + [f"{slope[(f, s)][0]:.2f}"
                                               for f, _ in FAMS]) + r" \\")
    (OUT / "slope.tex").write_text("\n".join([
        r"\begin{tabular}{lrrr}", r"\toprule",
        "Scenario & " + " & ".join(t for _, t in FAMS) + r" \\", r"\midrule"]
        + rows + [r"\bottomrule", r"\end{tabular}", ""]))
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()

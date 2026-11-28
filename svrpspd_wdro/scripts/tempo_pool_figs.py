"""Figures and sensitivity tables for papers/tempo/tempo.tex.

Reads results/tempo_pool/{trace_*,*_sens_*,timing_*,<family>}.csv and writes
papers/tempo/figures/{fig_rmse,fig_sz}.pdf and papers/tempo/tables/{sens_H,sens_Z,timing}.tex.

Usage (from svrpspd_wdro/): python scripts/tempo_pool_figs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_W))
sys.path.insert(0, str(_W / "scripts"))
from tempo_pool_tables import SEED, _matrix, boot_ci  # noqa: E402

RES = _W / "results" / "tempo_pool"
PAPER = _W.parent / "papers" / "tempo"
FAMS = [("Dethloff", "Dethloff"), ("SalhiNagy", "Salhi--Nagy"), ("City", "City")]
FAM_PLAIN = {"Dethloff": "Dethloff", "SalhiNagy": "Salhi-Nagy", "City": "City"}
# validated categorical slots 1-4 (light); markers give a second encoding
STYLE = {
    "route": ("Route", "#257de0", "o"),
    "full": ("Full", "#f47230", "s"),
    "zone": ("Zone", "#00b894", "^"),
    "hier_eb": ("EB (TEMPO)", "#f2a600", "D"),
}
INK = "#333333"
GRID = "#dddddd"


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#999999")
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def fig_rmse(kmax=10):
    scns = [("city", "City factor only"), ("mixed", "Mixed"), ("zone", "Zone factors only")]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.6), sharex=True)
    for i, (fam, _) in enumerate(FAMS):
        for j, (s, stitle) in enumerate(scns):
            ax = axes[i, j]
            _style(ax)
            p = RES / f"trace_{fam}_{s}.csv"
            if not p.exists():
                ax.set_visible(False)
                continue
            t = read(p)
            g = t.groupby(["policy", "k"])[["sse", "n"]].sum().reset_index()
            for pol, (lab, col, mk) in STYLE.items():
                h = g[(g.policy == pol) & (g.k <= kmax)].sort_values("k")
                ax.plot(h.k, np.sqrt(h.sse / h.n), color=col, marker=mk,
                        markersize=4, linewidth=1.6, label=lab)
            if i == 0:
                ax.set_title(stitle, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{FAM_PLAIN[fam]}\nRMSE of factor", fontsize=8, color=INK)
            if i == 2:
                ax.set_xlabel("Stop index $k$", fontsize=8, color=INK)
            ax.set_ylim(bottom=0)
    vis = [a for a in axes.flat if a.get_visible() and a.get_lines()]
    h, l = vis[0].get_legend_handles_labels() if vis else ([], [])
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = PAPER / "figures" / "fig_rmse.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def _gap_shares(df):
    m = {c: df[f"cost_{c}"].mean() for c in ["fixed", "known"] + list(STYLE)}
    den = m["fixed"] - m["known"]
    return {p: (m["fixed"] - m[p]) / den for p in STYLE}


def fig_sz():
    szs = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=False)
    for ax, (fam, _) in zip(axes, FAMS):
        _style(ax)
        rows = []
        for sz in szs:
            if abs(sz - 0.2) < 1e-9:
                d = read(RES / f"{fam}.csv")
                d = d[d.scenario == "mixed"]
            else:
                p = RES / f"{fam}_sens_sz{sz:.2f}.csv"
                if not p.exists():
                    continue
                d = read(p)
            rows.append((sz, _gap_shares(d)))
        for pol, (lab, col, mk) in STYLE.items():
            ax.plot([r[0] for r in rows], [r[1][pol] for r in rows], color=col,
                    marker=mk, markersize=4, linewidth=1.6, label=lab)
        ax.set_title(FAM_PLAIN[fam], fontsize=9, color=INK)
        ax.set_xlabel("Zone-level sd $s_z$ ($s_c = 0.2$)", fontsize=8, color=INK)
    axes[0].set_ylabel("Share of gap closed", fontsize=8, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = PAPER / "figures" / "fig_sz.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def _fmt(pt, lo, hi):
    t = f"{pt:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    t = t.replace("-", "$-$").replace("+", "$+$")
    return f"\\textbf{{{t}}}" if (hi < 0 or lo > 0) else t


def table_H(rng):
    Hs = [5, 10, 20, 30, 60]
    lines = [r"\begin{tabular}{l" + "l" * len(FAMS) + "}", r"\toprule",
             "Past days $D$ & " + " & ".join(t for _, t in FAMS) + r" \\", r"\midrule"]
    for H in Hs:
        cells = [str(H)]
        for fam, _ in FAMS:
            if H == 60:
                d = read(RES / f"{fam}.csv")
                d = d[d.scenario == "mixed"]
            else:
                p = RES / f"{fam}_sens_H{H}.csv"
                if not p.exists():
                    cells.append("---")
                    continue
                d = read(p)
            M = _matrix(d, "cost_hier_eb") - _matrix(d, "cost_hier")
            cells.append(_fmt(*boot_ci(M, rng)))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (PAPER / "tables" / "sens_H.tex").write_text("\n".join(lines))
    print("wrote sens_H.tex")


def table_Z(rng):
    lines = [r"\begin{tabular}{llrrrll}", r"\toprule",
             r"Family & $Z$ & Route & Zone & EB & EB $-$ route & EB $-$ zone \\",
             r"\midrule"]
    for fam, ftex in FAMS:
        for k, Z in enumerate([2, 4, 8]):
            p = RES / f"{fam}_sens_Z{Z}.csv"
            if not p.exists():
                continue
            d = read(p)
            g = _gap_shares(d)
            c1 = _fmt(*boot_ci(_matrix(d, "cost_hier_eb") - _matrix(d, "cost_route"), rng))
            c2 = _fmt(*boot_ci(_matrix(d, "cost_hier_eb") - _matrix(d, "cost_zone"), rng))
            lead = ftex if k == 0 else ""
            lines.append(" & ".join([lead, str(Z), f"{g['route']:.2f}", f"{g['zone']:.2f}",
                                     f"{g['hier_eb']:.2f}", c1, c2]) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", ""]
    (PAPER / "tables" / "sens_Z.tex").write_text("\n".join(lines))
    print("wrote sens_Z.tex")


def table_timing():
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Family & Customers & Routes & Library (s) & Posterior (ms) & Variances (ms) \\",
             r"\midrule"]
    for fam, ftex in FAMS:
        p = RES / f"timing_{fam}.csv"
        if not p.exists():
            continue
        t = read(p)
        lines.append(" & ".join([
            ftex, f"{t.n_cust.min()}--{t.n_cust.max()}", f"{t.K.min()}--{t.K.max()}",
            f"{t.lib_seconds.min():.0f}--{t.lib_seconds.max():.0f}",
            f"{t.hier_ms.mean():.2f}", f"{t.history_ms.mean():.2f}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (PAPER / "tables" / "timing.tex").write_text("\n".join(lines))
    print("wrote timing.tex")


def main():
    rng = np.random.default_rng(SEED)
    fig_rmse()
    fig_sz()
    table_H(rng)
    table_Z(rng)
    table_timing()


if __name__ == "__main__":
    main()

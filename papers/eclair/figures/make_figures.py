#!/usr/bin/env python3
"""Generate the manuscript's figures from the committed result files
(computes nothing new except the Fig-1 wealth-trajectory trace, which
re-runs one seeded certification with the trace hook).

Run from papers/eclair/figures:  ../../../.venv/bin/python make_figures.py
Outputs Fig1..Fig4 as PDF, sized for the Constraints format (119 mm).
"""

import json
import math
import pathlib
import random
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent / "code"
RES = CODE / "results"
sys.path.insert(0, str(CODE))

# validated palette (dataviz reference instance, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK = "#9a9891", "#0b0b0b"

W = 119 / 25.4                       # Constraints figure width (inches)
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "pdf.fonttype": 42,
})


def fig1_trajectories():
    """Wealth trajectories of one certified pool (mechanism figure)."""
    from eclair.certify import Bets, run_certification
    from eclair.mutations import make_faithful, make_mutant
    from eclair.problems import EVAL_SPECS

    bets = Bets(json.loads((RES / "calibration.json").read_text()))
    budget = 80 * statistics.fmean(bets.cost.values())
    rng = random.Random(7)
    spec = EVAL_SPECS[0]
    pool, labels = [], []
    for faithful in (True, True, True, False, False, False):
        pool.append(make_faithful(spec) if faithful
                    else make_mutant(spec, rng))
        labels.append(faithful)
    trace = []
    run_certification(pool, bets, "kelly", budget, 0.05, rng, trace=trace)

    paths = {i: [(0.0, 0.0)] for i in range(len(pool))}
    for idx, tier, cost, logE in trace:
        paths[idx].append((1000 * cost, logE))

    fig, ax = plt.subplots(figsize=(W, 2.5))
    thresh = math.log(1 / 0.05)
    ax.axhline(thresh, color=GRAY, lw=1, ls="--")
    ax.text(0.4, thresh + 0.14, r"reject: $\log(1/\alpha)$",
            color="#52514e", fontsize=7)
    seen = {True: False, False: False}
    for i, pts in paths.items():
        xs, ys = zip(*pts)
        ys = [min(y, thresh + 0.9) for y in ys]
        c, ls = (AQUA, "-") if labels[i] else (ORANGE, "-")
        lab = None
        if not seen[labels[i]]:
            lab = "faithful" if labels[i] else "mutant"
            seen[labels[i]] = True
        ax.plot(xs, ys, color=c, lw=1.6, ls=ls, label=lab,
                solid_capstyle="round")
    ax.set_xlim(-12, 360)
    ax.set_xlabel("cumulative probe time (ms)")
    ax.set_ylabel(r"$\log E_t(m)$")
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(HERE / "Fig1.pdf"); fig.savefig(HERE / "Fig1.png", dpi=150)
    plt.close(fig)


def fig2_alpha():
    data = json.loads((RES / "alpha_sweep.json").read_text())["rows"]
    alphas = [r["alpha"] for r in data]
    frej = [r["false_rej"] for r in data]
    det = [r["detect"] for r in data]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.2))
    a1.plot(alphas, alphas, color=GRAY, lw=1, ls="--")
    a1.text(0.105, 0.125, "nominal $\\alpha$", color="#52514e",
            fontsize=7, rotation=38)
    a1.plot(alphas, frej, color=BLUE, lw=1.6, marker="o", ms=4)
    a1.set_xscale("log")
    a1.set_xticks(alphas, [str(a) for a in alphas])
    a1.set_ylim(-0.012, 0.22)
    a1.set_xlabel(r"nominal level $\alpha$")
    a1.set_ylabel("false-rejection rate")
    a1.set_title("(a) validity", loc="left")

    a2.plot(alphas, det, color=BLUE, lw=1.6, marker="o", ms=4)
    for x, y in zip(alphas, det):
        a2.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=6.5,
                    color="#52514e")
    a2.set_xscale("log")
    a2.set_xticks(alphas, [str(a) for a in alphas])
    a2.set_ylim(0.55, 1.02)
    a2.set_xlabel(r"nominal level $\alpha$")
    a2.set_ylabel("detection rate")
    a2.set_title("(b) power", loc="left")
    fig.tight_layout()
    fig.savefig(HERE / "Fig2.pdf"); fig.savefig(HERE / "Fig2.png", dpi=150)
    plt.close(fig)


def fig3_ablation():
    rows = json.loads((RES / "tier_ablation.json").read_text())["rows"]
    names = ["+".join(r["tiers"]) for r in rows]
    det = [r["detect"] for r in rows]
    frej = [r["false_rej"] for r in rows]
    colors = [ORANGE if r["tiers"] == "C" else BLUE for r in rows]
    ys = range(len(rows))[::-1]

    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(W, 2.2), gridspec_kw={"width_ratios": [3, 2]})
    a1.barh(ys, det, height=0.62, color=colors)
    for y, d in zip(ys, det):
        a1.text(d + 0.015, y, f"{d:.2f}", va="center", fontsize=6.5,
                color="#52514e")
    a1.set_yticks(ys, names)
    a1.set_xlim(0, 1.12)
    a1.set_xlabel("detection rate")
    a1.set_title("(a) power by probe pool", loc="left")

    a2.barh(ys, frej, height=0.62, color=colors)
    a2.axvline(0.05, color=GRAY, lw=1, ls="--")
    a2.text(0.052, ys[0] + 0.1, r"$\alpha$", color="#52514e", fontsize=7)
    for y, f in zip(ys, frej):
        a2.text(f + 0.002, y, f"{f:.3f}", va="center", fontsize=6.5,
                color="#52514e")
    a2.set_yticks(ys, ["" for _ in ys])
    a2.set_xlim(0, 0.075)
    a2.set_xlabel("false-rejection rate")
    a2.set_title("(b) validity margin", loc="left")
    fig.tight_layout()
    fig.savefig(HERE / "Fig3.pdf"); fig.savefig(HERE / "Fig3.png", dpi=150)
    plt.close(fig)


def fig4_routing():
    """Cost-heterogeneity study (stdlib prototype numbers, 20:1 spread)."""
    policies = ["Kelly", "round-robin", "cost-blind"]
    det = [0.864, 0.596, 0.766]
    cost = [12.2, 16.6, 22.6]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.0))
    ys = range(len(policies))[::-1]
    a1.barh(ys, det, height=0.55, color=BLUE)
    for y, d in zip(ys, det):
        a1.text(d + 0.02, y, f"{d:.2f}", va="center", fontsize=6.5,
                color="#52514e")
    a1.set_yticks(ys, policies)
    a1.set_xlim(0, 1.05)
    a1.set_xlabel("detection rate")
    a1.set_title("(a) power", loc="left")

    a2.barh(ys, cost, height=0.55, color=BLUE)
    for y, c in zip(ys, cost):
        a2.text(c + 0.5, y, f"{c:.1f}", va="center", fontsize=6.5,
                color="#52514e")
    a2.set_yticks(ys, ["" for _ in ys])
    a2.set_xlim(0, 26)
    a2.set_xlabel("budget units per rejection")
    a2.set_title("(b) cost per rejection", loc="left")
    fig.tight_layout()
    fig.savefig(HERE / "Fig4.pdf"); fig.savefig(HERE / "Fig4.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig1_trajectories()
    fig2_alpha()
    fig3_ablation()
    fig4_routing()
    print("wrote", ", ".join(f"Fig{i}.pdf" for i in range(1, 5)))

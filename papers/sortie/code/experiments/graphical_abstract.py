"""Graphical abstract for the EJOR submission (submission_ejor/graphical_abstract.{pdf,png}).

Run from papers/sortie/code:  python experiments/graphical_abstract.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import common as cm  # noqa: E402
from sortie.sim import simulate  # noqa: E402
from sortie.tsp import tour_stats  # noqa: E402

RES = os.path.join(HERE, "..", "..", "results")
OUT = os.path.join(HERE, "..", "..", "submission_ejor")
DET, SOR = "#c0392b", "#2471a3"

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})

# (a) planned versus realised workloads of the Det range plan, X641-n30-b0, B = 1.10
e3 = pd.read_csv(os.path.join(RES, "e3_main.csv"))
row = e3[(e3.instance == "X641-n30-b0") & (e3.measure == "range") & (e3.B == 1.1) & (e3.method == "det")].iloc[0]
routes = json.load(open(os.path.join(RES, row.plan_file)))["routes"]
inst = cm.load_instance(30, 0)
X, Y = simulate(inst, routes, 0.3, 0.15, 20000, cm.inst_seed(30, 0))
Phi = X * Y[:, None]
L = np.array([tour_stats(r, inst.C)[0] for r in routes])
order = np.argsort(L)
ax = axes[0]
parts = ax.violinplot([Phi[:, k] for k in order], positions=range(1, 6), showextrema=False, widths=0.8)
for b in parts["bodies"]:
    b.set_facecolor(DET); b.set_alpha(0.35); b.set_edgecolor(DET)
ax.scatter(range(1, 6), L[order], marker="D", s=60, color="black", zorder=3, label="planned length")
ax.set_xticks(range(1, 6)); ax.set_xticklabels([f"driver {i}" for i in range(1, 6)])
ax.set_ylabel("workload (distance units)")
rng_det = L.max() - L.min()
rng_real = float((Phi.max(axis=1) - Phi.min(axis=1)).mean())
ax.set_title("A deterministic fair plan: what the planner sees vs.\nwhat the drivers experience", fontsize=12)
ax.text(0.03, 0.96, f"planned range {rng_det:.0f}\nexpected realised range {rng_real:.0f} (+{100*(rng_real/rng_det-1):.0f}%)",
        transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#bbbbbb"))
ax.legend(loc="lower right", fontsize=10, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

# (b) gain of ex-post optimisation (SORTIE) over the deterministic plan by budget
e9 = pd.read_csv(os.path.join(RES, "e9_budget.csv"))
p = e9.pivot_table(index=["instance", "measure", "B"], columns="method", values="oos_obj")
imp = 100 * (p.det - p.sortie0) / p.det
g = imp.groupby(level="B")
ax = axes[1]
Bs = np.array(sorted(g.groups))
ax.bar(range(len(Bs)), g.mean().values, color=SOR, alpha=0.85)
for i, (b, v) in enumerate(zip(Bs, g.mean().values)):
    wins = int((imp.xs(b, level="B") > 1e-9).sum())
    ax.text(i, v + 0.15, f"{v:.1f}%\n{wins}/30", ha="center", va="bottom", fontsize=10)
ax.set_xticks(range(len(Bs))); ax.set_xticklabels([f"{b:.2f}" for b in Bs])
ax.set_xlabel("budget: total length allowed / cost-optimal length")
ax.set_ylabel("lower expected realised unfairness (%)")
ax.set_ylim(0, max(g.mean().values) * 1.35)
ax.set_title("Optimising realised (ex-post) fairness:\ngain over the deterministic plan", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Fair vehicle routing with random route durations", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "graphical_abstract.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "graphical_abstract.png"), dpi=150, bbox_inches="tight")
print("written")

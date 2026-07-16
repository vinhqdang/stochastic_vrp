"""Generates Figure 1: a schematic illustration of the running example
(Example 1 in main.tex) used throughout the paper.

This is a THEORY paper with no real geographic instances (unlike the
BATON/TEMPO papers in this repository, which use real OSM road
networks) -- see README.md's "What the paper is (and is not)" section.
So this figure is deliberately an abstract schematic, not a real map:
site positions are placed at radius proportional to each site's
round-trip dispatch time p_i purely for visual spacing, not real
geographic coordinates.

All numeric values plotted here (p, d, w, the optimal dispatch order,
and its completion times) are exactly Example 1's verified numbers
from main.tex / Appendix A's exhaustive check, not independently
re-derived -- kept in sync by construction, since this script does not
recompute them, only lays them out.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Example 1's data (sites 1..4), already verified in main.tex / Appendix A.
sites = [1, 2, 3, 4]
p = {1: 2, 2: 3, 3: 1, 4: 4}
d = {1: 4, 2: 6, 3: 2, 4: 9}
w = {1: 5, 2: 8, 3: 3, 4: 10}
optimal_order = [1, 2, 4]          # S* = {1,2,4}, dispatched in this order
completion = {1: 2, 2: 5, 4: 9}    # cumulative completion times, verified
excluded = [3]                     # site 3 is sacrificed

angle = {1: 70, 2: 160, 3: 250, 4: 350}  # purely for visual spacing

fig, ax = plt.subplots(figsize=(6.4, 6.0))

# Depot
ax.scatter([0], [0], marker="s", s=220, color="black", zorder=5)
ax.annotate("Depot", (0, 0), textcoords="offset points", xytext=(0, 14),
            ha="center", fontsize=11, fontweight="bold")

site_xy = {}
for i in sites:
    r = p[i]
    a = np.deg2rad(angle[i])
    x, y = r * np.cos(a), r * np.sin(a)
    site_xy[i] = (x, y)

# Dispatch path: depot -> site1 -> site2 -> site4 (round trips back to depot
# are omitted visually for clarity; each leg is drawn depot-to-site since a
# round-trip dispatch always returns to the depot before the next leg).
path_color = "#1a6b3c"
for k, i in enumerate(optimal_order):
    x, y = site_xy[i]
    ax.annotate("", xy=(x, y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=path_color, lw=2.4,
                                shrinkA=12, shrinkB=14,
                                connectionstyle=f"arc3,rad={0.12*(k-1)}"))

for i in sites:
    x, y = site_xy[i]
    included = i in optimal_order
    face = "#2e8b57" if included else "#b0b0b0"
    edge = "#1a6b3c" if included else "#707070"
    size = 300 + 40 * w[i]
    ax.scatter([x], [y], s=size, color=face, edgecolor=edge, linewidth=1.8,
               zorder=4, alpha=0.9 if included else 0.55)
    label = (f"site {i}\n$p={p[i]}, d={d[i]}, w={w[i]}$")
    if included:
        label += f"\ncompletes at {completion[i]}"
    else:
        label += "\nsacrificed"
    dx = 0.32 * np.sign(x) if x != 0 else 0.0
    dy = 0.32 if y >= 0 else -0.55
    ax.annotate(label, (x, y), textcoords="offset points",
                xytext=(18 * np.sign(x if x != 0 else 1), 8),
                ha="left" if x >= -0.3 else "right", fontsize=9.3,
                color="black" if included else "#555555")

xs = [0.0] + [xy[0] for xy in site_xy.values()]
ys = [0.0] + [xy[1] for xy in site_xy.values()]
pad = 2.6
ax.set_xlim(min(xs) - pad, max(xs) + pad)
ax.set_ylim(min(ys) - pad - 1.0, max(ys) + pad)
ax.set_aspect("equal")
ax.axis("off")

legend_handles = [
    mpatches.Patch(color="#2e8b57", label="dispatched (optimal $S^\\ast=\\{1,2,4\\}$)"),
    mpatches.Patch(color="#b0b0b0", label="sacrificed (excluded from $S^\\ast$)"),
]
ax.legend(handles=legend_handles, loc="lower center", frameon=False,
          fontsize=9.5, bbox_to_anchor=(0.5, -0.06))

ax.set_title("Schematic illustration of Example 1 (running example)\n"
              "marker size $\\propto$ criticality weight $w_i$; layout is "
              "abstract, not a real road network",
              fontsize=10.5)

fig.tight_layout()
fig.savefig("running_example_schematic.pdf", bbox_inches="tight")
fig.savefig("running_example_schematic.png", dpi=200, bbox_inches="tight")
print("wrote running_example_schematic.pdf/.png")

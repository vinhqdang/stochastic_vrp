"""Generates Figure 5: a real-geography map of the Camp Fire case study
(Section 5.5 / case_study_campfire.py). Every coordinate plotted here
is the same real, cited coordinate used in case_study_campfire.py --
this script only visualizes them, it does not introduce new data.

Longitude is scaled by cos(mean latitude) so the plot is
locally distance-proportional (an equirectangular approximation, fine
at this ~50 km regional scale); this is a schematic geographic plot,
not a projected basemap.
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

GREEN = "#008300"
GRAY = "#8a8a86"
RED = "#e34948"
BLUE = "#2a78d6"

DEPOT = (39.5022, -121.5522, "Depot\n(Oroville, CA)")
IGNITION = (39.81028, -121.43722, "Ignition point\n(PG&E Tower 27/222)")

# name: (lat, lon, population, dispatched_to)
SITES = {
    "Concow":      (39.73722, -121.51444, 710, False),
    "Paradise":    (39.75972, -121.62194, 26218, True),
    "Magalia":     (39.833,   -121.583,   11310, False),
    "Yankee Hill": (39.70361, -121.52222, 333, False),
}

mean_lat = (DEPOT[0] + IGNITION[0]) / 2
km_per_deg_lat = 111.0
km_per_deg_lon = 111.0 * math.cos(math.radians(mean_lat))


def to_km(lat, lon, origin_lat, origin_lon):
    x = (lon - origin_lon) * km_per_deg_lon
    y = (lat - origin_lat) * km_per_deg_lat
    return x, y


origin_lat, origin_lon = DEPOT[0], DEPOT[1]

fig, ax = plt.subplots(figsize=(6.4, 6.6))

# Depot
dx, dy = to_km(DEPOT[0], DEPOT[1], origin_lat, origin_lon)
ax.scatter([dx], [dy], marker="s", s=220, color="black", zorder=5)
ax.annotate(DEPOT[2], (dx, dy), textcoords="offset points", xytext=(16, 6),
            ha="left", fontsize=9.5, fontweight="bold")

# Ignition point
ix, iy = to_km(IGNITION[0], IGNITION[1], origin_lat, origin_lon)
ax.scatter([ix], [iy], marker="*", s=420, color=RED, edgecolor="black",
           linewidth=0.8, zorder=5)
ax.annotate(IGNITION[2], (ix, iy), textcoords="offset points", xytext=(14, -4),
            ha="left", fontsize=9.5, color=RED)

# Dashed lines from ignition to each site (real distance the hazard travels)
for name, (lat, lon, pop, dispatched) in SITES.items():
    sx, sy = to_km(lat, lon, origin_lat, origin_lon)
    ax.plot([ix, sx], [iy, sy], color=RED, linestyle=":", linewidth=1.2,
            alpha=0.6, zorder=1)

# Dispatch arrow: depot -> Paradise (the optimal choice in both scenarios)
px, py = to_km(SITES["Paradise"][0], SITES["Paradise"][1], origin_lat, origin_lon)
ax.annotate("", xy=(px, py), xytext=(dx, dy),
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.4,
                            shrinkA=14, shrinkB=16))

label_offsets = {
    "Concow": (16, 8),
    "Paradise": (0, -34),
    "Magalia": (16, -22),
    "Yankee Hill": (16, -8),
}
label_ha = {"Concow": "left", "Paradise": "center", "Magalia": "left", "Yankee Hill": "left"}

for name, (lat, lon, pop, dispatched) in SITES.items():
    sx, sy = to_km(lat, lon, origin_lat, origin_lon)
    color = GREEN if dispatched else GRAY
    edge = "#1a6b3c" if dispatched else "#666663"
    size = 250 + 0.014 * pop
    ax.scatter([sx], [sy], s=size, color=color, edgecolor=edge,
               linewidth=1.6, alpha=0.9 if dispatched else 0.6, zorder=4)
    label = f"{name}\npop. {pop:,}"
    ax.annotate(label, (sx, sy), textcoords="offset points",
                xytext=label_offsets[name], ha=label_ha[name], fontsize=9.5,
                color="black" if dispatched else "#555552")

ax.set_xlabel("east-west distance from depot (km)")
ax.set_ylabel("north-south distance from depot (km)")
ax.set_aspect("equal")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.2)

xs_all = [dx, ix] + [to_km(lat, lon, origin_lat, origin_lon)[0] for lat, lon, *_ in SITES.values()]
ys_all = [dy, iy] + [to_km(lat, lon, origin_lat, origin_lon)[1] for lat, lon, *_ in SITES.values()]
ax.set_xlim(min(xs_all) - 6, max(xs_all) + 8)
ax.set_ylim(min(ys_all) - 5, max(ys_all) + 7)

legend_handles = [
    mpatches.Patch(color=GREEN, label="dispatched (optimal choice, both scenarios)"),
    mpatches.Patch(color=GRAY, label="not dispatched"),
    plt.Line2D([0], [0], color=RED, linestyle=":", label="real distance from ignition point"),
]
ax.legend(handles=legend_handles, loc="upper left", frameon=True,
          facecolor="white", edgecolor="none", framealpha=0.95, fontsize=8)

ax.set_title("The 2018 Camp Fire case study: real geography\n"
              "(marker size $\\propto$ 2010 census population)", fontsize=11)

fig.tight_layout()
fig.savefig("campfire_map.pdf", bbox_inches="tight")
fig.savefig("campfire_map.png", dpi=200, bbox_inches="tight")
print("wrote campfire_map.pdf/.png")

"""Generates the Camp Fire map figure (Section 5.6 / case_study_campfire.py).
Every coordinate is the same published coordinate used in
case_study_campfire.py; the hazard minutes are the documented NIST
TN 2135 events (minutes after the 06:31 initial dispatch). This script
only visualizes them.

Longitude is scaled by cos(mean latitude) so the plot is locally
distance-proportional (an equirectangular approximation, fine at this
~50 km regional scale); it is a schematic geographic plot, not a
projected basemap.
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
IGNITION = (39.81461, -121.43400, "Ignition\n(NIST origin point)")

# name: (lat, lon, population, hazard minute after 06:31, in the 50 km/h
# depot-spoke optimum)
SITES = {
    "Concow":      (39.73722, -121.51444, 710, "+54 min", False),
    "Paradise":    (39.75972, -121.62194, 26218, "+73 min", True),
    "Magalia":     (39.833,   -121.583,   11310, "+129 min", True),
    "Yankee Hill": (39.70361, -121.52222, 333, "next day", True),
}
ROUTE = ["Concow", "Paradise", "Magalia", "Yankee Hill"]   # chained route

mean_lat = (DEPOT[0] + IGNITION[0]) / 2
km_per_deg_lat = 111.0
km_per_deg_lon = 111.0 * math.cos(math.radians(mean_lat))


def to_km(lat, lon, origin_lat, origin_lon):
    return ((lon - origin_lon) * km_per_deg_lon,
            (lat - origin_lat) * km_per_deg_lat)


origin_lat, origin_lon = DEPOT[0], DEPOT[1]
fig, ax = plt.subplots(figsize=(6.4, 6.6))

dx, dy = to_km(DEPOT[0], DEPOT[1], origin_lat, origin_lon)
ax.scatter([dx], [dy], marker="s", s=220, color="black", zorder=5)
ax.annotate(DEPOT[2], (dx, dy), textcoords="offset points", xytext=(16, 6),
            ha="left", fontsize=9.5, fontweight="bold")

ix, iy = to_km(IGNITION[0], IGNITION[1], origin_lat, origin_lon)
ax.scatter([ix], [iy], marker="*", s=420, color=RED, edgecolor="black",
           linewidth=0.8, zorder=5)
ax.annotate(IGNITION[2], (ix, iy), textcoords="offset points", xytext=(12, 2),
            ha="left", fontsize=9, color=RED)

pts = {n: to_km(v[0], v[1], origin_lat, origin_lon) for n, v in SITES.items()}

# depot spokes (the model's round trips)
for n, (sx, sy) in pts.items():
    ax.plot([dx, sx], [dy, sy], color=GRAY, linestyle="-", linewidth=1.0,
            alpha=0.7, zorder=1)

# chained route: depot -> Concow -> Paradise -> Magalia -> Yankee Hill
path = [(dx, dy)] + [pts[n] for n in ROUTE]
for a, b in zip(path[:-1], path[1:]):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8,
                                linestyle="--", shrinkA=9, shrinkB=11),
                zorder=3)

offsets = {"Concow": (14, 6), "Paradise": (-16, 2), "Magalia": (14, -6),
           "Yankee Hill": (14, -12)}
ha = {"Concow": "left", "Paradise": "right", "Magalia": "left",
      "Yankee Hill": "left"}
for n, (lat, lon, pop, ev, in_opt) in SITES.items():
    sx, sy = pts[n]
    ax.scatter([sx], [sy], s=250 + 0.014 * pop,
               color=GREEN if in_opt else GRAY,
               edgecolor="#1a6b3c" if in_opt else "#666663", linewidth=1.6,
               alpha=0.9 if in_opt else 0.7, zorder=4)
    ax.annotate(f"{n}\npop. {pop:,}\nhazard {ev}", (sx, sy),
                textcoords="offset points", xytext=offsets[n], ha=ha[n],
                fontsize=8.5)

ax.set_xlabel("east-west distance from depot (km)")
ax.set_ylabel("north-south distance from depot (km)")
ax.set_aspect("equal")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.2)
xs = [dx, ix] + [p[0] for p in pts.values()]
ys = [dy, iy] + [p[1] for p in pts.values()]
ax.set_xlim(min(xs) - 12, max(xs) + 12)
ax.set_ylim(min(ys) - 11, max(ys) + 7)

handles = [
    mpatches.Patch(color=GREEN, label="in the 50 km/h depot-spoke optimum"),
    mpatches.Patch(color=GRAY, label="not in it (Concow)"),
    plt.Line2D([0], [0], color=GRAY, label="depot round trips (the model)"),
    plt.Line2D([0], [0], color=BLUE, linestyle="--",
               label="chained route (comparison model)"),
]
ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0), frameon=True, facecolor="white",
          edgecolor="none", framealpha=0.95, fontsize=7)
ax.set_title("The 2018 Camp Fire case study: geography and hazard times\n"
             "(marker size $\\propto$ 2010 census population)", fontsize=11)
fig.tight_layout()
fig.savefig("campfire_map.pdf", bbox_inches="tight")
fig.savefig("campfire_map.png", dpi=200, bbox_inches="tight")
print("wrote campfire_map.pdf/.png")

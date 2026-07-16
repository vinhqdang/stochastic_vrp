"""case_study_campfire.py -- a real-world-grounded MWHED instance built
from the 2018 Camp Fire (Butte County, California).

This is NOT a synthetic instance: every input number below is either a
real, independently verifiable quantity (geographic coordinates, 2010
US Census populations) or is derived by simple, disclosed arithmetic
from real, cited sources. Two quantities are explicitly modeling
ASSUMPTIONS rather than measured facts, and are labeled as such below:
the depot-to-site travel speed (used to convert real great-circle
distances into a round-trip dispatch time p_i), and the extrapolation
of a hazard-arrival deadline d_i for the two sites without a directly
documented fire-arrival timestamp.

Sources
-------
- Fire-progression timeline (ignition time; Concow buildings-burning
  time; Paradise spot-fire-ignition time): Maranghides, A., Link, E.D.,
  Brown, C.U., Mell, W., Hawks, S., Wilson, M., Brewer, W., Vihnanek,
  R., Walton, W.D. (2021). "A Case Study of the Camp Fire -- Fire
  Progression Timeline." NIST Technical Note 2135. National Institute
  of Standards and Technology. https://doi.org/10.6028/NIST.TN.2135
- Ignition-point coordinates (PG&E Transmission Tower 27/222 near
  Pulga, CA) and the Concow/Paradise timeline cross-reference:
  Wikipedia, "Camp Fire (2018)."
- Site coordinates and 2010 US Census populations for Concow,
  Paradise, Magalia, and Yankee Hill, Butte County, CA: Wikipedia
  articles for each place (each citing the U.S. Census Bureau's 2010
  Decennial Census), independently verified against each article
  directly.
- Depot: the CAL FIRE / Butte County Fire Department's regional
  headquarters (176 Nelson Ave, Oroville, CA) is used as a real,
  pre-existing (not fire-triggered) dispatch depot; since a
  citation-grade building-precise geocode was not available, the
  general Oroville, CA coordinate is used as a stand-in -- adequate
  for this illustration's precision (round-trip times to the nearest
  minute), but not claimed to be building-exact.

Modeling assumptions (NOT measured facts -- disclosed explicitly)
-------------------------------------------------------------------
1. Round-trip dispatch time p_i = 2 * (great-circle distance from
   depot to site i) / (assumed average response speed). Great-circle
   distance is a real, exactly-computable lower bound on the true road
   distance (mountain roads would be longer); two illustrative speed
   assumptions are used (50 km/h, conservative for winding mountain
   roads; 80 km/h, an optimistic highway-speed upper bound), to show
   how sensitive the outcome is to this assumption.
2. Hazard-arrival deadline d_i: for Concow and Paradise, this is taken
   directly from NIST's documented timeline (fire reaches Concow ~52
   minutes after ignition; spot fires ignite in Paradise ~71 minutes
   after ignition). Magalia and Yankee Hill have no directly documented
   arrival timestamp in the sources reviewed, so their deadlines are
   estimated by (a) computing the two real anchor points' implied
   average fire-spread speed from real distances and real times, then
   (b) applying that averaged rate to their own real distance from the
   ignition point. This is an extrapolation, not a documented fact, and
   is reported as such in the paper.

This script is self-contained and does not import anything from
BATON/TEMPO or their real-world datasets (Amazon LMRRC, OSM city
networks) -- it uses an entirely separate, independently sourced real
dataset, consistent with this paper's deliberate independence from the
other two papers in this repository.
"""
from __future__ import annotations

import math

from experiment import solve_exact, solve_edd_naive, solve_fptas, solve_greedy_repair


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


DEPOT = (39.5022, -121.5522)          # Oroville, CA area (CAL FIRE / Butte Co. Fire HQ)
IGNITION = (39.81028, -121.43722)     # PG&E Tower 27/222 near Pulga, CA (verified)

# name: (lat, lon, 2010 census population) -- all independently verified
SITES = {
    "Concow":      (39.73722, -121.51444, 710),
    "Paradise":    (39.75972, -121.62194, 26218),
    "Magalia":     (39.833,   -121.583,   11310),
    "Yankee Hill": (39.70361, -121.52222, 333),
}

# Minutes elapsed since the 6:33 a.m. ignition time (CAL FIRE's identified
# start), for the two sites with a directly NIST-documented arrival time.
ANCHOR_ARRIVAL_MINUTES = {"Concow": 52.0, "Paradise": 71.0}


def build_instance(response_speed_kmh):
    names = list(SITES.keys())
    dist_depot = {n: haversine_km(*DEPOT, SITES[n][0], SITES[n][1]) for n in names}
    dist_ignition = {n: haversine_km(*IGNITION, SITES[n][0], SITES[n][1]) for n in names}
    pop = {n: SITES[n][2] for n in names}

    rates = [dist_ignition[n] / (ANCHOR_ARRIVAL_MINUTES[n] / 60) for n in ANCHOR_ARRIVAL_MINUTES]
    avg_rate_kmh = sum(rates) / len(rates)

    d_minutes = {}
    for n in names:
        if n in ANCHOR_ARRIVAL_MINUTES:
            d_minutes[n] = ANCHOR_ARRIVAL_MINUTES[n]
        else:
            d_minutes[n] = 60 * dist_ignition[n] / avg_rate_kmh

    p_minutes = {n: round(2 * dist_depot[n] / response_speed_kmh * 60) for n in names}
    d_int = {n: round(d_minutes[n]) for n in names}

    order = sorted(names, key=lambda n: d_int[n])
    p_list = [p_minutes[n] for n in order]
    d_list = [d_int[n] for n in order]
    w_list = [pop[n] for n in order]
    return order, p_list, d_list, w_list, avg_rate_kmh


def main():
    for speed_kmh in (50, 80):
        order, p, d, w, avg_rate = build_instance(speed_kmh)
        print(f"=== Response speed = {speed_kmh} km/h "
             f"(derived hazard spread rate = {avg_rate:.2f} km/h) ===")
        for i, n in enumerate(order):
            feas = "OK" if p[i] <= d[i] else "INFEASIBLE ALONE"
            print(f"  {n:12s} p={p[i]:3d} d={d[i]:3d} w={w[i]:6d}  [{feas}]")

        opt, inc = solve_exact(p, d, w)
        naive, ninc = solve_edd_naive(p, d, w)
        rep, rinc = solve_greedy_repair(p, d, w)
        fp, finc = solve_fptas(p, d, w, 0.1)
        print(f"  Exact optimum   W* = {opt:>8.0f}  dispatched = {[order[i] for i in inc]}")
        print(f"  Naive EDD          = {naive:>8.0f}  dispatched = {[order[i] for i in ninc]}")
        print(f"  Greedy repair      = {rep:>8.0f}  dispatched = {[order[i] for i in rinc]}")
        print(f"  FPTAS (eps=0.1)    = {fp:>8.0f}  dispatched = {[order[i] for i in finc]}")
        print()


if __name__ == "__main__":
    main()

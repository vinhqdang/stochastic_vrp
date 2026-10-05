"""case_study_campfire.py -- a real-world-grounded MWHED instance built
from the 2018 Camp Fire (Butte County, California). Revised version.

Every input is either a real, independently checkable quantity
(coordinates, 2010 US Census populations) or derived by disclosed
arithmetic from a cited source; two quantities are modeling ASSUMPTIONS
(response speed -> round-trip time p_i; extrapolated deadlines for two
sites). See main.tex Section 5.5.

What changed relative to the submitted version
----------------------------------------------
1. Deadline semantics. The model's on-time test is "round trip finished
   by d_i" (the RETURN reading). The narrative of the paper is that a
   crew must REACH a site before the hazard does, which is the ARRIVAL
   reading: the crew is at site i after only a_i = p_i/2 of its p_i
   occupancy, so the equivalent MWHED deadline is d_i' = d_i + p_i/2
   (Remark 2 of the paper). The arrival reading is now primary; the
   return reading is reported for comparison.
2. Assumption-1 preprocessing (delete sites with p_i > d_i) is applied
   before every algorithm, including the naive baseline.
3. A stronger baseline (EDD with admission control) is reported.
4. A Monte-Carlo sensitivity analysis over the uncertain inputs
   (response speed, deadline error, dispatch delay).

Sources
-------
- Fire timeline anchors (Concow ~52 min, Paradise spot fires ~71 min after
  the timeline's time zero): Maranghides et al. (2021), NIST TN 2135.
- Ignition-point coordinates (PG&E Tower 27/222 near Pulga, CA) and site
  coordinates / 2010 Census populations: Wikipedia articles citing the
  U.S. Census Bureau; depot: general Oroville, CA coordinate.

Assumptions (NOT measured facts)
--------------------------------
1. p_i = 2 * great-circle distance(depot, i) / speed; speeds 50 and
   80 km/h (great-circle distance is a lower bound on road distance).
2. d_i for Magalia and Yankee Hill: the average spread rate implied by
   the two anchors, applied to straight-line distance from the ignition
   point (an isotropic extrapolation; the real spread was wind-driven and
   directional, which the sensitivity analysis below stresses).
3. The crew may leave at time zero (no detection/dispatch delay); the
   sensitivity analysis relaxes this.

Self-contained: imports nothing from BATON/TEMPO/PARCEL.
"""
from __future__ import annotations

import math
import random
from collections import Counter

from experiment import (preprocess, solve_edd_naive, solve_edd_skip,
                        solve_exact, solve_fptas, solve_greedy_repair)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


DEPOT = (39.5022, -121.5522)          # Oroville, CA area
IGNITION = (39.81028, -121.43722)     # PG&E Tower 27/222 near Pulga, CA

SITES = {  # name: (lat, lon, 2010 census population)
    "Concow":      (39.73722, -121.51444, 710),
    "Paradise":    (39.75972, -121.62194, 26218),
    "Magalia":     (39.833,   -121.583,   11310),
    "Yankee Hill": (39.70361, -121.52222, 333),
}
ANCHOR = {"Concow": 52.0, "Paradise": 71.0}   # minutes after time zero
NAMES = list(SITES)


def raw_inputs():
    dist_dep = {n: haversine_km(*DEPOT, *SITES[n][:2]) for n in NAMES}
    dist_ign = {n: haversine_km(*IGNITION, *SITES[n][:2]) for n in NAMES}
    rate = sum(dist_ign[n] / (ANCHOR[n] / 60) for n in ANCHOR) / len(ANCHOR)
    d_min = {n: ANCHOR.get(n, 60 * dist_ign[n] / rate) for n in NAMES}
    return dist_dep, d_min, rate


def instance(speed_kmh, semantics="arrival", d_scale=None, delay=0.0,
             dist_dep=None, d_min=None):
    """Return (names, p, d, w) after preprocessing. `semantics` is
    'arrival' (d' = d + p/2) or 'return' (d' = d). `d_scale[n]` scales a
    site's hazard-arrival minute; `delay` shifts departure (minutes)."""
    if dist_dep is None:
        dist_dep, d_min, _ = raw_inputs()
    p_all, d_all, w_all = [], [], []
    for n in NAMES:
        p = round(2 * dist_dep[n] / speed_kmh * 60)
        d = d_min[n] * (d_scale[n] if d_scale else 1.0) - delay
        if semantics == "arrival":
            d = d + p / 2
        p_all.append(max(1, p))
        d_all.append(round(d))
        w_all.append(SITES[n][2])
    pp, dd, ww, keep = preprocess(p_all, d_all, w_all)
    return [NAMES[i] for i in keep], pp, dd, ww


def report():
    dist_dep, d_min, rate = raw_inputs()
    print(f"derived hazard spread rate = {rate:.2f} km/h; "
          "extrapolated arrival minutes: "
          + ", ".join(f"{n}={d_min[n]:.0f}" for n in NAMES))
    for sem in ("arrival", "return"):
        for speed in (50, 80):
            names, p, d, w = instance(speed, sem)
            print(f"\n=== {sem} reading, {speed} km/h ===")
            all_p = [round(2 * dist_dep[n] / speed * 60) for n in NAMES]
            for n in NAMES:
                i = NAMES.index(n)
                one_way = all_p[i] / 2
                tag = "kept" if n in names else "DELETED (p>d)"
                print(f"  {n:12s} p={all_p[i]:3d} one-way={one_way:5.1f} "
                      f"d_arrival={d_min[n]:4.0f} w={SITES[n][2]:6d}  [{tag}]")
            res = {
                "Exact optimum": solve_exact(p, d, w),
                "FPTAS eps=0.1": solve_fptas(p, d, w, 0.1),
                "Greedy repair": solve_greedy_repair(p, d, w),
                "EDD with skip": solve_edd_skip(p, d, w),
                "Naive EDD": solve_edd_naive(p, d, w),
            }
            for label, (v, S) in res.items():
                print(f"  {label:14s} {v:8.0f}  {[names[i] for i in sorted(S)]}")


def sensitivity(draws=5000, seed=20260702):
    """Monte-Carlo over uncertain inputs under the arrival reading.
    Speed ~ U[40, 90] km/h; anchored deadlines scaled by U[0.85, 1.15],
    extrapolated ones by U[0.70, 1.30]; dispatch delay ~ U[0, 20] min.
    For each draw compare the plan that is optimal for the NOMINAL
    instance (80 km/h, no delay) against the optimum of the drawn one."""
    rng = random.Random(seed)
    dist_dep, d_min, _ = raw_inputs()
    nom_names, np_, nd, nw = instance(80, "arrival")
    nom_val, nom_set = solve_exact(np_, nd, nw)
    nominal_plan = [nom_names[i] for i in sorted(nom_set)]
    plans = {"nominal-optimal plan": nominal_plan,
             "Paradise-only plan": ["Paradise"]}
    sets = Counter()
    retained = {k: [] for k in plans}
    on_time = {k: 0 for k in plans}
    paradise_in = 0
    for _ in range(draws):
        speed = rng.uniform(40, 90)
        scale = {n: (rng.uniform(0.85, 1.15) if n in ANCHOR
                     else rng.uniform(0.70, 1.30)) for n in NAMES}
        delay = rng.uniform(0, 20)
        names, p, d, w = instance(speed, "arrival", scale, delay,
                                  dist_dep, d_min)
        if not names:
            sets["(nothing reachable)"] += 1
            continue
        v, S = solve_exact(p, d, w)
        sets["+".join(sorted(names[i] for i in S)) or "(empty)"] += 1
        paradise_in += "Paradise" in [names[i] for i in S]
        for label, plan_names in plans.items():
            # dispatch the plan's sites in EDD order; count weight on time
            idx = sorted(names.index(n) for n in plan_names if n in names)
            c, got, all_ok = 0, 0, len(idx) == len(plan_names)
            for i in idx:
                c += p[i]
                if c <= d[i]:
                    got += w[i]
                else:
                    all_ok = False
            retained[label].append(got / v if v else 1.0)
            on_time[label] += all_ok
    print(f"\n=== Sensitivity ({draws} draws, arrival reading) ===")
    print(f"nominal plan (80 km/h, no delay): {nominal_plan}, weight {nom_val:.0f}")
    for s_, c in sets.most_common():
        print(f"  optimal set {s_:32s} {c / draws:6.1%}")
    print(f"  Paradise in optimal set:         {paradise_in / draws:6.1%}")
    for label in plans:
        r = sorted(retained[label])
        print(f"  {label:22s} fully on time {on_time[label] / draws:6.1%}; "
              f"share of drawn optimum retained: mean {sum(r) / len(r):.3f}, "
              f"5th pct {r[int(0.05 * len(r))]:.3f}")


if __name__ == "__main__":
    report()
    sensitivity()

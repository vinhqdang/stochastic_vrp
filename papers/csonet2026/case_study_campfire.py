"""case_study_campfire.py -- a real-world-grounded MWHED instance built
from the 2018 Camp Fire (Butte County, California). Revision 2.

Every hazard time is now a DOCUMENTED clock time from NIST TN 2135
(Maranghides et al., 2021); nothing is extrapolated. Quantities that are
modelling assumptions (response speed -> round-trip time, time zero =
initial dispatch, one event type per community) are listed below.

Documented events (all clock times from NIST TN 2135)
-----------------------------------------------------
- Concow    8 Nov 07:25  first structures burning / spot fires igniting
                         (NIST p. xviii; Table 11)
- Paradise  8 Nov 07:44  first spot fires arrive (NIST p. xviii; the
                         detailed sections give 07:49 for the first 911
                         call about a spot fire; both lie inside the
                         sensitivity range below)
- Magalia   8 Nov 08:40  spot fires in Old Magalia, the earliest-affected
                         part of the community (NIST Table 32; most of
                         Magalia burned much later)
- Yankee Hill 9 Nov 08:40 fire "well established" and running through the
                         community (NIST Table 31): about 26 h after
                         ignition, so effectively unconstrained
Time zero = 06:31 on 8 Nov, the initial dispatch time in NIST's dispatch
log (the ignition estimate is ~06:20, the first 911 call 06:25).
Census (2010) populations are the proxy weights; coordinates are the
communities' published centroids; the depot (Oroville, CA) is a modelling
choice.

Assumptions (NOT measured facts)
--------------------------------
1. p_i = 2 * rho * great-circle distance(depot, i) / speed, rho = 1 in
   the base case (great-circle is a lower bound on road distance);
   speeds 50 and 80 km/h.
2. The crew may leave at time zero; the sensitivity analysis adds a
   departure delay.
3. One event type per community is used as the hazard-arrival time, but
   the events are not identical across communities (first structures
   burning, first spot fires, spot fires in a part of the community,
   fire well established).
4. Arrival reading: a site is protected if the crew ARRIVES before the
   event, so the MWHED deadline is d_i = h_i + p_i/2.

Self-contained: imports nothing from BATON/TEMPO/PARCEL.
"""
from __future__ import annotations

import itertools
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


DEPOT = (39.5022, -121.5522)          # Oroville, CA area (modelling choice)

SITES = {  # name: (lat, lon, 2010 census population)
    "Concow":      (39.73722, -121.51444, 710),
    "Paradise":    (39.75972, -121.62194, 26218),
    "Magalia":     (39.833,   -121.583,   11310),
    "Yankee Hill": (39.70361, -121.52222, 333),
}
T0 = 6 * 60 + 31                       # 06:31 on 8 Nov: initial dispatch
EVENT = {                              # clock minute of the documented event
    "Concow":      7 * 60 + 25,
    "Paradise":    7 * 60 + 44,
    "Magalia":     8 * 60 + 40,
    "Yankee Hill": 24 * 60 + 8 * 60 + 40,   # 9 Nov 08:40
}
HAZ = {n: EVENT[n] - T0 for n in EVENT}      # minutes after time zero
ANCHOR = {"Concow": HAZ["Concow"], "Paradise": HAZ["Paradise"]}
NAMES = list(SITES)


def raw_inputs():
    dist_dep = {n: haversine_km(*DEPOT, *SITES[n][:2]) for n in NAMES}
    d_min = {n: float(HAZ[n]) for n in NAMES}
    return dist_dep, d_min, None


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
    dist_dep, d_min, _ = raw_inputs()
    print("documented hazard minutes after time zero (06:31): "
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


def route_optimum(speed, rho=1.0, d_scale=None, delay=0.0, dist_dep=None,
                  d_min=None):
    """Chained-route (deadline-TSP) comparison model on the same data.

    One crew leaves the depot at `delay` minutes after time zero and visits
    an ordered subset of sites by direct travel between consecutive sites
    (great-circle distance times the road factor `rho`, at `speed` km/h);
    a site is protected iff the crew ARRIVES no later than its hazard
    minute. Exhaustive over all ordered subsets (n = 4). Returns
    (protected weight, visiting order).
    """
    if dist_dep is None:
        dist_dep, d_min, _ = raw_inputs()
    best = (0, ())
    for k in range(1, len(NAMES) + 1):
        for perm in itertools.permutations(NAMES, k):
            t, pos, got = delay, None, 0
            for n in perm:
                km = (dist_dep[n] if pos is None
                      else haversine_km(*SITES[pos][:2], *SITES[n][:2]))
                t += rho * km / speed * 60
                pos = n
                hz = d_min[n] * (d_scale[n] if d_scale else 1.0)
                if t <= hz + 1e-9:
                    got += SITES[n][2]
            if got > best[0]:
                best = (got, perm)
    return best


def spoke_optimum(speed, rho=1.0):
    """The paper's depot-spoke MWHED optimum (arrival reading) with the
    same road factor (p_i = 2 * rho * great-circle / speed)."""
    dist_dep, d_min, _ = raw_inputs()
    scaled = {n: rho * dist_dep[n] for n in NAMES}
    names, p, d, w = instance(speed, "arrival", dist_dep=scaled, d_min=d_min)
    if not names:
        return 0, []
    v, S = solve_exact(p, d, w)
    return v, [names[i] for i in sorted(S)]


def route_vs_spoke():
    """Table: depot-spoke optimum versus chained-route optimum."""
    print("\n=== Depot-spoke model vs chained route (arrival reading) ===")
    total = sum(SITES[n][2] for n in NAMES)
    for rho in (1.0, 1.3, 1.6):
        for speed in (50, 80):
            sv, ss = spoke_optimum(speed, rho)
            rv, rs = route_optimum(speed, rho)
            print(f"  road factor {rho:.1f}, {speed} km/h: spoke {sv:6.0f} "
                  f"{ss}   route {rv:6d} {list(rs)}   (total {total})")


def tipping(speed):
    """Smallest multiplier x of Paradise's weight at which Paradise leaves
    the (spoke-model) optimum, by bisection; and the multiplier at which
    the optimal set changes at all."""
    dist_dep, d_min, _ = raw_inputs()
    names, p, d, w = instance(speed, "arrival")
    ip = names.index("Paradise")

    def opt_set(x):
        ww = list(w)
        ww[ip] = w[ip] * x
        v, S = solve_exact(p, d, ww)
        return frozenset(names[i] for i in S)

    base = opt_set(1.0)
    lo, hi = 0.0, 1.0
    if "Paradise" in opt_set(1e-6):
        return base, None
    for _ in range(40):
        mid = (lo + hi) / 2
        if "Paradise" in opt_set(mid):
            hi = mid
        else:
            lo = mid
    return base, hi


def draw_inputs(rng, correlated):
    speed = rng.uniform(40, 90)
    if correlated:
        common = rng.uniform(0.70, 1.30)
        scale = {n: common * rng.uniform(0.95, 1.05) for n in NAMES}
    else:
        scale = {n: (rng.uniform(0.85, 1.15) if n in ANCHOR
                     else rng.uniform(0.70, 1.30)) for n in NAMES}
    return speed, scale, rng.uniform(0, 20)


def run_plan(plan_names, speed, scale, delay, dist_dep, d_min):
    """On-time weight of a fixed visiting sequence (sites visited in the
    order of NAMES) under one draw; sites deleted by Assumption 1 are
    skipped. Returns (weight, every planned site on time, optimum)."""
    names, p, d, w = instance(speed, "arrival", scale, delay, dist_dep, d_min)
    if not names:
        return 0, False, 0
    v, S = solve_exact(p, d, w)
    idx = sorted(names.index(n) for n in plan_names if n in names)
    c, got, ok = 0, 0, len(idx) == len(plan_names)
    for i in idx:
        c += p[i]
        if c <= d[i]:
            got += w[i]
        else:
            ok = False
    return got, ok, v


def sample_average_plan(correlated, seed=7, draws=2000):
    """The subset of sites (visited in NAMES order) with the largest
    average on-time weight over an independent training sample."""
    rng = random.Random(seed)
    dist_dep, d_min, _ = raw_inputs()
    train = [draw_inputs(rng, correlated) for _ in range(draws)]
    best, best_val = None, -1.0
    for k in range(1, len(NAMES) + 1):
        for sub in itertools.combinations(NAMES, k):
            val = sum(run_plan(sub, sp, sc, de, dist_dep, d_min)[0]
                      for sp, sc, de in train) / draws
            if val > best_val:
                best, best_val = list(sub), val
    return best, best_val


def nominal_plan_at(speed):
    """Optimal plan of the nominal instance (no delay, no scaling) at the
    given response speed, as a list of site names in NAMES order."""
    names, p, d, w = instance(speed, "arrival")
    v, S = solve_exact(p, d, w)
    return [names[i] for i in sorted(S)]


def plan_threshold():
    """Smallest response speed (km/h, 0.01 grid) from which the nominal
    optimum includes Concow."""
    sp = 40.0
    while sp <= 90.0:
        if "Concow" in nominal_plan_at(sp):
            return sp
        sp = round(sp + 0.01, 2)
    return None


def sensitivity(draws=20000, seed=20260702, correlated=False):
    """Monte-Carlo over uncertain inputs under the arrival reading.

    Independent mode: speed ~ U[40, 90] km/h; anchored hazard minutes
    (Concow, Paradise) scaled by U[0.85, 1.15] and those of Magalia and
    Yankee Hill, whose documented events are less sharply defined, by
    U[0.70, 1.30];
    dispatch delay ~ U[0, 20] min. Correlated mode: ONE common factor
    U[0.70, 1.30] scales every hazard minute (a systematic bias of the
    fire model), plus an independent U[0.95, 1.05] per site.

    Plan evaluation protocol (stated explicitly):
      * the plan is a FIXED visiting sequence (depot-spoke model: every
        visit is a round trip and a late visit still consumes its time);
        a site of the plan that Assumption 1 deletes in a draw is
        simply not visited;
      * 'fixed': the plan is executed as is, with sites that Assumption 1
        deletes in the draw skipped (a hopeless first visit does not
        consume time; for these data this coincides with a
        skip-if-late rule, because Concow can only be late when its round
        trip exceeds its deadline);
      * 'literal': every site of the plan is visited in order even when
        its round trip alone exceeds its deadline (no deletion), so a
        hopeless visit still consumes its time.
    Reported shares are with Monte-Carlo standard errors.
    """
    rng = random.Random(seed)
    dist_dep, d_min, _ = raw_inputs()
    nom_names, np_, nd, nw = instance(80, "arrival")
    nom_val, nom_set = solve_exact(np_, nd, nw)
    nominal_plan = [nom_names[i] for i in sorted(nom_set)]
    saa_plan, saa_val = sample_average_plan(correlated)
    plans = {"nominal plan @80 km/h": nominal_plan,
             "nominal plan @65 km/h": nominal_plan_at(65),
             "nominal plan @50 km/h": nominal_plan_at(50),
             "sample-average plan": saa_plan}
    sets = Counter()
    ret = {(k, m): [] for k in plans for m in ("fixed", "literal")}
    on_time = {k: 0 for k in plans}
    paradise_in = 0
    route_full = {1.0: 0, 1.3: 0}
    route_share = {1.0: [], 1.3: []}
    total = sum(SITES[n][2] for n in NAMES)
    for _ in range(draws):
        speed, scale, delay = draw_inputs(rng, correlated)
        for rho in route_full:
            rv, _ = route_optimum(speed, rho, scale, delay, dist_dep, d_min)
            route_full[rho] += rv == total
            route_share[rho].append(rv / total)
        names, p, d, w = instance(speed, "arrival", scale, delay,
                                  dist_dep, d_min)
        if not names:
            sets["(nothing reachable)"] += 1
            continue
        v, S = solve_exact(p, d, w)
        sets["+".join(sorted(names[i] for i in S)) or "(empty)"] += 1
        paradise_in += "Paradise" in [names[i] for i in S]
        for label, plan_names in plans.items():
            idx = sorted(names.index(n) for n in plan_names if n in names)
            c, got, all_ok = 0, 0, len(idx) == len(plan_names)
            for i in idx:
                c += p[i]
                if c <= d[i]:
                    got += w[i]
                else:
                    all_ok = False
            ret[(label, "fixed")].append(got / v if v else 1.0)
            on_time[label] += all_ok
            c, got = 0, 0
            for n in plan_names:
                pn = max(1, round(2 * dist_dep[n] / speed * 60))
                dn = round(d_min[n] * scale[n] - delay + pn / 2)
                c += pn
                if c <= dn:
                    got += SITES[n][2]
            ret[(label, "literal")].append(got / v if v else 1.0)

    def se(prop):
        return (prop * (1 - prop) / draws) ** 0.5

    def mean_se(xs):
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return m, (var / len(xs)) ** 0.5

    mode = "correlated" if correlated else "independent"
    print(f"\n=== Sensitivity ({draws} draws, arrival reading, "
          f"{mode} hazard-minute errors) ===")
    print(f"nominal plan (80 km/h, no delay): {nominal_plan}, "
          f"weight {nom_val:.0f}")
    for k_, v_ in plans.items():
        print(f"plan {k_}: {v_}")
    print(f"sample-average plan (best of all subsets on an independent "
          f"2000-draw sample): {saa_plan}, mean weight {saa_val:.0f}")
    for s_, c in sets.most_common():
        print(f"  optimal set {s_:32s} {c / draws:6.1%} "
              f"(se {se(c / draws):.1%})")
    print(f"  Paradise in optimal set:         {paradise_in / draws:6.1%}")
    for label in plans:
        f = on_time[label] / draws
        mf, sf = mean_se(ret[(label, 'fixed')])
        ms, ss = mean_se(ret[(label, 'literal')])
        r = sorted(ret[(label, "fixed")])
        print(f"  {label:22s} fully on time {f:6.1%} (se {se(f):.1%}); "
              f"retained, fixed route: mean {mf:.3f} (se {sf:.3f}), "
              f"5th pct {r[int(0.05 * len(r))]:.3f}; "
              f"literal route: mean {ms:.3f} (se {ss:.3f})")
    for rho in route_full:
        m, s = mean_se(route_share[rho])
        print(f"  chained-route optimum, road factor {rho:.1f}: all four "
              f"protected in {route_full[rho] / draws:6.1%} of draws; "
              f"mean protected share {m:.3f} (se {s:.3f})")


if __name__ == "__main__":
    report()
    route_vs_spoke()
    for sp in (50, 80):
        base, x = tipping(sp)
        wp = SITES["Paradise"][2]
        if x is None:
            msg = "Paradise stays in the optimum for every positive weight"
        else:
            msg = (f"Paradise leaves the optimum when its weight falls "
                   f"below {x:.3f} x {wp} = {x * wp:.0f} people")
        print(f"\nTipping, {sp} km/h: optimum {sorted(base)}; {msg}")
    print(f"\nthe nominal optimum includes Concow from {plan_threshold()} km/h")
    sensitivity()
    sensitivity(correlated=True)

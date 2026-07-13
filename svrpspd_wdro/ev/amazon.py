"""amazon.py — adapter for the 2021 Amazon Last-Mile Routing Research
Challenge data set (Merchan et al., Transportation Science 58(1), 2024,
DOI 10.1287/trsc.2022.1173; CC-BY-4.0, AWS Open Data bucket
amazon-last-mile-challenges).

A pilot slice lives in data/AmazonLMC/pilot_*.json (25 evaluation
routes, 5 per metro: LA, Boston, Seattle, Chicago, Austin), extracted
by streaming the monolithic bucket files. Each route provides REAL
inter-stop travel-time matrices (the planner's forecast = TEMPO's null
P0), real per-package planned service times (dwell), real package
volumes (demand) and the vehicle capacity.

The dataset is delivery-only and single-realization per route, so:
- the stop sequence is built by our own NN+2-opt over Amazon's travel
  times from the station (we need A plan, not the driver's sequence);
- day-to-day dispersion is not identifiable per stop; we use the
  cross-stop dispersion as its proxy (pilot convention, noted in the
  paper's limitations).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .world import DayParams
from .replan import resequence_nn2opt

DATA = Path(__file__).resolve().parent.parent / "data" / "AmazonLMC"


def load_pilot():
    routes = json.loads((DATA / "pilot_route_data.json").read_text())
    tts = json.loads((DATA / "pilot_travel_times.json").read_text())
    pkgs = json.loads((DATA / "pilot_package_data.json").read_text())
    return routes, tts, pkgs


def route_params(rid, routes, tts, pkgs, sig_T=0.25) -> DayParams:
    """Build a DayParams for one Amazon route."""
    r = routes[rid]
    tt = tts[rid]
    pk = pkgs[rid]
    stops = r["stops"]
    station = next(s for s, v in stops.items() if v["type"] == "Station")
    custs = [s for s, v in stops.items() if v["type"] != "Station"]

    # index map + seconds matrix over [station] + customers
    ids = [station] + custs
    idx = {s: i for i, s in enumerate(ids)}
    n = len(ids)
    D = np.zeros((n, n))
    for a in ids:
        row = tt[a]
        for b in ids:
            D[idx[a], idx[b]] = row[b]

    # our plan: NN+2-opt from the station over Amazon's travel times
    order = resequence_nn2opt(0, list(range(1, n)), D)

    tau = np.array([D[a, b] for a, b in
                    zip([0] + order[:-1], order)]) / 3600.0   # hours

    # demand: total package volume per stop (litres, negative = dropoff)
    vol = np.zeros(n)
    svc = np.full(n, np.nan)
    for s, packs in pk.items():
        v = sum((p.get("dimensions") or {}).get("depth_cm", 0)
                * (p.get("dimensions") or {}).get("height_cm", 0)
                * (p.get("dimensions") or {}).get("width_cm", 0)
                for p in packs.values()) / 1000.0
        t = sum((p.get("planned_service_time_seconds") or 0)
                for p in packs.values())
        vol[idx[s]] = v
        svc[idx[s]] = t / 3600.0
    svc = np.where(np.isnan(svc), np.nanmedian(svc), svc)

    mu_g = -vol[order]                       # deliveries reduce load
    sig_g = np.maximum(0.35 * vol[order], 1e-3)
    cap_l = float(r["executor_capacity_cm3"]) / 1000.0
    B = cap_l                                # delivery-only slack proxy

    p = DayParams(tau=np.maximum(tau, 1e-4), mu_g=mu_g, sig_g=sig_g,
                  B=B, sig_T=sig_T,
                  dwell_a=float(np.mean(svc[order])), dwell_b=0.0,
                  sig_S=float(max(np.std(svc[order]), 1e-3)))
    return p, order, ids

#!/usr/bin/env python3
"""ev_amazon_pilot.py — TEMPO on real Amazon last-mile routes.

25 evaluation routes of the 2021 Amazon Last-Mile Routing Research
Challenge (5 per metro: LA, Boston, Seattle, Chicago, Austin; Merchan
et al. 2024, Transportation Science, CC-BY-4.0). The null model P0 is
built from the data set's own quantities: Amazon's inter-stop travel
times (the planner's forecast), planned service times (dwell), package
volumes (demand). Scenarios from ev/scenarios.py; monitors TEMPO v2 vs
oracle-calibrated CUSUM / Page-Hinkley at TEMPO's realized false rate.

Output: results/results_ev_amazon.csv
Usage (from svrpspd_wdro/): python scripts/ev_amazon_pilot.py [n_days=15]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_WDRO = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_SCRIPTS))

from ev.amazon import load_pilot, route_params                 # noqa: E402
from ev.world import DriftSpec, simulate_route_day             # noqa: E402
from ev.scenarios import SCENARIOS, NULL_NAMES                 # noqa: E402
from ev.eprocess import TempoMonitor, run_day                  # noqa: E402
from ev.baselines import (CusumMonitor, PageHinkleyMonitor,    # noqa: E402
                          calibrate_threshold)

OUT = _WDRO / "results" / "results_ev_amazon.csv"
ALPHA = 0.05
TEMPO_FA = 0.016
N_CAL = 60
SCEN_SUBSET = tuple(SCENARIOS)      # the FULL grid, incl. mild /
                                    # transient / late-onset scenarios


def main():
    n_days = 15
    for a in sys.argv[1:]:
        if a.startswith("n_days="):
            n_days = int(a[7:])
    routes, tts, pkgs = load_pilot()
    rng = np.random.default_rng(20260714)
    rows = []
    for ri, rid in enumerate(sorted(routes)):
        p, order, ids = route_params(rid, routes, tts, pkgs)
        metro = routes[rid]["station_code"][:3]

        cal = [simulate_route_day(p, DriftSpec(kind="none"),
                                  np.random.default_rng(
                                      int(rng.integers(1 << 31))))[0]
               for _ in range(N_CAL)]
        hcm = calibrate_threshold(
            lambda: CusumMonitor(h=np.inf),
            lambda m: max(m.stat.values(), default=0.0), cal, TEMPO_FA)
        hpm = calibrate_threshold(
            lambda: PageHinkleyMonitor(h=np.inf),
            lambda m: m.m - m.m_min, cal, TEMPO_FA)

        for scen in SCEN_SUBSET:
            drift, force_rain = SCENARIOS[scen](p.t0)
            is_null = scen in NULL_NAMES
            stats = {n: dict(fired=0, delay=[], false=0)
                     for n in ("tempo2", "cusum_match", "ph_match")}
            for _ in range(n_days):
                seed = int(rng.integers(1 << 31))
                events, _ = simulate_route_day(
                    p, drift, np.random.default_rng(seed),
                    force_rain=force_rain)
                first_drift = next((i for i, e in enumerate(events)
                                    if e["ctx"]["drifted"]), None)
                hard_idx = next((i for i, e in enumerate(events)
                                 if e["channel"] == "breakdown"
                                 and e.get("x", 0) == 1), None)
                mons = {"tempo2": TempoMonitor(alpha=ALPHA),
                        "cusum_match": CusumMonitor(h=hcm),
                        "ph_match": PageHinkleyMonitor(h=hpm)}
                for name, mon in mons.items():
                    out = run_day(mon, events)
                    if hard_idx is not None and (
                            out["alarm_idx"] is None
                            or out["alarm_idx"] > hard_idx):
                        if not is_null:
                            out = dict(fired=True, alarm_idx=hard_idx)
                    if is_null:
                        stats[name]["false"] += int(out["fired"])
                    elif out["fired"]:
                        if first_drift is not None and \
                                out["alarm_idx"] >= first_drift:
                            stats[name]["fired"] += 1
                            stats[name]["delay"].append(
                                out["alarm_idx"] - first_drift)
                        else:
                            stats[name]["false"] += 1
            for name, s in stats.items():
                rows.append(dict(
                    Route=rid[:12], Metro=metro, n_stops=len(order),
                    scenario=scen, monitor=name,
                    false_rate=s["false"] / n_days,
                    detect_rate=(np.nan if is_null
                                 else s["fired"] / n_days),
                    mean_delay=(float(np.mean(s["delay"]))
                                if s["delay"] else np.nan)))
        print(f"  [{ri+1}/{len(routes)}] {metro} {rid[:12]} "
              f"({len(order)} stops)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print("\n== false-alarm on null scenarios ==")
    print(df[df.scenario.isin(NULL_NAMES)]
          .pivot_table(index="scenario", columns="monitor",
                       values="false_rate").round(3).to_string())
    print("\n== detection rate by scenario ==")
    print(df[~df.scenario.isin(NULL_NAMES)]
          .pivot_table(index="scenario", columns="monitor",
                       values="detect_rate").round(2).to_string())
    print("\n== detection by metro (tempo2) ==")
    print(df[(~df.scenario.isin(NULL_NAMES))
             & (df.monitor == "tempo2")]
          .groupby("Metro").detect_rate.mean().round(2).to_string())


if __name__ == "__main__":
    main()

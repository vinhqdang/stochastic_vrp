#!/usr/bin/env python3
"""ev_grid_eval.py — the comprehensive TEMPO sweep: every instance
family x the full scenario grid.

Instances: Dethloff (40), Salhi-Nagy (14), City real-shop (19) — the
BATON evaluation set, longest cached Det-gate route per instance.
Scenarios: ev/scenarios.py — nulls (incl. the hostile forecast-rain
null), single-factor x severity, ramps, transients, late onsets, and
compound days (storm / rush-crush / black day).

Monitors: TEMPO v2, flat ablation, CUSUM (oracle at alpha and at
TEMPO's realized rate), Page-Hinkley (matched), Bonferroni-fixed,
periodic. Observed breakdowns are hard reactive triggers for everyone.

Output: results/results_ev_grid.csv
Usage (from svrpspd_wdro/): python scripts/ev_grid_eval.py [n_days=15]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_WDRO = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_SCRIPTS))

from dethloff_runner import parse_dethloff                     # noqa: E402
from ev.world import DriftSpec, simulate_route_day, params_from_route  # noqa: E402
from ev.scenarios import SCENARIOS, NULL_NAMES                 # noqa: E402
from ev.eprocess import MasterEProcess, TempoMonitor, run_day  # noqa: E402
from ev.baselines import (CusumMonitor, PeriodicMonitor,       # noqa: E402
                          BonferroniFixed, PageHinkleyMonitor,
                          calibrate_threshold)

PLANS = _WDRO / "results" / "plans"
OUT = _WDRO / "results" / "results_ev_grid.csv"
DATASETS = {
    "Dethloff": _WDRO / "data" / "Dethloff",
    "SalhiNagy": _WDRO / "data" / "SalhiNagy",
    "City": _WDRO / "data" / "City",
}

ALPHA = 0.05
TEMPO_FA = 0.016
ROUTE_HOURS = 6.0
N_CAL = 100


def monitors(hc, hp, hcm, hpm):
    return {
        "tempo2": TempoMonitor(alpha=ALPHA),
        "tempo_flat": MasterEProcess(alpha=ALPHA, use_sensitivity=False),
        "cusum_cal": CusumMonitor(h=hc),
        "cusum_match": CusumMonitor(h=hcm),
        "ph_match": PageHinkleyMonitor(h=hpm),
        "bonferroni": BonferroniFixed(alpha=ALPHA),
        "periodic": PeriodicMonitor(period=40),
    }


def eval_instance(ds, stem, data_dir, n_days, rng):
    doc = json.loads((PLANS / f"{stem}.json").read_text())
    D, dem, Q, n, scale = parse_dethloff(str(data_dir / f"{stem}.vrpspd"))
    gate = "Det" if "Det" in doc["res"] else list(doc["res"])[0]
    route = max(doc["res"][gate]["plan"], key=len)
    p = params_from_route(route, D, dem, Q, scale)
    p.tau = p.tau * (ROUTE_HOURS / max(p.tau.sum(), 1e-9))

    cal = [simulate_route_day(p, DriftSpec(kind="none"),
                              np.random.default_rng(
                                  int(rng.integers(1 << 31))))[0]
           for _ in range(N_CAL)]
    hc = calibrate_threshold(lambda: CusumMonitor(h=np.inf),
                             lambda m: max(m.stat.values(), default=0.0),
                             cal, ALPHA)
    hcm = calibrate_threshold(lambda: CusumMonitor(h=np.inf),
                              lambda m: max(m.stat.values(), default=0.0),
                              cal, TEMPO_FA)
    hp = calibrate_threshold(lambda: PageHinkleyMonitor(h=np.inf),
                             lambda m: m.m - m.m_min, cal, ALPHA)
    hpm = calibrate_threshold(lambda: PageHinkleyMonitor(h=np.inf),
                              lambda m: m.m - m.m_min, cal, TEMPO_FA)

    rows = []
    for scen, make in SCENARIOS.items():
        drift, force_rain = make(p.t0)
        is_null = scen in NULL_NAMES
        stats = {name: dict(fired=0, delay=[], false=0)
                 for name in monitors(hc, hp, hcm, hpm)}
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
            for name, mon in monitors(hc, hp, hcm, hpm).items():
                out = run_day(mon, events)
                if hard_idx is not None and (out["alarm_idx"] is None
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
                Dataset=ds, Instance=stem, scenario=scen, monitor=name,
                n_days=n_days, false_rate=s["false"] / n_days,
                detect_rate=(np.nan if is_null else s["fired"] / n_days),
                mean_delay=(float(np.mean(s["delay"]))
                            if s["delay"] else np.nan)))
    return rows


def main():
    n_days = 15
    for a in sys.argv[1:]:
        if a.startswith("n_days="):
            n_days = int(a[7:])
    rng = np.random.default_rng(20260713)
    rows = []
    for ds, ddir in DATASETS.items():
        stems = sorted(q.stem for q in ddir.glob("*.vrpspd")
                       if (PLANS / f"{q.stem}.json").exists())
        print(f"{ds}: {len(stems)} instances", flush=True)
        for i, stem in enumerate(stems):
            rows += eval_instance(ds, stem, ddir, n_days, rng)
            print(f"  [{i+1}/{len(stems)}] {stem}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)   # checkpoint

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(df)} rows)")

    print("\n== false-alarm rate, plain null vs forecast-rain null ==")
    print(df[df.scenario.isin(NULL_NAMES)]
          .pivot_table(index="scenario", columns="monitor",
                       values="false_rate").round(3).to_string())
    print("\n== detection rate by scenario (pooled over datasets) ==")
    print(df[~df.scenario.isin(NULL_NAMES)]
          .pivot_table(index="scenario", columns="monitor",
                       values="detect_rate").round(2).to_string())
    print("\n== detection rate by dataset (tempo2 vs cusum_match) ==")
    sub = df[~df.scenario.isin(NULL_NAMES)
             & df.monitor.isin(["tempo2", "cusum_match"])]
    print(sub.pivot_table(index="Dataset", columns="monitor",
                          values="detect_rate").round(2).to_string())


if __name__ == "__main__":
    main()

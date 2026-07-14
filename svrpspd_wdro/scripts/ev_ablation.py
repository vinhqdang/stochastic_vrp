#!/usr/bin/env python3
"""ev_ablation.py — component ablation for TEMPO's three power sources:
decision-relevant sensitivity tilt, adaptive (EWMA) betting, and the
dual-regime combination. Isolates each incrementally:

  flat            no tilt, fixed grid, single (product) regime
  +tilt           sensitivity tilt, fixed grid, single regime
  +tilt+adapt     sensitivity tilt, adaptive EWMA bet, single regime
  tempo2          sensitivity tilt, adaptive bet, dual regime (full)

Smaller-scale than the full grid (results_ev_grid.csv) for tractable
runtime: 15 instances (5 per family) x 8 representative scenarios x 15
days. Output: results/results_ev_ablation.csv.
Usage (from svrpspd_wdro/): python scripts/ev_ablation.py [n_days=15]
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
from ev.eprocess import TempoMonitor, run_day                  # noqa: E402

PLANS = _WDRO / "results" / "plans"
OUT = _WDRO / "results" / "results_ev_ablation.csv"
DATASETS = {
    "Dethloff": _WDRO / "data" / "Dethloff",
    "SalhiNagy": _WDRO / "data" / "SalhiNagy",
    "City": _WDRO / "data" / "City",
}
N_PER_FAMILY = 5
SCEN_SUBSET = ("null", "traffic_mild", "traffic_late", "demand_mild",
              "demand_severe", "demand_ramp", "dwell_mild", "rush_crush")
ALPHA = 0.05
ROUTE_HOURS = 6.0


def monitors():
    return {
        "flat":        TempoMonitor(alpha=ALPHA, use_sensitivity=False,
                                    use_adaptive=False, use_dual=False),
        "tilt":        TempoMonitor(alpha=ALPHA, use_sensitivity=True,
                                    use_adaptive=False, use_dual=False),
        "tilt_adapt":  TempoMonitor(alpha=ALPHA, use_sensitivity=True,
                                    use_adaptive=True, use_dual=False),
        "tempo2":      TempoMonitor(alpha=ALPHA, use_sensitivity=True,
                                    use_adaptive=True, use_dual=True),
    }


def eval_instance(ds, stem, data_dir, n_days, rng):
    doc = json.loads((PLANS / f"{stem}.json").read_text())
    D, dem, Q, n, scale = parse_dethloff(str(data_dir / f"{stem}.vrpspd"))
    gate = "Det" if "Det" in doc["res"] else list(doc["res"])[0]
    route = max(doc["res"][gate]["plan"], key=len)
    p = params_from_route(route, D, dem, Q, scale)
    p.tau = p.tau * (ROUTE_HOURS / max(p.tau.sum(), 1e-9))

    rows = []
    for scen in SCEN_SUBSET:
        drift, force_rain = SCENARIOS[scen](p.t0)
        is_null = scen in NULL_NAMES
        stats = {name: dict(fired=0, false=0) for name in monitors()}
        for _ in range(n_days):
            seed = int(rng.integers(1 << 31))
            events, _ = simulate_route_day(
                p, drift, np.random.default_rng(seed),
                force_rain=force_rain)
            first_drift = next((i for i, e in enumerate(events)
                                if e["ctx"]["drifted"]), None)
            for name, mon in monitors().items():
                out = run_day(mon, events)
                if is_null:
                    stats[name]["false"] += int(out["fired"])
                elif out["fired"]:
                    if first_drift is not None and \
                            out["alarm_idx"] >= first_drift:
                        stats[name]["fired"] += 1
                    else:
                        stats[name]["false"] += 1
        for name, s in stats.items():
            rows.append(dict(
                Dataset=ds, Instance=stem, scenario=scen, monitor=name,
                n_days=n_days, false_rate=s["false"] / n_days,
                detect_rate=(np.nan if is_null else s["fired"] / n_days)))
    return rows


def main():
    n_days = 15
    for a in sys.argv[1:]:
        if a.startswith("n_days="):
            n_days = int(a[7:])
    rng = np.random.default_rng(20260714)
    rows = []
    for ds, ddir in DATASETS.items():
        stems = sorted(q.stem for q in ddir.glob("*.vrpspd")
                       if (PLANS / f"{q.stem}.json").exists())[:N_PER_FAMILY]
        print(f"{ds}: {len(stems)} instances", flush=True)
        for i, stem in enumerate(stems):
            rows += eval_instance(ds, stem, ddir, n_days, rng)
            print(f"  [{i+1}/{len(stems)}] {stem}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(df)} rows)")

    print("\n== false-alarm rate (null) ==")
    print(df[df.scenario.isin(NULL_NAMES)]
          .pivot_table(index="scenario", columns="monitor",
                       values="false_rate").round(3).to_string())
    print("\n== detection rate by scenario ==")
    print(df[~df.scenario.isin(NULL_NAMES)]
          .pivot_table(index="scenario", columns="monitor",
                       values="detect_rate").round(2).to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ev_detect_eval.py — TEMPO milestone 1: detection layer on the BATON
evaluation set.

For every Dethloff instance (the same 40 instances and cached Det-gate
ALNS plans as paper 1), take the longest route, build its planning
model (DayParams), and simulate days under the null and under five
injected drift scenarios. Compare monitors:

    tempo        master e-process, sensitivity-tilted bets (ours)
    tempo_flat   master e-process, uniform tilts (ablation)
    cusum        per-channel one-sided CUSUM, h tuned crudely
    bonferroni   fixed-sample z-tests at pre-committed checkpoints
    periodic     replan every P events regardless of data

Metrics per (instance, scenario, monitor): false-alarm rate on null
days, detection rate and mean detection delay (events after the
change-point) on drift days.

Output: results/results_ev_detect.csv  (+ a printed summary).

Usage (from svrpspd_wdro/):  python scripts/ev_detect_eval.py [n_days=25]
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
from ev.eprocess import MasterEProcess, TempoMonitor, run_day  # noqa: E402
from ev.baselines import (CusumMonitor, PeriodicMonitor,       # noqa: E402
                          BonferroniFixed, PageHinkleyMonitor,
                          calibrate_threshold)

DATA = _WDRO / "data" / "Dethloff"
PLANS = _WDRO / "results" / "plans"
OUT = _WDRO / "results" / "results_ev_detect.csv"

ALPHA = 0.05
T_STAR_OFFSET = 2.0        # drift starts 2 h after departure
ROUTE_HOURS = 6.0          # normalize each route's driving time
SCENARIOS = [
    ("none", 0.0),
    ("traffic", 1.6),      # log-mean multiplier post t*
    ("demand", 1.0),       # +1 sd shift
    ("accident", 5.0),     # x5 incident rate
    ("dwell", 3.0),        # +3 sd dwell inflation
    ("breakdown", 50.0),   # p 0.002 -> 0.1 per leg
]


N_CAL = 100          # null days for ORACLE calibration of the foils


TEMPO_FA = 0.016     # TEMPO's realized false rate: matched-rate foils


def monitors(h_cusum: float, h_ph: float, h_cusum_m: float,
             h_ph_m: float):
    return {
        "tempo2": TempoMonitor(alpha=ALPHA),
        "tempo1": MasterEProcess(alpha=ALPHA, use_sensitivity=True),
        "tempo_flat": MasterEProcess(alpha=ALPHA, use_sensitivity=False),
        "cusum_cal": CusumMonitor(h=h_cusum),
        "cusum_match": CusumMonitor(h=h_cusum_m),
        "ph_cal": PageHinkleyMonitor(h=h_ph),
        "ph_match": PageHinkleyMonitor(h=h_ph_m),
        "bonferroni": BonferroniFixed(alpha=ALPHA),
        "periodic": PeriodicMonitor(period=40),
    }


def eval_instance(stem: str, n_days: int, rng: np.random.Generator):
    plan_doc = json.loads((PLANS / f"{stem}.json").read_text())
    D, dem, Q, n, scale = parse_dethloff(str(DATA / f"{stem}.vrpspd"))
    routes = plan_doc["res"]["Det"]["plan"]
    route = max(routes, key=len)
    p = params_from_route(route, D, dem, Q, scale)
    p.tau = p.tau * (ROUTE_HOURS / max(p.tau.sum(), 1e-9))

    # oracle calibration of the non-anytime foils on this instance's
    # null distribution (a courtesy no real dispatcher can extend)
    cal_streams = [simulate_route_day(p, DriftSpec(kind="none"),
                                      np.random.default_rng(
                                          int(rng.integers(1 << 31))))[0]
                   for _ in range(N_CAL)]
    h_cusum = calibrate_threshold(
        lambda: CusumMonitor(h=np.inf), lambda m: max(m.stat.values(),
                                                      default=0.0),
        cal_streams, ALPHA)
    h_ph = calibrate_threshold(
        lambda: PageHinkleyMonitor(h=np.inf),
        lambda m: m.m - m.m_min, cal_streams, ALPHA)
    h_cusum_m = calibrate_threshold(
        lambda: CusumMonitor(h=np.inf), lambda m: max(m.stat.values(),
                                                      default=0.0),
        cal_streams, TEMPO_FA)
    h_ph_m = calibrate_threshold(
        lambda: PageHinkleyMonitor(h=np.inf),
        lambda m: m.m - m.m_min, cal_streams, TEMPO_FA)
    mons = lambda: monitors(h_cusum, h_ph, h_cusum_m, h_ph_m)

    rows = []
    for kind, mag in SCENARIOS:
        drift = DriftSpec(kind=kind, t_star=p.t0 + T_STAR_OFFSET,
                          magnitude=mag)
        stats = {name: dict(fired=0, delay=[], false=0)
                 for name in mons()}
        for day in range(n_days):
            seed = int(rng.integers(1 << 31))
            events, _ = simulate_route_day(
                p, drift, np.random.default_rng(seed))
            # index of first drifted event (for delay scoring)
            first_drift = next((i for i, e in enumerate(events)
                                if e["ctx"]["drifted"]), None)
            # a breakdown is OBSERVED, not inferred: every monitor gets
            # the hard reactive trigger at the first realized breakdown
            hard_idx = next((i for i, e in enumerate(events)
                             if e["channel"] == "breakdown"
                             and e.get("x", 0) == 1), None)
            for name, mon in mons().items():
                out = run_day(mon, events)
                if hard_idx is not None and (out["alarm_idx"] is None
                                             or out["alarm_idx"] > hard_idx):
                    if kind != "none":       # null hard events are recourse,
                        out = dict(fired=True, alarm_idx=hard_idx,  # not false
                                   alarm_clock=events[hard_idx]["clock"],
                                   alarm_on_drifted=bool(
                                       events[hard_idx]["ctx"]["drifted"]))
                if kind == "none":
                    stats[name]["false"] += int(out["fired"])
                elif out["fired"]:
                    if first_drift is not None and \
                            out["alarm_idx"] >= first_drift:
                        stats[name]["fired"] += 1
                        stats[name]["delay"].append(
                            out["alarm_idx"] - first_drift)
                    else:                       # fired before drift began
                        stats[name]["false"] += 1
        for name, s in stats.items():
            rows.append(dict(
                Instance=stem, scenario=kind, monitor=name,
                n_days=n_days,
                false_rate=s["false"] / n_days,
                detect_rate=(s["fired"] / n_days
                             if kind != "none" else np.nan),
                mean_delay=(float(np.mean(s["delay"]))
                            if s["delay"] else np.nan)))
    return rows


def main():
    n_days = 25
    for a in sys.argv[1:]:
        if a.startswith("n_days="):
            n_days = int(a[7:])
    stems = sorted(p.stem for p in DATA.glob("*.vrpspd")
                   if (PLANS / f"{p.stem}.json").exists())
    print(f"{len(stems)} Dethloff instances, {n_days} days per scenario")
    rng = np.random.default_rng(20260712)
    rows = []
    for i, stem in enumerate(stems):
        rows += eval_instance(stem, n_days, rng)
        print(f"  [{i+1}/{len(stems)}] {stem}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")

    # summary: false alarms on null days; detection/delay by scenario
    print("\n== false-alarm rate on null days (target <= %.2f) ==" % ALPHA)
    print(df[df.scenario == "none"].groupby("monitor")
          .false_rate.mean().round(3).to_string())
    print("\n== detection rate ==")
    print(df[df.scenario != "none"]
          .pivot_table(index="scenario", columns="monitor",
                       values="detect_rate").round(2).to_string())
    print("\n== mean detection delay (events after t*) ==")
    print(df[df.scenario != "none"]
          .pivot_table(index="scenario", columns="monitor",
                       values="mean_delay").round(1).to_string())


if __name__ == "__main__":
    main()

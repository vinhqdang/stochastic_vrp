#!/usr/bin/env python3
"""TEMPO-2 prototype driver: fleet-level replanning under drift.

Usage (from svrpspd_wdro/):
    python scripts/tempo2_proto.py [n_inst=10] [days=60] [tune_days=30]
        [workers=4] [out=results/tempo2/proto.csv]

Replans switch each alive route to a precomputed policy library entry
(pickup scale on the grid A_GRID, nearest to the estimate). Limitation:
heavy_del changes deliveries, but a replan only re-scales pickups, so the
heavy_del scenario is evaluated with the pickups-only refit.

For each instance (gate Det, first n_inst cached Dethloff instances):
  * tuning (seed stream 1/4): EveryK's k in {1,2,3,4,6} per C_rp on the
    scenarios null, light, surge20, surge30; Cusum's h on null days
    (z-score alarm rate of 0.05 over the tuning days, no replans).
  * test (seed stream 2 for demands, 3 for replan refits): every scenario,
    C_rp in {0,5,20} and trigger on identical day samples.

Triggers whose decisions do not depend on C_rp are simulated once and
charged C_rp per replan afterwards (exact). CostBased is simulated per
C_rp, sharing a per-day cache of its saving estimates.
"""

from __future__ import annotations

import os
import sys
import time
import zlib

# One BLAS thread per worker process: multithreaded BLAS under a process
# pool oversubscribes the cores and slows the RouteValue builds sharply.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_WDRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_WDRO))

from ev.fleet import (FleetPlan, PolicyLibrary, sample_day,  # noqa: E402
                      simulate_fleet_day, make_drift, Never, EveryK,
                      Continuous, ValueInnovation, ModelFitLR, Cusum,
                      CostBased, Oracle, _z_score)

C_RPS = (0.0, 5.0, 20.0)
K_GRID = (1, 2, 3, 4, 6)
TUNE_SCEN = ("null", "light", "surge20", "surge30")
TEST_SCEN = ("null", "light", "heavy_del", "surge20", "surge30", "surge50",
             "late30")
SCEN_ID = {s: i for i, s in enumerate(
    ("null", "light", "heavy_del", "surge20", "surge30", "surge50", "late30"))}
H_GRID = np.arange(2.0, 20.0001, 0.25)
ALPHA_DAY = 0.05
# Trigger rows whose decision is independent of C_rp.
C_FREE = ("Never", "Continuous", "ValueInnovation", "ModelFitLR", "Cusum",
          "Oracle")


def _inst_id(stem: str) -> int:
    return zlib.crc32(stem.encode()) % 1000003


def _stop_clock_z(fleet, gs):
    """Standardised residuals of one day in stop-clock order (no replans)."""
    zs = []
    for k in range(1, fleet.m_max + 1):
        for r, rt in enumerate(fleet.routes):
            if k <= rt.m:
                o = {"r": r, "k": k, "g": float(gs[r][k - 1])}
                z = _z_score(o, {"fleet": fleet, "a_cur": 1.0})
                if z is not None:
                    zs.append(z)
    return zs


def tune_cusum_h(fleet, tune_days, iid):
    """Pick h so that the day-level alarm rate under the planning law is
    closest to ALPHA_DAY (ties: smaller h)."""
    seqs = []
    for d in range(tune_days):
        gs, _ = sample_day(fleet, "null",
                           np.random.default_rng([1, iid, SCEN_ID["null"], d]))
        seqs.append(_stop_clock_z(fleet, gs))
    best_h, best_gap = float(H_GRID[0]), np.inf
    for h in H_GRID:
        alarms = 0
        for zs in seqs:
            sp = sm = 0.0
            hit = False
            for z in zs:
                sp = max(0.0, sp + z - 0.25)
                sm = max(0.0, sm - z - 0.25)
                if sp > h or sm > h:
                    hit = True
                    break
            alarms += int(hit)
        gap = abs(alarms / max(tune_days, 1) - ALPHA_DAY)
        if gap < best_gap - 1e-12:
            best_gap, best_h = gap, float(h)
    return best_h


def run_instance(path_str, days, tune_days):
    t0 = time.time()
    path = Path(path_str)
    stem = path.stem
    iid = _inst_id(stem)
    fleet = FleetPlan.from_instance(path, "Det")
    lib = PolicyLibrary(fleet, seed=iid)
    h = tune_cusum_h(fleet, tune_days, iid)

    # ---- EveryK tuning (replan charge applied afterwards, exact) ----
    acc = {k: {C: [] for C in C_RPS} for k in K_GRID}
    for scen in TUNE_SCEN:
        sid = SCEN_ID[scen]
        for d in range(tune_days):
            day = sample_day(fleet, scen, np.random.default_rng([1, iid, sid, d]))
            for k in K_GRID:
                res = simulate_fleet_day(lib, day, EveryK(k), 0.0)
                for C in C_RPS:
                    acc[k][C].append((scen, res["recourse_cost"]
                                      + C * res["n_replans"]))
    tuned_k = {}
    for C in C_RPS:
        best, best_k = np.inf, K_GRID[0]
        for k in K_GRID:
            per_scen = []
            for scen in TUNE_SCEN:
                vals = [c for s, c in acc[k][C] if s == scen]
                per_scen.append(np.mean(vals))
            m = float(np.mean(per_scen))
            if m < best - 1e-12:
                best, best_k = m, k
        tuned_k[C] = best_k

    # ---- test ----
    rows = []
    for scen in TEST_SCEN:
        drift = make_drift(scen, fleet)
        sid = SCEN_ID[scen]
        for d in range(days):
            day = sample_day(fleet, drift, np.random.default_rng([2, iid, sid, d]))

            free = {
                "Never": Never(),
                "Continuous": Continuous(),
                "ValueInnovation": ValueInnovation(ALPHA_DAY),
                "ModelFitLR": ModelFitLR(ALPHA_DAY),
                "Cusum": Cusum(h),
                "Oracle": Oracle(drift),
            }
            res_free = {}
            for name, trig in free.items():
                res_free[name] = (trig, simulate_fleet_day(lib, day, trig, 0.0))
            ek = {}
            for C in C_RPS:
                k = tuned_k[C]
                if k not in ek:
                    ek[k] = simulate_fleet_day(lib, day, EveryK(k), 0.0)

            cache = {}
            cb = {}
            for C in C_RPS:
                cb[C] = simulate_fleet_day(
                    lib, day, CostBased(C, cache=cache), C)

            for C in C_RPS:
                base = dict(instance=stem, scenario=scen, C_rp=C, day=d)
                for name in C_FREE:
                    trig, res = res_free[name]
                    res = dict(res)
                    res["cost"] = res["recourse_cost"] + C * res["n_replans"]
                    param = getattr(trig, "param", None)
                    rows.append({**base, "trigger": name, "param": param, **_pick(res)})
                k = tuned_k[C]
                res = dict(ek[k])
                res["cost"] = res["recourse_cost"] + C * res["n_replans"]
                rows.append({**base, "trigger": "EveryK", "param": k, **_pick(res)})
                rows.append({**base, "trigger": "CostBased", "param": C,
                             **_pick(cb[C])})
    tuning = {"instance": stem, "tuned_k": tuned_k, "cusum_h": h,
              "seconds": round(time.time() - t0, 1)}
    return rows, tuning


def _pick(res):
    return {k: res[k] for k in ("cost", "recourse_cost", "n_replans",
                                "breaches", "handoffs", "restocks")}


def list_instances(n_inst):
    plans = _WDRO / "results" / "plans"
    out = []
    for p in sorted((_WDRO / "data" / "Dethloff").glob("*.vrpspd")):
        pj = plans / f"{p.stem}.json"
        if pj.exists() and "Det" in __import__("json").loads(pj.read_text())["res"]:
            out.append(p)
        if len(out) == n_inst:
            break
    return out


def main(argv):
    kw = dict(n_inst=10, days=60, tune_days=30, workers=4,
              out="results/tempo2/proto.csv")
    for a in argv:
        k, v = a.split("=", 1)
        kw[k] = type(kw[k])(v)
    insts = list_instances(int(kw["n_inst"]))
    print(f"instances: {[p.stem for p in insts]}", flush=True)
    t0 = time.time()
    rows, tunings = [], []
    jobs = [(str(p), int(kw["days"]), int(kw["tune_days"])) for p in insts]
    with ProcessPoolExecutor(max_workers=int(kw["workers"])) as ex:
        futs = [ex.submit(run_instance, *j) for j in jobs]
        for f in futs:
            r, t = f.result()
            rows.extend(r)
            tunings.append(t)
            print(f"done {t['instance']} in {t['seconds']} s  "
                  f"tuned_k={t['tuned_k']} cusum_h={t['cusum_h']}", flush=True)

    out = _WDRO / kw["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pd.DataFrame(tunings).to_csv(out.with_name(out.stem + "_tuning.csv"),
                                 index=False)
    print(f"wrote {out} ({len(df)} rows) in {time.time() - t0:.1f} s")

    summ = (df.groupby(["scenario", "C_rp", "trigger"])
              .agg(mean_cost=("cost", "mean"),
                   mean_replans=("n_replans", "mean"))
              .reset_index())
    with pd.option_context("display.max_rows", 500, "display.width", 120):
        print(summ.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])

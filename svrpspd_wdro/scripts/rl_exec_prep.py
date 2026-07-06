#!/usr/bin/env python3
"""
rl_exec_prep.py — Package route data for the RL execution-policy baseline.

Exports, for every route of the Det-gate plans of a set of instances, the
training/test scenario increments and price schedules into one compressed
.npz bundle that the self-contained Colab trainer (rl_exec_train.py)
consumes. Keeping the bundle small: increments as float32, test days
capped at 2000 per route (matching the evaluation protocol).

Usage:
    python scripts/rl_exec_prep.py [out=results/rl_bundle.npz] [max_dethloff=10]
"""

import sys
import json
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_WDRO    = _SCRIPTS.parent
sys.path.insert(0, str(_WDRO))
sys.path.insert(0, str(_SCRIPTS))

from dethloff_runner import parse_dethloff, sample_demands, CV, DIST, SEED
from core.costs import LastMileCosts, route_cost_schedules
from core.otr2 import calibrate_B_empirical_peak

PLANS_DIR = _WDRO / "results" / "plans"


def collect(instances, n_train=1000, n_test=2000):
    costs = LastMileCosts()
    routes = []
    for inst in instances:
        fam = "Dethloff" if not inst.startswith(("HANOI", "HCMC", "NYC",
                                                 "PARIS", "SHANGHAI")) else "City"
        path = _WDRO / "data" / fam / f"{inst}.vrpspd"
        D, dem, Q, n, scale = parse_dethloff(str(path))
        plan_file = PLANS_DIR / f"{inst}.json"
        if not plan_file.exists():
            print(f"  skip {inst}: no persisted plan")
            continue
        plan = json.loads(plan_file.read_text())["res"].get("Det", {}).get("plan")
        if not plan:
            continue
        dbar = dem[:, 0].astype(float)
        pbar = dem[:, 1].astype(float)
        seed = SEED + abs(hash(inst)) % 10_000
        rng = np.random.default_rng(seed)
        dsc_tr = sample_demands(dbar, n, n_train, CV, DIST, rng)
        psc_tr = sample_demands(pbar, n, n_train, CV, DIST, rng)
        rng2 = np.random.default_rng(seed + 99_991)
        dsc_te = sample_demands(dbar, n, n_test, CV, DIST, rng2)
        psc_te = sample_demands(pbar, n, n_test, CV, DIST, rng2)
        for route in plan:
            if not route or len(route) < 3:
                continue
            r = np.array(route)
            B = float(Q - dbar[r].sum())
            g_tr = (psc_tr[:, r] - dsc_tr[:, r]).astype(np.float32)
            if B <= 0:
                B = calibrate_B_empirical_peak(g_tr, alpha=0.10)
            H, E = route_cost_schedules(route, D, scale, costs)
            routes.append(dict(
                inst=inst, m=len(r), B=np.float32(B),
                g_train=g_tr,
                g_test=(psc_te[:, r] - dsc_te[:, r]).astype(np.float32),
                H=H.astype(np.float32), E=E.astype(np.float32)))
        print(f"  {inst}: {sum(1 for x in routes if x['inst'] == inst)} routes")
    return routes


def main():
    out = _WDRO / "results" / "rl_bundle.npz"
    max_deth = 10
    for a in sys.argv[1:]:
        if a.startswith("out="):          out = Path(a[4:])
        elif a.startswith("max_dethloff="): max_deth = int(a[13:])

    deth = sorted(p.stem for p in (_WDRO / "data" / "Dethloff").glob("*.vrpspd"))
    instances = deth[:max_deth] + ["HANOI-100-1", "HCMC-100-1"]
    routes = collect(instances)

    payload = {"n_routes": np.array([len(routes)])}
    for i, rt in enumerate(routes):
        for key in ("g_train", "g_test", "H", "E"):
            payload[f"r{i}_{key}"] = rt[key]
        payload[f"r{i}_B"] = np.array([rt["B"]])
        payload[f"r{i}_inst"] = np.array([rt["inst"]])
    np.savez_compressed(out, **payload)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(routes)} routes)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ev_replan_demo.py — milestone 2 opener: the replanner is pluggable.

Fixes the trigger (TEMPO v2) and swaps the optimizer underneath on
demand-surge days, where re-planning has unambiguous value: loads run
hot, some vehicles head for capacity breaches (priced as emergency
transfers), and a good replan rebalances the remaining customers
across the fleet's residual slack.

Policies compared on identical realized days (per-customer demand is
pre-drawn, so the realization is independent of visit order):

    never                 ride the plan (lower anchor)
    tempo+resequence      TEMPO alarm -> per-vehicle NN+2-opt re-seq
    tempo+rebalance       TEMPO alarm -> fleet 2-regret reinsertion
    tempo+exact           TEMPO alarm -> Gurobi open-TSP per vehicle
    oracle+rebalance      rebalance at the first stop (upper anchor)

Day cost = travel distance (c_km per unit) + F_emg per capacity breach
(emergency transfer: the vehicle empties, pays, continues).

Output: results/results_ev_replan.csv (+ printed summary).
Usage (from svrpspd_wdro/): python scripts/ev_replan_demo.py [n_days=20]
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
from ev.eprocess import TempoMonitor                           # noqa: E402
from ev.replan import (resequence_nn2opt, exact_open_tsp,      # noqa: E402
                       rebalance_regret)

DATA = _WDRO / "data" / "Dethloff"
PLANS = _WDRO / "results" / "plans"
OUT = _WDRO / "results" / "results_ev_replan.csv"

ALPHA = 0.05
SURGE = 0.8            # demand mean shift (in sd units) on surge days
C_KM = 0.1             # travel cost per distance unit (paper-1 economics)
F_EMG = 40.0           # emergency transfer on breach
CV = 0.35


def day_cost(routes, curs, done_dist, W, g, D, B):
    """Cost of finishing the day: drive each route from its current
    node, realize the pre-drawn demands g[c], pay F_EMG and empty on
    breach. Returns (cost, n_breach)."""
    cost = done_dist * C_KM
    nb = 0
    for v, route in enumerate(routes):
        c = curs[v]
        Wv = W[v]
        for nxt in route:
            cost += D[c, nxt] * C_KM
            Wv += g[nxt]
            if Wv > B[v]:
                nb += 1
                cost += F_EMG
                Wv = 0.0
            c = nxt
        cost += D[c, 0] * C_KM
    return cost, nb


def eval_instance(stem, n_days, rng):
    doc = json.loads((PLANS / f"{stem}.json").read_text())
    D, dem, Q, n, scale = parse_dethloff(str(DATA / f"{stem}.vrpspd"))
    D = D / scale                     # back to instance units (~km)
    routes0 = [list(r) for r in doc["res"]["Det"]["plan"] if r]
    dbar = dem[:, 0].astype(float)
    pbar = dem[:, 1].astype(float)
    mu = pbar - dbar
    sig = np.maximum(CV * np.sqrt(pbar ** 2 + dbar ** 2), 1e-6)
    B = [float(Q - dbar[r].sum()) for r in routes0]

    rows = []
    for day in range(n_days):
        drng = np.random.default_rng(int(rng.integers(1 << 31)))
        g = mu + SURGE * sig + drng.normal(0.0, sig)     # surge day
        g[0] = 0.0

        # --- trigger: TEMPO v2 on the pooled demand stream, stops
        # interleaved across vehicles in service order
        mon = TempoMonitor(alpha=ALPHA)
        order = []                       # (v, k) service order round-robin
        mlen = max(len(r) for r in routes0)
        for k in range(mlen):
            for v, r in enumerate(routes0):
                if k < len(r):
                    order.append((v, k))
        alarm_at = None                  # position in `order`
        for i, (v, k) in enumerate(order):
            c = routes0[v][k]
            z = float((g[c] - mu[c]) / sig[c])
            if mon.update(dict(channel="demand", z=z,
                               ctx=dict(B=B[v],
                                        W_prev=0.0, rem_frac=1.0))) \
                    and alarm_at is None:
                alarm_at = i
                break

        def state_at(pos):
            """Fleet state after the first `pos` services."""
            curs = [0] * len(routes0)
            W = [0.0] * len(routes0)
            done = [0] * len(routes0)    # customers served per vehicle
            dist = 0.0
            for (v, k) in order[:pos + 1]:
                c = routes0[v][k]
                dist += D[curs[v], c]
                W[v] += g[c]
                if W[v] > B[v]:          # breach before any replan
                    W[v] = 0.0
                curs[v] = c
                done[v] = k + 1
            return curs, W, done, dist

        # --- policies
        res = {}
        res["never"], res["never_nb"] = day_cost(
            routes0, [0] * len(routes0), 0.0,
            [0.0] * len(routes0), g, D, B)

        variants = []
        if alarm_at is not None:
            variants.append(("tempo", alarm_at))
        variants.append(("oracle", 0))
        for tag, pos in variants:
            curs, W, done, dist = state_at(pos)
            rem = [routes0[v][done[v]:] for v in range(len(routes0))]
            slack = [B[v] - W[v] for v in range(len(routes0))]
            gbar_adj = mu + SURGE * sig          # post-alarm knowledge
            backends = {
                "resequence": lambda: [resequence_nn2opt(curs[v], rem[v], D)
                                       for v in range(len(routes0))],
                "rebalance": lambda: rebalance_regret(curs, rem, slack,
                                                      D, gbar_adj),
                "exact": lambda: [exact_open_tsp(curs[v], rem[v], D)
                                  for v in range(len(routes0))],
            }
            wanted = (("resequence", "rebalance", "exact")
                      if tag == "tempo" else ("rebalance",))
            for bk in wanted:
                new_routes = backends[bk]()
                cost, nb = day_cost(new_routes, curs, 0.0, W, g, D, B)
                res[f"{tag}+{bk}"] = dist * C_KM + cost
                res[f"{tag}+{bk}_nb"] = nb
        if alarm_at is None:
            for bk in ("resequence", "rebalance", "exact"):
                res[f"tempo+{bk}"] = res["never"]
                res[f"tempo+{bk}_nb"] = res["never_nb"]

        rows.append(dict(Instance=stem, day=day, alarmed=alarm_at
                         is not None, **res))
    return rows


def main():
    n_days = 20
    n_inst = 12
    for a in sys.argv[1:]:
        if a.startswith("n_days="):
            n_days = int(a[7:])
        if a.startswith("n_inst="):
            n_inst = int(a[7:])
    stems = sorted(p.stem for p in DATA.glob("*.vrpspd")
                   if (PLANS / f"{p.stem}.json").exists())[:n_inst]
    rng = np.random.default_rng(20260713)
    rows = []
    for i, stem in enumerate(stems):
        rows += eval_instance(stem, n_days, rng)
        print(f"  [{i+1}/{len(stems)}] {stem}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")

    pols = ["never", "tempo+resequence", "tempo+rebalance", "tempo+exact",
            "oracle+rebalance"]
    print(f"\nalarm rate on surge days: {df.alarmed.mean():.2f}")
    print("\n== mean day cost (lower is better) ==")
    base = df["never"].mean()
    for p in pols:
        if p in df:
            sv = 100 * (base - df[p].mean()) / base
            nb = df.get(p + "_nb", pd.Series(np.nan)).mean()
            print(f"  {p:20s} {df[p].mean():8.1f}   saving {sv:5.1f}%   "
                  f"breaches/day {nb:.2f}")


if __name__ == "__main__":
    main()

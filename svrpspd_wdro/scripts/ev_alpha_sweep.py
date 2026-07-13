#!/usr/bin/env python3
"""ev_alpha_sweep.py — empirical witness for THEORY.md Corollary 1:
day cost as a function of the evidence level alpha is U-shaped with an
interior optimum alpha*.

Mechanism: small alpha demands more evidence -> longer detection delay
-> staleness cost on drift days; large alpha alarms casually -> false
replans on on-model days -> unnecessary-detour cost. The optimum trades
Delta_c/g against C_fr exactly as the corollary prices it.

Population: 50% on-model days, 50% demand-surge days (+0.8 sd), the
ev_replan_demo cost harness (0.1/unit travel, 40/breach), trigger =
TEMPO v2 at each alpha, replanner = fleet rebalance.

Output: results/results_ev_alpha.csv + printed curve.
Usage (from svrpspd_wdro/): python scripts/ev_alpha_sweep.py
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
from ev.replan import rebalance_regret                         # noqa: E402
from ev_replan_demo import day_cost, C_KM, F_EMG, CV, SURGE    # noqa: E402

DATA = _WDRO / "data" / "Dethloff"
PLANS = _WDRO / "results" / "plans"
OUT = _WDRO / "results" / "results_ev_alpha.csv"
ALPHAS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.005, 0.001, 1e-4)


def run_day_policy(alpha, surge, routes0, mu, sig, B, D, drng):
    shift = SURGE * sig if surge else 0.0
    g = mu + shift + drng.normal(0.0, sig)
    g[0] = 0.0
    mon = TempoMonitor(alpha=alpha)
    order = [(v, k) for k in range(max(map(len, routes0)))
             for v, r in enumerate(routes0) if k < len(r)]
    alarm_at = None
    for i, (v, k) in enumerate(order):
        c = routes0[v][k]
        z = float((g[c] - mu[c]) / sig[c])
        if mon.update(dict(channel="demand", z=z,
                           ctx=dict(B=B[v], W_prev=0.0, rem_frac=1.0))):
            alarm_at = i
            break
    if alarm_at is None:
        return day_cost(routes0, [0] * len(routes0), 0.0,
                        [0.0] * len(routes0), g, D, B)[0], False
    # roll state to the alarm, then rebalance
    curs = [0] * len(routes0)
    W = [0.0] * len(routes0)
    done = [0] * len(routes0)
    dist = 0.0
    for (v, k) in order[:alarm_at + 1]:
        c = routes0[v][k]
        dist += D[curs[v], c]
        W[v] += g[c]
        if W[v] > B[v]:
            W[v] = 0.0
        curs[v] = c
        done[v] = k + 1
    rem = [routes0[v][done[v]:] for v in range(len(routes0))]
    slack = [B[v] - W[v] for v in range(len(routes0))]
    gbar = mu + (shift if surge else 0.0)
    new_routes = rebalance_regret(curs, rem, slack, D, gbar)
    cost, _ = day_cost(new_routes, curs, 0.0, W, g, D, B)
    return dist * C_KM + cost, True


def main():
    n_days, n_inst = 15, 10
    stems = sorted(p.stem for p in DATA.glob("*.vrpspd")
                   if (PLANS / f"{p.stem}.json").exists())[:n_inst]
    rows = []
    for stem in stems:
        doc = json.loads((PLANS / f"{stem}.json").read_text())
        D, dem, Q, n, scale = parse_dethloff(str(DATA / f"{stem}.vrpspd"))
        D = D / scale
        routes0 = [list(r) for r in doc["res"]["Det"]["plan"] if r]
        dbar = dem[:, 0].astype(float)
        pbar = dem[:, 1].astype(float)
        mu = pbar - dbar
        sig = np.maximum(CV * np.sqrt(pbar ** 2 + dbar ** 2), 1e-6)
        B = [float(Q - dbar[r].sum()) for r in routes0]
        rng = np.random.default_rng(hash(stem) % (1 << 31))
        for day in range(n_days):
            seeds = [int(rng.integers(1 << 31)) for _ in range(2)]
            for alpha in ALPHAS:
                for surge, seed in zip((False, True), seeds):
                    cost, fired = run_day_policy(
                        alpha, surge, routes0, mu, sig, B, D,
                        np.random.default_rng(seed))
                    rows.append(dict(Instance=stem, day=day, alpha=alpha,
                                     surge=surge, cost=cost, fired=fired))
        print(f"  {stem}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print("\n== mean day cost by alpha (50/50 null-surge population) ==")
    piv = df.pivot_table(index="alpha", columns="surge", values="cost")
    piv["mixed"] = piv.mean(axis=1)
    fr = df[~df.surge].groupby("alpha").fired.mean()
    piv["false_replan_rate"] = fr
    print(piv.round(1).sort_index(ascending=False).to_string())


if __name__ == "__main__":
    main()

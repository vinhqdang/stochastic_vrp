"""Per-route baseline ladder for the two-action handoff problem
({continue, hand off}), the setting of direction A.

none   : never hand off (pay emergency cost at breach)
baton  : BATON v2_lsm (LS optimal stopping, isotonic continuation models)
oracle : clairvoyant lower bound
Neural policies plug in by returning per-scenario costs on g_test.
"""
from __future__ import annotations

import numpy as np

from .baton.costs import (fit_lsm_general, simulate_v2_general,
                          oracle_costs_general)
from .data import route_problem


def run_baselines(p: dict) -> dict:
    g_tr, g_te, B, H, E = p["g_train"], p["g_test"], p["B"], p["H"], p["E"]
    cm = fit_lsm_general(g_tr, B, H, E)
    baton = simulate_v2_general(g_te, B, H, E, cm)["mean_cost"]
    # "none": an unreachable-cheap handoff never triggers => emergency only
    none = simulate_v2_general(g_te, B, np.full_like(H, 1e12), E,
                               fit_lsm_general(g_tr, B, np.full_like(H, 1e12), E),
                               H_bill=H)["mean_cost"]
    oracle = float(oracle_costs_general(g_te, B, H, E).mean())
    return dict(none=none, baton=baton, oracle=oracle)


def instance_table(name, n_train=500, n_test=2000, plan_key="Det"):
    rows = []
    for p in route_problem(name, plan_key, n_train, n_test):
        r = run_baselines(p)
        r.update(route=p["route_id"], m=len(p["route"]))
        rows.append(r)
    return rows


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "CON3-0"
    rows = instance_table(name)
    tot = {k: float(np.sum([r[k] for r in rows])) for k in ("none", "baton", "oracle")}
    print(name, "routes", len(rows), {k: round(v, 3) for k, v in tot.items()})

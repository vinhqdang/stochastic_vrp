"""replan.py — pluggable replanning backends for TEMPO's trigger.

TEMPO decides WHEN to replan; these decide WHAT the new plan is. Every
backend implements the same signature so milestone-2 experiments can
cross triggers with optimizers:

    replan(cur, remaining, D) -> new visit order (list of customer ids)
        cur        node index the vehicle is at now
        remaining  customer ids still to serve (this vehicle)
        D          full distance matrix (depot = 0)

Fleet-level backends additionally rebalance customers across vehicles:

    rebalance(curs, remainings, slacks, D, gbar) -> list of orders

Backends:
    resequence_nn2opt   open TSP from the vehicle position, nearest-
                        neighbour construction + 2-opt improvement,
                        return-to-depot leg included. The rolling-
                        horizon workhorse; always available, O(m^2).
    rebalance_regret    pool every vehicle's remaining customers and
                        re-insert by 2-regret cheapest insertion into
                        open routes anchored at the current vehicle
                        positions, respecting each vehicle's remaining
                        expected slack; each route then 2-opt'ed.
    exact_open_tsp      MTZ formulation solved by Gurobi (falls back
                        to resequence beyond `max_n` or without a
                        licence) — the perfect-replanner anchor.

ALNS itself (scripts/dethloff_runner.solve / solve_fast) remains the
planning-time optimizer; for mid-route replans its neighbourhood
assumes depot-rooted routes, so the fleet backend here plays the
warm-started-reconstruction role in its place.
"""

from __future__ import annotations

import numpy as np


# ── single-vehicle: open TSP from current position ──────────────────────────

def _tour_len(D, cur, order, depot=0):
    if not order:
        return float(D[cur, depot])
    L = D[cur, order[0]]
    for a, b in zip(order[:-1], order[1:]):
        L += D[a, b]
    return float(L + D[order[-1], depot])


def _two_opt(D, cur, order, depot=0, max_passes=30):
    order = list(order)
    for _ in range(max_passes):
        improved = False
        n = len(order)
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if _tour_len(D, cur, cand, depot) < \
                        _tour_len(D, cur, order, depot) - 1e-9:
                    order = cand
                    improved = True
        if not improved:
            break
    return order


def resequence_nn2opt(cur, remaining, D, depot=0):
    """Nearest-neighbour from the vehicle position, then 2-opt."""
    rem = list(remaining)
    order = []
    c = cur
    while rem:
        nxt = min(rem, key=lambda x: D[c, x])
        order.append(nxt)
        rem.remove(nxt)
        c = nxt
    return _two_opt(D, cur, order, depot)


def exact_open_tsp(cur, remaining, D, depot=0, max_n=12, time_limit=5.0):
    """Exact open TSP (start cur, visit all, end depot) via Gurobi MTZ.
    Falls back to resequence_nn2opt beyond max_n or without Gurobi."""
    if len(remaining) > max_n:
        return resequence_nn2opt(cur, remaining, D, depot)
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return resequence_nn2opt(cur, remaining, D, depot)
    nodes = [cur] + list(remaining) + [depot]      # 0 = start, last = end
    n = len(nodes)
    try:
        env = gp.Env(params={"OutputFlag": 0})
        m = gp.Model(env=env)
        x = m.addVars(n, n, vtype=GRB.BINARY)
        u = m.addVars(n, lb=0, ub=n)
        m.setObjective(gp.quicksum(
            D[nodes[i], nodes[j]] * x[i, j]
            for i in range(n) for j in range(n) if i != j), GRB.MINIMIZE)
        for i in range(n):
            x[i, i].ub = 0
        m.addConstr(gp.quicksum(x[0, j] for j in range(1, n)) == 1)
        m.addConstr(gp.quicksum(x[i, 0] for i in range(1, n)) == 0)
        m.addConstr(gp.quicksum(x[i, n - 1] for i in range(n - 1)) == 1)
        m.addConstr(gp.quicksum(x[n - 1, j] for j in range(n)) == 0)
        for k in range(1, n - 1):
            m.addConstr(gp.quicksum(x[i, k] for i in range(n - 1)
                                    if i != k) == 1)
            m.addConstr(gp.quicksum(x[k, j] for j in range(1, n)
                                    if j != k) == 1)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)
        m.Params.TimeLimit = time_limit
        m.optimize()
        if m.SolCount == 0:
            return resequence_nn2opt(cur, remaining, D, depot)
        order, c = [], 0
        while True:
            nxt = next(j for j in range(n)
                       if j != c and x[c, j].X > 0.5)
            if nxt == n - 1:
                break
            order.append(nodes[nxt])
            c = nxt
        return order
    except Exception:
        return resequence_nn2opt(cur, remaining, D, depot)


# ── fleet-level: rebalance customers across vehicles ─────────────────────────

def rebalance_regret(curs, remainings, slacks, D, gbar, depot=0):
    """2-regret cheapest insertion of the pooled remaining customers
    into open routes anchored at the current vehicle positions.

    slacks[v]  expected remaining net-load headroom of vehicle v; a
               customer c consumes gbar[c] (its mean net increment,
               clipped at 0 — deliveries free capacity, pickups use it)
    Returns a list of visit orders, one per vehicle; every route is
    2-opt'ed at the end.
    """
    pool = [c for rem in remainings for c in rem]
    routes = [[] for _ in curs]
    cap = [float(s) for s in slacks]

    def ins_cost(v, pos, c):
        seq = [curs[v]] + routes[v] + [depot]
        a, b = seq[pos], seq[pos + 1]
        return float(D[a, c] + D[c, b] - D[a, b])

    while pool:
        best = None                    # (regret, -best_cost, c, v, pos)
        for c in pool:
            need = max(gbar[c], 0.0)
            opts = []
            for v in range(len(curs)):
                if cap[v] < need:
                    continue
                costs = [(ins_cost(v, p, c), v, p)
                         for p in range(len(routes[v]) + 1)]
                if costs:
                    opts.append(min(costs))
            if not opts:               # nobody has slack: cheapest anyway
                opts = [min((ins_cost(v, p, c), v, p)
                            for v in range(len(curs))
                            for p in range(len(routes[v]) + 1))]
            opts.sort()
            regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else 1e9
            cand = (regret, -opts[0][0], c, opts[0][1], opts[0][2])
            if best is None or cand > best:
                best = cand
        _, _, c, v, pos = best
        routes[v].insert(pos, c)
        cap[v] -= max(gbar[c], 0.0)
        pool.remove(c)

    return [_two_opt(D, curs[v], routes[v], depot)
            for v in range(len(curs))]


BACKENDS = {
    "resequence": resequence_nn2opt,
    "exact": exact_open_tsp,
}

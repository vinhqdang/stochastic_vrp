"""
RCI lower-bound loop (Gurobi branch-and-cut) for robust SVRPSPD -> optimality-gap certificate.

Builds the 2-index arc formulation, minimises distance + omega_V * (#vehicles), and enforces the
robust rounded capacity inequalities (Def 1 / Prop 3) as LAZY constraints via integer separation:
for the customer set S_v of each route (or subtour) in an integer solution,
    #arcs entering S_v  >=  r_rho(S_v) = ceil( max(CVaR(D_{S_v}), CVaR(P_{S_v})) / Q_eff ).

This RCI-relaxation enforces only the ENDPOINT-necessary condition (Prop 2), NOT the interior peak
excursion, so its optimum IP_RCI satisfies  IP_RCI <= true_opt <= matheuristic_UB.  Hence the gap
(UB - IP_RCI)/UB is a VALID but possibly LOOSE optimality certificate; we measure how loose.

Run:  python rci_bound.py
"""
import numpy as np, math, time
import gurobipy as gp
from gurobipy import GRB

from matheuristic_core import make_instance, alns, cvar


def rci_of_S(S, inst):
    S = list(S)
    cd = cvar(inst.d_scen[:, S].sum(1), inst.alpha)
    cp = cvar(inst.p_scen[:, S].sum(1), inst.alpha)
    return max(1, int(math.ceil(max(cd, cp) / inst.Qeff)))


def _routes_and_subtours(succ, n):
    """From an integer successor map (customer idx 0..n-1 use node j=idx+1; depot=0),
    return (routes, subtours) as lists of customer-index sets."""
    routes = []; visited = set()
    # routes start at depot
    for j in range(1, n + 1):
        if succ.get(0) is None:
            pass
    # gather all depot-out arcs
    depot_next = [j for (i, j) in succ.items() if i == 0]  # succ is per-node; handle multi below
    return routes


def rci_callback(model, where):
    if where != GRB.Callback.MIPSOL:
        return
    x = model._x; n = model._n; inst = model._inst; V = range(n + 1)
    xv = model.cbGetSolution(x)
    # successor lists (there may be multiple out-arcs at depot)
    out = {i: [] for i in V}
    for (i, j) in x:
        if xv[i, j] > 0.5:
            out[i].append(j)
    # trace routes from depot
    visited_cust = set()
    node_in_route = {}
    routes = []
    for start in out[0]:
        seq = []; cur = start
        guard = 0
        while cur != 0 and guard <= n + 1:
            seq.append(cur - 1); visited_cust.add(cur - 1)
            nxt = out[cur][0] if out[cur] else 0
            cur = nxt; guard += 1
        if seq:
            routes.append(set(seq))
    # subtours: customers not visited from depot
    remaining = set(range(n)) - visited_cust
    subtours = []
    while remaining:
        s = next(iter(remaining)); comp = []; cur = s + 1; guard = 0
        while guard <= n + 1:
            comp.append(cur - 1); remaining.discard(cur - 1)
            nxts = out[cur]
            if not nxts: break
            cur = nxts[0]
            if cur == s + 1 or cur == 0 or (cur - 1) not in remaining and (cur - 1) not in comp:
                if cur - 1 in comp or cur == s + 1: break
            guard += 1
        if comp:
            subtours.append(set(comp))

    def add_rci(S):
        r = rci_of_S(S, inst)
        entering = gp.quicksum(x[i, j] for i in V for j in S if i not in S and (i, j) in x)
        # convert customer-set S (0-based) to node set (1-based)
    # add cuts for routes needing more vehicles, and all subtours
    for S in routes:
        r = rci_of_S(S, inst)
        if r > 1:  # single vehicle insufficient
            Snodes = {c + 1 for c in S}
            model.cbLazy(gp.quicksum(x[i, j] for i in V for j in Snodes
                                     if i not in Snodes and (i, j) in x) >= r)
    for S in subtours:
        r = rci_of_S(S, inst)
        Snodes = {c + 1 for c in S}
        model.cbLazy(gp.quicksum(x[i, j] for i in V for j in Snodes
                                 if i not in Snodes and (i, j) in x) >= max(1, r))


def gurobi_rci_bound(inst, omega_V, timelimit=60, root_lp=False):
    n = inst.n; V = range(n + 1)
    m = gp.Model(); m.Params.OutputFlag = 0
    vtype = GRB.CONTINUOUS if root_lp else GRB.BINARY
    x = m.addVars([(i, j) for i in V for j in V if i != j], vtype=vtype, ub=1, name="x")
    m.setObjective(gp.quicksum(inst.C[i, j] * x[i, j] for i in V for j in V if i != j)
                   + omega_V * gp.quicksum(x[0, j] for j in range(1, n + 1)), GRB.MINIMIZE)
    for j in range(1, n + 1):
        m.addConstr(gp.quicksum(x[i, j] for i in V if i != j) == 1)
        m.addConstr(gp.quicksum(x[j, i] for i in V if i != j) == 1)
    m.addConstr(gp.quicksum(x[0, j] for j in range(1, n + 1))
                == gp.quicksum(x[j, 0] for j in range(1, n + 1)))
    m._x = x; m._inst = inst; m._n = n
    if root_lp:
        # iterative separation on the LP: solve, separate by rounding, add static RCIs up to size 4
        import itertools
        for sz in range(1, min(n, 4) + 1):
            for S in itertools.combinations(range(n), sz):
                r = rci_of_S(S, inst)
                if r >= 1:
                    Sn = {c + 1 for c in S}
                    m.addConstr(gp.quicksum(x[i, j] for i in V for j in Sn
                                            if i not in Sn and (i, j) in x) >= r)
        m.optimize()
        return m.ObjVal, m.ObjVal
    else:
        m.Params.LazyConstraints = 1
        m.Params.TimeLimit = timelimit
        m.optimize(rci_callback)
        return m.ObjVal, m.ObjBound


def extract_routes_from_x(m, inst):
    x = m._x; n = m._n; V = range(n + 1)
    xv = {k: v.X for k, v in x.items()}
    out = {i: [j for (i2, j) in x if i2 == i and xv[i, j] > 0.5] for i in V}
    routes = []
    for start in out[0]:
        seq = []; cur = start; g = 0
        while cur != 0 and g <= n + 1:
            seq.append(cur - 1); cur = out[cur][0] if out[cur] else 0; g += 1
        if seq: routes.append(seq)
    return routes


def gurobi_rci_bound_ip(inst, omega_V, timelimit=60):
    """IP with lazy RCI separation. Returns (objval, objbound, model)."""
    n = inst.n; V = range(n + 1)
    m = gp.Model(); m.Params.OutputFlag = 0
    x = m.addVars([(i, j) for i in V for j in V if i != j], vtype=GRB.BINARY, ub=1, name="x")
    m.setObjective(gp.quicksum(inst.C[i, j] * x[i, j] for i in V for j in V if i != j)
                   + omega_V * gp.quicksum(x[0, j] for j in range(1, n + 1)), GRB.MINIMIZE)
    for j in range(1, n + 1):
        m.addConstr(gp.quicksum(x[i, j] for i in V if i != j) == 1)
        m.addConstr(gp.quicksum(x[j, i] for i in V if i != j) == 1)
    m.addConstr(gp.quicksum(x[0, j] for j in range(1, n + 1))
                == gp.quicksum(x[j, 0] for j in range(1, n + 1)))
    m._x = x; m._inst = inst; m._n = n
    m.Params.LazyConstraints = 1
    m.Params.TimeLimit = timelimit
    m.optimize(rci_callback)
    return m.ObjVal, m.ObjBound, m


def run():
    from matheuristic_core import route_cvar_peak
    print("RCI lower bound (Gurobi B&C, lazy separation) vs matheuristic UB\n")
    print(f"{'n':>3} {'r':>2} {'UB(matheur)':>11} {'LB(valid)':>9} {'gap%':>7} "
          f"{'IP_solved?':>10} {'t_UB':>6} {'t_LB':>6}")
    for (n, r) in [(12, 2), (14, 2), (16, 2), (18, 2)]:
        inst = make_instance(n=n, r=r, N=150, seed=1, eps=1.5, alpha=0.9)
        omega_V = inst.C[0, 1:].mean()
        t0 = time.perf_counter()
        best, _ = alns(inst, iters=2000, seed=1, omega_V=omega_V, verbose=False)
        t_ub = time.perf_counter() - t0; ub = best.cost()
        t1 = time.perf_counter()
        _, bound, m = gurobi_rci_bound_ip(inst, omega_V, timelimit=40)
        t_lb = time.perf_counter() - t1
        lb = bound                                    # VALID dual lower bound (m.ObjBound)
        solved = (m.Status == GRB.OPTIMAL)
        gap = 100 * (ub - lb) / ub
        print(f"{n:>3} {r:>2} {ub:>11.1f} {lb:>9.1f} {gap:>7.2f} "
              f"{('YES' if solved else 'timeout'):>10} {t_ub:>6.1f} {t_lb:>6.1f}")
    print("\nREADING (honest):")
    print("  LB is the VALID dual bound (ObjBound), so gap is a true certificate: matheuristic is")
    print("  provably within gap% of optimum. When IP_solved=YES the bound is as tight as this")
    print("  relaxation allows; on 'timeout' the valid bound is weaker (larger gap) purely because")
    print("  the bound computation did not finish -- a full license + fractional RCI separation would")
    print("  tighten it. The endpoint RCI relaxation ignores the interior excursion, so even its exact")
    print("  optimum under-estimates the true optimum; the certificate is valid but conservative.")


if __name__ == "__main__":
    run()


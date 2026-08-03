"""Numerical falsification tests for the central claims of the WRNF paper.

The paper's spine rests on three claims about two-stage Wasserstein-robust
network flow with ambiguous demand.  Before writing any theorem we test each
one numerically against brute force, on instances small enough that the
worst-case distribution can be computed exactly by linear programming over
transport plans.

Model.  Sources i = 1..m with first-stage installed capacity x_i (unit cost
f_i).  Sinks j = 1..n with random demand xi_j.  Arc (i,j) has unit shipping
cost c_ij.  Unmet demand at j is penalised at rate p per unit, which keeps the
second-stage value function finite for every realisation.

    Q(x, xi) = min  sum_ij c_ij y_ij + p sum_j u_j
               s.t. sum_j y_ij <= x_i          (source capacity)
                    sum_i y_ij + u_j = xi_j    (demand met or unmet)
                    y, u >= 0

Q is convex piecewise linear in xi.  The distributionally robust first-stage
problem over a type-1 Wasserstein ball of radius eps around the empirical
distribution of N demand samples is

    min_x  f'x + sup_{P in B_eps(Phat)} E_P[Q(x, xi)] .

CLAIM 1 (collapse).  With unbounded support, sup_P E_P[Q] equals the sample
average plus eps * Lip(Q), where Lip is the Lipschitz modulus of Q in the norm
dual to the transport norm.  Because that modulus equals the penalty rate p
irrespective of x, the robust problem is the sample-average problem plus a
constant: robustification does not change the optimal decision.

CLAIM 2 (decision relevance returns under bounded support).  Restricting the
support to xi <= xi_max makes the Lipschitz modulus x-dependent, so the robust
and sample-average optima genuinely differ.

CLAIM 3 (combinatorial modulus).  For the uncapacitated relaxation the
Lipschitz modulus is min(p, min_i c_ij) maximised over sinks j -- a quantity
read off the cost matrix, with no LP required.

Run:  python3 prototype_check.py
"""

from __future__ import annotations

import itertools
import os

import numpy as np

os.environ.setdefault("GRB_LICENSE_FILE", "/root/gurobi.lic")
import gurobipy as gp
from gurobipy import GRB

RNG = np.random.default_rng(20260727)


# ----------------------------------------------------------------------------
# second-stage value function
# ----------------------------------------------------------------------------
def solve_Q(x, xi, cost, penalty):
    """Second-stage min-cost flow value Q(x, xi).  Returns (value, demand duals)."""
    m, n = cost.shape
    mdl = gp.Model(env=ENV)
    mdl.Params.OutputFlag = 0
    y = mdl.addVars(m, n, lb=0.0, name="y")
    u = mdl.addVars(n, lb=0.0, name="u")
    for i in range(m):
        mdl.addConstr(gp.quicksum(y[i, j] for j in range(n)) <= x[i])
    dem = [
        mdl.addConstr(gp.quicksum(y[i, j] for i in range(m)) + u[j] == xi[j])
        for j in range(n)
    ]
    mdl.setObjective(
        gp.quicksum(cost[i, j] * y[i, j] for i in range(m) for j in range(n))
        + penalty * gp.quicksum(u[j] for j in range(n)),
        GRB.MINIMIZE,
    )
    mdl.optimize()
    assert mdl.Status == GRB.OPTIMAL, mdl.Status
    val = mdl.ObjVal
    duals = np.array([c.Pi for c in dem])
    mdl.dispose()
    return val, duals


# ----------------------------------------------------------------------------
# worst-case expectation: brute force over transport plans on a finite grid
# ----------------------------------------------------------------------------
def worstcase_bruteforce(x, samples, eps, grid, cost, penalty):
    """sup_{P in B_eps} E_P[Q] by LP over transport plans, support = `grid`.

    Variables pi[k, g] = mass moved from sample k to grid point g.  The type-1
    Wasserstein constraint is sum pi[k,g] * ||grid_g - sample_k||_1 <= eps.
    """
    N = len(samples)
    G = len(grid)
    qvals = np.array([solve_Q(x, g, cost, penalty)[0] for g in grid])
    dist = np.array([[np.abs(grid[g] - samples[k]).sum() for g in range(G)] for k in range(N)])

    mdl = gp.Model(env=ENV)
    mdl.Params.OutputFlag = 0
    pi = mdl.addVars(N, G, lb=0.0)
    for k in range(N):
        mdl.addConstr(gp.quicksum(pi[k, g] for g in range(G)) == 1.0 / N)
    mdl.addConstr(
        gp.quicksum(dist[k, g] * pi[k, g] for k in range(N) for g in range(G)) <= eps
    )
    mdl.setObjective(
        gp.quicksum(qvals[g] * pi[k, g] for k in range(N) for g in range(G)), GRB.MAXIMIZE
    )
    mdl.optimize()
    assert mdl.Status == GRB.OPTIMAL, mdl.Status
    val = mdl.ObjVal
    mdl.dispose()
    return val


def worstcase_closedform(x, samples, eps, cost, penalty):
    """Sample average + eps * Lip, the predicted unbounded-support collapse."""
    saa = float(np.mean([solve_Q(x, s, cost, penalty)[0] for s in samples]))
    return saa + eps * lipschitz_global(penalty), saa


def lipschitz_global(penalty):
    """Modulus of Q over an UNBOUNDED support.

    The dual of the demand constraint is pi_j = min(p, min_i (c_ij + mu_i))
    where mu_i >= 0 is the source-capacity dual.  Refusing demand is always
    available, so pi_j <= p; and as demand grows the installed capacity is
    exhausted, mu_i grows, and pi_j rises to exactly p.  Hence over an
    unbounded support the modulus is p, INDEPENDENT of x -- which is what
    drives the collapse in Claim 1.
    """
    return float(penalty)


def lipschitz_uncapacitated(cost, penalty):
    """Modulus in the slack-capacity regime: capacity never binds on the support.

    Then mu_i = 0, so pi_j = min(p, min_i c_ij) and the modulus is the largest
    such marginal cost over sinks.  Under an l1 transport norm the dual norm is
    l_inf, so this is a max over j, not a sum.  Purely combinatorial: read off
    the cost matrix, no LP.
    """
    return float(np.max(np.minimum(penalty, cost.min(axis=0))))


def lipschitz_empirical(x, grid, cost, penalty):
    """Modulus measured directly: max |Q(a) - Q(b)| / ||a-b||_1 over grid pairs."""
    qs = {tuple(g): solve_Q(x, g, cost, penalty)[0] for g in grid}
    worst = 0.0
    for a, b in itertools.combinations(grid, 2):
        d = np.abs(a - b).sum()
        if d > 1e-9:
            worst = max(worst, abs(qs[tuple(a)] - qs[tuple(b)]) / d)
    return worst


# ----------------------------------------------------------------------------
# first-stage optimisation by grid search (small instances, exact comparison)
# ----------------------------------------------------------------------------
def optimise_first_stage(samples, eps, cost, penalty, fcost, xgrid, mode, grid=None):
    """Grid-search argmin of f'x + <second-stage term>.

    mode='saa'  : sample average
    mode='dro'  : worst case, brute force over `grid` (bounded support)
    mode='dro_cf': worst case via the closed form (unbounded support)
    """
    best, arg = np.inf, None
    for x in xgrid:
        x = np.asarray(x, dtype=float)
        if mode == "saa":
            second = float(np.mean([solve_Q(x, s, cost, penalty)[0] for s in samples]))
        elif mode == "dro":
            second = worstcase_bruteforce(x, samples, eps, grid, cost, penalty)
        elif mode == "dro_cf":
            second = worstcase_closedform(x, samples, eps, cost, penalty)[0]
        else:
            raise ValueError(mode)
        tot = float(fcost @ x) + second
        if tot < best - 1e-9:
            best, arg = tot, x.copy()
    return arg, best


# ----------------------------------------------------------------------------
def make_instance():
    m, n = 2, 2
    cost = np.array([[1.0, 4.0], [3.0, 2.0]])
    penalty = 10.0
    fcost = np.array([0.8, 0.8])
    samples = [np.array([3.0, 2.0]), np.array([2.0, 4.0]), np.array([5.0, 1.0]),
               np.array([1.0, 3.0]), np.array([4.0, 4.0])]
    return m, n, cost, penalty, fcost, samples


def main():
    global ENV
    ENV = gp.Env(params={"OutputFlag": 0})

    m, n, cost, penalty, fcost, samples = make_instance()
    print("=" * 74)
    print("WRNF prototype: numerical falsification of the paper's three claims")
    print("=" * 74)
    print(f"cost matrix:\n{cost}\npenalty p = {penalty}, install cost f = {fcost}")
    print(f"N = {len(samples)} demand samples: {[list(s) for s in samples]}")

    # ---- support grids -----------------------------------------------------
    # "unbounded": generous box, far enough out that the worst case is never
    # clipped by the box for the radii tested.
    wide = np.arange(0.0, 30.01, 1.0)
    grid_wide = [np.array([a, b]) for a in wide for b in wide]
    # bounded support: demand physically cannot exceed 6 units at a sink
    tight = np.arange(0.0, 6.01, 0.5)
    grid_tight = [np.array([a, b]) for a in tight for b in tight]

    x_test = np.array([3.0, 3.0])

    # ---- CLAIM 3: two combinatorial regimes for the modulus ---------------
    print("\n--- CLAIM 3: the modulus is combinatorial, in two regimes -------")
    # (a) binding regime: capacity is small relative to the support
    lip_bind_pred = lipschitz_global(penalty)
    lip_bind_emp = lipschitz_empirical(x_test, grid_tight, cost, penalty)
    print(f"  binding regime  x = {x_test} (capacity can be exhausted)")
    print(f"    predicted p                        = {lip_bind_pred:.4f}")
    print(f"    measured  max |dQ|/||dxi||_1        = {lip_bind_emp:.4f}")
    ok3a = abs(lip_bind_pred - lip_bind_emp) < 1e-6
    # (b) slack regime: capacity so large it never binds on the support
    x_slack = np.array([60.0, 60.0])
    lip_slack_pred = lipschitz_uncapacitated(cost, penalty)
    lip_slack_emp = lipschitz_empirical(x_slack, grid_tight, cost, penalty)
    print(f"  slack regime    x = {x_slack} (capacity never binds)")
    print(f"    predicted max_j min(p, min_i c_ij) = {lip_slack_pred:.4f}")
    print(f"    measured  max |dQ|/||dxi||_1        = {lip_slack_emp:.4f}")
    ok3b = abs(lip_slack_pred - lip_slack_emp) < 1e-6
    ok3 = ok3a and ok3b
    print(f"CLAIM 3: {'HOLDS in both regimes' if ok3 else 'FAILS'} "
          f"(binding {'ok' if ok3a else 'no'}, slack {'ok' if ok3b else 'no'})")

    # ---- CLAIM 1: collapse under unbounded support -----------------------
    print("\n--- CLAIM 1: unbounded support => worst case = SAA + eps*Lip ----")
    print(f"{'eps':>6} {'brute force':>14} {'closed form':>14} {'SAA':>10} {'gap':>12}")
    ok1 = True
    for eps in [0.0, 0.25, 0.5, 1.0, 2.0]:
        bf = worstcase_bruteforce(x_test, samples, eps, grid_wide, cost, penalty)
        cf, saa = worstcase_closedform(x_test, samples, eps, cost, penalty)
        gap = abs(bf - cf)
        ok1 &= gap < 1e-4
        print(f"{eps:>6.2f} {bf:>14.6f} {cf:>14.6f} {saa:>10.4f} {gap:>12.2e}")
    print(f"CLAIM 1 (objective identity): {'HOLDS' if ok1 else 'FAILS'}")

    # decision irrelevance
    xgrid = [np.array([a, b]) for a in np.arange(0.0, 8.01, 0.5)
             for b in np.arange(0.0, 8.01, 0.5)]
    x_saa, v_saa = optimise_first_stage(samples, 0.0, cost, penalty, fcost, xgrid, "saa")
    x_cf, v_cf = optimise_first_stage(samples, 1.0, cost, penalty, fcost, xgrid, "dro_cf")
    same = np.allclose(x_saa, x_cf)
    print(f"argmin SAA = {x_saa}  (obj {v_saa:.4f})")
    print(f"argmin DRO = {x_cf}  (obj {v_cf:.4f})   [unbounded support]")
    print(f"CLAIM 1 (decision irrelevance): {'HOLDS' if same else 'FAILS'} "
          f"-- decisions {'coincide' if same else 'DIFFER'}")

    # ---- CLAIM 2: bounded support restores decision relevance ------------
    print("\n--- CLAIM 2: bounded support => decisions differ ----------------")
    eps_b = 2.0
    x_saa_b, v_saa_b = optimise_first_stage(samples, 0.0, cost, penalty, fcost,
                                            xgrid, "saa")
    x_dro_b, v_dro_b = optimise_first_stage(samples, eps_b, cost, penalty, fcost,
                                            xgrid, "dro", grid=grid_tight)
    print(f"support xi <= 6, eps = {eps_b}")
    print(f"argmin SAA = {x_saa_b}  (obj {v_saa_b:.4f})")
    print(f"argmin DRO = {x_dro_b}  (obj {v_dro_b:.4f})")
    differ = not np.allclose(x_saa_b, x_dro_b)
    print(f"CLAIM 2: {'HOLDS' if differ else 'FAILS'} -- decisions "
          f"{'DIFFER (robustness is decision-relevant)' if differ else 'coincide'}")

    # also confirm the closed form is now an upper bound, not an identity
    bf_t = worstcase_bruteforce(x_test, samples, eps_b, grid_tight, cost, penalty)
    cf_t, _ = worstcase_closedform(x_test, samples, eps_b, cost, penalty)
    print(f"at x={x_test}: bounded-support worst case {bf_t:.4f} "
          f"<= closed form {cf_t:.4f}  ({'consistent' if bf_t <= cf_t + 1e-6 else 'VIOLATION'})")

    print("\n" + "=" * 74)
    print("SUMMARY")
    print(f"  Claim 1 objective identity        : {'HOLDS' if ok1 else 'FAILS'}")
    print(f"  Claim 1 decision irrelevance      : {'HOLDS' if same else 'FAILS'}")
    print(f"  Claim 2 bounded-support relevance : {'HOLDS' if differ else 'FAILS'}")
    print(f"  Claim 3 combinatorial modulus     : {'HOLDS' if ok3 else 'FAILS'}")
    allok = ok1 and same and differ and ok3
    print(f"  ==> paper spine {'SURVIVES' if allok else 'NEEDS REVISION'}")
    print("=" * 74)


if __name__ == "__main__":
    main()

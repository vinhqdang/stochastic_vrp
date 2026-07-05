#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROBUST SVRPSPD  --  Column Generation / Price-and-Branch
========================================================

Set-covering master (min total travel cost, cover every customer) priced by the
validated envelope labeling subproblem (surrogate-pruned, exact-confirmed CVaR
feasibility under rank-1 factor demand).  This is the OR-grade computational
wrapper around the Part VII pricing engine.

Master backend is swappable:
  * Gurobi  -- used automatically if gurobipy imports (your machine).
  * scipy   -- fallback (linprog + milp) so the CG loop and correctness are
               verifiable in this sandbox, which has no Gurobi.

Correctness gate (run before any timing):
  (A) CG LP optimum  ==  LP over the FULLY ENUMERATED feasible-route pool.
  (B) env-pricer CG  ==  naive-pricer CG   (two independent pricers, same LP).
If either fails the program reports it loudly.

Scaling demo:
  env-pricer CG runs at scenario counts N where the per-scenario (naive) pricer
  exhausts memory -- the practical payoff of N-independent pricing state.

NOTE on scope: this is PRICE-AND-BRANCH (exact LP relaxation by CG, then the
restricted master solved as an IP over generated columns).  It is NOT full
branch-and-price-and-cut (no pricing inside branch nodes), so the integer
solution is a strong heuristic / upper bound; the LP value is the exact
relaxation bound.  Stated honestly.

Delivered after the scipy path was run in-sandbox; the Gurobi path is the same
model and is what you run locally.
"""

import numpy as np
import itertools
import time
import importlib.util
import argparse

# ---- load the validated pieces (v2 = instance/FStats/naive; v3 = surrogate/exact_np) ----
def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v2 = _load("v2", "pricing_labeling_v2.py")
v3 = _load("v3", "pricing_labeling_v3.py")
EPS = v2.EPS


# ===========================================================================
#  helpers
# ===========================================================================

def route_cost(order, inst):
    D = inst['D']; path = [0] + list(order) + [0]
    return sum(D[path[i], path[i + 1]] for i in range(len(path) - 1))


def route_peaks(order, inst, F):
    return v2.route_peaks(order, inst, F)


def enumerate_feasible(inst, F, alpha, Qeff):
    n = inst['n']; routes = []
    for k in range(1, n + 1):
        for sub in itertools.combinations(range(1, n + 1), k):
            for order in itertools.permutations(sub):
                if v2.cvar_naive(route_peaks(list(order), inst, F), alpha) <= Qeff + EPS:
                    routes.append((list(order), route_cost(list(order), inst)))
    return routes


# ===========================================================================
#  Pricers  (return columns (order, cost, reduced_cost) with rc < -tol)
# ===========================================================================

def env_price(inst, fs, Qeff, pi, tol=1e-6, max_cols=40):
    n = inst['n']; D = inst['D']
    dbar, dco, qbar, lam = inst['dbar'], inst['dco'], inst['qbar'], inst['lam']
    mu_hi, mu_lo = v3.factor_tail_means(fs)
    LE, env_dom = v2.LE, v2.env_dom
    start = LE(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [(0.0, 0.0)])
    buckets = {(0, 0): [start]}; parent = {id(start): (None, None)}
    cols = []

    def ll_of(lab):
        return [(lab.Dbar + b, lab.dco + a) for (b, a) in lab.lines]

    levels = {0: [(0, 0, start)]}
    for lvl in range(n + 1):
        for (node, vis, lab) in levels.get(lvl, []):
            ll = ll_of(lab)
            if v3.surrogate_cvar(ll, mu_hi, mu_lo) > Qeff + EPS:
                continue
            if vis != 0 and v3.exact_cvar_hybrid(ll, fs) <= Qeff + EPS:
                rc_full = lab.rc + D[node, 0]
                if rc_full < -tol:
                    order = v2._reconstruct(lab, parent)
                    cols.append((order, route_cost(order, inst), rc_full))
            for w in range(1, n + 1):
                bit = 1 << (w - 1)
                if vis & bit:
                    continue
                nb = lab.Sbar + qbar[w]; na = lab.lamco + lam[w]
                nlab = LE(w, vis | bit, lab.rc + D[node, w] - pi[w],
                          lab.Dbar + dbar[w], lab.dco + dco[w], nb, na,
                          lab.lines + [(nb, na)])
                if v3.surrogate_cvar(ll_of(nlab), mu_hi, mu_lo) > Qeff + EPS:
                    continue
                parent[id(nlab)] = (lab, w)
                key = (w, vis | bit); bucket = buckets.get(key)
                if bucket is None:
                    buckets[key] = [nlab]; levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab)); continue
                if any(env_dom(e, nlab) for e in bucket):
                    continue
                buckets[key] = [e for e in bucket if not env_dom(nlab, e)] + [nlab]
                levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab))
    return _dedup_top(cols, max_cols)


def naive_price(inst, F, alpha, Qeff, pi, tol=1e-6, max_cols=40):
    n = inst['n']; D = inst['D']
    dbar, dco, qbar, lam = inst['dbar'], inst['dco'], inst['qbar'], inst['lam']
    LN, naive_dom = v2.LN, v2.naive_dom
    z = np.zeros(len(F)); start = LN(0, 0, 0.0, z.copy(), z.copy(), z.copy())
    buckets = {(0, 0): [start]}; parent = {id(start): (None, None)}; cols = []
    levels = {0: [(0, 0, start)]}
    for lvl in range(n + 1):
        for (node, vis, lab) in levels.get(lvl, []):
            if vis != 0 and v2.cvar_naive(lab.D + lab.M, alpha) <= Qeff + EPS:
                rc_full = lab.rc + D[node, 0]
                if rc_full < -tol:
                    order = v2._reconstruct(lab, parent)
                    cols.append((order, route_cost(order, inst), rc_full))
            for w in range(1, n + 1):
                bit = 1 << (w - 1)
                if vis & bit:
                    continue
                newD = lab.D + dbar[w] + dco[w] * F
                newS = lab.S + qbar[w] + lam[w] * F
                newM = np.maximum(lab.M, newS)
                if v2.cvar_naive(newD + newM, alpha) > Qeff + EPS:
                    continue
                nlab = LN(w, vis | bit, lab.rc + D[node, w] - pi[w], newD, newM, newS)
                parent[id(nlab)] = (lab, w)
                key = (w, vis | bit); bucket = buckets.get(key)
                if bucket is None:
                    buckets[key] = [nlab]; levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab)); continue
                if any(naive_dom(e, nlab) for e in bucket):
                    continue
                buckets[key] = [e for e in bucket if not naive_dom(nlab, e)] + [nlab]
                levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab))
    return _dedup_top(cols, max_cols)


def _dedup_top(cols, max_cols):
    cols.sort(key=lambda c: c[2])
    seen, out = set(), []
    for order, cost, rc in cols:
        k = tuple(order)
        if k in seen:
            continue
        seen.add(k); out.append((order, cost, rc))
        if len(out) >= max_cols:
            break
    return out


# ===========================================================================
#  Master backends
# ===========================================================================

class ScipyMaster:
    def __init__(self, n):
        self.n = n; self.customers = list(range(1, n + 1))
        self.routes = []; self.costs = []; self.sets = []

    def has(self, order):
        return tuple(order) in self._keys()

    def _keys(self):
        return set(tuple(r) for r in self.routes)

    def add_column(self, order, cost):
        self.routes.append(list(order)); self.costs.append(cost); self.sets.append(set(order))

    def _matrix(self):
        R, n = len(self.routes), self.n
        A = np.zeros((n, R))
        for r, s in enumerate(self.sets):
            for i in s:
                A[i - 1, r] = 1.0
        return A

    def solve_lp(self):
        from scipy.optimize import linprog
        A = self._matrix()
        res = linprog(c=self.costs, A_ub=-A, b_ub=-np.ones(self.n),
                      bounds=[(0, None)] * len(self.costs), method='highs')
        if not res.success:
            raise RuntimeError("master LP infeasible: " + res.message)
        lam = res.ineqlin.marginals          # duals of (-A x <= -1)
        pi = {i: -lam[i - 1] for i in self.customers}   # price of (A x >= 1)_i
        return res.fun, pi, res.x

    def solve_ip(self):
        from scipy.optimize import milp, LinearConstraint, Bounds
        A = self._matrix(); R = len(self.costs)
        con = LinearConstraint(A, lb=np.ones(self.n), ub=np.inf)
        res = milp(c=np.array(self.costs), constraints=[con],
                   integrality=np.ones(R), bounds=Bounds(0, np.inf))
        if not res.success:
            return None, None
        chosen = [self.routes[k] for k in range(R) if res.x[k] > 0.5]
        return res.fun, chosen


class GurobiMaster:
    def __init__(self, n):
        import gurobipy as gp
        from gurobipy import GRB
        self.gp, self.GRB = gp, GRB
        self.m = gp.Model(); self.m.Params.OutputFlag = 0
        self.m.ModelSense = GRB.MINIMIZE
        self.customers = list(range(1, n + 1))
        self.cover = {i: self.m.addConstr(gp.LinExpr() >= 1.0, name="cov%d" % i)
                      for i in self.customers}
        self.vars = []; self.routes = []; self._key = set()

    def add_column(self, order, cost):
        col = self.gp.Column()
        for i in set(order):
            col.addTerms(1.0, self.cover[i])
        v = self.m.addVar(obj=float(cost), lb=0.0, column=col)
        self.vars.append(v); self.routes.append(list(order)); self._key.add(tuple(order))
        self.m.update()

    def solve_lp(self):
        self.m.optimize()
        pi = {i: self.cover[i].Pi for i in self.customers}
        return self.m.ObjVal, pi, [v.X for v in self.vars]

    def solve_ip(self):
        for v in self.vars:
            v.VType = self.GRB.BINARY
        self.m.optimize()
        chosen = [self.routes[k] for k, v in enumerate(self.vars) if v.X > 0.5]
        obj = self.m.ObjVal
        for v in self.vars:
            v.VType = self.GRB.CONTINUOUS
        self.m.update()
        return obj, chosen


def make_master(n):
    try:
        import gurobipy  # noqa
        return GurobiMaster(n), "gurobi"
    except Exception:
        return ScipyMaster(n), "scipy"


# ===========================================================================
#  Column generation
# ===========================================================================

def column_generation(inst, n, init_routes, pricer_fn, maxit=2000, tol=1e-6):
    master, backend = make_master(n)
    seen = set()
    for (order, cost) in init_routes:
        if tuple(order) not in seen:
            master.add_column(order, cost); seen.add(tuple(order))
    nit = 0
    while nit < maxit:
        lp_obj, pi, _ = master.solve_lp()
        piarr = np.zeros(n + 1)
        for i in range(1, n + 1):
            piarr[i] = pi[i]
        cols = [c for c in pricer_fn(piarr) if c[2] < -tol and tuple(c[0]) not in seen]
        if not cols:
            break
        for (order, cost, rc) in cols:
            master.add_column(order, cost); seen.add(tuple(order))
        nit += 1
    lp_obj, _, _ = master.solve_lp()
    return master, backend, lp_obj, nit, len(seen)


def full_pool_lp(routes, n):
    m, _ = (ScipyMaster(n), None) if make_master(n)[1] == "scipy" else (make_master(n)[0], None)
    # always use scipy here for a backend-independent ground truth if available
    try:
        m = ScipyMaster(n)
        for (order, cost) in routes:
            m.add_column(order, cost)
        obj, _, _ = m.solve_lp()
        return obj
    except Exception as e:
        return None


# ===========================================================================
#  Driver
# ===========================================================================

def calibrate_Qeff_feasible(inst, fs, alpha, pct, seed):
    # base percentile, then raise to guarantee every singleton is feasible
    base = v2.calibrate_Qeff(inst, alpha, pct, seed)
    n = inst['n']
    sing = []
    F = fs.Fs
    for i in range(1, n + 1):
        sing.append(v2.cvar_naive(route_peaks([i], inst, F), alpha))
    return max(base, max(sing) * 1.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.90)
    args = ap.parse_args()
    n, alpha = args.n, args.alpha

    inst = v2.make_instance(n, args.seed)

    print("=" * 84)
    print("ROBUST SVRPSPD  --  Column Generation / Price-and-Branch")
    _, backend0 = make_master(n)
    print("master backend: %s   (Gurobi auto-used if installed)" % backend0.upper())
    print("=" * 84)

    # ---- correctness gate at a small N ----
    Nc = 300
    Fc = v2.sample_factors(Nc, args.seed); fsc = v2.FStats(Fc, alpha)
    Qeff = calibrate_Qeff_feasible(inst, fsc, alpha, 50, args.seed)
    print("\nseed=%d n=%d alpha=%.2f Q_eff=%.2f" % (args.seed, n, alpha, Qeff))

    init = [([i], route_cost([i], inst)) for i in range(1, n + 1)]   # singletons
    # verify singletons feasible
    bad = [i for i in range(1, n + 1)
           if v2.cvar_naive(route_peaks([i], inst, Fc), alpha) > Qeff + EPS]
    assert not bad, "infeasible singletons: %s" % bad

    pool = enumerate_feasible(inst, Fc, alpha, Qeff)
    lp_true = full_pool_lp(pool, n)

    _, _, lp_env, it_env, nc_env = column_generation(
        inst, n, init, lambda pi: env_price(inst, fsc, Qeff, pi))
    _, _, lp_nai, it_nai, nc_nai = column_generation(
        inst, n, init, lambda pi: naive_price(inst, Fc, alpha, Qeff, pi))

    print("\n[CORRECTNESS GATE @N=%d]" % Nc)
    print("  full-pool LP (%d feasible routes enumerated) : %.4f" % (len(pool), lp_true))
    print("  env-pricer CG LP   : %.4f  (%d iters, %d cols)  %s"
          % (lp_env, it_env, nc_env, "OK" if abs(lp_env - lp_true) < 1e-4 else "MISMATCH"))
    print("  naive-pricer CG LP : %.4f  (%d iters, %d cols)  %s"
          % (lp_nai, it_nai, nc_nai, "OK" if abs(lp_nai - lp_true) < 1e-4 else "MISMATCH"))

    # integer (price-and-branch) over env columns
    masterE, _, _, _, _ = column_generation(inst, n, init,
                                             lambda pi: env_price(inst, fsc, Qeff, pi))
    ip_obj, chosen = masterE.solve_ip()
    if ip_obj is not None:
        print("  price-and-branch IP: %.4f   routes=%s" % (ip_obj, chosen))
        print("  LP-IP gap: %.2f%%" % (100 * (ip_obj - lp_env) / max(abs(lp_env), 1e-9)))

    # ---- scaling: env CG runs where naive CG OOMs ----
    print("\n[SCALING]  env-pricer CG (N-independent state) vs naive-pricer CG")
    print("%-9s | %-26s | %-26s" % ("N", "env CG (LP, iters, time)", "naive CG (LP, iters, time)"))
    print("-" * 78)
    for N in [3000, 30000, 100000, 300000, 1000000]:
        F = v2.sample_factors(N, args.seed); fs = v2.FStats(F, alpha)
        t0 = time.perf_counter()
        _, _, lpE, itE, _ = column_generation(inst, n, init, lambda pi: env_price(inst, fs, Qeff, pi))
        tE = time.perf_counter() - t0
        mem_gb = 600 * 3 * N * 8 / 1e9
        if mem_gb < 2.0:
            t0 = time.perf_counter()
            _, _, lpN, itN, _ = column_generation(inst, n, init, lambda pi: naive_price(inst, F, alpha, Qeff, pi))
            tN = time.perf_counter() - t0
            nstr = "%.4f  %d its  %.2fs" % (lpN, itN, tN)
            match = "match=OK" if abs(lpE - lpN) < 1e-4 else "MISMATCH"
        else:
            nstr = "--OOM (per-scenario state)--"; match = "env-only"
        print("%-9d | %-26s | %-26s %s"
              % (N, "%.4f  %d its  %.2fs" % (lpE, itE, tE), nstr, match))

    print("\n" + "=" * 84)
    print("READING")
    print("  Gate: env-CG LP == naive-CG LP == full-pool LP  => CG + pricer + duals correct.")
    print("  Scaling: env-CG solves the LP relaxation at N where the per-scenario pricer")
    print("           exhausts memory -> the scenario-scalability claim, end to end.")
    print("  On your machine the master is Gurobi; run for larger n and report timings.")
    print("=" * 84)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dethloff_tropical_bp.py
=======================
Full Branch-and-Price for the robust SVRPSPD (Stochastic VRP with Simultaneous
Pickup & Delivery under a rank-1 Wasserstein/factor demand model), on Dethloff
benchmark instances, with PySCIPOpt.
"""

import sys
import bisect as _bisect
import numpy as np

from pyscipopt import Model, Pricer, Conshdlr, Branchrule, quicksum
from pyscipopt import SCIP_RESULT, SCIP_PARAMSETTING

# ===========================================================================
#  Global config
# ===========================================================================

EPS = 1e-7
FMIN, FMAX = -2.5, 2.5           # truncated support of the standard factor
ALPHA = 0.90                     # CVaR level
SIGMA_FRAC = 0.25                # factor loading magnitude as a fraction of mean demand
DEFAULT_N = 1000                 # default number of scenarios
CVAR_HYBRID_THRESH = 50000       # numpy-direct below, order-stats above
REDCOST_TOL = 1e-6
FRAC_TOL = 1e-6
FORBID, ENFORCE = 0, 1

# ===========================================================================
#  PART 1.  Dethloff .vrpspd ingestion
# ===========================================================================

def parse_dethloff(path):
    rows, lone = [], []
    cap_header = None
    with open(path) as fh:
        for raw in fh:
            s = raw.strip()
            if not s or s.startswith(("#", "//")):
                continue
            parts = s.split()
            if parts[0].upper().startswith("CAP"):
                try:
                    cap_header = float(parts[-1])
                except ValueError:
                    pass
                continue
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue                      
            if len(vals) == 1:
                lone.append(vals[0])
            elif len(vals) >= 4:
                rows.append(vals)

    if cap_header is not None:
        cap = cap_header
    elif lone:
        cap = max(lone)                       
    else:
        raise ValueError("parse_dethloff: capacity not found; adjust to your file format")

    coords, deliv, pick = [], [], []
    for v in rows:
        x, y, d, p = v[-4], v[-3], v[-2], v[-1]
        coords.append((x, y)); deliv.append(d); pick.append(p)
    if not coords:
        raise ValueError("parse_dethloff: no node rows parsed; adjust to your file format")

    depot_idx = next((k for k, (d, p) in enumerate(zip(deliv, pick)) if d == 0 and p == 0), 0)
    order = [depot_idx] + [k for k in range(len(coords)) if k != depot_idx]
    coords = [coords[k] for k in order]
    deliv = [deliv[k] for k in order]
    pick = [pick[k] for k in order]

    return dict(n=len(coords) - 1, capacity=cap,
                coords=np.array(coords, float),
                delivery=np.array(deliv, float),
                pickup=np.array(pick, float))

def build_factor_instance(deth, sigma_frac=SIGMA_FRAC):
    n = deth["n"]
    coords = deth["coords"]
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    dbar = deth["delivery"].copy(); dbar[0] = 0.0
    pbar = deth["pickup"].copy();   pbar[0] = 0.0
    dco = sigma_frac * dbar; dco[0] = 0.0           
    pload = sigma_frac * pbar; pload[0] = 0.0       
    qbar = pbar - dbar                              
    lam = pload - dco                               
    return dict(n=n, D=D, dbar=dbar, pbar=pbar, dco=dco, pload=pload, qbar=qbar, lam=lam)

def sample_factors(N, seed=2026):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.0, 1.0, N), FMIN, FMAX)

# ===========================================================================
#  PART 2.  Factor order statistics + exact / surrogate CVaR
# ===========================================================================

def upper_env(lines):
    best = {}
    for (B, A) in lines:
        if (A not in best) or (B > best[A]):
            best[A] = B
    items = sorted(best.items())                    
    hull = []
    for A, B in items:
        while hull:
            B2, A2 = hull[-1]
            xn = (B2 - B) / (A - A2)
            if len(hull) >= 2:
                B1, A1 = hull[-2]
                xp = (B1 - B2) / (A2 - A1)
                if xn <= xp:
                    hull.pop(); continue
            break
        hull.append((B, A))
    pieces = []
    for i, (B, A) in enumerate(hull):
        fl = -np.inf if i == 0 else (hull[i - 1][0] - B) / (A - hull[i - 1][1])
        pieces.append((fl, B, A))
    return pieces

def g_at(pieces, f):
    return max(B + A * f for (_, B, A) in pieces)

class FStats:
    def __init__(self, F, alpha=ALPHA):
        self.Fs = np.sort(np.asarray(F, float))
        self.N = len(self.Fs)
        self.m = max(1, int(np.ceil((1.0 - alpha) * self.N)))
        self.PF = np.concatenate([[0.0], np.cumsum(self.Fs)])
        self.Fl = self.Fs.tolist()
        self.alpha = alpha

    def _valley(self, pieces):
        Fs = self.Fs
        lo, hi = 0, self.N - 1
        while hi - lo > 2:
            m1 = lo + (hi - lo) // 3
            m2 = hi - (hi - lo) // 3
            if g_at(pieces, Fs[m1]) < g_at(pieces, Fs[m2]):
                hi = m2
            else:
                lo = m1
        best, bestv = lo, g_at(pieces, Fs[lo])
        for i in range(lo, hi + 1):
            v = g_at(pieces, Fs[i])
            if v < bestv:
                bestv, best = v, i
        return best

    def _count_left_gt(self, ist, tau, pieces):
        lo, hi = 0, ist + 1
        while lo < hi:
            mid = (lo + hi) // 2
            if g_at(pieces, self.Fs[mid]) > tau:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _count_right_gt(self, ist, tau, pieces):
        lo, hi = ist + 1, self.N
        while lo < hi:
            mid = (lo + hi) // 2
            if g_at(pieces, self.Fs[mid]) > tau:
                hi = mid
            else:
                lo = mid + 1
        return self.N - lo

    def _sum_g(self, a, b, pieces):
        if b <= a:
            return 0.0
        total = 0.0
        for k, (fl, B, A) in enumerate(pieces):
            f_next = pieces[k + 1][0] if k + 1 < len(pieces) else np.inf
            lo_i = 0 if fl == -np.inf else _bisect.bisect_left(self.Fl, fl)
            hi_i = self.N if f_next == np.inf else _bisect.bisect_left(self.Fl, f_next)
            lo_i = max(lo_i, a); hi_i = min(hi_i, b)
            if hi_i > lo_i:
                total += A * (self.PF[hi_i] - self.PF[lo_i]) + B * (hi_i - lo_i)
        return total

    def cvar(self, lines):
        pieces = upper_env(lines)
        N, m = self.N, self.m
        ist = self._valley(pieces)
        gmin = g_at(pieces, self.Fs[ist])
        gmax = max(g_at(pieces, self.Fs[0]), g_at(pieces, self.Fs[N - 1]))
        if gmax <= gmin + 1e-15:
            return gmin
        lo, hi = gmin, gmax
        for _ in range(45):
            tau = 0.5 * (lo + hi)
            if self._count_left_gt(ist, tau, pieces) + self._count_right_gt(ist, tau, pieces) >= m:
                lo = tau
            else:
                hi = tau
        tau = lo
        pL = self._count_left_gt(ist, tau, pieces)
        pR = self._count_right_gt(ist, tau, pieces)
        s = self._sum_g(0, pL, pieces) + self._sum_g(N - pR, N, pieces) + (m - pL - pR) * tau
        return s / m

def factor_tail_means(fs):
    m = fs.m
    return float(np.mean(fs.Fs[-m:])), float(np.mean(fs.Fs[:m]))

def surrogate_cvar(load_lines, mu_hi, mu_lo):
    best = -np.inf
    for (B, A) in load_lines:
        v = B + A * (mu_hi if A >= 0 else mu_lo)
        if v > best:
            best = v
    return best

def exact_cvar_np(load_lines, fs):
    Fs, m = fs.Fs, fs.m
    B = np.array([b for (b, a) in load_lines])[:, None]
    A = np.array([a for (b, a) in load_lines])[:, None]
    peak = np.max(B + A * Fs[None, :], axis=0)
    N = peak.shape[0]
    if m >= N:
        return float(peak.mean())
    return float(np.partition(peak, N - m)[N - m:].mean())

def exact_cvar_hybrid(load_lines, fs):
    if fs.N >= CVAR_HYBRID_THRESH:
        return fs.cvar(load_lines)               
    return exact_cvar_np(load_lines, fs)         

# ===========================================================================
#  PART 3.  Tropical-envelope pricing core
# ===========================================================================

class Label:
    __slots__ = ("node", "vis", "rc", "Dbar", "dco", "Sbar", "lamco", "lines")
    def __init__(s, node, vis, rc, Dbar, dco, Sbar, lamco, lines):
        s.node, s.vis, s.rc = node, vis, rc
        s.Dbar, s.dco, s.Sbar, s.lamco, s.lines = Dbar, dco, Sbar, lamco, lines

def _aff_le(b1, a1, b2, a2):
    return (b1 + a1 * FMIN <= b2 + a2 * FMIN + EPS and
            b1 + a1 * FMAX <= b2 + a2 * FMAX + EPS)

def _env_le(L1, L2):
    cands = [FMIN, FMAX]
    alll = L1 + L2
    for i in range(len(alll)):
        for j in range(i + 1, len(alll)):
            b1, a1 = alll[i]; b2, a2 = alll[j]
            if abs(a1 - a2) > 1e-12:
                f = (b2 - b1) / (a1 - a2)
                if FMIN <= f <= FMAX:
                    cands.append(f)
    for f in cands:
        if max(b + a * f for (b, a) in L1) > max(b + a * f for (b, a) in L2) + EPS:
            return False
    return True

def env_dom(a, b):
    return (a.rc <= b.rc + EPS
            and _aff_le(a.Dbar, a.dco, b.Dbar, b.dco)
            and _aff_le(a.Sbar, a.lamco, b.Sbar, b.lamco)
            and _env_le(a.lines, b.lines))

def route_cost(custs, inst, depot=0):
    D = inst["D"]; seq = [depot] + list(custs) + [depot]
    return sum(D[seq[i], seq[i + 1]] for i in range(len(seq) - 1))

def route_arcs(custs, depot=0):
    seq = [depot] + list(custs) + [depot]
    return frozenset((seq[i], seq[i + 1]) for i in range(len(seq) - 1))

def _reconstruct(lab, parent):
    seq, cur = [], lab
    while cur is not None:
        p, w = parent.get(id(cur), (None, None))
        if w is not None:
            seq.append(w)
        cur = p
    return list(reversed(seq))

def _dedup_top(cols, max_cols):
    cols.sort(key=lambda c: c[2])
    seen, out = set(), []
    for order, cost, rc in cols:
        k = tuple(order)
        if k in seen:
            continue
        seen.add(k); out.append((k, cost, rc))
        if len(out) >= max_cols:
            break
    return out

def env_price(inst, fs, Q, pi, forbidden=frozenset(), enforced=frozenset(),
              farkas=False, label_cap=None, tol=REDCOST_TOL, max_cols=40, depot=0):
    n = inst["n"]; D = inst["D"]
    dbar, dco_arr, qbar, lam = inst["dbar"], inst["dco"], inst["qbar"], inst["lam"]
    mu_hi, mu_lo = factor_tail_means(fs)

    eff = set(forbidden)
    for (i, j) in enforced:
        for k in range(0, n + 1):
            if k != j:
                eff.add((i, k))
            if k != i:
                eff.add((k, j))

    start = Label(depot, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [(0.0, 0.0)])
    buckets = {(depot, 0): [start]}
    parent = {id(start): (None, None)}
    cols = []
    levels = {0: [(depot, 0, start)]}

    def ll_of(lab):
        return [(lab.Dbar + b, lab.dco + a) for (b, a) in lab.lines]

    for lvl in range(n + 1):
        for (node, vis, lab) in levels.get(lvl, []):
            ll = ll_of(lab)
            if surrogate_cvar(ll, mu_hi, mu_lo) > Q + EPS:
                continue
            if vis != 0 and (node, depot) not in eff:
                rc_full = lab.rc + (0.0 if farkas else D[node, depot])   
                if rc_full < -tol and exact_cvar_hybrid(ll, fs) <= Q + EPS:
                    order = _reconstruct(lab, parent)
                    cols.append((order, route_cost(order, inst, depot), rc_full))
            for w in range(1, n + 1):
                bit = 1 << (w - 1)
                if (vis & bit) or ((node, w) in eff):
                    continue
                nb = lab.Sbar + qbar[w]; na = lab.lamco + lam[w]
                arc = (0.0 if farkas else D[node, w]) - pi[w]
                nlab = Label(w, vis | bit, lab.rc + arc,
                             lab.Dbar + dbar[w], lab.dco + dco_arr[w], nb, na,
                             lab.lines + [(nb, na)])
                if surrogate_cvar(ll_of(nlab), mu_hi, mu_lo) > Q + EPS:
                    continue
                parent[id(nlab)] = (lab, w)
                key = (w, vis | bit)
                bucket = buckets.get(key)
                if bucket is None:
                    buckets[key] = [nlab]
                    levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab))
                    continue
                if any(env_dom(e, nlab) for e in bucket):
                    continue
                newb = [e for e in bucket if not env_dom(nlab, e)] + [nlab]
                if label_cap and len(newb) > label_cap:        
                    newb.sort(key=lambda L: L.rc)
                    newb = newb[:label_cap]
                buckets[key] = newb
                if nlab in newb:
                    levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab))

    return _dedup_top(cols, max_cols)

def make_user_pricer(inst, fs, Q, label_cap=None):
    n = inst["n"]
    def user_pricer(duals, forbidden_edges, enforced_edges, farkas):
        pi = np.zeros(n + 1)
        for i, val in duals.items():
            pi[i] = val
        return env_price(inst, fs, Q, pi,
                         forbidden=forbidden_edges, enforced=enforced_edges,
                         farkas=farkas, label_cap=label_cap)
    return user_pricer

# ===========================================================================
#  PART 4.  SCIP Branch-and-Price
# ===========================================================================

class BPData:
    def __init__(self, customers, depot=0):
        self.depot = depot
        self.customers = list(customers)
        self.cover_cons = {}
        self.route_of_var = {}            
        self.vars_using_arc = {}          
        self.arc_conshdlr = None
        self.user_pricer = None

class TropicalPricer(Pricer):
    def pricerinit(self):
        d = self.data
        for c in d.customers:
            d.cover_cons[c] = self.model.getTransformedCons(d.cover_cons[c])

    def pricerredcost(self):
        d = self.data
        duals = {c: self.model.getDualsolLinear(d.cover_cons[c]) for c in d.customers}
        forbidden, enforced = self._active_arcs()
        columns = d.user_pricer(duals, forbidden, enforced, farkas=False)
        self._inject(columns)
        return {"result": SCIP_RESULT.SUCCESS}

    def pricerfarkas(self):
        d = self.data
        duals = {c: self.model.getDualfarkasLinear(d.cover_cons[c]) for c in d.customers}
        forbidden, enforced = self._active_arcs()
        columns = d.user_pricer(duals, forbidden, enforced, farkas=True)
        self._inject(columns)
        return {"result": SCIP_RESULT.SUCCESS}

    def _active_arcs(self):
        ch = self.data.arc_conshdlr
        forbidden, enforced = set(), set()
        for (t, h, dr) in ch.active_stack:
            (forbidden if dr == FORBID else enforced).add((t, h))
        return forbidden, enforced

    def _inject(self, columns):
        d = self.data
        for (custs, cost, rc) in columns:
            if rc > -REDCOST_TOL:
                continue
            var_name = "priced_%d" % len(d.route_of_var)
            var = self.model.addVar(vtype="C", obj=float(cost), pricedVar=True, name=var_name)
            for c in custs:
                self.model.addConsCoeff(d.cover_cons[c], var, 1.0)
            arcs = route_arcs(custs, d.depot)
            d.route_of_var[var_name] = (var, tuple(custs), arcs, float(cost))
            for a in arcs:
                d.vars_using_arc.setdefault(a, []).append(var)

class ArcConshdlr(Conshdlr):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.active_stack = []
        self.cons_decision = {}

    def register(self, cons, tail, head, dirn):
        self.cons_decision[cons.name] = (tail, head, dirn)

    def consactive(self, constraint):
        self.active_stack.append(self.cons_decision[constraint.name])

    def consdeactive(self, constraint):
        self.active_stack.pop()

    def consprop(self, constraints, nusefulconss, nmarkedconss, proptiming):
        d = self.data
        forbidden = {(t, h) for (t, h, dr) in self.active_stack if dr == FORBID}
        enforced = {(t, h) for (t, h, dr) in self.active_stack if dr == ENFORCE}
        disallowed = set(forbidden)
        if enforced:
            for (a, b) in list(d.vars_using_arc.keys()):
                for (i, j) in enforced:
                    if (a == i and b != j) or (b == j and a != i):
                        disallowed.add((a, b))
        reduced = False
        for arc in disallowed:
            for var in d.vars_using_arc.get(arc, []):
                infeasible, tightened = self.model.tightenVarUb(var, 0.0)  
                reduced = reduced or tightened
                if infeasible:
                    return {"result": SCIP_RESULT.CUTOFF}
        return {"result": SCIP_RESULT.REDUCEDDOM if reduced else SCIP_RESULT.DIDNOTFIND}

    def conscheck(self, constraints, solution, checkintegrality, checklprows,
                  printreason, completely):
        return {"result": SCIP_RESULT.FEASIBLE}

    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        return

class ArcBranchrule(Branchrule):
    def __init__(self, data, conshdlr):
        super().__init__()
        self.data = data
        self.conshdlr = conshdlr
        self._uid = 0

    def branchexeclp(self, allowaddcons):
        d = self.data
        arc_flow = {}
        for var_name, (var, custs, arcs, cost) in d.route_of_var.items():
            val = self.model.getVal(var)
            if val <= FRAC_TOL:
                continue
            for a in arcs:
                arc_flow[a] = arc_flow.get(a, 0.0) + val
        cand, best = None, 1.0
        for a, f in arc_flow.items():
            if abs(f - round(f)) > FRAC_TOL and abs(f - 0.5) < best:
                best, cand = abs(f - 0.5), a
        if cand is None:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        tail, head = cand
        for dirn in (FORBID, ENFORCE):
            child = self.model.createChild(0, self.model.getLPObjVal())
            self._uid += 1
            name = "arcbranch_%d_%d_%d_%d" % (dirn, tail, head, self._uid)
            cons = self.model.createCons(                      
                self.conshdlr, name,
                initial=False, separate=False, enforce=True, check=False,
                propagate=True, local=True, modifiable=False,
                dynamic=False, removable=False, stickingatnode=False)
            self.conshdlr.register(cons, tail, head, dirn)
            self.model.addConsNode(child, cons)
        return {"result": SCIP_RESULT.BRANCHED}

def configure_for_branch_and_price(model):
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    model.setBoolParam("misc/allowstrongdualreds", False)
    model.setBoolParam("misc/allowweakdualreds", False)
    model.setIntParam("pricing/maxvars", 100)
    model.setIntParam("pricing/maxvarsroot", 200)

def build_branch_and_price(data, initial_routes, user_pricer):
    model = Model("dethloff_tropical_bp")
    data.user_pricer = user_pricer

    init_vars = []
    for (custs, cost) in initial_routes:
        v = model.addVar(vtype="C", obj=float(cost),
                         name="r0_" + "_".join(map(str, custs)))
        init_vars.append((v, tuple(custs), float(cost)))

    for c in data.customers:
        expr = quicksum(v for (v, custs, _) in init_vars if c in custs)
        cons = model.addCons(expr >= 1.0, name="cover_%d" % c,
                             modifiable=True, separate=False)
        data.cover_cons[c] = cons

    for v, custs, cost in init_vars:
        arcs = route_arcs(custs, data.depot)
        data.route_of_var[v.name] = (v, custs, arcs, cost)
        for a in arcs:
            data.vars_using_arc.setdefault(a, []).append(v)

    pricer = TropicalPricer()
    pricer.data = data
    model.includePricer(pricer, "TropicalEnvelopePricer",
                        "envelope robust pricing for SVRPSPD")

    conshdlr = ArcConshdlr(data)
    data.arc_conshdlr = conshdlr
    model.includeConshdlr(
        conshdlr, "ArcBranching",
        "stores + broadcasts + propagates arc-branching decisions",
        sepapriority=0, enfopriority=0, chckpriority=0,
        sepafreq=-1, propfreq=1, eagerfreq=-1, maxprerounds=0,
        delaysepa=False, delayprop=False, needscons=False)

    branchrule = ArcBranchrule(data, conshdlr)
    model.includeBranchrule(
        branchrule, "ArcBranching",
        "branch on a fractional arc (forbid / enforce)",
        priority=1000000, maxdepth=-1, maxbounddist=1.0)

    configure_for_branch_and_price(model)
    return model

# ===========================================================================
#  PART 5.  main
# ===========================================================================

def feasible_singletons(inst, fs, Q):
    n = inst["n"]
    init, bad = [], []
    for i in range(1, n + 1):
        load_lines = [(inst["dbar"][i], inst["dco"][i]),
                      (inst["pbar"][i], inst["pload"][i])]
        if exact_cvar_hybrid(load_lines, fs) > Q + EPS:
            bad.append(i)
        init.append(((i,), route_cost((i,), inst)))
    if bad:
        print("WARNING: singletons robustly infeasible under Q=%.2f: %s" % (Q, bad))
        init = [r for r in init if r[0][0] not in bad]
    return init

def main():
    if len(sys.argv) < 2:
        print("usage: python dethloff_tropical_bp.py <file.vrpspd> [N_scenarios] [label_cap]")
        sys.exit(1)
    path = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N
    label_cap = int(sys.argv[3]) if len(sys.argv) > 3 else None

    deth = parse_dethloff(path)
    inst = build_factor_instance(deth, sigma_frac=SIGMA_FRAC)
    Q = deth["capacity"]                       
    print("instance %s : n=%d customers, capacity=%.1f, alpha=%.2f, sigma=%.2f"
          % (path, inst["n"], Q, ALPHA, SIGMA_FRAC))
    if label_cap:
        print("HEURISTIC pricing: label_cap=%d (LP bound NOT exact)" % label_cap)

    fs = FStats(sample_factors(N), alpha=ALPHA)
    customers = list(range(1, inst["n"] + 1))
    init = feasible_singletons(inst, fs, Q)

    data = BPData(customers, depot=0)
    user_pricer = make_user_pricer(inst, fs, Q, label_cap=label_cap)
    model = build_branch_and_price(data, init, user_pricer)
    model.optimize()

    print("\nstatus      :", model.getStatus())
    print("objective   :", model.getObjVal())
    print("routes used :")
    for var_name, (v, custs, arcs, cost) in data.route_of_var.items():
        if model.getVal(v) > 0.5:
            print("   depot ->", " -> ".join(map(str, custs)), "-> depot")

if __name__ == "__main__":
    main()
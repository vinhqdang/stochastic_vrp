#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dethloff_tropical_bp.py
=======================
Full Branch-and-Price for the robust SVRPSPD (Stochastic VRP with Simultaneous
Pickup & Delivery under a rank-1 Wasserstein/factor demand model), on Dethloff
benchmark instances, with PySCIPOpt.

    python dethloff_tropical_bp.py  dethloff_data/SCA8-8.vrpspd  [N_scenarios] [label_cap]

Pipeline
--------
    parse_dethloff(path)            -> deterministic coords + delivery/pickup + capacity
    build_factor_instance(...)      -> rank-1 factor-demand overlay (loadings)
    FStats(sample_factors(N))       -> sorted factors + prefix sums, BUILT ONCE in main
    env_price(...)                  -> tropical-envelope labeling (exact CVaR feasibility,
                                       surrogate pruning, hybrid completion), now honouring
                                       forbidden / enforced edges + a Farkas flag
    SCIP Branch-and-Price           -> Pricer (redcost + Farkas), arc-branching constraint
                                       handler (broadcast + propagate decisions), branch rule

Infinite-loop prevention (two halves):
    (1) the pricer skips forbidden edges -> never regenerates eliminated columns;
    (2) the constraint handler fixes existing columns that violate active decisions.

!!! SCALING REALITY CHECK !!!
The pricing core is an EXACT ELEMENTARY pure-Python labeling.  Dethloff instances
have ~50 customers; exact elementary pricing on n=50 will NOT terminate in pure
Python (the visited-set space explodes — this is the customer-count wall, which
ng-route does not cure in this tight-capacity regime).  Two heuristic knobs make
the pure-Python pricer terminate on big instances:
    neighbor_k    -- only extend to the k nearest customers (caps the branching
                     factor; this is the real lever -- a per-state label cap does
                     NOT help, since every visited-set is its own state).
    label_budget  -- global cap on labels expanded per pricing call (guarantees
                     termination; the LP bound is then heuristic, not exact).
Use this file to validate the full SCIP wiring / branching / Farkas on a SMALL
instance (exact), then run big instances heuristically.  A production-grade exact
run on n=50 needs a compiled / bidirectional pricer (standard VRP engineering,
orthogonal to the scenario-scalability contribution).

The factor-demand overlay (loadings) is synthetic and configurable (SIGMA_FRAC);
swap in your real factor model where marked.
"""

import sys
import bisect as _bisect
import numpy as np

from pyscipopt import Model, Pricer, Conshdlr, quicksum
from pyscipopt import SCIP_RESULT, SCIP_PARAMSETTING


# ===========================================================================
#  Global config
# ===========================================================================

EPS = 1e-7
FMIN, FMAX = -2.5, 2.5           # truncated support of the standard factor
ALPHA = 0.90                     # CVaR level
SIGMA_FRAC = 0.25                # factor loading magnitude as a fraction of mean demand
DEFAULT_N = 1000                 # default number of scenarios
LOAD_SCALE = 1e-4                # demands & capacity are stored x10000 in the file
DIST_SCALE = 1.0                 # distances ARE the large numbers inside the SCIP LP;
                                 # set to 1e-4 if SCIP flags LP tolerances (routes unchanged,
                                 # reported objective then = raw distance total x 1e-4)
CVAR_HYBRID_THRESH = 50000       # numpy-direct below, order-stats above
REDCOST_TOL = 1e-6
FRAC_TOL = 1e-6
FORBID, ENFORCE = 0, 1


# ===========================================================================
#  PART 1.  Dethloff .vrpspd ingestion
# ===========================================================================

def parse_dethloff(path):
    """
    Parse a TSPLIB-style VRPSPD .vrpspd instance (Dethloff SCA/CON), e.g.:

        NAME : SCA8-8
        TYPE : VRPSPD
        DIMENSION : 51
        VEHICLES : 9
        CAPACITY : 3101288
        EDGE_WEIGHT_TYPE : EXPLICIT
        EDGE_WEIGHT_FORMAT : FULL_MATRIX
        EDGE_WEIGHT_SECTION
        <DIMENSION*DIMENSION numbers : full distance matrix, depot = node 1>
        PICKUP_AND_DELIVERY_SECTION
        <id  demand  tw_start  tw_end  service  pickup  delivery>   x DIMENSION
        DEPOT_SECTION
        <depot_id>
        -1
        EOF

    Distances are EXPLICIT (no coordinates).  Each PD row is
        [id, demand, tw_start, tw_end, service, pickup, delivery]
    so columns 5,6 (0-indexed) are taken as (pickup, delivery).  If your model
    convention is the opposite, swap PICKUP_COL / DELIVERY_COL below and re-run.

    Returns dict: n (customers), capacity, D ((n+1)x(n+1) explicit cost matrix,
                  depot at index 0), delivery[(n+1)], pickup[(n+1)].
    """
    PICKUP_COL, DELIVERY_COL = 5, 6          # <-- swap these two if results look wrong

    header = {}
    section = None
    ew_nums, pd_rows = [], []
    depot_id = 1
    with open(path) as fh:
        for raw in fh:
            s = raw.strip()
            if not s:
                continue
            up = s.upper()
            if up.startswith("EDGE_WEIGHT_SECTION"):
                section = "EW"; continue
            if up.startswith("PICKUP_AND_DELIVERY_SECTION") or up.startswith("DEMAND_SECTION"):
                section = "PD"; continue
            if up.startswith("DEPOT_SECTION"):
                section = "DEPOT"; continue
            if up.startswith("EOF"):
                section = None; continue
            if section is None and ":" in s:                 # header  KEY : VALUE
                k, v = s.split(":", 1)
                header[k.strip().upper()] = v.strip()
                continue
            if section == "EW":
                ew_nums.extend(float(x) for x in s.split())
            elif section == "PD":
                pd_rows.append([float(x) for x in s.split()])
            elif section == "DEPOT":
                tok = s.split()[0]
                if tok != "-1":
                    depot_id = int(float(tok))

    dim = int(header["DIMENSION"])
    cap = float(header["CAPACITY"])
    if len(ew_nums) < dim * dim:
        raise ValueError("EDGE_WEIGHT_SECTION: expected %d numbers, got %d"
                         % (dim * dim, len(ew_nums)))
    D = np.array(ew_nums[:dim * dim], float).reshape(dim, dim)

    delivery = np.zeros(dim); pickup = np.zeros(dim)
    for row in pd_rows:
        idx = int(row[0]) - 1                            # 1-based id -> 0-based index
        pickup[idx] = row[PICKUP_COL]
        delivery[idx] = row[DELIVERY_COL]

    # move the depot to index 0 (these instances already list it as node 1)
    d0 = depot_id - 1
    if d0 != 0:
        order = [d0] + [k for k in range(dim) if k != d0]
        D = D[np.ix_(order, order)]
        delivery = delivery[order]; pickup = pickup[order]
    delivery[0] = 0.0; pickup[0] = 0.0

    return dict(n=dim - 1, capacity=cap, D=D, delivery=delivery, pickup=pickup)


def build_factor_instance(deth, sigma_frac=SIGMA_FRAC):
    """
    Overlay a rank-1 (single common factor F) demand model on the deterministic
    Dethloff instance (explicit distance matrix):
        delivery_i(F) = dbar_i + dco_i * F,   pickup_i(F) = pbar_i + pload_i * F
    Loadings are SYNTHETIC: a systematic shock proportional to the mean
    (sigma_frac * mean).  >>> Replace with your real factor model if you have one. <<<
    """
    n = deth["n"]
    D = np.asarray(deth["D"], float) * DIST_SCALE
    dbar = deth["delivery"].astype(float) * LOAD_SCALE; dbar[0] = 0.0
    pbar = deth["pickup"].astype(float) * LOAD_SCALE;   pbar[0] = 0.0
    dco = sigma_frac * dbar; dco[0] = 0.0           # delivery loading on F
    pload = sigma_frac * pbar; pload[0] = 0.0       # pickup loading on F
    qbar = pbar - dbar                              # mean net (pickup - delivery)
    lam = pload - dco                               # net loading on F
    return dict(n=n, D=D, dbar=dbar, pbar=pbar, dco=dco, pload=pload,
                qbar=qbar, lam=lam, capacity=float(deth["capacity"]) * LOAD_SCALE)


def sample_factors(N, seed=2026):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.0, 1.0, N), FMIN, FMAX)


# ===========================================================================
#  PART 2.  Factor order statistics + exact / surrogate CVaR
# ===========================================================================

def upper_env(lines):
    """Upper envelope (max) of lines value = B + A*F -> pieces [(f_left, B, A)]."""
    best = {}
    for (B, A) in lines:
        if (A not in best) or (B > best[A]):
            best[A] = B
    items = sorted(best.items())                    # (A, B) ascending in A
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
    """Precomputed factor order statistics + O(pieces*logN) exact CVaR."""

    def __init__(self, F, alpha=ALPHA):
        self.Fs = np.sort(np.asarray(F, float))
        self.N = len(self.Fs)
        self.m = max(1, int(np.ceil((1.0 - alpha) * self.N)))
        self.PF = np.concatenate([[0.0], np.cumsum(self.Fs)])
        self.Fl = self.Fs.tolist()
        self.alpha = alpha

    # --- order-statistics exact CVaR of a convex-PWL function of the scalar F ---
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
    """Per-cut surrogate: a sound O(pieces) lower bound on CVaR(peak)."""
    best = -np.inf
    for (B, A) in load_lines:
        v = B + A * (mu_hi if A >= 0 else mu_lo)
        if v > best:
            best = v
    return best


def exact_cvar_np(load_lines, fs):
    """Vectorised exact CVaR (good constant): peak = max_k(B_k+A_k F); top-m mean."""
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
        return fs.cvar(load_lines)               # sub-linear
    return exact_cvar_np(load_lines, fs)         # good constant


# ===========================================================================
#  PART 3.  Tropical-envelope pricing core (extended for B&P)
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
              farkas=False, label_budget=None, neighbor_k=None, bucket_cap=None,
              tol=REDCOST_TOL, max_cols=40, depot=0):
    """
    Min-reduced-cost robustly-feasible elementary route pricing.

    duals pi : array length n+1, pi[depot] = 0.
    forbidden / enforced : directed edges; enforce(i,j) is reduced to forbidding
               every (i,k!=j) and (k!=i,j).
    farkas   : drop travel cost in the reduced-cost test (rc = -sum pi); the
               returned 'cost' is still the REAL travel cost (master objective).

    HEURISTIC knobs (all None -> exact pricing):
      neighbor_k   : only extend to the k nearest customers.  This caps the
                     branching factor and is the real lever that tames n=50;
                     a per-bucket cap does NOT, because each visited-set is its
                     own state and there are exponentially many of them.
      label_budget : global cap on labels expanded per call -> the call is
                     guaranteed to terminate (it may then miss the true min-rc
                     column, so the LP bound is no longer exact).
      bucket_cap   : optional max labels kept per (node,vis) state (memory safety).

    Returns list of (customers_tuple, travel_cost, reduced_cost) with rc < -tol.
    """
    n = inst["n"]; D = inst["D"]
    dbar, dco_arr, qbar, lam = inst["dbar"], inst["dco"], inst["qbar"], inst["lam"]
    mu_hi, mu_lo = factor_tail_means(fs)

    # nearest-neighbour extension lists (customers only), if sparsified
    if neighbor_k is not None:
        neigh = {u: sorted((w for w in range(1, n + 1) if w != u),
                           key=lambda w: D[u, w])[:neighbor_k]
                 for u in range(0, n + 1)}
    else:
        allcust = list(range(1, n + 1))

    # enforce -> forbid expansion
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
    expanded = 0
    stop = False

    def ll_of(lab):
        return [(lab.Dbar + b, lab.dco + a) for (b, a) in lab.lines]

    for lvl in range(n + 1):
        if stop:
            break
        for (node, vis, lab) in levels.get(lvl, []):
            if label_budget is not None and expanded >= label_budget:
                stop = True
                break
            expanded += 1
            ll = ll_of(lab)
            if surrogate_cvar(ll, mu_hi, mu_lo) > Q + EPS:
                continue
            # completion (return to depot) -- always permitted
            if vis != 0 and (node, depot) not in eff:
                rc_full = lab.rc + (0.0 if farkas else D[node, depot])   # pi[depot] = 0
                if rc_full < -tol and exact_cvar_hybrid(ll, fs) <= Q + EPS:
                    order = _reconstruct(lab, parent)
                    cols.append((order, route_cost(order, inst, depot), rc_full))
            # extensions
            ext = neigh[node] if neighbor_k is not None else allcust
            for w in ext:
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
                if bucket_cap and len(newb) > bucket_cap:        # memory safety
                    newb.sort(key=lambda L: L.rc)
                    newb = newb[:bucket_cap]
                buckets[key] = newb
                if nlab in newb:
                    levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab))

    return _dedup_top(cols, max_cols)


def make_user_pricer(inst, fs, Q, label_budget=None, neighbor_k=None, bucket_cap=None):
    """Adapt env_price to the TropicalPricer contract."""
    n = inst["n"]

    def user_pricer(duals, forbidden_edges, enforced_edges, farkas):
        pi = np.zeros(n + 1)
        for i, val in duals.items():
            pi[i] = val
        return env_price(inst, fs, Q, pi,
                         forbidden=forbidden_edges, enforced=enforced_edges,
                         farkas=farkas, label_budget=label_budget,
                         neighbor_k=neighbor_k, bucket_cap=bucket_cap)

    return user_pricer


# ===========================================================================
#  PART 4.  SCIP Branch-and-Price
# ===========================================================================

class BPData:
    def __init__(self, customers, depot=0):
        self.depot = depot
        self.customers = list(customers)
        self.cover_cons = {}
        self.route_of_var = []            # list of (var, custs, arcs, cost)
        self.vars_using_arc = {}          # (i,j) -> [vars]  (arc key is hashable; vars in a list)
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
        return self.data.arc_conshdlr.active_decisions()

    def _inject(self, columns):
        d = self.data
        for (custs, cost, rc) in columns:
            if rc > -REDCOST_TOL:
                continue
            var = self.model.addVar(vtype="C", obj=float(cost), pricedVar=True)
            for c in custs:
                self.model.addConsCoeff(d.cover_cons[c], var, 1.0)
            arcs = route_arcs(custs, d.depot)
            d.route_of_var.append((var, tuple(custs), arcs, float(cost)))
            for a in arcs:
                d.vars_using_arc.setdefault(a, []).append(var)


class ArcConshdlr(Conshdlr):
    """Drives arc-branching from consenfolp and enforces decisions by propagation.

    Branching avoids createCons/addConsNode (whose signatures vary across PySCIPOpt
    versions): children are made with createChild, the (tail,head,dir) decision is
    stored by child node number, and the active decisions at any node are recovered
    by walking from the current node up to the root via getParent()."""

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.node_decision = {}        # node number -> (tail, head, dir)

    # ---- recover the branching decisions active at the current node ----
    def active_decisions(self):
        forbidden, enforced = set(), set()
        node = self.model.getCurrentNode()
        while node is not None:
            dec = self.node_decision.get(node.getNumber())
            if dec is not None:
                t, h, dr = dec
                (forbidden if dr == FORBID else enforced).add((t, h))
            node = node.getParent()
        return forbidden, enforced

    # ---- arc-flow helpers ----
    def _fractional_arc(self, sol):
        d = self.data
        flow = {}
        for (var, custs, arcs, cost) in d.route_of_var:
            val = self.model.getSolVal(sol, var)          # sol=None -> current LP
            if val <= FRAC_TOL:
                continue
            for a in arcs:
                flow[a] = flow.get(a, 0.0) + val
        cand, best = None, 1.0
        for a, f in flow.items():
            if abs(f - round(f)) > FRAC_TOL and abs(f - 0.5) < best:
                best, cand = abs(f - 0.5), a
        return cand

    def _branch_on(self, arc):
        tail, head = arc
        for dirn in (FORBID, ENFORCE):
            child = self.model.createChild(0, self.model.getLPObjVal())
            self.node_decision[child.getNumber()] = (tail, head, dirn)

    # ---- enforcement: branch on a fractional arc (drives the whole tree) ----
    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        arc = self._fractional_arc(None)                  # None -> current LP solution
        if arc is None:
            return {"result": SCIP_RESULT.FEASIBLE}
        self._branch_on(arc)
        return {"result": SCIP_RESULT.BRANCHED}

    # a candidate solution is integer-feasible iff its arc-flow is integral
    def conscheck(self, constraints, solution, checkintegrality, checklprows,
                  printreason, completely):
        if self._fractional_arc(solution) is None:
            return {"result": SCIP_RESULT.FEASIBLE}
        return {"result": SCIP_RESULT.INFEASIBLE}

    # ---- propagation: fix existing columns that violate active decisions ----
    def consprop(self, constraints, nusefulconss, nmarkedconss, proptiming):
        d = self.data
        forbidden, enforced = self.active_decisions()
        disallowed = set(forbidden)
        if enforced:
            for (a, b) in list(d.vars_using_arc.keys()):
                for (i, j) in enforced:
                    if (a == i and b != j) or (b == j and a != i):
                        disallowed.add((a, b))
        reduced = False
        for arc in disallowed:
            for var in d.vars_using_arc.get(arc, []):
                infeasible, tightened = self.model.tightenVarUb(var, 0.0)  # VERSION
                reduced = reduced or tightened
                if infeasible:
                    return {"result": SCIP_RESULT.CUTOFF}
        return {"result": SCIP_RESULT.REDUCEDDOM if reduced else SCIP_RESULT.DIDNOTFIND}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        return


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
        init_vars.append((v, tuple(custs)))

    # one >= 1 set-COVERING row per customer (always coverable; overlaps are repaired
    # into a clean partition at reporting). MODIFIABLE so pricing may add columns.
    for c in data.customers:
        expr = quicksum(v for (v, custs) in init_vars if c in custs)
        cons = model.addCons(expr >= 1.0, name="cover_%d" % c,
                             modifiable=True, separate=False)
        data.cover_cons[c] = cons

    for v, custs in init_vars:
        arcs = route_arcs(custs, data.depot)
        data.route_of_var.append((v, custs, arcs, None))
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
        "stores/propagates arc decisions and branches on fractional arc-flow",
        sepapriority=0, enfopriority=-1, chckpriority=-1,
        sepafreq=-1, propfreq=1, eagerfreq=-1, maxprerounds=0,
        delaysepa=False, delayprop=False, needscons=False)

    configure_for_branch_and_price(model)
    return model


# ===========================================================================
#  PART 5.  main
# ===========================================================================

def feasible_singletons(inst, fs, Q):
    """Singleton routes depot-i-depot. Keep ALL of them so every customer has a
    covering column (a dropped singleton would make its row 0 >= 1, infeasible)."""
    n = inst["n"]
    init, bad = [], []
    for i in range(1, n + 1):
        # peak cuts of a single-customer route: deliver d_i (k=0), pickup p_i (k=1)
        load_lines = [(inst["dbar"][i], inst["dco"][i]),
                      (inst["pbar"][i], inst["pload"][i])]
        if exact_cvar_hybrid(load_lines, fs) > Q + EPS:
            bad.append(i)
        init.append(((i,), route_cost((i,), inst)))
    if bad:
        print("WARNING: %d singleton(s) exceed robust capacity Q=%.2f: %s" % (len(bad), Q, bad))
        print("         (kept anyway to preserve coverage; the instance may be "
              "robustly infeasible for these customers)")
    return init


def main():
    if len(sys.argv) < 2:
        print("usage: python dethloff_tropical_bp.py <file.vrpspd> "
              "[N_scenarios] [label_budget] [neighbor_k]")
        print("  exact (small n only) : python dethloff_tropical_bp.py inst.vrpspd")
        print("  heuristic (n=50)     : python dethloff_tropical_bp.py inst.vrpspd 1000 200000 10")
        sys.exit(1)
    path = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N
    label_budget = int(sys.argv[3]) if len(sys.argv) > 3 else None   # global expansion cap
    neighbor_k = int(sys.argv[4]) if len(sys.argv) > 4 else None     # nearest-neighbour fan-out

    # ---- data ingestion ----
    deth = parse_dethloff(path)
    inst = build_factor_instance(deth, sigma_frac=SIGMA_FRAC)
    Q = inst["capacity"]                       # scaled shadow capacity Q_eff (CVaR constraint)
    print("instance %s : n=%d customers, capacity=%.3f (raw %.0f x %g), alpha=%.2f, sigma=%.2f"
          % (path, inst["n"], Q, deth["capacity"], LOAD_SCALE, ALPHA, SIGMA_FRAC))
    heuristic = (label_budget is not None) or (neighbor_k is not None)
    if heuristic:
        print("HEURISTIC pricing: label_budget=%s neighbor_k=%s  (LP bound NOT exact)"
              % (label_budget, neighbor_k))
    elif inst["n"] > 15:
        print("NOTE: exact elementary pricing on n=%d is pure-Python and will likely NOT "
              "terminate.\n      Run heuristically, e.g.:  python %s %s 1000 200000 10"
              % (inst["n"], sys.argv[0], path))

    # ---- FStats built EXACTLY ONCE, reference passed into the pricer ----
    fs = FStats(sample_factors(N), alpha=ALPHA)

    # ---- master initial feasible singletons ----
    customers = list(range(1, inst["n"] + 1))
    init = feasible_singletons(inst, fs, Q)

    # ---- assemble + solve ----
    data = BPData(customers, depot=0)
    data.inst = inst                                   # for repaired-route costing in the report
    user_pricer = make_user_pricer(inst, fs, Q,
                                   label_budget=label_budget, neighbor_k=neighbor_k)
    model = build_branch_and_price(data, init, user_pricer)
    model.optimize()

    # ---- report (repair covering overlaps into a clean partition) ----
    print("\nstatus      :", model.getStatus())
    if model.getNSols() == 0:
        print("no feasible solution found")
        return
    sol = model.getBestSol()                                 # the INCUMBENT, not the LP
    selected = [custs for (v, custs, arcs, cost) in data.route_of_var
                if model.getSolVal(sol, v) > 0.5]
    assigned, repaired = set(), []
    for custs in sorted(selected, key=lambda r: -len(r)):   # keep longer routes intact
        kept = tuple(c for c in custs if c not in assigned)
        if kept:
            assigned.update(kept)
            repaired.append(kept)
    part_cost = sum(route_cost(r, inst) for r in repaired)
    print("incumbent objective: %.4f" % model.getObjVal())
    print("partition objective: %.4f  (after removing any over-covered customers)" % part_cost)
    print("routes (%d):" % len(repaired))
    for r in repaired:
        print("   depot -> " + " -> ".join(map(str, r)) + " -> depot")
    missing = [c for c in customers if c not in assigned]
    if missing:
        print("WARNING: customers not covered at all:", missing)
    else:
        print("coverage    : OK (every customer visited exactly once)")


if __name__ == "__main__":
    main()
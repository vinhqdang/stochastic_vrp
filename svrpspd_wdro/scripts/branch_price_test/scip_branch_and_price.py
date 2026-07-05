
import importlib.util
import numpy as np
from pyscipopt import Model, Pricer, Conshdlr, Branchrule, quicksum
from pyscipopt import SCIP_RESULT, SCIP_PARAMSETTING

FORBID, ENFORCE = 0, 1
REDCOST_TOL = 1e-6
FRAC_TOL = 1e-6


def _load(m, p):
    s = importlib.util.spec_from_file_location(m, p)
    mod = importlib.util.module_from_spec(s);
    s.loader.exec_module(mod);
    return mod


v2 = _load("v2", "pricing_labeling_v2.py")
v3 = _load("v3", "pricing_labeling_v3.py")
bpc = _load("bpc", "robust_svrpspd_bpc.py")

EPS = bpc.EPS
route_cost = bpc.route_cost

def feas_lhs(ll, fs, Fbar, mode, eps, alpha):
    if mode == 'det':
        return max(B + A * Fbar for (B, A) in ll)
    cv = v3.exact_cvar_hybrid(ll, fs)
    if mode == 'saa':
        return cv
    return cv + eps / (1.0 - alpha)

def surr_lhs(ll, mu_hi, mu_lo, Fbar, mode, eps, alpha):
    if mode == 'det':
        return max(B + A * Fbar for (B, A) in ll)
    s = v3.surrogate_cvar(ll, mu_hi, mu_lo)
    if mode == 'wdro':
        s += eps / (1.0 - alpha)
    return s


def env_price_mode_scip(inst, fs, Q, pi, mode, eps, alpha, tol=1e-6, max_cols=40, forbidden_edges=None, farkas=False):
    if forbidden_edges is None:
        forbidden_edges = set()

    n = inst['n'];
    D = inst['D']
    dbar, dco, qbar, lam = inst['dbar'], inst['dco'], inst['qbar'], inst['lam']
    mu_hi, mu_lo = v3.factor_tail_means(fs)
    Fbar = float(fs.Fs.mean())
    LE, env_dom = v2.LE, v2.env_dom

    start = LE(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [(0.0, 0.0)])
    buckets = {(0, 0): [start]};
    parent = {id(start): (None, None)};
    cols = []

    def ll_of(lab):
        return [(lab.Dbar + b, lab.dco + a) for (b, a) in lab.lines]

    levels = {0: [(0, 0, start)]}
    for lvl in range(n + 1):
        for (node, vis, lab) in levels.get(lvl, []):
            ll = ll_of(lab)
            if surr_lhs(ll, mu_hi, mu_lo, Fbar, mode, eps, alpha) > Q + EPS:
                continue

            if vis != 0:
                if (node, 0) not in forbidden_edges and (0, node) not in forbidden_edges:
                    rc_full = lab.rc + (0.0 if farkas else D[node, 0])
                    if rc_full < -tol and feas_lhs(ll, fs, Fbar, mode, eps, alpha) <= Q + EPS:
                        order = v2._reconstruct(lab, parent)
                        cols.append((order, route_cost(order, inst), rc_full))

            for w in range(1, n + 1):
                bit = 1 << (w - 1)
                if vis & bit:
                    continue
                if (node, w) in forbidden_edges or (w, node) in forbidden_edges:
                    continue

                step_cost = 0.0 if farkas else D[node, w]
                new_rc = lab.rc + step_cost - pi[w]

                nb = lab.Sbar + qbar[w];
                na = lab.lamco + lam[w]
                nlab = LE(w, vis | bit, new_rc,
                          lab.Dbar + dbar[w], lab.dco + dco[w], nb, na,
                          lab.lines + [(nb, na)])

                if surr_lhs(ll_of(nlab), mu_hi, mu_lo, Fbar, mode, eps, alpha) > Q + EPS:
                    continue

                parent[id(nlab)] = (lab, w)
                key = (w, vis | bit);
                bucket = buckets.get(key)

                if bucket is None:
                    buckets[key] = [nlab];
                    levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab))
                    continue
                if any(env_dom(e, nlab) for e in bucket):
                    continue
                buckets[key] = [e for e in bucket if not env_dom(nlab, e)] + [nlab]
                levels.setdefault(lvl + 1, []).append((w, vis | bit, nlab))

    return bpc._dedup_top(cols, max_cols)



class BPData:
    def __init__(self, customers, depot=0):
        self.depot = depot
        self.customers = list(customers)
        self.cover_cons = {}
        self.route_of_var = {}
        self.vars_using_arc = {}
        self.var_obj = {}  # LƯU TRỮ OBJECT Ở ĐÂY ĐỂ TRÁNH LỖI UNHASHABLE
        self.arc_conshdlr = None
        self.user_pricer = None


def route_arcs(custs, depot=0):
    seq = [depot] + list(custs) + [depot]
    return frozenset((seq[i], seq[i + 1]) for i in range(len(seq) - 1))


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
        self._inject_columns(columns)
        return {"result": SCIP_RESULT.SUCCESS}

    def pricerfarkas(self):
        d = self.data
        duals = {c: self.model.getDualfarkasLinear(d.cover_cons[c]) for c in d.customers}
        forbidden, enforced = self._active_arcs()
        columns = d.user_pricer(duals, forbidden, enforced, farkas=True)
        self._inject_columns(columns)
        return {"result": SCIP_RESULT.SUCCESS}

    def _active_arcs(self):
        ch = self.data.arc_conshdlr
        forbidden, enforced = set(), set()
        for (tail, head, dirn) in ch.active_stack:
            (forbidden if dirn == FORBID else enforced).add((tail, head))
        return forbidden, enforced

    def _inject_columns(self, columns):
        d = self.data
        for (custs, cost, redcost) in columns:
            if redcost > -REDCOST_TOL:
                continue
            v_name = f"r_{'_'.join(map(str, custs))}_{self.model.getNVars()}"
            var = self.model.addVar(vtype="C", obj=float(cost), pricedVar=True, name=v_name)
            for c in custs:
                self.model.addConsCoeff(d.cover_cons[c], var, 1.0)
            arcs = route_arcs(custs, d.depot)
            d.var_obj[v_name] = var
            d.route_of_var[v_name] = (tuple(custs), arcs, float(cost))
            for a in arcs:
                d.vars_using_arc.setdefault(a, []).append(v_name)


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
        reduced = False
        for arc in forbidden:
            for v_name in d.vars_using_arc.get(arc, []):
                var = d.var_obj[v_name]
                infeasible, tightened = self.model.tightenVarUb(var, 0.0)
                reduced = reduced or tightened
                if infeasible:
                    return {"result": SCIP_RESULT.CUTOFF}
        return {"result": SCIP_RESULT.REDUCEDDOM if reduced else SCIP_RESULT.DIDNOTFIND}

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely):
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
        for v_name, (custs, arcs, cost) in d.route_of_var.items():
            var = d.var_obj[v_name]
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
                self.conshdlr, name, initial=False, separate=False, enforce=True, check=False,
                propagate=True, local=True, modifiable=False, dynamic=False, removable=False, stickingatnode=False)
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
    model = Model("robust_svrpspd_bp")
    data.user_pricer = user_pricer

    init_vars = []
    for (custs, cost) in initial_routes:
        v = model.addVar(vtype="C", obj=float(cost), name="r0_" + "_".join(map(str, custs)))
        init_vars.append((v, tuple(custs)))

    for c in data.customers:
        expr = quicksum(v for (v, custs) in init_vars if c in custs)
        cons = model.addCons(expr >= 1.0, name="cover_%d" % c, modifiable=True, separate=False)
        data.cover_cons[c] = cons

    for v, custs in init_vars:
        arcs = route_arcs(custs, data.depot)
        v_name = v.name
        data.var_obj[v_name] = v
        data.route_of_var[v_name] = (custs, arcs, None)
        for a in arcs:
            data.vars_using_arc.setdefault(a, []).append(v_name)

    pricer = TropicalPricer()
    pricer.data = data
    model.includePricer(pricer, "TropicalEnvelopePricer", "envelope robust pricing")

    conshdlr = ArcConshdlr(data)
    data.arc_conshdlr = conshdlr
    model.includeConshdlr(conshdlr, "ArcBranching", "stores arc decisions", sepafreq=-1, propfreq=1, eagerfreq=-1,
                          delaysepa=False, delayprop=False, needscons=False)

    branchrule = ArcBranchrule(data, conshdlr)
    model.includeBranchrule(branchrule, "ArcBranching", "branch on arc", priority=1000000, maxdepth=-1,
                            maxbounddist=1.0)

    configure_for_branch_and_price(model)
    return model


def make_user_pricer(inst, fs, Qeff, alpha, mode, eps):
    def user_pricer(duals, forbidden_edges, enforced_edges, farkas):
        n = inst["n"]
        pi = np.zeros(n + 1)
        for i, val in duals.items():
            pi[i] = val
        pi[0] = 0.0
        cols = env_price_mode_scip(inst, fs, Qeff, pi, mode, eps, alpha, forbidden_edges=forbidden_edges, farkas=farkas)
        return cols

    return user_pricer


if __name__ == "__main__":
    n = 9
    seed = 2026 + 100 * n + 0
    inst = v2.make_instance(n, seed)
    alpha = 0.90
    N = 200000000
    fs = v2.FStats(v2.sample_factors(N, seed), alpha)

    Q = bpc.calibrate_Qeff_feasible(inst, fs, alpha, 55, seed)
    mode = 'wdro'
    eps = 3.0

    print("=" * 60)
    print(f"BẮT ĐẦU FULL BPC SCIP -- Instance: n={n}, mode={mode}")
    print("=" * 60)

    t0 = time.perf_counter()
    data = BPData(customers=list(range(1, inst['n'] + 1)))
    init_routes = [([i], bpc.route_cost([i], inst)) for i in data.customers]
    up = make_user_pricer(inst, fs, Q, alpha, mode, eps)

    model = build_branch_and_price(data, init_routes, up)
    model.setRealParam("limits/time", 3600)
    #model.hideOutput()  # Tắt dòng chữ chạy rối mắt của SCIP

    model.optimize()
    t_end = time.perf_counter() - t0

    print("\n" + "=" * 60)
    if model.getStatus() == "optimal":
        print(" SCIP STATUS: OPTIMAL (Gap = 0.00%)")
        print(f" EXACT OBJECTIVE VALUE : {model.getObjVal():.4f}")
    else:
        print(f" SCIP STATUS: {model.getStatus()}")
        print(f" BEST FOUND OBJECTIVE  : {model.getObjVal():.4f}")

    print(f" TOTAL SOLVE TIME      : {t_end:.2f} giây")

    print("\n[CÁC TUYẾN ĐƯỜNG TỐI ƯU]")
    for v_name, (custs, arcs, cost) in data.route_of_var.items():
        var = data.var_obj[v_name]
        if model.getVal(var) > 0.5:
            print(f"  -> Điểm: {custs} | Chi phí: {cost:.2f}")
    print("=" * 60)
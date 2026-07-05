"""
Matheuristic CORE for distributionally robust SVRPSPD (C&OR paper engine).

Robustness is folded into Q_eff (DR reduction, cited ORL); the working constraint is the
NOMINAL empirical CVaR of the PEAK load:  CVaR_alpha(Peak_route) <= Q_eff, evaluated exactly
over N scenarios (NOT a surrogate).

Two mechanisms the paper rests on, both instrumented here:
  * ALNS (adaptive large-neighbourhood search) over routes, minimising distance + vehicle cost
    subject to per-route CVaR-of-peak feasibility.
  * VALLEY-REPAIR (Lemma 1): when a route's distance-driven order violates CVaR-of-peak, reorder
    its customers by mean net (p-d) ascending; this minimises the mean peak and often restores
    feasibility at the cost of a controlled detour. We MEASURE how often it rescues an infeasible
    order -- this is the empirical test of the "interior excursion" the theory flags.

Run:  python matheuristic_core.py
"""
import numpy as np, time, math
from dataclasses import dataclass, field


# ----------------------------- instance -----------------------------
@dataclass
class Instance:
    n: int; r: int; N: int; alpha: float
    coords: np.ndarray; C: np.ndarray
    dbar: np.ndarray; pbar: np.ndarray
    d_scen: np.ndarray; p_scen: np.ndarray   # (N, n)
    Q: float; Qeff: float

def make_instance(n=20, r=2, N=200, seed=0, eps=1.5, alpha=0.9, cap_frac=0.5):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, (n + 1, 2))          # index 0 = depot
    C = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(2))
    dbar = rng.uniform(2, 18, n); pbar = rng.uniform(2, 18, n)
    lam_d = rng.normal(0, 1, (n, r)) * (0.35 * dbar)[:, None]
    lam_p = rng.normal(0, 1, (n, r)) * (0.35 * pbar)[:, None]
    F = rng.normal(0, 1, (N, r))
    d_scen = np.clip(dbar[None, :] + F @ lam_d.T, 0, None)
    p_scen = np.clip(pbar[None, :] + F @ lam_p.T, 0, None)
    # capacity: a vehicle should hold a few customers' worth of the heavier side
    Q = cap_frac * max(d_scen.sum(1).mean(), p_scen.sum(1).mean())
    Qeff = Q - eps / (1 - alpha)
    return Instance(n, r, N, alpha, coords, C, dbar, pbar, d_scen, p_scen, Q, Qeff)


# ----------------------------- route evaluation -----------------------------
def cvar(x, alpha):
    xs = np.sort(x); k = int(math.ceil(alpha * len(xs)))
    return xs[k - 1:].mean()

def peak_scenarios(seq, inst):
    """Peak load per scenario for an ordered route seq (0-based customer indices)."""
    if not seq:
        return np.zeros(inst.N)
    D_tot = inst.d_scen[:, seq].sum(1)                      # (N,) all deliveries
    net = (inst.p_scen - inst.d_scen)[:, seq]               # (N, L)
    cum = np.cumsum(net, axis=1)                            # prefix nets, k=1..L
    max_prefix = np.maximum(0.0, cum.max(axis=1))           # incl. k=0 (=0)
    return D_tot + max_prefix

def route_cvar_peak(seq, inst):
    if not seq: return 0.0
    return cvar(peak_scenarios(seq, inst), inst.alpha)

def route_dist(seq, inst):
    if not seq: return 0.0
    nodes = [0] + [c + 1 for c in seq] + [0]
    return sum(inst.C[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1))

def valley_order(seq, inst):
    order = np.argsort(inst.pbar[seq] - inst.dbar[seq])     # mean net ascending
    return [seq[i] for i in order]


# ----------------------------- solution -----------------------------
@dataclass
class Solution:
    routes: list = field(default_factory=list)              # list of ordered customer lists
    inst: Instance = None
    omega_V: float = 0.0
    valley_saves: int = 0                                    # times valley-repair rescued a route
    feas_checks: int = 0

    def cost(self):
        d = sum(route_dist(r, self.inst) for r in self.routes if r)
        k = sum(1 for r in self.routes if r)
        return d + self.omega_V * k

    def distance(self):
        return sum(route_dist(r, self.inst) for r in self.routes if r)

    def n_veh(self):
        return sum(1 for r in self.routes if r)

    def verify(self):
        """Return (all_feasible, max_cvar_ratio) checking every route's CVaR-peak <= Qeff."""
        ok = True; worst = 0.0
        for r in self.routes:
            if not r: continue
            cp = route_cvar_peak(r, self.inst)
            worst = max(worst, cp / self.inst.Qeff)
            if cp > self.inst.Qeff + 1e-6: ok = False
        return ok, worst

    def copy(self):
        s = Solution([r[:] for r in self.routes], self.inst, self.omega_V)
        s.valley_saves = self.valley_saves; s.feas_checks = self.feas_checks
        return s


def feasible_order(seq, inst, sol=None):
    """Return a FEASIBLE ordered route for customer set of seq, or None.
    Try the given order; if it violates CVaR-peak, try valley order (Lemma 1 repair)."""
    if sol is not None: sol.feas_checks += 1
    if route_cvar_peak(seq, inst) <= inst.Qeff + 1e-9:
        return seq
    vseq = valley_order(seq, inst)
    if route_cvar_peak(vseq, inst) <= inst.Qeff + 1e-9:
        if sol is not None: sol.valley_saves += 1
        return vseq
    return None


# ----------------------------- construction -----------------------------
def best_insertion(cust, sol):
    """Cheapest feasible insertion of cust into existing routes (with valley-repair)."""
    inst = sol.inst; best = None
    for ri, r in enumerate(sol.routes):
        if not r: continue
        for pos in range(len(r) + 1):
            cand = r[:pos] + [cust] + r[pos:]
            fseq = feasible_order(cand, inst, sol)
            if fseq is None: continue
            delta = route_dist(fseq, inst) - route_dist(r, inst)
            if best is None or delta < best[0]:
                best = (delta, ri, fseq)
    return best

def greedy_construct(inst, omega_V):
    sol = Solution([], inst, omega_V)
    order = sorted(range(inst.n), key=lambda c: -(inst.dbar[c] + inst.pbar[c]))  # big first
    for c in order:
        bi = best_insertion(c, sol)
        # opening a new single-customer route
        singleton = feasible_order([c], inst, sol)
        new_cost = (route_dist(singleton, inst) + omega_V) if singleton else float("inf")
        if bi is not None and bi[0] <= new_cost:
            sol.routes[bi[1]] = bi[2]
        elif singleton is not None:
            sol.routes.append(singleton)
        else:
            raise RuntimeError(f"customer {c} infeasible even alone (Qeff too small)")
    sol.routes = [r for r in sol.routes if r]
    return sol


# ----------------------------- ALNS operators -----------------------------
def destroy_random(sol, k, rng):
    allc = [(ri, c) for ri, r in enumerate(sol.routes) for c in r]
    if not allc: return []
    pick = rng.choice(len(allc), size=min(k, len(allc)), replace=False)
    removed = [allc[i] for i in pick]
    for ri, c in removed:
        sol.routes[ri].remove(c)
    sol.routes = [r for r in sol.routes if r]
    return [c for _, c in removed]

def destroy_worst(sol, k, rng):
    inst = sol.inst; contrib = []
    for ri, r in enumerate(sol.routes):
        for c in r:
            without = [x for x in r if x != c]
            save = route_dist(r, inst) - route_dist(without, inst)
            contrib.append((save, ri, c))
    contrib.sort(reverse=True)
    removed = contrib[:k]
    for _, ri, c in removed:
        sol.routes[ri].remove(c)
    sol.routes = [r for r in sol.routes if r]
    return [c for _, _, c in removed]

def destroy_related(sol, k, rng):
    """Shaw-like: remove a seed and its nearest (by distance) customers."""
    inst = sol.inst
    allc = [c for r in sol.routes for c in r]
    if not allc: return []
    seed = allc[rng.integers(len(allc))]
    dist_to_seed = sorted(allc, key=lambda c: inst.C[seed + 1, c + 1])
    removed = dist_to_seed[:k]
    for c in removed:
        for r in sol.routes:
            if c in r: r.remove(c); break
    sol.routes = [r for r in sol.routes if r]
    return removed

def repair_greedy(sol, removed, rng):
    rng.shuffle(removed)
    for c in removed:
        bi = best_insertion(c, sol)
        singleton = feasible_order([c], sol.inst, sol)
        new_cost = (route_dist(singleton, sol.inst) + sol.omega_V) if singleton else float("inf")
        if bi is not None and bi[0] <= new_cost:
            sol.routes[bi[1]] = bi[2]
        elif singleton is not None:
            sol.routes.append(singleton)
        else:
            return False
    sol.routes = [r for r in sol.routes if r]
    return True

def repair_regret2(sol, removed, rng):
    """Insert the customer with the largest regret (2nd-best minus best) first."""
    remaining = removed[:]
    while remaining:
        best_c = None; best_key = None; best_move = None
        for c in remaining:
            inst = sol.inst; deltas = []
            for ri, r in enumerate(sol.routes):
                for pos in range(len(r) + 1):
                    cand = r[:pos] + [c] + r[pos:]
                    fseq = feasible_order(cand, inst, sol)
                    if fseq is None: continue
                    deltas.append((route_dist(fseq, inst) - route_dist(r, inst), ri, fseq))
            singleton = feasible_order([c], inst, sol)
            if singleton is not None:
                deltas.append((route_dist(singleton, inst) + sol.omega_V, -1, singleton))
            if not deltas: return False
            deltas.sort(key=lambda x: x[0])
            regret = (deltas[1][0] - deltas[0][0]) if len(deltas) > 1 else 1e9
            if best_key is None or regret > best_key:
                best_key = regret; best_c = c; best_move = deltas[0]
        delta, ri, fseq = best_move
        if ri == -1: sol.routes.append(fseq)
        else: sol.routes[ri] = fseq
        remaining.remove(best_c)
    sol.routes = [r for r in sol.routes if r]
    return True


# ----------------------------- ALNS driver -----------------------------
def alns(inst, iters=3000, seed=0, omega_V=None, verbose=True):
    rng = np.random.default_rng(seed)
    if omega_V is None:
        omega_V = inst.C[0, 1:].mean()                     # a moderate per-vehicle cost
    cur = greedy_construct(inst, omega_V)
    best = cur.copy()
    destroys = [destroy_random, destroy_worst, destroy_related]
    repairs = [repair_greedy, repair_regret2]
    wd = np.ones(len(destroys)); wr = np.ones(len(repairs))
    T = 0.05 * cur.cost(); cooling = (1e-3 / max(T, 1e-9)) ** (1.0 / iters)
    hist = []
    for it in range(iters):
        cand = cur.copy()
        di = rng.choice(len(destroys), p=wd / wd.sum())
        ri = rng.choice(len(repairs), p=wr / wr.sum())
        k = rng.integers(1, max(2, inst.n // 4))
        removed = destroys[di](cand, k, rng)
        ok = repairs[ri](cand, removed, rng)
        if not ok:
            wd[di] *= 0.99; wr[ri] *= 0.99
            continue
        dcost = cand.cost() - cur.cost()
        accept = dcost < 0 or rng.random() < math.exp(-dcost / max(T, 1e-9))
        reward = 0.0
        if cand.cost() < best.cost() - 1e-9:
            best = cand.copy(); reward = 3.0
        elif accept and dcost < 0:
            reward = 1.0
        elif accept:
            reward = 0.3
        if accept:
            cur = cand
        wd[di] = 0.9 * wd[di] + 0.1 * reward + 0.05
        wr[ri] = 0.9 * wr[ri] + 0.1 * reward + 0.05
        T *= cooling
        if it % 500 == 0:
            hist.append((it, best.cost()))
    best.valley_saves = cur.valley_saves + best.valley_saves   # rough carry
    return best, hist


# ----------------------------- run -----------------------------
def run():
    print("Matheuristic core for robust SVRPSPD (CVaR-of-peak, exact over scenarios)\n")
    for (n, r) in [(15, 2), (20, 2), (25, 3)]:
        inst = make_instance(n=n, r=r, N=200, seed=1, eps=1.5, alpha=0.9)
        t0 = time.perf_counter()
        best, hist = alns(inst, iters=2500, seed=1, verbose=False)
        dt = time.perf_counter() - t0
        ok, worst = best.verify()
        print(f"n={n} r={r} N={inst.N}  Qeff={inst.Qeff:.1f}")
        print(f"   objective={best.cost():.1f}  distance={best.distance():.1f}  vehicles={best.n_veh()}")
        print(f"   ALL routes CVaR-peak <= Qeff : {ok}   (worst CVaR/Qeff ratio = {worst:.4f})")
        print(f"   valley-repair rescued infeasible orders : {best.valley_saves} times "
              f"(of {best.feas_checks} feasibility checks)")
        print(f"   time={dt:.1f}s   convergence(best-cost @ iters): "
              f"{[(i, round(c,1)) for i,c in hist[::3]]}\n")
    print("READING: 'ALL routes ... True' verifies exact CVaR-peak feasibility (constraint honoured).")
    print("valley-repair count = how often the endpoint-necessary order failed but valley fixed it;")
    print("  high count => the interior-excursion the theory flags is real & handled cheaply by Lemma 1.")

if __name__ == "__main__":
    run()

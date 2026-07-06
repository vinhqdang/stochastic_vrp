

import sys, re, math, random, time, glob
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st


ALPHA      = 0.90      # CVaR level -> per-route failure target 1-alpha = 0.10
CV         = 0.30      # coefficient of variation of each demand component
RHO        = 0.6       # inter-node correlation (Gaussian copula) for gate + out-of-sample
DIST       = "gamma"   # data family for the empirical Phat (gate)
EPS_FRAC   = 0.15      # W-DRO ambiguity: Q_eff = Q*(1-EPS_FRAC)
N_DATA     = 1000      # scenarios in the empirical Phat (the gate)
N_MC       = 10000     # out-of-sample evaluation days, per test shape
SHAPE_W    = {"normal": 0.70, "right": 0.10, "left": 0.10, "heavy": 0.10}  # Gauss/SkewR/SkewL/Heavy (sum=1)


MIXTURES = [
    ("headline_70/10/10/10_logn",  {"normal": 0.70, "right": 0.10, "left": 0.10, "heavy": 0.10}, "lognormal"),
    ("balanced_40/20/20/20_logn",  {"normal": 0.40, "right": 0.20, "left": 0.20, "heavy": 0.20}, "lognormal"),
    ("mid_55/15/15/15_logn",       {"normal": 0.55, "right": 0.15, "left": 0.15, "heavy": 0.15}, "lognormal"),
    ("heavywt_50/10/10/30_logn",   {"normal": 0.50, "right": 0.10, "left": 0.10, "heavy": 0.30}, "lognormal"),
    ("noheavy_70/15/15/0",         {"normal": 0.70, "right": 0.15, "left": 0.15, "heavy": 0.00}, "lognormal"),
    ("headline_70/10/10/10_studt", {"normal": 0.70, "right": 0.10, "left": 0.10, "heavy": 0.10}, "studentt"),
    ("heavywt_50/10/10/30_studt",  {"normal": 0.50, "right": 0.10, "left": 0.10, "heavy": 0.30}, "studentt"),
]
SEED       = 7
OMEGA_RATIO = 50.0     
GOUNARIS_ALPHA = 1.0     # factor = 0.50  (nếu overshoot K=7/EVx<0.01 thì lùi 0.80 -> factor 0.40)
GOUNARIS_BETA  = 0.50
CUI_ALPHA      = 0.30     # hat_d = 0.30*(p+d)
CUI_GAMMA      = 0.80     # budget = 0.80 * #visited
TLIM        = 60.0     
NO_IMPROVE  = 15.0     # early-stop: break if no improving move for this many seconds
DATA_DIR    = "dethloff_data"

Z_CVAR = float(st.norm.pdf(st.norm.ppf(ALPHA)) / (1.0 - ALPHA))  


def _lines_after(txt, tag):
    out, cap = [], False
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        if not cap:
            if s.upper().startswith(tag):
                cap = True
            continue
        if any(ch.isalpha() for ch in s):          
            break
        out.append(s)
    return out


def _header_int(txt, key):
    for line in txt.splitlines():
        if line.strip().upper().startswith(key):
            m = re.findall(r"-?\d+", line)
            if m:
                return int(m[-1])
    return None


def _header_float(txt, key):
    for line in txt.splitlines():
        if line.strip().upper().startswith(key):
            m = re.findall(r"-?\d+\.?\d*", line)
            if m:
                return float(m[-1])
    return None


def _parse_pd(txt, n):
    dem = np.zeros((n, 2), dtype=float)
    for row in _lines_after(txt, "PICKUP_AND_DELIVERY_SECTION"):
        t = row.split()
        if len(t) < 3:
            continue
        idx = int(float(t[0])) - 1
        if 0 <= idx < n:
            dem[idx, 0] = float(t[-2])             # delivery
            dem[idx, 1] = float(t[-1])             # pickup
    return dem


def parse_dethloff(path):
    txt = Path(path).read_text(errors="ignore")
    n = _header_int(txt, "DIMENSION")
    Q = _header_float(txt, "CAPACITY")
    toks = []
    for row in _lines_after(txt, "EDGE_WEIGHT_SECTION"):
        toks.extend(row.split())
    vals = [int(float(t)) for t in toks]
    if n is None or len(vals) != n * n:
        raise ValueError("EDGE_WEIGHT_SECTION: got %d tokens, expected n*n=%s (need the FULL matrix)"
                         % (len(vals), None if n is None else n * n))
    D = np.array(vals, dtype=np.int64).reshape(n, n)
    dem = _parse_pd(txt, n)
    return D, dem, Q, n, 10000                       


def _marginal_ppf(u, mu, cv, dist):
    if dist == "normal":
        return np.clip(st.norm.ppf(u, mu, cv * mu), 0, None)
    if dist == "uniform":
        half = math.sqrt(3.0) * cv * mu
        return np.clip(mu - half + 2.0 * half * u, 0, None)
    if dist == "left":
        k = 1.0 / (cv * cv)
        g = st.gamma.ppf(1.0 - u, k, scale=mu / k)
        return np.clip(2.0 * mu - g, 0, None)
    if dist == "lognormal":                          # heavy (sub-exponential), matched mean & cv
        s2 = math.log(1.0 + cv * cv)
        return np.exp((math.log(mu) - 0.5 * s2) + math.sqrt(s2) * st.norm.ppf(u))
    if dist == "studentt":                           # heavy (power-law tail, df=4), matched mean & cv
        nu = 4.0
        s = cv * mu * math.sqrt((nu - 2.0) / nu)     # scale so Var = (cv*mu)^2
        return np.clip(mu + s * st.t.ppf(u, nu), 0, None)
    k = 1.0 / (cv * cv)                              # gamma / right (default)
    return st.gamma.ppf(u, k, scale=mu / k)


def sample_demands(mean, n, N, cv, dist, rng, rho=None):
    """(N, n) non-negative demands, per-column mean `mean`, cv `cv`. rho>0 -> Gaussian copula
    equicorrelation across active customers (marginals unchanged). rho=0 -> independent draws."""
    if rho is None:
        rho = RHO
    out = np.zeros((N, n))
    active = [i for i in range(1, n) if mean[i] > 0]
    if not active:
        return out
    if rho and rho > 0.0:
        m = len(active)
        Sigma = np.full((m, m), float(rho)); np.fill_diagonal(Sigma, 1.0)
        L = np.linalg.cholesky(Sigma)
        Z = rng.standard_normal((N, m)) @ L.T
        U = np.clip(st.norm.cdf(Z), 1e-12, 1.0 - 1e-12)
        for j, i in enumerate(active):
            out[:, i] = _marginal_ppf(U[:, j], mean[i], cv, dist)
        return out
    for i in active:
        mu = mean[i]; sd = cv * mu
        if dist == "normal":
            out[:, i] = np.clip(rng.normal(mu, sd, N), 0, None)
        elif dist == "uniform":
            half = math.sqrt(3.0) * sd
            out[:, i] = np.clip(rng.uniform(mu - half, mu + half, N), 0, None)
        elif dist == "left":
            k = 1.0 / (cv * cv)
            g = rng.gamma(k, mu / k, N)
            out[:, i] = np.clip(2 * mu - g, 0, None)
        elif dist == "lognormal":
            s2 = math.log(1.0 + cv * cv)
            out[:, i] = rng.lognormal(math.log(mu) - 0.5 * s2, math.sqrt(s2), N)
        elif dist == "studentt":
            nu = 4.0
            s = cv * mu * math.sqrt((nu - 2.0) / nu)
            out[:, i] = np.clip(mu + s * rng.standard_t(nu, N), 0, None)
        else:
            k = 1.0 / (cv * cv)
            out[:, i] = rng.gamma(k, mu / k, N)
    return out


def make_scenarios(dbar, pbar, N, cv, dist, seed):
    """Gate scenarios: delivery and pickup drawn independently from one rng stream (each with the
    inter-node copula). dist='gamma' = the SAA/W-DRO empirical Phat."""
    rng = np.random.default_rng(seed)
    return (sample_demands(dbar, len(dbar), N, cv, dist, rng),
            sample_demands(pbar, len(pbar), N, cv, dist, rng))


# ============================================================ peak / cost / CVaR
def route_cost(route, D):
    """Closed-tour distance depot->...->depot (D integer for Dethloff)."""
    if not route:
        return 0.0
    c = D[0, route[0]] + D[route[-1], 0]
    for i in range(len(route) - 1):
        c += D[route[i], route[i + 1]]
    return float(c)


def route_peaks(route, dsc, psc):
    """Model-A peak load of `route` for every scenario. dsc,psc: (N, n). Returns (N,)."""
    if not route:
        return np.zeros(dsc.shape[0])
    d = dsc[:, route]; p = psc[:, route]
    total_d = d.sum(1)
    Lmid = total_d[:, None] - np.cumsum(d, 1) + np.cumsum(p, 1)   # L_k, k=1..m
    return np.maximum(total_d, Lmid.max(1))                       # include L_0 = total_d


def cvar(samples, alpha):
    """Empirical CVaR_alpha = mean of the worst (1-alpha) tail."""
    s = np.sort(samples)
    k = int(math.ceil(alpha * len(s)))
    tail = s[k:] if k < len(s) else s[-1:]
    return float(tail.mean())


def nominal_peak(route, dbar, pbar):
    if not route:
        return 0.0
    d = np.array([dbar[c] for c in route]); p = np.array([pbar[c] for c in route])
    total_d = d.sum()
    L = [total_d]
    for k in range(len(route)):
        L.append(total_d - d[:k + 1].sum() + p[:k + 1].sum())
    return float(max(L))


# ============================================================ gates
class DetGate:
    """Deterministic: nominal (mean-demand) peak <= Q."""
    mode = "det"
    def __init__(self, cap, dbar, pbar):
        self.cap, self.dbar, self.pbar = cap, dbar, pbar
        self.calls = self.pruned = 0
    def feasible(self, route):
        if not route:
            return True
        return nominal_peak(route, self.dbar, self.pbar) <= self.cap + 1e-9


class TwoPhaseGate:
    """SAA / W-DRO: Phase-1 O(route_len) rho-surrogate LOWER-BOUND prune (reject-only),
    Phase-2 exact empirical CVaR certificate (the only acceptance gate)."""
    mode = "cvar"
    def __init__(self, cap, alpha, dbar, pbar, sig_d, sig_p, z, rho, dsc, psc, prune=True):
        self.cap, self.alpha = cap, alpha
        self.dbar, self.pbar = dbar, pbar
        self.sig_d, self.sig_p = sig_d, sig_p        # per-node std = cv*mean
        self.z, self.rho = z, rho                    # CVaR multiplier + equicorrelation
        self.dsc, self.psc = dsc, psc                # gate scenarios (N_DATA, n)
        self.prune = prune                           # False -> pure exact CVaR (audit mode)
        self.calls = self.pruned = 0                 # diagnostics: Phase-1 prune rate

    def rho_route(self, route):
        """max_k [ M_k + z*sqrt(Var_k) ] : nominal Model-A load + z * equicorrelated std, per
        position. Var_k = (1-rho)*sum(sig^2) + rho*(sum sig)^2  (exact equicorr variance of the
        on-board components: rho=0 -> independent, rho=1 -> comonotone)."""
        d = self.dbar[route]; p = self.pbar[route]
        sd = self.sig_d[route]; sp = self.sig_p[route]
        total_d = d.sum()
        M = np.concatenate(([total_d], total_d - np.cumsum(d) + np.cumsum(p)))        # mean profile
        v2 = (sd ** 2).sum()
        Vind = np.concatenate(([v2], v2 - np.cumsum(sd ** 2) + np.cumsum(sp ** 2)))   # sum sig^2 onboard
        s1 = sd.sum()
        S = np.concatenate(([s1], s1 - np.cumsum(sd) + np.cumsum(sp)))                # sum sig  onboard
        Vcorr = (1.0 - self.rho) * Vind + self.rho * S ** 2
        return float(np.max(M + self.z * np.sqrt(np.clip(Vcorr, 0.0, None))))

    def feasible(self, route):
        if not route:
            return True
        self.calls += 1
        if self.prune and self.rho_route(route) > self.cap + 1e-9:   # Phase 1: lower-bound prune
            self.pruned += 1
            return False
        return cvar(route_peaks(route, self.dsc, self.psc),          # Phase 2: exact certificate
                    self.alpha) <= self.cap + 1e-9

class InflationGate:
    """Simple static-inflation gate (didactic; used by figure/validation
    scripts): nominal Model-A peak with every demand inflated by (1+alpha)
    must fit the capacity. The full quadrant-budget robust gate is
    GounarisGate below."""
    mode = "inflate"

    def __init__(self, cap, dbar, pbar, alpha=0.2):
        self.cap = cap
        self.dbar = dbar * (1.0 + alpha)
        self.pbar = pbar * (1.0 + alpha)
        self.calls = self.pruned = 0

    def feasible(self, route):
        if not route:
            return True
        return nominal_peak(route, self.dbar, self.pbar) <= self.cap + 1e-9


class CantelliGate:
    """M-DRO Baseline (Cantelli-Chebyshev). 
    Enforces P(load > Q) <= 1-alpha using only mean and variance."""
    mode = "cantelli"
    def __init__(self, cap, alpha, dbar, pbar, sig_d, sig_p, rho):
        self.cap = cap
        self.alpha = alpha
        self.dbar, self.pbar = dbar, pbar
        self.sig_d, self.sig_p = sig_d, sig_p
        self.rho = rho
        self.multiplier = math.sqrt(alpha / (1.0 - alpha))
        self.calls = self.pruned = 0

    def feasible(self, route):
        if not route:
            return True
        self.calls += 1
        d = self.dbar[route]; p = self.pbar[route]
        sd = self.sig_d[route]; sp = self.sig_p[route]
        
        total_d = d.sum()
        M = np.concatenate(([total_d], total_d - np.cumsum(d) + np.cumsum(p)))
        
        v2 = (sd ** 2).sum()
        Vind = np.concatenate(([v2], v2 - np.cumsum(sd ** 2) + np.cumsum(sp ** 2)))
        s1 = sd.sum()
        S = np.concatenate(([s1], s1 - np.cumsum(sd) + np.cumsum(sp)))
        Vcorr = (1.0 - self.rho) * Vind + self.rho * S ** 2
        
        peak_cantelli = float(np.max(M + self.multiplier * np.sqrt(np.clip(Vcorr, 0.0, None))))
        return peak_cantelli <= self.cap + 1e-9
class CuiGate:
    """
    Cui et al. (2025) / Bertsimas & Sim (2004) Budgeted Uncertainty.
    Core Idea: At most 'Gamma' fraction of nodes in a route can deviate
    up to their maximum deviation bound (hat_d).
    """
    mode = "cui_budget"
    
    def __init__(self, cap, dbar, pbar, alpha=0.1, gamma=0.5):
        self.cap = cap
        self.alpha = alpha  # Max fractional deviation
        self.gamma = gamma  # Budget fraction (0.0 to 1.0)
        self.dbar = dbar
        self.pbar = pbar
        
        # Max deviation bound for each node: hat_d_i = alpha * (P_i + D_i)
        # This matches the logic from the Cui script
        self.hat_d = alpha * (pbar + dbar)
        
        self.calls = 0
        self.pruned = 0

    def feasible(self, route):
        if not route:
            return True
        self.calls += 1
        
        m = len(route)
        d = self.dbar[route]
        p = self.pbar[route]
        h = self.hat_d[route]
        
        # 1. Tính Tải trọng danh nghĩa (Nominal Load) tại mọi trạm
        total_d = d.sum()
        nominal_M = np.concatenate(([total_d], total_d - np.cumsum(d) + np.cumsum(p)))
        
        # Nếu chỉ riêng Nominal Load đã vượt sức chứa -> Vứt luôn
        if np.any(nominal_M > self.cap):
            return False
            
        # 2. Xử lý Budgeted Worst-case cho từng trạm
        # Mặc dù vòng lặp for trong python chậm hơn numpy, nhưng vì số trạm m nhỏ (thường < 20)
        # nên việc tính Worst-case tại từng điểm dừng (như script của Cui) là bắt buộc.
        
        for k in range(m):
            # Lấy các deviation của các trạm đã đi qua từ đầu đến trạm k
            devs = h[:k+1]
            
            # Sắp xếp giảm dần để lấy những độ lệch tồi tệ nhất
            sorted_devs = np.sort(devs)[::-1]
            
            # Tính Budget: Gamma * số trạm đã đi qua
            budget_float = self.gamma * (k + 1)
            budget_floor = int(math.floor(budget_float))
            budget_frac = budget_float - budget_floor
            
            # Cộng dồn các deviation lớn nhất nằm trong Budget
            worst_extra = np.sum(sorted_devs[:budget_floor])
            if budget_frac > 0 and budget_floor < len(sorted_devs):
                worst_extra += budget_frac * sorted_devs[budget_floor]
            
            # Tính Worst-case Load tại trạm k
            # nominal_M có size m+1 (vị trí 0 là depot), nên trạm k tương ứng với nominal_M[k+1]
            worst_load = nominal_M[k+1] + worst_extra
            
            if worst_load > self.cap + 1e-9:
                return False  # Chỉ cần 1 trạm vỡ tải Worst-case là cấm cửa route này
                
        # Sống sót qua mọi kịch bản Worst-case
        return True
class GounarisGate:
    """
    Gounaris (2013) Robust CVRP - Static Demand Inflation (QB Support).
    Core Idea: Inflate demand by (1 + alpha * beta) before checking capacity.
    """
    mode = "gounaris_qb"
    
    def __init__(self, cap, dbar, pbar, alpha=0.1, beta=0.5):
        self.cap = cap
        self.alpha = alpha
        self.beta = beta
        
        # Mô phỏng chính xác logic inflate_demands_QB từ Gounaris 2013
        factor = alpha if beta >= 1.0 else alpha * beta
        
        # Bơm phồng toàn bộ array nhu cầu một lần duy nhất (Vectorized)
        self.d_inf = dbar * (1.0 + factor)
        self.p_inf = pbar * (1.0 + factor)
        self.calls = 0
        self.pruned = 0 # Thêm biến này để tránh lỗi nếu code sếp có gọi tới

    def feasible(self, route):
        if not route:
            return True
        self.calls += 1
        
        # Lấy nhu cầu ĐÃ BỊ BƠM PHỒNG của các trạm trong tuyến
        d = self.d_inf[route]
        p = self.p_inf[route]
        
        # Tính toán Peak Load vật lý như bình thường
        total_d = d.sum()
        M = np.concatenate(([total_d], total_d - np.cumsum(d) + np.cumsum(p)))
        peak_load = np.max(M)
        
        return float(peak_load) <= self.cap + 1e-9
# ============================================================ ALNS (verbatim core + early stop)
def cw_init(D, gate, n):
    routes = [[c] for c in range(1, n) if gate.feasible([c])]
    placed = {c for r in routes for c in r}
    leftovers = [c for c in range(1, n) if c not in placed]
    sav = []
    for a in range(1, n):
        for b in range(a + 1, n):
            sav.append((D[0, a] + D[0, b] - D[a, b], a, b))
    sav.sort(reverse=True)
    rt = {i: r for i, r in enumerate(routes)}
    where = {r[0]: i for i, r in enumerate(routes)}
    for s, a, b in sav:
        if s <= 0:
            break
        ra, rb = where.get(a), where.get(b)
        if ra is None or rb is None or ra == rb:
            continue
        Ra, Rb = rt[ra], rt[rb]
        if Ra[-1] == a and Rb[0] == b:
            merged = Ra + Rb
        elif Ra[0] == a and Rb[-1] == b:
            merged = Rb + Ra
        elif Ra[-1] == a and Rb[-1] == b:
            merged = Ra + Rb[::-1]
        elif Ra[0] == a and Rb[0] == b:
            merged = Ra[::-1] + Rb
        else:
            continue
        if not gate.feasible(merged):
            continue
        rt[ra] = merged
        for c in Rb:
            where[c] = ra
        del rt[rb]
    sol = [r for r in rt.values()]
    for c in leftovers:
        sol.append([c])
    return sol


def two_opt_gate(route, D, gate, max_passes=200):
    """First-improvement 2-opt. The gain test assumes symmetric D; the pass
    cap is a safety net against cycling if D is (accidentally) asymmetric."""
    if len(route) < 4:
        return route
    r = route[:]
    improved = True
    passes = 0
    while improved and passes < max_passes:
        passes += 1
        improved = False
        for i in range(len(r) - 1):
            for k in range(i + 1, len(r)):
                a = r[i - 1] if i > 0 else 0
                b = r[i]; c = r[k]
                d = r[k + 1] if k + 1 < len(r) else 0
                if a == c or b == d:
                    continue
                if D[a, c] + D[b, d] - D[a, b] - D[c, d] < -1e-9:
                    cand = r[:i] + r[i:k + 1][::-1] + r[k + 1:]
                    if gate.feasible(cand):
                        r = cand; improved = True
                        break
            if improved:
                break
    return r


def relocate_gate(sol, D, gate):
    improved = True
    while improved:
        improved = False
        for ri in range(len(sol)):
            R = sol[ri]
            for pi in range(len(R)):
                c = R[pi]
                a = R[pi - 1] if pi > 0 else 0
                b = R[pi + 1] if pi + 1 < len(R) else 0
                gain = D[a, c] + D[c, b] - D[a, b]
                for rj in range(len(sol)):
                    if rj == ri:
                        continue
                    Rj = sol[rj]
                    for q in range(len(Rj) + 1):
                        u = Rj[q - 1] if q > 0 else 0
                        v = Rj[q] if q < len(Rj) else 0
                        if D[u, c] + D[c, v] - D[u, v] - gain < -1e-9:
                            newR = R[:pi] + R[pi + 1:]
                            newRj = Rj[:q] + [c] + Rj[q:]
                            if gate.feasible(newR) and gate.feasible(newRj):
                                sol[ri] = newR; sol[rj] = newRj
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        sol = [r for r in sol if r]
    return sol


def greedy_insert(sol, c, D, gate):
    best = (None, None, math.inf)
    for ri, R in enumerate(sol):
        for p in range(len(R) + 1):
            cand = R[:p] + [c] + R[p:]
            if not gate.feasible(cand):
                continue
            u = R[p - 1] if p > 0 else 0
            v = R[p] if p < len(R) else 0
            delta = D[u, c] + D[c, v] - D[u, v]
            if delta < best[2]:
                best = (ri, p, delta)
    return best


def ruin_recreate(sol, D, gate, rng, q_frac=0.2):
    sol = [r[:] for r in sol]
    custs = [c for r in sol for c in r]
    q = max(1, int(q_frac * len(custs)))
    removed = rng.sample(custs, min(q, len(custs)))
    rem = set(removed)
    sol = [[c for c in r if c not in rem] for r in sol]
    sol = [r for r in sol if r]
    rng.shuffle(removed)
    for c in removed:
        ri, p, _ = greedy_insert(sol, c, D, gate)
        if ri is None:
            sol.append([c])
        else:
            sol[ri].insert(p, c)
    return sol


def local_search(sol, D, gate):
    sol = [two_opt_gate(r, D, gate) for r in sol if r]
    sol = relocate_gate(sol, D, gate)
    return [r for r in sol if r]


def econ_cost(sol, D, omega_V):
    return sum(route_cost(r, D) for r in sol) + omega_V * sum(1 for r in sol if r)


def solve_fast(D, gate, n):
    """Large-instance planning: Clarke-Wright + per-route 2-opt only.

    The pure-Python relocate/ILS machinery of solve() runs its initial
    local search to convergence BEFORE the time limit applies, which is
    impractical beyond ~150 customers. CW + 2-opt builds a reasonable plan
    in seconds; every execution policy is evaluated on the SAME plan, so
    the policy comparison is unaffected by planner optimality."""
    sol = [two_opt_gate(r, D, gate) for r in cw_init(D, gate, n) if r]
    return [r for r in sol if r]


def solve(D, gate, n, omega_V, time_limit, seed, no_improve=NO_IMPROVE):
    """Penalised-distance ILS (distance + omega_V*K). Early stop: break if no improving move for
    `no_improve` seconds, or when `time_limit` is hit."""
    rng = random.Random(seed)
    cur = local_search(cw_init(D, gate, n), D, gate)
    best, best_c = [r[:] for r in cur], econ_cost(cur, D, omega_V)
    t0 = time.time(); t_imp = t0
    while time.time() - t0 < time_limit:
        cand = local_search(
            ruin_recreate(best, D, gate, rng, rng.choice([0.1, 0.15, 0.2, 0.3])), D, gate)
        c = econ_cost(cand, D, omega_V)
        if c < best_c - 1e-9:
            best, best_c = [r[:] for r in cand], c
            t_imp = time.time()
        elif time.time() - t_imp > no_improve:
            break
    return [r for r in best if r]


# ============================================================ out-of-sample E[V_extra]
def eval_evextra(plan, dbar, pbar, n, Q, mixture=None, heavy_family="lognormal", seed=SEED):
    """Out-of-sample E_w[V_extra] = sum_r sum_s w_s * Vbar^s_r   (eq. 12), with
    Vbar^s_r = E[ ceil( max(0, peak_r - Q) / Q ) ]   (eq. 11) under stress shape s.
    `mixture` = {shape: weight} (defaults to SHAPE_W); the 'heavy' slot is drawn from `heavy_family`
    ('lognormal' or 'studentt'). Same global RHO copula. Returns (worst_route_fail, E_w[V_extra])."""
    if mixture is None:
        mixture = SHAPE_W
    fagg = [0.0] * len(plan); vagg = [0.0] * len(plan)
    for si, (shape, w) in enumerate(mixture.items()):
        if w <= 0:
            continue
        dist = heavy_family if shape == "heavy" else shape    # 'heavy' slot -> chosen tail family
        rng = np.random.default_rng(seed + 7919 * (si + 1))
        dsc = sample_demands(dbar, n, N_MC, CV, dist, rng)
        psc = sample_demands(pbar, n, N_MC, CV, dist, rng)
        for ri, r in enumerate(plan):
            pk = route_peaks(r, dsc, psc)
            fagg[ri] += w * float((pk > Q).mean())
            vagg[ri] += w * float(np.ceil(np.clip(pk - Q, 0, None) / Q).mean())
    return (max(fagg) if fagg else 0.0), float(sum(vagg))


# ============================================================ one instance
def solve_instance(path, tlim, no_improve, use_prune=True, which=None):
    """Solve the requested policies ONCE (default: all). Returns plans +
    per-policy K/dist/time/prune plus the data needed to (re-)price
    economics under any stress mixture (omega_V is mixture-independent).
    `which`: optional list of gate names to solve, e.g. ["Det"] — the
    scenario-gated SAA/WDRO gates are O(N_DATA) per feasibility call and
    impractical inside O(n^2) local search at n >= 200."""
    D, dem, Q, n, scale = parse_dethloff(path)
    dbar = dem[:, 0].astype(float).copy()            # mean delivery (depot idx 0 = 0)
    pbar = dem[:, 1].astype(float).copy()            # native pickup
    sig_d = CV * dbar; sig_p = CV * pbar
    Qeff = Q * (1 - EPS_FRAC)
    omega_V_solve = float(np.mean(D[D > 0]))         # per-vehicle search penalty (one mean edge)

    dsc, psc = make_scenarios(dbar, pbar, N_DATA, CV, DIST, SEED)    # empirical Phat (the gate)
    gates = {
        "Det":      DetGate(Q, dbar, pbar),
        "Gounaris": GounarisGate(Q, dbar, pbar, alpha=GOUNARIS_ALPHA, beta=GOUNARIS_BETA),
        "Cui":      CuiGate(Q, dbar, pbar, alpha=CUI_ALPHA, gamma=CUI_GAMMA),
        "MDRO":     CantelliGate(Q, ALPHA, dbar, pbar, sig_d, sig_p, RHO), # Phương sai
        "SAA":      TwoPhaseGate(Q, ALPHA, dbar, pbar, sig_d, sig_p, Z_CVAR, RHO, dsc, psc, use_prune), # Lấy mẫu
        "WDRO":     TwoPhaseGate(Qeff, ALPHA, dbar, pbar, sig_d, sig_p, Z_CVAR, RHO, dsc, psc, use_prune), # Trùm cuối
    }
    if which:
        gates = {k: v for k, v in gates.items() if k in which}
    res = {}
    for name, gate in gates.items():
        t0 = time.time()
        if n > 90:                      # ILS impractical: CW + 2-opt (see solve_fast)
            plan = solve_fast(D, gate, n)
        else:
            plan = solve(D, gate, n, omega_V_solve, tlim, SEED, no_improve)
        elapsed = time.time() - t0
        K = sum(1 for r in plan if r)
        dist = sum(route_cost(r, D) for r in plan) / scale          # real units
        prune = (gate.pruned / gate.calls) if getattr(gate, "calls", 0) else 0.0
        res[name] = {"plan": plan, "K": K, "dist": dist, "time": elapsed, "prune": prune}
    maxdist = max(res[k]["dist"] for k in res)
    maxK    = max(res[k]["K"] for k in res)
    omega_V = maxdist / max(1, maxK)                 # one route's distance (mixture-independent)
    return {"name": Path(path).stem, "dbar": dbar, "pbar": pbar, "n": n, "Q": Q,
            "omega_V": omega_V, "res": res}


def price_instance(sol, mixture=None, heavy_family="lognormal"):
    """Price TBC@OMEGA_RATIO for already-solved plans under a given stress mixture.
    Returns (per-policy dict with evx+tbc, winner)."""
    dbar, pbar, n, Q = sol["dbar"], sol["pbar"], sol["n"], sol["Q"]
    omega_V = sol["omega_V"]; omega_F = OMEGA_RATIO * omega_V
    out = {}
    for k, r in sol["res"].items():
        _, evx = eval_evextra(r["plan"], dbar, pbar, n, Q, mixture, heavy_family)
        out[k] = {"K": r["K"], "dist": r["dist"], "time": r["time"], "prune": r["prune"],
                  "evx": evx, "tbc": r["dist"] + omega_V * r["K"] + omega_F * evx}
    winner = min(("Det", "SAA", "WDRO"), key=lambda k: out[k]["tbc"])
    return out, winner


def run_one(path, tlim, no_improve, use_prune=True):
    sol = solve_instance(path, tlim, no_improve, use_prune)
    out, winner = price_instance(sol, SHAPE_W, "lognormal")          # headline mixture
    row = {"Instance": sol["name"]}
    for k in ("Det", "SAA", "WDRO"):
        row[f"K_{k}"]       = out[k]["K"]
        row[f"Dist_{k}"]    = round(out[k]["dist"], 2)
        row[f"EVextra_{k}"] = round(out[k]["evx"], 4)
        row[f"Time_{k}"]    = round(out[k]["time"], 1)
        row[f"TBC50_{k}"]   = round(out[k]["tbc"], 1)
    row["omega_V"] = round(sol["omega_V"], 2)
    row["Winner"]  = winner
    print(f"  {sol['name']:<10}  "
          f"K(det/saa/wdro)={out['Det']['K']}/{out['SAA']['K']}/{out['WDRO']['K']}  "
          f"TBC50={out['Det']['tbc']:.0f}/{out['SAA']['tbc']:.0f}/{out['WDRO']['tbc']:.0f}  "
          f"win={winner}  "
          f"t={out['Det']['time']:.0f}/{out['SAA']['time']:.0f}/{out['WDRO']['time']:.0f}s  "
          f"prune(saa/wdro)={out['SAA']['prune']*100:.0f}%/{out['WDRO']['prune']*100:.0f}%")
    return row


def run_sweep(files, tlim, no_improve, use_prune):
    """Solve every instance ONCE, then re-price under each mixture in MIXTURES (plans are independent
    of the test mixture; only the economics change). Reports WDRO win-rate + mean TBC per config."""
    print("\n--- SOLVING (once per instance; plans reused across all mixtures) ---")
    sols, t0 = [], time.time()
    for i, f in enumerate(files, 1):
        print(f"[solve {i}/{len(files)}] {Path(f).stem:<10}", end="  ")
        try:
            s = solve_instance(f, tlim, no_improve, use_prune)
            sols.append(s)
            print(f"K={s['res']['Det']['K']}/{s['res']['SAA']['K']}/{s['res']['WDRO']['K']}")
        except Exception as e:
            print(f"ERROR: {e}")
    print(f"   solved {len(sols)} instances in {(time.time() - t0) / 60:.1f} min")
    if not sols:
        print("No instances solved."); return

    print(f"\n=== MIXTURE SENSITIVITY  (same {len(sols)} plans re-stressed under {len(MIXTURES)} mixtures) ===")
    print(f"   {'config':<30}{'WDRO win':>10}{'meanTBC SAA':>13}{'meanTBC WDRO':>14}{'winner':>9}")
    summary = []
    for label, mix, hf in MIXTURES:
        wins = {"Det": 0, "SAA": 0, "WDRO": 0}
        tsum = {"Det": 0.0, "SAA": 0.0, "WDRO": 0.0}
        esum = {"SAA": 0.0, "WDRO": 0.0}
        for s in sols:
            out, w = price_instance(s, mix, hf)
            wins[w] += 1
            for k in tsum: tsum[k] += out[k]["tbc"]
            for k in esum: esum[k] += out[k]["evx"]
        nN = len(sols)
        mwin = min(("Det", "SAA", "WDRO"), key=lambda k: tsum[k])
        summary.append({"config": label, "heavy": hf, "WDRO_wins": f"{wins['WDRO']}/{nN}",
                        "TBC_Det": round(tsum["Det"] / nN, 1), "TBC_SAA": round(tsum["SAA"] / nN, 1),
                        "TBC_WDRO": round(tsum["WDRO"] / nN, 1),
                        "EVx_SAA": round(esum["SAA"] / nN, 4), "EVx_WDRO": round(esum["WDRO"] / nN, 4),
                        "mean_winner": mwin})
        print(f"   {label:<30}{wins['WDRO']:>7}/{nN:<2}{tsum['SAA'] / nN:>13,.0f}{tsum['WDRO'] / nN:>14,.0f}{mwin:>9}")

    df = pd.DataFrame(summary)
    try:
        df.to_excel("results_mixture_sensitivity.xlsx", index=False)
        print("\n   wrote results_mixture_sensitivity.xlsx")
    except Exception as e:
        df.to_csv("results_mixture_sensitivity.csv", index=False)
        print(f"\n   (xlsx unavailable: {e}) wrote results_mixture_sensitivity.csv")


def main():
    argv = sys.argv[1:]
    data_dir, tlim, no_improve, max_n, use_prune, sweep = DATA_DIR, TLIM, NO_IMPROVE, None, True, False
    for a in argv:
        if   a.startswith("dir="):   data_dir = a[4:]
        elif a.startswith("t="):     tlim = float(a[2:])
        elif a.startswith("noimp="): no_improve = float(a[6:])
        elif a.startswith("max="):   max_n = int(a[4:])
        elif a == "noprune":         use_prune = False     # audit mode: pure exact CVaR, no Phase-1
        elif a == "sweep":           sweep = True           # mixture-sensitivity mode (reuses solves)

    files = sorted(glob.glob(str(Path(data_dir) / "*.vrpspd")))
    if not files:
        # be forgiving about the extension / location
        files = sorted(glob.glob(str(Path(data_dir) / "*.txt"))) or \
                sorted(glob.glob(str(Path(data_dir) / "*")))
        files = [f for f in files if Path(f).is_file()]
    if max_n:
        files = files[:max_n]
    if not files:
        print(f"ERROR: no instances found in '{data_dir}/'. "
              f"Put the Dethloff .vrpspd files there or pass dir=<folder>.")
        return

    print("=" * 92)
    print(" DETHLOFF BATCH RUNNER  --  SVRPSPD W-DRO economics sweep")
    print("=" * 92)
    print(f"   instances={len(files)} in '{data_dir}/'   policies=Det/SAA/WDRO   "
          f"operating point omega_F/omega_V={OMEGA_RATIO:g}")
    print(f"   alpha={ALPHA}  cv={CV}  rho={RHO}  eps_frac={EPS_FRAC}  "
          f"N_data={N_DATA}  N_mc={N_MC:,}  seed={SEED}")
    print(f"   per-policy tlim={tlim:g}s  early-stop no-improve={no_improve:g}s  "
          f"Phase-1 z_cvar={Z_CVAR:.4f}")
    print("   gate = Two-Phase (rho-prune -> exact empirical CVaR certificate)" if use_prune
          else "   gate = EXACT empirical CVaR only  (Phase-1 prune OFF -- AUDIT MODE)")
    print("-" * 92)

    if sweep:
        run_sweep(files, tlim, no_improve, use_prune)
        return

    rows = []
    t_start = time.time()
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}]", end=" ")
        try:
            rows.append(run_one(f, tlim, no_improve, use_prune))
        except Exception as e:
            print(f"  {Path(f).stem:<10}  ERROR: {e}")
    print("-" * 92)
    print(f"   total wall time = {(time.time() - t_start)/60:.1f} min")

    if not rows:
        print("No successful instances; nothing to write."); return

    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c not in ("Instance", "Winner")]
    avg = {c: round(float(df[c].mean()), 2) for c in num_cols}
    avg["Instance"] = "AVERAGE"
    avg["Winner"] = min(("Det", "SAA", "WDRO"), key=lambda k: avg[f"TBC50_{k}"])
    df_out = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

    print("\n=== MEAN TBC @ omega_F/omega_V=50 (the headline) ===")
    for k in ("Det", "SAA", "WDRO"):
        print(f"   {k:<5}: mean TBC50 = {avg[f'TBC50_{k}']:>12,.1f}   "
              f"mean K = {avg[f'K_{k}']:.2f}   mean E[V_extra] = {avg[f'EVextra_{k}']:.4f}")
    print(f"   -> lowest mean TBC50: {avg['Winner']}   "
          f"(W-DRO wins on {sum(1 for r in rows if r['Winner']=='WDRO')}/{len(rows)} instances)")

    out = "results_dethloff_summary.xlsx"
    try:
        df_out.to_excel(out, index=False)
        print(f"\n   wrote {out}  ({len(rows)} instances + AVERAGE row)")
    except Exception as e:
        out_csv = "results_dethloff_summary.csv"
        df_out.to_csv(out_csv, index=False)
        print(f"\n   (xlsx engine unavailable: {e})\n   wrote {out_csv} instead "
              f"-- `pip install openpyxl` for the .xlsx")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Dethloff SVRPSPD planning runner with a genuine classical ALNS search layer.

Changes relative to the archived runner:
  * random + worst removal; greedy + regret-2 repair;
  * adaptive roulette-wheel operator weights and simulated annealing;
  * current and global-best solutions are distinct;
  * the entire Dethloff data set is divided by 10,000 at parse time;
  * Cantelli, Gounaris, Cui, SAA, and WDRO are all included in reporting.

The risk-gate formulas themselves are intentionally unchanged in this revision.
"""

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


# Classical ALNS search settings.  These match the user's existing ALNS implementation.
ALNS_MAX_ITERATIONS = 5000
ALNS_DESTROY_MIN_FRAC = 0.10
ALNS_DESTROY_MAX_FRAC = 0.40
ALNS_SA_TEMP_INIT = 100.0
ALNS_SA_COOLING = 0.9997
ALNS_SEGMENT_SIZE = 10
ALNS_REACTION_FACTOR = 0.80
ALNS_SIGMA_1 = 33.0          # new global best
ALNS_SIGMA_2 = 9.0           # improves current solution
ALNS_SIGMA_3 = 13.0          # accepted non-improving solution
ALNS_WORST_DETERMINISM = 3.0
ALNS_WEIGHT_FLOOR = 0.01

POLICY_ORDER = ("Det", "Gounaris", "Cui", "Cantelli", "SAA", "WDRO")

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
    # The distributed Dethloff files store distance, demands, and capacity
    # multiplied by 10,000.  Normalize every quantity here so all downstream
    # calculations, SA temperatures, and reported objectives use real units.
    data_scale = 10000.0
    D = np.array(vals, dtype=float).reshape(n, n) / data_scale
    dem = _parse_pd(txt, n) / data_scale
    Q = float(Q) / data_scale
    return D, dem, Q, n, 1.0


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
# ============================================================ Classical ALNS search layer
# The original runner used one random ruin operator, one greedy repair operator,
# and strict-improvement acceptance.  The implementation below is a genuine
# classical ALNS: two destroy operators, two repair operators, adaptive roulette
# weights, simulated-annealing acceptance, and separate current/best solutions.

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
        if not gate.feasible([c]):
            raise ValueError(f"customer {c} is infeasible as a singleton route")
        sol.append([c])
    return sol


def two_opt_gate(route, D, gate, max_passes=200):
    """First-improvement 2-opt under the active stochastic feasibility gate."""
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
                b = r[i]
                c = r[k]
                d = r[k + 1] if k + 1 < len(r) else 0
                if a == c or b == d:
                    continue
                if D[a, c] + D[b, d] - D[a, b] - D[c, d] < -1e-9:
                    cand = r[:i] + r[i:k + 1][::-1] + r[k + 1:]
                    if gate.feasible(cand):
                        r = cand
                        improved = True
                        break
            if improved:
                break
    return r


def relocate_gate(sol, D, gate):
    sol = [r[:] for r in sol if r]
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
                                sol[ri] = newR
                                sol[rj] = newRj
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


def local_search(sol, D, gate):
    sol = [two_opt_gate(r, D, gate) for r in sol if r]
    sol = relocate_gate(sol, D, gate)
    return [r for r in sol if r]


def econ_cost(sol, D, omega_V):
    return sum(route_cost(r, D) for r in sol) + omega_V * sum(1 for r in sol if r)


def greedy_insert(sol, c, D, gate, omega_V=0.0):
    """Backward-compatible single-customer greedy insertion helper."""
    options = _insertion_options(sol, c, D, gate, omega_V)
    if not options:
        return None, None, math.inf
    delta, ri, p = min(options, key=lambda x: (x[0], x[1], x[2]))
    return ri, p, delta


def _removal_count(total_customers, rng, min_frac, max_frac):
    if total_customers <= 0:
        return 0
    frac = rng.uniform(min_frac, max_frac)
    return min(total_customers, max(1, int(round(frac * total_customers))))


def random_removal(sol, D, omega_V, rng, min_frac, max_frac):
    del D, omega_V
    work = [r[:] for r in sol if r]
    customers = [c for r in work for c in r]
    q = _removal_count(len(customers), rng, min_frac, max_frac)
    removed = rng.sample(customers, q) if q else []
    removed_set = set(removed)
    partial = [[c for c in r if c not in removed_set] for r in work]
    return [r for r in partial if r], removed


def worst_removal(sol, D, omega_V, rng, min_frac, max_frac,
                  determinism=ALNS_WORST_DETERMINISM):
    """Randomized worst removal based on marginal economic routing cost."""
    work = [r[:] for r in sol if r]
    q = _removal_count(sum(len(r) for r in work), rng, min_frac, max_frac)
    removed = []
    for _ in range(q):
        ranked = []
        for ri, route in enumerate(work):
            old = route_cost(route, D) + omega_V
            for pi, customer in enumerate(route):
                trial = route[:pi] + route[pi + 1:]
                new = route_cost(trial, D) + omega_V if trial else 0.0
                ranked.append((old - new, ri, pi, customer))
        if not ranked:
            break
        ranked.sort(key=lambda x: (-x[0], x[3]))
        idx = min(int(len(ranked) * (rng.random() ** determinism)), len(ranked) - 1)
        _, ri, pi, customer = ranked[idx]
        removed.append(customer)
        del work[ri][pi]
        if not work[ri]:
            del work[ri]
    return work, removed


def _insertion_options(sol, customer, D, gate, omega_V):
    options = []
    for ri, route in enumerate(sol):
        before = route_cost(route, D)
        for p in range(len(route) + 1):
            cand = route[:p] + [customer] + route[p:]
            if gate.feasible(cand):
                options.append((route_cost(cand, D) - before, ri, p))
    if gate.feasible([customer]):
        options.append((route_cost([customer], D) + omega_V, -1, 0))
    return options


def greedy_repair(partial, removed, D, gate, omega_V, rng):
    sol = [r[:] for r in partial if r]
    pool = list(removed)
    rng.shuffle(pool)
    for customer in pool:
        options = _insertion_options(sol, customer, D, gate, omega_V)
        if not options:
            raise ValueError(f"no feasible insertion for customer {customer}")
        _, ri, p = min(options, key=lambda x: (x[0], x[1], x[2]))
        if ri == -1:
            sol.append([customer])
        else:
            sol[ri].insert(p, customer)
    return sol


def regret2_repair(partial, removed, D, gate, omega_V, rng):
    del rng
    sol = [r[:] for r in partial if r]
    pool = list(removed)
    while pool:
        records = []
        for customer in pool:
            options = _insertion_options(sol, customer, D, gate, omega_V)
            if not options:
                raise ValueError(f"no feasible insertion for customer {customer}")
            options.sort(key=lambda x: (x[0], x[1], x[2]))
            best = options[0]
            second = options[1][0] if len(options) > 1 else best[0]
            records.append((second - best[0], best[0], customer, best[1], best[2]))
        _, _, customer, ri, p = max(records, key=lambda x: (x[0], -x[1], -x[2]))
        if ri == -1:
            sol.append([customer])
        else:
            sol[ri].insert(p, customer)
        pool.remove(customer)
    return sol


def ruin_recreate(sol, D, gate, rng, q_frac=0.2):
    """Legacy compatibility wrapper: random removal followed by greedy repair."""
    partial, removed = random_removal(sol, D, 0.0, rng, q_frac, q_frac)
    return greedy_repair(partial, removed, D, gate, 0.0, rng)


def _roulette_index(weights, rng):
    total = float(sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("operator weights must have a positive finite sum")
    target = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += float(w)
        if target <= acc:
            return i
    return len(weights) - 1


def solve_fast(D, gate, n):
    """Optional explicit fallback: Clarke-Wright plus per-route 2-opt."""
    return [two_opt_gate(r, D, gate) for r in cw_init(D, gate, n) if r]


def solve(D, gate, n, omega_V, time_limit, seed, no_improve=NO_IMPROVE,
          max_iterations=ALNS_MAX_ITERATIONS, segment_size=ALNS_SEGMENT_SIZE,
          return_diagnostics=False):
    """Classical time-limited ALNS for distance + omega_V * fleet size.

    Operators: random/worst removal and greedy/regret-2 repair.  Operators are
    selected by adaptive roulette weights.  Simulated annealing accepts selected
    non-improving moves.  The current and global-best solutions are kept separate.
    """
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if segment_size <= 0:
        raise ValueError("segment_size must be positive")

    rng = random.Random(seed)
    destroy_ops = [random_removal, worst_removal]
    repair_ops = [greedy_repair, regret2_repair]
    d_w = [1.0, 1.0]
    r_w = [1.0, 1.0]
    d_score = [0.0, 0.0]
    r_score = [0.0, 0.0]
    d_count = [0, 0]
    r_count = [0, 0]
    d_use = [0, 0]
    r_use = [0, 0]

    t0 = time.perf_counter()
    current = local_search(cw_init(D, gate, n), D, gate)
    current_cost = econ_cost(current, D, omega_V)
    best = [r[:] for r in current]
    best_cost = current_cost
    last_improve = time.perf_counter()
    temp = ALNS_SA_TEMP_INIT
    accepted = accepted_worse = new_bests = 0
    iterations = 0

    while iterations < max_iterations and time.perf_counter() - t0 < time_limit:
        if no_improve is not None and no_improve > 0 and time.perf_counter() - last_improve > no_improve:
            break
        iterations += 1
        di = _roulette_index(d_w, rng)
        ri = _roulette_index(r_w, rng)
        d_count[di] += 1
        r_count[ri] += 1
        d_use[di] += 1
        r_use[ri] += 1

        partial, removed = destroy_ops[di](
            current, D, omega_V, rng, ALNS_DESTROY_MIN_FRAC, ALNS_DESTROY_MAX_FRAC)
        cand = repair_ops[ri](partial, removed, D, gate, omega_V, rng)
        cand = local_search(cand, D, gate)
        cand = [r for r in cand if r]
        cand_cost = econ_cost(cand, D, omega_V)
        delta = cand_cost - current_cost

        improves_current = delta < -1e-9
        accept = improves_current
        accepted_nonimproving = False
        if not accept:
            prob = math.exp(-max(delta, 0.0) / max(temp, 1e-15))
            if rng.random() < prob:
                accept = True
                accepted_nonimproving = True

        if accept:
            current = [r[:] for r in cand]
            current_cost = cand_cost
            accepted += 1
            if accepted_nonimproving:
                accepted_worse += 1

            if cand_cost < best_cost - 1e-9:
                best = [r[:] for r in cand]
                best_cost = cand_cost
                new_bests += 1
                last_improve = time.perf_counter()
                score = ALNS_SIGMA_1
            elif improves_current:
                score = ALNS_SIGMA_2
            else:
                score = ALNS_SIGMA_3
            d_score[di] += score
            r_score[ri] += score

        if iterations % segment_size == 0:
            for i in range(2):
                if d_count[i] > 0:
                    perf = d_score[i] / d_count[i]
                    d_w[i] = max(ALNS_WEIGHT_FLOOR,
                                 ALNS_REACTION_FACTOR * d_w[i]
                                 + (1.0 - ALNS_REACTION_FACTOR) * perf)
                if r_count[i] > 0:
                    perf = r_score[i] / r_count[i]
                    r_w[i] = max(ALNS_WEIGHT_FLOOR,
                                 ALNS_REACTION_FACTOR * r_w[i]
                                 + (1.0 - ALNS_REACTION_FACTOR) * perf)
            d_score = [0.0, 0.0]
            r_score = [0.0, 0.0]
            d_count = [0, 0]
            r_count = [0, 0]

        temp *= ALNS_SA_COOLING
        if temp < 0.1:
            temp = ALNS_SA_TEMP_INIT * 0.3

    plan = [r for r in best if r]
    diag = {
        "solver": "ALNS",
        "iterations": iterations,
        "accepted": accepted,
        "accepted_worse": accepted_worse,
        "new_bests": new_bests,
        "destroy_random": d_use[0],
        "destroy_worst": d_use[1],
        "repair_greedy": r_use[0],
        "repair_regret2": r_use[1],
        "destroy_weight_random": d_w[0],
        "destroy_weight_worst": d_w[1],
        "repair_weight_greedy": r_w[0],
        "repair_weight_regret2": r_w[1],
        "best_cost": best_cost,
    }
    return (plan, diag) if return_diagnostics else plan


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
def solve_instance(path, tlim, no_improve, use_prune=True, which=None,
                   max_iterations=ALNS_MAX_ITERATIONS,
                   segment_size=ALNS_SEGMENT_SIZE,
                   fast_large=False):
    """Solve the requested planning policies once using the full ALNS layer."""
    D, dem, Q, n, scale = parse_dethloff(path)
    dbar = dem[:, 0].astype(float).copy()
    pbar = dem[:, 1].astype(float).copy()
    sig_d = CV * dbar
    sig_p = CV * pbar
    Qeff = Q * (1 - EPS_FRAC)
    positive = D[D > 0]
    omega_V_solve = float(np.mean(positive)) if positive.size else 1.0

    dsc, psc = make_scenarios(dbar, pbar, N_DATA, CV, DIST, SEED)
    gates = {
        "Det":       DetGate(Q, dbar, pbar),
        "Gounaris":  GounarisGate(Q, dbar, pbar, alpha=GOUNARIS_ALPHA, beta=GOUNARIS_BETA),
        "Cui":       CuiGate(Q, dbar, pbar, alpha=CUI_ALPHA, gamma=CUI_GAMMA),
        "Cantelli":  CantelliGate(Q, ALPHA, dbar, pbar, sig_d, sig_p, RHO),
        "SAA":       TwoPhaseGate(Q, ALPHA, dbar, pbar, sig_d, sig_p, Z_CVAR, RHO,
                                   dsc, psc, use_prune),
        "WDRO":      TwoPhaseGate(Qeff, ALPHA, dbar, pbar, sig_d, sig_p, Z_CVAR, RHO,
                                   dsc, psc, use_prune),
    }
    if which:
        wanted = {"Cantelli" if str(k).upper() == "MDRO" else str(k) for k in which}
        gates = {k: v for k, v in gates.items() if k in wanted}
    if not gates:
        raise ValueError("no requested policies matched " + ",".join(POLICY_ORDER))

    res = {}
    for name, gate in gates.items():
        t0 = time.perf_counter()
        if fast_large and n > 90:
            plan = solve_fast(D, gate, n)
            diag = {
                "solver": "CW+2OPT", "iterations": 0, "accepted": 0,
                "accepted_worse": 0, "new_bests": 0,
                "destroy_random": 0, "destroy_worst": 0,
                "repair_greedy": 0, "repair_regret2": 0,
                "destroy_weight_random": 1.0, "destroy_weight_worst": 1.0,
                "repair_weight_greedy": 1.0, "repair_weight_regret2": 1.0,
                "best_cost": econ_cost(plan, D, omega_V_solve),
            }
        else:
            plan, diag = solve(
                D, gate, n, omega_V_solve, tlim, SEED, no_improve,
                max_iterations=max_iterations, segment_size=segment_size,
                return_diagnostics=True)
        elapsed = time.perf_counter() - t0
        K = sum(1 for r in plan if r)
        dist = sum(route_cost(r, D) for r in plan) / scale
        prune = (gate.pruned / gate.calls) if getattr(gate, "calls", 0) else 0.0
        res[name] = {
            "plan": plan, "K": K, "dist": dist, "time": elapsed, "prune": prune,
            **diag,
        }

    maxdist = max(res[k]["dist"] for k in res)
    maxK = max(res[k]["K"] for k in res)
    omega_V = maxdist / max(1, maxK)
    return {
        "name": Path(path).stem, "dbar": dbar, "pbar": pbar, "n": n, "Q": Q,
        "omega_V": omega_V, "res": res,
    }


def price_instance(sol, mixture=None, heavy_family="lognormal"):
    """Price TBC@OMEGA_RATIO for every plan produced by solve_instance."""
    dbar, pbar, n, Q = sol["dbar"], sol["pbar"], sol["n"], sol["Q"]
    omega_V = sol["omega_V"]
    omega_F = OMEGA_RATIO * omega_V
    out = {}
    for k, r in sol["res"].items():
        worst_fail, evx = eval_evextra(r["plan"], dbar, pbar, n, Q, mixture, heavy_family)
        out[k] = {
            **r,
            "worst_fail": worst_fail,
            "evx": evx,
            "tbc": r["dist"] + omega_V * r["K"] + omega_F * evx,
        }
    policies = [k for k in POLICY_ORDER if k in out]
    winner = min(policies, key=lambda k: out[k]["tbc"])
    return out, winner


def run_one(path, tlim, no_improve, use_prune=True, which=None,
            max_iterations=ALNS_MAX_ITERATIONS,
            segment_size=ALNS_SEGMENT_SIZE,
            fast_large=False):
    sol = solve_instance(path, tlim, no_improve, use_prune, which,
                         max_iterations, segment_size, fast_large)
    out, winner = price_instance(sol, SHAPE_W, "lognormal")
    policies = [k for k in POLICY_ORDER if k in out]
    row = {"Instance": sol["name"]}
    for k in policies:
        row[f"K_{k}"] = out[k]["K"]
        row[f"Dist_{k}"] = round(out[k]["dist"], 4)
        row[f"WorstFail_{k}"] = round(out[k]["worst_fail"], 6)
        row[f"EVextra_{k}"] = round(out[k]["evx"], 6)
        row[f"Time_{k}"] = round(out[k]["time"], 4)
        row[f"TBC50_{k}"] = round(out[k]["tbc"], 4)
        row[f"Iter_{k}"] = out[k]["iterations"]
        row[f"Accepted_{k}"] = out[k]["accepted"]
        row[f"AcceptedWorse_{k}"] = out[k]["accepted_worse"]
        row[f"WorstRemovalUse_{k}"] = out[k]["destroy_worst"]
        row[f"Regret2Use_{k}"] = out[k]["repair_regret2"]
    row["omega_V"] = round(sol["omega_V"], 6)
    row["Winner"] = winner

    k_text = "/".join(f"{k}:{out[k]['K']}" for k in policies)
    tbc_text = "/".join(f"{k}:{out[k]['tbc']:.1f}" for k in policies)
    iter_text = "/".join(f"{k}:{out[k]['iterations']}" for k in policies)
    print(f"  {sol['name']:<10} K[{k_text}]  TBC50[{tbc_text}]  win={winner}")
    print(f"              iterations[{iter_text}]")
    return row


def run_sweep(files, tlim, no_improve, use_prune, which=None,
              max_iterations=ALNS_MAX_ITERATIONS,
              segment_size=ALNS_SEGMENT_SIZE,
              fast_large=False):
    print("\n--- SOLVING ONCE PER INSTANCE; PLANS REUSED ACROSS STRESS MIXTURES ---")
    sols, t0 = [], time.perf_counter()
    for i, f in enumerate(files, 1):
        print(f"[solve {i}/{len(files)}] {Path(f).stem:<10}", end="  ")
        try:
            s = solve_instance(f, tlim, no_improve, use_prune, which,
                               max_iterations, segment_size, fast_large)
            sols.append(s)
            print("K=" + "/".join(f"{k}:{s['res'][k]['K']}" for k in POLICY_ORDER if k in s["res"]))
        except Exception as e:
            print(f"ERROR: {e}")
    print(f"   solved {len(sols)} instances in {(time.perf_counter() - t0) / 60:.1f} min")
    if not sols:
        print("No instances solved.")
        return

    summary = []
    policies = [k for k in POLICY_ORDER if k in sols[0]["res"]]
    print(f"\n=== MIXTURE SENSITIVITY ({len(sols)} plans, {len(MIXTURES)} mixtures) ===")
    for label, mix, hf in MIXTURES:
        wins = {k: 0 for k in policies}
        tsum = {k: 0.0 for k in policies}
        esum = {k: 0.0 for k in policies}
        for s in sols:
            priced, winner = price_instance(s, mix, hf)
            wins[winner] += 1
            for k in policies:
                tsum[k] += priced[k]["tbc"]
                esum[k] += priced[k]["evx"]
        nsol = len(sols)
        mean_winner = min(policies, key=lambda k: tsum[k])
        row = {"config": label, "heavy": hf, "mean_winner": mean_winner}
        for k in policies:
            row[f"wins_{k}"] = wins[k]
            row[f"mean_TBC_{k}"] = tsum[k] / nsol
            row[f"mean_EVextra_{k}"] = esum[k] / nsol
        summary.append(row)
        print(f"   {label:<30} mean winner={mean_winner:<10} "
              + "  ".join(f"{k}:{wins[k]}/{nsol}" for k in policies))

    df = pd.DataFrame(summary)
    out = "results_mixture_sensitivity_full_alns.xlsx"
    try:
        df.to_excel(out, index=False)
        print(f"\n   wrote {out}")
    except Exception as e:
        out_csv = "results_mixture_sensitivity_full_alns.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n   (xlsx unavailable: {e}) wrote {out_csv}")


def main():
    argv = sys.argv[1:]
    data_dir = DATA_DIR
    tlim = TLIM
    no_improve = NO_IMPROVE
    max_n = None
    use_prune = True
    sweep = False
    max_iterations = ALNS_MAX_ITERATIONS
    segment_size = ALNS_SEGMENT_SIZE
    fast_large = False
    which = None

    for a in argv:
        if a.startswith("dir="):
            data_dir = a[4:]
        elif a.startswith("t="):
            tlim = float(a[2:])
        elif a.startswith("noimp="):
            no_improve = float(a[6:])
        elif a.startswith("max="):
            max_n = int(a[4:])
        elif a.startswith("iters="):
            max_iterations = int(a[6:])
        elif a.startswith("segment="):
            segment_size = int(a[8:])
        elif a.startswith("policies="):
            raw = [x.strip() for x in a.split("=", 1)[1].split(",") if x.strip()]
            which = ["Cantelli" if x.upper() == "MDRO" else x for x in raw]
        elif a == "noprune":
            use_prune = False
        elif a == "sweep":
            sweep = True
        elif a == "fastlarge":
            fast_large = True

    files = sorted(glob.glob(str(Path(data_dir) / "*.vrpspd")))
    if not files:
        files = sorted(glob.glob(str(Path(data_dir) / "*.txt"))) or sorted(glob.glob(str(Path(data_dir) / "*")))
        files = [f for f in files if Path(f).is_file()]
    if max_n:
        files = files[:max_n]
    if not files:
        print(f"ERROR: no instances found in '{data_dir}/'.")
        return

    policies = which if which else list(POLICY_ORDER)
    print("=" * 110)
    print(" DETHLOFF FULL-ALNS RUNNER -- stochastic/robust planning and TBC50 stress test")
    print("=" * 110)
    print(f"   instances={len(files)}  policies={','.join(policies)}")
    print(f"   alpha={ALPHA} cv={CV} rho={RHO} eps_frac={EPS_FRAC} N_data={N_DATA} N_mc={N_MC:,}")
    print(f"   ALNS: max_iters={max_iterations} tlim={tlim:g}s no-improve={no_improve:g}s "
          f"destroy=[{ALNS_DESTROY_MIN_FRAC:.2f},{ALNS_DESTROY_MAX_FRAC:.2f}] "
          f"SA=({ALNS_SA_TEMP_INIT:g},{ALNS_SA_COOLING:g}) segment={segment_size}")
    print(f"   operators=random/worst removal + greedy/regret-2 repair; adaptive roulette weights")
    print("   Dethloff distance, demands, and capacity are divided by 10,000 at parse time")
    if fast_large:
        print("   WARNING: fastlarge enabled; n>90 uses CW+2OPT rather than ALNS")
    print("-" * 110)

    if sweep:
        run_sweep(files, tlim, no_improve, use_prune, which,
                  max_iterations, segment_size, fast_large)
        return

    rows = []
    t_start = time.perf_counter()
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}]", end=" ")
        try:
            rows.append(run_one(f, tlim, no_improve, use_prune, which,
                                max_iterations, segment_size, fast_large))
        except Exception as e:
            print(f"  {Path(f).stem:<10} ERROR: {e}")
    print("-" * 110)
    print(f"   total wall time = {(time.perf_counter() - t_start) / 60:.1f} min")
    if not rows:
        print("No successful instances; nothing to write.")
        return

    df = pd.DataFrame(rows)
    numeric = df.select_dtypes(include=[np.number]).columns
    avg = {c: float(df[c].mean()) for c in numeric}
    avg["Instance"] = "AVERAGE"
    available = [k for k in POLICY_ORDER if f"TBC50_{k}" in avg]
    avg["Winner"] = min(available, key=lambda k: avg[f"TBC50_{k}"])
    df_out = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

    print("\n=== MEAN TBC @ omega_F/omega_V=50 ===")
    for k in available:
        print(f"   {k:<10}: TBC50={avg[f'TBC50_{k}']:>12,.2f}  "
              f"K={avg[f'K_{k}']:.2f}  EVextra={avg[f'EVextra_{k}']:.5f}  "
              f"worst-fail={avg[f'WorstFail_{k}']:.5f}")
    print(f"   -> lowest mean TBC50: {avg['Winner']}")

    out = "results_dethloff_full_alns_summary.xlsx"
    try:
        df_out.to_excel(out, index=False)
        print(f"\n   wrote {out} ({len(rows)} instances + AVERAGE row)")
    except Exception as e:
        out_csv = "results_dethloff_full_alns_summary.csv"
        df_out.to_csv(out_csv, index=False)
        print(f"\n   (xlsx unavailable: {e}) wrote {out_csv}")


if __name__ == "__main__":
    main()
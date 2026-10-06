"""ALG (class decomposition + point-to-point orienteering oracle) for h_i = beta + lam*a_i, lam>1.
The oracle is pluggable: exact (alpha=1, small n) or any routine returning a route over the class
sites whose excess is <= Phi at every visited site (that is all the feasibility proof needs)."""
import math, numpy as np
from common import *

def gap_m(rho, theta):
    """smallest m>=1 with (2 theta + 4/rho) <= (1-theta)(2^m - 1)"""
    m = 1
    while (2 * theta + 4 / rho) > (1 - theta) * (2 ** m - 1):
        m += 1
    return m

def pieces(theta):
    return math.ceil(2 / theta - 1e-12)

def classes(D, lam, beta):
    """class index k for each site 1..n; returns dict k -> sites, and alpha_k (lower boundary), a0."""
    n = len(D) - 1; a = D[0]; rho = lam - 1
    a0 = beta / rho if beta > 0 else 1.0
    cls = {}
    for i in range(1, n + 1):
        if beta > 0 and a[i] < a0:
            k = -1
        else:
            k = math.floor(math.log2(a[i] / a0) + 1e-12)
            # guard against fp
            while a0 * 2 ** k > a[i] * (1 + 1e-12): k -= 1
            while a0 * 2 ** (k + 1) <= a[i] * (1 - 1e-12): k += 1
        cls.setdefault(k, []).append(i)
    return cls, a0

def alpha_k(k, a0, beta):
    if beta > 0 and k == -1: return 0.0
    return a0 * 2.0 ** k

def alg(D, w, lam, beta=0.0, theta=0.5, oracle=None, m=None, return_info=False):
    rho = lam - 1; assert rho > 0
    oracle = oracle or v_unif_exact
    cls, a0 = classes(D, lam, beta)
    if m is None: m = gap_m(rho, theta)
    ks = sorted(cls)
    val = {}; rt = {}
    for k in ks:
        flo = beta + rho * alpha_k(k, a0, beta)
        Phi = theta * flo
        v, r = oracle(D, w, cls[k], Phi)
        val[k] = v; rt[k] = r
    # weighted independent set with index gap >= m
    best = {}; choice = {}
    for idx, k in enumerate(ks):
        # take k: previous allowed index with ks[j] <= k-m
        j = idx - 1
        while j >= 0 and ks[j] > k - m: j -= 1
        take = val[k] + (best[ks[j]] if j >= 0 else 0.0)
        skip = best[ks[idx - 1]] if idx > 0 else 0.0
        if take >= skip: best[k] = take; choice[k] = ('t', ks[j] if j >= 0 else None)
        else: best[k] = skip; choice[k] = ('s', ks[idx - 1] if idx > 0 else None)
    chosen = []
    k = ks[-1] if ks else None
    while k is not None:
        c, p = choice[k]
        if c == 't': chosen.append(k)
        k = p
    chosen = sorted(chosen)
    route = []
    for k in chosen: route += rt[k]
    out = (route, chosen, val)
    return out if return_info else route

def sup_alpha_ratio(lam, beta, theta=0.5):
    rho = lam - 1
    return pieces(theta) * gap_m(rho, theta)

def alg_plus(D, w, lam, beta=0.0, theta=0.5, oracle=None, m=None):
    """ALG followed by (heuristic) augmentation: splice un-chosen class routes (and then single sites)
    into the route at the position that keeps every site protected. Never worse than ALG."""
    rho = lam - 1
    route, chosen, val = alg(D, w, lam, beta, theta, oracle, m, return_info=True)
    cls, a0 = classes(D, lam, beta)
    oracle = oracle or v_unif_exact
    rt = {}
    for k in cls:
        Phi = theta * (beta + rho * alpha_k(k, a0, beta))
        rt[k] = oracle(D, w, cls[k], Phi)[1]
    cur = list(route)
    def feasible(r): return all_protected(D, r, lam, beta)
    for k in sorted(cls, key=lambda k: -val[k]):
        if k in chosen: continue
        for cand in (rt[k],):
            if not cand: continue
            best = None
            for pos in range(len(cur) + 1):
                r2 = cur[:pos] + cand + cur[pos:]
                if feasible(r2): best = r2; break
            if best is not None: cur = best
    # single-site insertion, heaviest first
    inroute = set(cur)
    for s in sorted(range(1, len(D)), key=lambda s: -w[s]):
        if s in inroute: continue
        for pos in range(len(cur) + 1):
            r2 = cur[:pos] + [s] + cur[pos:]
            if feasible(r2): cur = r2; inroute.add(s); break
    return cur

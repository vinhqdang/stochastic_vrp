#!/usr/bin/env python3
"""ITEM 3: refined FPTAS guarantee.  Exact rational arithmetic throughout.
Refined theorem (K>1, n>=2): y=n/eps, N=floor(y), f=y-N.  For ANY feasible set S^ maximizing sum floor(w_i/K):
   W(S^)/W* > rho := 1/(1+max{(n-1)/y, (n-2+f)/N})  >=  N/(N+n-1)  >  1/(1+eps)  >  1-eps.
Checks: random instances (algorithm as in the paper, DP tie-breaking by min time) AND pessimistic worst maximizer
(brute force over all feasible sets), adversarial families (Prop. 8 family and the 'second family')."""
import random, sys, itertools
from fractions import Fraction as Fr
from math import floor

def feasible_sets(p, d):
    n = len(p); idx = sorted(range(n), key=lambda i: (d[i], i)); out = []
    for mask in range(1 << n):
        t = 0; ok = True
        for i in idx:
            if mask >> i & 1:
                t += p[i]
                if t > d[i]: ok = False; break
        if ok: out.append(mask)
    return out

def fptas(p, d, w, eps):
    """Algorithm 2 verbatim (value-indexed DP, strict '<' tie rule); returns (set mask, K, scaled)"""
    n = len(p); wmax = max(w)
    K = max(Fr(1), eps * wmax / n)
    ws = [floor(Fr(x) / K) for x in w]
    order = sorted(range(n), key=lambda i: (d[i], i))
    V = sum(ws); INF = float('inf')
    g = [[INF] * (V + 1) for _ in range(n + 1)]; ch = [[False] * (V + 1) for _ in range(n + 1)]
    g[0][0] = 0
    for a, i in enumerate(order, 1):
        for v in range(V + 1):
            g[a][v] = g[a - 1][v]
            if v >= ws[i] and g[a - 1][v - ws[i]] < INF and g[a - 1][v - ws[i]] + p[i] <= d[i]:
                if g[a - 1][v - ws[i]] + p[i] < g[a][v]:
                    g[a][v] = g[a - 1][v - ws[i]] + p[i]; ch[a][v] = True
    v = max(x for x in range(V + 1) if g[n][x] < INF)
    S = 0
    for a in range(n, 0, -1):
        if ch[a][v]: S |= 1 << order[a - 1]; v -= ws[order[a - 1]]
    return S, K, ws

def rho_exact(n, eps):
    y = Fr(n) / eps; N = floor(y); f = y - N
    return 1 / (1 + max((n - 1) / y, (n - 2 + f) / N)), Fr(N, N + n - 1)

def check_instance(p, d, w, eps, stats):
    n = len(p)
    keep = [i for i in range(n) if p[i] <= d[i]]
    p = [p[i] for i in keep]; d = [d[i] for i in keep]; w = [w[i] for i in keep]; n = len(p)
    if n < 2: return True
    fs = feasible_sets(p, d)
    W = lambda m: sum(w[i] for i in range(n) if m >> i & 1)
    Wstar = max(W(m) for m in fs)
    S, K, ws = fptas(p, d, w, eps)
    sv = lambda m: sum(ws[i] for i in range(n) if m >> i & 1)
    assert S in set(fs)
    x = W(S)
    best = max(sv(m) for m in fs); assert sv(S) == best
    worst = min(W(m) for m in fs if sv(m) == best)         # worst maximiser of the scaled value
    stats['inst'] += 1
    ok = True
    if K == 1:
        stats['K1'] += 1
        ok = (x == Wstar) and (worst == Wstar)
    else:
        stats['Kgt1'] += 1
        rho, simple = rho_exact(n, eps)
        wmax = max(w)
        for xx in (x, worst):
            ok &= Fr(xx) > rho * Wstar                    # refined (strict)
            ok &= rho >= simple
            ok &= simple > 1 / (1 + eps)
            ok &= Fr(xx) > (1 / (1 + eps)) * Wstar
            ok &= Fr(xx) > (1 - eps) * Wstar
            ok &= Fr(xx) > Wstar - eps * wmax              # classical
            ok &= Fr(xx) >= K * floor(Fr(wmax) / K) and Fr(xx) > wmax - K
        stats['minslack'] = min(stats['minslack'], float(Fr(x) / Wstar - rho))
        stats['minratio_over_rho'] = min(stats['minratio_over_rho'], float(Fr(worst) / Wstar / rho))
    if not ok: stats['viol'] += 1; print("VIOLATION", p, d, w, eps)
    return ok

def family1(n, eps, M):         # Proposition 8 family
    K = eps * M / n; z = int(-(-K // 1)) - 1                       # ceil(K)-1
    return [1] * n, [n] * n, [M] + [z] * (n - 1)

def family2(n, eps, Kint):      # second family: b (w=K*N), h (w=K*n/eps), n-2 zero-scaled sites
    y = Fr(n) / eps; N = floor(y)
    M = Kint * y                                               # w_max = K n/eps, must be an integer
    assert M.denominator == 1
    M = int(M)
    p = [1, 2] + [1] * (n - 2); d = [1, 2] + [n] * (n - 2)
    w = [Kint * N, M] + [Kint - 1] * (n - 2)
    return p, d, w

def family_value(p, d, w, eps):
    """exact: use the algorithm to get x; W* by brute force (n small) else by formula"""
    n = len(p)
    S, K, ws = fptas(p, d, w, eps)
    x = sum(w[i] for i in range(n) if S >> i & 1)
    fs = feasible_sets(p, d)
    Ws = max(sum(w[i] for i in range(n) if m >> i & 1) for m in fs)
    return Fr(x, Ws), K, S


def grid_check():
    """closed-form inequalities on a grid: rho >= N/(N+n-1) >= (n-eps)/(n+(n-2)eps) > 1/(1+eps) > 1-eps;
       (n-eps)/(n(1+eps)) <= rho; rho1 - rho < eps^2/n where rho1 = n/(n+(n-1)eps) (Prop. 8 limit)."""
    cnt = bad = 0
    for n in range(2, 61):
        for k in range(1, 200):
            eps = Fr(k, 200)
            rho, simple = rho_exact(n, eps)
            rho1 = Fr(n) / (n + (n - 1) * eps)
            ok = (rho >= simple >= (n - eps) / (n + (n - 2) * eps) > 1 / (1 + eps) > 1 - eps
                  and (n - eps) / (n * (1 + eps)) <= rho and 0 <= rho1 - rho < eps * eps / n and rho < 1)
            cnt += 1; bad += (not ok)
            if not ok: print("GRID FAIL", n, eps)
    return cnt, bad

def main():
    rng = random.Random(424242)
    stats = dict(inst=0, K1=0, Kgt1=0, viol=0, minslack=1e9, minratio_over_rho=1e9)
    epss = [Fr(1, 2), Fr(1, 3), Fr(1, 5), Fr(1, 10), Fr(3, 4), Fr(4, 5), Fr(9, 10), Fr(3, 10), Fr(7, 10), Fr(1, 20)]
    target = 4400
    while stats['inst'] < target:
        n = rng.randint(2, 10)
        regime = rng.random()
        p = [rng.randint(1, 8) for _ in range(n)]
        horizon = max(2, int(sum(p) * rng.choice([0.4, 0.7, 1.0])))
        d = [rng.randint(1, horizon) for _ in range(n)]
        if regime < 0.5: w = [rng.randint(1, 10 ** rng.choice([2, 3, 4, 6])) for _ in range(n)]
        elif regime < 0.8:                       # heavy hitter + dust (just below granule) -> stresses rounding
            big = rng.randint(10 ** 3, 10 ** 5); w = [big] + [rng.randint(1, big // n + 3) for _ in range(n - 1)]
        else: w = [rng.randint(1, 30) for _ in range(n)]
        check_instance(p, d, w, rng.choice(epss), stats)
    print(f"random instances: {stats['inst']} (K=1: {stats['K1']}, K>1: {stats['Kgt1']}); violations: {stats['viol']}")
    print(f"  min over K>1 instances of (x/W* - rho) = {stats['minslack']:.3e}  (>0 means strict bound held), min worst-maximiser/(rho W*) = {stats['minratio_over_rho']:.6f}")
    # hill-climbing adversary on small n to try to break rho (pessimistic maximiser)
    viol2 = 0; trials = 0; best_gap = 1e9
    for restart in range(60):
        n = rng.randint(2, 5); eps = rng.choice(epss)
        p = [rng.randint(1, 4) for _ in range(n)]; d = [rng.randint(1, sum(p)) for _ in range(n)]
        w = [rng.randint(5, 400) for _ in range(n)]
        def score(w_):
            keep = [i for i in range(n) if p[i] <= d[i]]
            if len(keep) < 2: return 1e9
            pp = [p[i] for i in keep]; dd = [d[i] for i in keep]; ww = [w_[i] for i in keep]
            fs = feasible_sets(pp, dd); nn = len(pp)
            Ws = max(sum(ww[i] for i in range(nn) if m >> i & 1) for m in fs)
            S, K, ws = fptas(pp, dd, ww, eps)
            if K == 1: return 1e9
            sv = lambda m: sum(ws[i] for i in range(nn) if m >> i & 1)
            best = max(sv(m) for m in fs)
            worst = min(sum(ww[i] for i in range(nn) if m >> i & 1) for m in fs if sv(m) == best)
            return float(Fr(worst, Ws) / rho_exact(nn, eps)[0])
        cur = score(w)
        for step in range(150):
            w2 = list(w); j = rng.randrange(n); w2[j] = max(1, w2[j] + rng.randint(-60, 60))
            s2 = score(w2); trials += 1
            if s2 <= cur: w, cur = w2, s2
        best_gap = min(best_gap, cur)
        if cur <= 1.0: viol2 += 1
    print(f"local-search adversary: {trials} evaluations, min (worst-maximiser ratio)/rho = {best_gap:.5f}; violations: {viol2}")
    # families
    print("Adversarial families (exact arithmetic, algorithm output):")
    fam_viol = 0
    for (n, eps) in [(2, Fr(1, 2)), (3, Fr(3, 4)), (4, Fr(1, 2)), (5, Fr(1, 5)), (6, Fr(1, 3)), (3, Fr(10, 13)), (4, Fr(4, 7)), (5, Fr(5, 24))]:
        rho, simple = rho_exact(n, eps); y = Fr(n) / eps; N = floor(y); f = y - N
        case1 = (n - 1) / y; case2 = (n - 2 + f) / N
        for Mscale in (10 ** 2, 10 ** 4, 10 ** 6):
            M = Mscale * n * eps.denominator * 1  # integer M
            if (M * eps) % n != 0:
                pass
            p1, d1, w1 = family1(n, eps, M)
            r1, K1, S1 = family_value(p1, d1, w1, eps)
            Kint = Mscale
            try:
                p2, d2, w2 = family2(n, eps, Kint * eps.numerator)
                r2, K2, S2 = family_value(p2, d2, w2, eps)
            except AssertionError:
                r2 = None
            rr = [float(r1)] + ([float(r2)] if r2 is not None else [])
            if r1 <= rho: fam_viol += 1
            if r2 is not None and r2 <= rho: fam_viol += 1
        print(f"  n={n} eps={eps} y=n/eps={float(y):.3f}: bound rho={float(rho):.5f} (case1 gives {float(1/(1+case1)):.5f}, case2 gives {float(1/(1+case2)):.5f}); "
              f"family1 ratio (K={float(K1):.0f}) = {float(r1):.5f}" + (f"; family2 ratio (K={float(K2):.0f}) = {float(r2):.5f}" if r2 is not None else ""))
    print("family violations of strict bound:", fam_viol)
    # exact attainment (n=2): b=(p1,d1,w40), h=(p2,d2,w59), eps=40/59 -> K=20, ratio = rho exactly (so the theorem must say '>=', not '>')
    S_, K_, ws_ = fptas([1, 2], [1, 2], [40, 59], Fr(40, 59)); x_ = sum([40, 59][i] for i in range(2) if S_ >> i & 1)
    eq_ok = (Fr(x_, 59) == rho_exact(2, Fr(40, 59))[0] and K_ == 20)
    print(f"exact attainment example n=2, eps=40/59: K={K_}, x/W* = {Fr(x_,59)} = rho? {eq_ok}")
    gc, gb = grid_check(); print(f"closed-form inequality grid: {gc} (n,eps) pairs, failures {gb}")
    return stats['viol'] + viol2 + fam_viol + gb + (0 if eq_ok else 1)

if __name__ == '__main__':
    sys.exit(1 if main() else 0)

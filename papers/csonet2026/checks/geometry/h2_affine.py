"""H2 part 3: MWHED with affine deadlines d_i = beta + alpha * p_i.

  A0  solver validation: subset/EDD solver == permutation brute force
  A1  feasibility formula for alpha<=1:  S feasible  <=>  sum_S p - alpha*max_S p <= beta
  A2  reduction PARTITION -> MWHED, 0<alpha<=1 (beta part of input, w=p)   [Thm C1]
  A3  reduction PARTITION -> MWHED, alpha>1, beta=0 (ballast + geometric levels) [Thm C2]
  A4  geometric size bound for beta=0, alpha>1:  |S| <= 1 + log_lambda(P/p_min), lambda=alpha/(alpha-1)
"""
import itertools
import math
import random
from fractions import Fraction


def edd_feasible(items):
    """items: list of (p, d).  EDD (sorted by d) feasibility."""
    t = 0
    for p, d in sorted(items, key=lambda x: (x[1], x[0])):
        t += p
        if t > d:
            return False
    return True


def solve_subsets(p, d, w):
    """exact optimum by enumerating subsets + EDD feasibility (Lemma 'edd')."""
    n = len(p)
    best = 0
    bestS = ()
    for mask in range(1 << n):
        idx = [i for i in range(n) if mask >> i & 1]
        if edd_feasible([(p[i], d[i]) for i in idx]):
            val = sum(w[i] for i in idx)
            if val > best:
                best, bestS = val, tuple(idx)
    return best, bestS


def solve_perms(p, d, w):
    n = len(p)
    best = 0
    for perm in itertools.permutations(range(n)):
        t = 0; tot = 0
        for i in perm:
            t += p[i]
            if t <= d[i]:
                tot += w[i]
        best = max(best, tot)
    return best


def partition_yes(a):
    s = {0}
    for x in a:
        s |= {v + x for v in s}
    return sum(a) % 2 == 0 and sum(a) // 2 in s


def rand_partition(rng, n, hi=9):
    while True:
        a = [rng.randint(1, hi) for _ in range(n)]
        if sum(a) % 2 == 0:
            return a


# -------------------------------------------------------------------- A0, A1
def A0_A1(seed=31, trials=1500):
    rng = random.Random(seed)
    n0 = n1 = 0
    for tr in range(trials):
        n = rng.randint(1, 6)
        p = [rng.randint(1, 12) for _ in range(n)]
        d = [rng.randint(1, 40) for _ in range(n)]
        w = [rng.randint(1, 9) for _ in range(n)]
        assert solve_subsets(p, d, w)[0] == solve_perms(p, d, w)
        n0 += 1
    for tr in range(trials):
        al = Fraction(rng.randint(1, 8), 8)
        n = rng.randint(1, 7)
        p = [rng.randint(1, 20) for _ in range(n)]
        beta = rng.randint(0, 40)
        d = [beta + al * pi for pi in p]
        for mask in range(1, 1 << n):
            S = [i for i in range(n) if mask >> i & 1]
            lhs = edd_feasible([(p[i], d[i]) for i in S])
            rhs = sum(p[i] for i in S) - al * max(p[i] for i in S) <= beta
            assert lhs == rhs, (al, beta, p, S)
        n1 += 1
    return n0, n1


# -------------------------------------------------------------------- A2
def reduce_A(a, u, q):
    """alpha = u/q in (0,1].  returns p,d,w,target"""
    A = sum(a)
    assert A % 2 == 0
    alpha = Fraction(u, q)
    p = [2 * q * x for x in a] + [2 * q * A + q]
    w = list(p)
    beta = q * A + (q - u) * (2 * A + 1)
    d = [beta + alpha * pi for pi in p]
    assert all(x.denominator == 1 if isinstance(x, Fraction) else True for x in d)
    d = [int(x) for x in d]
    assert all(pi <= di for pi, di in zip(p, d))
    target = p[-1] + q * A
    return p, d, w, target, beta


def A2(seed=32, trials=60):
    rng = random.Random(seed)
    stats = {"yes": 0, "no": 0}
    alphas = [(1, 2), (1, 3), (2, 3), (1, 1), (3, 4), (1, 5), (4, 5), (7, 9), (1, 10)]
    for tr in range(trials):
        u, q = alphas[tr % len(alphas)]
        n = rng.randint(3, 8)
        a = rand_partition(rng, n)
        p, d, w, target, beta = reduce_A(a, u, q)
        opt, _ = solve_subsets(p, d, w)
        assert opt <= target
        yes = partition_yes(a)
        assert (opt >= target) == yes, (a, u, q, opt, target)
        stats["yes" if yes else "no"] += 1
    return stats


# -------------------------------------------------------------------- A3
def reduce_C(a, u, q):
    """alpha=u/q>1, beta=0.  Levels (X_i, X_i+a_i), ballast of ceil(c) equal items
    (c = alpha-1), all scaled to integers divisible by q.  returns p,d,w,target."""
    A = sum(a)
    n = len(a)
    assert A % 2 == 0 and u > q
    c = Fraction(u - q, q)
    lam = 1 + 1 / c
    m = math.ceil(c) - 1              # number of ballast items besides the last; c-m in (0,1]
    rho = max(3, math.floor(lam) + 1)  # integer > lam and >= 3
    if rho == lam:
        rho += 1
    slack = c - Fraction(1, rho - 1)
    assert slack > 0
    M0 = int(math.ceil(Fraction(A, 1) / slack)) + A + 1
    X = [M0 * rho ** (i + 1) for i in range(n)]
    SX = sum(X)
    r_den = (c - m)                    # in (0,1]
    # scale so that every size is an integer multiple of q and g integer
    s = r_den.numerator * q * q
    Xs = [s * x for x in X]
    bs = [s * x for x in a]
    capS = s * (SX + Fraction(A, 2))
    g = capS / r_den                   # c-m = r_den :  chosen + m g <= c g  <=>  chosen <= (c-m) g
    assert g.denominator == 1, g
    g = int(g)
    assert g % q == 0
    p = []
    kind = []
    for i in range(n):
        p += [Xs[i], Xs[i] + bs[i]]
        kind += [("lv", i, 0), ("lv", i, 1)]
    nlev = len(p)
    levw = sum(p)
    p += [g] * (m + 1)
    kind += [("bal",)] * (m + 1)
    WB = 2 * levw + 1
    w = list(p[:nlev]) + [WB] * (m + 1)
    d = [Fraction(u, q) * pi for pi in p]
    assert all(x.denominator == 1 for x in d)
    d = [int(x) for x in d]
    target = (m + 1) * WB + capS
    meta = dict(c=c, m=m, rho=rho, g=g, cap=capS)
    return p, d, w, target, meta


def A3(seed=33, trials=40, nmax=4):
    rng = random.Random(seed)
    stats = {"yes": 0, "no": 0}
    alphas = [(6, 5), (3, 2), (2, 1), (5, 2), (3, 1), (9, 2), (11, 2), (7, 5), (13, 10), (4, 1)]
    for tr in range(trials):
        u, q = alphas[tr % len(alphas)]
        n = rng.randint(3, nmax)
        a = rand_partition(rng, n, hi=6)
        p, d, w, target, meta = reduce_C(a, u, q)
        assert len(p) <= 15
        opt, S = solve_subsets(p, d, w)
        assert opt <= target, (a, u, q, opt, target)
        yes = partition_yes(a)
        assert (opt >= target) == yes, (a, u, q, opt, target, meta)
        stats["yes" if yes else "no"] += 1
    return stats


# -------------------------------------------------------------------- A4
def A4(seed=34, trials=400):
    rng = random.Random(seed)
    cnt = 0
    for tr in range(trials):
        n = rng.randint(2, 11)
        al = Fraction(rng.randint(11, 40), 10)
        p = [rng.randint(1, 60) for _ in range(n)]
        d = [al * x for x in p]
        w = [rng.randint(1, 9) for _ in range(n)]
        best, S = solve_subsets(p, d, w)
        lam = al / (al - 1)
        # take any feasible set (the optimum) and check the growth law T_k >= lam T_{k-1}
        ps = sorted(p[i] for i in S)
        T = 0
        for k, x in enumerate(ps):
            if k > 0:
                assert T <= (al - 1) * x
                assert T + x >= lam * T - 1e-12
            T += x
        if len(ps) > 1:
            assert len(ps) <= 1 + math.log(T / ps[0]) / math.log(float(lam)) + 1e-9
        cnt += 1
    return cnt


if __name__ == "__main__":
    print("A0/A1 (solver validation; feasibility-formula checks):", A0_A1())
    print("A2 alpha<=1 reduction (yes/no instances):", A2())
    print("A3 alpha>1, beta=0 reduction (yes/no instances):", A3())
    print("A4 geometric-growth law checks:", A4())

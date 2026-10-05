"""H1 basic verification: geometry, feasibility closed form, solvers, cardinality bound.
Seeded; prints a count table.  Run: python3 h1_verify_basic.py
"""
import random, math
from fractions import Fraction as F
from h1_lib import *

random.seed(20261005)
rows = []


def report(name, n_inst, n_checks, bad):
    rows.append((name, n_inst, n_checks, bad))
    print(f"{name:<78s} inst={n_inst:6d} checks={n_checks:8d} violations={bad}")


# ---- 1. geometry: both readings reduce to d = lambda * p ------------------
bad = chk = 0
for _ in range(20000):
    u = F(random.randint(1, 30), random.randint(1, 10))   # front speed
    s = F(random.randint(1, 60), random.randint(1, 10))   # crew speed
    r = F(random.randint(1, 500), random.randint(1, 7))   # distance from depot
    Cprev = F(random.randint(0, 800), random.randint(1, 5))  # crew busy until
    p = 2 * r / s
    rho = s / (2 * u)
    # return reading: crew back at depot by the time the front reaches r
    lhs = (Cprev + p <= r / u)
    rhs = (Cprev + p <= rho * p)
    chk += 1; bad += lhs != rhs
    # arrival reading: crew ARRIVES (Cprev + r/s) before the front r/u
    lhs = (Cprev + r / s <= r / u)
    rhs = (Cprev + p <= (rho + F(1, 2)) * p)
    chk += 1; bad += lhs != rhs
report("geometry: front from depot, return/arrival reading == d=rho*p / (rho+1/2)*p", 20000, chk, bad)

# ---- 2. closed form P_{k-1} <= mu p_k  vs literal EDD test -----------------
RHOS = [F(1, 2), F(1), F(11, 10), F(3, 2), F(2), F(5, 2), F(3), F(7, 2), F(5), F(8), F(25, 2)]
bad = chk = ninst = 0
for _ in range(1500):
    n = random.randint(1, 9)
    rho = random.choice(RHOS)
    spread = random.choice([3, 10, 100, 10 ** 4])
    p = [random.randint(1, spread) for _ in range(n)]
    d = [rho * x for x in p]
    ninst += 1
    for mask in range(1 << n):
        S = [i for i in range(n) if mask >> i & 1]
        chk += 1
        bad += feasible_by_definition(S, p, d) != feasible_prefix(S, p, rho)
report("closed form (sorted by p, P_{k-1}<=mu p_k) == literal EDD test, all subsets", ninst, chk, bad)

# ---- 3. all-permutation optimum == subset DFS == Lawler-Moore DP ----------
bad = chk = ninst = 0
for _ in range(400):
    n = random.randint(1, 7)
    rho = random.choice([F(1), F(3, 2), F(2), F(5, 2), F(3), F(7, 2), F(6)])
    p = [random.randint(1, 30) for _ in range(n)]
    w = [random.randint(1, 20) for _ in range(n)]
    d = [rho * x for x in p]
    ninst += 1
    a = best_over_permutations(p, w, d)
    b = brute_opt(p, w, rho)
    pi, den = scale_to_integers(p, rho)
    di = [int(rho * x) for x in pi]
    wi = w
    c = dp_lawler_moore(pi, wi, di)
    chk += 2
    bad += (a != b) + (b != c)
report("optimum: all n! orders == subset DFS == Lawler-Moore DP (integerised)", ninst, chk, bad)

# ---- 4. equal weights: Moore-Hodgson == brute -------------------------------
bad = chk = ninst = 0
for _ in range(600):
    n = random.randint(1, 12)
    rho = random.choice(RHOS)
    p = [random.randint(1, 10 ** random.choice([1, 2, 4])) for _ in range(n)]
    pi, den = scale_to_integers(p, rho)
    d = [int(rho * x) for x in pi]
    ninst += 1
    mh = moore_hodgson(pi, d)
    bf = brute_opt(p, [1] * n, rho)
    chk += 1; bad += mh != bf
report("equal weights: Moore-Hodgson == exhaustive", ninst, chk, bad)

# ---- 5. cardinality bound  m <= 1 + log(rho R)/log(1+1/mu) -------------------
bad = chk = ninst = 0
worst_slack = 10 ** 9
for _ in range(1500):
    n = random.randint(2, 16)
    rho = random.choice([F(11, 10), F(3, 2), F(2), F(5, 2), F(3), F(5), F(8)])
    mu = rho - 1
    spread = random.choice([5, 50, 10 ** 3, 10 ** 6])
    p = sorted(random.randint(1, spread) for _ in range(n))
    R = F(p[-1], p[0])
    m = max_card_feasible(p, rho)
    bound = 1 + math.log(float(rho * R)) / math.log(1 + 1 / float(mu))
    ninst += 1; chk += 1
    bad += m > bound + 1e-9
    worst_slack = min(worst_slack, bound - m)
report("cardinality bound  max|S| <= 1+log(rho R)/log(1+1/mu)", ninst, chk, bad)
print("   smallest slack observed:", round(worst_slack, 3))

# ---- 6. rho<=1: only singletons (rho=1) or nothing (rho<1) -----------------
bad = chk = ninst = 0
for _ in range(300):
    n = random.randint(1, 8)
    p = [random.randint(1, 50) for _ in range(n)]
    w = [random.randint(1, 50) for _ in range(n)]
    ninst += 1
    chk += 2
    bad += brute_opt(p, w, F(1)) != max(w)
    bad += brute_opt(p, w, F(9, 10)) != 0
report("rho=1: optimum = max w ; rho<1: optimum = 0", ninst, chk, bad)

print("\nTOTAL violations:", sum(r[3] for r in rows))
